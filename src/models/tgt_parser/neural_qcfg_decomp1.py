from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ...datamodules.components.vocab import Vocabulary
from ..components.common import MultiResidualLayer
from .base import TgtParserBase
from .struct.td_pcfg import FastestTDPCFG


class NeuralQCFGDecomp1TgtParser(TgtParserBase):
    def __init__(
        self,
        vocab=100,
        dim=256,
        num_layers=3,
        src_dim=256,
        nt_states=10,
        pt_states=1,
        rule_constraint_type=0,
        use_copy=False,
        nt_span_range=[2, 1000],
        pt_span_range=[1, 1],
        num_samples=10,
        check_ppl=False,
        cpd_args=None,
    ):
        super(NeuralQCFGDecomp1TgtParser, self).__init__()
        self.neg_huge = -1e5
        self.pcfg = FastestTDPCFG()
        self.vocab = vocab
        self.dim = dim
        self.src_dim = src_dim
        self.nt_states = nt_states
        self.pt_states = pt_states
        self.rule_constraint_type = rule_constraint_type
        self.use_copy = use_copy
        self.nt_span_range = nt_span_range
        self.pt_span_range = pt_span_range
        self.num_samples = num_samples
        self.check_ppl = check_ppl

        self.src_nt_emb = nn.Parameter(torch.randn(nt_states, dim))
        self.src_nt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.src_pt_emb = nn.Parameter(torch.randn(pt_states, dim))
        self.src_pt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.rule_mlp_parent = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_mlp_left = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_mlp_right = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_child = nn.Linear(dim, 1, bias=False)
        self.vocab_out = MultiResidualLayer(
            dim, dim, num_layers=num_layers, out_dim=vocab
        )

        assert cpd_args is not None
        self.rank_proj = nn.Parameter(torch.randn(dim, cpd_args.rank))
        self.head_ln = nn.LayerNorm(cpd_args.rank)
        self.left_ln = nn.LayerNorm(cpd_args.rank)
        self.right_ln = nn.LayerNorm(cpd_args.rank)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.src_nt_emb.data)
        nn.init.xavier_uniform_(self.src_pt_emb.data)

    def forward(self, x, lengths, node_features, spans, copy_position=None):
        params, *_ = self.get_params(node_features, spans, x, x_str=None)
        out = self.pcfg(params, lengths, False)
        return out

    def parse(self, x, lengths, node_features, spans, copy_position=None):
        params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = self.get_params(
            node_features, spans, x, x_str=None
        )
        out = self.pcfg(params, lengths, True)

        # out: list of list, containing spans (i, j, label)
        src_nt_states = self.nt_states * nt_num_nodes
        src_pt_states = self.pt_states * pt_num_nodes
        all_spans_node = []
        for b, (all_span, pt_span, nt_span) in enumerate(zip(out, pt_spans, nt_spans)):
            all_span_node = []
            for l, r, label in all_span:
                if l == r:
                    if label < src_pt_states:
                        all_span_node.append(pt_span[label % pt_num_nodes])
                    else:
                        # these are for tgt_nt_states, which are removed for now.
                        all_span_node.append([-1, -1, label - src_pt_states])
                else:
                    if label < src_nt_states:
                        all_span_node.append(nt_span[label % nt_num_nodes])
                    else:
                        all_span_node.append([-1, -1, label - src_nt_states])
            all_spans_node.append(all_span_node)
        return out, all_spans_node

    def generate(self, node_features, spans, tokenizer: Vocabulary):
        # if check_ppl=True, I will compute ppl for samples, return the one with minimum ppl
        # else, just return the one with the maximum score
        params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = self.get_params(
            node_features, spans
        )
        preds = self.pcfg.sampled_decoding(
            params,
            nt_spans,
            self.nt_states,
            pt_spans,
            self.pt_states,
            num_samples=self.num_samples,
            use_copy=self.use_copy,
            max_length=100,
        )

        if self.check_ppl:
            new_preds = []
            for i, preds_inst in enumerate(preds):
                _ids = [
                    inst[0] for inst in preds_inst if 0 < len(inst[0]) < 60
                ]  # <60: ~20gb
                if len(_ids) == 0:
                    _ids = [[0, 0]]  # unk, unk
                _ids.sort(key=lambda x: len(x), reverse=True)
                _lens = [len(inst) for inst in _ids]
                _ids_t = torch.full((len(_ids), _lens[0]), tokenizer.pad_token_id)
                for j, (snt, length) in enumerate(zip(_ids, _lens)):
                    _ids_t[j, :length] = torch.tensor(snt)
                _ids_t = _ids_t.to(node_features[0].device)

                batch_size = max(200 // _lens[0], 1)  # ~17gb, 3min
                ppl = []
                for j in range(0, len(_ids), batch_size):
                    real_batch_size = min(batch_size, len(_ids) - j)
                    _node_ft = [node_features[i] for _ in range(real_batch_size)]
                    _spans = [spans[i] for _ in range(real_batch_size)]
                    nll = (
                        self(
                            _ids_t[j : j + batch_size],
                            _lens[j : j + batch_size],
                            _node_ft,
                            _spans,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    ppl.append(np.exp(nll / np.array(_lens[j : j + batch_size])))
                ppl = np.concatenate(ppl, 0)
                chosen = np.argmin(ppl)
                new_preds.append((_ids[chosen], ppl[chosen]))
            preds = new_preds
        else:
            preds = [inst[0] for inst in preds]

        pred_strings = []
        for pred in preds:
            snt, score = pred
            try:
                pred_strings.append((tokenizer.convert_ids_to_tokens(snt), score))
            except IndexError:
                print("bad pred:", snt)
                pred_strings.append([("", -999)])
        return pred_strings

    def get_params(
        self,
        node_features,
        spans,
        x: Optional[torch.Tensor] = None,
        x_str: Optional[List[str]] = None,
    ):
        batch_size = len(spans)

        # seperate nt and pt features according to span width
        pt_node_features, nt_node_features = [], []
        pt_spans, nt_spans = [], []
        for span, node_feature in zip(spans, node_features):
            pt_node_feature = []
            nt_node_feature = []
            pt_span = []
            nt_span = []
            for i, s in enumerate(span):
                s_len = s[1] - s[0] + 1
                if s_len >= self.nt_span_range[0] and s_len <= self.nt_span_range[1]:
                    nt_node_feature.append(node_feature[i])
                    nt_span.append(s)
                if s_len >= self.pt_span_range[0] and s_len <= self.pt_span_range[1]:
                    pt_node_feature.append(node_feature[i])
                    pt_span.append(s)
            if len(nt_node_feature) == 0:
                nt_node_feature.append(node_feature[-1])
                nt_span.append(span[-1])
            pt_node_features.append(torch.stack(pt_node_feature))
            nt_node_features.append(torch.stack(nt_node_feature))
            pt_spans.append(pt_span)
            nt_spans.append(nt_span)
        nt_node_features = pad_sequence(
            nt_node_features, batch_first=True, padding_value=0.0
        )
        pt_node_features = pad_sequence(
            pt_node_features, batch_first=True, padding_value=0.0
        )
        pt_num_nodes = pt_node_features.size(1)
        nt_num_nodes = nt_node_features.size(1)
        device = nt_node_features.device

        # e = u + h
        nt_emb = []
        src_nt_node_emb = self.src_nt_node_mlp(nt_node_features)
        src_nt_emb = self.src_nt_emb.unsqueeze(0).expand(
            batch_size, self.nt_states, self.dim
        )
        src_nt_emb = src_nt_emb.unsqueeze(2) + src_nt_node_emb.unsqueeze(1)
        src_nt_emb = src_nt_emb.view(batch_size, self.nt_states * nt_num_nodes, -1)
        nt_emb.append(src_nt_emb)
        nt_emb = torch.cat(nt_emb, 1)
        pt_emb = []
        src_pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        src_pt_emb = self.src_pt_emb.unsqueeze(0).expand(
            batch_size, self.pt_states, self.dim
        )
        src_pt_emb = src_pt_emb.unsqueeze(2) + src_pt_node_emb.unsqueeze(1)
        src_pt_emb = src_pt_emb.view(batch_size, self.pt_states * pt_num_nodes, -1)
        pt_emb.append(src_pt_emb)
        pt_emb = torch.cat(pt_emb, 1)

        # S->A
        roots = self.root_mlp_child(nt_emb)
        roots = roots.view(batch_size, -1)
        roots += self.neg_huge
        # root must align to root
        for s in range(self.nt_states):
            roots[:, s * nt_num_nodes + nt_num_nodes - 1] -= self.neg_huge
        roots = F.log_softmax(roots, 1)

        # A->BC
        nt = nt_emb.size(1)
        pt = pt_emb.size(1)
        all_emb = torch.cat([nt_emb, pt_emb], 1)
        rule_emb_parent = self.rule_mlp_parent(nt_emb) @ self.rank_proj
        rule_emb_left = self.rule_mlp_left(all_emb) @ self.rank_proj
        rule_emb_right = self.rule_mlp_right(all_emb) @ self.rank_proj
        rule_emb_parent = self.head_ln(rule_emb_parent).softmax(-1)
        rule_emb_left = self.left_ln(rule_emb_left).softmax(-2)
        rule_emb_right = self.right_ln(rule_emb_right).softmax(-2)
        # rule_emb_child = rule_emb_left[:, :, None, :] + rule_emb_right[:, None, :, :]
        # rule_emb_child = rule_emb_child.view(batch_size, (nt+pt)**2, self.dim)
        # rules = torch.matmul(rule_emb_parent, rule_emb_child.transpose(1,2))
        # rules = rules.view(batch_size, nt, nt + pt, nt + pt)

        src_pt = pt

        # src_nt_idx = slice(0, src_nt)
        # src_pt_idx = slice(src_nt + tgt_nt, src_nt + tgt_nt + src_pt)
        # tgt_nt_idx = slice(src_nt, src_nt + tgt_nt)
        # tgt_pt_idx = slice(src_nt + tgt_nt + src_pt, src_nt + tgt_nt + src_pt + tgt_pt)

        # if self.rule_constraint_type > 0:
        #   if self.rule_constraint_type == 1:
        #     mask = self.get_rules_mask1(batch_size, nt_num_nodes, pt_num_nodes,
        #                                 nt_spans, pt_spans, device)
        #   elif self.rule_constraint_type == 2:
        #     mask = self.get_rules_mask2(batch_size, nt_num_nodes, pt_num_nodes,
        #                                 nt_spans, pt_spans, device)

        #   rules[:, src_nt_idx, src_nt_idx, src_nt_idx] += mask[:, :, :src_nt, :src_nt]
        #   rules[:, src_nt_idx, src_nt_idx, src_pt_idx] += mask[:, :, :src_nt, src_nt:]
        #   rules[:, src_nt_idx, src_pt_idx, src_nt_idx] += mask[:, :, src_nt:, :src_nt]
        #   rules[:, src_nt_idx, src_pt_idx, src_pt_idx] += mask[:, :, src_nt:, src_nt:]

        # rules = rules
        # rules = rules.view(batch_size, nt, (nt+pt)**2).log_softmax(2).view(
        #   batch_size, nt, nt+pt, nt+pt)

        terms = F.log_softmax(self.vocab_out(pt_emb), 2)

        if x is not None:
            n = x.size(1)
            terms = terms.unsqueeze(1).expand(batch_size, n, pt, terms.size(2))
            x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
            terms = torch.gather(terms, 3, x_expand).squeeze(3)
            if self.use_copy:
                copy_pt = (
                    torch.zeros(batch_size, n, pt).fill_(self.neg_huge * 0.1).to(device)
                )
                copy_pt_view = copy_pt[:, :, :src_pt].view(
                    batch_size, n, self.pt_states, pt_num_nodes
                )
                for b in range(batch_size):
                    for c, s in enumerate(pt_spans[b]):
                        if s[-1] == None:
                            continue
                        copy_str = " ".join(s[-1])
                        for j in range(n):
                            if x_str[b][j] == copy_str:
                                copy_pt_view[:, j, -1, c] = 0.0
                copy_mask = torch.zeros_like(copy_pt)
                copy_mask_view = copy_mask[:, :, :src_pt].view(
                    batch_size, n, self.pt_states, pt_num_nodes
                )
                # COPY is a special nonterminal
                copy_mask_view[:, :, -1].fill_(1.0)
                # copy_pt has binary weight
                terms = terms * (1 - copy_mask) + copy_pt * copy_mask
        # TODO: copy_nt
        params = {
            "term": terms,
            "root": roots,
            "left": rule_emb_left,
            "right": rule_emb_right,
            "head": rule_emb_parent,
        }
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes
