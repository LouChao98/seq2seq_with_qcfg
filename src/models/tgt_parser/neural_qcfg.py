from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ...datamodules.components.vocab import Vocabulary
from ..components.common import MultiResidualLayer
from .base import TgtParserBase
from .struct.pcfg import PCFG


class NeuralQCFGTgtParser(TgtParserBase):
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
    ):
        super(NeuralQCFGTgtParser, self).__init__()
        self.neg_huge = -1e5
        self.pcfg = PCFG()
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
            dim, dim, out_dim=vocab, num_layers=num_layers
        )

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

    def generate(
        self,
        node_features,
        spans,
        tokenizer: Vocabulary,
        src: Optional[List[str]] = None,
    ):
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

        # TODO handle copy
        preds_ = []
        for item in preds:
            # here just drop types
            preds_.append((item[0], item[2]))
        preds = preds_

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
                _ids_t = torch.full((len(_ids), _lens[0]), tokenizer.pad_token_id or 0)
                for j, (snt, length) in enumerate(zip(_ids, _lens)):
                    _ids_t[j, :length] = torch.tensor(snt)
                _ids_t = _ids_t.to(node_features[0].device)

                batch_size = len(node_features)
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
            assert False, "Bad impl of score. see sample."
            preds_ = []
            for item in preds:
                item = max(item, key=lambda x: x[1])
                preds_.append(item)
            preds = preds_

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
        ignore_src=False,
    ):
        batch_size = len(spans)

        # seperate nt and pt features according to span width
        # TODO sanity check: the root node must be the last element of nt_spans.
        # NOTE TreeLSTM guarantees this.
        pt_node_features, nt_node_features = [], []
        pt_spans, nt_spans = [], []
        for spans_inst, node_features_inst in zip(spans, node_features):
            pt_node_feature = []
            nt_node_feature = []
            pt_span = []
            nt_span = []
            for i, s in enumerate(spans_inst):
                s_len = s[1] - s[0] + 1
                if s_len >= self.nt_span_range[0] and s_len <= self.nt_span_range[1]:
                    nt_node_feature.append(node_features_inst[i])
                    nt_span.append(s)
                if s_len >= self.pt_span_range[0] and s_len <= self.pt_span_range[1]:
                    pt_node_feature.append(node_features_inst[i])
                    pt_span.append(s)
            if len(nt_node_feature) == 0:
                nt_node_feature.append(node_features_inst[-1])
                nt_span.append(spans_inst[-1])
            pt_node_features.append(torch.stack(pt_node_feature))
            nt_node_features.append(torch.stack(nt_node_feature))
            pt_spans.append(pt_span)
            nt_spans.append(nt_span)
        nt_num_nodes_list = [len(inst) for inst in nt_node_features]
        pt_num_nodes_list = [len(inst) for inst in pt_node_features]
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
        src_nt_node_emb = self.src_nt_node_mlp(nt_node_features)
        if ignore_src:
            src_nt_node_emb.zero_()
        src_nt_emb = self.src_nt_emb.expand(batch_size, self.nt_states, self.dim)
        src_nt_emb = src_nt_emb.unsqueeze(2) + src_nt_node_emb.unsqueeze(1)
        src_nt_emb = src_nt_emb.view(batch_size, self.nt_states * nt_num_nodes, -1)
        nt_emb = src_nt_emb

        src_pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        if ignore_src:
            src_pt_node_emb.zero_()
        src_pt_emb = self.src_pt_emb.expand(batch_size, self.pt_states, self.dim)
        src_pt_emb = src_pt_emb.unsqueeze(2) + src_pt_node_emb.unsqueeze(1)
        src_pt_emb = src_pt_emb.view(batch_size, self.pt_states * pt_num_nodes, -1)
        pt_emb = src_pt_emb

        # S->A
        roots = self.root_mlp_child(nt_emb)
        roots = roots.view(batch_size, self.nt_states, nt_num_nodes)
        mask = (
            torch.arange(nt_num_nodes, device=device)
            .view(1, 1, -1)
            .expand(batch_size, 1, -1)
        )
        allowed = (torch.tensor(nt_num_nodes_list, device=device) - 1).view(-1, 1, 1)
        roots = torch.where(mask == allowed, roots, roots.new_full((1,), self.neg_huge))
        roots = roots.view(batch_size, -1)
        roots = F.log_softmax(roots, 1)

        # A->BC
        nt = nt_emb.size(1)
        pt = pt_emb.size(1)
        all_emb = torch.cat([nt_emb, pt_emb], 1)
        rule_emb_parent = self.rule_mlp_parent(nt_emb)  # b x nt_all x dm
        rule_emb_left = self.rule_mlp_left(all_emb)
        rule_emb_right = self.rule_mlp_right(all_emb)

        rule_emb_child = rule_emb_left[:, :, None, :] + rule_emb_right[:, None, :, :]
        rule_emb_child = rule_emb_child.view(batch_size, (nt + pt) ** 2, self.dim)
        rules = torch.matmul(rule_emb_parent, rule_emb_child.transpose(1, 2))
        rules = rules.view(batch_size, nt, nt + pt, nt + pt)

        # fmt: off
        # TODO lhs_mask seems to be unnecessary.
        nt_mask = torch.arange(nt_num_nodes, device=rules.device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=rules.device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=rules.device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=rules.device).unsqueeze(1)
        lhs_mask = nt_mask.unsqueeze(1).expand(-1, self.nt_states, -1).reshape(batch_size, -1)
        _pt_rhs_mask = pt_mask.unsqueeze(1).expand(-1, self.pt_states, -1).reshape(batch_size, -1)
        # fmt: on
        rhs_mask = torch.cat([lhs_mask, _pt_rhs_mask], dim=1)
        mask = torch.einsum("bx,by,bz->bxyz", lhs_mask, rhs_mask, rhs_mask)
        rules[~mask] = self.neg_huge

        if self.rule_constraint_type > 0:
            if self.rule_constraint_type == 1:
                mask = self.get_rules_mask1(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )
            elif self.rule_constraint_type == 2:
                mask = self.get_rules_mask2(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )
            rules[~mask] = self.neg_huge

        rules = (
            rules.view(batch_size, nt, (nt + pt) ** 2)
            .log_softmax(2)
            .view(batch_size, nt, nt + pt, nt + pt)
        )

        # A->a
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
                copy_pt_view = copy_pt[:, :, :pt].view(
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
                copy_mask_view = copy_mask[:, :, :pt].view(
                    batch_size, n, self.pt_states, pt_num_nodes
                )
                # COPY is a special nonterminal
                copy_mask_view[:, :, -1].fill_(1.0)
                # copy_pt has binary weight
                terms = terms * (1 - copy_mask) + copy_pt * copy_mask

        params = {"term": terms, "root": roots, "rule": rules}
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes

    def get_rules_mask1(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # A[a i]->B[a j] C[a k], a i must be the parent of a j and a k.
        # return 1 for not masked
        nt = nt_num_nodes * self.nt_states
        pt = pt_num_nodes * self.pt_states
        nt_node_mask = torch.ones(
            batch_size, nt_num_nodes, nt_num_nodes, dtype=torch.bool
        )
        pt_node_mask = torch.ones(
            batch_size, nt_num_nodes, pt_num_nodes, dtype=torch.bool
        )

        def is_parent(parent, child):
            return child[0] >= parent[0] and child[1] <= parent[1]

        # TODO vectorization
        for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):
            for i, parent_span in enumerate(nt_span):
                for j, child_span in enumerate(nt_span):
                    if not (is_parent(parent_span, child_span)):
                        nt_node_mask[b, i, j].fill_(0.0)
                for j, child_span in enumerate(pt_span):
                    if not (is_parent(parent_span, child_span)):
                        pt_node_mask[b, i, j].fill_(0.0)

        nt_node_mask = (
            nt_node_mask[:, None, :, None, :]
            .expand(
                batch_size, self.nt_states, nt_num_nodes, self.nt_states, nt_num_nodes,
            )
            .contiguous()
        )
        pt_node_mask = (
            pt_node_mask[:, None, :, None, :]
            .expand(
                batch_size, self.nt_states, nt_num_nodes, self.pt_states, pt_num_nodes,
            )
            .contiguous()
        )

        nt_node_mask = nt_node_mask.view(batch_size, nt, nt)
        pt_node_mask = pt_node_mask.view(batch_size, nt, pt)
        node_mask = torch.cat([nt_node_mask, pt_node_mask], 2).to(device)
        node_mask = node_mask.unsqueeze(3) * node_mask.unsqueeze(2)
        return node_mask.view(batch_size, nt, nt + pt, nt + pt)

    def get_rules_mask2(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # A[a i]->B[a j] C[a k], a i must be the DIRECT parent of a j and a k, j!=k.
        #   if a i has no child, a j/k = a i.
        # TODO review this comment
        nt = nt_num_nodes * self.nt_states
        pt = pt_num_nodes * self.pt_states
        bsz = batch_size
        src_nt = self.nt_states
        src_pt = self.pt_states
        node_nt = nt_num_nodes
        node_pt = pt_num_nodes
        node_mask = torch.zeros(
            bsz,
            src_nt * node_nt,
            src_nt * node_nt + src_pt * node_pt,
            src_nt * node_nt + src_pt * node_pt,
        ).to(device)

        nt_idx = slice(0, src_nt * node_nt)
        pt_idx = slice(src_nt * node_nt, src_nt * node_nt + src_pt * node_pt)

        nt_ntnt = node_mask[:, nt_idx, nt_idx, nt_idx].view(
            bsz, src_nt, node_nt, src_nt, node_nt, src_nt, node_nt
        )
        nt_ntpt = node_mask[:, nt_idx, nt_idx, pt_idx].view(
            bsz, src_nt, node_nt, src_nt, node_nt, src_pt, node_pt
        )
        nt_ptnt = node_mask[:, nt_idx, pt_idx, nt_idx].view(
            bsz, src_nt, node_nt, src_pt, node_pt, src_nt, node_nt
        )
        nt_ptpt = node_mask[:, nt_idx, pt_idx, pt_idx].view(
            bsz, src_nt, node_nt, src_pt, node_pt, src_pt, node_pt
        )

        def is_parent(parent, child):
            if child[0] >= parent[0] and child[1] <= parent[1]:
                return True
            else:
                return False

        def is_strict_parent(parent, child):
            return is_parent(parent, child) and parent != child

        def span_len(span):
            return span[1] - span[0] + 1

        def covers(parent, child1, child2):
            return (span_len(parent) == (span_len(child1) + span_len(child2))) and (
                (parent[0] == child1[0] and parent[1] == child2[1])
                or (parent[0] == child2[0] and parent[1] == child1[1])
            )

        # fill_(1.) = not masked
        for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):
            min_nt_span = min([span_len(s) for s in nt_span])
            for i, parent in enumerate(nt_span):
                if span_len(parent) == min_nt_span:
                    nt_ntnt[b, :, i, :, i, :, i].fill_(1.0)
                    for j, child in enumerate(pt_span):
                        if is_strict_parent(parent, child):
                            nt_ntpt[b, :, i, :, i, :, j].fill_(1.0)
                            nt_ptnt[b, :, i, :, j, :, i].fill_(1.0)
                if span_len(parent) == 1:
                    for j, child in enumerate(pt_span):
                        if parent == child:
                            nt_ptnt[b, :, i, :, j, :, i].fill_(1.0)
                            nt_ntpt[b, :, i, :, i, :, j].fill_(1.0)
                            nt_ptpt[b, :, i, :, j, :, j].fill_(1.0)
                for j, child1 in enumerate(nt_span):
                    for k, child2 in enumerate(nt_span):
                        if covers(parent, child1, child2):
                            nt_ntnt[b, :, i, :, j, :, k].fill_(1.0)
                            nt_ntnt[b, :, i, :, k, :, j].fill_(1.0)
                    for k, child2 in enumerate(pt_span):
                        if covers(parent, child1, child2):
                            nt_ntpt[b, :, i, :, j, :, k].fill_(1.0)
                            nt_ptnt[b, :, i, :, k, :, j].fill_(1.0)
                for j, child1 in enumerate(pt_span):
                    for k, child2 in enumerate(pt_span):
                        if covers(parent, child1, child2):
                            nt_ptpt[b, :, i, :, j, :, k].fill_(1.0)
                            nt_ptpt[b, :, i, :, k, :, j].fill_(1.0)

        node_mask = (1.0 - node_mask) * self.neg_huge

        return node_mask.contiguous().view(batch_size, nt, nt + pt, nt + pt)

