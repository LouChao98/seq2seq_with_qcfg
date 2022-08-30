import copy
import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ...datamodules.components.vocab import VocabularyPair
from ..components.common import MultiResidualLayer
from .neural_qcfg import NeuralQCFGTgtParser
from .struct.pcfg import PCFG, TokenType

log = logging.getLogger(__file__)


class NeuralQCFGSoftTgtParser(NeuralQCFGTgtParser):
    """Impl of Yoon Kim's footnote 22.
    Instead of A[i] -> B[j]C[k], we only have A->BC, but emb of ABC are attentions
    to ijk. (I guess). I use ABC's emb as key. No constraint can be applied.
    """

    def __init__(self, att_dropout=0.0, att_num_heads=4, **kwargs):
        super(NeuralQCFGSoftTgtParser, self).__init__(**kwargs)
        assert self.rule_constraint_type == 0, "Do not support any constraint."
        assert not self.use_copy, "Do not support copy."  # TODO we can impl this
        self.att_dropout = att_dropout
        self.att_num_heads = att_num_heads
        self.att = nn.MultiheadAttention(
            self.src_dim,
            self.att_num_heads,
            self.att_dropout,
            kdim=self.dim,
            vdim=self.dim,
            batch_first=True,
        )

    def parse(self, x, lengths, node_features, spans, copy_position=None):
        (
            params,
            pt_spans,
            pt_num_nodes,
            pt_att,
            nt_spans,
            nt_num_nodes,
            nt_att,
        ) = self.get_params(node_features, spans, x, copy_position)
        out = self.pcfg(params, lengths, True)

        pt_aligned = pt_att.argmax(-1).cpu().numpy()
        nt_aligned = nt_att.argmax(-1).cpu().numpy()

        # out: list of list, containing spans (i, j, label)
        all_spans_node = []
        for b, (all_span, pt_span, nt_span) in enumerate(zip(out, pt_spans, nt_spans)):
            all_span_node = []
            for l, r, label in all_span:
                if l == r:
                    all_span_node.append(pt_span[pt_aligned[b, label]])
                else:
                    all_span_node.append(nt_span[nt_aligned[b, label]])

            all_spans_node.append(all_span_node)
        return out, all_spans_node, pt_spans, nt_spans

    def generate(
        self,
        node_features,
        spans,
        vocab_pair: VocabularyPair,
        src_ids: torch.Tensor,
        src: List[List[str]],
    ):
        # if check_ppl=True, I will compute ppl for samples, return the one with minimum ppl
        # else, just return the one with the maximum score

        (
            params,
            pt_spans,
            pt_num_nodes,
            pt_att,
            nt_spans,
            nt_num_nodes,
            nt_att,
        ) = self.get_params(node_features, spans)

        max_len = 30
        preds = self.pcfg.sampled_decoding(
            params,
            nt_spans,
            self.nt_states,
            pt_spans,
            self.pt_states,
            num_samples=self.num_samples,
            use_copy=self.use_copy,
            max_length=max_len,
        )

        # expand copied spans and build copy_position
        preds_ = []
        src_ids = src_ids.tolist()
        for batch in preds:
            expanded_batch = []
            for inst in batch:
                expanded_batch.append((inst[0], inst[2]))
            preds_.append(expanded_batch)

        preds = preds_

        if self.check_ppl:
            padid = vocab_pair.tgt.pad_token_id or 0
            new_preds = []
            for i, preds_one_inp in enumerate(preds):
                to_keep = [1 < len(inst[0]) <= 60 for inst in preds_one_inp]
                _ids = [inst[0] for inst, flag in zip(preds_one_inp, to_keep) if flag]

                sort_id = list(range(len(_ids)))
                sort_id.sort(key=lambda x: len(_ids[x]), reverse=True)
                _ids = [_ids[i] for i in sort_id]
                _lens = [len(inst) for inst in _ids]
                _ids_t = torch.full((len(_ids), _lens[0]), padid)
                for j, (snt, length) in enumerate(zip(_ids, _lens)):
                    _ids_t[j, :length] = torch.tensor(snt)
                _ids_t = _ids_t.to(node_features[0].device)

                batch_size = (
                    len(node_features)
                    if self.check_ppl_batch_size is None
                    else self.check_ppl_batch_size
                )
                ppl = []
                for j in range(0, len(_ids), batch_size):
                    real_batch_size = min(batch_size, len(_ids) - j)
                    _node_ft = [node_features[i] for _ in range(real_batch_size)]
                    _spans = [spans[i] for _ in range(real_batch_size)]
                    max_len = max(_lens[j : j + batch_size])
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
                new_preds.append((_ids[chosen], ppl[chosen], None))
            preds = new_preds
        else:
            assert False, "Bad impl of score. see sample()."
            preds_ = []
            for item in preds:
                item = max(item, key=lambda x: x[1])
                preds_.append(item)
            preds = preds_

        pred_strings = []
        for pred, src_sent in zip(preds, src):
            snt, score, copy_unk = pred
            try:
                sent = vocab_pair.tgt.convert_ids_to_tokens(snt)
                if copy_unk is not None:
                    for t, s in copy_unk.items():
                        sent[t] = src_sent[s]
                pred_strings.append((sent, score))
            except IndexError:
                print("Bad pred:", snt)
                pred_strings.append([("", -999)])
        return pred_strings

    def get_params(
        self,
        node_features,
        spans,
        x: Optional[torch.Tensor] = None,
        copy_position=None,  # (pt, nt)
    ):
        if copy_position is None or not self.use_copy:
            copy_position = (None, None)

        batch_size = len(spans)

        (
            nt_spans,
            nt_num_nodes_list,
            nt_num_nodes,
            nt_node_features,
            pt_spans,
            pt_num_nodes_list,
            pt_num_nodes,
            pt_node_features,
        ) = self.build_src_features(spans, node_features)
        device = node_features[0][0].device

        # fmt: off
        nt_mask = torch.arange(nt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=device).unsqueeze(1)

        # e = u + h
        nt_emb, nt_att = self.att(
            self.src_nt_emb.unsqueeze(0).expand(batch_size, -1, -1),
            nt_node_features,
            nt_node_features,
            ~nt_mask,
            average_attn_weights=True,
        )
        pt_emb, pt_att = self.att(
            self.src_pt_emb.unsqueeze(0).expand(batch_size, -1, -1),
            pt_node_features,
            pt_node_features,
            ~pt_mask,
            average_attn_weights=True,
        )

        # S->A
        roots = self.root_mlp_child(nt_emb)
        roots = roots.view(batch_size, self.nt_states)
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
        rules = (
            torch.matmul(rule_emb_parent, rule_emb_child.transpose(1, 2))
            .log_softmax(-1)
        )
        rules = rules.view(batch_size, nt, nt + pt, nt + pt)

        # A->a
        terms = F.log_softmax(self.vocab_out(pt_emb), 2)

        if x is not None:
            n = x.size(1)
            terms = terms.unsqueeze(1).expand(batch_size, n, pt, terms.size(2))
            x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
            terms = torch.gather(terms, 3, x_expand).squeeze(3)


        params = {"term": terms, "root": roots, "rule": rules}
        return params, pt_spans, pt_num_nodes, pt_att, nt_spans, nt_num_nodes, nt_att
