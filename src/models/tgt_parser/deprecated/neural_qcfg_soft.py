import copy
import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...datamodules.components.vocab import VocabularyPair
from .neural_qcfg import NeuralQCFGTgtParser

log = logging.getLogger(__file__)


class NeuralQCFGSoftTgtParser(NeuralQCFGTgtParser):
    """Impl of Yoon Kim's footnote 22.
    Instead of A[i] -> B[j]C[k], we only have A->BC, but emb of ABC are attentions
    to ijk. (I guess). I use ABC's emb as key. No constraint can be applied.
    """

    def __init__(self, att_dropout=0.0, att_num_heads=4, **kwargs):
        super(NeuralQCFGSoftTgtParser, self).__init__(**kwargs)
        assert self.rule_hard_constraint is None, "Do not support any constraint."
        assert self.rule_soft_constraint is None, "Do not support any constraint."
        assert not self.use_copy, "Do not support copy."
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

    def parse(self, x, lengths, node_features, spans, params=None, **kwargs):
        if params is None:
            params = self.get_params(node_features, spans, x, **kwargs)
        params, pt_spans, pt_num_nodes, pt_att, nt_spans, nt_num_nodes, nt_att = params

        out = self.pcfg(params, lengths, decode=True)

        pt_aligned = pt_att.argmax(-1).cpu().numpy()
        nt_aligned = nt_att.argmax(-1).cpu().numpy()

        # find alignments
        aligned_spans = []
        for b, (all_span, pt_span, nt_span) in enumerate(zip(out, pt_spans, nt_spans)):
            aligned_spans_item = []
            for l, r, label in all_span:
                if l == r:
                    aligned_spans_item.append(pt_span[pt_aligned[b, label]])
                else:
                    aligned_spans_item.append(nt_span[nt_aligned[b, label]])
            aligned_spans.append(aligned_spans_item)
        return out, aligned_spans, pt_spans, nt_spans

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
        device = nt_node_features.device

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
        nt = self.nt_states * nt_num_nodes
        pt = self.pt_states * pt_num_nodes
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
