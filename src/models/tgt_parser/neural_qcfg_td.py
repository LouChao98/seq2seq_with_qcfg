from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .neural_qcfg import NeuralQCFGTgtParser
from .struct.td_pcfg import FastestTDPCFG


class NeuralQCFGDecomp1TgtParser(NeuralQCFGTgtParser):
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
        cpd_rank=32,
    ):
        super().__init__(
            vocab,
            dim,
            num_layers,
            src_dim,
            nt_states,
            pt_states,
            rule_constraint_type,
            use_copy,
            nt_span_range,
            pt_span_range,
            num_samples,
            check_ppl,
        )

        self.pcfg = FastestTDPCFG()
        self.rank_proj_head = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(dim, cpd_rank), nn.LayerNorm(cpd_rank)
        )
        self.rank_proj_left = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(dim, cpd_rank), nn.LayerNorm(cpd_rank)
        )
        self.rank_proj_right = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(dim, cpd_rank), nn.LayerNorm(cpd_rank)
        )

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
        src_nt_emb = self.src_nt_emb.expand(batch_size, self.nt_states, self.dim)
        src_nt_emb = src_nt_emb.unsqueeze(2) + src_nt_node_emb.unsqueeze(1)
        src_nt_emb = src_nt_emb.view(batch_size, self.nt_states * nt_num_nodes, -1)
        nt_emb = src_nt_emb

        src_pt_node_emb = self.src_pt_node_mlp(pt_node_features)
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
        rule_emb_head = self.rank_proj_head(self.rule_mlp_parent(nt_emb))
        rule_emb_left = self.rank_proj_left(self.rule_mlp_left(all_emb))
        rule_emb_right = self.rank_proj_right(self.rule_mlp_right(all_emb))

        # fmt: off
        device = all_emb.device
        nt_mask = torch.arange(nt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=device).unsqueeze(1)
        nt_mask = nt_mask.unsqueeze(1).expand(-1, self.nt_states, -1).reshape(batch_size, -1)
        pt_mask = pt_mask.unsqueeze(1).expand(-1, self.pt_states, -1).reshape(batch_size, -1)
        # fmt: on
        mask = torch.cat([nt_mask, pt_mask], dim=1)
        rule_emb_left[~mask] = self.neg_huge
        rule_emb_right[~mask] = self.neg_huge

        rule_emb_head = rule_emb_head.softmax(-1)
        rule_emb_left = rule_emb_left.softmax(-2)
        rule_emb_right = rule_emb_right.softmax(-2)

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

        params = {
            "term": terms,
            "root": roots,
            "left": rule_emb_left,
            "right": rule_emb_right,
            "head": rule_emb_head,
        }
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes
