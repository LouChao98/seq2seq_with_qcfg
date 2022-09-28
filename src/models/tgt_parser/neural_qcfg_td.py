from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..components.common import MultiResidualLayer
from .base import TgtParserBase
from .struct.td_pcfg import FastestTDPCFG


class NeuralQCFGDecomp1TgtParser(TgtParserBase):
    def __init__(
        self,
        pt_states=1,
        nt_states=10,
        pt_span_range=(1, 1),
        nt_span_range=(2, 1000),
        use_copy=False,
        vocab_pair=None,
        rule_hard_constraint=None,
        rule_soft_constraint=None,
        rule_soft_constraint_solver=None,
        generation_max_length=40,
        generation_num_samples=10,
        generation_ppl_batch_size=None,
        vocab=100,
        dim=256,
        cpd_rank=64,
        num_layers=3,
        src_dim=256,
    ):
        super().__init__(
            pt_states,
            nt_states,
            pt_span_range,
            nt_span_range,
            use_copy,
            vocab_pair,
            rule_hard_constraint,
            rule_soft_constraint,
            rule_soft_constraint_solver,
            generation_max_length,
            generation_num_samples,
            generation_ppl_batch_size,
        )
        assert self.rule_hard_constraint is None, "Do not support any constraint."
        assert self.rule_soft_constraint is None, "Do not support any constraint."

        self.pcfg = FastestTDPCFG()
        self.vocab = vocab
        self.dim = dim
        self.cpd_rank = cpd_rank
        self.src_dim = src_dim
        self.num_layers = num_layers

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
        self.rank_proj_head = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(dim, cpd_rank)  # , nn.LayerNorm(cpd_rank)
        )
        self.rank_proj_left = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(dim, cpd_rank)  # , nn.LayerNorm(cpd_rank)
        )
        self.rank_proj_right = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(dim, cpd_rank)  # , nn.LayerNorm(cpd_rank)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.src_nt_emb.data)
        nn.init.xavier_uniform_(self.src_pt_emb.data)

    def get_params(
        self,
        node_features,
        spans,
        x: Optional[torch.Tensor] = None,
        copy_position=None,  # (pt, nt)
        impossible_span_mask=None,
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

        # e = u + h
        src_nt_node_emb = self.src_nt_node_mlp(nt_node_features)
        src_nt_emb = self.src_nt_emb.expand(batch_size, self.nt_states, self.dim)
        nt_emb = src_nt_emb.unsqueeze(2) + src_nt_node_emb.unsqueeze(1)
        nt_emb = nt_emb.view(batch_size, self.nt_states * nt_num_nodes, -1)

        src_pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        src_pt_emb = self.src_pt_emb.expand(batch_size, self.pt_states, self.dim)
        pt_emb = src_pt_emb.unsqueeze(2) + src_pt_node_emb.unsqueeze(1)
        pt_emb = pt_emb.view(batch_size, self.pt_states * pt_num_nodes, -1)

        # S->A
        roots = self.root_mlp_child(nt_emb)
        roots = roots.view(batch_size, self.nt_states, nt_num_nodes)
        mask = (
            torch.arange(nt_num_nodes, device=device)
            .view(1, 1, -1)
            .expand(batch_size, 1, -1)
        )
        allowed = (torch.tensor(nt_num_nodes_list, device=device) - 1).view(-1, 1, 1)
        roots = torch.where(mask == allowed, roots, roots.new_tensor(self.neg_huge))
        roots = roots.view(batch_size, -1)
        roots = F.log_softmax(roots, 1)

        # A->BC
        nt = self.nt_states * nt_num_nodes
        pt = self.pt_states * pt_num_nodes
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

        nt_constraint = None
        if x is not None:
            terms, roots, nt_constraint, _, _ = self.build_rules_give_tgt(
                x,
                terms,
                roots,
                pt_num_nodes,
                pt_spans,
                nt_num_nodes,
                nt_spans,
                pt,
                nt,
                pt_copy=copy_position[0],
                nt_copy=copy_position[1],
                observed_mask=impossible_span_mask,
            )

        params = {
            "term": terms,
            "root": roots,
            "left": rule_emb_left,
            "right": rule_emb_right,
            "head": rule_emb_head,
            "constraint": nt_constraint,
        }
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes
