from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.fn import spans2tree
from ..components.common import MultiResidualLayer
from .base import TgtParserBase
from .neural_qcfg import NeuralQCFGTgtParser
from .struct.d1_pcfg_flex import D1PCFGFlex


def get_nn(dim, cpd_rank):
    return nn.Sequential(
        nn.LeakyReLU(), nn.Linear(dim, cpd_rank)  # , nn.LayerNorm(cpd_rank)
    )


def normalize(t: torch.Tensor):
    shape = t.shape
    return t.flatten(-2).softmax(-1).view(shape)


class NeuralQCFGD1FlexTgtParser(TgtParserBase):
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
        direction=0,
        cpd_rank=128,
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

        self.pcfg = D1PCFGFlex(self.nt_states, self.pt_states, direction)
        self.vocab = vocab
        self.dim = dim
        self.direction = direction
        self.cpd_rank = cpd_rank
        self.num_layers = num_layers
        self.src_dim = src_dim

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

        self.root_mlp_i = get_nn(dim, dim)
        self.root_mlp_j = get_nn(dim, dim)
        self.root_mlp_k = get_nn(dim, dim)
        self.rijk_weight = nn.Parameter(torch.empty(cpd_rank, dim, dim))
        self.ai_r_nn = get_nn(dim, cpd_rank)
        self.r_b_nn = get_nn(dim, cpd_rank)
        self.r_c_nn = get_nn(dim, cpd_rank)
        self.r_jk_nn = get_nn(dim, cpd_rank)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.src_nt_emb)
        nn.init.xavier_uniform_(self.src_pt_emb)
        nn.init.xavier_uniform_(self.rijk_weight)

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
        nt_node_emb = self.src_nt_node_mlp(nt_node_features)
        nt_state_emb = self.src_nt_emb.expand(batch_size, self.nt_states, self.dim)
        nt_emb = nt_state_emb.unsqueeze(2) + nt_node_emb.unsqueeze(1)
        nt = self.nt_states * nt_num_nodes

        pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        pt_state_emb = self.src_pt_emb.expand(batch_size, self.pt_states, self.dim)
        pt_emb = pt_state_emb.unsqueeze(2) + pt_node_emb.unsqueeze(1)
        pt = self.pt_states * pt_num_nodes

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
        state_emb = torch.cat([nt_state_emb, pt_state_emb], 1)
        node_emb = torch.cat([nt_node_emb, pt_node_emb], 1)
        rule_head = self.ai_r_nn(
            self.rule_mlp_parent(nt_emb.view(batch_size, -1, self.dim))
        ).softmax(-1)
        rule_left = self.r_b_nn(self.rule_mlp_left(state_emb)).transpose(1, 2)
        rule_right = self.r_c_nn(self.rule_mlp_right(state_emb)).transpose(1, 2)

        i = self.root_mlp_i(nt_node_emb)
        j = self.root_mlp_j(node_emb)
        k = self.root_mlp_k(node_emb)
        rule_slr = torch.einsum(
            "rab,xia,xjkb->xrijk",
            self.rijk_weight,
            F.leaky_relu(i),
            F.leaky_relu(j[:, :, None] + k[:, None, :]),
        )

        # fmt: off
        nt_mask = torch.arange(nt_num_nodes, device=i.device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=i.device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=i.device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=i.device).unsqueeze(1)
        # fmt: on
        valid_mask = torch.cat([nt_mask, pt_mask], dim=1)
        valid_mask = torch.einsum("bx,by,bz->bxyz", nt_mask, valid_mask, valid_mask)
        valid_mask = valid_mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
        rule_slr[~valid_mask] = self.neg_huge

        if self.rule_hard_constraint is not None:
            mask = self.rule_hard_constraint.get_mask(
                batch_size, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device
            )
            mask = mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
            rule_slr[~mask] = self.neg_huge

        if self.direction == 0:
            rule_left = rule_left.softmax(-1)
            rule_right = rule_right.softmax(-1)
            new_slr = torch.empty_like(rule_slr)
            nnn = nt_num_nodes
            new_slr[..., :nnn, :nnn] = normalize(rule_slr[..., :nnn, :nnn])
            new_slr[..., :nnn, nnn:] = normalize(rule_slr[..., :nnn, nnn:])
            new_slr[..., nnn:, :nnn] = normalize(rule_slr[..., nnn:, :nnn])
            new_slr[..., nnn:, nnn:] = normalize(rule_slr[..., nnn:, nnn:])
            rule_slr = new_slr
        else:
            new_left = torch.empty_like(rule_left)
            new_left[..., : self.nt_states] = rule_left[..., : self.nt_states].softmax(
                -1
            )
            new_left[..., self.nt_states :] = rule_left[..., self.nt_states :].softmax(
                -1
            )
            rule_left = new_left

            new_right = torch.empty_like(rule_right)
            new_right[..., : self.nt_states] = rule_right[
                ..., : self.nt_states
            ].softmax(-1)
            new_right[..., self.nt_states :] = rule_right[
                ..., self.nt_states :
            ].softmax(-1)
            rule_right = new_right

            shape = rule_slr.shape
            rule_slr = rule_slr.flatten(3).softmax(-1).view(shape).clone()

        mask = nt_mask.unsqueeze(1)
        mask = mask.expand(-1, rule_slr.shape[1], -1)
        rule_slr[~mask] = 0

        # A->a
        terms = self.vocab_out(pt_emb).log_softmax(-1)
        terms = terms.view(batch_size, -1, terms.shape[-1])

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
            "left": rule_left,
            "right": rule_right,
            "head": rule_head,
            "slr": rule_slr,
            "constraint": nt_constraint,
        }
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes

    def post_process_nt_constraint(self, constraint, device):
        constraint_ = []
        for value, mask in constraint:
            value = torch.from_numpy(value)
            mask = torch.from_numpy(mask)
            constraint_.append((value.to(device), mask.to(device)))
        return constraint_
