import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.common import MultiResidualLayer
from ..struct.decomp7 import Decomp7, Decomp7Sampler
from ..struct.decomp7_fast import Decomp7Fast, Decomp7FastSampler
from .base import TgtParserBase, TgtParserPrediction

log = logging.getLogger(__file__)


class NeuralDecomp7Impl2TgtParser(TgtParserBase):
    def __init__(
        self,
        cpd_rank=32,
        vocab=100,
        dim=256,
        num_layers=3,
        src_dim=256,
        use_fast=False,
        tie_r=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab = vocab
        self.dim = dim
        self.src_dim = src_dim
        self.num_layers = num_layers
        self.cpd_rank = cpd_rank
        self.use_fast = use_fast

        self.src_nt_emb = nn.Parameter(torch.randn(self.nt_states, dim))
        self.src_nt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.src_pt_emb = nn.Parameter(torch.randn(self.pt_states, dim))
        self.src_pt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)

        self.root_mlp_child = nn.Linear(dim, 1, bias=False)
        self.vocab_out = MultiResidualLayer(dim, dim, out_dim=vocab, num_layers=num_layers)

        self.rule_mlp_parent = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.root_mlp_i = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_j = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_k = MultiResidualLayer(dim, dim, num_layers=num_layers)

        self.rj_b_nt = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rj_b_pt = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rk_c_nt = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rk_c_pt = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)

        if tie_r:
            _w, _b = self.rj_b_nt.out_linear.weight, self.rj_b_nt.out_linear.bias
            self.rj_b_pt.out_linear.weight = _w
            self.rj_b_pt.out_linear.bias = _b
            self.rk_c_nt.out_linear.weight = _w
            self.rk_c_nt.out_linear.bias = _b
            self.rk_c_pt.out_linear.weight = _w
            self.rk_c_pt.out_linear.bias = _b

        self.rijk_weight = nn.Parameter(torch.empty(cpd_rank, dim, dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.src_nt_emb.data)
        nn.init.xavier_uniform_(self.src_nt_emb.data)
        nn.init.xavier_uniform_(self.rijk_weight.data)

    def forward(self, node_features, spans, **kwargs):
        batch_size = len(spans)
        device = node_features[0].device

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

        nt = self.nt_states * nt_num_nodes
        pt = self.pt_states * pt_num_nodes

        # e = u + h
        nt_node_emb = self.src_nt_node_mlp(nt_node_features)
        nt_state_emb = self.src_nt_emb.expand(batch_size, self.nt_states, self.dim)
        nt_emb = nt_state_emb.unsqueeze(2) + nt_node_emb.unsqueeze(1)
        nt_emb = nt_emb.view(batch_size, nt, -1)

        pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        pt_state_emb = self.src_pt_emb.expand(batch_size, self.pt_states, self.dim)
        pt_emb = pt_state_emb.unsqueeze(2) + pt_node_emb.unsqueeze(1)
        pt_emb = pt_emb.view(batch_size, pt, -1)

        # S->A
        roots = self.root_mlp_child(nt_emb)
        roots = roots.view(batch_size, self.nt_states, nt_num_nodes)
        mask = torch.arange(nt_num_nodes, device=device).view(1, 1, -1).expand(batch_size, 1, -1)
        allowed = (torch.tensor(nt_num_nodes_list, device=device) - 1).view(-1, 1, 1)
        roots = torch.where(mask == allowed, roots, roots.new_tensor(self.neg_huge))
        roots = roots.view(batch_size, -1)
        roots = F.log_softmax(roots, 1)

        # A->BC
        reg = 0

        all_node_emb = torch.cat([nt_node_emb, pt_node_emb], 1)
        rule_head = self.rule_mlp_parent(nt_emb)
        if self.score_regularziation > 0:
            reg += torch.norm(rule_head)
        rule_head = rule_head.softmax(-1)

        i = self.root_mlp_i(nt_node_emb)
        j = self.root_mlp_j(all_node_emb)
        k = self.root_mlp_k(all_node_emb)

        num = max(self.nt_states, self.pt_states)
        if num != self.pt_states:
            new_pt_state_emb = torch.zeros_like(nt_state_emb)
            new_pt_state_emb[:, : self.pt_states] = pt_state_emb
            pt_state_emb = new_pt_state_emb
        elif num != self.nt_states:
            new_nt_state_emb = torch.zeros_like(pt_state_emb)
            new_nt_state_emb[:, : self.nt_states] = nt_state_emb
            nt_state_emb = new_nt_state_emb

        rj_b_nt = self.rj_b_nt(j[:, :nt_num_nodes, None] + nt_state_emb[:, None]).movedim(-1, 1)
        rj_b_pt = self.rj_b_pt(j[:, nt_num_nodes:, None] + pt_state_emb[:, None]).movedim(-1, 1)
        rk_c_nt = self.rk_c_nt(k[:, :nt_num_nodes, None] + nt_state_emb[:, None]).movedim(-1, 1)
        rk_c_pt = self.rk_c_pt(k[:, nt_num_nodes:, None] + pt_state_emb[:, None]).movedim(-1, 1)

        if self.score_regularziation > 0:
            reg += torch.norm(rj_b_nt) + torch.norm(rj_b_pt) + torch.norm(rk_c_nt) + torch.norm(rk_c_pt)
            rj_b_nt = rj_b_nt.clone()
            rj_b_pt = rj_b_pt.clone()
            rk_c_nt = rk_c_nt.clone()
            rk_c_pt = rk_c_pt.clone()

        if num != self.pt_states:
            rj_b_pt[..., self.pt_states :] = self.neg_huge
            rk_c_pt[..., self.pt_states :] = self.neg_huge
        elif num != self.nt_states:
            rj_b_nt[..., self.nt_states :] = self.neg_huge
            rk_c_nt[..., self.nt_states :] = self.neg_huge
        if self.use_fast:
            rj_b_nt = rj_b_nt.softmax(-1)
            rj_b_pt = rj_b_pt.softmax(-1)
            rk_c_nt = rk_c_nt.softmax(-1)
            rk_c_pt = rk_c_pt.softmax(-1)
        else:
            rj_b_nt = rj_b_nt.log_softmax(-1)
            rj_b_pt = rj_b_pt.log_softmax(-1)
            rk_c_nt = rk_c_nt.log_softmax(-1)
            rk_c_pt = rk_c_pt.log_softmax(-1)
        rule_left = torch.cat([rj_b_nt, rj_b_pt], dim=2)
        rule_right = torch.cat([rk_c_nt, rk_c_pt], dim=2)

        rule_slr = torch.einsum(
            "rab,xia,xjkb->xrijk",
            self.rijk_weight,
            F.leaky_relu(i),
            F.leaky_relu(j[:, :, None] + k[:, None, :]),
        )

        if self.score_regularziation > 0:
            reg += torch.norm(rule_slr)
            rule_slr = rule_slr.clone()

        # fmt: off
        nt_mask = torch.arange(nt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=device).unsqueeze(1)
        # fmt: on
        valid_mask = torch.cat([nt_mask, pt_mask], dim=1)
        # _mask = valid_mask.unsqueeze(1).expand(-1, self.cpd_rank, -1)
        # rule_left[~_mask] = 0
        # rule_right[~_mask] = 0

        valid_mask = torch.einsum("bx,by,bz->bxyz", nt_mask, valid_mask, valid_mask)
        valid_mask = valid_mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
        rule_slr[~valid_mask] = self.neg_huge

        if self.rule_hard_constraint is not None:
            mask = self.rule_hard_constraint.get_mask(
                batch_size, self.pt_states, self.nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device
            )
            mask = mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
            rule_slr[~mask] = self.neg_huge

        shape = rule_slr.shape
        rule_slr = rule_slr.flatten(3).softmax(-1).view(shape).clone()

        mask = nt_mask.unsqueeze(1)
        mask = mask.expand(-1, rule_slr.shape[1], -1)
        rule_slr[~valid_mask] = 0

        # A->a
        terms = F.log_softmax(self.vocab_out(pt_emb), 2)

        params = {
            "term": terms,
            "root": roots,
            "left": rule_left,
            "right": rule_right,
            "slr": rule_slr,
            "head": rule_head,
        }
        pred = TgtParserPrediction(
            batch_size=batch_size,
            nt=nt,
            nt_states=self.nt_states,
            nt_nodes=nt_spans,
            nt_num_nodes=nt_num_nodes,
            nt_features=nt_node_features,
            pt=pt,
            pt_states=self.pt_states,
            pt_nodes=pt_spans,
            pt_num_nodes=pt_num_nodes,
            pt_features=pt_node_features,
            params=params,
            device=device,
        )

        if self.score_regularziation > 0:
            pred.score_reg_loss = reg

        return pred

    def observe_x(self, pred: TgtParserPrediction, x, lengths, inplace=True, **kwargs) -> TgtParserPrediction:
        pred = super().observe_x(pred, x, lengths, inplace, **kwargs)
        pred.dist = (Decomp7Fast if self.use_fast else Decomp7)(pred.posterior_params, pred.lengths, **pred.common())
        return pred

    def prepare_sampler(self, pred: TgtParserPrediction, src, src_ids, inplace=True) -> TgtParserPrediction:
        pred = super().prepare_sampler(pred, src, src_ids, inplace)
        pred.sampler = (Decomp7FastSampler if self.use_fast else Decomp7Sampler)(
            pred.params, **pred.common(), **self.sampler_common()
        )
        return pred

    def post_process_nt_constraint(self, constraint, device):
        constraint_ = []
        for value, mask in constraint:
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
                mask = torch.from_numpy(mask)
            else:
                value = value.flatten(2)
                mask = mask.flatten(2)
            constraint_.append((value.to(device), mask.to(device)))
        return constraint_
