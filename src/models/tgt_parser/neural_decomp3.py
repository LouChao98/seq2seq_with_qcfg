import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.common import MultiResidualLayer
from .base import DirTgtParserPrediction, TgtParserBase
from .struct.decomp3 import Decomp3, Decomp3Sampler

log = logging.getLogger(__file__)


def normalize(t: torch.Tensor):
    shape = t.shape
    return t.flatten(-2).softmax(-1).view(shape)


class NeuralDecomp3TgtParser(TgtParserBase):
    def __init__(
        self,
        pt_states=1,
        nt_states=10,
        pt_span_range=(1, 1),
        nt_span_range=(2, 1000),
        cpd_rank=32,
        direction=0,
        use_copy=False,
        datamodule=None,
        rule_hard_constraint=None,
        rule_soft_constraint=None,
        rule_soft_constraint_solver=None,
        rule_reweight_constraint=None,
        generation_criteria="ppl",
        generation_max_length=40,
        generation_max_actions=80,
        generation_num_samples=10,
        generation_ppl_batch_size=None,
        generation_strict=False,
        vocab=100,
        dim=256,
        num_layers=3,
        src_dim=256,
    ):
        super().__init__(
            pt_states,
            nt_states,
            pt_span_range,
            nt_span_range,
            use_copy,
            datamodule,
            rule_hard_constraint,
            rule_soft_constraint,
            rule_soft_constraint_solver,
            rule_reweight_constraint,
            generation_criteria,
            generation_max_length,
            generation_max_actions,
            generation_num_samples,
            generation_ppl_batch_size,
            generation_strict,
        )

        self.vocab = vocab
        self.dim = dim
        self.src_dim = src_dim
        self.num_layers = num_layers
        self.cpd_rank = cpd_rank
        self.direction = direction

        self.src_nt_emb = nn.Parameter(torch.randn(nt_states, dim))
        self.src_nt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.src_pt_emb = nn.Parameter(torch.randn(pt_states, dim))
        self.src_pt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)

        self.root_mlp_child = nn.Linear(dim, 1, bias=False)
        self.vocab_out = MultiResidualLayer(dim, dim, out_dim=vocab, num_layers=num_layers)

        self.rule_mlp_parent = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rule_mlp_left = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rule_mlp_right = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.root_mlp_i = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_j = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_k = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rijk_weight = nn.Parameter(torch.empty(cpd_rank, dim, dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.src_nt_emb.data)
        nn.init.xavier_uniform_(self.src_pt_emb.data)
        nn.init.xavier_uniform_(self.rijk_weight)

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
        all_node_emb = torch.cat([nt_node_emb, pt_node_emb], 1)
        all_state_emb = torch.cat([nt_state_emb, pt_state_emb], 1)
        rule_head = self.rule_mlp_parent(nt_emb).softmax(-1)
        rule_left = self.rule_mlp_left(all_state_emb).transpose(1, 2)
        rule_right = self.rule_mlp_right(all_state_emb).transpose(1, 2)

        i = self.root_mlp_i(nt_node_emb)
        j = self.root_mlp_j(all_node_emb)
        k = self.root_mlp_k(all_node_emb)
        rule_slr = torch.einsum(
            "rab,xia,xjkb->xrijk",
            self.rijk_weight,
            F.leaky_relu(i),
            F.leaky_relu(j[:, :, None] + k[:, None, :]),
        )

        # fmt: off
        nt_mask = torch.arange(nt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=device).unsqueeze(1)
        # fmt: on
        valid_mask = torch.cat([nt_mask, pt_mask], dim=1)
        valid_mask = torch.einsum("bx,by,bz->bxyz", nt_mask, valid_mask, valid_mask)
        valid_mask = valid_mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
        rule_slr[~valid_mask] = self.neg_huge

        if self.rule_hard_constraint is not None:
            mask = self.rule_hard_constraint.get_mask(
                batch_size, self.pt_states, self.nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device
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
            new_left[..., : self.nt_states] = rule_left[..., : self.nt_states].softmax(-1)
            new_left[..., self.nt_states :] = rule_left[..., self.nt_states :].softmax(-1)
            rule_left = new_left

            new_right = torch.empty_like(rule_right)
            new_right[..., : self.nt_states] = rule_right[..., : self.nt_states].softmax(-1)
            new_right[..., self.nt_states :] = rule_right[..., self.nt_states :].softmax(-1)
            rule_right = new_right

            shape = rule_slr.shape
            rule_slr = rule_slr.flatten(3).softmax(-1).view(shape).clone()

        mask = nt_mask.unsqueeze(1)
        mask = mask.expand(-1, rule_slr.shape[1], -1)
        rule_slr[~mask] = 0

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
        pred = DirTgtParserPrediction(
            batch_size=batch_size,
            nt=nt,
            nt_states=self.nt_states,
            nt_nodes=nt_spans,
            nt_num_nodes=nt_num_nodes,
            pt=pt,
            pt_states=self.pt_states,
            pt_nodes=pt_spans,
            pt_num_nodes=pt_num_nodes,
            params=params,
            device=device,
            direction=self.direction,
        )
        return pred

    def observe_x(self, pred: DirTgtParserPrediction, x, lengths, inplace=True, **kwargs) -> DirTgtParserPrediction:
        pred = super().observe_x(pred, x, lengths, inplace, **kwargs)
        pred.dist = Decomp3(pred.posterior_params, pred.lengths, **pred.common())
        return pred

    def prepare_sampler(self, pred: DirTgtParserPrediction, src, src_ids, inplace=True) -> DirTgtParserPrediction:
        pred = super().prepare_sampler(pred, src, src_ids, inplace)
        pred.sampler = Decomp3Sampler(pred.params, **pred.common(), **self.sampler_common())
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
