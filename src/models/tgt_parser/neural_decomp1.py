import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.common import MultiResidualLayer
from .base import TgtParserBase, TgtParserPrediction
from .struct3.decomp1 import Decomp1, Decomp1Sampler

log = logging.getLogger(__file__)


class NeuralDecomp1TgtParser(TgtParserBase):
    def __init__(
        self,
        pt_states=1,
        nt_states=10,
        pt_span_range=(1, 1),
        nt_span_range=(2, 1000),
        cpd_rank=32,
        use_copy=False,
        datamodule=None,
        rule_hard_constraint=None,
        rule_soft_constraint=None,
        rule_soft_constraint_solver=None,
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
            generation_max_length,
            generation_max_actions,
            generation_num_samples,
            generation_ppl_batch_size,
            generation_strict,
        )

        assert self.rule_hard_constraint is None, "Do not support any constraint."
        assert self.rule_soft_constraint is None, "Do not support any constraint."

        self.vocab = vocab
        self.dim = dim
        self.src_dim = src_dim
        self.num_layers = num_layers
        self.cpd_rank = cpd_rank

        self.src_nt_emb = nn.Parameter(torch.randn(nt_states, dim))
        self.src_nt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.src_pt_emb = nn.Parameter(torch.randn(pt_states, dim))
        self.src_pt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)

        self.root_mlp_child = nn.Linear(dim, 1, bias=False)
        self.vocab_out = MultiResidualLayer(dim, dim, out_dim=vocab, num_layers=num_layers)

        self.rule_mlp_parent = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rule_mlp_left = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rule_mlp_right = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.src_nt_emb.data)
        nn.init.xavier_uniform_(self.src_pt_emb.data)

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
        all_emb = torch.cat([nt_emb, pt_emb], 1)
        rule_head = self.rule_mlp_parent(nt_emb)
        rule_left = self.rule_mlp_left(all_emb)
        rule_right = self.rule_mlp_right(all_emb)

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
        rule_left[~mask] = self.neg_huge
        rule_right[~mask] = self.neg_huge

        rule_head = rule_head.log_softmax(-1)
        rule_left = rule_left.transpose(1, 2).log_softmax(-1)
        rule_right = rule_right.transpose(1, 2).log_softmax(-1)

        # A->a
        terms = F.log_softmax(self.vocab_out(pt_emb), 2)

        params = {
            "term": terms,
            "root": roots,
            "left": rule_left,
            "right": rule_right,
            "head": rule_head,
        }
        pred = TgtParserPrediction(
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
        )
        return pred

    def observe_x(self, pred: TgtParserPrediction, x, lengths, inplace=True, **kwargs) -> TgtParserPrediction:
        pred = super().observe_x(pred, x, lengths, inplace, **kwargs)
        pred.dist = Decomp1(pred.posterior_params, pred.lengths, **pred.common())
        return pred

    def prepare_sampler(self, pred: TgtParserPrediction, src, src_ids, inplace=True) -> TgtParserPrediction:
        pred = super().prepare_sampler(pred, src, src_ids, inplace)
        pred.sampler = Decomp1Sampler(pred.params, **pred.common(), **self.sampler_common())
        return pred
