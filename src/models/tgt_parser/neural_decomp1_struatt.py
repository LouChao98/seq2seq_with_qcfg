import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..components.common import MultiResidualLayer
from ..struct.decomp1 import Decomp1, Decomp1Sampler
from ..struct.decomp1_fast import Decomp1Fast
from .base import TgtParserBase, TgtParserPrediction

log = logging.getLogger(__file__)


class NeuralDecomp1StruAttTgtParser(TgtParserBase):
    def __init__(
        self,
        cpd_rank=32,
        vocab=100,
        dim=256,
        num_layers=3,
        src_dim=256,
        ignore_weight=False,
        tie_r=False,
        use_fast=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert self.rule_hard_constraint is None, "Do not support any constraint."
        assert self.rule_soft_constraint is None, "Do not support any constraint."

        self.vocab = vocab
        self.dim = dim
        self.src_dim = src_dim
        self.num_layers = num_layers
        self.cpd_rank = cpd_rank
        self.ignore_weight = ignore_weight
        self.tie_r = tie_r
        self.use_fast = use_fast

        self.src_nt_emb = nn.Parameter(torch.randn(self.nt_states, dim))
        self.src_nt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.src_pt_emb = nn.Parameter(torch.randn(self.pt_states, dim))
        self.src_pt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)

        self.root_mlp_child = nn.Linear(dim, 1, bias=False)
        self.vocab_out = MultiResidualLayer(dim, dim, out_dim=vocab, num_layers=num_layers)

        self.rule_mlp_parent = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rule_mlp_left = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rule_mlp_right = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)

        if tie_r:
            _w, _b = self.rule_mlp_parent.out_linear.weight, self.rule_mlp_parent.out_linear.bias
            self.rule_mlp_left.out_linear.weight = _w
            self.rule_mlp_left.out_linear.bias = _b
            self.rule_mlp_right.out_linear.weight = _w
            self.rule_mlp_right.out_linear.bias = _b

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.src_nt_emb.data)
        nn.init.xavier_uniform_(self.src_pt_emb.data)

    def build_src_features_with_weight(self, spans, node_features, weights):
        # seperate nt and pt features according to span width
        pt_node_features, nt_node_features = [], []
        pt_spans, nt_spans = [], []
        pt_weights, nt_weights = [], []
        for spans_item, node_features_item, weights_item in zip(spans, node_features, weights):
            pt_node_feature = []
            nt_node_feature = []
            pt_span = []
            nt_span = []
            pt_weight = []
            nt_weight = []
            for s, f, w in zip(spans_item, node_features_item, weights_item):
                s_len = s[1] - s[0]
                if self.nt_span_range[0] <= s_len <= self.nt_span_range[1]:
                    nt_node_feature.append(f)
                    nt_span.append(s)
                    nt_weight.append(w)
                if self.pt_span_range[0] <= s_len <= self.pt_span_range[1]:
                    pt_node_feature.append(f)
                    pt_span.append(s)
                    pt_weight.append(w)
            self.sanity_check_spans(nt_span, pt_span)
            pt_node_features.append(torch.stack(pt_node_feature))
            nt_node_features.append(torch.stack(nt_node_feature))
            pt_spans.append(pt_span)
            nt_spans.append(nt_span)
            pt_weights.append(torch.stack(pt_weight))
            nt_weights.append(torch.stack(nt_weight))
        nt_num_nodes_list = [len(item) for item in nt_node_features]
        pt_num_nodes_list = [len(item) for item in pt_node_features]
        nt_node_features = pad_sequence(nt_node_features, batch_first=True, padding_value=0.0)
        pt_node_features = pad_sequence(pt_node_features, batch_first=True, padding_value=0.0)
        nt_weights = pad_sequence(nt_weights, batch_first=True, padding_value=0.0).clamp(1e-32).log()
        pt_weights = pad_sequence(pt_weights, batch_first=True, padding_value=0.0).clamp(1e-32).log()
        if self.ignore_weight:
            nt_weights.zero_()
            pt_weights.zero_()
        pt_num_nodes = pt_node_features.size(1)
        nt_num_nodes = nt_node_features.size(1)
        return (
            nt_spans,
            nt_num_nodes_list,
            nt_num_nodes,
            nt_node_features,
            nt_weights,
            pt_spans,
            pt_num_nodes_list,
            pt_num_nodes,
            pt_node_features,
            pt_weights,
        )

    def forward(self, node_features, spans, weights, **kwargs):
        batch_size = len(spans)
        device = node_features[0].device

        (
            nt_spans,
            nt_num_nodes_list,
            nt_num_nodes,
            nt_node_features,
            nt_weights,
            pt_spans,
            pt_num_nodes_list,
            pt_num_nodes,
            pt_node_features,
            pt_weights,
        ) = self.build_src_features_with_weight(spans, node_features, weights)

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
        roots = F.log_softmax(roots, 1) + nt_weights.unsqueeze(1)
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

        ntw = nt_weights.unsqueeze(1).repeat(1, self.nt_states, 1).view(batch_size, -1)
        ptw = pt_weights.unsqueeze(1).repeat(1, self.pt_states, 1).view(batch_size, -1)
        w = torch.cat([ntw, ptw], dim=1).unsqueeze(-1)
        rule_left = rule_left.log_softmax(1) + w
        rule_right = rule_right.log_softmax(1) + w

        rule_left[~mask] = self.neg_huge
        rule_right[~mask] = self.neg_huge

        if self.use_fast:
            rule_head = rule_head.softmax(-1)
            rule_left = rule_left.transpose(1, 2).softmax(-1)
            rule_right = rule_right.transpose(1, 2).softmax(-1)
        else:
            rule_head = rule_head.log_softmax(-1)
            rule_left = rule_left.transpose(1, 2).log_softmax(-1)
            rule_right = rule_right.transpose(1, 2).log_softmax(-1)

        # A->a
        terms = self.vocab_out(pt_emb)
        # terms = terms.view(batch_size, self.pt_states, pt_num_nodes, -1)
        # terms = terms + pt_weights[:, None, :, None]
        terms = F.log_softmax(terms.view(batch_size, pt, -1), 2)

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
        if self.use_fast:
            pred.dist = Decomp1Fast(pred.posterior_params, pred.lengths, **pred.common())
        else:
            pred.dist = Decomp1(pred.posterior_params, pred.lengths, **pred.common())
        return pred

    def prepare_sampler(self, pred: TgtParserPrediction, src, src_ids, inplace=True) -> TgtParserPrediction:
        pred = super().prepare_sampler(pred, src, src_ids, inplace)
        pred.sampler = Decomp1Sampler(pred.params, **pred.common(), **self.sampler_common())
        return pred
