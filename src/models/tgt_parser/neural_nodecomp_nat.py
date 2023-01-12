import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roformer.modeling_roformer import RoFormerSinusoidalPositionalEmbedding

from ..components.common import MultiResidualLayer
from ..struct.no_decomp import NoDecomp, NoDecompNATSampler
from .base import TgtParserBase, TgtParserPrediction

log = logging.getLogger(__file__)


class NeuralNoDecompNATTgtParser(TgtParserBase):
    def __init__(self, vocab=100, dim=256, num_layers=3, src_dim=256, learnable_positional_embedding=True, **kwargs):
        super().__init__(**kwargs)
        assert not self.use_copy

        self.vocab = vocab
        self.dim = dim
        self.src_dim = src_dim
        self.num_layers = num_layers

        self.vocab_out = MultiResidualLayer(dim, dim, out_dim=vocab, num_layers=num_layers)
        if learnable_positional_embedding:
            self.position_emb = nn.Embedding(self.generation_max_length, dim)
        else:
            self.position_emb = RoFormerSinusoidalPositionalEmbedding(self.generation_max_length, dim)

        self.src_nt_emb = nn.Parameter(torch.randn(self.nt_states, dim))
        self.src_nt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.src_pt_emb = nn.Parameter(torch.randn(self.pt_states, dim))
        self.src_pt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)

        self.rule_mlp_parent = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_mlp_left = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_mlp_right = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_child = nn.Linear(dim, 1, bias=False)

    def forward(
        self,
        node_features,
        spans,
        filtered_spans=None,  # list of spans
    ):
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
        roots = F.log_softmax(roots, dim=1)

        # A->BC

        all_emb = torch.cat([nt_emb, pt_emb], 1)
        rule_emb_parent = self.rule_mlp_parent(nt_emb)  # b x nt_all x dm
        rule_emb_left = self.rule_mlp_left(all_emb)
        rule_emb_right = self.rule_mlp_right(all_emb)

        rule_emb_child = rule_emb_left[:, :, None, :] + rule_emb_right[:, None, :, :]
        rule_emb_child = rule_emb_child.view(batch_size, (nt + pt) ** 2, self.dim)

        rules = torch.matmul(rule_emb_parent, rule_emb_child.transpose(1, 2))
        rules = rules.view(batch_size, nt, nt + pt, nt + pt)

        # fmt: off
        nt_mask = torch.arange(nt_num_nodes, device=rules.device).unsqueeze(0) \
                  < torch.tensor(nt_num_nodes_list, device=rules.device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=rules.device).unsqueeze(0) \
                  < torch.tensor(pt_num_nodes_list, device=rules.device).unsqueeze(1)
        lhs_mask = nt_mask.unsqueeze(1).expand(-1, self.nt_states, -1).reshape(batch_size, -1)
        _pt_rhs_mask = pt_mask.unsqueeze(1).expand(-1, self.pt_states, -1).reshape(batch_size, -1)
        # fmt: on
        rhs_mask = torch.cat([lhs_mask, _pt_rhs_mask], dim=1)
        valid_mask = torch.einsum("bx,by,bz->bxyz", lhs_mask, rhs_mask, rhs_mask)
        rules[~valid_mask] = self.neg_huge

        common = batch_size, self.pt_states, self.nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device
        if self.rule_hard_constraint is not None:
            mask = self.rule_hard_constraint.get_mask(*common)
            rules[~mask] = self.neg_huge

        rules = rules.flatten(2).log_softmax(dim=-1).clone()
        rules = rules.view(batch_size, nt, nt + pt, nt + pt)

        if self.rule_reweight_constraint is not None and (not self.rule_reweight_test_only or not self.training):
            weight = self.rule_reweight_constraint.get_weight(*common)
            rules = rules + weight

        rules[~lhs_mask] = self.neg_huge

        # A->a
        # 1 x nposition x 1 x hidden + bsz x 1 x state x hidden
        term_emb = self.position_emb.weight[None, :, None] + pt_emb.unsqueeze(1)
        terms = F.log_softmax(self.score_normalize(self.vocab_out(term_emb), dims=2), dim=3)

        if filtered_spans is not None:
            roots, rules, nt_num_nodes, nt_num_nodes_list = self.filter_rules(
                nt_spans, nt, pt, filtered_spans, roots, rules
            )

        params = {
            "term": terms,
            "root": roots,
            "rule": rules,
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
        return pred

    def build_term_rules_given_tgt(self, term, batch_size, n, pt, tgt):
        term = term[:, :n]
        tgt_expand = tgt.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
        term = torch.gather(term, 3, tgt_expand).squeeze(3)
        return term

    def observe_x(self, pred: TgtParserPrediction, x, lengths, inplace=True, **kwargs) -> TgtParserPrediction:
        pred = super().observe_x(pred, x, lengths, inplace, **kwargs)
        pred.dist = NoDecomp(pred.posterior_params, pred.lengths, **pred.common())
        return pred

    def prepare_sampler(self, pred: TgtParserPrediction, src, src_ids, inplace=True) -> TgtParserPrediction:
        pred = super().prepare_sampler(pred, src, src_ids, inplace)
        pred.sampler = NoDecompNATSampler(pred.params, **pred.common(), **self.sampler_common())
        return pred
