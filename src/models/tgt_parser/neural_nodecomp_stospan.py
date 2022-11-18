import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..components.common import MultiResidualLayer
from ..struct.no_decomp import NoDecomp, NoDecompSampler
from .base import TgtParserBase, TgtParserPrediction
from .neural_nodecomp import NeuralNoDecompTgtParser

log = logging.getLogger(__file__)


class NeuralNoDecompStoSpanTgtParser(NeuralNoDecompTgtParser):
    def forward(
        self,
        sequence_features,  # bsz x max_len x hidden
        node_features,  # bsz x max_len+1 x max_len+1 x hidden
        span_indicator,  # bsz x max_len+1 x max_len+1 x max_len-1
        src_lens,
    ):
        batch_size = len(src_lens)
        device = node_features[0].device

        pt_spans = [[(i, i + 1, -1) for i in range(l)] for l in src_lens]
        nt_spans = [[] for _ in range(batch_size)]
        for bidx, l, r, i in span_indicator.nonzero().tolist():
            nt_spans[bidx].append((l, r, i))
        for pt_spans_item, nt_spans_item in zip(pt_spans, nt_spans):
            nt_spans_item.sort(key=lambda x: x[2])
            nt_spans_item[:] = [(item[0], item[1], -1) for item in nt_spans_item]
            self.sanity_check_spans(nt_spans_item, pt_spans_item)
        nt_num_nodes_list = [len(item) for item in nt_spans]
        nt_num_nodes = sequence_features.shape[1] - 1
        pt_num_nodes_list = [len(item) for item in pt_spans]
        pt_num_nodes = sequence_features.shape[1]
        pt_node_features = sequence_features
        nt_node_features = torch.einsum("bxyi,bxyh->bih", span_indicator, node_features)

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

        rules = self.normalizer(rules.flatten(2), dim=-1).clone()
        rules = rules.view(batch_size, nt, nt + pt, nt + pt)

        if self.rule_reweight_constraint is not None:
            weight = self.rule_reweight_constraint.get_weight(*common)
            rules = rules + weight

        rules[~lhs_mask] = self.neg_huge

        # A->a
        terms = F.log_softmax(self.vocab_out(pt_emb), 2)

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
            pt=pt,
            pt_states=self.pt_states,
            pt_nodes=pt_spans,
            pt_num_nodes=pt_num_nodes,
            params=params,
            device=device,
        )
        return pred
