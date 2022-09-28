import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .neural_qcfg_d1 import NeuralQCFGD1TgtParser
from .struct.d1_pcfg_tse import D1PCFGTSE


class NeuralQCFGD1LogParamTgtParser(NeuralQCFGD1TgtParser):

    # This produce log-space params.
    # The impl of struct is much slower than counterparts.
    # Just for debug.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pcfg = D1PCFGTSE(self.nt_states, self.pt_states)

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

        # PT can only align to leavs
        is_multi = np.ones((batch_size, pt_num_nodes), dtype=np.bool8)
        for b, pt_spans_inst in enumerate(pt_spans):
            for span in pt_spans_inst:
                if span[0] == span[1]:
                    is_multi[b, span[0]] = False
        is_multi = torch.from_numpy(is_multi).to(device)

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
        node_emb = nt_node_emb  # torch.cat([nt_node_emb, pt_node_emb], 1)
        rule_head = self.ai_r_nn(
            self.rule_mlp_parent(nt_emb.view(batch_size, -1, self.dim))
        ).log_softmax(-1)
        rule_left = (
            self.r_b_nn(self.rule_mlp_left(state_emb)).transpose(1, 2).log_softmax(-1)
        )
        rule_right = (
            self.r_c_nn(self.rule_mlp_right(state_emb)).transpose(1, 2).log_softmax(-1)
        )

        i = self.root_mlp_i(node_emb)
        j = self.root_mlp_j(node_emb)
        k = self.root_mlp_k(node_emb)
        rule_slr = torch.einsum(
            "rab,xia,xjkb->xrijk",
            self.rijk_weight,
            F.leaky_relu(i),
            F.leaky_relu(j[:, :, None] + k[:, None, :]),
        ).clone()

        # fmt: off
        lhs_mask = torch.arange(nt_num_nodes, device=i.device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=i.device).unsqueeze(1)
        # fmt: on
        valid_mask = torch.einsum("bx,by,bz->bxyz", lhs_mask, lhs_mask, lhs_mask)
        valid_mask = valid_mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
        rule_slr[~valid_mask] = self.neg_huge

        if self.rule_hard_constraint is not None:
            mask = self.rule_hard_constraint.get_mask(
                batch_size, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device
            )
            mask = mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
            rule_slr[~mask] = self.neg_huge

        rule_slr = rule_slr.flatten(3).log_softmax(-1).view(rule_slr.shape).clone()

        mask = (is_multi & lhs_mask).unsqueeze(1)
        mask = mask.expand(-1, rule_slr.shape[1], -1)
        rule_slr[~mask] = self.neg_huge

        # A->a
        terms = self.vocab_out(pt_emb).log_softmax(-1).clone()
        mask = is_multi[:, None, :, None]
        mask = mask.expand(-1, terms.shape[1], -1, terms.shape[3])
        terms[mask] = self.neg_huge
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
