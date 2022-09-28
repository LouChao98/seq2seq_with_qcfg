import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..components.common import MultiResidualLayer
from .base import TgtParserBase
from .struct.pcfg import PCFG

log = logging.getLogger(__file__)


class NeuralQCFGTgtParser(TgtParserBase):
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

        self.pcfg = PCFG()
        self.vocab = vocab
        self.dim = dim
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
        filtered_spans=None,  # list of spans
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
        nt = self.nt_states * nt_num_nodes

        src_pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        src_pt_emb = self.src_pt_emb.expand(batch_size, self.pt_states, self.dim)
        pt_emb = src_pt_emb.unsqueeze(2) + src_pt_node_emb.unsqueeze(1)
        pt_emb = pt_emb.view(batch_size, self.pt_states * pt_num_nodes, -1)
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

        if self.rule_hard_constraint is not None:
            mask = self.rule_hard_constraint.get_mask(
                batch_size, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device
            )
            rules[~mask] = self.neg_huge

        rules = rules.flatten(2).log_softmax(-1).clone()
        rules = rules.view(batch_size, nt, nt + pt, nt + pt)
        rules[~lhs_mask] = self.neg_huge

        # A->a
        terms = F.log_softmax(self.vocab_out(pt_emb), 2)

        if filtered_spans is not None:
            roots, rules, nt_num_nodes, nt_num_nodes_list = self.filter_rules(
                nt_spans, nt, pt, filtered_spans, roots, rules
            )

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
            "rule": rules,
            "constraint": nt_constraint,
        }
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes

    def filter_rules(self, nt_spans, nt, pt, filtered_spans, roots, rules):
        batch_size = len(nt_spans)
        masks = []
        inds = []
        new_nt_spans = []
        for spans, to_be_preserved in zip(nt_spans, filtered_spans):
            new_span = []
            mask = []
            ind = []
            for i, span in enumerate(spans):
                if span in to_be_preserved:
                    new_span.append(span)
                    mask.append(False)
                    ind.append(i)
                else:
                    mask.append(True)
            new_nt_spans.append(new_span)
            mask = torch.tensor(mask)
            mask = mask.unsqueeze(0).expand(self.nt_states, -1).contiguous().flatten()
            masks.append(mask)
            inds.append(torch.tensor(ind))
        masks = pad_sequence(masks, batch_first=True).to(roots.device)
        roots = roots.clone()
        roots[masks] = self.neg_huge
        rules[masks] = self.neg_huge
        inds_ = pad_sequence(inds, batch_first=True).to(roots.device)
        inds_ = inds_.unsqueeze(1)
        _state_offset = (
            torch.arange(self.nt_states, device=roots.device)[None, :, None]
            * nt_num_nodes
        )
        inds = (_state_offset + inds_).view(batch_size, -1)
        roots = roots.gather(1, inds)
        rules = rules.gather(1, inds[..., None, None].expand(-1, -1, nt + pt, nt + pt))
        rules11 = rules[:, :, :nt, :nt]
        rules12 = rules[:, :, :nt, nt:]
        rules21 = rules[:, :, nt:, :nt]
        rules22 = rules[:, :, nt:, nt:]
        n = inds.shape[1]
        rules11 = rules11.gather(
            2, inds[:, None, :, None].expand(-1, n, -1, nt)
        ).gather(3, inds[:, None, None, :].expand(-1, n, n, -1))
        rules12 = rules12.gather(2, inds[:, None, :, None].expand(-1, n, -1, pt))
        rules21 = rules21.gather(3, inds[:, None, None, :].expand(-1, n, pt, -1))
        rules = torch.empty((batch_size, n, n + pt, n + pt), device=rules11.device)
        rules[:, :, :n, :n] = rules11
        rules[:, :, :n, n:] = rules12
        rules[:, :, n:, :n] = rules21
        rules[:, :, n:, n:] = rules22
        nt_spans = new_nt_spans
        nt_num_nodes = n // self.nt_states
        nt_num_nodes_list = [len(item) for item in new_nt_spans]

        return roots, rules, nt_num_nodes, nt_num_nodes_list
