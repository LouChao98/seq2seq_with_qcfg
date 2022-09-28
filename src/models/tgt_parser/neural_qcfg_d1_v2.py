from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..components.common import MultiResidualLayer
from .base import TgtParserBase
from .neural_qcfg_d1 import NeuralQCFGD1TgtParser
from .struct.d1_pcfg import D1PCFG


class NeuralQCFGD1V2TgtParser(NeuralQCFGD1TgtParser):
    def __init__(
        self,
        pcfg=None,
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
        cpd_rank=128,
        num_layers=3,
        src_dim=256,
    ):
        # Different parameterization. Init nothing from bases.
        TgtParserBase.__init__(
            self,
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

        assert self.nt_states == self.pt_states

        self.pcfg = D1PCFG(self.nt_states, self.pt_states)
        self.vocab = vocab
        self.dim = dim
        self.cpd_rank = cpd_rank
        self.num_layers = num_layers
        self.src_dim = src_dim

        self.nt_emb = nn.Parameter(torch.empty(nt_states, dim))
        self.nt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.pt_emb = nn.Parameter(torch.empty(pt_states, dim))
        self.pt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.r_emb = nn.Parameter(torch.empty(cpd_rank, dim))

        self.root_mlp_child = nn.Linear(dim, 1, bias=False)
        self.vocab_out = MultiResidualLayer(
            dim, dim, out_dim=vocab, num_layers=num_layers
        )

        self.root_mlp_i = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_j = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_k = MultiResidualLayer(dim, dim, num_layers=num_layers)

        self.act = nn.LeakyReLU()
        self.ai_r_weight = nn.Parameter(torch.empty(dim, dim))
        self.ri_b_weight = nn.Parameter(torch.empty(dim, dim))
        self.ri_c_weight = nn.Parameter(torch.empty(dim, dim))
        self.ri_jk_weight = nn.Parameter(torch.empty(dim, dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.nt_emb)
        nn.init.xavier_uniform_(self.pt_emb)
        nn.init.xavier_uniform_(self.r_emb)
        nn.init.xavier_uniform_(self.ai_r_weight)
        nn.init.xavier_uniform_(self.ri_b_weight)
        nn.init.xavier_uniform_(self.ri_c_weight)
        nn.init.xavier_uniform_(self.ri_jk_weight)

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
        nt_node_emb = self.nt_node_mlp(nt_node_features)
        nt_state_emb = self.nt_emb.expand(batch_size, self.nt_states, self.dim)
        nt_emb = nt_state_emb.unsqueeze(2) + nt_node_emb.unsqueeze(1)
        nt = self.nt_states * nt_num_nodes

        pt_node_emb = self.pt_node_mlp(pt_node_features)
        pt_state_emb = self.pt_emb.expand(batch_size, self.pt_states, self.dim)
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
        node_emb = nt_node_emb  # torch.cat([nt_node_emb, pt_node_emb], 1)
        i = self.root_mlp_i(node_emb)
        j = self.root_mlp_j(node_emb)
        k = self.root_mlp_k(node_emb)
        state_emb = torch.cat([nt_state_emb, pt_state_emb], 1)

        rule_head = (
            torch.einsum(
                "baix,xy,ry->bair", self.act(nt_emb), self.ai_r_weight, self.r_emb
            )
            .softmax(-1)
            .view(batch_size, -1, self.cpd_rank)
        )
        rule_left = torch.einsum(
            "rx,xy,bny->brn", self.r_emb, self.ri_b_weight, state_emb
        ).softmax(-1)
        rule_right = torch.einsum(
            "rx,xy,bny->brn", self.r_emb, self.ri_c_weight, state_emb
        ).softmax(-1)

        ri = self.act(self.r_emb[None, :, None] + i.unsqueeze(1))
        jk = self.act(j.unsqueeze(2) + k.unsqueeze(1))
        rule_slr = torch.einsum(
            "brix,xy,bjky->brijk", ri, self.ri_jk_weight, jk
        ).clone()
        num_nodes = nt_num_nodes  # + pt_num_nodes

        # fmt: off
        mask = torch.arange(nt_num_nodes, device=i.device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=i.device).unsqueeze(1)
        # fmt: on
        mask = torch.einsum("bx,by,bz->bxyz", mask, mask, mask)
        mask = mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
        rule_slr[~mask] = self.neg_huge

        mask = is_multi[:, None, :, None, None]
        mask = mask.expand(-1, rule_slr.shape[1], -1, *rule_slr.shape[-2:])
        rule_slr[~mask] = self.neg_huge

        if self.rule_hard_constraint is not None:
            mask = self.rule_hard_constraint.get_mask(
                batch_size, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device
            )
            mask = mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
            rule_slr[~mask] = 0

        rule_slr = (
            rule_slr.flatten(3)
            .softmax(-1)
            .view(batch_size, self.cpd_rank, num_nodes, num_nodes, num_nodes)
        )

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
