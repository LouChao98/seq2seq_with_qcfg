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
        vocab=100,
        dim=256,
        num_layers=3,
        src_dim=256,
        nt_states=10,
        pt_states=1,
        rule_constraint_type=0,
        use_copy=False,
        nt_span_range=[2, 1000],
        pt_span_range=[1, 1],
        cpd_rank=96,
        num_samples=10,
        check_ppl=False,
        check_ppl_batch_size=None,
    ):
        # Different parameterization. Init nothing from bases.
        TgtParserBase.__init__(self)
        self.neg_huge = -1e9
        self.vocab = vocab
        self.dim = dim
        self.src_dim = src_dim
        self.num_layers = num_layers
        self.nt_states = nt_states
        self.pt_states = pt_states
        self.rule_constraint_type = rule_constraint_type
        self.use_copy = use_copy
        self.nt_span_range = nt_span_range
        self.pt_span_range = pt_span_range
        self.cpd_rank = cpd_rank
        self.num_samples = num_samples
        self.check_ppl = check_ppl  # If true, there is a length limitation.
        self.check_ppl_batch_size = check_ppl_batch_size
        self.pcfg = D1PCFG(self.nt_states, self.pt_states)

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
        copy_position=None,  # (pt, nt), nt not implemented
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
        nt_node_emb = self.nt_node_mlp(nt_node_features)
        nt_state_emb = self.nt_emb.expand(batch_size, self.nt_states, self.dim)
        nt_emb = nt_state_emb.unsqueeze(2) + nt_node_emb.unsqueeze(1)

        pt_node_emb = self.pt_node_mlp(pt_node_features)
        pt_state_emb = self.pt_emb.expand(batch_size, self.pt_states, self.dim)
        pt_emb = pt_state_emb.unsqueeze(2) + pt_node_emb.unsqueeze(1)

        # S->A
        roots = self.root_mlp_child(nt_emb)
        roots = roots.view(batch_size, self.nt_states, nt_num_nodes)
        mask = (
            torch.arange(nt_num_nodes, device=device)
            .view(1, 1, -1)
            .expand(batch_size, 1, -1)
        )
        allowed = (torch.tensor(nt_num_nodes_list, device=device) - 1).view(-1, 1, 1)
        roots = torch.where(mask == allowed, roots, roots.new_full((1,), self.neg_huge))
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
            .log_softmax(-1)
            .view(batch_size, -1, self.cpd_rank)
        )
        rule_left = torch.einsum(
            "rx,xy,bny->brn", self.r_emb, self.ri_b_weight, state_emb
        ).log_softmax(-1)
        rule_right = torch.einsum(
            "rx,xy,bny->brn", self.r_emb, self.ri_c_weight, state_emb
        ).log_softmax(-1)

        ri = self.act(self.r_emb[None, :, None] + i.unsqueeze(1))
        jk = self.act(j.unsqueeze(2) + k.unsqueeze(1))
        rule_slr = torch.einsum("brix,xy,bjky->brijk", ri, self.ri_jk_weight, jk)
        num_nodes = nt_num_nodes  # + pt_num_nodes
        rule_slr = (
            rule_slr.view(batch_size, self.cpd_rank, num_nodes, -1)
            .log_softmax(-1)
            .view(batch_size, self.cpd_rank, num_nodes, num_nodes, num_nodes)
            .clone()
        )

        # fmt: off
        nt_mask = torch.arange(nt_num_nodes, device=i.device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=i.device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=i.device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=i.device).unsqueeze(1)
        # fmt: on
        mask = nt_mask  # torch.cat([nt_mask, pt_mask], dim=1)
        mask = torch.einsum("bx,by,bz->bxyz", mask, mask, mask)
        mask = mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
        rule_slr[~mask] = self.neg_huge

        if self.rule_constraint_type > 0:
            if self.rule_constraint_type == 1:
                mask = self.get_rules_mask1(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )
            elif self.rule_constraint_type == 2:
                raise NotImplementedError
                mask = self.get_rules_mask2(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )
            mask = mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
            rule_slr[~mask] = self.neg_huge

        # A->a
        terms = self.vocab_out(pt_emb).log_softmax(-1)
        # temperory fix
        is_multi = np.ones((batch_size, pt_num_nodes), dtype=np.bool8)
        for b, pt_spans_inst in enumerate(pt_spans):
            for span in pt_spans_inst:
                if span[0] == span[1]:
                    is_multi[b, span[0]] = False
        terms = terms.clone()
        mask = torch.from_numpy(is_multi)[:, None, :, None]
        mask = mask.expand(-1, terms.shape[1], -1, terms.shape[3])
        terms[mask] = self.neg_huge
        terms = terms.view(batch_size, -1, terms.shape[-1])
        mask = torch.from_numpy(is_multi)[:, None, :, None, None]
        mask = mask.expand(-1, rule_slr.shape[1], -1, *rule_slr.shape[-2:])
        rule_slr[~mask] = self.neg_huge

        copy_nt = None
        if x is not None:
            n = x.size(1)
            pt = terms.size(1)
            terms = terms.unsqueeze(1).expand(batch_size, n, pt, terms.size(2))
            x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
            terms = torch.gather(terms, 3, x_expand).squeeze(3)
            if copy_position[0] is not None:
                terms = terms.view(batch_size, n, self.pt_states, -1)
                terms[:, :, -1, : copy_position[0].shape[1]] = (
                    0.1 * self.neg_huge * ~copy_position[0].transpose(1, 2)
                )
                terms = terms.view(batch_size, n, -1)
            if copy_position[1] is not None:
                # mask=True will set to value
                copy_nt = [
                    np.full(
                        (batch_size, n - w, self.nt_states, nt_num_nodes),
                        self.neg_huge,
                        dtype=np.float32,
                    )
                    for w in range(1, n)
                ]
                for batch_idx, (nt_spans_inst, possible_copy) in enumerate(
                    zip(nt_spans, copy_position[1])
                ):
                    for i, (l, r, _) in enumerate(nt_spans_inst):
                        w = r - l - 1
                        t = None
                        if w >= len(possible_copy) or w <= 0:
                            continue
                        for possible_s, possible_t in possible_copy[w]:
                            if possible_s == l:
                                t = possible_t
                                break
                        if t is not None:
                            copy_nt[w][batch_idx, t, -1, i] = 0
                copy_nt_ = []
                for item in copy_nt:
                    mask = np.zeros_like(item, dtype=np.bool8)
                    mask[:, :, -1] = True
                    item = torch.from_numpy(item)
                    mask = torch.from_numpy(mask)
                    copy_nt_.append((item.to(terms.device), mask.to(terms.device)))
                copy_nt = copy_nt_

        params = {
            "term": terms,
            "root": roots,
            "left": rule_left,
            "right": rule_right,
            "head": rule_head,
            "slr": rule_slr,
            "copy_nt": copy_nt,
        }
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes
