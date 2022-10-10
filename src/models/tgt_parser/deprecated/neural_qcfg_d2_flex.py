from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..components.common import MultiResidualLayer
from .neural_qcfg import NeuralQCFGTgtParser
from .neural_qcfg_d1_flex import NeuralQCFGD1FlexTgtParser
from .struct.d2_pcfg_flex import D2PCFGFlex
from .struct.pcfg import PCFG


def get_nn(dim, cpd_rank):
    return nn.Sequential(nn.LeakyReLU(), nn.Linear(dim, cpd_rank))  # , nn.LayerNorm(cpd_rank)


@torch.jit.script
def normalize(t: torch.Tensor):
    t = t - t.amax((-2, -1), keepdim=True)
    t = t.exp()
    t = t / (t.sum((-2, -1), keepdim=True) + 1e-9)
    return t


class NeuralQCFGD2FlexTgtParser(NeuralQCFGD1FlexTgtParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pcfg = D2PCFGFlex(self.nt_states, self.pt_states)

    def get_params(
        self,
        node_features,
        spans,
        x: Optional[torch.Tensor] = None,
        copy_position=None,  # (pt, nt)
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
        nt_node_emb = self.src_nt_node_mlp(nt_node_features)
        nt_state_emb = self.src_nt_emb.expand(batch_size, self.nt_states, self.dim)
        nt_emb = nt_state_emb.unsqueeze(2) + nt_node_emb.unsqueeze(1)

        pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        pt_state_emb = self.src_pt_emb.expand(batch_size, self.pt_states, self.dim)
        pt_emb = pt_state_emb.unsqueeze(2) + pt_node_emb.unsqueeze(1)

        # S->A
        roots = self.root_mlp_child(nt_emb)
        roots = roots.view(batch_size, self.nt_states, nt_num_nodes)
        mask = torch.arange(nt_num_nodes, device=device).view(1, 1, -1).expand(batch_size, 1, -1)
        allowed = (torch.tensor(nt_num_nodes_list, device=device) - 1).view(-1, 1, 1)
        roots = torch.where(mask == allowed, roots, roots.new_tensor(self.neg_huge))
        roots = roots.view(batch_size, -1)
        roots = F.log_softmax(roots, 1)

        # A->BC
        state_emb = torch.cat([nt_state_emb, pt_state_emb], 1)
        node_emb = torch.cat([nt_node_emb, pt_node_emb], 1)
        rule_head = self.ai_r_nn(self.rule_mlp_parent(nt_emb.view(batch_size, -1, self.dim))).softmax(-1)

        combined_emb = nt_node_emb[:, :, None] + state_emb[:, None, :]
        rule_left = self.r_b_nn(self.rule_mlp_left(combined_emb)).movedim(3, 1).softmax(-1)
        rule_right = self.r_c_nn(self.rule_mlp_right(combined_emb)).movedim(3, 1).softmax(-1)

        i = self.root_mlp_i(nt_node_emb)
        j = self.root_mlp_j(node_emb)
        k = self.root_mlp_k(node_emb)
        # ijk = i[:, :, None, None] + j[:, None, :, None] + k[:, None, None, :]
        # num_nodes = nt_num_nodes  # + pt_num_nodes
        # rule_slr = (
        #     self.r_jk_nn(ijk)
        #     .movedim(4, 1)
        #     .view(batch_size, self.cpd_rank, num_nodes, -1)
        #     .log_softmax(-1)
        #     .view(batch_size, self.cpd_rank, num_nodes, num_nodes, num_nodes)
        #     .clone()
        # )
        rule_slr = torch.einsum(
            "rab,xia,xjkb->xrijk",
            self.rijk_weight,
            F.leaky_relu(i),
            F.leaky_relu(j[:, :, None] + k[:, None, :]),
        )  # softmax

        new_slr = torch.empty_like(rule_slr)
        nnn = nt_num_nodes
        new_slr[..., :nnn, :nnn] = normalize(rule_slr[..., :nnn, :nnn])
        new_slr[..., :nnn, nnn:] = normalize(rule_slr[..., :nnn, nnn:])
        new_slr[..., nnn:, :nnn] = normalize(rule_slr[..., nnn:, :nnn])
        new_slr[..., nnn:, nnn:] = normalize(rule_slr[..., nnn:, nnn:])
        rule_slr = new_slr

        # fmt: off
        nt_mask = torch.arange(nt_num_nodes, device=i.device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=i.device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=i.device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=i.device).unsqueeze(1)
        # fmt: on
        mask = torch.cat([nt_mask, pt_mask], dim=1)
        mask = torch.einsum("bx,by,bz->bxyz", nt_mask, mask, mask)
        mask = mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
        rule_slr[~mask] = 0  # = self.neg_huge

        if self.rule_constraint_type > 0:
            if self.rule_constraint_type == 1:
                mask = self.get_rules_mask1(batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device)
            elif self.rule_constraint_type == 2:
                mask = self.get_rules_mask2(batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device)
            elif self.rule_constraint_type == 3:
                mask = self.get_rules_mask3(batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device)
            else:
                raise ValueError("Bad constraint_type")
            mask = mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
            rule_slr[~mask] = 0  # = self.neg_huge

        # A->a
        terms = self.vocab_out(pt_emb).log_softmax(-1)
        terms = terms.view(batch_size, -1, terms.shape[-1])

        copy_nt = None
        if x is not None:
            n = x.size(1)
            pt = terms.size(1)
            terms = terms.unsqueeze(1).expand(batch_size, n, pt, terms.size(2))
            x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
            terms = torch.gather(terms, 3, x_expand).squeeze(3)
            if copy_position[0] is not None:
                terms = terms.view(batch_size, n, self.pt_states, -1)
                copy_m = copy_position[0][:, : terms.shape[-1]].transpose(1, 2)
                terms[:, :, -1, : copy_m.shape[2]] = self.neg_huge * ~copy_m
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
                for batch_idx, (nt_spans_inst, possible_copy) in enumerate(zip(nt_spans, copy_position[1])):
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
