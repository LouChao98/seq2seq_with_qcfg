import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ...datamodules.components.vocab import VocabularyPair
from ..components.common import MultiResidualLayer
from .base import TgtParserBase
from .struct.pcfg import PCFG, TokenType
from .struct.d1_pcfg import D1PCFG
from .neural_qcfg import NeuralQCFGTgtParser


def get_nn(dim, cpd_rank):
    return nn.Sequential(
        nn.LeakyReLU(), nn.Linear(dim, cpd_rank), nn.LayerNorm(cpd_rank)
    )


class NeuralQCFGD1TgtParser(NeuralQCFGTgtParser):
    def __init__(self, cpd_rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.nt_states == self.pt_states
        dim = self.dim
        num_layers = self.num_layers
        self.cpd_rank = cpd_rank
        self.pcfg = D1PCFG(self.nt_states, self.pt_states)
        self.root_mlp_i = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_j = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_k = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.ai_r_nn = get_nn(dim, cpd_rank)
        self.r_b_nn = get_nn(dim, cpd_rank)
        self.r_c_nn = get_nn(dim, cpd_rank)
        self.r_jk_nn = get_nn(dim, cpd_rank)

    def parse(self, x, lengths, node_features, spans, copy_position=None):
        params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = self.get_params(
            node_features, spans, x, copy_position
        )

        head = params["head"].view(len(x), self.nt_states, -1, self.cpd_rank)
        rule = torch.einsum(
            "xair,xrb,xrc,xrijk->xaibjck",
            head.exp(),
            params["left"].exp(),
            params["right"].exp(),
            params["slr"].exp(),
        )
        shape = rule.shape
        rule = rule.reshape(
            shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]
        ).log()
        params2 = {"term": params["term"], "rule": rule, "root": params["root"]}
        out = PCFG()(params2, lengths, decode=True)
        # out = self.pcfg(params, lengths, True)

        # out: list of list, containing spans (i, j, label)
        src_nt_states = self.nt_states * nt_num_nodes
        src_pt_states = self.pt_states * pt_num_nodes
        all_spans_node = []
        for b, (all_span, pt_span, nt_span) in enumerate(zip(out, pt_spans, nt_spans)):
            all_span_node = []
            for l, r, label in all_span:
                if l == r:
                    if label < src_pt_states:
                        all_span_node.append(pt_span[label % pt_num_nodes])
                    else:
                        # these are for tgt_nt_states, which are removed for now.
                        all_span_node.append([-1, -1, label - src_pt_states])
                else:
                    if label < src_nt_states:
                        all_span_node.append(nt_span[label % nt_num_nodes])
                    else:
                        all_span_node.append([-1, -1, label - src_nt_states])
            all_spans_node.append(all_span_node)
        return out, all_spans_node

    def get_params(
        self,
        node_features,
        spans,
        x: Optional[torch.Tensor] = None,
        copy_position=None,  # (pt, nt), nt not implemented
    ):
        if copy_position is None:
            copy_position = (None, None)

        batch_size = len(spans)

        # seperate nt and pt features according to span width
        # TODO sanity check: the root node must be the last element of nt_spans.
        # NOTE TreeLSTM guarantees this.
        pt_node_features, nt_node_features = [], []
        pt_spans, nt_spans = [], []
        for spans_inst, node_features_inst in zip(spans, node_features):
            pt_node_feature = []
            nt_node_feature = []
            pt_span = []
            nt_span = []
            for i, s in enumerate(spans_inst):
                s_len = s[1] - s[0] + 1
                if s_len >= self.nt_span_range[0] and s_len <= self.nt_span_range[1]:
                    nt_node_feature.append(node_features_inst[i])
                    nt_span.append(s)
                if s_len >= self.pt_span_range[0] and s_len <= self.pt_span_range[1]:
                    pt_node_feature.append(node_features_inst[i])
                    pt_span.append(s)
            if len(nt_node_feature) == 0:
                nt_node_feature.append(node_features_inst[-1])
                nt_span.append(spans_inst[-1])
            pt_node_features.append(torch.stack(pt_node_feature))
            nt_node_features.append(torch.stack(nt_node_feature))
            pt_spans.append(pt_span)
            nt_spans.append(nt_span)
        nt_num_nodes_list = [len(inst) for inst in nt_node_features]
        pt_num_nodes_list = [len(inst) for inst in pt_node_features]
        nt_node_features = pad_sequence(
            nt_node_features, batch_first=True, padding_value=0.0
        )
        pt_node_features = pad_sequence(
            pt_node_features, batch_first=True, padding_value=0.0
        )
        pt_num_nodes = pt_node_features.size(1)
        nt_num_nodes = nt_node_features.size(1)
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
        ijk = torch.einsum("bin,bjn,bkn->bijkn", i, j, k)
        num_nodes = nt_num_nodes  # + pt_num_nodes
        rule_slr = (
            self.r_jk_nn(ijk)
            .movedim(4, 1)
            .view(batch_size, self.cpd_rank, num_nodes, -1)
            .log_softmax(-1)
            .view(batch_size, self.cpd_rank, num_nodes, num_nodes, num_nodes)
            .clone()
        )

        # fmt: off
        # TODO lhs_mask seems to be unnecessary.
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
                terms[:, :, -1] = (
                    0.1 * self.neg_huge * ~copy_position[0].transpose(1, 2)
                )
                terms = terms.view(batch_size, n, -1)
            if copy_position[1] is not None:
                # mask True= will set to value
                # TODO this waste memory to store TGT * SRC. Does this matter?
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
                    for l, r, _ in nt_spans_inst:
                        w = r - l - 1
                        t = None
                        if w >= len(possible_copy):
                            continue
                        for possible_s, possible_t in possible_copy[w]:
                            if possible_s == l:
                                t = possible_t
                                break
                        if t is not None:
                            copy_nt[w][batch_idx, t, -1, l] = 0
                copy_nt_ = []
                for item in copy_nt:
                    mask = np.zeros_like(item, dtype=np.bool8)
                    mask[:, :, -1] = True
                    item = torch.from_numpy(item.reshape(item.shape[:2] + (-1,)))
                    mask = torch.from_numpy(mask.reshape(item.shape))
                    copy_nt_.append((item.to(terms.device), mask.to(terms.device)))
                copy_nt = copy_nt_

        params = {
            "term": terms,
            "root": roots,
            "left": rule_left,
            "right": rule_right,
            "head": rule_head,
            "slr": rule_slr,
        }
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes

    def get_rules_mask1(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # A[a i]->B[a j] C[a k], a i must be the parent of a j and a k.
        # return 1 for not masked
        nt = nt_num_nodes
        nt_node_mask = torch.ones(
            batch_size, nt_num_nodes, nt_num_nodes, dtype=torch.bool
        )

        def is_parent(parent, child):
            return child[0] >= parent[0] and child[1] <= parent[1]

        for b, nt_span in enumerate(nt_spans):
            for i, parent_span in enumerate(nt_span):
                for j, child_span in enumerate(nt_span):
                    if not (is_parent(parent_span, child_span)):
                        nt_node_mask[b, i, j] = False

        node_mask = nt_node_mask.unsqueeze(3) * nt_node_mask.unsqueeze(2)
        return node_mask.view(batch_size, nt, nt, nt)

    def get_rules_mask2(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # A[a i]->B[a j] C[a k], a i must be the DIRECT parent of a j and a k, j!=k.
        #   if a i has no child, a j/k = a i.
        # TODO review this comment
        nt = nt_num_nodes * self.nt_states
        pt = pt_num_nodes * self.pt_states
        bsz = batch_size
        src_nt = self.nt_states
        src_pt = self.pt_states
        node_nt = nt_num_nodes
        node_pt = pt_num_nodes
        node_mask = torch.zeros(
            bsz,
            src_nt * node_nt,
            src_nt * node_nt + src_pt * node_pt,
            src_nt * node_nt + src_pt * node_pt,
        ).to(device)

        nt_idx = slice(0, src_nt * node_nt)
        pt_idx = slice(src_nt * node_nt, src_nt * node_nt + src_pt * node_pt)

        nt_ntnt = node_mask[:, nt_idx, nt_idx, nt_idx].view(
            bsz, src_nt, node_nt, src_nt, node_nt, src_nt, node_nt
        )
        nt_ntpt = node_mask[:, nt_idx, nt_idx, pt_idx].view(
            bsz, src_nt, node_nt, src_nt, node_nt, src_pt, node_pt
        )
        nt_ptnt = node_mask[:, nt_idx, pt_idx, nt_idx].view(
            bsz, src_nt, node_nt, src_pt, node_pt, src_nt, node_nt
        )
        nt_ptpt = node_mask[:, nt_idx, pt_idx, pt_idx].view(
            bsz, src_nt, node_nt, src_pt, node_pt, src_pt, node_pt
        )

        def is_parent(parent, child):
            if child[0] >= parent[0] and child[1] <= parent[1]:
                return True
            else:
                return False

        def is_strict_parent(parent, child):
            return is_parent(parent, child) and parent != child

        def span_len(span):
            return span[1] - span[0] + 1

        def covers(parent, child1, child2):
            return (span_len(parent) == (span_len(child1) + span_len(child2))) and (
                (parent[0] == child1[0] and parent[1] == child2[1])
                or (parent[0] == child2[0] and parent[1] == child1[1])
            )

        # fill_(1.) = not masked
        for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):
            min_nt_span = min([span_len(s) for s in nt_span])
            for i, parent in enumerate(nt_span):
                if span_len(parent) == min_nt_span:
                    nt_ntnt[b, :, i, :, i, :, i].fill_(1.0)
                    for j, child in enumerate(pt_span):
                        if is_strict_parent(parent, child):
                            nt_ntpt[b, :, i, :, i, :, j].fill_(1.0)
                            nt_ptnt[b, :, i, :, j, :, i].fill_(1.0)
                if span_len(parent) == 1:
                    for j, child in enumerate(pt_span):
                        if parent == child:
                            nt_ptnt[b, :, i, :, j, :, i].fill_(1.0)
                            nt_ntpt[b, :, i, :, i, :, j].fill_(1.0)
                            nt_ptpt[b, :, i, :, j, :, j].fill_(1.0)
                for j, child1 in enumerate(nt_span):
                    for k, child2 in enumerate(nt_span):
                        if covers(parent, child1, child2):
                            nt_ntnt[b, :, i, :, j, :, k].fill_(1.0)
                            nt_ntnt[b, :, i, :, k, :, j].fill_(1.0)
                    for k, child2 in enumerate(pt_span):
                        if covers(parent, child1, child2):
                            nt_ntpt[b, :, i, :, j, :, k].fill_(1.0)
                            nt_ptnt[b, :, i, :, k, :, j].fill_(1.0)
                for j, child1 in enumerate(pt_span):
                    for k, child2 in enumerate(pt_span):
                        if covers(parent, child1, child2):
                            nt_ptpt[b, :, i, :, j, :, k].fill_(1.0)
                            nt_ptpt[b, :, i, :, k, :, j].fill_(1.0)

        node_mask = (1.0 - node_mask) * self.neg_huge

        return node_mask.contiguous().view(batch_size, nt, nt + pt, nt + pt)

