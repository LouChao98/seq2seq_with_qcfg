import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ...utils.fn import spans2tree
from ..components.common import MultiResidualLayer
from .neural_qcfg import NeuralQCFGTgtParser
from .struct.d1_pcfg import D1PCFG
from .struct.pcfg import PCFG


def get_nn(dim, cpd_rank):
    return nn.Sequential(
        nn.LeakyReLU(), nn.Linear(dim, cpd_rank)  # , nn.LayerNorm(cpd_rank)
    )


class NeuralQCFGD1TgtParser(NeuralQCFGTgtParser):
    def __init__(self, cpd_rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.nt_states == self.pt_states
        dim = self.dim
        num_layers = self.num_layers
        self.cpd_rank = cpd_rank
        self.pcfg = D1PCFG(self.nt_states, self.pt_states)
        # self.root_mlp_i = MultiResidualLayer(dim, dim, num_layers=num_layers)
        # self.root_mlp_j = MultiResidualLayer(dim, dim, num_layers=num_layers)
        # self.root_mlp_k = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_i = get_nn(dim, dim)
        self.root_mlp_j = get_nn(dim, dim)
        self.root_mlp_k = get_nn(dim, dim)
        self.rijk_weight = nn.Parameter(torch.empty(cpd_rank, dim, dim))
        nn.init.kaiming_uniform_(self.rijk_weight)
        self.ai_r_nn = get_nn(dim, cpd_rank)
        self.r_b_nn = get_nn(dim, cpd_rank)
        self.r_c_nn = get_nn(dim, cpd_rank)
        self.r_jk_nn = get_nn(dim, cpd_rank)

    def forward(self, x, lengths, node_features, spans, copy_position=None):
        params, *_ = self.get_params(node_features, spans, x, copy_position)
        out = self.pcfg(params, lengths, False)
        return out

    def parse(self, x, lengths, node_features, spans, copy_position=None):
        params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = self.get_params(
            node_features, spans, x, copy_position
        )

        # params2 = D1PCFG.get_pcfg_rules(params, self.nt_states)
        # out = PCFG()(params2, lengths, decode=True)
        out = self.pcfg(params, lengths, True)

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
        return out, all_spans_node, pt_spans, nt_spans

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
        rule_slr = torch.einsum(
            "rab,xia,xjkb->xrijk",
            self.rijk_weight,
            F.leaky_relu(i),
            F.leaky_relu(j[:, :, None] + k[:, None, :]),
        )
        num_nodes = nt_num_nodes  # + pt_num_nodes
        rule_slr = (
            rule_slr.view(batch_size, self.cpd_rank, num_nodes, -1)
            .log_softmax(-1)
            .view(batch_size, self.cpd_rank, num_nodes, num_nodes, num_nodes)
            .clone()
        )
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
                mask = self.get_rules_mask2(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )
            elif self.rule_constraint_type == 3:
                mask = self.get_rules_mask3(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )
            else:
                raise ValueError("Bad constraint_type")
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

        # debug_m = (rule_slr[0, 0].exp() > 1e-4).nonzero().tolist()
        # debug_spans = nt_spans[0]
        # children = defaultdict(set)
        # for i, j, k in debug_m:
        #     children[i].add(j)
        #     children[i].add(k)
        # for i, vset in children.items():
        #     print(f'Parent={debug_spans[i]}')
        #     print('  ' + ', '.join(str(debug_spans[j]) for j in vset))
        # print('===')
        # debug_m = terms.view(batch_size, self.pt_states, -1, terms.shape[-1])
        # debug_m = debug_m[0, 0, :, 0].exp().nonzero().squeeze(-1).tolist()
        # for i in debug_m:
        #     print(pt_spans[0][i], end=', ')

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
                        if w >= len(possible_copy) or w < 0:
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
                # if i == len(nt_span) - 1:
                #     nt_node_mask[b, i, i] = False

        node_mask = nt_node_mask.unsqueeze(3) * nt_node_mask.unsqueeze(2)
        return node_mask.view(batch_size, nt, nt, nt)

    def get_rules_mask2(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # A[a i]->B[a j] C[a k], a i must be the DIRECT parent of a j and a k, j!=k.
        #   if a i has no child, a j/k = a i.
        nt = nt_num_nodes
        node_mask = torch.zeros(batch_size, nt, nt, nt, device=device, dtype=torch.bool)

        def is_parent(parent, child):
            return child[0] >= parent[0] and child[1] <= parent[1]

        def is_strict_parent(parent, child):
            return is_parent(parent, child) and parent != child

        def span_len(span):
            return span[1] - span[0] + 1

        def covers(parent, child1, child2):
            return (span_len(parent) == (span_len(child1) + span_len(child2))) and (
                (parent[0] == child1[0] and parent[1] == child2[1])
                or (parent[0] == child2[0] and parent[1] == child1[1])
            )

        for b, nt_span in enumerate(nt_spans):
            min_nt_span = min([span_len(s) for s in nt_span])
            for i, parent in enumerate(nt_span):
                if span_len(parent) == min_nt_span:
                    node_mask[b, i, i, i].fill_(True)
                    for j, child in enumerate(nt_span):
                        if is_strict_parent(parent, child):
                            node_mask[b, i, i, j].fill_(True)
                            node_mask[b, i, j, i].fill_(True)
                for j, child1 in enumerate(nt_span):
                    for k, child2 in enumerate(nt_span):
                        if covers(parent, child1, child2):
                            node_mask[b, i, j, k].fill_(True)
                            node_mask[b, i, k, j].fill_(True)

        return node_mask.contiguous().view(batch_size, nt, nt, nt)

    def get_rules_mask3(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # mask1 + children

        # A[a i]->B[a j] C[a k], a i must be the parent of a j and a k.
        # return True for not masked
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

        for b, nt_spans_inst in enumerate(nt_spans):
            spans, parents, mapping = spans2tree(nt_spans_inst, return_mapping=True)
            for i, span1 in enumerate(spans):
                for j, span2 in enumerate(spans[i + 1 :], start=i + 1):
                    if not (is_parent(span1, span2)):
                        continue
                    depth = 1
                    k = parents[j]
                    while k != -1:
                        if k == i:
                            break
                        k = parents[k]
                        depth += 1
                    if depth > 2:
                        nt_node_mask[b, mapping[i], mapping[j]] = False

        # for mask, spans_inst in zip(nt_node_mask, nt_spans):
        #     self.show_constraint(mask, spans_inst)

        node_mask = nt_node_mask.unsqueeze(3) * nt_node_mask.unsqueeze(2)
        return node_mask.view(batch_size, nt, nt, nt)

    def show_constraint(self, mask, spans):
        # mask should be N * N. not allow batch.
        position = mask[: len(spans), : len(spans)].nonzero(as_tuple=True)
        allowed = defaultdict(list)
        for i, j in zip(*position):
            allowed[i].append(j)
        for i, vset in allowed.items():
            print("Parent: ", spans[i])
            print("Possible children: ", [spans[j] for j in vset])
            print("===")
