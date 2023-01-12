from collections import defaultdict
from copy import deepcopy

import torch
from torch.nn.utils.rnn import pad_sequence

from src.utils.fn import spans2tree

from .base import RuleConstraintBase


class USimpleHierarchy(RuleConstraintBase):
    def get_mask(self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device):
        assert nt_num_nodes == pt_num_nodes
        assert pt_spans == nt_spans
        nt_spans = deepcopy(nt_spans)
        nt_spans[0] += [(0, 0, -999)] * (nt_num_nodes - len(nt_spans[0]))
        nt_spans = pad_sequence(
            [torch.tensor([(l, r) for l, r, t in item]) for item in nt_spans],
            batch_first=True,
        ).to(device)
        nt_node_mask = nt_spans[..., 0].unsqueeze(2) <= nt_spans[..., 0].unsqueeze(1)
        nt_node_mask &= nt_spans[..., 1].unsqueeze(2) >= nt_spans[..., 1].unsqueeze(1)
        node_mask = nt_node_mask.unsqueeze(3) * nt_node_mask.unsqueeze(2)
        return node_mask

    # def get_mask(self, batch_size, max_pt_spans, max_nt_spans, pt_spans, nt_spans, device):
    #     # A[a i]->B[a j] C[a k], a i must be the parent of a j and a k.
    #     # return 1 for not masked
    #     nt = max_nt_spans
    #     nt_node_mask = torch.ones(batch_size, max_nt_spans, max_nt_spans, dtype=torch.bool)

    #     def is_parent(parent, child):
    #         return child[0] >= parent[0] and child[1] <= parent[1]

    #     for b, nt_span in enumerate(nt_spans):
    #         for i, parent_span in enumerate(nt_span):
    #             for j, child_span in enumerate(nt_span):
    #                 if not (is_parent(parent_span, child_span)):
    #                     nt_node_mask[b, i, j] = False

    #     node_mask = nt_node_mask.unsqueeze(3) * nt_node_mask.unsqueeze(2)
    #     return node_mask.view(batch_size, nt, nt, nt)


def is_parent(parent, child):
    return child[0] >= parent[0] and child[1] <= parent[1]


class UDepthBoundedHierarchy(RuleConstraintBase):
    def __init__(self, depth):
        self.depth = depth

    def get_mask(self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device):
        # mask1 + children

        # A[a i]->B[a j] C[a k], a i must be the parent of a j and a k.
        # return True for not masked
        nt = nt_num_nodes
        nt_node_mask = torch.ones(batch_size, nt, nt, dtype=torch.bool)

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
        return node_mask.view(batch_size, nt, nt, nt).to(device)


def show_constraint(mask, spans):
    # mask should be N * N. not allow batch.
    position = mask[: len(spans), : len(spans)].nonzero(as_tuple=True)
    allowed = defaultdict(list)
    for i, j in zip(*position):
        allowed[i].append(j)
    for i, vset in allowed.items():
        print("Parent: ", spans[i])
        print("Possible children: ", [spans[j] for j in vset])
        print("===")


# def get_rules_mask2(
#     self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
# ):
#     # A[a i]->B[a j] C[a k], a i must be the DIRECT parent of a j and a k, j!=k.
#     #   if a i has no child, a j/k = a i.
#     nt = nt_num_nodes
#     node_mask = torch.zeros(batch_size, nt, nt, nt, device=device, dtype=torch.bool)

#     def is_parent(parent, child):
#         return child[0] >= parent[0] and child[1] <= parent[1]

#     def is_strict_parent(parent, child):
#         return is_parent(parent, child) and parent != child

#     def span_len(span):
#         return span[1] - span[0] + 1

#     def covers(parent, child1, child2):
#         return (span_len(parent) == (span_len(child1) + span_len(child2))) and (
#             (parent[0] == child1[0] and parent[1] == child2[1])
#             or (parent[0] == child2[0] and parent[1] == child1[1])
#         )

#     for b, nt_span in enumerate(nt_spans):
#         min_nt_span = min([span_len(s) for s in nt_span])
#         for i, parent in enumerate(nt_span):
#             if span_len(parent) == min_nt_span:
#                 node_mask[b, i, i, i].fill_(True)
#                 for j, child in enumerate(nt_span):
#                     if is_strict_parent(parent, child):
#                         node_mask[b, i, i, j].fill_(True)
#                         node_mask[b, i, j, i].fill_(True)
#             for j, child1 in enumerate(nt_span):
#                 for k, child2 in enumerate(nt_span):
#                     if covers(parent, child1, child2):
#                         node_mask[b, i, j, k].fill_(True)
#                         node_mask[b, i, k, j].fill_(True)

#     return node_mask.contiguous().view(batch_size, nt, nt, nt)


if __name__ == "__main__":
    c = USimpleHierarchy(3, 3)
    batch_size = 2
    max_pt_spans = 6
    max_nt_spans = 6
    pt_spans = [
        [(0, 0, -1), (1, 1, -1), (2, 2, -1), (0, 3, -1), (1, 2, -1), (1, 3, -1)],
        [(1, 1, -1), (2, 2, -1), (0, 4, -1), (3, 4, -1)],
    ]
    nt_spans = pt_spans
    m1 = c.get_mask(batch_size, max_pt_spans, max_nt_spans, pt_spans, nt_spans, "cpu")
    m2 = c.get_mask_impl2(batch_size, max_pt_spans, max_nt_spans, pt_spans, nt_spans, "cpu")
