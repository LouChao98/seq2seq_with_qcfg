from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

from src.utils.fn import spans2tree


class FSimpleHierarchy:
    def __init__(self, pt_states, nt_states):
        self.pt_states = pt_states
        self.nt_states = nt_states

    def get_mask(
        self, batch_size, max_pt_spans, max_nt_spans, pt_spans, nt_spans, device
    ):
        pt_spans[0] += [(0, 0, 0)] * (max_pt_spans - len(pt_spans[0]))
        nt_spans[0] += [(0, 0, 0)] * (max_nt_spans - len(nt_spans[0]))
        pt_spans = pad_sequence(
            [torch.tensor([(l, r) for l, r, t in item]) for item in pt_spans],
            batch_first=True,
        ).to(device)
        nt_spans = pad_sequence(
            [torch.tensor([(l, r) for l, r, t in item]) for item in nt_spans],
            batch_first=True,
        ).to(device)

        nt_node_mask = nt_spans[..., 0].unsqueeze(2) <= nt_spans[..., 0].unsqueeze(1)
        nt_node_mask &= nt_spans[..., 1].unsqueeze(2) >= nt_spans[..., 1].unsqueeze(1)
        pt_node_mask = nt_spans[..., 0].unsqueeze(2) <= pt_spans[..., 0].unsqueeze(1)
        pt_node_mask &= nt_spans[..., 1].unsqueeze(2) >= pt_spans[..., 1].unsqueeze(1)

        nt = max_nt_spans
        pt = max_pt_spans
        node_mask = torch.cat([nt_node_mask, pt_node_mask], 2).to(device)
        node_mask = node_mask.unsqueeze(3) * node_mask.unsqueeze(2)
        return node_mask.view(batch_size, nt, nt + pt, nt + pt)

    def get_feature(self, *args, **kwargs):
        return 1.0 - self.get_mask(*args, **kwargs).float()


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
