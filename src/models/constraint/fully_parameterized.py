import torch
from torch.nn.utils.rnn import pad_sequence


class FPSimpleHierarchy:
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

        nt = max_nt_spans * self.nt_states
        pt = max_pt_spans * self.pt_states
        nt_node_mask = (
            nt_node_mask[:, None, :, None, :]
            .repeat(1, self.nt_states, 1, self.nt_states, 1)
            .view(batch_size, nt, nt)
        )
        pt_node_mask = (
            pt_node_mask[:, None, :, None, :]
            .repeat(1, self.nt_states, 1, self.pt_states, 1)
            .view(batch_size, nt, pt)
        )
        node_mask = torch.cat([nt_node_mask, pt_node_mask], 2).to(device)
        node_mask = node_mask.unsqueeze(3) * node_mask.unsqueeze(2)
        return node_mask.view(batch_size, nt, nt + pt, nt + pt)

    def get_feature(self, *args, **kwargs):
        return 1.0 - self.get_mask(*args, **kwargs).float()

    def get_mask_impl2(
        self, batch_size, max_pt_spans, max_nt_spans, pt_spans, nt_spans, device
    ):
        # A[a i]->B[a j] C[a k], a i must be the parent of a j and a k.
        # return True for not masked
        nt = max_nt_spans * self.nt_states
        pt = max_pt_spans * self.pt_states
        nt_node_mask = torch.ones(
            batch_size, max_nt_spans, max_nt_spans, dtype=torch.bool
        )
        pt_node_mask = torch.ones(
            batch_size, max_nt_spans, max_pt_spans, dtype=torch.bool
        )

        def is_parent(parent, child):
            return child[0] >= parent[0] and child[1] <= parent[1]

        for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):
            for i, parent_span in enumerate(nt_span):
                for j, child_span in enumerate(nt_span):
                    if not (is_parent(parent_span, child_span)):
                        nt_node_mask[b, i, j] = False
                for j, child_span in enumerate(pt_span):
                    if not (is_parent(parent_span, child_span)):
                        pt_node_mask[b, i, j] = False

        nt_node_mask = (
            nt_node_mask[:, None, :, None, :]
            .expand(
                batch_size,
                self.nt_states,
                max_nt_spans,
                self.nt_states,
                max_nt_spans,
            )
            .contiguous()
        )
        pt_node_mask = (
            pt_node_mask[:, None, :, None, :]
            .expand(
                batch_size,
                self.nt_states,
                max_nt_spans,
                self.pt_states,
                max_pt_spans,
            )
            .contiguous()
        )

        nt_node_mask = nt_node_mask.view(batch_size, nt, nt)
        pt_node_mask = pt_node_mask.view(batch_size, nt, pt)
        node_mask = torch.cat([nt_node_mask, pt_node_mask], 2).to(device)
        node_mask = node_mask.unsqueeze(3) * node_mask.unsqueeze(2)
        return node_mask.view(batch_size, nt, nt + pt, nt + pt)


class FPFewerSame:
    def __init__(self, pt_states, nt_states):
        self.pt_states = pt_states
        self.nt_states = nt_states

    def get_mask(
        self, batch_size, max_pt_spans, max_nt_spans, pt_spans, nt_spans, device
    ):
        nt_node_mask = ~torch.eye(max_nt_spans, device=device, dtype=torch.bool)
        pt_node_mask = torch.ones(
            max_nt_spans, max_pt_spans, device=device, dtype=torch.bool
        )
        nt = max_nt_spans * self.nt_states
        pt = max_pt_spans * self.pt_states
        nt_node_mask = (
            nt_node_mask[None, None, :, None, :]
            .repeat(1, self.nt_states, 1, self.nt_states, 1)
            .view(1, nt, nt)
        )
        pt_node_mask = (
            pt_node_mask[None, None, :, None, :]
            .repeat(1, self.nt_states, 1, self.pt_states, 1)
            .view(1, nt, pt)
        )
        node_mask = torch.cat([nt_node_mask, pt_node_mask], 2).to(device)
        node_mask = node_mask.unsqueeze(3) * node_mask.unsqueeze(2)
        return node_mask.view(1, nt, nt + pt, nt + pt).expand(batch_size, -1, -1, -1)

    def get_feature(
        self, batch_size, max_pt_spans, max_nt_spans, pt_spans, nt_spans, device
    ):
        nt_node_mask = torch.eye(max_nt_spans, device=device, dtype=torch.bool)
        pt_node_mask = torch.zeros(
            max_nt_spans, max_pt_spans, device=device, dtype=torch.bool
        )
        nt = max_nt_spans * self.nt_states
        pt = max_pt_spans * self.pt_states
        nt_node_mask = (
            nt_node_mask[None, None, :, None, :]
            .repeat(1, self.nt_states, 1, self.nt_states, 1)
            .view(1, nt, nt)
        )
        pt_node_mask = (
            pt_node_mask[None, None, :, None, :]
            .repeat(1, self.nt_states, 1, self.pt_states, 1)
            .view(1, nt, pt)
        )
        node_mask = torch.cat([nt_node_mask, pt_node_mask], 2).to(device)
        node_mask = node_mask.unsqueeze(3) + node_mask.unsqueeze(2)
        return node_mask.view(1, nt, nt + pt, nt + pt).expand(batch_size, -1, -1, -1)


#   def get_rules_mask2(
#         self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
#     ):
#         # A[a i]->B[a j] C[a k], a i must be the DIRECT parent of a j and a k, j!=k.
#         #   if a i has no child, a j/k = a i.
#         nt = nt_num_nodes * self.nt_states
#         pt = pt_num_nodes * self.pt_states
#         bsz = batch_size
#         src_nt = self.nt_states
#         src_pt = self.pt_states
#         node_nt = nt_num_nodes
#         node_pt = pt_num_nodes
#         node_mask = torch.zeros(
#             bsz,
#             src_nt * node_nt,
#             src_nt * node_nt + src_pt * node_pt,
#             src_nt * node_nt + src_pt * node_pt,
#             dtype=torch.bool,
#             device=device,
#         )

#         nt_idx = slice(0, src_nt * node_nt)
#         pt_idx = slice(src_nt * node_nt, src_nt * node_nt + src_pt * node_pt)

#         nt_ntnt = node_mask[:, nt_idx, nt_idx, nt_idx].view(
#             bsz, src_nt, node_nt, src_nt, node_nt, src_nt, node_nt
#         )
#         nt_ntpt = node_mask[:, nt_idx, nt_idx, pt_idx].view(
#             bsz, src_nt, node_nt, src_nt, node_nt, src_pt, node_pt
#         )
#         nt_ptnt = node_mask[:, nt_idx, pt_idx, nt_idx].view(
#             bsz, src_nt, node_nt, src_pt, node_pt, src_nt, node_nt
#         )
#         nt_ptpt = node_mask[:, nt_idx, pt_idx, pt_idx].view(
#             bsz, src_nt, node_nt, src_pt, node_pt, src_pt, node_pt
#         )

#         def is_parent(parent, child):
#             return child[0] >= parent[0] and child[1] <= parent[1]

#         def is_strict_parent(parent, child):
#             return is_parent(parent, child) and parent != child

#         def span_len(span):
#             return span[1] - span[0] + 1

#         def covers(parent, child1, child2):
#             return (span_len(parent) == (span_len(child1) + span_len(child2))) and (
#                 (parent[0] == child1[0] and parent[1] == child2[1])
#                 or (parent[0] == child2[0] and parent[1] == child1[1])
#             )

#         for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):
#             min_nt_span = min([span_len(s) for s in nt_span])
#             for i, parent in enumerate(nt_span):
#                 if span_len(parent) == min_nt_span:
#                     nt_ntnt[b, :, i, :, i, :, i].fill_(True)
#                     for j, child in enumerate(pt_span):
#                         if is_strict_parent(parent, child):
#                             nt_ntpt[b, :, i, :, i, :, j].fill_(True)
#                             nt_ptnt[b, :, i, :, j, :, i].fill_(True)
#                 if span_len(parent) == 1:
#                     for j, child in enumerate(pt_span):
#                         if parent == child:
#                             nt_ptnt[b, :, i, :, j, :, i].fill_(True)
#                             nt_ntpt[b, :, i, :, i, :, j].fill_(True)
#                             nt_ptpt[b, :, i, :, j, :, j].fill_(True)
#                 for j, child1 in enumerate(nt_span):
#                     for k, child2 in enumerate(nt_span):
#                         if covers(parent, child1, child2):
#                             nt_ntnt[b, :, i, :, j, :, k].fill_(True)
#                             nt_ntnt[b, :, i, :, k, :, j].fill_(True)
#                     for k, child2 in enumerate(pt_span):
#                         if covers(parent, child1, child2):
#                             nt_ntpt[b, :, i, :, j, :, k].fill_(True)
#                             nt_ptnt[b, :, i, :, k, :, j].fill_(True)
#                 for j, child1 in enumerate(pt_span):
#                     for k, child2 in enumerate(pt_span):
#                         if covers(parent, child1, child2):
#                             nt_ptpt[b, :, i, :, j, :, k].fill_(True)
#                             nt_ptpt[b, :, i, :, k, :, j].fill_(True)

#         return node_mask.view(batch_size, nt, nt + pt, nt + pt)


if __name__ == "__main__":
    c = FPSimpleHierarchy(3, 3)
    batch_size = 2
    max_pt_spans = 3
    max_nt_spans = 3
    pt_spans = [[(0, 0, -1), (1, 1, -1), (2, 2, -1)], [(1, 1, -1), (2, 2, -1)]]
    nt_spans = [[(0, 3, -1), (1, 2, -1), (1, 3, -1)], [(0, 4, -1), (3, 4, -1)]]
    m1 = c.get_mask(batch_size, max_pt_spans, max_nt_spans, pt_spans, nt_spans, "cpu")
    m2 = c.get_mask_impl2(
        batch_size, max_pt_spans, max_nt_spans, pt_spans, nt_spans, "cpu"
    )
