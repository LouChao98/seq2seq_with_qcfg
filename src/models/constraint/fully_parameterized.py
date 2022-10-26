from collections import defaultdict
from logging import getLogger
from pprint import pprint

import torch
from torch.nn.utils.rnn import pad_sequence

from src.utils.fn import spans2tree

from .base import RuleConstraintBase

logger = getLogger(__file__)


def is_parent(parent, child):
    return child[0] >= parent[0] and child[1] <= parent[1]


def is_strict_parent(parent, child):
    return is_parent(parent, child) and parent != child


def span_len(span):
    return span[1] - span[0] + 1


def covers(parent, child1, child2):
    return (span_len(parent) == (span_len(child1) + span_len(child2))) and (
        (parent[0] == child1[0] and parent[1] == child2[1]) or (parent[0] == child2[0] and parent[1] == child1[1])
    )


class FPSimpleHierarchy(RuleConstraintBase):
    def get_mask(self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device):
        pt_spans[0] += [(0, 0, -999)] * (pt_num_nodes - len(pt_spans[0]))
        nt_spans[0] += [(0, 0, -999)] * (nt_num_nodes - len(nt_spans[0]))
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

        nt = nt_states * nt_num_nodes
        pt = pt_states * pt_num_nodes
        nt_node_mask = nt_node_mask[:, None, :, None, :].repeat(1, nt_states, 1, nt_states, 1)
        nt_node_mask = nt_node_mask.view(batch_size, nt, nt)

        pt_node_mask = pt_node_mask[:, None, :, None, :].repeat(1, nt_states, 1, pt_states, 1)
        pt_node_mask = pt_node_mask.view(batch_size, nt, pt)

        node_mask = torch.cat([nt_node_mask, pt_node_mask], 2).to(device)
        node_mask = node_mask.unsqueeze(3) * node_mask.unsqueeze(2)
        return node_mask.view(batch_size, nt, nt + pt, nt + pt)


class FPSynchronous(RuleConstraintBase):
    def __init__(self) -> None:
        super().__init__()
        # if len(src) >> len(tgt), tgt may be not able to reach PT.
        logger.warning("src pt should include src nt to avoid infeasible tree.")

    def get_mask(self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device):

        # A[a i]->B[a j] C[a k], a i must be the DIRECT parent of a j and a k, j!=k.
        #   if a i has no child, a j/k = a i.
        nt = nt_num_nodes * nt_states
        pt = pt_num_nodes * pt_states
        node_mask = torch.zeros(batch_size, nt, nt + pt, nt + pt, dtype=torch.bool)

        nt_idx = slice(0, nt)
        pt_idx = slice(nt, nt + pt)

        nt_ntnt = node_mask[:, nt_idx, nt_idx, nt_idx].view(
            batch_size, nt_states, nt_num_nodes, nt_states, nt_num_nodes, nt_states, nt_num_nodes
        )
        nt_ntpt = node_mask[:, nt_idx, nt_idx, pt_idx].view(
            batch_size, nt_states, nt_num_nodes, nt_states, nt_num_nodes, pt_states, pt_num_nodes
        )
        nt_ptnt = node_mask[:, nt_idx, pt_idx, nt_idx].view(
            batch_size, nt_states, nt_num_nodes, pt_states, pt_num_nodes, nt_states, nt_num_nodes
        )
        nt_ptpt = node_mask[:, nt_idx, pt_idx, pt_idx].view(
            batch_size, nt_states, nt_num_nodes, pt_states, pt_num_nodes, pt_states, pt_num_nodes
        )

        for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):
            min_nt_span = min([span_len(s) for s in nt_span])
            for i, parent in enumerate(nt_span):
                if span_len(parent) == min_nt_span:
                    nt_ntnt[b, :, i, :, i, :, i].fill_(True)
                    for j, child in enumerate(pt_span):
                        if is_strict_parent(parent, child):
                            nt_ntpt[b, :, i, :, i, :, j].fill_(True)
                            nt_ptnt[b, :, i, :, j, :, i].fill_(True)
                if span_len(parent) == 1:
                    for j, child in enumerate(pt_span):
                        if parent == child:
                            nt_ptnt[b, :, i, :, j, :, i].fill_(True)
                            nt_ntpt[b, :, i, :, i, :, j].fill_(True)
                            nt_ptpt[b, :, i, :, j, :, j].fill_(True)
                for j, child1 in enumerate(nt_span):
                    for k, child2 in enumerate(nt_span):
                        if covers(parent, child1, child2):
                            nt_ntnt[b, :, i, :, j, :, k].fill_(True)
                            nt_ntnt[b, :, i, :, k, :, j].fill_(True)
                    for k, child2 in enumerate(pt_span):
                        if covers(parent, child1, child2):
                            nt_ntpt[b, :, i, :, j, :, k].fill_(True)
                            nt_ptnt[b, :, i, :, k, :, j].fill_(True)
                for j, child1 in enumerate(pt_span):
                    for k, child2 in enumerate(pt_span):
                        if covers(parent, child1, child2):
                            nt_ptpt[b, :, i, :, j, :, k].fill_(True)
                            nt_ptpt[b, :, i, :, k, :, j].fill_(True)

        return node_mask.view(batch_size, nt, nt + pt, nt + pt).to(device)


class FPPenaltyDepth(RuleConstraintBase):
    def __init__(
        self, upwards_score=0.7, stay_score=0.9, down1_score=1, down2_score=0.9, down3_score=0.8, down_score=0.7
    ) -> None:
        super().__init__()
        self.upwards_score = upwards_score
        self.stay_score = stay_score
        self.down1_score = down1_score
        self.down2_score = down2_score
        self.down3_score = down3_score
        self.down_score = down_score

    def get_mask(self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device):
        raise NotImplementedError

    def get_reward(self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device):
        nt = nt_num_nodes * nt_states
        pt = pt_num_nodes * pt_states
        node_score = torch.zeros(batch_size, nt, nt + pt)

        nt_idx = slice(0, nt)
        pt_idx = slice(nt, nt + pt)

        nt_ntnt = node_score[:, nt_idx, nt_idx].view(batch_size, nt_states, nt_num_nodes, nt_states, nt_num_nodes)
        nt_ntpt = node_score[:, nt_idx, pt_idx].view(batch_size, nt_states, nt_num_nodes, pt_states, pt_num_nodes)
        nt_ntpt.fill_(1.0)

        for b, nt_spans_inst in enumerate(nt_spans):
            spans, parents, mapping_ = spans2tree(nt_spans_inst, return_mapping=True)
            mapping = list(range(len(mapping_)))
            mapping.sort(key=lambda x: mapping_[x])
            for i, span1 in enumerate(spans):
                nt_ntnt[b, :, mapping[i], :, mapping[i]] = self.stay_score
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
                    if depth == 1:
                        nt_ntnt[b, :, mapping[i], :, mapping[j]] = self.down1_score
                    elif depth == 2:
                        nt_ntnt[b, :, mapping[i], :, mapping[j]] = self.down2_score
                    elif depth == 3:
                        nt_ntnt[b, :, mapping[i], :, mapping[j]] = self.down3_score
                    else:
                        nt_ntnt[b, :, mapping[i], :, mapping[j]] = self.down_score
                    nt_ntnt[b, :, mapping[j], :, mapping[i]] = self.upwards_score

        node_score = node_score.to(device)
        node_score = node_score.unsqueeze(2) * node_score.unsqueeze(3)
        return node_score


def show_constraint(ntnt, ntpt, ptnt, ptpt, nt_spans, pt_spans):
    # ntnt/ntpt/ptnt/ptpt: src_nt * src_{nt,pt} * src_{nt,pt},
    # show_constraint(nt_ntnt[0, 0, :, 0, :, 0, :], nt_ntpt[0, 0, :, 0, :, 0, :], nt_ptnt[0, 0, :, 0, :, 0, :], nt_ptpt[0, 0, :, 0, :, 0, :], nt_spans[0], pt_spans[0])
    ntnt = ntnt[:, : len(nt_spans), : len(pt_spans)].nonzero(as_tuple=True)
    ntpt = ntpt[:, : len(nt_spans), : len(pt_spans)].nonzero(as_tuple=True)
    ptnt = ptnt[:, : len(pt_spans), : len(nt_spans)].nonzero(as_tuple=True)
    ptpt = ptpt[:, : len(pt_spans), : len(pt_spans)].nonzero(as_tuple=True)
    allowed = defaultdict(list)
    nt_spans = [(i, j, "n") for i, j, _ in nt_spans]
    pt_spans = [(i, j, "p") for i, j, _ in pt_spans]
    for i, j, k in zip(*ntnt):
        allowed[nt_spans[i]].append((nt_spans[j], nt_spans[k]))
    for i, j, k in zip(*ntpt):
        allowed[nt_spans[i]].append((nt_spans[j], pt_spans[k]))
    for i, j, k in zip(*ptnt):
        allowed[nt_spans[i]].append((pt_spans[j], nt_spans[k]))
    for i, j, k in zip(*ptpt):
        allowed[nt_spans[i]].append((pt_spans[j], pt_spans[k]))
    for i, vset in allowed.items():
        print("Parent: ", i)
        print("Possible children: ")
        pprint(vset)
        print("===")


if __name__ == "__main__":
    c = FPSimpleHierarchy(3, 3)
    batch_size = 2
    max_pt_spans = 3
    max_nt_spans = 3
    pt_spans = [[(0, 0, -1), (1, 1, -1), (2, 2, -1)], [(1, 1, -1), (2, 2, -1)]]
    nt_spans = [[(0, 3, -1), (1, 2, -1), (1, 3, -1)], [(0, 4, -1), (3, 4, -1)]]
    m1 = c.get_mask(batch_size, max_pt_spans, max_nt_spans, pt_spans, nt_spans, "cpu")
    m2 = c.get_mask_impl2(batch_size, max_pt_spans, max_nt_spans, pt_spans, nt_spans, "cpu")
