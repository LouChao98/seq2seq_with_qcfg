from logging import getLogger

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from src.utils.fn import spans2tree

from .base import RuleConstraintBase

logger = getLogger(__file__)


def is_parent(parent, child):
    return child[0] >= parent[0] and child[1] <= parent[1]


def is_strict_parent(parent, child):
    return is_parent(parent, child) and parent[:2] != child[:2]


def span_len(span):
    return span[1] - span[0]


def covers(parent, child1, child2):
    return (span_len(parent) == (span_len(child1) + span_len(child2))) and (
        (parent[0] == child1[0] and parent[1] == child2[1]) or (parent[0] == child2[0] and parent[1] == child1[1])
    )


def intersect(span1, span2):
    return max(span1[0], span2[0]) < min(span1[1], span2[1])


class FPSimpleHierarchy(RuleConstraintBase):
    def get_mask(self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device):
        pt_spans[0] += [(0, 0, -999)] * (pt_num_nodes - len(pt_spans[0]))  # TODO any issue here?
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
    # p(m) ^ k > p(0) ^ {k-1} p(mk), assume p(0) -> 1
    # k log p(m) >= log p(mk)

    def __init__(
        self, upwards_score=1e-4, stay_score=0.9, nt_temperature=1.0, pt_temperature=1.0, infeasible_score=1e-4
    ) -> None:
        super().__init__()
        self.infeasible_score = np.log(infeasible_score)
        self.upwards_score = np.log(upwards_score)
        self.stay_score = np.log(stay_score)
        self.nt_temperature = nt_temperature
        self.pt_temperature = pt_temperature
        self.nt_score = -np.linspace(0, 99 / nt_temperature, 100)
        self.pt_score = -np.linspace(0, 99 / pt_temperature, 100)

    def get_weight(self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device):
        nt = nt_num_nodes * nt_states
        pt = pt_num_nodes * pt_states
        node_score = torch.full((batch_size, nt, nt + pt), self.infeasible_score)

        nt_idx = slice(0, nt)
        pt_idx = slice(nt, nt + pt)

        nt_ntnt = node_score[:, nt_idx, nt_idx].view(batch_size, nt_states, nt_num_nodes, nt_states, nt_num_nodes)
        nt_ntpt = node_score[:, nt_idx, pt_idx].view(batch_size, nt_states, nt_num_nodes, pt_states, pt_num_nodes)

        for b, (nt_spans_inst, pt_spans_inst) in enumerate(zip(nt_spans, pt_spans)):

            parents = spans2tree(nt_spans_inst)

            nt_depths = []
            pt_depths = [0] * len(pt_spans_inst)
            for j in range(len(nt_spans_inst)):
                depth = 0
                k = parents[j]
                while k != -1:
                    k = parents[k]
                    depth += 1
                nt_depths.append(depth)

            for i, span1 in enumerate(nt_spans_inst):
                nt_ntnt[b, :, i, :, i] = self.stay_score
                for j, span2 in enumerate(nt_spans_inst):
                    if not (is_parent(span1, span2)) or i == j:
                        continue
                    distance = nt_depths[j] - nt_depths[i]
                    assert distance > 0
                    nt_ntnt[b, :, i, :, j] = self.nt_score[distance - 1]
                    nt_ntnt[b, :, j, :, i] = self.upwards_score
                for j, pt_span in enumerate(pt_spans_inst):
                    if not (is_parent(span1, pt_span)):
                        continue
                    pt_depths[j] = max(pt_depths[j], nt_depths[i] + 1)

            for i, span1 in enumerate(nt_spans_inst):
                for j, span2 in enumerate(pt_spans_inst):
                    if is_parent(span1, span2):
                        distance = pt_depths[j] - nt_depths[i]
                        assert distance > 0
                        nt_ntpt[b, :, i, :, j] = self.pt_score[distance - 1]

        node_score = node_score.to(device)
        node_score = torch.min(node_score.unsqueeze(2), node_score.unsqueeze(3))
        return node_score

    def get_feature(self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device):
        return -self.get_weight(
            batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device
        )


class FPPenaltyDepthAndNonIntersection(FPPenaltyDepth):
    def __init__(self, intersect_score=0.8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.intersect_score = np.log(intersect_score)

    def get_weight(self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device):
        node_score = super().get_weight(
            batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device
        )
        nt = nt_num_nodes * nt_states
        pt = pt_num_nodes * pt_states
        nt_idx = slice(0, nt)
        pt_idx = slice(nt, nt + pt)

        nt_ntnt = node_score[:, nt_idx, nt_idx, nt_idx].view(
            batch_size, nt_states, nt_num_nodes, nt_states, nt_num_nodes, nt_states, nt_num_nodes
        )
        nt_ntpt = node_score[:, nt_idx, nt_idx, pt_idx].view(
            batch_size, nt_states, nt_num_nodes, nt_states, nt_num_nodes, pt_states, pt_num_nodes
        )
        nt_ptnt = node_score[:, nt_idx, pt_idx, nt_idx].view(
            batch_size, nt_states, nt_num_nodes, pt_states, pt_num_nodes, nt_states, nt_num_nodes
        )
        nt_ptpt = node_score[:, nt_idx, pt_idx, pt_idx].view(
            batch_size, nt_states, nt_num_nodes, pt_states, pt_num_nodes, pt_states, pt_num_nodes
        )

        for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):
            for j, child1 in enumerate(nt_span):
                for k, child2 in enumerate(nt_span):
                    if intersect(child1, child2):
                        nt_ntnt[b, :, :, :, j, :, k] += self.intersect_score
                        if j == k:
                            nt_ntnt[b, :, j, :, j, :, k] -= self.intersect_score
            for j, child1 in enumerate(nt_span):
                for k, child2 in enumerate(pt_span):
                    if intersect(child1, child2):
                        nt_ntpt[b, :, :, :, j, :, k] += self.intersect_score
                        nt_ptnt[b, :, :, :, k, :, j] += self.intersect_score
            for j, child1 in enumerate(pt_span):
                for k, child2 in enumerate(pt_span):
                    if intersect(child1, child2):
                        nt_ptpt[b, :, :, :, j, :, k] += self.intersect_score
        return node_score


class FPPenaltyBranching(RuleConstraintBase):
    def get_feature(self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device):
        nt = nt_num_nodes * nt_states
        pt = pt_num_nodes * pt_states
        node_score = torch.zeros((batch_size, nt, nt + pt, nt + pt), device=device)

        nt_idx = slice(0, nt)
        pt_idx = slice(nt, nt + pt)

        node_score[:, :, nt_idx, nt_idx] = -1.0
        node_score[:, :, pt_idx, pt_idx] = -1.0
        return node_score

    def get_mask(self, *args, **kwargs):
        raise NotImplementedError

    def get_weight(self, *args, **kwargs):
        raise NotImplementedError
