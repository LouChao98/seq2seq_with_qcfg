import logging
from typing import List, Tuple

import numpy as np
import torch

from ..struct.decomp1_copy_as_t import Decomp1, Decomp1Sampler
from .base import NO_COPY_SPAN, TgtParserBase, TgtParserPrediction
from .neural_decomp1 import NeuralDecomp1TgtParser as _BaseModel

log = logging.getLogger(__file__)


class NeuralDecomp1TgtParser(_BaseModel):
    def __init__(self, *args, max_copy_width=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_copy_width = max_copy_width

    def observe_x(self, pred: TgtParserPrediction, x, lengths, inplace=True, **kwargs) -> TgtParserPrediction:
        pred = super().observe_x(pred, x, lengths, inplace, **kwargs)
        pred.dist = Decomp1(pred.posterior_params, pred.lengths, **pred.common())
        return pred

    def prepare_sampler(self, pred: TgtParserPrediction, src, src_ids, inplace=True) -> TgtParserPrediction:
        pred = super().prepare_sampler(pred, src, src_ids, inplace)
        pred.sampler = Decomp1Sampler(pred.params, **pred.common(), **self.sampler_common())
        return pred

    def build_rules_give_tgt(
        self,
        tgt: torch.Tensor,
        term: torch.Tensor,
        root: torch.Tensor,
        max_pt_spans: int,
        pt_spans: List[List[Tuple[int, int, int]]],
        max_nt_spans: int,
        nt_spans: List[List[Tuple[int, int, int]]],
        pt: int,
        nt: int,
        pt_copy=None,
        nt_copy=None,
        observed_mask=None,
        prior_alignment=None,
    ):
        assert observed_mask is None or not self.use_observed
        assert prior_alignment is None
        batch_size, n = tgt.shape[:2]
        term = term.unsqueeze(1).expand(batch_size, n, pt, term.size(2))
        tgt_expand = tgt.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
        term = torch.gather(term, 3, tgt_expand).squeeze(3)
        term2d = term.new_full(
            (batch_size, term.shape[1], self.max_copy_width, self.pt_states, max_pt_spans), -1e9, device="cpu"
        )

        if self.use_copy:
            if pt_copy is not None:
                term = self.build_pt_copy_constraint(term, pt_copy)
            if nt_copy is not None:
                # NOTE WE USE PT_SPANS!!!
                term2d = self.build_nt_copy_constraint(term2d, pt_spans, nt_copy)

        term2d = term2d.to(term.device)
        term2d = term2d.flatten(3)
        term2d[:, :, 0] = term
        term = term2d

        constraint_scores = None
        lse_scores = None
        add_scores = None

        if constraint_scores is not None:
            constraint_scores = self.post_process_nt_constraint(constraint_scores, tgt.device)

        if observed_mask is not None and self.use_observed:
            constraint_scores = self.build_observed_span_constraint(
                batch_size, n, max_nt_spans, observed_mask, constraint_scores
            )

        return {"term": term, "root": root, "constraint": constraint_scores, "lse": lse_scores, "add": add_scores}

    def build_nt_copy_constraint(self, terms2d, pt_spans, copy_position):
        for batch_idx, (pt_spans_inst, possible_copy) in enumerate(zip(pt_spans, copy_position)):
            for i, (l, r, tag) in enumerate(pt_spans_inst):
                if tag == NO_COPY_SPAN:
                    continue
                w = r - l - 2
                t = None
                if w >= len(possible_copy) or w < 0 or (w + 1) >= self.max_copy_width:
                    continue
                for possible_s, possible_t in possible_copy[w]:
                    if possible_s == l:
                        t = possible_t
                        break
                if t is not None:
                    terms2d[batch_idx, t, w + 1, -1, i] = 0.0
        return terms2d

    def get_init_nt_constraint(self, batch_size, n):
        return [
            (
                np.full(
                    (batch_size, n - w, self.cpd_rank),
                    self.neg_huge,
                    dtype=np.float32,
                ),
                np.zeros((batch_size, n - w, self.cpd_rank), dtype=np.bool8),
            )
            for w in range(1, n)
        ]

    def build_observed_span_constraint(self, batch_size, n, max_nt_spans, observed_constraint, constraint=None):
        if constraint is None:
            constraint = self.post_process_nt_constraint(
                self.get_init_nt_constraint(batch_size, n), observed_constraint[0].device
            )
        for item, (value, mask) in zip(observed_constraint, constraint):
            mask |= item.view(list(item.shape) + [1] * (mask.ndim - item.ndim))
            value[item] = self.neg_huge
        return constraint
