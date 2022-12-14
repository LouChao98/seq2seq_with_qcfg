import logging
from contextlib import contextmanager
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import torch
from numba import jit
from torch import Tensor
from torch.distributions.utils import lazy_property

from src.models.struct.semiring import EntropySemiring, LogSemiring

from ._fn import ns_stripe as stripe
from ._utils import (
    check_full_marginal,
    checkpoint,
    compare_marginal,
    compare_unlabeled_marginal,
    compute_unnormalized_prob,
    enumerate_seq,
    weighted_random_v2,
)
from .base import DecompBase
from .decomp1 import Decomp1Sampler

log = logging.getLogger(__file__)


class Decomp1Fast(DecompBase):
    KEYS = ("term", "head", "left", "right", "root")
    LOGSPACE = (True, False, False, False, True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace_rank = False

    def inside(self, params, semiring, trace=False, use_reentrant=True) -> Tuple[Tensor, Tensor]:
        assert semiring == LogSemiring
        return self.inside_log_semiring(params, trace, use_reentrant)

    def inside_log_semiring(self, params, trace=False, use_reentrant=True):
        merge = checkpoint(g_merge, use_reentrant=use_reentrant)
        term: Tensor = params["term"]  # (batch, seq_len, PT)
        root: Tensor = params["root"]  # (batch, NT)
        H: Tensor = params["head"]  # (batch, NT, r), A[i] -> R
        L: Tensor = params["left"].transpose(1, 2)  # (batch, r, NT + T), r->L
        R: Tensor = params["right"].transpose(1, 2)  # (batch, r, NT + T), r->R

        batch, N, PT = term.shape
        NT = root.shape[1]
        N += 1
        rank = L.shape[2]

        TLNT = L[:, :NT]
        TLPT = L[:, NT:]
        TRNT = R[:, :NT]
        TRPT = R[:, NT:]

        if trace:
            if self._trace_rank:
                _span_indicator = term.new_ones(batch, N, N, rank, requires_grad=True)
                span_indicator = _span_indicator
            else:
                _span_indicator = term.new_ones(batch, N, N, requires_grad=True)
                span_indicator = _span_indicator.unsqueeze(-1)
            span_indicator_running = span_indicator
        else:
            _span_indicator = None

        sn = term.new_full((batch, N, N), -1e9)
        norm: Tensor = term.max(-1)[0]
        sn.diagonal(1, 1, 2).copy_(norm)
        term = (term - norm.unsqueeze(-1)).exp()

        s = term.new_zeros(batch, N, N, rank)

        # b n pt x b pt r -> b n r
        left_term = torch.matmul(term, TLPT)
        right_term = torch.matmul(term, TRPT)

        s.diagonal(1, 1, 2).copy_(left_term.transpose(1, 2))
        s.diagonal(-1, 1, 2).copy_(right_term.transpose(1, 2))

        H = H.transpose(-1, -2)
        # (m_A, r_A) + (m_B, r_B) -> (r_A, r_B)
        H_L = torch.matmul(H, TLNT)
        H_R = torch.matmul(H, TRNT)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (torch.arange(2, N + 1).unsqueeze(1) <= self.lens.cpu().unsqueeze(0)).sum(1)

        # w: span width
        final = []
        final_normalizer = []
        for step, w in enumerate(range(2, N)):

            # n: the number of spans of width w.
            n = N - w
            y = stripe(s, n, w - 1, (0, 1))
            z = stripe(s, n, w - 1, (w, 1))
            yn = stripe(sn, n, w - 1, (0, 1))
            zn = stripe(sn, n, w - 1, (1, w), 0)

            if y.requires_grad:
                y = y.clone()  # due to checkpoint
                z = z.clone()
                yn = yn.clone()
                zn = zn.clone()

            x, xn = merge(y, z, yn, zn)

            if trace:
                indicator = span_indicator_running.diagonal(w, 1, 2).movedim(-1, 1)
                x = x * indicator

            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            if current_bsz - unfinished > 0:
                final.insert(0, x[unfinished:current_bsz, :1])
                final_normalizer.insert(0, xn[unfinished:current_bsz, :1])

            if unfinished > 0:
                if current_bsz > unfinished:
                    x = x[:unfinished]
                    s = s[:unfinished]

                    H_L = H_L[:unfinished]
                    H_R = H_R[:unfinished]
                    sn = sn[:unfinished]
                    xn = xn[:unfinished]
                    if trace:
                        span_indicator_running = span_indicator_running[:unfinished]

                left_x = torch.matmul(x, H_L)
                right_x = torch.matmul(x, H_R)
                s.diagonal(w, 1, 2).copy_(left_x.transpose(1, 2))
                s.diagonal(-w, 1, 2).copy_(right_x.transpose(1, 2))
                sn.diagonal(w, 1, 2).copy_(xn)

            if unfinished == 0:
                break

        final = torch.cat(final, dim=0)
        final_normalizer = torch.cat(final_normalizer, dim=0)
        final = torch.matmul(final, H)
        final = (final + 1e-9).squeeze(1).log() + root
        logZ = final.logsumexp(-1) + final_normalizer.squeeze(-1)
        return logZ, _span_indicator

    # NOTE: entropy, cross_entropy and kl are slower than the normal version but save memory
    # because we need marginals here

    @lazy_property
    def entropy(self):
        marginal = self.rule_marginal_with_grad
        result = self.partition.clone()
        for k in self.KEYS:
            result -= (self.log_params[k] * marginal[k]).flatten(1).sum(-1)
        return result

    def cross_entropy(self, other: "DecompBase", fix_left=False):
        # self = p, other = q, ce(q, p)
        q = self
        p = other
        qm = q.rule_marginal_with_grad
        result = p.partition.clone()
        for k in self.KEYS:
            qmk = qm[k]
            if fix_left:
                qmk = qmk.detach()
            result -= (qmk * p.log_params[k]).flatten(1).sum(-1)
        return result

    def kl(self, other: "DecompBase", fix_left=False):
        q = self
        p = other
        qm = q.rule_marginal_with_grad
        result = p.partition - q.partition
        for k in self.KEYS:
            qmk = qm[k]
            if fix_left:
                qmk = qmk.detach()
            result += (qmk * (q.log_params[k] - p.log_params[k])).flatten(1).sum(-1)
        return result

    @staticmethod
    def random(bsz, max_len, tgt_pt, src_pt, tgt_nt, src_nt, r):
        nt = tgt_nt * src_nt
        pt = tgt_pt * src_pt
        return {
            "term": torch.randn(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "head": torch.randn(bsz, nt, r).softmax(-1).requires_grad_(True),
            "left": torch.randn(bsz, r, nt + pt).softmax(-1).requires_grad_(True),
            "right": torch.randn(bsz, r, nt + pt).softmax(-1).requires_grad_(True),
        }


def convert_decomp1_to_pcfg(p, tgt_nt):
    rule = torch.einsum("xar,xrb,xrc->xabc", p["head"], p["left"], p["right"])
    rule = (rule + 1e-9).log()
    output = {"term": p["term"], "rule": rule, "root": p["root"]}
    if "constraint" in p:
        output["constraint"] = p["constraint"]
    if "add" in p:
        output["add"] = p["add"]
    if "lse" in p:
        output["lse"] = p["lse"]
    return output


# @torch.jit.script
def g_merge(y, z, yn, zn):
    # contract dimension w.
    y = (y + 1e-9).log() + yn.unsqueeze(-1)
    z = (z + 1e-9).log() + zn.unsqueeze(-1)
    b_n_r = (y + z).logsumexp(-2)
    normalizer = b_n_r.max(-1)[0]
    b_n_r = (b_n_r - normalizer.unsqueeze(-1)).exp()
    return b_n_r, normalizer


if __name__ == "__main__":
    from torch_struct import SentCFG

    from .decomp1 import Decomp1

    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 2, 4, 2, 5, 3, 7
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    r = 3
    lens = [max(2, N - i) for i in range(B)]
    params = Decomp1Fast.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    params_ref = convert_decomp1_to_pcfg(params, TGT_NT)
    params_ref2 = {key: value.detach().requires_grad_() for key, value in params.items()}
    params_ref2["head"] = params_ref2["head"].detach().clamp(1e-9).log().requires_grad_()
    params_ref2["left"] = params_ref2["left"].detach().clamp(1e-9).log().requires_grad_()
    params_ref2["right"] = params_ref2["right"].detach().clamp(1e-9).log().requires_grad_()
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }

    pcfg = Decomp1Fast(params, lens, **meta)
    pcfg_ref = SentCFG((params_ref["term"], params_ref["rule"], params_ref["root"]), lens)
    pcfg_ref2 = Decomp1(params_ref2, lens, **meta)

    print("test nll")
    nll = pcfg.nll
    nll_ref = -pcfg_ref.partition
    assert torch.allclose(nll, nll_ref), (nll, nll_ref)

    print("test marginal")
    m1 = pcfg.marginal
    check_full_marginal(m1["term"], m1["trace"].sum(-1), lens)

    m2 = pcfg_ref.marginals[-1]
    compare_unlabeled_marginal(m1["trace"].sum(-1), m2)

    print("test entropy")
    print(pcfg.entropy)
    print(pcfg_ref2.entropy)

    pcfg.entropy.sum().backward()
    pcfg_ref2.entropy.sum().backward()

    assert torch.allclose(pcfg.params["term"].grad, pcfg_ref2.params["term"].grad, atol=1e-6, rtol=1e-4)
    print("pass")
