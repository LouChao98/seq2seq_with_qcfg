import logging
from contextlib import contextmanager
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import torch
from numba import jit
from torch import Tensor
from torch.autograd import grad
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
from .decomp7 import Decomp7Sampler

log = logging.getLogger(__file__)


class Decomp7Fast(DecompBase):
    KEYS = ("term", "head", "left", "right", "slr", "root")
    LOGSPACE = (True, False, False, False, False, True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace_rank = False

    @contextmanager
    def trace_rank(self):
        self._trace_rank = True
        # reset lazy property
        if hasattr(self, "marginal"):
            delattr(self, "marginal")
        yield
        self._trace_rank = False

    def inside(self, params, semiring, trace=False, use_reentrant=True) -> Tuple[Tensor, Tensor]:
        assert semiring == LogSemiring
        return self.inside_log_semiring(params, trace, use_reentrant)

    def inside_log_semiring(self, params, trace=False, use_reentrant=True) -> Tuple[Tensor, Tensor]:
        merge_h = checkpoint(g_merge_h, use_reentrant=use_reentrant)
        merge_h2 = checkpoint(g_merge_h2, use_reentrant=use_reentrant)
        merge_h3 = checkpoint(g_merge_h3, use_reentrant=use_reentrant)

        head: Tensor = params["head"]  # (batch, nt, r), A[i] -> R
        term: Tensor = params["term"]  # (batch, seq_len, PT)
        root: Tensor = params["root"]  # (batch, NT)
        constraint = params.get("constraint")

        batch, N, PT = term.shape
        _, NT, R = head.shape
        N += 1
        nt_spans = NT // self.nt_states
        pt_spans = PT // self.pt_states

        head = head.view(batch, self.nt_states, nt_spans, R)
        term = term.view(batch, -1, self.pt_states, pt_spans)
        root = root.view(batch, self.nt_states, nt_spans)

        # (batch, r, SRC, TGT_NT), R -> B/C
        # (batch, r, SRC, TGT_PT), R -> B/C
        size = [nt_spans, pt_spans]
        TLNT, TLPT = torch.split(params["left"], size, 2)
        TRNT, TRPT = torch.split(params["right"], size, 2)
        TLNT = TLNT[..., : self.nt_states]
        TLPT = TLPT[..., : self.pt_states]
        TRNT = TRNT[..., : self.nt_states]
        TRPT = TRPT[..., : self.pt_states]

        SLR = params["slr"]
        SL1R1 = SLR[:, :, :, nt_spans:, nt_spans:]
        SL1R = SLR[:, :, :, nt_spans:, :nt_spans]
        SLR1 = SLR[:, :, :, :nt_spans, nt_spans:]
        SLR = SLR[:, :, :, :nt_spans, :nt_spans]

        if trace:
            span_indicator = term.new_ones(batch, N, N, self.nt_states, nt_spans, requires_grad=True)
            span_indicator_running = span_indicator
        else:
            span_indicator = None
            span_indicator_running = None

        norm: Tensor = term.amax((2, 3), keepdim=True)
        term = (term - norm).exp()
        left_term = (torch.einsum("xlpi,xrip->xlir", term, TLPT) + 1e-9).log() + norm
        right_term = (torch.einsum("xlpi,xrip->xlir", term, TRPT) + 1e-9).log() + norm

        s = term.new_full((batch, N, N, nt_spans, R), -1e9)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (torch.arange(2, N + 1).unsqueeze(1) <= self.lens.cpu().unsqueeze(0)).sum(1)

        # w: span width
        final = []
        final_normalizer = []
        for step, w in enumerate(range(2, N)):
            # n: the number of spans of width w.
            n = N - w
            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            y_term = left_term[:, : N - w].unsqueeze(2)
            z_term = right_term[:, w - 1 :].unsqueeze(2)

            if w == 2:
                x, xn = merge_h(y_term, z_term, SL1R1, head)
            else:
                y = stripe(s, n, w - 2, (0, 2)).clone()
                z = stripe(s, n, w - 2, (w, 1)).clone()

                if w == 3:
                    x, xn = merge_h2(y, z, y_term, z_term, SL1R, SLR1, head)
                else:
                    x, xn = merge_h3(y, z, y_term, z_term, SL1R, SLR1, SLR, head)

            if constraint is not None:
                value, mask = constraint[step]
                if value.ndim > 0:
                    value = value[:current_bsz]
                mask = mask[:current_bsz]
                x, xn = set_score(x, xn, mask, value)

            if trace:
                indicator = span_indicator_running.diagonal(w, 1, 2).movedim(-1, 1)
                x = x * indicator

            if current_bsz - unfinished > 0:
                final.insert(0, x[unfinished:current_bsz, :1])
                final_normalizer.insert(0, xn[unfinished:current_bsz, :1])

            if unfinished > 0:
                if current_bsz > unfinished:
                    x = x[:unfinished]
                    xn = xn[:unfinished]
                    s = s[:unfinished]
                    left_term = left_term[:unfinished]
                    right_term = right_term[:unfinished]
                    head = head[:unfinished]
                    SLR = SLR[:unfinished]
                    SL1R = SL1R[:unfinished]
                    SLR1 = SLR1[:unfinished]
                    TLNT = TLNT[:unfinished]
                    TRNT = TRNT[:unfinished]
                    if trace:
                        span_indicator_running = span_indicator_running[:unfinished]

                left_x = (torch.einsum("xlpi,xrip->xirl", x, TLNT) + 1e-9).log() + xn[:, None, None]
                right_x = (torch.einsum("xlpi,xrip->xirl", x, TRNT) + 1e-9).log() + xn[:, None, None]
                s.diagonal(w, 1, 2).copy_(left_x)
                s.diagonal(-w, 1, 2).copy_(right_x)

            if unfinished == 0:
                break

        final = torch.cat(final, dim=0).squeeze(1)
        final_normalizer = torch.cat(final_normalizer, dim=0)
        final = (final + 1e-9).log() + final_normalizer.unsqueeze(-1)
        final = final + root
        logZ = final.flatten(1).logsumexp(1)
        return logZ, span_indicator

    @lazy_property
    def marginal(self):
        params = {}
        for key, value in self.params.items():
            if key in self.KEYS:
                params[key] = value.detach().requires_grad_()
            else:
                params[key] = value
        logZ, trace = self.inside(params, LogSemiring, trace=True)
        logZ.sum().backward()
        output = {}
        for k, is_in_log_space in zip(self.KEYS, self.LOGSPACE):
            g = params[k].grad
            if is_in_log_space:
                output[k] = g
            else:
                output[k] = g * params[k].detach()
        if isinstance(trace, tuple):
            output["trace"] = trace[0].grad
            output["term"] = trace[1].grad
        else:
            output["trace"] = trace.grad
        return output

    @lazy_property
    def marginal_with_grad(self):
        params = {}
        # TODO only check term should be enough
        for key, value in self.params.items():
            if key in self.KEYS and not value.requires_grad:
                params[key] = value.requires_grad_()
            else:
                params[key] = value
        logZ, trace = self.inside(self.params, LogSemiring, trace=True, use_reentrant=False)
        if isinstance(trace, tuple):
            grads = grad(logZ.sum(), [trace[1], trace[0]], create_graph=True)
        else:
            grads = grad(logZ.sum(), [self.params["term"], trace], create_graph=True)
        return grads

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
        src = src_nt + src_pt
        slr = (
            (5 * torch.randn(bsz, r, src_nt, src_nt + src_pt, src_nt + src_pt))
            .flatten(3)
            .softmax(-1)
            .view(bsz, r, src_nt, src_nt + src_pt, src_nt + src_pt)
        )
        return {
            "term": (5 * torch.randn(bsz, max_len, pt)).log_softmax(-1).requires_grad_(True),
            "root": (5 * torch.randn(bsz, nt)).log_softmax(-1).requires_grad_(True),
            "head": (5 * torch.randn(bsz, nt, r)).softmax(-1).requires_grad_(True),
            "left": (5 * torch.randn(bsz, r, src, max(tgt_nt, tgt_pt))).softmax(-1).requires_grad_(True),
            "right": (5 * torch.randn(bsz, r, src, max(tgt_nt, tgt_pt))).softmax(-1).requires_grad_(True),
            "slr": slr,
        }


def convert_decomp7fast_to_pcfg(params, nt_states):
    slr = params["slr"]
    term = params["term"]
    left = params["left"]
    right = params["right"]
    bsz, r, src_nt, src, _ = slr.shape
    src_pt = src - src_nt
    pt_states = term.shape[2] // src_pt
    head = params["head"].view(bsz, nt_states, src_nt, r)
    rule11 = torch.einsum(
        "xair,xrjb,xrkc,xrijk->xaibjck",
        head,
        left[:, :, :src_nt, :nt_states],
        right[:, :, :src_nt, :nt_states],
        params["slr"][:, :, :, :src_nt, :src_nt],
    )
    rule12 = torch.einsum(
        "xair,xrjb,xrkc,xrijk->xaibjck",
        head,
        left[:, :, :src_nt, :nt_states],
        right[:, :, src_nt:, :pt_states],
        params["slr"][:, :, :, :src_nt, src_nt:],
    )
    rule21 = torch.einsum(
        "xair,xrjb,xrkc,xrijk->xaibjck",
        head,
        left[:, :, src_nt:, :pt_states],
        right[:, :, :src_nt, :nt_states],
        params["slr"][:, :, :, src_nt:, :src_nt],
    )
    rule22 = torch.einsum(
        "xair,xrjb,xrkc,xrijk->xaibjck",
        head,
        left[:, :, src_nt:, :pt_states],
        right[:, :, src_nt:, :pt_states],
        params["slr"][:, :, :, src_nt:, src_nt:],
    )
    rule = rule11.new_zeros(
        bsz,
        nt_states * src_nt,
        (nt_states * src_nt) + (pt_states * src_pt),
        (nt_states * src_nt) + (pt_states * src_pt),
    )
    shape = rule11.shape
    rule[:, :, : nt_states * src_nt, : nt_states * src_nt] = (
        rule11.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]) + 1e-9
    ).log()
    shape = rule12.shape
    rule[:, :, : nt_states * src_nt, nt_states * src_nt :] = (
        rule12.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]) + 1e-9
    ).log()
    shape = rule21.shape
    rule[:, :, nt_states * src_nt :, : nt_states * src_nt] = (
        rule21.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]) + 1e-9
    ).log()
    shape = rule22.shape
    rule[:, :, nt_states * src_nt :, nt_states * src_nt :] = (
        rule22.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]) + 1e-9
    ).log()
    output = {"term": term, "rule": rule, "root": params["root"]}
    if "constraint" in params:
        output["constraint"] = params["constraint"]
    if "add" in params:
        output["add"] = params["add"]
    if "lse" in params:
        output["lse"] = params["lse"]
    return output


class Decomp7FastSampler(Decomp7Sampler):
    # not faster than Decomp1Sampler. just match the name
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(2).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        H = params["head"].cumsum(2).cpu().numpy()
        L = params["left"].cumsum(3).cpu().numpy()
        R = params["right"].cumsum(3).cpu().numpy()
        SLR = params["slr"].flatten(3).cumsum(3).cpu().numpy()

        # SLR_debug = self.threshold(params["slr"]).flatten(3).cpu().numpy()

        return terms, H, L, R, SLR, roots


def compose_bigger_span(y, z):
    # qnwjr, qnwkr -> qnrjk
    x = y.transpose(-1, -2).unsqueeze(-1) + z.transpose(-1, -2).unsqueeze(-2)
    x = x.logsumexp(2)
    return x


def normalize_tensors(*tlist):
    max_val = [t.amax((2, 3, 4)) for t in tlist]
    normalizer = torch.stack(max_val, dim=-1).max(-1)[0]
    shape = list(normalizer.shape) + [1] * len((2, 3, 4))
    tlist = [(t - normalizer.view(shape)).exp() for t in tlist]
    return tlist, normalizer


def g_merge_h(y, z, slr, h):
    x = compose_bigger_span(y, z)
    (x,), xn = normalize_tensors(x)
    x = torch.einsum("qnrjk,qrijk,qair->qnai", x, slr, h)
    return x, xn


def g_merge_h2(y, z, y_term, z_term, sl1r, slr1, h):
    x1 = compose_bigger_span(y_term, z)
    x3 = compose_bigger_span(y, z_term)
    (x1, x3), xn = normalize_tensors(x1, x3)
    x1 = torch.einsum("qnrjk,qrijk->qnir", x1, sl1r)
    x3 = torch.einsum("qnrjk,qrijk->qnir", x3, slr1)
    x = torch.einsum("qnir,qair->qnai", x1 + x3, h)
    return x, xn


def g_merge_h3(y, z, y_term, z_term, sl1r, slr1, slr, h):
    x1 = compose_bigger_span(y_term, z[:, :, :1])
    x2 = compose_bigger_span(y[:, :, :-1], z[:, :, 1:])
    x3 = compose_bigger_span(y[:, :, -1:], z_term)
    (x1, x2, x3), xn = normalize_tensors(x1, x2, x3)
    x1 = torch.einsum("qnrjk,qrijk->qnir", x1, sl1r)
    x2 = torch.einsum("qnrjk,qrijk->qnir", x2, slr)
    x3 = torch.einsum("qnrjk,qrijk->qnir", x3, slr1)
    x = torch.einsum("qnir,qair->qnai", x1 + x2 + x3, h)
    return x, xn


def set_score(x, xn, mask, value):
    x_real = (x + 1e-9).log() + xn[..., None, None]
    x_real = torch.where(mask, value, x_real)
    xn = x_real.flatten(2).max(-1)[0]
    x = (x_real - xn[..., None, None]).exp()
    return x, xn


if __name__ == "__main__":
    from torch_struct import SentCFG

    from .decomp7 import Decomp7

    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 2, 4, 2, 5, 3, 7
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    r = 3
    lens = [max(2, N - i) for i in range(B)]
    params = Decomp7Fast.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    params_ref = convert_decomp7fast_to_pcfg(params, TGT_NT)
    params_ref2 = {key: value.detach().requires_grad_() for key, value in params.items()}
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

    pcfg = Decomp7Fast(params, lens, **meta)
    pcfg_ref = SentCFG((params_ref["term"], params_ref["rule"], params_ref["root"]), lens)
    pcfg_ref2 = Decomp7(params_ref2, lens, **meta)

    print("test nll")
    nll = pcfg.nll
    nll_ref = -pcfg_ref2.partition
    assert torch.allclose(nll, nll_ref), (nll, nll_ref)

    print("test marginal")
    m1 = pcfg.marginal
    check_full_marginal(m1["term"], m1["trace"].sum(-1), lens)

    # m2 = pcfg_ref.marginals[-1]
    # compare_unlabeled_marginal(m1["trace"].sum(-1), m2)

    print("test entropy")
    print(pcfg.entropy)
    print(pcfg_ref2.entropy)

    pcfg.params["term"].grad = None
    pcfg_ref2.params["term"].grad = None
    pcfg.entropy.sum().backward()
    pcfg_ref2.entropy.sum().backward()

    assert torch.allclose(pcfg.params["term"].grad, pcfg_ref2.params["term"].grad, atol=1e-6, rtol=1e-4)
    print("pass")
