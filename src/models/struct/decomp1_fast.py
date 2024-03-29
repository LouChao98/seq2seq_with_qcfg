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
from .decomp1 import Decomp1, Decomp1NATSampler, Decomp1Sampler

log = logging.getLogger(__file__)


class Decomp1Fast(Decomp1):
    KEYS = ("term", "head", "left", "right", "root")
    LOGSPACE = (True, False, False, False, True)

    def __init__(self, *args, impl2=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace_rank = False
        self.impl2 = impl2

    @contextmanager
    def trace_rank(self):
        self._trace_rank = True
        # reset lazy property
        if hasattr(self, "marginal"):
            delattr(self, "marginal")
        yield
        self._trace_rank = False

    def inside(self, params, semiring, trace=False, use_reentrant=True) -> Tuple[Tensor, Tensor]:
        if semiring == LogSemiring:
            if not self.impl2:
                return self.inside_log_semiring(params, trace, use_reentrant)
            else:
                return self.inside_log_semiring2(params, trace, use_reentrant)
        else:
            params = {**params}
            params["head"] = params["head"].clamp(1e-9).log()
            params["left"] = params["left"].clamp(1e-9).log()
            params["right"] = params["right"].clamp(1e-9).log()
            return super().inside(params, semiring, trace)

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

        sn = term.new_full((batch, N, N), -1e9)
        norm: Tensor = term.max(-1)[0]
        sn.diagonal(1, 1, 2).copy_(norm)
        term = (term - norm.unsqueeze(-1)).exp()

        s = term.new_zeros(batch, N, N, rank)

        H = H.transpose(-1, -2)
        # (m_A, r_A) + (m_B, r_B) -> (r_A, r_B)
        H_L = torch.matmul(H, TLNT)
        H_R = torch.matmul(H, TRNT)

        # b n pt x b pt r -> b n r
        left_term = torch.matmul(term, TLPT)
        right_term = torch.matmul(term, TRPT)

        if trace:
            if self._trace_rank:
                _span_indicator = term.new_ones(batch, N, N, rank, requires_grad=True)
                span_indicator = _span_indicator
                term_indicator = torch.ones_like(left_term, requires_grad=True)
                left_term = left_term * term_indicator
                right_term = right_term * term_indicator
            else:
                _span_indicator = term.new_ones(batch, N, N, requires_grad=True)
                span_indicator = _span_indicator.unsqueeze(-1)
            span_indicator_running = span_indicator
        else:
            _span_indicator = None

        s.diagonal(1, 1, 2).copy_(left_term.transpose(1, 2))
        s.diagonal(-1, 1, 2).copy_(right_term.transpose(1, 2))

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

        if trace and self._trace_rank:
            return logZ, (_span_indicator, term_indicator)
        else:
            return logZ, _span_indicator

    def inside_log_semiring2(self, params, trace=False, use_reentrant=True):
        # slight faster, but use more memory at peak

        merge = checkpoint(g_merge2, use_reentrant=use_reentrant)
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

        norm: Tensor = term.max(-1, keepdim=True)[0]
        term = (term - norm).exp()

        H = H.transpose(-1, -2)
        # (m_A, r_A) + (m_B, r_B) -> (r_A, r_B)
        H_L = torch.matmul(H, TLNT)
        H_R = torch.matmul(H, TRNT)

        # b n pt x b pt r -> b n r
        left_term = (torch.matmul(term, TLPT) + 1e-9).log() + norm
        right_term = (torch.matmul(term, TRPT) + 1e-9).log() + norm

        s = term.new_full((batch, N, N, rank), -1e9)
        s.diagonal(1, 1, 2).copy_(left_term.transpose(1, 2))
        s.diagonal(-1, 1, 2).copy_(right_term.transpose(1, 2))

        if trace:
            if self._trace_rank:
                _span_indicator = term.new_ones(batch, N, N, rank, requires_grad=True)
                span_indicator = _span_indicator
                term_indicator = torch.ones_like(left_term, requires_grad=True)
                left_term = left_term * term_indicator
                right_term = right_term * term_indicator
            else:
                _span_indicator = term.new_ones(batch, N, N, requires_grad=True)
                span_indicator = _span_indicator.unsqueeze(-1)
            span_indicator_running = span_indicator
        else:
            _span_indicator = None

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

            if y.requires_grad:
                y = y.clone()  # due to checkpoint
                z = z.clone()

            x, xn = merge(y, z)

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

                left_x = (torch.matmul(x, H_L) + 1e-9).log() + xn
                right_x = (torch.matmul(x, H_R) + 1e-9).log() + xn

                s.diagonal(w, 1, 2).copy_(left_x.transpose(1, 2))
                s.diagonal(-w, 1, 2).copy_(right_x.transpose(1, 2))

            if unfinished == 0:
                break

        final = torch.cat(final, dim=0)
        final_normalizer = torch.cat(final_normalizer, dim=0)
        final = torch.matmul(final, H)
        final = (final + 1e-9).squeeze(1).log() + root
        logZ = final.logsumexp(-1) + final_normalizer.view(-1)

        if trace and self._trace_rank:
            return logZ, (_span_indicator, term_indicator)
        else:
            return logZ, _span_indicator

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


class Decomp1FastSampler(Decomp1Sampler):
    # not faster than Decomp1Sampler. just match the name
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(2).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        H = params["head"].cumsum(2).cpu().numpy()
        L = params["left"].cumsum(2).cpu().numpy()
        R = params["right"].cumsum(2).cpu().numpy()
        return terms, H, L, R, roots


class Decomp1FastNATSampler(Decomp1NATSampler):
    # not faster than Decomp1Sampler. just match the name
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(3).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        H = params["head"].cumsum(2).cpu().numpy()
        L = params["left"].cumsum(2).cpu().numpy()
        R = params["right"].cumsum(2).cpu().numpy()
        return terms, H, L, R, roots


# @torch.jit.script
def g_merge(y, z, yn, zn):
    # contract dimension w.
    y = (y + 1e-9).log() + yn.unsqueeze(-1)
    z = (z + 1e-9).log() + zn.unsqueeze(-1)
    b_n_r = (y + z).logsumexp(-2)
    normalizer = b_n_r.max(-1)[0]
    b_n_r = (b_n_r - normalizer.unsqueeze(-1)).exp()
    return b_n_r, normalizer


def g_merge2(y, z):
    b_n_r = (y + z).logsumexp(-2)
    normalizer = b_n_r.max(-1, keepdim=True)[0]
    b_n_r = (b_n_r - normalizer).exp()
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

    # m2 = pcfg_ref.marginals[-1]
    # compare_unlabeled_marginal(m1["trace"].sum(-1), m2)

    print("test entropy")
    print(pcfg.entropy)
    print(pcfg_ref2.entropy)

    pcfg.entropy.sum().backward()
    pcfg_ref2.entropy.sum().backward()

    assert torch.allclose(pcfg.params["term"].grad, pcfg_ref2.params["term"].grad, atol=1e-6, rtol=1e-4)
    print("pass")
