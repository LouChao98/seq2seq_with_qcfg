import logging
from collections import Counter
from copy import copy
from functools import partial
from typing import Dict, List

import numpy as np
import torch
from numba import jit
from torch import Tensor

from ._fn import diagonal_copy_, stripe, stripe2
from ._utils import (
    check_full_marginal,
    checkpoint,
    compare_marginal,
    compute_unnormalized_prob,
    enumerate_seq,
    weighted_random_v2,
)
from .base import _COPY_NT, _COPY_PT, _OK, _REACHLIMIT, _SONMASK, _VOCAB, DecompBase, DecompSamplerBase

log = logging.getLogger(__file__)


class Decomp1(DecompBase):
    # A,i->r r->B,j r->C,k

    KEYS = ("term", "head", "left", "right", "root")
    LOGSPACE = (True, True, True, True, True)

    def inside(self, params, semiring, trace=False):
        params = self.preprocess(params, semiring)
        merge = checkpoint(partial(g_merge, semiring=semiring))

        head: Tensor = params["head"]  # (batch, NT, r), A[i] -> R
        term: Tensor = params["term"]  # (batch, seq_len, max_copy_width, PT)
        root: Tensor = params["root"]  # (batch, NT)
        L: Tensor = params["left"]  # (batch, r, NT + T), r->L
        R: Tensor = params["right"]  # (batch, r, NT + T), r->R
        constraint = params.get("constraint")

        bsz = self.batch_size
        N = term.shape[2] + 1
        NT = self.nt_states * self.nt_num_nodes
        rank = L.shape[2]

        TLNT = L[..., :NT]
        TLPT = L[..., NT:]
        TRNT = R[..., :NT]
        TRPT = R[..., NT:]

        if trace:
            _span_indicator = term.new_zeros(bsz, N, N, requires_grad=True)
            span_indicator = _span_indicator.view(1, bsz, N, N)
            span_indicator_running = span_indicator
        else:
            _span_indicator = None
            span_indicator = None
            span_indicator_running = None

        s = semiring.new_zeros((bsz, N, N, rank))

        # cxmn1p,cx11rp -> cxmnr
        left_term = semiring.sum(semiring.mul(term.unsqueeze(4), TLPT[:, :, None, None]), dim=5)
        right_term = semiring.sum(semiring.mul(term.unsqueeze(4), TRPT[:, :, None, None]), dim=5)

        # c, b, r, nt * c, b, nt, r = batch, parentr, childr
        HL = semiring.sum(semiring.mul(TLNT.unsqueeze(4), head.unsqueeze(2)), dim=3)
        HR = semiring.sum(semiring.mul(TRNT.unsqueeze(4), head.unsqueeze(2)), dim=3)
        # c b nt r  * c b nt
        root = semiring.sum(semiring.mul(head, root.unsqueeze(3)), dim=2)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (torch.arange(2, N + 1).unsqueeze(1) <= self.lens.cpu().unsqueeze(0)).sum(1)

        # w: span width
        final = []
        for step, w in enumerate(range(2, N)):

            # n: the number of spans of width w.
            n = N - w
            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            pt_w = min(w - 1, left_term.shape[3])
            y_term = left_term[:, :current_bsz, :n, :pt_w]
            offset = [1, w - 2]
            if offset[1] >= right_term.shape[3]:
                offset[0] += offset[1] - right_term.shape[3] + 1
                offset[1] = right_term.shape[3] - 1
            z_term = stripe2(right_term, current_bsz, n, pt_w, offset)
            y = stripe(s, current_bsz, n, w - 1, (0, 1)).clone()
            z = stripe(s, current_bsz, n, w - 1, (w, 1)).clone()

            x = merge(y, z, y_term, z_term)
            if constraint is not None:
                value, mask = constraint[step]
                if value.ndim > 0:
                    value = value[:current_bsz]
                mask = mask[:current_bsz]
                x = torch.where(mask, value, x)

            if trace:
                indicator = span_indicator_running.diagonal(w, 2, 3).movedim(-1, 2)
                x = x + indicator.unsqueeze(-1)

            if current_bsz - unfinished > 0:
                final.insert(0, x[:, unfinished:current_bsz, :1])

            if unfinished > 0:
                if current_bsz > unfinished:
                    x = x[:, :unfinished]
                    HL = HL[:, :unfinished]
                    HR = HR[:, :unfinished]
                    if trace:
                        span_indicator_running = span_indicator_running[:, :unfinished]
                # cx1nr,cxR1r -> cxRn
                left_x = semiring.sum(semiring.mul(x.unsqueeze(2), HL.unsqueeze(3)), dim=4)
                right_x = semiring.sum(semiring.mul(x.unsqueeze(2), HR.unsqueeze(3)), dim=4)
                s[:, :unfinished].diagonal(w, 2, 3).copy_(left_x)
                s[:, :unfinished].diagonal(-w, 2, 3).copy_(right_x)
            if unfinished == 0:
                break

        final = torch.cat(final, dim=1)
        final = semiring.mul(final.squeeze(2), root)
        logZ = semiring.sum(final, dim=-1)
        logZ = semiring.unconvert(logZ)
        return logZ, _span_indicator

    @staticmethod
    def random(bsz, max_len, tgt_pt, src_pt, tgt_nt, src_nt, r, ngram=1):
        nt = tgt_nt * src_nt
        pt = tgt_pt * src_pt
        term = torch.randn(bsz, max_len, ngram, pt).zero_()
        return {
            "term": term.requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "head": torch.randn(bsz, nt, r).log_softmax(-1).requires_grad_(True),
            "left": torch.randn(bsz, r, nt + pt).log_softmax(-1).requires_grad_(True),
            "right": torch.randn(bsz, r, nt + pt).log_softmax(-1).requires_grad_(True),
        }


class Decomp1Sampler(DecompSamplerBase):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(2).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        H = params["head"].exp().cumsum(2).cpu().numpy()
        L = params["left"].exp().cumsum(2).cpu().numpy()
        R = params["right"].exp().cumsum(2).cpu().numpy()
        return terms, H, L, R, roots

    @staticmethod
    # @jit(nopython=True)
    def sample_impl(
        terms: np.ndarray,  # seqlen x pt, in normal space
        rules_head: np.ndarray,  # nt x r, in normal space
        rules_left: np.ndarray,  # (nt+pt) x r, in normal space
        rules_right: np.ndarray,  # (nt+pt) x r, in normal space
        roots: np.ndarray,  # nt, in normal space
        nt_states: int,
        nt_num_nodes: int,
        pt_states: int,
        pt_num_nodes: int,
        use_copy=True,
        num_samples=1,
        max_length=100,
        max_actions=100,
        unk=1,
    ):
        NT = rules_head.shape[0]
        COPY_NT = nt_states - 1
        COPY_PT = pt_states - 1
        samples = [[0] for _ in range(num_samples)]
        types = [[0] for _ in range(num_samples)]
        status = [_OK for _ in range(num_samples)]

        for i in range(num_samples):
            nonterminals: List[int] = []
            preterminals: List[int] = []
            actions = 0
            try:
                sample = weighted_random_v2(roots)
                nonterminals.append(sample)

                while len(nonterminals) > 0 and len(preterminals) < max_length and actions < max_actions:
                    s = nonterminals.pop()
                    if s < NT:
                        actions += 1
                        head = weighted_random_v2(rules_head[s])
                        left = weighted_random_v2(rules_left[head])
                        right = weighted_random_v2(rules_right[head])
                        nonterminals.extend([right, left])
                    else:
                        preterminals.append(s - NT)
            except Exception:
                status[i] = _SONMASK

            if actions == max_actions or (len(preterminals) == max_length and len(nonterminals) > 0):
                status[i] = _REACHLIMIT

            terminals: List[int] = []
            terminal_type: List[int] = []  # 0=vocab, 1=nt span, 2=pt span
            try:
                for s in preterminals:
                    src_pt_state = s // pt_num_nodes
                    if use_copy and src_pt_state == COPY_PT:
                        src_node = s % pt_num_nodes
                        terminals.append(src_node)
                        terminal_type.append(_COPY_PT)
                    else:
                        sample = weighted_random_v2(terms[s])
                        if use_copy and sample == unk:
                            # force <unk> tokens to copy
                            src_node = s % pt_num_nodes
                            terminals.append(src_node)
                            terminal_type.append(_COPY_PT)
                        else:
                            terminals.append(sample)
                            terminal_type.append(_VOCAB)
            except Exception:
                status[i] = _SONMASK

            samples[i] = terminals
            types[i] = terminal_type
        return samples, types, status


def convert_decomp1copypt_to_pcfg(p, tgt_nt):
    print("Conversion only support 1-gram PT.")
    rule = torch.einsum("xar,xrb,xrc->xabc", p["head"].exp(), p["left"].exp(), p["right"].exp())
    rule = (rule + 1e-9).log()
    output = {"term": p["term"][:, :, 0], "rule": rule, "root": p["root"]}
    if "constraint" in p:
        output["constraint"] = p["constraint"]
    if "add" in p:
        output["add"] = p["add"]
    if "lse" in p:
        output["lse"] = p["lse"]
    return output


def g_merge(y, z, y_term, z_term, semiring):
    # contract dimension w.
    shape = list(y.shape)
    del shape[3]
    if y_term.shape[3] == y.shape[3]:
        output = y.new_empty(shape + [4])  # c, b, n, r, 4
        output[..., 3] = semiring.sum(semiring.mul(y_term, z_term), dim=3)
    else:
        output = y.new_empty(shape + [3])
    output[..., 0] = semiring.sum(semiring.mul(y, z), dim=3)
    output[..., 1] = semiring.sum(semiring.mul(y_term, z[:, :, :, : y_term.shape[3]]), dim=3)
    output[..., 2] = semiring.sum(semiring.mul(y[:, :, :, -z_term.shape[3] :], z_term), dim=3)

    return semiring.sum(output, dim=4)


if __name__ == "__main__":
    from .decomp1 import Decomp1 as Decomp1Ref

    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 1, 4, 2, 5, 3, 7
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    r = 2
    ngram = 1
    lens = [max(2, N - i) for i in range(B)]
    params = Decomp1.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r, ngram=ngram)
    # params_ref = convert_decomp1copypt_to_pcfg(params, TGT_NT)
    params_ref = {k: v[:, :, 0] if k == "term" else v for k, v in params.items()}
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }

    pcfg = Decomp1(params, lens, **meta)
    pcfg_ref = Decomp1Ref(params_ref, lens, **meta)

    print("test nll")
    nll = pcfg.nll
    nll_ref = pcfg_ref.nll
    assert torch.allclose(nll, nll_ref), (nll, nll_ref)

    marginals = pcfg.marginal

    length = torch.arange(1, ngram + 1)[None, None, :, None]
    print((length * marginals["term"]).sum())
    print(marginals["head"].flatten(1).sum(1))
    print(nll)
    # breakpoint()
