import logging
from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np
import torch
from numba import jit
from torch import Tensor

from ._fn import diagonal_copy_, stripe
from ._utils import checkpoint, process_param_for_trace, weighted_random_v2
from .decomp_base import DecompBase, DecompSamplerBase

log = logging.getLogger(__file__)

_VOCAB, _COPY_NT, _COPY_PT = 0, 1, 2
_OK, _SONMASK, _REACHLIMIT = 0, 1, 2


class TokenType(IntEnum):
    VOCAB = _VOCAB
    COPY_NT = _COPY_NT
    COPY_PT = _COPY_PT


class Decomp7(DecompBase):
    # A[i] -> B[j], C[k]
    # ================
    # A[i] -> R
    # R, j -> B      Folded
    # R, k -> C      Folded
    # R, i -> j, k
    # ================
    # Time complexity: 6
    # Flex

    KEYS = ("term", "head", "left", "right", "slr", "root")
    LOGSPACE = (True, False, False, False, False, True)

    def inside(self, trace) -> Tuple[Tensor, Tensor]:
        if trace:
            params = {k: process_param_for_trace(v) for k, v in self.params.items()}
        else:
            params = self.params

        head = params["head"]  # (batch, nt, r), A[i] -> R
        term = params["term"]  # (batch, seq_len, PT)
        root = params["root"]  # (batch, NT)
        constraint = params.get("constraint")

        batch, N, PT = term.shape
        _, NT, R = head.shape
        N += 1
        nt_spans = NT // self.nt_states
        pt_spans = PT // self.pt_states
        max_spans = max(nt_spans, pt_spans)

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
            span_indicator = term.new_ones(batch, N, N, self.max_states, max_spans, requires_grad=True)
            span_indicator_running = span_indicator[:, :, :, : self.nt_states, :nt_spans]
        else:
            span_indicator = None
            span_indicator_running = None

        normalizer = term.new_full((batch, N, N), -1e9)
        norm = term.flatten(2).max(-1)[0]
        diagonal_copy_(normalizer, norm, w=1)
        term = (term - norm[..., None, None]).exp()

        left_s = term.new_full((batch, N, N, max_spans, R), 0.0)
        right_s = term.new_full((batch, N, N, max_spans, R), 0.0)
        if trace:
            indicator = span_indicator.diagonal(1, 1, 2).movedim(-1, 1)
            term = term * indicator[..., : self.pt_states, :pt_spans]
        left_term = torch.einsum("xlpi,xrip->xlir", term, TLPT)
        right_term = torch.einsum("xlpi,xrip->xlir", term, TRPT)
        diagonal_copy_(left_s, left_term, w=1, s3=pt_spans)
        diagonal_copy_(right_s, right_term, w=1, s3=pt_spans)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (torch.arange(2, N + 1).unsqueeze(1) <= self.lens.cpu().unsqueeze(0)).sum(1)

        # w: span width
        final = []
        final_normalizer = []
        for step, w in enumerate(range(2, N)):

            # n: the number of spans of width w.
            n = N - w
            y = stripe(left_s, n, w - 1, (0, 1)).clone()
            z = stripe(right_s, n, w - 1, (1, w), 0).clone()
            yn = stripe(normalizer, n, w - 1, (0, 1))
            zn = stripe(normalizer, n, w - 1, (1, w), 0)

            if w == 2:
                x, xn = merge_h(y, z, yn, zn, SL1R1, head)
            elif w == 3:
                x, xn = merge_h2(y, z, yn, zn, SL1R, SLR1, head)
            else:
                x, xn = merge_h3(y, z, yn, zn, SL1R, SLR1, SLR, head)

            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

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
                    left_s = left_s[:unfinished]
                    right_s = right_s[:unfinished]
                    SLR = SLR[:unfinished]
                    SL1R = SL1R[:unfinished]
                    SLR1 = SLR1[:unfinished]
                    head = head[:unfinished]
                    TLNT = TLNT[:unfinished]
                    TRNT = TRNT[:unfinished]
                    normalizer = normalizer[:unfinished]
                    xn = xn[:unfinished]
                    if trace:
                        span_indicator_running = span_indicator_running[:unfinished]

                left_x = torch.einsum("qnai,qria->qnir", x, TLNT)
                right_x = torch.einsum("qnai,qria->qnir", x, TRNT)
                diagonal_copy_(left_s, left_x, w, s3=nt_spans)
                diagonal_copy_(right_s, right_x, w, s3=nt_spans)
                diagonal_copy_(normalizer, xn, w)

            if unfinished == 0:
                break

        final = torch.cat(final, dim=0)
        final_normalizer = torch.cat(final_normalizer, dim=0)
        final = (final + 1e-9).squeeze(1).log() + root
        logZ = final.logsumexp((-2, -1)) + final_normalizer.squeeze(-1)
        return logZ, span_indicator

    @staticmethod
    def random(bsz, max_len, tgt_pt, src_pt, tgt_nt, src_nt, r):
        nt = tgt_nt * src_nt
        pt = tgt_pt * src_pt
        src = src_nt + src_pt
        slr = (
            torch.randn(bsz, r, src_nt, src_nt + src_pt, src_nt + src_pt)
            .flatten(3)
            .softmax(-1)
            .view(bsz, r, src_nt, src_nt + src_pt, src_nt + src_pt)
            .requires_grad_(True)
        )
        return {
            "term": torch.randn(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "head": torch.randn(bsz, nt, r).softmax(-1).requires_grad_(True),
            "left": torch.randn(bsz, r, src, max(tgt_nt, tgt_pt)).softmax(-1).requires_grad_(True),
            "right": torch.randn(bsz, r, src, max(tgt_nt, tgt_pt)).softmax(-1).requires_grad_(True),
            "slr": slr,
        }


class Decomp7Sampler(DecompSamplerBase):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(2).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        H = params["head"].cumsum(2).cpu().numpy()
        L = params["left"].cumsum(3).cpu().numpy()
        R = params["right"].cumsum(3).cpu().numpy()

        SLR = params["slr"].flatten(3).cumsum(3).cpu().numpy()
        return terms, H, L, R, SLR, roots

    @staticmethod
    @jit(nopython=True)
    def sample_impl(
        terms: np.ndarray,  # seqlen x pt, in normal space
        rules_head: np.ndarray,  # nt x r, in normal space
        rules_left: np.ndarray,  # (nt+pt) x r, in normal space
        rules_right: np.ndarray,  # (nt+pt) x r, in normal space
        rules_slr: np.ndarray,
        roots: np.ndarray,
        nt_num_nodes: int,
        nt_states: int,
        pt_num_nodes: int,
        pt_states: int,
        use_copy=True,
        num_samples=1,
        max_length=100,
        max_actions=100,
        unk=1,
    ):
        COPY_NT = nt_states - 1
        COPY_PT = pt_states - 1
        samples = [[0] for _ in range(num_samples)]
        types = [[0] for _ in range(num_samples)]
        status = [_OK for _ in range(num_samples)]

        for sidx in range(num_samples):
            nonterminals: List[Tuple[int, int]] = []
            preterminals: List[Tuple[int, int, bool]] = []
            actions = 0
            try:
                s = weighted_random_v2(roots)
                state, i = divmod(s, nt_num_nodes)
                nonterminals.append((state, i))
                while len(nonterminals) > 0 and len(preterminals) < max_length and actions < max_actions:
                    actions += 1
                    s, i = nonterminals.pop()

                    if s >= nt_states:
                        preterminals.append((s - nt_states, i, False))
                        continue
                    if use_copy and s == COPY_NT:
                        preterminals.append((s, i, True))
                        continue

                    r = weighted_random_v2(rules_head[s * nt_num_nodes + i])
                    jk = weighted_random_v2(rules_slr[r, i])
                    j, k = divmod(jk, nt_num_nodes + pt_num_nodes)

                    left = weighted_random_v2(rules_left[r, j])
                    right = weighted_random_v2(rules_right[r, k])

                    if j >= nt_num_nodes:
                        left += nt_states
                        j -= nt_num_nodes
                    if k >= nt_num_nodes:
                        right += nt_states
                        k -= nt_num_nodes

                    nonterminals.extend([(right, k), (left, j)])

            except Exception:
                status[sidx] = _SONMASK

            if actions == max_actions or (len(preterminals) == max_length and len(nonterminals) > 0):
                status[sidx] = _REACHLIMIT

            terminals: List[int] = []
            terminal_type: List[int] = []
            try:
                for s, i, flag in preterminals:
                    if flag:
                        terminals.append(i)
                        terminal_type.append(_COPY_NT)
                        continue
                    if use_copy and s == COPY_PT:
                        terminals.append(i)
                        terminal_type.append(_COPY_PT)
                    else:
                        sample = weighted_random_v2(terms[s * pt_num_nodes + i])
                        if use_copy and sample == unk:
                            # force <unk> tokens to copy
                            terminals.append(i)
                            terminal_type.append(_COPY_PT)
                        else:
                            terminals.append(sample)
                            terminal_type.append(_VOCAB)
            except Exception:
                status[sidx] = _SONMASK

            samples[sidx] = terminals
            types[sidx] = terminal_type
        return samples, types, status


def convert_decomp7_to_pcfg(params, nt_states):
    slr = params["slr"]
    term = params["term"]
    bsz, r, src_nt, src, _ = slr.shape
    src_pt = src - src_nt
    pt_states = term.shape[2] // src_pt
    head = params["head"].view(bsz, nt_states, src_nt, r)
    rule11 = torch.einsum(
        "xair,xrjb,xrkc,xrijk->xaibjck",
        head,
        params["left"][:, :, :src_nt, :nt_states],
        params["right"][:, :, :src_nt, :nt_states],
        params["slr"][:, :, :, :src_nt, :src_nt],
    )
    rule12 = torch.einsum(
        "xair,xrjb,xrkc,xrijk->xaibjck",
        head,
        params["left"][:, :, :src_nt, :nt_states],
        params["right"][:, :, src_nt:, :pt_states],
        params["slr"][:, :, :, :src_nt, src_nt:],
    )
    rule21 = torch.einsum(
        "xair,xrjb,xrkc,xrijk->xaibjck",
        head,
        params["left"][:, :, src_nt:, :pt_states],
        params["right"][:, :, :src_nt, :nt_states],
        params["slr"][:, :, :, src_nt:, :src_nt],
    )
    rule22 = torch.einsum(
        "xair,xrjb,xrkc,xrijk->xaibjck",
        head,
        params["left"][:, :, src_nt:, :pt_states],
        params["right"][:, :, src_nt:, :pt_states],
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


@torch.jit.script
def eq_qnkrj(v1, v2):
    # "qnwjr,qnwkr->qnkrj"
    v = v1.transpose(-1, -2).unsqueeze(-3) + v2.unsqueeze(-1)
    return torch.logsumexp(v, dim=2)


@checkpoint
@torch.jit.script
def merge_h(y, z, y_normalizer, z_normalizer, slr, h):
    num = slr.shape[3]
    y = (y[:, :, :, :num] + 1e-9).log() + y_normalizer[..., None, None]
    z = (z[:, :, :, :num] + 1e-9).log() + z_normalizer[..., None, None]
    qnkrj = eq_qnkrj(y, z)
    normalizer = qnkrj.flatten(2).max(-1)[0]
    qnkrj = (qnkrj - normalizer[..., None, None, None]).exp()
    x = torch.einsum("qnkrj,qrijk,qair->qnai", qnkrj, slr, h)
    return x, normalizer


@checkpoint
@torch.jit.script
def merge_h2(y, z, y_normalizer, z_normalizer, sl1r, slr1, h):
    num_pt, num_nt = sl1r.shape[-2:]
    y = (y + 1e-9).log() + y_normalizer[..., None, None]
    z = (z + 1e-9).log() + z_normalizer[..., None, None]
    qnkrj1 = eq_qnkrj(y[:, :, :1, :num_pt], z[:, :, :1, :num_nt])
    qnkrj3 = eq_qnkrj(y[:, :, -1:, :num_nt], z[:, :, -1:, :num_pt])
    normalizer = torch.stack([qnkrj1.flatten(2).max(-1)[0], qnkrj3.flatten(2).max(-1)[0],], dim=-1,).max(
        -1
    )[0]
    qnkrj1 = (qnkrj1 - normalizer[..., None, None, None]).exp()
    qnkrj3 = (qnkrj3 - normalizer[..., None, None, None]).exp()
    x1 = torch.einsum("qnkrj,qrijk->qnri", qnkrj1, sl1r)
    x3 = torch.einsum("qnkrj,qrijk->qnri", qnkrj3, slr1)
    x = torch.einsum("qnri,qair->qnai", x1 + x3, h)
    return x, normalizer


@checkpoint
@torch.jit.script
def merge_h3(y, z, y_normalizer, z_normalizer, sl1r, slr1, slr, h):
    num_pt, num_nt = sl1r.shape[-2:]
    y = (y + 1e-9).log() + y_normalizer[..., None, None]
    z = (z + 1e-9).log() + z_normalizer[..., None, None]
    qnkrj1 = eq_qnkrj(y[:, :, :1, :num_pt], z[:, :, :1, :num_nt])
    qnkrj2 = eq_qnkrj(y[:, :, 1:-1, :num_nt], z[:, :, 1:-1, :num_nt])
    qnkrj3 = eq_qnkrj(y[:, :, -1:, :num_nt], z[:, :, -1:, :num_pt])
    normalizer = torch.stack(
        [
            qnkrj1.flatten(2).max(-1)[0],
            qnkrj2.flatten(2).max(-1)[0],
            qnkrj3.flatten(2).max(-1)[0],
        ],
        dim=-1,
    ).max(-1)[0]
    qnkrj1 = (qnkrj1 - normalizer[..., None, None, None]).exp()
    qnkrj2 = (qnkrj2 - normalizer[..., None, None, None]).exp()
    qnkrj3 = (qnkrj3 - normalizer[..., None, None, None]).exp()
    x1 = torch.einsum("qnkrj,qrijk->qnri", qnkrj1, sl1r)
    x2 = torch.einsum("qnkrj,qrijk->qnri", qnkrj2, slr)
    x3 = torch.einsum("qnkrj,qrijk->qnri", qnkrj3, slr1)
    x = torch.einsum("qnri,qair->qnai", x1 + x2 + x3, h)
    return x, normalizer


@torch.jit.script
def set_score(x, xn, mask, value):
    x_real = (x + 1e-9).log() + xn[..., None, None]
    x_real = torch.where(mask, value, x_real)
    xn = x_real.flatten(2).max(-1)[0]
    x = (x_real - xn[..., None, None]).exp()
    return x, xn
