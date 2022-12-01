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


class Decomp5(DecompBase):
    # A[i] -> B[j], C[k]
    # ================
    # A[i] -> R
    # R -> B
    # R -> C
    # A, i -> j, k
    # ================
    # Time complexity: 7
    # Flex

    KEYS = ("term", "head", "left", "right", "slr", "root")
    LOGSPACE = (True, False, False, False, False, True)

    def __init__(self, *args, direction, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction = direction

    def inside(self, trace) -> Tuple[Tensor, Tensor]:
        if trace:
            params = {k: process_param_for_trace(v) for k, v in self.params.items()}
        else:
            params = self.params

        head = params["head"]  # (batch, NT, r), A[i] -> R
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

        # (batch, r, TGT_NT), R -> B/C
        # (batch, r, TGT_PT), R -> B/C
        size = [self.nt_states, self.pt_states]
        TLNT, TLPT = torch.split(params["left"], size, -1)
        TRNT, TRPT = torch.split(params["right"], size, -1)

        SLR = params["slr"]

        # NOTE SL1R1, SL1R, SLR1, SLR should be normalized for each. p(B)P(j|B)
        #   or TL(TR)NT, TL(TR)PT should be normalized for each. p(j)p(B|j)

        SL1R1 = SLR[:, :, :, nt_spans:, nt_spans:]
        SL1R = SLR[:, :, :, nt_spans:, :nt_spans]
        SLR1 = SLR[:, :, :, :nt_spans, nt_spans:]
        SLR = SLR[:, :, :, :nt_spans, :nt_spans]

        # ===== End =====

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
        left_term = torch.einsum("xlpi,xrp->xlir", term, TLPT)
        right_term = torch.einsum("xlpi,xrp->xlir", term, TRPT)
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

                left_x = torch.einsum("qnai,qra->qnir", x, TLNT)
                right_x = torch.einsum("qnai,qra->qnir", x, TRNT)
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
    def random_dir0(bsz, max_len, tgt_pt, src_pt, tgt_nt, src_nt, r):
        nt = tgt_nt * src_nt
        pt = tgt_pt * src_pt
        slr = torch.rand(bsz, tgt_nt, src_nt, src_nt + src_pt, src_nt + src_pt)
        slr[..., :src_nt, :src_nt] /= slr[..., :src_nt, :src_nt].sum((3, 4), keepdim=True)
        slr[..., src_nt:, :src_nt] /= slr[..., src_nt:, :src_nt].sum((3, 4), keepdim=True)
        slr[..., :src_nt, src_nt:] /= slr[..., :src_nt, src_nt:].sum((3, 4), keepdim=True)
        slr[..., src_nt:, src_nt:] /= slr[..., src_nt:, src_nt:].sum((3, 4), keepdim=True)
        return {
            "term": torch.randn(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "head": torch.randn(bsz, nt, r).softmax(-1).requires_grad_(True),
            "left": torch.randn(bsz, r, tgt_nt + tgt_pt).softmax(-1).requires_grad_(True),
            "right": torch.randn(bsz, r, tgt_nt + tgt_pt).softmax(-1).requires_grad_(True),
            "slr": slr,
        }

    @staticmethod
    def random_dir1(bsz, max_len, tgt_pt, src_pt, tgt_nt, src_nt, r):
        nt = tgt_nt * src_nt
        pt = tgt_pt * src_pt
        slr = torch.rand(bsz, tgt_nt, src_nt, src_nt + src_pt, src_nt + src_pt)
        slr /= slr.sum((3, 4), keepdim=True)

        left = torch.rand(bsz, r, tgt_nt + tgt_pt)
        left[..., :tgt_nt] /= left[..., :tgt_nt].sum(-1, keepdim=True)
        left[..., tgt_nt:] /= left[..., tgt_nt:].sum(-1, keepdim=True)

        right = torch.rand(bsz, r, tgt_nt + tgt_pt)
        right[..., :tgt_nt] /= right[..., :tgt_nt].sum(-1, keepdim=True)
        right[..., tgt_nt:] /= right[..., tgt_nt:].sum(-1, keepdim=True)
        return {
            "term": torch.randn(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "head": torch.randn(bsz, nt, r).softmax(-1).requires_grad_(True),
            "left": left.requires_grad_(True),
            "right": right.requires_grad_(True),
            "slr": slr.requires_grad_(True),
        }


class Decomp5Sampler:
    def __new__(cls, *args, direction, **kwargs):
        if direction == 0:
            return Decomp5Dir0Sampler(*args, **kwargs)
        elif direction == 1:
            return Decomp5Dir1Sampler(*args, **kwargs)
        raise ValueError


class Decomp5Dir0Sampler(DecompSamplerBase):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(2).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        H = params["head"].cumsum(2).cpu().numpy()
        L = params["left"].cumsum(2).cpu().numpy()
        R = params["right"].cumsum(2).cpu().numpy()

        SLR = params["slr"]
        n = SLR.shape[2]
        SL1R1 = SLR[..., n:, n:].flatten(3).cumsum(3).cpu().numpy()
        SL1R = SLR[..., n:, :n].flatten(3).cumsum(3).cpu().numpy()
        SLR1 = SLR[..., :n, n:].flatten(3).cumsum(3).cpu().numpy()
        SLR = SLR[..., :n, :n].flatten(3).cumsum(3).cpu().numpy()
        assert n == self.nt_num_nodes
        return terms, H, L, R, SL1R1, SL1R, SLR1, SLR, roots

    @staticmethod
    @jit(nopython=True)
    def sample_impl(
        terms: np.ndarray,  # seqlen x pt, in normal space
        rules_head: np.ndarray,  # nt x r, in normal space
        rules_left: np.ndarray,  # (nt+pt) x r, in normal space
        rules_right: np.ndarray,  # (nt+pt) x r, in normal space
        rules_sl1r1: np.ndarray,
        rules_sl1r: np.ndarray,
        rules_slr1: np.ndarray,
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
                    left = weighted_random_v2(rules_left[r])
                    right = weighted_random_v2(rules_right[r])

                    if left < nt_states and right < nt_states:
                        jk = weighted_random_v2(rules_slr[s, i])
                        j, k = divmod(jk, nt_num_nodes)
                    elif left < nt_states and right >= nt_states:
                        jk = weighted_random_v2(rules_slr1[s, i])
                        j, k = divmod(jk, pt_num_nodes)
                    elif left >= nt_states and right < nt_states:
                        jk = weighted_random_v2(rules_sl1r[s, i])
                        j, k = divmod(jk, nt_num_nodes)
                    elif left >= nt_states and right >= nt_states:
                        jk = weighted_random_v2(rules_sl1r1[s, i])
                        j, k = divmod(jk, pt_num_nodes)

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


class Decomp5Dir1Sampler(DecompSamplerBase):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(2).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        H = params["head"].cumsum(2).cpu().numpy()
        SLR = params["slr"].flatten(3).cumsum(3).cpu().numpy()

        L = params["left"]  # (batch, r, TGT_NT + TGT_PT)
        R = params["right"]  # (batch, r, TGT_NT + TGT_PT)
        LNT = L[..., : self.nt_states].cumsum(2).cpu().numpy()
        LPT = L[..., self.nt_states :].cumsum(2).cpu().numpy()
        RNT = R[..., : self.nt_states].cumsum(2).cpu().numpy()
        RPT = R[..., self.nt_states :].cumsum(2).cpu().numpy()
        return terms, H, LNT, LPT, RNT, RPT, SLR, roots

    @staticmethod
    @jit(nopython=True)
    def sample_impl(
        terms: np.ndarray,  # seqlen x pt, in normal space
        rules_head: np.ndarray,  # nt x r, in normal space
        rules_left_nt: np.ndarray,  # (nt+pt) x r, in normal space
        rules_left_pt: np.ndarray,  # (nt+pt) x r, in normal space
        rules_right_nt: np.ndarray,  # (nt+pt) x r, in normal space
        rules_right_pt: np.ndarray,  # (nt+pt) x r, in normal space
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
        num_nodes = nt_num_nodes + pt_num_nodes

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
                    jk = weighted_random_v2(rules_slr[s, i])
                    j, k = divmod(jk, num_nodes)

                    if j < nt_num_nodes:
                        left = weighted_random_v2(rules_left_nt[r])
                    else:
                        left = weighted_random_v2(rules_left_pt[r]) + nt_states
                        j -= nt_num_nodes
                    if k < nt_num_nodes:
                        right = weighted_random_v2(rules_right_nt[r])
                    else:
                        right = weighted_random_v2(rules_right_pt[r]) + nt_states
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


def convert_decomp5_to_pcfg(params, nt_states):
    B, _, r = params["head"].shape
    TGT_NT = nt_states
    TGT_PT = params["left"].shape[-1] - TGT_NT
    head = params["head"].view(B, nt_states, -1, r)
    SRC_NT = head.shape[2]
    SRC_PT = params["slr"].shape[-1] - SRC_NT
    rule11 = torch.einsum(
        "xair,xrb,xrc,xaijk->xaibjck",
        head,
        params["left"][:, :, :TGT_NT],
        params["right"][:, :, :TGT_NT],
        params["slr"][:, :, :, :SRC_NT, :SRC_NT],
    )
    rule12 = torch.einsum(
        "xair,xrb,xrc,xaijk->xaibjck",
        head,
        params["left"][:, :, :TGT_NT],
        params["right"][:, :, TGT_NT:],
        params["slr"][:, :, :, :SRC_NT, SRC_NT:],
    )
    rule21 = torch.einsum(
        "xair,xrb,xrc,xaijk->xaibjck",
        head,
        params["left"][:, :, TGT_NT:],
        params["right"][:, :, :TGT_NT],
        params["slr"][:, :, :, SRC_NT:, :SRC_NT],
    )
    rule22 = torch.einsum(
        "xair,xrb,xrc,xaijk->xaibjck",
        head,
        params["left"][:, :, TGT_NT:],
        params["right"][:, :, TGT_NT:],
        params["slr"][:, :, :, SRC_NT:, SRC_NT:],
    )

    rule = rule11.new_zeros(
        B,
        TGT_NT * SRC_NT,
        (TGT_NT * SRC_NT) + (TGT_PT * SRC_PT),
        (TGT_NT * SRC_NT) + (TGT_PT * SRC_PT),
    )
    shape = rule11.shape
    rule[:, :, : TGT_NT * SRC_NT, : TGT_NT * SRC_NT] = (
        rule11.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]) + 1e-9
    ).log()
    shape = rule12.shape
    rule[:, :, : TGT_NT * SRC_NT, TGT_NT * SRC_NT :] = (
        rule12.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]) + 1e-9
    ).log()
    shape = rule21.shape
    rule[:, :, TGT_NT * SRC_NT :, : TGT_NT * SRC_NT] = (
        rule21.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]) + 1e-9
    ).log()
    shape = rule22.shape
    rule[:, :, TGT_NT * SRC_NT :, TGT_NT * SRC_NT :] = (
        rule22.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]) + 1e-9
    ).log()
    output = {"term": params["term"], "rule": rule, "root": params["root"]}
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
    x = torch.einsum("qnkrj,qaijk,qair->qnai", qnkrj, slr, h)
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
    x1 = torch.einsum("qnkrj,qaijk->qnrai", qnkrj1, sl1r)
    x3 = torch.einsum("qnkrj,qaijk->qnrai", qnkrj3, slr1)
    x = torch.einsum("qnrai,qair->qnai", x1 + x3, h)
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
    x1 = torch.einsum("qnkrj,qaijk->qnrai", qnkrj1, sl1r)
    x2 = torch.einsum("qnkrj,qaijk->qnrai", qnkrj2, slr)
    x3 = torch.einsum("qnkrj,qaijk->qnrai", qnkrj3, slr1)
    x = torch.einsum("qnrai,qair->qnai", x1 + x2 + x3, h)
    return x, normalizer


@torch.jit.script
def set_score(x, xn, mask, value):
    x_real = (x + 1e-9).log() + xn[..., None, None]
    x_real = torch.where(mask, value, x_real)
    xn = x_real.flatten(2).max(-1)[0]
    x = (x_real - xn[..., None, None]).exp()
    return x, xn
