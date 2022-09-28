import logging
from enum import IntEnum
from typing import Dict, List

import numpy as np
import torch
from numba import jit
from torch import Tensor
from torch.autograd import grad

from ._fn import diagonal_copy_, stripe
from ._utils import (
    checkpoint,
    process_param_for_marginal,
    weighted_random,
    weighted_random_v2,
)
from .td_style_base import TDStyleBase

# import torch_semiring_einsum as tse


log = logging.getLogger(__file__)

_VOCAB, _COPY_NT, _COPY_PT = 0, 1, 2
_OK, _SONMASK, _REACHLIMIT = 0, 1, 2


class TokenType(IntEnum):
    VOCAB = _VOCAB
    COPY_NT = _COPY_NT
    COPY_PT = _COPY_PT


class D1PCFG(TDStyleBase):
    # A[i] -> B[j], C[k]
    # ================
    # A[i] -> R
    # R -> B
    # R -> C
    # R, i -> j, k
    # ================
    # Time complexity: 6

    # This impl assume tgt_nt_states = tgt_pt_states
    # This should be faster if PT and NT has no seperation in alignment
    def __init__(self, tgt_nt_states, tgt_pt_states) -> None:
        self.tgt_nt_states = tgt_nt_states
        self.tgt_pt_states = tgt_pt_states
        self.max_states = max(tgt_nt_states, tgt_pt_states)
        self.threshold = torch.nn.Threshold(1e-8, 0, True)

        # self.eq_slr = tse.compile_equation("qrij, qrik->qrijk")
        # self.eq_qnkrj = tse.compile_equation("qnwjr,qnwkr->qnkrj")
        # self.eq_qnri = tse.compile_equation("qnkrj,qrijk->qnri")
        # self.eq_qnai = tse.compile_equation("qnri,qair->qnai")
        # self.eq_tor = tse.compile_equation("xlpi,xrp->xlir")

    def __call__(self, params: Dict[str, Tensor], lens, decode=False, marginal=False):
        # if not decode and not marginal and params.get("copy_nt") is None:
        #     return self.logZ(params, lens)
        if decode:
            marginal = True  # MBR decoding
        if marginal:
            grad_state = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            cm = torch.inference_mode(False)
            cm.__enter__()
            params = {k: process_param_for_marginal(v) for k, v in params.items()}
        if not isinstance(lens, torch.Tensor):
            lens = torch.tensor(lens)
        assert (lens[1:] <= lens[:-1]).all(), "Expect lengths in descending."

        head = params["head"]  # (batch, NT, r), A[i] -> R
        term = params["term"]  # (batch, seq_len, PT)
        root = params["root"]  # (batch, NT)
        constraint = params.get("constraint")

        # ===== This routine is optimized for src_len < tgt_len =====

        if "slr" in params:
            SLR = params["slr"]
        else:
            SL = params["sl"]  # (batch, r, SRC_NT, SRC_NT), R, i -> j
            SR = params["sr"]  # (batch, r, SRC_NT, SRC_NT), R, i -> k
            SLR = SL.unsqueeze(-1) + SR.unsqueeze(-2)

        # ===== End =====

        batch, N, PT = term.shape
        _, NT, R = head.shape
        N += 1
        nt_spans = NT // self.tgt_nt_states
        pt_spans = PT // self.tgt_pt_states
        max_spans = max(nt_spans, pt_spans)

        head = head.view(batch, self.tgt_nt_states, nt_spans, R)
        term = term.view(batch, -1, self.tgt_pt_states, pt_spans)
        root = root.view(batch, self.tgt_nt_states, nt_spans)

        # (batch, r, TGT_NT), R -> B/C
        # (batch, r, TGT_PT), R -> B/C
        size = (self.tgt_nt_states, self.tgt_pt_states)
        TLNT, TLPT = torch.split(params["left"], size, -1)
        TRNT, TRPT = torch.split(params["right"], size, -1)

        if marginal:
            span_indicator = term.new_ones(
                batch, N, N, self.max_states, nt_spans, requires_grad=True
            )
            span_indicator_running = span_indicator[:, :, :, : self.tgt_nt_states]
        else:
            span_indicator = None

        normalizer = term.new_full((batch, N, N), -1e9)
        norm = term.flatten(2).max(-1)[0]
        diagonal_copy_(normalizer, norm, w=1)
        term = (term - norm[..., None, None]).exp()

        left_s = term.new_full((batch, N, N, max_spans, R), -1e9)
        right_s = term.new_full((batch, N, N, max_spans, R), -1e9)
        if marginal:
            indicator = (
                span_indicator[:, :, :, : self.tgt_pt_states]
                .diagonal(1, 1, 2)
                .movedim(-1, 1)
            )
            term = term * indicator
        left_term = torch.einsum("xlpi,xrp->xlir", term, TLPT)
        right_term = torch.einsum("xlpi,xrp->xlir", term, TRPT)
        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (
            torch.arange(2, N + 1).unsqueeze(1) <= lens.cpu().unsqueeze(0)
        ).sum(1)

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
            x, xn = merge_h(y, z, yn, zn, SLR, head)

            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            if constraint is not None:
                value, mask = constraint[step]
                if value.ndim > 0:
                    value = value[:current_bsz]
                mask = mask[:current_bsz]
                x, xn = set_score(x, xn, mask, value)

            if marginal:
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
                    head = head[:unfinished]
                    TLNT = TLNT[:unfinished]
                    TRNT = TRNT[:unfinished]
                    normalizer = normalizer[:unfinished]
                    xn = xn[:unfinished]
                    if marginal:
                        span_indicator_running = span_indicator_running[:unfinished]

                left_x = torch.einsum("qnai,qra->qnir", x, TLNT)
                right_x = torch.einsum("qnai,qra->qnir", x, TRNT)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
                diagonal_copy_(normalizer, xn, w)
            if unfinished == 0:
                break
        final = torch.cat(final, dim=0)
        final_normalizer = torch.cat(final_normalizer, dim=0)
        final = (final + 1e-9).squeeze(1).log() + root
        logZ = final.logsumexp((-2, -1)) + final_normalizer.squeeze(-1)
        if decode:
            spans = self.mbr_decoding(logZ, span_indicator, lens)
            return spans
        if marginal:
            torch.set_grad_enabled(grad_state)
            cm.__exit__(None, None, None)
            return grad(logZ.sum(), [span_indicator])[0]
        return -logZ

    def inside_with_fused_factor(self, params: Dict[str, Tensor], lens):
        if not isinstance(lens, torch.Tensor):
            lens = torch.tensor(lens)
        assert (lens[1:] <= lens[:-1]).all(), "Expect lengths in descending."

        terms = params["term"]  # (batch, seq_len, PT)
        root = params["root"]  # (batch, NT)

        batch, N, PT = terms.shape
        N += 1
        NT = root.shape[1]
        nt_spans = NT // self.tgt_nt_states
        pt_spans = PT // self.tgt_pt_states

        terms = terms.view(batch, -1, self.tgt_pt_states, pt_spans)
        root = root.view(batch, self.tgt_nt_states, nt_spans)

        H = params["head"]  # (batch, NT, r), A[i] -> R

        # (batch, r, TGT_NT), R -> B/C
        # (batch, r, TGT_PT), R -> B/C
        size = (self.tgt_nt_states, self.tgt_pt_states)
        TLNT, TLPT = torch.split(params["left"], size, -1)
        TRNT, TRPT = torch.split(params["right"], size, -1)

        R = H.shape[-1]
        H = H.view(batch, self.tgt_nt_states, nt_spans, R)
        HL = torch.einsum("qair,qla->qril", H, TLNT)
        HR = torch.einsum("qair,qla->qril", H, TRNT)

        # ===== This routine is optimized for src_len < tgt_len =====

        if "slr" in params:
            SLR = params["slr"]
        else:
            SL = params["sl"]  # (batch, r, SRC_NT, SRC_NT), R, i -> j
            SR = params["sr"]  # (batch, r, SRC_NT, SRC_NT), R, i -> k
            SLR = SL.unsqueeze(-1) + SR.unsqueeze(-2)

        # ===== End =====

        normalizer = terms.new_full((batch, N, N), -1e9)
        norm = terms.flatten(2).max(-1)[0]
        diagonal_copy_(normalizer, norm, w=1)
        terms = (terms - norm[..., None, None]).exp()

        left_s = terms.new_full((batch, N, N, nt_spans, R), -1e9)
        right_s = terms.new_full((batch, N, N, nt_spans, R), -1e9)
        left_term = torch.einsum("xlpi,xrp->xlir", terms, TLPT)
        right_term = torch.einsum("xlpi,xrp->xlir", terms, TRPT)
        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (
            torch.arange(2, N + 1).unsqueeze(1) <= lens.cpu().unsqueeze(0)
        ).sum(1)

        # w: span width
        final = []
        final_normalizer = []
        for step, w in enumerate(range(2, N)):

            # n: the number of spans of width w.
            n = N - w
            y = stripe(left_s, n, w - 1, (0, 1)).clone()
            z = stripe(right_s, n, w - 1, (1, w), 0).clone()
            y_normalizer = stripe(normalizer, n, w - 1, (0, 1))
            z_normalizer = stripe(normalizer, n, w - 1, (1, w), 0)
            x, x_normalizer = merge(y, z, y_normalizer, z_normalizer, SLR)

            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            if current_bsz - unfinished > 0:
                final.insert(
                    0,
                    torch.einsum(
                        "qnri,qair->qnai",
                        x[unfinished:current_bsz, :1],
                        H[unfinished:current_bsz],
                    ),
                )
                final_normalizer.insert(0, x_normalizer[unfinished:current_bsz, :1])
            if unfinished > 0:
                if current_bsz > unfinished:
                    x = x[:unfinished]
                    left_s = left_s[:unfinished]
                    right_s = right_s[:unfinished]
                    SLR = SLR[:unfinished]
                    HL = HL[:unfinished]
                    HR = HR[:unfinished]
                    normalizer = normalizer[:unfinished]
                    x_normalizer = x_normalizer[:unfinished]

                left_x = torch.einsum("qnri,qril->qnil", x, HL)
                right_x = torch.einsum("qnri,qril->qnil", x, HR)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
                diagonal_copy_(normalizer, x_normalizer, w)
            if unfinished == 0:
                break
        final = torch.cat(final, dim=0)
        final_normalizer = torch.cat(final_normalizer, dim=0)
        final = (final + 1e-9).squeeze(1).log() + root
        logZ = final.logsumexp((-2, -1)) + final_normalizer.squeeze(-1)
        return -logZ

    @torch.no_grad()
    def sampled_decoding(
        self,
        params: Dict[str, Tensor],
        nt_spans,
        src_nt_states,
        pt_spans,
        src_pt_states,
        use_copy=True,
        num_samples=10,
        max_length=100,
        strict=False,
    ):
        terms = params["term"]
        roots = params["root"]
        H = params["head"]  # (batch, NT, r) r:=rank
        L = params["left"]  # (batch, r, TGT_NT + TGT_PT)
        R = params["right"]  # (batch, r, TGT_NT + TGT_PT)
        SLR = params["slr"]

        terms = terms.exp().cumsum(2)
        roots = roots.exp().cumsum(1)
        H = H.cumsum(2)
        LNT, LPT = L.split((self.tgt_nt_states, self.tgt_pt_states), 2)
        LNT = LNT.cumsum(2)
        LPT = LPT.cumsum(2)
        RNT, RPT = R.split((self.tgt_nt_states, self.tgt_pt_states), 2)
        RNT = RNT.cumsum(2)
        RPT = RPT.cumsum(2)
        SLR = self.threshold(SLR).flatten(3).cumsum(3)
        # SLR = SLR.flatten(3).cumsum(3)

        terms = terms.cpu().numpy()
        roots = roots.cpu().numpy()
        H = H.cpu().numpy()
        LNT = LNT.cpu().numpy()
        LPT = LPT.cpu().numpy()
        RNT = RNT.cpu().numpy()
        RPT = RPT.cpu().numpy()
        SLR = SLR.cpu().numpy()

        max_nt_spans = max(len(item) for item in nt_spans)
        max_pt_spans = max(len(item) for item in pt_spans)

        preds = []
        for b in range(len(terms)):
            samples, types, status = self.sample(
                terms[b],
                H[b],
                LNT[b],
                LPT[b],
                RNT[b],
                RPT[b],
                SLR[b],
                roots[b],
                max_nt_spans,
                src_nt_states,
                max_pt_spans,
                src_pt_states,
                use_copy=use_copy,
                num_samples=num_samples,
                max_length=max_length,
            )
            if (cnt := sum(item == _REACHLIMIT for item in status)) > 0:
                log.warning(f"{cnt} trials are terminated due to REACHLIMIT")
            if (cnt := sum(item == _SONMASK for item in status)) > 0:
                log.warning(f"{cnt} trials are terminated due to SONMASK")
            samples = [
                (sample, type_)
                for sample, type_, status_ in zip(samples, types, status)
                if len(sample) > 1 and (not strict or status_ == _OK)
            ]  # len=0 when max_actions is reached but no PT rules applied
            if len(samples) == 0:
                log.warning("All trials are failed.")
                samples = [([0, 0], [TokenType.VOCAB, TokenType.VOCAB])]
            preds.append(samples)
        return preds

    @staticmethod
    @jit(nopython=True)
    def sample(
        terms: np.ndarray,  # seqlen x pt, in normal space
        rules_head: np.ndarray,  # nt x r, in normal space
        rules_left_nt: np.ndarray,  # nt x r, in normal space
        rules_left_pt: np.ndarray,  # pt x r, in normal space
        rules_right_nt: np.ndarray,  # nt x r, in normal space
        rules_right_pt: np.ndarray,  # pt x r, in normal space
        rules_src: np.ndarray,  # r x src x src x src, in normal space
        roots: np.ndarray,  # nt, in normal space
        nt_num_nodes: int,
        nt_states: int,
        pt_num_nodes: int,
        pt_states: int,
        use_copy=True,
        num_samples=1,
        max_length=100,
        max_actions=100,
        UNK=1,
    ):
        NT = rules_head.shape[0]
        COPY_NT = nt_states - 1
        COPY_PT = pt_states - 1
        samples = [[0] for _ in range(num_samples)]
        types = [[0] for _ in range(num_samples)]
        status = [_OK for _ in range(num_samples)]

        for i in range(num_samples):
            try:
                sample = weighted_random_v2(roots)
                nonterminals: List[int] = [sample]
                preterminals: List[int] = []
                is_copy_nt: List[bool] = []
                actions = 0

                while (
                    len(nonterminals) > 0
                    and len(preterminals) < max_length
                    and actions < max_actions
                ):
                    s = nonterminals.pop()
                    if s < NT:
                        nt_state, nt_node = divmod(s, nt_num_nodes)
                        if use_copy and nt_state == COPY_NT:
                            preterminals.append(nt_node)
                            is_copy_nt.append(True)
                            continue
                        actions += 1
                        r = weighted_random_v2(rules_head[s])
                        jk = weighted_random_v2(rules_src[r, nt_node])
                        j, k = divmod(jk, nt_num_nodes)
                        if rules_src[0, j, -1] < 1e-6:
                            left = weighted_random_v2(rules_left_pt[r]) + nt_states
                        else:
                            left = weighted_random_v2(rules_left_nt[r])
                        if rules_src[0, k, -1] < 1e-6:
                            right = weighted_random_v2(rules_right_pt[r]) + pt_states
                        else:
                            right = weighted_random_v2(rules_right_nt[r])
                        nonterminals.extend(
                            [right * nt_num_nodes + k, left * nt_num_nodes + j]
                        )
                    else:
                        preterminals.append(s - NT)
                        is_copy_nt.append(False)

            except Exception:
                status[i] = _SONMASK

            if actions == max_actions or (
                len(preterminals) == max_length and len(nonterminals) > 0
            ):
                status[i] = _REACHLIMIT

            try:
                terminals: List[int] = []
                terminal_type: List[int] = []  # 0=vocab, 1=nt span, 2=pt span
                for s, flag in zip(preterminals, is_copy_nt):
                    if flag:
                        terminals.append(s)
                        terminal_type.append(_COPY_NT)
                    else:
                        src_pt_state = s // pt_num_nodes
                        if use_copy and src_pt_state == COPY_PT:
                            src_node = s % pt_num_nodes
                            terminals.append(src_node)
                            terminal_type.append(_COPY_PT)
                        else:
                            sample = weighted_random_v2(terms[s])
                            if use_copy and sample == UNK:
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

    @staticmethod
    def get_pcfg_rules(params, nt_states):
        head = params["head"]
        b, _, r = head.shape
        head = params["head"].view(b, nt_states, -1, r)
        rule = torch.einsum(
            "xair,xrb,xrc,xrijk->xaibjck",
            head,
            params["left"],
            params["right"],
            params["slr"],
        )
        shape = rule.shape
        rule = (
            rule.reshape(
                shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]
            )
            + 1e-9
        ).log()
        return {"term": params["term"], "rule": rule, "root": params["root"]}

    @torch.enable_grad()
    def ce(self, q, p, lens):
        # ce(q, p)
        # TODO consider yzhang's impl
        ql = [q["term"], q["root"], q["head"], q["left"], q["right"], q["slr"]]
        ql = [item.requires_grad_() for item in ql]
        pl = [
            p["term"],
            p["root"],
            (p["head"] + 1e-12).log(),
            (p["left"] + 1e-12).log(),
            (p["right"] + 1e-12).log(),
            (p["slr"] + 1e-12).log(),
        ]
        logZ = -self(q, lens)
        q_margin = grad(logZ.sum(), ql)

        ce = (
            -self(p, lens)
            - (q_margin[0].detach() * pl[0]).sum((1, 2))
            - (q_margin[1].detach() * pl[1]).sum(1)
            - ((q_margin[2].detach() * ql[2].detach()) * pl[2]).sum((1, 2))
            - ((q_margin[3].detach() * ql[3].detach()) * pl[3]).sum((1, 2))
            - ((q_margin[4].detach() * ql[4].detach()) * pl[4]).sum((1, 2))
            - ((q_margin[5].detach() * ql[5].detach()) * pl[5]).sum((1, 2, 3, 4))
        )
        return ce


@torch.jit.script
def eq_qnkrj(v1, v2):
    # "qnwjr,qnwkr->qnkrj"
    v = v1.transpose(-1, -2).unsqueeze(-3) + v2.unsqueeze(-1)
    return torch.logsumexp(v, dim=2)


@checkpoint
@torch.jit.script
def merge(y, z, y_normalizer, z_normalizer, slr):
    y = (y + 1e-9).log() + y_normalizer[..., None, None]
    z = (z + 1e-9).log() + z_normalizer[..., None, None]
    qnkrj = eq_qnkrj(y, z)
    normalizer = qnkrj.flatten(2).max(-1)[0]
    qnkrj = (qnkrj - normalizer[..., None, None, None]).exp()
    x = torch.einsum("qnkrj,qrijk->qnri", qnkrj, slr)
    return x, normalizer


@checkpoint
@torch.jit.script
def merge_h(y, z, y_normalizer, z_normalizer, slr, h):
    y = (y + 1e-9).log() + y_normalizer[..., None, None]
    z = (z + 1e-9).log() + z_normalizer[..., None, None]
    qnkrj = eq_qnkrj(y, z)
    normalizer = qnkrj.flatten(2).max(-1)[0]
    qnkrj = (qnkrj - normalizer[..., None, None, None]).exp()
    x = torch.einsum("qnkrj,qrijk,qair->qnai", qnkrj, slr, h)
    return x, normalizer


@torch.jit.script
def set_score(x, xn, mask, value):
    x_real = (x + 1e-9).log() + xn[..., None, None]
    x_real = torch.where(mask, value, x_real)
    xn = x_real.flatten(2).max(-1)[0]
    x = (x_real - xn[..., None, None]).exp()
    return x, xn
