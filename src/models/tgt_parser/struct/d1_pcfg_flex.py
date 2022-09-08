from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.autograd import grad

from ._fn import diagonal_copy_, stripe
from ._utils import checkpoint, process_param_for_marginal, weighted_random_v2
from .td_style_base import TDStyleBase

_VOCAB, _COPY_NT, _COPY_PT = 0, 1, 2


class TokenType(IntEnum):
    VOCAB = _VOCAB
    COPY_NT = _COPY_NT
    COPY_PT = _COPY_PT


class D1PCFGFlex(TDStyleBase):
    # A[i] -> B[j], C[k]
    # ================
    # A[i] -> R
    # R -> B
    # R -> C
    # R, i -> j, k
    # ================
    # Time complexity: 6

    # This impl allow tgt_nt_states != tgt_pt_states
    def __init__(self, tgt_nt_states, tgt_pt_states) -> None:
        self.tgt_nt_states = tgt_nt_states
        self.tgt_pt_states = tgt_pt_states
        self.max_states = max(self.tgt_nt_states, self.tgt_pt_states)
        self.threshold = torch.nn.Threshold(1e-3, 0)

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
        copy_nt = params.get("copy_nt")

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

        # ===== This routine is optimized for src_len < tgt_len =====

        if "slr" in params:
            SLR = params["slr"]
        else:
            SL = params["sl"]  # (batch, r, SRC_NT, SRC_NT), R, i -> j
            SR = params["sr"]  # (batch, r, SRC_NT, SRC_NT), R, i -> k
            SLR = SL.unsqueeze(-1) + SR.unsqueeze(-2)

        # NOTE SL1R1, SL1R, SLR1, SLR should be normalized for each.
        SL1R1 = SLR[:, :, :, nt_spans:, nt_spans:]
        SL1R = SLR[:, :, :, nt_spans:, :nt_spans]
        SLR1 = SLR[:, :, :, :nt_spans, nt_spans:]
        SLR = SLR[:, :, :, :nt_spans, :nt_spans]

        # ===== End =====

        if marginal:
            span_indicator = term.new_ones(
                batch, N, N, self.max_states, max_spans, requires_grad=True
            )
            span_indicator_running = span_indicator[
                :, :, :, : self.tgt_nt_states, :nt_spans
            ]
        else:
            span_indicator = None

        normalizer = term.new_full((batch, N, N), -1e9)
        norm = term.flatten(2).max(-1)[0]
        diagonal_copy_(normalizer, norm, w=1)
        term = (term - norm[..., None, None]).exp()

        left_s = term.new_full((batch, N, N, max_spans, R), 0.0)
        right_s = term.new_full((batch, N, N, max_spans, R), 0.0)
        if marginal:
            indicator = span_indicator.diagonal(1, 1, 2).movedim(-1, 1)
            term = term * indicator[..., : self.tgt_pt_states, :pt_spans]
        left_term = torch.einsum("xlpi,xrp->xlir", term, TLPT)
        right_term = torch.einsum("xlpi,xrp->xlir", term, TRPT)
        diagonal_copy_(left_s, left_term, w=1, s3=pt_spans)
        diagonal_copy_(right_s, right_term, w=1, s3=pt_spans)

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

            if w == 2:
                x, xn = merge_h(y, z, yn, zn, SL1R1, head)
            elif w == 3:
                x, xn = merge_h2(y, z, yn, zn, SL1R, SLR1, head)
            else:
                x, xn = merge_h3(y, z, yn, zn, SL1R, SLR1, SLR, head)

            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            if copy_nt is not None:
                value, mask = copy_nt[step]
                if value.ndim > 0:
                    value = value[:current_bsz]
                mask = mask[:current_bsz]
                x = torch.where(mask, (value - xn[..., None, None]).exp(), x)

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
                    SL1R = SL1R[:unfinished]
                    SLR1 = SLR1[:unfinished]
                    head = head[:unfinished]
                    TLNT = TLNT[:unfinished]
                    TRNT = TRNT[:unfinished]
                    normalizer = normalizer[:unfinished]
                    xn = xn[:unfinished]
                    if marginal:
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
        if decode:
            spans = self.mbr_decoding(logZ, span_indicator, lens)
            return spans
        if marginal:
            torch.set_grad_enabled(grad_state)
            cm.__exit__(None, None, None)
            return grad(logZ.sum(), [span_indicator])[0]
        return -logZ

    def logZ(self, params: Dict[str, Tensor], lens):
        if not isinstance(lens, torch.Tensor):
            lens = torch.tensor(lens)
        assert (lens[1:] <= lens[:-1]).all(), "Expect lengths in descending."

        term = params["term"]  # (batch, seq_len, PT)
        root = params["root"]  # (batch, NT)
        head = params["head"]  # (batch, NT, r), A[i] -> R

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
        HL = torch.einsum("qair,qla->qril", head, TLNT)
        HR = torch.einsum("qair,qla->qril", head, TRNT)

        # ===== This routine is optimized for src_len < tgt_len =====

        if "slr" in params:
            SLR = params["slr"]
        else:
            SL = params["sl"]  # (batch, r, SRC_NT, SRC_NT), R, i -> j
            SR = params["sr"]  # (batch, r, SRC_NT, SRC_NT), R, i -> k
            SLR = SL.unsqueeze(-1) + SR.unsqueeze(-2)

        SL1R1 = SLR[:, :, :, nt_spans:, nt_spans:]
        SL1R = SLR[:, :, :, nt_spans:, :nt_spans]
        SLR1 = SLR[:, :, :, :nt_spans, nt_spans:]
        SLR = SLR[:, :, :, :nt_spans, :nt_spans]

        # ===== End =====

        normalizer = term.new_full((batch, N, N), -1e9)
        norm = term.flatten(2).max(-1)[0]
        diagonal_copy_(normalizer, norm, w=1)
        term = (term - norm[..., None, None]).exp()

        left_s = term.new_full((batch, N, N, max_spans, R), 0.0)
        right_s = term.new_full((batch, N, N, max_spans, R), 0.0)
        left_term = torch.einsum("xlpi,xrp->xlir", term, TLPT)
        right_term = torch.einsum("xlpi,xrp->xlir", term, TRPT)
        diagonal_copy_(left_s, left_term, w=1, s3=pt_spans)
        diagonal_copy_(right_s, right_term, w=1, s3=pt_spans)

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
            # x, x_normalizer = merge(y, z, y_normalizer, z_normalizer, SLR)

            if w == 2:
                x, x_normalizer = merge(y, z, y_normalizer, z_normalizer, SL1R1)
            elif w == 3:
                x, x_normalizer = merge2(y, z, y_normalizer, z_normalizer, SL1R, SLR1)
            else:
                x, x_normalizer = merge3(
                    y, z, y_normalizer, z_normalizer, SL1R, SLR1, SLR
                )

            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            if current_bsz - unfinished > 0:
                final.insert(
                    0,
                    torch.einsum(
                        "qnri,qair->qnai",
                        x[unfinished:current_bsz, :1],
                        head[unfinished:current_bsz],
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
                diagonal_copy_(left_s, left_x, w, s3=nt_spans)
                diagonal_copy_(right_s, right_x, w, s3=nt_spans)
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
        src_nt_spans,
        tgt_nt_states,
        src_pt_spans,
        tgt_pt_states,
        use_copy=True,
        num_samples=10,
        max_length=100,
    ):
        terms = params["term"]
        roots = params["root"]
        H = params["head"]  # (batch, NT, r) r:=rank
        L = params["left"]  # (batch, r, TGT_NT + TGT_PT)
        R = params["right"]  # (batch, r, TGT_NT + TGT_PT)
        SLR = params["slr"]

        terms = self.threshold(terms.exp()).cumsum(2)
        roots = self.threshold(roots.exp()).cumsum(1)
        H = self.threshold(H).cumsum(2)
        L = self.threshold(L).cumsum(2)
        R = self.threshold(R).cumsum(2)
        SLR = self.threshold(SLR)
        SL1R1 = SLR[..., tgt_nt_states:, tgt_nt_states:].flatten(3).cumsum(3)
        SL1R = SLR[..., tgt_nt_states:, :tgt_nt_states].flatten(3).cumsum(3)
        SLR1 = SLR[..., :tgt_nt_states, tgt_nt_states:].flatten(3).cumsum(3)
        SLR = SLR[..., :tgt_nt_states, :tgt_nt_states].flatten(3).cumsum(3)

        terms = terms.cpu().numpy()
        roots = roots.cpu().numpy()
        H = H.cpu().numpy()
        L = L.cpu().numpy()
        R = R.cpu().numpy()
        SL1R1 = SL1R1.cpu().numpy()
        SL1R = SL1R.cpu().numpy()
        SLR1 = SLR1.cpu().numpy()
        SLR = SLR.cpu().numpy()

        max_nt_spans = max(len(item) for item in src_nt_spans)
        max_pt_spans = max(len(item) for item in src_pt_spans)

        preds = []
        for b in range(len(terms)):
            samples, types = self.sample(
                terms[b],
                H[b],
                L[b],
                R[b],
                SL1R1[b],
                SL1R[b],
                SLR1[b],
                SLR[b],
                roots[b],
                max_nt_spans,
                tgt_nt_states,
                max_pt_spans,
                tgt_pt_states,
                use_copy=use_copy,
                num_samples=num_samples,
                max_length=max_length,
            )
            sample_scores = [
                (sample, type_)
                for sample, type_ in zip(samples, types)
                # if len(sample) > 0
            ]  # any case for len(sample) = 0 in new imp?
            if len(sample_scores) == 0:
                sample_scores = ([0, 0], [TokenType.VOCAB, TokenType.VOCAB], 0)
            preds.append(sample_scores)
        return preds

    @staticmethod
    def sample(
        terms: np.ndarray,  # seqlen x pt, in normal space
        rules_head: np.ndarray,  # nt x r, in normal space
        rules_left: np.ndarray,  # (nt+pt) x r, in normal space
        rules_right: np.ndarray,  # (nt+pt) x r, in normal space
        rules_sl1r1: np.ndarray,  # r x src x src x src, in normal space
        rules_sl1r: np.ndarray,  # r x src x src x src, in normal space
        rules_slr1: np.ndarray,  # r x src x src x src, in normal space
        rules_slr: np.ndarray,  # r x src x src x src, in normal space
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
        COPY_NT = nt_states - 1
        COPY_PT = pt_states - 1
        samples = [[0] for _ in range(num_samples)]
        types = [[0] for _ in range(num_samples)]

        for _ in range(num_samples):
            try:
                sample = weighted_random_v2(roots)
                state, i = divmod(s, nt_states)
                nonterminals: List[Tuple[int, int]] = [(state, i)]
                preterminals: List[int] = []
                actions = 0

                while (
                    len(nonterminals) > 0
                    and len(preterminals) < max_length
                    and actions < max_actions
                ):
                    actions += 1
                    s, i = nonterminals.pop()

                    if s > nt_states:
                        preterminals.append((s - nt_states, i, False))
                        continue
                    if use_copy and s == COPY_NT:
                        preterminals.append((s, i, True))
                        continue

                    r = weighted_random_v2(rules_head[s])
                    left = weighted_random_v2(rules_left[r])
                    right = weighted_random_v2(rules_right[r])

                    if left < nt_states and right < nt_states:
                        jk = weighted_random_v2(rules_slr[r, i])
                        j, k = divmod(jk, nt_num_nodes)
                    elif left < nt_states and right >= nt_states:
                        jk = weighted_random_v2(rules_slr1[r, i])
                        j, k = divmod(jk, pt_num_nodes)
                    elif left >= nt_states and right < nt_states:
                        jk = weighted_random_v2(rules_sl1r[r, i])
                        j, k = divmod(jk, nt_num_nodes)
                    elif left >= nt_states and right >= nt_states:
                        jk = weighted_random_v2(rules_sl1r1[r, i])
                        j, k = divmod(jk, pt_num_nodes)

                    nonterminals.extend([(right, k), (left, j)])

                # try to generate something. just use NT->PT PT
                while len(nonterminals) > 0 and len(preterminals) < max_length:
                    s, i = nonterminals.pop()

                    if s > nt_states:
                        preterminals.append((s - nt_states, i, False))
                        continue
                    if use_copy and s == COPY_NT:
                        preterminals.append((s, i, True))
                        continue

                    r = weighted_random_v2(rules_head[s])
                    left = weighted_random_v2(rules_left[r, nt_states:]) + nt_states
                    right = weighted_random_v2(rules_right[r, nt_states:]) + nt_states

                    jk = weighted_random_v2(rules_sl1r1[r, i])
                    j, k = divmod(jk, pt_num_nodes)
                    nonterminals.extend([(right, k), (left, j)])

                terminals: List[int] = []
                terminal_type: List[int] = []
                for s, i, flag in preterminals:
                    if flag:
                        terminals.append(i)
                        terminal_type.append(_COPY_NT)
                        continue
                    if use_copy and s == COPY_PT:
                        terminals.append(i)
                        terminal_type.append(_COPY_PT)
                    else:
                        sample = weighted_random_v2(terms[s])
                        if use_copy and sample == UNK:
                            # force <unk> tokens to copy
                            terminals.append(i)
                            terminal_type.append(_COPY_PT)
                        else:
                            terminals.append(sample)
                            terminal_type.append(_VOCAB)
            except ValueError as e:
                if str(e) == "Sampling on masked NT.":
                    continue
                else:
                    raise e
            samples[i] = terminals
            types[i] = terminal_type
        return samples, types

    @staticmethod
    def get_pcfg_rules(params, nt_states):
        B, _, r = params["head"].shape
        TGT_NT = nt_states
        TGT_PT = params["left"].shape[-1] - TGT_NT
        head = params["head"].view(B, nt_states, -1, r)
        SRC_NT = head.shape[2]
        SRC_PT = params["slr"].shape[-1] - SRC_NT
        rule11 = torch.einsum(
            "xair,xrb,xrc,xrijk->xaibjck",
            head,
            params["left"][:, :, :TGT_NT],
            params["right"][:, :, :TGT_NT],
            params["slr"][:, :, :, :SRC_NT, :SRC_NT],
        )
        rule12 = torch.einsum(
            "xair,xrb,xrc,xrijk->xaibjck",
            head,
            params["left"][:, :, :TGT_NT],
            params["right"][:, :, TGT_NT:],
            params["slr"][:, :, :, :SRC_NT, SRC_NT:],
        )
        rule21 = torch.einsum(
            "xair,xrb,xrc,xrijk->xaibjck",
            head,
            params["left"][:, :, TGT_NT:],
            params["right"][:, :, :TGT_NT],
            params["slr"][:, :, :, SRC_NT:, :SRC_NT],
        )
        rule22 = torch.einsum(
            "xair,xrb,xrc,xrijk->xaibjck",
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
        rule[:, :, : TGT_NT * SRC_NT, : TGT_NT * SRC_NT] = rule11.reshape(
            shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]
        ).log()
        shape = rule12.shape
        rule[:, :, : TGT_NT * SRC_NT, TGT_NT * SRC_NT :] = rule12.reshape(
            shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]
        ).log()
        shape = rule21.shape
        rule[:, :, TGT_NT * SRC_NT :, : TGT_NT * SRC_NT] = rule21.reshape(
            shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]
        ).log()
        shape = rule22.shape
        rule[:, :, TGT_NT * SRC_NT :, TGT_NT * SRC_NT :] = rule22.reshape(
            shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]
        ).log()
        return {"term": params["term"], "rule": rule, "root": params["root"]}


@torch.jit.script
def eq_qnkrj(v1, v2):
    # "qnwjr,qnwkr->qnkrj"
    v = v1.transpose(-1, -2).unsqueeze(-3) + v2.unsqueeze(-1)
    return torch.logsumexp(v, dim=2)


@checkpoint
@torch.jit.script
def merge(y, z, y_normalizer, z_normalizer, slr):
    num = slr.shape[3]
    y = (y[:, :, :, :num] + 1e-9).log() + y_normalizer[..., None, None]
    z = (z[:, :, :, :num] + 1e-9).log() + z_normalizer[..., None, None]
    qnkrj = eq_qnkrj(y, z)
    normalizer = qnkrj.flatten(2).max(-1)[0]
    qnkrj = (qnkrj - normalizer[..., None, None, None]).exp()
    x = torch.einsum("qnkrj,qrijk->qnri", qnkrj, slr)
    return x, normalizer


@checkpoint
@torch.jit.script
def merge2(y, z, y_normalizer, z_normalizer, sl1r, slr1):
    num_pt, num_nt = sl1r.shape[-2:]
    y = (y + 1e-9).log() + y_normalizer[..., None, None]
    z = (z + 1e-9).log() + z_normalizer[..., None, None]
    qnkrj1 = eq_qnkrj(y[:, :, :1, :num_pt], z[:, :, :1, :num_nt])
    qnkrj3 = eq_qnkrj(y[:, :, -1:, :num_nt], z[:, :, -1:, :num_pt])
    normalizer = torch.stack(
        [
            qnkrj1.flatten(2).max(-1)[0],
            qnkrj3.flatten(2).max(-1)[0],
        ],
        dim=-1,
    ).max(-1)[0]
    qnkrj1 = (qnkrj1 - normalizer[..., None, None, None]).exp()
    qnkrj3 = (qnkrj3 - normalizer[..., None, None, None]).exp()
    x1 = torch.einsum("qnkrj,qrijk->qnri", qnkrj1, sl1r)
    x3 = torch.einsum("qnkrj,qrijk->qnri", qnkrj3, slr1)
    x = x1 + x3
    return x, normalizer


@checkpoint
@torch.jit.script
def merge3(y, z, y_normalizer, z_normalizer, sl1r, slr1, slr):
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
    x = x1 + x2 + x3
    return x, normalizer


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
    normalizer = torch.stack(
        [
            qnkrj1.flatten(2).max(-1)[0],
            qnkrj3.flatten(2).max(-1)[0],
        ],
        dim=-1,
    ).max(-1)[0]
    qnkrj1 = (qnkrj1 - normalizer[..., None, None, None]).exp()
    qnkrj3 = (qnkrj3 - normalizer[..., None, None, None]).exp()
    x1 = torch.einsum("qnkrj,qrijk,qair->qnai", qnkrj1, sl1r, h)
    x3 = torch.einsum("qnkrj,qrijk,qair->qnai", qnkrj3, slr1, h)
    x = x1 + x3
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
    x1 = torch.einsum("qnkrj,qrijk,qair->qnai", qnkrj1, sl1r, h)
    x2 = torch.einsum("qnkrj,qrijk,qair->qnai", qnkrj2, slr, h)
    x3 = torch.einsum("qnkrj,qrijk,qair->qnai", qnkrj3, slr1, h)
    x = x1 + x2 + x3
    return x, normalizer
