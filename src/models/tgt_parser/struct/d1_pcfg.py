from enum import IntEnum
from typing import Dict, List

import numpy as np
import torch
from numba import jit, prange
from torch import Tensor
from torch.autograd import grad

from ._fn import diagonal, diagonal_copy_, stripe
from ._utils import checkpoint, process_param_for_marginal, weighted_random
from .td_style_base import TDStyleBase

# import torch_semiring_einsum as tse

_VOCAB, _COPY_NT, _COPY_PT = 0, 1, 2


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
        # import torch_semiring_einsum as tse

        # self.eq_slr = tse.compile_equation("qrij, qrik->qrijk")
        # self.eq_qnkrj = tse.compile_equation("qnwjr,qnwkr->qnkrj")
        # self.eq_qnri = tse.compile_equation("qnkrj,qrijk->qnri")
        # self.eq_qnai = tse.compile_equation("qnri,qair->qnai")
        # self.eq_tor = tse.compile_equation("xlpi,xrp->xlir")
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
            y_normalizer = stripe(normalizer, n, w - 1, (0, 1))
            z_normalizer = stripe(normalizer, n, w - 1, (1, w), 0)
            x, x_normalizer = merge_h(y, z, y_normalizer, z_normalizer, SLR, head)

            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            if copy_nt is not None:
                value, mask = copy_nt[step]
                if value.ndim > 0:
                    value = value[:current_bsz]
                mask = mask[:current_bsz]
                x = torch.where(mask, (value - x_normalizer[..., None, None]).exp(), x)

            if marginal:
                indicator = span_indicator_running.diagonal(w, 1, 2).movedim(-1, 1)
                x = x * indicator

            if current_bsz - unfinished > 0:
                final.insert(0, x[unfinished:current_bsz, :1])
                final_normalizer.insert(0, x_normalizer[unfinished:current_bsz, :1])
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
                    x_normalizer = x_normalizer[:unfinished]
                    if marginal:
                        span_indicator_running = span_indicator_running[:unfinished]

                left_x = torch.einsum("qnai,qra->qnir", x, TLNT)
                right_x = torch.einsum("qnai,qra->qnir", x, TRNT)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
                diagonal_copy_(normalizer, x_normalizer, w)
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

    def inside_semiring_einsum(
        self, params: Dict[str, Tensor], lens, decode=False, marginal=False
    ):
        # NOTE current scorer do not support this. because this need everything in log-space.
        if decode:
            marginal = True  # MBR decoding
        if marginal:
            grad_state = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            cm = torch.inference_mode(False)
            cm.__enter__()
            # NOTE I assume marginals are only used for decoding.
            params = {k: process_param_for_marginal(v) for k, v in params.items()}
        if not isinstance(lens, torch.Tensor):
            lens = torch.tensor(lens)
        assert (
            lens[1:] <= lens[:-1]
        ).all(), "You should sort samples by length descently."

        terms = params["term"]  # (batch, seq_len, PT)
        root = params["root"]  # (batch, NT)
        copy_nts = params.get("copy_nt")

        batch, N, PT = terms.shape
        N += 1
        NT = root.shape[1]
        nt_spans = NT // self.tgt_nt_states
        pt_spans = PT // self.tgt_pt_states

        terms = terms.view(batch, -1, self.tgt_pt_states, pt_spans)
        root = root.view(batch, self.tgt_nt_states, nt_spans)

        # {source,target}{left,right}[{nonterminal,preterminal}]
        H = params["head"]  # (batch, NT, r), A[i] -> R
        # (batch, r, TGT_NT), R -> B
        # (batch, r, TGT_PT), R -> B
        TLNT, TLPT = torch.split(
            params["left"], (self.tgt_nt_states, self.tgt_pt_states), -1
        )
        # (batch, r, TGT_NT), R -> C
        # (batch, r, TGT_PT), R -> C
        TRNT, TRPT = torch.split(
            params["right"], (self.tgt_nt_states, self.tgt_pt_states), -1
        )
        R = H.shape[-1]
        H = H.view(batch, self.tgt_nt_states, nt_spans, R)

        # ===== This routine is optimized for src_len < tgt_len =====

        if "slr" in params:
            SLR = params["slr"]
        else:
            SL = params["sl"]  # (batch, r, SRC_NT, SRC_NT), R, i -> j
            SR = params["sr"]  # (batch, r, SRC_NT, SRC_NT), R, i -> k
            SLR = tse.log_einsum(self.eq_slr, SL, SR, block_size=self.block_size)

        # ===== End =====

        if marginal:
            span_indicator = terms.new_zeros(
                batch, N, N, self.tgt_nt_states, nt_spans
            ).requires_grad_()
            # span_indicator = terms.new_zeros(batch, N, N).requires_grad_()
            span_indicator_running = span_indicator[:]
        else:
            span_indicator = None

        left_s = terms.new_full((batch, N, N, nt_spans, R), -1e9)
        right_s = terms.new_full((batch, N, N, nt_spans, R), -1e9)
        if marginal:
            indicator = span_indicator_running.diagonal(1, 1, 2).movedim(-1, 1)
            terms = terms + indicator
        left_term = tse.log_einsum(self.eq_tor, terms, TLPT, block_size=self.block_size)
        right_term = tse.log_einsum(
            self.eq_tor, terms, TRPT, block_size=self.block_size
        )
        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (
            torch.arange(2, N + 1).unsqueeze(1) <= lens.cpu().unsqueeze(0)
        ).sum(1)

        # w: span width
        final = []
        for step, w in enumerate(range(2, N)):

            # n: the number of spans of width w.
            n = N - w
            y = stripe(left_s, n, w - 1, (0, 1)).clone()
            z = stripe(right_s, n, w - 1, (1, w), 0).clone()

            qnkrj = tse.log_einsum(self.eq_qnkrj, y, z, block_size=self.block_size)
            qnri = tse.log_einsum(self.eq_qnri, qnkrj, SLR, block_size=self.block_size)
            x = tse.log_einsum(self.eq_qnai, qnri, H, block_size=self.block_size)

            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            if copy_nts is not None:
                value, mask = copy_nts[step]
                value = value[:current_bsz]
                mask = mask[:current_bsz]
                x = torch.where(mask, value, x)

            if marginal:
                indicator = span_indicator_running.diagonal(w, 1, 2).movedim(-1, 1)
                # indicator = span_indicator_running.diagonal(w, 1, 2)[..., None, None]
                x += indicator

            if current_bsz - unfinished > 0:
                final.insert(0, x[unfinished:current_bsz, :1])
            if unfinished > 0:
                x = x[:unfinished]
                left_s = left_s[:unfinished]
                right_s = right_s[:unfinished]
                SLR = SLR[:unfinished]
                H = H[:unfinished]
                TLNT = TLNT[:unfinished]
                TRNT = TRNT[:unfinished]
                if marginal:
                    span_indicator_running = span_indicator_running[:unfinished]

                left_x = tse.log_einsum(
                    self.eq_tor, x, TLNT, block_size=self.block_size
                )
                right_x = tse.log_einsum(
                    self.eq_tor, x, TRNT, block_size=self.block_size
                )
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
            if unfinished == 0:
                break

        final = torch.cat(final, dim=0)

        final = final.squeeze(1) + root
        logZ = final.logsumexp((-2, -1))
        if decode:
            spans = self.get_prediction(logZ, span_indicator, lens)
            # spans = [[(span[0], span[1] - 1, 0) for span in inst] for inst in spans]
            return spans
        if marginal:
            torch.set_grad_enabled(grad_state)
            cm.__exit__(None, None, None)
            return grad(logZ.sum(), [span_indicator])[0]
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
        SLR = self.threshold(SLR.flatten(3)).cumsum(3)

        terms = terms.cpu().numpy()
        roots = roots.cpu().numpy()
        H = H.cpu().numpy()
        L = L.cpu().numpy()
        R = R.cpu().numpy()
        SLR = SLR.cpu().numpy()

        max_nt_spans = max(len(item) for item in nt_spans)
        max_pt_spans = max(len(item) for item in pt_spans)

        preds = []
        for b in range(len(terms)):
            samples, types, scores = self.sample(
                terms[b],
                H[b],
                L[b],
                R[b],
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
            sample_scores = [
                (sample, type_, score)
                for sample, type_, score in zip(samples, types, scores)
                if len(sample) > 0
            ]  # len=0 when max_actions is reached but no PT rules applied
            if len(sample_scores) == 0:
                sample_scores = ([0, 0], [TokenType.VOCAB, TokenType.VOCAB], 0)
            preds.append(sample_scores)
        return preds

    @staticmethod
    @jit(nopython=True)
    def sample(
        terms: np.ndarray,  # seqlen x pt, in normal space
        rules_head: np.ndarray,  # nt x r, in normal space
        rules_left: np.ndarray,  # (nt+pt) x r, in normal space
        rules_right: np.ndarray,  # (nt+pt) x r, in normal space
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
        scores = [0.0 for _ in range(num_samples)]

        for i in prange(num_samples):
            actions = 0
            sample = weighted_random(roots)
            # score = roots[sample]
            nonterminals: List[int] = [sample]
            preterminals: List[int] = []
            is_copy_nt: List[bool] = []
            failed = False

            while (
                len(nonterminals) > 0
                and len(preterminals) < max_length
                and actions < max_actions
            ):
                s = nonterminals.pop()
                if s < NT:
                    nt_state, nt_node = divmod(s, nt_num_nodes)
                    if use_copy:
                        if nt_state == COPY_NT:
                            preterminals.append(nt_node)
                            is_copy_nt.append(True)
                            continue
                    actions += 1
                    r = weighted_random(rules_head[s])
                    left = weighted_random(rules_left[r])
                    right = weighted_random(rules_right[r])

                    check_left = not (use_copy and left == COPY_NT)
                    check_right = not (use_copy and right == COPY_NT)
                    for patience in range(5):
                        jk = weighted_random(rules_src[r, nt_node])
                        j, k = divmod(jk, nt_num_nodes)
                        ok = True
                        if check_left:
                            ok &= rules_src[0, j, -1] > 0
                        if check_right:
                            ok &= rules_src[0, k, -1] > 0
                        if ok:
                            break
                    if not ok:
                        failed = True
                        break
                    # score += (
                    #     rules_head[s, r]
                    #     + rules_left[r, left]
                    #     + rules_right[r, right]
                    #     + rules_src[r, nt_node, jk]
                    # )
                    nonterminals.extend(
                        [right * nt_num_nodes + k, left * nt_num_nodes + j]
                    )
                else:
                    preterminals.append(s - NT)
                    is_copy_nt.append(False)

            if failed:
                # print('failed')
                continue

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
                        sample = weighted_random(terms[s])
                        # score += terms[s, sample]
                        if use_copy and sample == UNK:
                            # force <unk> tokens to copy
                            src_node = s % pt_num_nodes
                            terminals.append(src_node)
                            terminal_type.append(_COPY_PT)
                        else:
                            terminals.append(sample)
                            terminal_type.append(_VOCAB)
            samples[i] = terminals
            types[i] = terminal_type
            # scores[i] = score / (len(terminals) + 1e-9)
        return samples, types, scores

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
        rule = rule.reshape(
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
