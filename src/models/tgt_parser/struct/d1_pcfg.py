from enum import IntEnum
from typing import Dict, List, Union

import numpy as np
import torch
import torch_semiring_einsum as tse
from numba import jit, prange
from torch import Tensor
from torch.autograd import grad

from ._fn import diagonal, diagonal_copy_, stripe
from ._utils import checkpoint, reorder, weighted_random
from .td_style_base import TDStyleBase

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

    # TODO fix nt_spans, and pt_spans. this impl assume nt_spans = pt_spans.
    def __init__(self, tgt_nt_states, tgt_pt_states) -> None:
        self.tgt_nt_states = tgt_nt_states
        self.tgt_pt_states = tgt_pt_states
        self.block_size = 32

        self.eq_slr = tse.compile_equation("qrij, qrik->qrijk")
        self.eq_qnkrj = tse.compile_equation("qnwjr,qnwkr->qnkrj")
        self.eq_qnri = tse.compile_equation("qnkrj,qrijk->qnri")
        self.eq_qnai = tse.compile_equation("qnri,qair->qnai")
        self.eq_tor = tse.compile_equation("xlpi,xrp->xlir")

    @reorder
    def __call__(self, params: Dict[str, Tensor], lens, decode=False, marginal=False):
        if decode:
            marginal = True  # MBR decoding
        if marginal:
            grad_state = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            # NOTE I assume marginals are only used for decoding.
            params = {
                k: v.detach() if isinstance(v, Tensor) else v for k, v in params.items()
            }
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

        left_s = terms.new_full((batch, N, N, nt_spans, R), -1e9)
        right_s = terms.new_full((batch, N, N, nt_spans, R), -1e9)
        left_term = tse.log_einsum(self.eq_tor, terms, TLPT, block_size=self.block_size)
        right_term = tse.log_einsum(
            self.eq_tor, terms, TRPT, block_size=self.block_size
        )
        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        if marginal:
            span_indicator = terms.new_zeros(
                batch, N, N, self.tgt_nt_states, nt_spans
            ).requires_grad_()
            # span_indicator = terms.new_zeros(batch, N, N).requires_grad_()
            span_indicator_running = span_indicator[:]
        else:
            span_indicator = None

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
        terms = params["term"].detach()
        roots = params["root"].detach()
        H = params["head"].detach()  # (batch, NT, r) r:=rank
        L = params["left"].detach()  # (batch, r, TGT_NT + TGT_PT)
        R = params["right"].detach()  # (batch, r, TGT_NT + TGT_PT)
        SLR = params["slr"].detach()

        terms = terms.softmax(2).clamp(1e-3).cumsum(2)
        roots = roots.softmax(1).clamp(1e-3).cumsum(1)
        H = H.softmax(2).clamp(1e-3).cumsum(2)
        L = L.softmax(2).clamp(1e-3).cumsum(2)
        R = R.softmax(2).clamp(1e-3).cumsum(2)
        SLR = SLR.flatten(3).softmax(2).clamp(1e-3).cumsum(2)

        terms = terms.cpu().numpy()
        roots = roots.cpu().numpy()
        H = H.cpu().numpy()
        L = L.cpu().numpy()
        R = R.cpu().numpy()
        SLR = SLR.cpu().numpy()

        preds = []
        for b in range(len(terms)):
            samples, types, scores = self.sample(
                terms[b],
                H[b],
                L[b],
                R[b],
                SLR[b],
                roots[b],
                len(nt_spans[b]),
                src_nt_states,
                len(pt_spans[b]),
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
    @jit(nopython=True, parallel=True)
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
            is_copy_pt: List[bool] = []

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
                            is_copy_pt.append(True)
                            continue
                    actions += 1
                    r = weighted_random(rules_head[s])
                    left = weighted_random(rules_left[r])
                    right = weighted_random(rules_right[r])
                    jk = weighted_random(rules_src[r, nt_node])
                    j, k = divmod(jk, nt_num_nodes)
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
                    is_copy_pt.append(False)

            terminals: List[int] = []
            terminal_type: List[int] = []  # 0=vocab, 1=nt span, 2=pt span
            for s, flag in zip(preterminals, is_copy_pt):
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
            head.exp(),
            params["left"].exp(),
            params["right"].exp(),
            params["slr"].exp(),
        )
        shape = rule.shape
        rule = rule.reshape(
            shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]
        ).log()
        return {"term": params["term"], "rule": rule, "root": params["root"]}


if __name__ == "__main__":

    from .pcfg import PCFG

    torch.autograd.set_detect_anomaly(True)
    torch.random.manual_seed(1)
    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r = 2, 6, 3, 3, 3, 3, 2
    NT = TGT_NT * SRC_NT
    T = TGT_PT * SRC_PT
    device = "cuda"

    slr = (
        torch.randn(B, r, SRC_NT, SRC_NT, SRC_NT, device=device)
        .view(B, r * SRC_NT, -1)
        .log_softmax(-1)
        .view(B, r, SRC_NT, SRC_NT, SRC_NT)
    )
    params = {
        "term": torch.randn(B, N, T, device=device)
        .log_softmax(-1)
        .requires_grad_(True),
        "root": torch.randn(B, NT, device=device).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r, device=device)
        .log_softmax(-1)
        .requires_grad_(True),
        "left": torch.randn(B, r, TGT_NT + TGT_PT, device=device)
        .log_softmax(-1)
        .requires_grad_(True),
        "right": torch.randn(B, r, TGT_NT + TGT_PT, device=device)
        .log_softmax(-1)
        .requires_grad_(True),
        "slr": slr,
    }
    lens = torch.tensor([N - 4, N - 2], dtype=torch.long, device=device)

    pcfg = D1PCFG(TGT_NT, TGT_PT)

    print(pcfg(params, lens))
    m1 = pcfg(params, lens, marginal=True)
    print(m1.sum((1, 2)))

    head = params["head"].view(B, TGT_NT, SRC_NT, r)
    rule = torch.einsum(
        "xair,xrb,xrc,xrijk->xaibjck",
        head.exp(),
        params["left"].exp(),
        params["right"].exp(),
        params["slr"].exp(),
    )
    shape = rule.shape
    rule = rule.reshape(
        shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]
    ).log()

    params2 = {"term": params["term"], "rule": rule, "root": params["root"]}
    pcfg2 = PCFG()
    print(pcfg2(params2, lens))
