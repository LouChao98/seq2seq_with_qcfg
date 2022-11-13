import logging
from enum import IntEnum
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from ._fn import diagonal_copy_, stripe
from ._utils import checkpoint, process_param_for_trace, weighted_random_v2
from .base import _COPY_NT, _COPY_PT, _OK, _REACHLIMIT, _SONMASK, _VOCAB, DecompBase, DecompSamplerBase

log = logging.getLogger(__file__)


class TokenType(IntEnum):
    VOCAB = _VOCAB
    COPY_NT = _COPY_NT
    COPY_PT = _COPY_PT


class Decomp7Impl4(DecompBase):
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
    LOGSPACE = (True, False, True, True, False, True)

    def inside(self, params, semiring, trace) -> Tuple[Tensor, Tensor]:
        params = self.preprocess(params, semiring, trace)
        merge_h = checkpoint(partial(g_merge_h, semiring=semiring))
        merge_h2 = checkpoint(partial(g_merge_h2, semiring=semiring))
        merge_h3 = checkpoint(partial(g_merge_h3, semiring=semiring))

        head = params["head"]  # (batch, nt, r), A[i] -> R
        term = params["term"]  # (batch, seq_len, PT)
        root = params["root"]  # (batch, NT)
        constraint = params.get("constraint")

        _, batch, N, PT = term.shape
        _, _, NT, R = head.shape
        N += 1
        nt_spans = NT // self.nt_states
        pt_spans = PT // self.pt_states

        head = head.view(semiring.size, batch, self.nt_states, nt_spans, R)
        term = term.view(semiring.size, batch, -1, self.pt_states, pt_spans)
        root = root.view(semiring.size, batch, self.nt_states, nt_spans)

        # (batch, r, SRC, TGT_NT), R -> B/C
        # (batch, r, SRC, TGT_PT), R -> B/C
        size = [nt_spans, pt_spans]
        TLNT, TLPT = torch.split(params["left"], size, 3)
        TRNT, TRPT = torch.split(params["right"], size, 3)
        TLNT = TLNT[..., : self.nt_states]
        TLPT = TLPT[..., : self.pt_states]
        TRNT = TRNT[..., : self.nt_states]
        TRPT = TRPT[..., : self.pt_states]

        SLR = params["slr"]
        SL1R1 = SLR[:, :, :, :, nt_spans:, nt_spans:]
        SL1R = SLR[:, :, :, :, nt_spans:, :nt_spans]
        SLR1 = SLR[:, :, :, :, :nt_spans, nt_spans:]
        SLR = SLR[:, :, :, :, :nt_spans, :nt_spans]

        if trace:
            span_indicator = term.new_zeros(1, batch, N, N, self.nt_states, nt_spans, requires_grad=True)
            span_indicator_running = span_indicator
        else:
            span_indicator = None
            span_indicator_running = None

        left_s = semiring.new_zeros((batch, N, N, nt_spans, R))
        right_s = semiring.new_zeros((batch, N, N, nt_spans, R))
        # left_term = torch.einsum("xlpi,xrip->xlir", term, TLPT)
        # right_term = torch.einsum("xlpi,xrip->xlir", term, TRPT)
        left_term = semiring.mul(term.permute(0, 1, 2, 4, 3).unsqueeze(-1), TLPT.permute(0, 1, 3, 4, 2).unsqueeze(2))
        left_term = semiring.sum(left_term, dim=4)
        right_term = semiring.mul(term.permute(0, 1, 2, 4, 3).unsqueeze(-1), TRPT.permute(0, 1, 3, 4, 2).unsqueeze(2))
        right_term = semiring.sum(right_term, dim=4)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (torch.arange(2, N + 1).unsqueeze(1) <= self.lens.cpu().unsqueeze(0)).sum(1)

        # w: span width
        final = []
        for step, w in enumerate(range(2, N)):
            # n: the number of spans of width w.
            n = N - w
            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            y_term = left_term[:, :, : N - w].unsqueeze(3)
            z_term = right_term[:, :, w - 1 :].unsqueeze(3)

            if w == 2:
                x = merge_h(y_term, z_term, SL1R1, head)
            else:
                y = stripe(left_s, current_bsz, n, w - 2, (0, 2)).clone()
                z = stripe(right_s, current_bsz, n, w - 2, (1, w), 0).clone()
                if w == 3:
                    x = merge_h2(y, z, y_term, z_term, SL1R, SLR1, head)
                else:
                    x = merge_h3(y, z, y_term, z_term, SL1R, SLR1, SLR, head)

            if constraint is not None:
                value, mask = constraint[step]
                if value.ndim > 0:
                    value = value[:, :current_bsz]
                mask = mask[:current_bsz]
                x = torch.where(mask.unsqueeze(0).expand([semiring.size] + list(mask.shape)), value, x)

            if trace:
                indicator = span_indicator_running.diagonal(w, 2, 3).movedim(-1, 2)
                x = x + indicator

            if current_bsz - unfinished > 0:
                final.insert(0, x[:, unfinished:current_bsz, :1])

            if unfinished > 0:
                if current_bsz > unfinished:
                    x = x[:, :unfinished]
                    left_term = left_term[:, :unfinished]
                    right_term = right_term[:, :unfinished]
                    SLR = SLR[:, :unfinished]
                    SL1R = SL1R[:, :unfinished]
                    SLR1 = SLR1[:, :unfinished]
                    head = head[:, :unfinished]
                    TLNT = TLNT[:, :unfinished]
                    TRNT = TRNT[:, :unfinished]
                    if trace:
                        span_indicator_running = span_indicator_running[:, :unfinished]
                left_x = semiring.mul(
                    x.permute(0, 1, 2, 4, 3).unsqueeze(-1), TLNT.permute(0, 1, 3, 4, 2).unsqueeze(2)
                )
                left_x = semiring.sum(left_x, dim=4)
                right_x = semiring.mul(
                    x.permute(0, 1, 2, 4, 3).unsqueeze(-1), TRNT.permute(0, 1, 3, 4, 2).unsqueeze(2)
                )
                right_x = semiring.sum(right_x, dim=4)
                diagonal_copy_(left_s, left_x, unfinished, w)
                diagonal_copy_(right_s, right_x, unfinished, w)

            if unfinished == 0:
                break

        final = torch.cat(final, dim=1)
        final = semiring.mul(final.squeeze(2), root)
        logZ = semiring.sum(final.flatten(-2), dim=-1)
        logZ = semiring.unconvert(logZ)
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
        )
        return {
            "term": torch.randn(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "head": torch.randn(bsz, nt, r).softmax(-1).requires_grad_(True),
            "left": torch.randn(bsz, r, src, max(tgt_nt, tgt_pt)).log_softmax(-1).requires_grad_(True),
            "right": torch.randn(bsz, r, src, max(tgt_nt, tgt_pt)).log_softmax(-1).requires_grad_(True),
            "slr": slr,
        }


# class Decomp7Sampler(DecompSamplerBase):
#     @torch.no_grad()
#     def process_params(self, params: Dict[str, Tensor]):
#         terms = params["term"].exp().cumsum(2).cpu().numpy()
#         roots = params["root"].exp().cumsum(1).cpu().numpy()
#         H = params["head"].cumsum(2).cpu().numpy()
#         L = params["left"].cumsum(3).cpu().numpy()
#         R = params["right"].cumsum(3).cpu().numpy()

#         SLR = params["slr"].flatten(3).cumsum(3).cpu().numpy()
#         return terms, H, L, R, SLR, roots

#     @staticmethod
#     @jit(nopython=True)
#     def sample_impl(
#         terms: np.ndarray,  # seqlen x pt, in normal space
#         rules_head: np.ndarray,  # nt x r, in normal space
#         rules_left: np.ndarray,  # (nt+pt) x r, in normal space
#         rules_right: np.ndarray,  # (nt+pt) x r, in normal space
#         rules_slr: np.ndarray,
#         roots: np.ndarray,
#         nt_num_nodes: int,
#         nt_states: int,
#         pt_num_nodes: int,
#         pt_states: int,
#         use_copy=True,
#         num_samples=1,
#         max_length=100,
#         max_actions=100,
#         unk=1,
#     ):
#         COPY_NT = nt_states - 1
#         COPY_PT = pt_states - 1
#         samples = [[0] for _ in range(num_samples)]
#         types = [[0] for _ in range(num_samples)]
#         status = [_OK for _ in range(num_samples)]

#         for sidx in range(num_samples):
#             nonterminals: List[Tuple[int, int]] = []
#             preterminals: List[Tuple[int, int, bool]] = []
#             actions = 0
#             try:
#                 s = weighted_random_v2(roots)
#                 state, i = divmod(s, nt_num_nodes)
#                 nonterminals.append((state, i))
#                 while len(nonterminals) > 0 and len(preterminals) < max_length and actions < max_actions:
#                     actions += 1
#                     s, i = nonterminals.pop()

#                     if s >= nt_states:
#                         preterminals.append((s - nt_states, i, False))
#                         continue
#                     if use_copy and s == COPY_NT:
#                         preterminals.append((s, i, True))
#                         continue

#                     r = weighted_random_v2(rules_head[s * nt_num_nodes + i])
#                     jk = weighted_random_v2(rules_slr[r, i])
#                     j, k = divmod(jk, nt_num_nodes + pt_num_nodes)

#                     left = weighted_random_v2(rules_left[r, j])
#                     right = weighted_random_v2(rules_right[r, k])

#                     if j >= nt_num_nodes:
#                         left += nt_states
#                         j -= nt_num_nodes
#                     if k >= nt_num_nodes:
#                         right += nt_states
#                         k -= nt_num_nodes

#                     nonterminals.extend([(right, k), (left, j)])

#             except Exception:
#                 status[sidx] = _SONMASK

#             if actions == max_actions or (len(preterminals) == max_length and len(nonterminals) > 0):
#                 status[sidx] = _REACHLIMIT

#             terminals: List[int] = []
#             terminal_type: List[int] = []
#             try:
#                 for s, i, flag in preterminals:
#                     if flag:
#                         terminals.append(i)
#                         terminal_type.append(_COPY_NT)
#                         continue
#                     if use_copy and s == COPY_PT:
#                         terminals.append(i)
#                         terminal_type.append(_COPY_PT)
#                     else:
#                         sample = weighted_random_v2(terms[s * pt_num_nodes + i])
#                         if use_copy and sample == unk:
#                             # force <unk> tokens to copy
#                             terminals.append(i)
#                             terminal_type.append(_COPY_PT)
#                         else:
#                             terminals.append(sample)
#                             terminal_type.append(_VOCAB)
#             except Exception:
#                 status[sidx] = _SONMASK

#             samples[sidx] = terminals
#             types[sidx] = terminal_type
#         return samples, types, status


# def convert_decomp7impl4_to_pcfg(params, nt_states):
#     params = {**params}
#     params["left"] = params["left"].exp()
#     params["right"] = params["right"].exp()
#     return convert_decomp7_to_pcfg(params, nt_states)


def eq_qnkrj(v1, v2, semiring):
    v = semiring.mul(v1.transpose(-1, -2).unsqueeze(-3), v2.unsqueeze(-1))
    v = semiring.sum(v, dim=3)
    return v


def g_merge_h(y, z, slr, h, semiring):
    x = eq_qnkrj(y, z, semiring)
    (x,), xn = semiring.to_normal_space([x], (3, 4, 5))
    # qn1rjk, q1irjk -> qnirjk -> qnir
    x = semiring.normal_space_sum(
        semiring.normal_space_mul(
            x.permute(0, 1, 2, 4, 5, 3).flatten(4).unsqueeze(3),
            slr.permute(0, 1, 3, 2, 4, 5).flatten(4).unsqueeze(2),
        ),
        dim=5,
    )
    # qn1ir, q1air -> qnair -> qnai
    x = semiring.normal_space_sum(semiring.normal_space_mul(x.unsqueeze(3), h.unsqueeze(2)), dim=5)
    x = semiring.to_log_space(x, xn)
    return x


def g_merge_h2(y, z, y_term, z_term, sl1r, slr1, h, semiring):
    x1 = eq_qnkrj(y_term, z, semiring)
    x3 = eq_qnkrj(y, z_term, semiring)
    (x1, x3), xn = semiring.to_normal_space([x1, x3], (3, 4, 5))
    # qn1rjk, q1irjk -> qnirjk -> qnir
    x1 = semiring.normal_space_sum(
        semiring.normal_space_mul(
            x1.permute(0, 1, 2, 4, 5, 3).flatten(4).unsqueeze(3),
            sl1r.permute(0, 1, 3, 2, 4, 5).flatten(4).unsqueeze(2),
        ),
        dim=5,
    )
    # qn1rjk, q1irjk -> qnirjk -> qnir
    x3 = semiring.normal_space_sum(
        semiring.normal_space_mul(
            x3.permute(0, 1, 2, 4, 5, 3).flatten(4).unsqueeze(3),
            slr1.permute(0, 1, 3, 2, 4, 5).flatten(4).unsqueeze(2),
        ),
        dim=5,
    )
    # qn1ir, q1air -> qnair -> qnai
    x = semiring.normal_space_sum(
        semiring.normal_space_mul(semiring.normal_space_add(x1, x3).unsqueeze(3), h.unsqueeze(2)),
        dim=5,
    )
    x = semiring.to_log_space(x, xn)
    return x


def g_merge_h3(y, z, y_term, z_term, sl1r, slr1, slr, h, semiring):
    x1 = eq_qnkrj(y_term, z[:, :, :, :1], semiring)
    x2 = eq_qnkrj(y[:, :, :, :-1], z[:, :, :, 1:], semiring)
    x3 = eq_qnkrj(y[:, :, :, -1:], z_term, semiring)
    (x1, x2, x3), xn = semiring.to_normal_space([x1, x2, x3], (3, 4, 5))
    # qn1rjk, q1irjk -> qnirjk -> qnir
    x1 = semiring.normal_space_sum(
        semiring.normal_space_mul(
            x1.permute(0, 1, 2, 4, 5, 3).flatten(4).unsqueeze(3),
            sl1r.permute(0, 1, 3, 2, 4, 5).flatten(4).unsqueeze(2),
        ),
        dim=5,
    )
    # qn1rjk, q1irjk -> qnirjk -> qnir
    x2 = semiring.normal_space_sum(
        semiring.normal_space_mul(
            x2.permute(0, 1, 2, 4, 5, 3).flatten(4).unsqueeze(3),
            slr.permute(0, 1, 3, 2, 4, 5).flatten(4).unsqueeze(2),
        ),
        dim=5,
    )
    # qn1rjk, q1irjk -> qnirjk -> qnir
    x3 = semiring.normal_space_sum(
        semiring.normal_space_mul(
            x3.permute(0, 1, 2, 4, 5, 3).flatten(4).unsqueeze(3),
            slr1.permute(0, 1, 3, 2, 4, 5).flatten(4).unsqueeze(2),
        ),
        dim=5,
    )
    # qn1ir, q1air -> qnair -> qnai
    x = semiring.normal_space_sum(
        semiring.normal_space_mul(semiring.normal_space_add(x1, x2, x3).unsqueeze(3), h.unsqueeze(2)),
        dim=5,
    )
    x = semiring.to_log_space(x, xn)
    return x
