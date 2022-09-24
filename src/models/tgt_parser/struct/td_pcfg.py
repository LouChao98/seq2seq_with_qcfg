import logging
from enum import IntEnum
from typing import Dict, List, Union

import numpy as np
import torch
from numba import jit, prange
from torch import Tensor
from torch.autograd import grad

from ._fn import diagonal, diagonal_copy_, stripe
from ._utils import checkpoint, process_param_for_marginal, weighted_random
from .td_style_base import TDStyleBase

log = logging.getLogger(__file__)
_VOCAB, _COPY_NT, _COPY_PT = 0, 1, 2


class TokenType(IntEnum):
    VOCAB = _VOCAB
    COPY_NT = _COPY_NT
    COPY_PT = _COPY_PT


class FastestTDPCFG(TDStyleBase):
    # based on Songlin Yang, Wei Liu and Kewei Tu's work
    # https://github.com/sustcsonglin/TN-PCFG/blob/main/parser/pcfgs/tdpcfg.py
    # modification:
    # 1. use new grad api
    # 2. remove unecessary tensor.clone()
    # 3. respect lens

    # This DO NOT support:
    # 1. copy
    # 2. mbr on label
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, params: Dict[str, Tensor], lens, decode=False, marginal=False):
        if decode:
            marginal = True  # MBR decoding
        if marginal:
            grad_state = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            cm = torch.inference_mode(False)
            cm.__enter__()
            params = {k: process_param_for_marginal(v) for k, v in params.items()}
        lens = torch.tensor(lens)
        assert (lens[1:] <= lens[:-1]).all(), "Expect lengths in descending."

        terms = params["term"]
        root = params["root"]

        # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        H = params["head"]  # (batch, NT, r), r:=rank, H->r
        L = params["left"]  # (batch, NT + T, r), r->L
        R = params["right"]  # (batch, NT + T, r), r->R

        batch, N, T = terms.shape
        S = L.shape[1]
        NT = S - T
        N += 1

        L_term = L[:, NT:]
        L_nonterm = L[:, :NT]
        R_term = R[:, NT:]
        R_nonterm = R[:, :NT]

        # (m_A, r_A) + (m_B, r_B) -> (r_A, r_B)
        H = H.transpose(-1, -2)
        H_L = torch.matmul(H, L_nonterm)
        H_R = torch.matmul(H, R_nonterm)
        if marginal:
            span_indicator = terms.new_ones(batch, N, N).requires_grad_()
            span_indicator_running = span_indicator[:]

        normalizer = terms.new_full((batch, N, N), -1e9)
        norm = terms.max(-1)[0]
        diagonal_copy_(normalizer, norm, w=1)
        terms = (terms - norm.unsqueeze(-1)).exp()
        if marginal:
            indicator = span_indicator_running.diagonal(1, 1, 2).unsqueeze(-1)
            terms = terms * indicator
        left_term = torch.matmul(terms, L_term)
        right_term = torch.matmul(terms, R_term)

        # for caching V^{T}s_{i,k} and W^{T}s_{k+1,j}
        left_s = terms.new_full((batch, N, N, L.shape[2]), 0.0)
        right_s = terms.new_full((batch, N, N, L.shape[2]), 0.0)
        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (
            torch.arange(2, N + 1).unsqueeze(1) <= lens.cpu().unsqueeze(0)
        ).sum(1)

        # w: span width
        final_m = []
        final_normalizer = []
        for step, w in enumerate(range(2, N)):

            # n: the number of spans of width w.
            n = N - w
            y = stripe(left_s, n, w - 1, (0, 1))
            z = stripe(right_s, n, w - 1, (1, w), 0)
            yn = stripe(normalizer, n, w - 1, (0, 1))
            zn = stripe(normalizer, n, w - 1, (1, w), 0)
            x, xn = merge(y, z, yn, zn)

            if marginal:
                indicator = span_indicator_running.diagonal(w, 1, 2).unsqueeze(-1)
                x = x * indicator

            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            if current_bsz - unfinished > 0:
                final_m.insert(
                    0,
                    torch.matmul(
                        x[unfinished:current_bsz, :1], H[unfinished:current_bsz]
                    ),
                )
                final_normalizer.insert(0, xn[unfinished:current_bsz, :1])
            if unfinished > 0:
                x = x[:unfinished]
                left_s = left_s[:unfinished]
                right_s = right_s[:unfinished]
                H_L = H_L[:unfinished]
                H_R = H_R[:unfinished]
                normalizer = normalizer[:unfinished]
                xn = xn[:unfinished]
                if marginal:
                    span_indicator_running = span_indicator_running[:unfinished]

                left_x = torch.matmul(x, H_L)
                right_x = torch.matmul(x, H_R)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
                diagonal_copy_(normalizer, xn, w)

        final_m = torch.cat(final_m, dim=0)
        final_normalizer = torch.cat(final_normalizer, dim=0)
        final = (final_m + 1e-9).squeeze(1).log() + root
        logZ = final.logsumexp(-1) + final_normalizer.squeeze(-1)
        if decode:
            spans = self.mbr_decoding(logZ, span_indicator, lens)
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
        terms = params["term"].detach()
        roots = params["root"].detach()
        H = params["head"].detach()  # (batch, NT, r) r:=rank
        L = params["left"].detach()  # (batch, NT + T, r)
        R = params["right"].detach()  # (batch, NT + T, r)

        terms = terms.softmax(2).cumsum(2)
        roots = roots.softmax(1).cumsum(1)
        H = H.cumsum(2)
        L = L.cumsum(1)
        R = R.cumsum(1)

        terms = terms.cpu().numpy()
        roots = roots.cpu().numpy()
        H = H.cpu().numpy()
        L = L.cpu().numpy()
        R = R.cpu().numpy()

        max_nt_spans = max(len(item) for item in nt_spans)
        max_pt_spans = max(len(item) for item in pt_spans)

        preds = []
        for b in range(len(terms)):
            samples, types = self.sample(
                terms[b],
                H[b],
                L[b],
                R[b],
                roots[b],
                max_nt_spans,
                src_nt_states,
                max_pt_spans,
                src_pt_states,
                use_copy=use_copy,
                num_samples=num_samples,
                max_length=max_length,
            )
            samples = [
                (sample, type_)
                for sample, type_ in zip(samples, types)
                if len(sample) > 1
            ]  # len=0 when max_actions is reached but no PT rules applied
            if len(samples) == 0:
                log.warning("All trials are failed.")
                samples = [([0, 0], [TokenType.VOCAB, TokenType.VOCAB])]
            preds.append(samples)
        return preds

    @staticmethod
    @jit(nopython=True, parallel=True)
    def sample(
        terms: np.ndarray,  # seqlen x pt, in normal space
        rules_head: np.ndarray,  # nt x r, in normal space
        rules_left: np.ndarray,  # (nt+pt) x r, in normal space
        rules_right: np.ndarray,  # (nt+pt) x r, in normal space
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

        for i in prange(num_samples):
            actions = 0
            sample = weighted_random(roots)
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
                    if use_copy:
                        nt_state = s // nt_num_nodes
                        if nt_state == COPY_NT:
                            nt_node = s % nt_num_nodes
                            preterminals.append(nt_node)
                            is_copy_pt.append(True)
                            continue
                    actions += 1
                    head = weighted_random(rules_head[s])
                    left = weighted_random(rules_left[:, head])
                    right = weighted_random(rules_right[:, head])
                    nonterminals.extend([right, left])
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
        return samples, types

    @staticmethod
    def get_pcfg_rules(params, tgt_nt):
        rule = torch.einsum(
            "xar,xbr,xcr->xabc", params["head"], params["left"], params["right"]
        ).log()
        return {"term": params["term"], "rule": rule, "root": params["root"]}


@checkpoint
@torch.jit.script
def merge(y, z, yn, zn):
    """
    :param Y: shape (batch, n, w, r)
    :param Z: shape (batch, n, w, r)
    :return: shape (batch, n, x)
    """
    # contract dimension w.
    y = (y + 1e-9).log() + yn.unsqueeze(-1)
    z = (z + 1e-9).log() + zn.unsqueeze(-1)
    b_n_r = (y + z).logsumexp(-2)
    normalizer = b_n_r.max(-1)[0]
    b_n_r = (b_n_r - normalizer.unsqueeze(-1)).exp()
    return b_n_r, normalizer
