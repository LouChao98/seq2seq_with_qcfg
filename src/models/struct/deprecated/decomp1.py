import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from numba import jit
from torch import Tensor

from ._fn import diagonal_copy_, stripe
from ._utils import checkpoint, process_param_for_trace, weighted_random_v2
from .decomp_base import _COPY_NT, _COPY_PT, _OK, _REACHLIMIT, _SONMASK, _VOCAB, DecompBase, DecompSamplerBase

log = logging.getLogger(__file__)


class Decomp1(DecompBase):
    # A,i->r r->B,j r->C,k
    #
    # Based on Songlin Yang, Wei Liu and Kewei Tu's work
    # https://github.com/sustcsonglin/TN-PCFG/blob/main/parser/pcfgs/tdpcfg.py
    #
    # Modification:
    # 1. use new grad api
    # 2. remove unnecessary tensor.clone()
    # 3. respect lens

    KEYS = ("term", "head", "left", "right", "root")
    LOGSPACE = (True, False, False, False, True)

    def inside(self, trace) -> Tuple[Tensor, Tensor]:
        if trace:
            params = {k: process_param_for_trace(v) for k, v in self.params.items()}
        else:
            params = self.params

        head: Tensor = params["head"]  # (batch, NT, r), A[i] -> R
        term: Tensor = params["term"]  # (batch, seq_len, PT)
        root: Tensor = params["root"]  # (batch, NT)
        L: Tensor = params["left"]  # (batch, r, NT + T), r->L
        R: Tensor = params["right"]  # (batch, r, NT + T), r->R
        constraint = params.get("constraint")

        batch, N, PT = term.shape
        NT = root.shape[1]
        N += 1

        TLNT = L[..., :NT]
        TLPT = L[..., NT:]
        TRNT = R[..., :NT]
        TRPT = R[..., NT:]

        if trace:
            span_indicator = term.new_ones(batch, N, N, max(NT, PT), requires_grad=True)
            span_indicator_running = span_indicator[..., :NT]
        else:
            span_indicator = None
            span_indicator_running = None

        normalizer = term.new_full((batch, N, N), -1e9)
        norm: Tensor = term.max(-1)[0]
        diagonal_copy_(normalizer, norm, w=1)
        term = (term - norm.unsqueeze(-1)).exp()

        left_s = term.new_full((batch, N, N, L.shape[1]), 0.0)
        right_s = term.new_full((batch, N, N, L.shape[1]), 0.0)
        if trace:
            indicator = span_indicator[..., :PT].diagonal(1, 1, 2).movedim(-1, 1)
            term = term * indicator
        left_term = torch.matmul(term, TLPT.transpose(1, 2))
        right_term = torch.matmul(term, TRPT.transpose(1, 2))
        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (torch.arange(2, N + 1).unsqueeze(1) <= self.lens.cpu().unsqueeze(0)).sum(1)

        # w: span width
        final = []
        final_normalizer = []
        for step, w in enumerate(range(2, N)):

            # n: the number of spans of width w.
            n = N - w
            y = stripe(left_s, n, w - 1, (0, 1))
            z = stripe(right_s, n, w - 1, (1, w), 0)
            yn = stripe(normalizer, n, w - 1, (0, 1))
            zn = stripe(normalizer, n, w - 1, (1, w), 0)
            x, xn = merge(y, z, yn, zn, head)

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
                    head = head[:unfinished]
                    TLNT = TLNT[:unfinished]
                    TRNT = TRNT[:unfinished]
                    normalizer = normalizer[:unfinished]
                    xn = xn[:unfinished]
                    if trace:
                        span_indicator_running = span_indicator_running[:unfinished]
                left_x = torch.matmul(x, TLNT.transpose(1, 2))
                right_x = torch.matmul(x, TRNT.transpose(1, 2))
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
                diagonal_copy_(normalizer, xn, w)

            if unfinished == 0:
                break

        final = torch.cat(final, dim=0)
        final_normalizer = torch.cat(final_normalizer, dim=0)
        final = (final + 1e-9).squeeze(1).log() + root
        logZ = final.logsumexp(-1) + final_normalizer.squeeze(-1)
        return logZ, span_indicator

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


class Decomp1Sampler(DecompSamplerBase):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(2).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        H = params["head"].cumsum(2).cpu().numpy()
        L = params["left"].cumsum(2).cpu().numpy()
        R = params["right"].cumsum(2).cpu().numpy()
        return terms, H, L, R, roots

    @staticmethod
    @jit(nopython=True)
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
            is_copy_pt: List[bool] = []
            actions = 0
            try:
                sample = weighted_random_v2(roots)
                nonterminals.append(sample)

                while len(nonterminals) > 0 and len(preterminals) < max_length and actions < max_actions:
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
                        head = weighted_random_v2(rules_head[s])
                        left = weighted_random_v2(rules_left[head])
                        right = weighted_random_v2(rules_right[head])
                        nonterminals.extend([right, left])
                    else:
                        preterminals.append(s - NT)
                        is_copy_pt.append(False)
            except Exception:
                status[i] = _SONMASK

            if actions == max_actions or (len(preterminals) == max_length and len(nonterminals) > 0):
                status[i] = _REACHLIMIT

            terminals: List[int] = []
            terminal_type: List[int] = []  # 0=vocab, 1=nt span, 2=pt span
            try:
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


@checkpoint
@torch.jit.script
def merge(y, z, yn, zn, h):
    # contract dimension w.
    y = (y + 1e-9).log() + yn.unsqueeze(-1)
    z = (z + 1e-9).log() + zn.unsqueeze(-1)
    b_n_r = (y + z).logsumexp(-2)
    normalizer = b_n_r.max(-1)[0]
    b_n_r = (b_n_r - normalizer.unsqueeze(-1)).exp()
    x = torch.matmul(b_n_r, h.transpose(1, 2))
    return x, normalizer


@torch.jit.script
def set_score(x, xn, mask, value):
    x_real = (x + 1e-9).log() + xn[..., None]
    x_real = torch.where(mask, value, x_real)
    xn = x_real.max(-1)[0]
    x = (x_real - xn[..., None]).exp()
    return x, xn
