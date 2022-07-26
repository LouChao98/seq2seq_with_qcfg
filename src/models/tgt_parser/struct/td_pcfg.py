from enum import IntEnum
from typing import Dict, List, Union

import numpy as np
import torch
from numba import jit, prange
from torch import Tensor
from torch.autograd import grad

from ._utils import checkpoint, weighted_random

_VOCAB, _COPY_NT, _COPY_PT = 0, 1, 2


class TokenType(IntEnum):
    VOCAB = _VOCAB
    COPY_NT = _COPY_NT
    COPY_PT = _COPY_PT


class FastestTDPCFG:
    # based on Songlin Yang, Wei Liu and Kewei Tu's work
    # https://github.com/sustcsonglin/TN-PCFG/blob/main/parser/pcfgs/tdpcfg.py
    # modification:
    # 1. use new grad api
    # 2. remove unecessary tensor.clone()
    # 3. respect lens

    # sampling: as the code generate samples one by one. I just recover the rules. Memory should be ok.

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
        lens = torch.tensor(lens)
        assert (
            lens[1:] <= lens[:-1]
        ).all(), "You should sort samples by length descently."

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

        normalizer = terms.new_full((batch, N, N), -1e9)
        norm = terms.max(-1)[0]
        diagonal_copy_(normalizer, norm, w=1)
        terms = (terms - norm.unsqueeze(-1)).exp()
        left_term = torch.matmul(terms, L_term)
        right_term = torch.matmul(terms, R_term)

        # for caching V^{T}s_{i,k} and W^{T}s_{k+1,j}
        left_s = terms.new_full((batch, N, N, L.shape[2]), -1e9)
        right_s = terms.new_full((batch, N, N, L.shape[2]), -1e9)

        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        if marginal:
            span_indicator = terms.new_zeros(batch, N, N).requires_grad_()
            span_indicator_running = span_indicator[:]
        else:
            span_indicator = None

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
            Y = stripe(left_s, n, w - 1, (0, 1))
            Z = stripe(right_s, n, w - 1, (1, w), 0)
            Y_normalizer = stripe(normalizer, n, w - 1, (0, 1))
            Z_normalizer = stripe(normalizer, n, w - 1, (1, w), 0)
            if marginal:
                x, x_normalizer = merge_with_indicator(
                    Y,
                    Z,
                    Y_normalizer,
                    Z_normalizer,
                    span_indicator_running.diagonal(w, 1, 2).unsqueeze(-1),
                )
            else:
                x, x_normalizer = merge_without_indicator(
                    Y, Z, Y_normalizer, Z_normalizer
                )

            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            if current_bsz - unfinished > 0:
                final_m.insert(
                    0,
                    torch.matmul(
                        x[unfinished:current_bsz, :1], H[unfinished:current_bsz]
                    ),
                )
                final_normalizer.insert(0, x_normalizer[unfinished:current_bsz, :1])
            if unfinished > 0:
                x = x[:unfinished]
                left_s = left_s[:unfinished]
                right_s = right_s[:unfinished]
                H_L = H_L[:unfinished]
                H_R = H_R[:unfinished]
                normalizer = normalizer[:unfinished]
                x_normalizer = x_normalizer[:unfinished]
                if marginal:
                    span_indicator_running = span_indicator_running[:unfinished]

                left_x = torch.matmul(x, H_L)
                right_x = torch.matmul(x, H_R)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)
                diagonal_copy_(normalizer, x_normalizer, w)

        final_m = torch.cat(final_m, dim=0)
        final_normalizer = torch.cat(final_normalizer, dim=0)
        final = (final_m + 1e-9).squeeze(1).log() + root
        logZ = final.logsumexp(-1) + final_normalizer.squeeze(-1)
        if decode:
            spans = self.get_prediction(logZ, span_indicator, lens)
            spans = [[(span[0], span[1] - 1, 0) for span in inst] for inst in spans]
            return spans
            # trees = []
            # for spans_inst, l in zip(spans, lens.tolist()):
            #     tree = self.convert_to_tree(spans_inst, l)
            #     trees.append(tree)
            # return trees, spans
        if marginal:
            torch.set_grad_enabled(grad_state)
            return grad(logZ.sum(), [span_indicator])[0]
        return -logZ

    @torch.no_grad()
    def sampled_decoding(
        self,
        params,
        nt_spans,
        src_nt_states,
        pt_spans,
        src_pt_states,
        use_copy=True,
        num_samples=10,
        max_length=100,
    ):
        terms: torch.Tensor = params["term"].detach()
        roots: torch.Tensor = params["root"].detach()
        H: torch.Tensor = params["head"].detach()  # (batch, NT, r) r:=rank
        L: torch.Tensor = params["left"].detach()  # (batch, NT + T, r)
        R: torch.Tensor = params["right"].detach()  # (batch, NT + T, r)

        zero = terms.new_full((1,), -1e9)
        threshold = terms.new_full((1,), np.log(1e-2))
        terms = torch.where(terms > threshold, terms, zero).softmax(2).cumsum(2)
        roots = torch.where(roots > threshold, roots, zero).softmax(1).cumsum(1)

        zero = terms.new_full((1,), 0.0)
        threshold = terms.new_full((1,), 1e-2)
        H = torch.where(H > threshold, H, zero)
        H /= H.sum(2)
        H = H.cumsum(2)
        L = torch.where(L > threshold, L, zero)
        L /= L.sum(1)
        L = L.cumsum(1)
        R = torch.where(R > threshold, R, zero)
        R /= R.sum(1)
        R = R.cumsum(1)

        terms[:, :, -1] += 1  # avoid out of bound
        roots[:, -1] += 1
        H[:, :, -1] += 1
        L[:, -1] += 1
        R[:, -1] += 1

        terms = terms.cpu().numpy()
        roots = roots.cpu().numpy()
        H = H.cpu().numpy()
        L = L.cpu().numpy()
        R = R.cpu().numpy()

        preds = []
        for b in range(len(terms)):
            samples, types, scores = self.sample(
                terms[b],
                H[b],
                L[b],
                R[b],
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
            ]
            preds.append(sample_scores)
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
        scores = [0.0 for _ in range(num_samples)]

        for i in prange(num_samples):
            actions = 0
            sample = weighted_random(roots)
            score = roots[sample]
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
                    score += (
                        rules_head[s, head]
                        + rules_left[left, head]
                        + rules_right[right, head]
                    )
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
                        score += terms[s, sample]
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
            scores[i] = score / len(terminals)
        return samples, types, scores

    def get_prediction(self, logZ, span_indicator, lens):
        batch, seq_len = span_indicator.shape[:2]
        # to avoid some trivial corner cases.
        if seq_len >= 3:
            marginals = grad(logZ.sum(), [span_indicator])[0].detach()
            return self._cky_zero_order(marginals.detach(), lens)
        else:
            # minimal length is 2
            predictions = [[(0, 0), (1, 1), (0, 1)] for _ in range(batch)]
            return predictions

    @torch.no_grad()
    def _cky_zero_order(self, marginals, lens):
        N = marginals.shape[-1]
        s = marginals.new_zeros(*marginals.shape).fill_(-1e9)
        p = marginals.new_zeros(*marginals.shape).long()
        diagonal_copy_(s, diagonal(marginals, 1), 1)
        for w in range(2, N):
            n = N - w
            starts = p.new_tensor(range(n))
            if w != 2:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            else:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            X, split = (Y + Z).max(2)
            x = X + diagonal(marginals, w)
            diagonal_copy_(s, x, w)
            diagonal_copy_(p, split + starts.unsqueeze(0) + 1, w)

        def backtrack(p, i, j):
            if j == i + 1:
                return [(i, j)]
            split = p[i][j]
            ltree = backtrack(p, i, split)
            rtree = backtrack(p, split, j)
            return [(i, j)] + ltree + rtree

        p = p.tolist()
        # lens = lens.tolist()
        spans = [backtrack(p[i], 0, length) for i, length in enumerate(lens)]
        for spans_inst in spans:
            spans_inst.sort(key=lambda x: x[1] - x[0])
        return spans

    def convert_to_tree(self, spans, length):
        tree = [(i, str(i)) for i in range(length)]
        tree = dict(tree)
        for l, r, _ in spans:
            if l != r:
                span = "({} {})".format(tree[l], tree[r])
                tree[r] = tree[l] = span
        return tree[0]


@checkpoint
def merge_with_indicator(Y, Z, y, z, indicator):
    Y = (Y + 1e-9).log() + y.unsqueeze(-1)
    Z = (Z + 1e-9).log() + z.unsqueeze(-1)
    b_n_r = (Y + Z).logsumexp(-2) + indicator
    normalizer = b_n_r.max(-1)[0]
    b_n_r = (b_n_r - normalizer.unsqueeze(-1)).exp()
    return b_n_r, normalizer


@checkpoint
def merge_without_indicator(Y, Z, y, z):
    """
    :param Y: shape (batch, n, w, r)
    :param Z: shape (batch, n, w, r)
    :return: shape (batch, n, x)
    """
    # contract dimension w.
    Y = (Y + 1e-9).log() + y.unsqueeze(-1)
    Z = (Z + 1e-9).log() + z.unsqueeze(-1)
    b_n_r = (Y + Z).logsumexp(-2)
    normalizer = b_n_r.max(-1)[0]
    b_n_r = (b_n_r - normalizer.unsqueeze(-1)).exp()
    return b_n_r, normalizer


def diagonal_copy_(x, y, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        x.as_strided(
            size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
            stride=new_stride,
            storage_offset=w * stride[2],
        ).copy_(y)
    else:
        x.as_strided(
            size=(x.shape[0], seq_len - w),
            stride=new_stride,
            storage_offset=w * stride[2],
        ).copy_(y)


def diagonal(x, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        return x.as_strided(
            size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
            stride=new_stride,
            storage_offset=w * stride[2],
        )
    else:
        return x.as_strided(
            size=(x.shape[0], seq_len - w),
            stride=new_stride,
            storage_offset=w * stride[2],
        )


def stripe(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        return x.as_strided(
            size=(x.shape[0], n, w, *list(x.shape[3:])),
            stride=stride,
            storage_offset=(offset[0] * seq_len + offset[1]) * numel,
        )
    else:
        return x.as_strided(
            size=(x.shape[0], n, w),
            stride=stride,
            storage_offset=(offset[0] * seq_len + offset[1]) * numel,
        )


if __name__ == "__main__":
    from .pcfg import PCFG

    torch.random.manual_seed(1)

    B, N, T, NT, r = 4, 5, 3, 7, 2
    device = "cpu"
    params = {
        "term": torch.randn(B, N, T, device=device)
        .log_softmax(-1)
        .requires_grad_(True),
        "root": torch.randn(B, NT, device=device).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r, device=device).softmax(-1).requires_grad_(True),
        "left": torch.randn(B, NT + T, r, device=device)
        .softmax(-2)
        .requires_grad_(True),
        "right": torch.randn(B, NT + T, r, device=device)
        .softmax(-2)
        .requires_grad_(True),
    }
    lens = torch.tensor([N, N - 1, N - 1, N - 3], dtype=torch.long, device=device)

    # pcfg = PCFG()

    # print(pcfg(params, lens))
    # print(pcfg(params, lens, decode=True))

    cfg = FastestTDPCFG()

    logZ = cfg(params, lens)
    print(logZ)

