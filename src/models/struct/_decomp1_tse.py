# From https://github.com/sustcsonglin/TN-PCFG/blob/main/parser/pcfgs/tdpcfg.py
# I place it here just for benchmarking.


import torch
import torch_semiring_einsum as tse
from torch.utils.checkpoint import checkpoint as ckp


def checkpoint(func):
    def wrapper(*args, **kwargs):
        return ckp(func, *args, **kwargs)

    return wrapper


def diagonal_copy_(x, y, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        x.as_strided(
            size=(x.shape[0], seq_len - w, *list(x.shape[3:])), stride=new_stride, storage_offset=w * stride[2]
        ).copy_(y)
    else:
        x.as_strided(size=(x.shape[0], seq_len - w), stride=new_stride, storage_offset=w * stride[2]).copy_(y)


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
            size=(x.shape[0], n, w), stride=stride, storage_offset=(offset[0] * seq_len + offset[1]) * numel
        )


def diagonal(x, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        return x.as_strided(
            size=(x.shape[0], seq_len - w, *list(x.shape[3:])), stride=new_stride, storage_offset=w * stride[2]
        )
    else:
        return x.as_strided(size=(x.shape[0], seq_len - w), stride=new_stride, storage_offset=w * stride[2])


class Fastest_TDPCFG_TSE:
    MATMUL = tse.compile_equation("bik,bkj->bij")
    MERGE = tse.compile_equation("bnwr,bnwr->bnr")

    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None):
        assert viterbi is not True
        unary = rules["term"].clone()
        root = rules["root"].clone()

        # 3d binary rule probabilities tensor decomposes to three 2d matrices after CP decomposition.
        H = rules["head"].clone()  # (batch, NT, r) r:=rank
        L = rules["left"].clone()  # (batch, NT+T, r)
        R = rules["right"].clone()  # (batch, NT+T, r)

        T = unary.shape[-1]
        S = L.shape[-2]
        NT = S - T
        # r = L.shape[-1]

        L_term = L[:, NT:, ...].contiguous()
        L_nonterm = L[:, :NT, ...].contiguous()
        R_term = R[:, NT:, ...].contiguous()
        R_nonterm = R[:, :NT, ...].contiguous()

        def transform(x, y):
            """
            :param x: shape (batch, n, T)
            :return: shape (batch, n, r)
            """
            return tse.log_einsum(self.MATMUL, x, y)

        H = H.transpose(-1, -2)
        # (m_A, r_A) + (m_B, r_B) -> (r_A, r_B)
        H_L = transform(H, L_nonterm)
        H_R = transform(H, R_nonterm)

        @checkpoint
        def merge(Y, Z, indicator):
            """
            :param Y: shape (batch, n, w, r)
            :param Z: shape (batch, n, w, r)
            :return: shape (batch, n, x)
            """
            # contract dimension w.

            b_n_r = tse.log_einsum(self.MERGE, Y, Z) + indicator
            return b_n_r

        batch, N, *_ = unary.shape
        N += 1

        # for estimating marginals.
        if s_span is None:
            span_indicator = unary.new_zeros(batch, N, N).requires_grad_(mbr)
        else:
            span_indicator = s_span
            if mbr or viterbi:
                span_indicator = span_indicator.detach().clone().requires_grad_(True)
            unary += diagonal(span_indicator, w=1).unsqueeze(-1)

        left_term = transform(unary, L_term)
        right_term = transform(unary, R_term)

        # for caching V^{T}s_{i,k} and W^{T}s_{k+1,j} as described in paper to decrease complexities.
        left_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)
        right_s = unary.new_zeros(batch, N, N, L.shape[2]).fill_(-1e9)

        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        # w: span width
        for w in range(2, N):
            # n: the number of spans of width w.
            n = N - w
            Y = stripe(left_s, n, w - 1, (0, 1))
            Z = stripe(right_s, n, w - 1, (1, w), 0)

            x = merge(
                Y.clone(),
                Z.clone(),
                span_indicator[:, torch.arange(n), w + torch.arange(n)].unsqueeze(-1),
            )

            if w + 1 < N:
                left_x = transform(x, H_L)
                right_x = transform(x, H_R)
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)

            else:
                final_m = transform(x, H)

        final = final_m.squeeze(1) + root
        logZ = final.logsumexp(-1)

        return logZ
