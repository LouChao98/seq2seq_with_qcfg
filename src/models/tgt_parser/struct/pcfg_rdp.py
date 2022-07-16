from typing import Dict, List, Union

import numpy as np
import torch
from torch import Tensor
from torch_struct import SentCFG

from . import _torch_model_utils as tmu
from ._utils import checkpoint, weighted_random


class PCFG:
    def __call__(self, params, lens, decode=False, marginal=False):
        # terms: bsz x seqlen x pt
        # rules: bsz x nt x (nt+pt) x (nt+pt)
        # roots: bsz x nt

        terms, rules, roots = params["term"], params["rule"], params["root"]
        logZ = self.forward_approx(terms, rules, roots, lens, marginal, 1, 1)
        # if marginal:
        #     return dist.marginals
        # elif not decode:
        #     return -dist.partition
        # else:
        #     spans_onehot = dist.argmax[-1].cpu().numpy()
        #     tags = dist.argmax[0].max(-1)[1].cpu().numpy()
        #     # lens = lens.cpu().tolist()
        #     all_spans = []
        #     for b in range(len(spans_onehot)):
        #         spans_inst = [(i, i, int(tags[b][i])) for i in range(lens[b])]
        #         for width, left, tag in zip(*spans_onehot[b].nonzero()):
        #             spans_inst.append((left, left + width + 1, tag))
        #         all_spans.append(spans_inst)
        #     return all_spans

    def forward_approx(
        self,
        term: torch.Tensor,
        rule: torch.Tensor,
        root: torch.Tensor,
        lens,
        marginal,
        topk,
        sample_size,
    ):
        batch, N, T = term.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T
        RS = topk + sample_size
        assert RS <= S, "Trival parameters"

        s = term.new_full((batch, N, N, RS), -1e9)
        s_ind = term.new_zeros(batch, N, N, RS, dtype=torch.long)

        NTs = slice(0, NT)
        Ts = slice(NT, S)
        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        if marginal:
            span_indicator = rule.new_zeros(batch, N, N, requires_grad=True)

        for w in range(2, N):
            n = N - w

            Y_term = term[:, :n, :, None]
            Z_term = term[:, w - 1 :, None, :]

            if w == 2:
                score = Xyz(Y_term, Z_term, X_y_z)  # bsz x n x nt

                diagonal_copy_(
                    s, +span_indicator.diagonal(w, 1, 2).unsqueeze(-1), w,
                )
                continue

            n = N - w
            x = term.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Y_ind = stripe(s_ind, n, w - 1, (0, 1))
            Z = stripe(s, n, w - 1, (1, w), 0).clone()
            Z_ind = stripe(s_ind, n, w - 1, (1, w), 0)

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Y_ind, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, Z_ind, X_y_Z))
            x = x.logsumexp(0)

            diagonal_copy_(
                s,
                x.logsumexp(dim=0) + span_indicator.diagonal(w, 1, 2).unsqueeze(-1),
                w,
            )

        logZ = (s[torch.arange(batch), 0, lens] + root).logsumexp(-1)

        return logZ


def sample(score: Tensor, topk: int, sample: int):
    # I use normalized scores p(l|i,j) as the proposal distribution

    # Get topk for exact marginalization
    b, n, c = score.shape
    proposal_p = score.softmax(-1)
    _, topk_ind = torch.topk(proposal_p, topk, dim=-1)
    topk_score = tmu.batch_index_select(
        score.view(b * n, c), topk_ind.view(b * n, -1),
    ).view(b, n, -1)

    # get renormalized proposal distribution
    proposal_p = tmu.batch_index_fill(
        proposal_p.view(b * n, c), topk_ind.view(b * n, -1), 1e-7,
    )
    proposal_p /= proposal_p.sum(-1, keepdim=True)

    # sample from the proposal.
    sampled_ind = torch.multinomial(proposal_p, sample)
    sample_log_prob = tmu.batch_index_select(proposal_p, sampled_ind)
    sample_log_prob = (sample_log_prob + 1e-8).log()
    sampled_score = tmu.batch_index_select(
        score.view(b * n, -1), sampled_ind.view(b * n, -1),
    ).view(b, n, sample)

    #  debias sampled emission
    sampled_ind = sampled_ind.view(b, n, sample)
    sample_log_prob = sample_log_prob.view(b, n, sample)
    correction = sample_log_prob + np.log(sample)
    sampled_score -= correction

    # Combine the emission
    combined_score = torch.cat([topk_score, sampled_score], dim=-1)
    combined_ind = torch.cat([topk_ind, sampled_ind], dim=-1)
    return combined_score, combined_ind


# @checkpoint
def Xyz(y: Tensor, z: Tensor, rule: Tensor):
    b_n_yz = (y + z).view(*y.shape[:2], -1)
    b_n_x = (b_n_yz.unsqueeze(-2) + rule.unsqueeze(1)).logsumexp(-1)
    return b_n_x


# @checkpoint
def XYZ(Y: Tensor, Y_ind: Tensor, Z: Tensor, Z_ind: Tensor, rule: Tensor):
    n = Y.shape[1]
    b_n_yz = (
        (Y[:, :, 1:-1, :].unsqueeze(-1) + Z[:, :, 1:-1, :].unsqueeze(-2))
        .logsumexp(2)
        .reshape(batch, n, -1)
    )
    b_n_x = (b_n_yz.unsqueeze(2) + rule.unsqueeze(1)).logsumexp(-1)
    return b_n_x


# @checkpoint
def XYz(Y: Tensor, Y_ind: Tensor, z: Tensor, rule: Tensor):
    n = Y.shape[1]
    Y = Y[:, :, -1, :, None]
    b_n_yz = (Y + z).reshape(batch, n, NT * T)
    b_n_x = (b_n_yz.unsqueeze(-2) + rule.unsqueeze(1)).logsumexp(-1)
    return b_n_x


# @checkpoint
def XyZ(y: Tensor, Z: Tensor, Z_ind: Tensor, rule: Tensor):
    n = Z.shape[1]
    Z = Z[:, :, 0, None, :]
    b_n_yz = (y + Z).reshape(batch, n, NT * T)
    b_n_x = (b_n_yz.unsqueeze(-2) + rule.unsqueeze(1)).logsumexp(-1)
    return b_n_x


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


if __name__ == "__main__":
    torch.random.manual_seed(1)

    B, N, T, NT = 4, 5, 3, 7
    device = "cpu"
    params = {
        "term": torch.randn(B, N, T, device=device)
        .log_softmax(-1)
        .requires_grad_(True),
        "root": torch.randn(B, NT, device=device).log_softmax(-1).requires_grad_(True),
        "rule": torch.randn(B, NT, (NT + T) ** 2, device=device)
        .log_softmax(-1)
        .view(B, NT, NT + T, NT + T)
        .requires_grad_(True),
    }
    lens = torch.tensor([N, N - 1, N - 1, N - 3], dtype=torch.long, device=device)

    pcfg = PCFG()
    print(pcfg(params, lens))

