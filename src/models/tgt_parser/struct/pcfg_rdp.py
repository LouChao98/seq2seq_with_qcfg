from typing import Dict, List, Union

import numpy as np
import torch
from torch import Tensor
import math

class PCFG:
    def __call__(self, params, lens, decode=False, marginal=False):
        # terms: bsz x seqlen x pt
        # rules: bsz x nt x (nt+pt) x (nt+pt)
        # roots: bsz x nt

        terms, rules, roots = params["term"], params["rule"], params["root"]
        logZ = self.forward_approx(terms, rules, roots, lens, marginal, 3, 3)
        if marginal:
            return dist.marginals
        elif not decode:
            return -logZ
        else:
            spans_onehot = dist.argmax[-1].cpu().numpy()
            tags = dist.argmax[0].max(-1)[1].cpu().numpy()
            # lens = lens.cpu().tolist()
            all_spans = []
            for b in range(len(spans_onehot)):
                spans_inst = [(i, i, int(tags[b][i])) for i in range(lens[b])]
                for width, left, tag in zip(*spans_onehot[b].nonzero()):
                    spans_inst.append((left, left + width + 1, tag))
                all_spans.append(spans_inst)
            return all_spans

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
        rXYZ = rule[:, :, NTs, NTs].contiguous()
        rXyZ = rule[:, :, Ts, NTs].contiguous()
        rXYz = rule[:, :, NTs, Ts].contiguous()
        rXyz = rule[:, :, Ts, Ts].contiguous()

        span_indicator = rule.new_zeros(batch, N, N, requires_grad=marginal)

        for w in range(2, N):
            n = N - w

            Y_term = term[:, :n, :, None]
            Z_term = term[:, w - 1 :, None, :]
            indicator = span_indicator.diagonal(w, 1, 2).unsqueeze(-1)

            if w == 2:
                score = Xyz(Y_term, Z_term, rXyz)
                score, ind = sample(score, topk, sample_size)
                diagonal_copy_(s, score + indicator, w)
                diagonal_copy_(s_ind, ind, w)
                continue

            x = term.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Y_ind = stripe(s_ind, n, w - 1, (0, 1))
            Z = stripe(s, n, w - 1, (1, w), 0).clone()
            Z_ind = stripe(s_ind, n, w - 1, (1, w), 0)

            if w > 3:
                x[0].copy_(XYZ(Y, Y_ind, Z, Z_ind, rXYZ))

            x[1].copy_(XYz(Y, Y_ind, Z_term, rXYz))
            x[2].copy_(XyZ(Y_term, Z, Z_ind, rXyZ))
            x = x.logsumexp(0)

            x, ind = sample(x, topk, sample_size)

            diagonal_copy_(s, x + indicator, w)
            diagonal_copy_(s_ind, ind, w)

        b_ind = torch.arange(batch, device=root.device)
        lens = lens.to(device=root.device)
        root_ind = s_ind[b_ind, 0, lens]
        root = root.gather(1, root_ind)
        logZ = (s[b_ind, 0, lens] + root).logsumexp(-1)

        return logZ


def sample(score: Tensor, topk: int, sample: int):
    # I use normalized scores p(l|i,j) as the proposal distribution

    # Get topk for exact marginalization
    b, n, c = score.shape
    proposal_p = score.softmax(-1)
    _, topk_ind = torch.topk(proposal_p, topk, dim=-1, sorted=False)
    topk_score = score.gather(-1, topk_ind)

    if sample == 0:
        return topk_score, topk_ind

    # get renormalized proposal distribution
    b_ind = torch.arange(b, device=score.device)
    b_ind = b_ind.unsqueeze(-1).expand(b, n * topk).flatten()
    n_ind = torch.arange(n, device=score.device)
    n_ind = n_ind.view(1, n, 1).expand(b, n, topk).flatten()
    proposal_p[b_ind, n_ind, topk_ind.flatten()] = 1e-6
    proposal_p /= proposal_p.sum(-1, keepdim=True) 

    # sample from the proposal.
    sampled_ind = torch.multinomial(proposal_p.view(-1, c), sample).view(b, n, sample)
    sample_log_prob = proposal_p.gather(-1, sampled_ind)
    sample_log_prob = (sample_log_prob + 1e-8).log()
    sampled_score = score.gather(-1, sampled_ind)

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
    b, n, t, _ = y.shape
    b_n_yz = (y + z).view(*y.shape[:2], -1)
    b_n_x = (b_n_yz.unsqueeze(-2) + rule.view(b, 1, -1, t * t)).logsumexp(-1)
    return b_n_x


# @checkpoint
def XYZ(Y: Tensor, Y_ind: Tensor, Z: Tensor, Z_ind: Tensor, rule: Tensor):
    b, n, w, nt = Y.shape
    w -= 2
    NT = rule.shape[1]
    Y_ind = Y_ind[:, :, 1:-1, None, :, None].expand(-1, -1, -1, NT, -1, NT)
    Z_ind = Z_ind[:, :, 1:-1, None, None, :].expand(-1, -1, -1, NT, nt, -1)
    rule = rule.view(b, 1, 1, NT, NT, NT).expand(-1, n, w, -1, -1, -1)
    rule = rule.gather(4, Y_ind).gather(5, Z_ind)

    b_n_yz = (Y[:, :, 1:-1, :].unsqueeze(-1) + Z[:, :, 1:-1, :].unsqueeze(-2))
    b_n_x = (b_n_yz.unsqueeze(3) + rule).logsumexp((2, 4, 5))
    return b_n_x


# @checkpoint
def XYz(Y: Tensor, Y_ind: Tensor, z: Tensor, rule: Tensor):
    b, n, _, nt = Y.shape
    t = z.shape[-1]
    Y = Y[:, :, -1, :, None]
    Y_ind = (
        Y_ind[:, :, -1]
        .view(b, n, 1, -1, 1)
        .expand(-1, -1, rule.shape[1], -1, rule.shape[3])
    )
    rule = rule.unsqueeze(1).expand(-1, n, -1, -1, -1)
    rule = rule.gather(3, Y_ind).flatten(3)
    b_n_yz = (Y + z).reshape(b, n, nt * t)
    b_n_x = (b_n_yz.unsqueeze(-2) + rule).logsumexp(-1)
    return b_n_x


# @checkpoint
def XyZ(y: Tensor, Z: Tensor, Z_ind: Tensor, rule: Tensor):
    b, n, _, nt = Z.shape
    t = y.shape[-2]
    Z = Z[:, :, 0, None, :]
    Z_ind = (
        Z_ind[:, :, 0]
        .view(b, n, 1, 1, -1)
        .expand(-1, -1, rule.shape[1], rule.shape[2], -1)
    )
    rule = rule.unsqueeze(1).expand(-1, n, -1, -1, -1)
    rule = rule.gather(4, Z_ind).flatten(3)
    b_n_yz = (y + Z).reshape(b, n, nt * t)
    b_n_x = (b_n_yz.unsqueeze(-2) + rule).logsumexp(-1)
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
    out = torch.zeros(B)
    print(pcfg(params, lens))
    for _ in range(1000):
        out += pcfg(params, lens)
    print(out / 1000)

    from .pcfg import PCFG as PCFG_ref
    pcfg = PCFG_ref()
    print(pcfg(params, lens))

