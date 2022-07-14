from typing import Dict, Union, List
import torch
from torch_struct import SentCFG

import numpy as np
from ._utils import weighted_random, checkpoint


class PCFG:
    def __call__(self, params, lens, decode=False, marginal=False):
        # terms: bsz x seqlen x pt
        # rules: bsz x nt x (nt+pt) x (nt+pt)
        # roots: bsz x nt

        terms, rules, roots = params["term"], params["rule"], params["root"]
        dist = SentCFG((terms, rules, roots), lens)
        if marginal:
            return dist.marginals
        elif not decode:
            return -dist.partition
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
        self, term: torch.Tensor, rule: torch.Tensor, root: torch.Tensor, lens, marginal, topk, sample_size
    ):
        batch, N, T = term.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T
        RS = topk + sample_size
        assert RS <= S, "Trival parameters"

        s = term.new_zeros(batch, N, N, NT).fill_(-1e9)
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
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z)
                    + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(
                        -1
                    ),
                    w,
                )
                continue

            n = N - w
            x = term.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            diagonal_copy_(
                s,
                x.logsumexp(dim=0)
                + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1),
                w,
            )

        logZ = (s[torch.arange(batch), 0, lens] + root).logsumexp(-1)

        if viterbi or mbr:
            prediction = self._get_prediction(logZ, span_indicator, lens, mbr=mbr)
            return {"partition": logZ, "prediction": prediction}
        else:
            return {"partition": logZ}


@checkpoint
def Xyz(y, z, rule):
    n = y.shape[1]
    b_n_yz = (y + z).reshape(batch, n, T * T)
    b_n_x = (b_n_yz.unsqueeze(-2) + rule.unsqueeze(1)).logsumexp(-1)
    return b_n_x

@checkpoint
def XYZ(Y, Z, rule):
    n = Y.shape[1]
    b_n_yz = (
        (Y[:, :, 1:-1, :].unsqueeze(-1) + Z[:, :, 1:-1, :].unsqueeze(-2))
        .logsumexp(2)
        .reshape(batch, n, -1)
    )
    b_n_x = (b_n_yz.unsqueeze(2) + rule.unsqueeze(1)).logsumexp(-1)
    return b_n_x

@checkpoint
def XYz(Y, z, rule):
    n = Y.shape[1]
    Y = Y[:, :, -1, :, None]
    b_n_yz = (Y + z).reshape(batch, n, NT * T)
    b_n_x = (b_n_yz.unsqueeze(-2) + rule.unsqueeze(1)).logsumexp(-1)
    return b_n_x

@checkpoint
def XyZ(y, Z, rule):
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
    from time import time

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

    pcfg = PCFG()

    # t = time()
    # print(pcfg(params, lens))
    print(pcfg(params, lens, decode=True))
    # print(time() - t)

    # cfg = FastestTDPCFG()
    # t = time()
    # logZ = cfg(params, lens)
    # print(logZ)
    # print(time() - t)
    # exit(0)

    # print(cfg(params, lens, argmax=True))
    # # logZ.sum().backward()
    # # print(params["head"].grad[0])

    # # mrg = cfg._inside(params, lens, marginal=True)
    # # print(mrg[0].sum())

    # for k in params.keys():
    #     params[k] = params[k].detach().clone()
    # cfg = PCFG()
    # params = (
    #     params["term"],
    #     torch.einsum(
    #         "bxr,byr,bzr->bxyz", params["head"], params["left"], params["right"]
    #     ).log(),
    #     params["root"],
    # )
    # logZ = -cfg(params, lens)
    # print(logZ)
    # cfg(params, lens, argmax=True)
    # # logZ.sum().backward()
    # # print(params['head'].grad[0])

    # print("Ok if same")

