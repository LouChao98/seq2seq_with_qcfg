import logging
from collections import Counter
from copy import copy
from functools import partial

import torch
from torch import Tensor

from ._utils import check_full_marginal, checkpoint, compare_marginal, compute_unnormalized_prob, enumerate_seq
from .base import DecompBase

log = logging.getLogger(__file__)


class NoDecompDepth(DecompBase):
    """
    term: b n tgt_pt src_pt
    rule: b nt nt+pt nt+pt
    root: b tgt_ntt src_nt
    NOTE: span_id start from 1
    """

    KEYS = ("term", "rule", "root")
    LOGSPACE = (True, True, True)

    def inside(self, params, semiring, trace=False, use_reentrant=True):
        params = self.preprocess(params, semiring)
        merge = checkpoint(partial(g_merge, semiring=semiring), use_reentrant=use_reentrant)
        merge2 = checkpoint(partial(g_merge2, semiring=semiring), use_reentrant=use_reentrant)
        merge3 = checkpoint(partial(g_merge3, semiring=semiring), use_reentrant=use_reentrant)

        term: Tensor = params["term"]
        rule: Tensor = params["rule"]
        root: Tensor = params["root"]
        constraint = params.get("constraint")
        lse_scores = params.get("lse")
        add_scores = params.get("add")

        bsz = self.batch_size
        N = term.shape[2] + 1
        NT = self.nt_states * self.nt_num_nodes

        NTNT = rule[..., :NT, :NT]
        NTPT = rule[..., :NT, NT:]
        PTNT = rule[..., NT:, :NT]
        PTPT = rule[..., NT:, NT:]

        term = term.view(term.shape[0], bsz, term.shape[2], 1, self.pt_states * self.pt_num_nodes)

        _span_indicator = term.new_zeros(bsz, N, N, term.shape[2], requires_grad=True)
        span_indicator = _span_indicator.view(1, bsz, N, N, term.shape[2])
        span_indicator_running = span_indicator

        s = semiring.new_zeros((bsz, N, N, term.shape[2], NT))

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (torch.arange(2, N + 1).unsqueeze(1) <= self.lens.cpu().unsqueeze(0)).sum(1)

        # w: span width
        final = []
        for step, w in enumerate(range(2, N)):
            # n: the number of spans of width w.
            n = N - w
            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            # s, b, n, 1, n, pt
            y_term = term[:, :, : N - w].unsqueeze(3)
            z_term = term[:, :, w - 1 :].unsqueeze(3)

            if w == 2:
                x = merge(y_term, z_term, PTPT)
            else:
                y = stripe_left(s, current_bsz, n, w - 2, w, (0, 2)).clone()
                z = stripe_right(s, current_bsz, n, w - 2, w, (1, w)).clone()
                if w == 3:
                    x = merge2(y, z, y_term, z_term, PTNT, NTPT)
                else:
                    x = merge3(y, z, y_term, z_term, PTNT, NTPT, NTNT)

            if constraint is not None:
                value, mask = constraint[step]
                if value.ndim > 0:
                    value = value[:, :current_bsz]
                mask = mask[:current_bsz]
                x = torch.where(mask.unsqueeze(0).expand([semiring.size] + list(mask.shape)), value, x)

            if add_scores is not None:
                x = x + add_scores[step]

            if lse_scores is not None:
                x = torch.logaddexp(x, lse_scores[step])

            if trace:
                indicator = span_indicator_running[:, :, :, :, w - 1 :].diagonal(w, 2, 3).movedim(-1, 2)
                x = x + indicator.unsqueeze(-1)

            if current_bsz - unfinished > 0:
                final.insert(0, x[:, unfinished:current_bsz, :1, 0])

            if unfinished > 0:
                if current_bsz > unfinished:
                    x = x[:, :unfinished]
                    term = term[:, :unfinished]
                    PTPT = PTPT[:, :unfinished]
                    PTNT = PTNT[:, :unfinished]
                    NTPT = NTPT[:, :unfinished]
                    NTNT = NTNT[:, :unfinished]
                    if trace:
                        span_indicator_running = span_indicator_running[:, :unfinished]
                diagonal_copy_with_id_(s, x, unfinished, w)

            if unfinished == 0:
                break

        final = torch.cat(final, dim=1)
        final = semiring.mul(final.squeeze(2), root)
        logZ = semiring.sum(final, dim=-1)
        logZ = semiring.unconvert(logZ)
        return logZ, _span_indicator

    @staticmethod
    def random(bsz, max_len, tgt_pt, src_pt, tgt_nt, src_nt, r):
        nt = tgt_nt * src_nt
        pt = tgt_pt * src_pt
        src = src_nt + src_pt
        return {
            "term": torch.rand(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.rand(bsz, nt).log_softmax(-1).requires_grad_(True),
            "rule": torch.rand(bsz, nt, nt + pt, nt + pt)
            .flatten(2)
            .log_softmax(-1)
            .view(bsz, nt, nt + pt, nt + pt)
            .requires_grad_(True),
        }


def g_merge(y, z, ptpt, semiring):
    # y: c, bsz, n, 1(w), 1(id), pt
    # z: c, bsz, n, 1(w), 1(id), pt
    # ptpt: c, bsz, nt, pt, pt

    # c, bsz, n, 1, pt, NEW * c, bsz, n, 1, NEW, pt -> c, bsz, n, 1(id), pt, pt
    x = semiring.mul(y.unsqueeze(6), z.unsqueeze(5)).squeeze(3)
    # c, bsz, n, 1(id), NEW(nt), pt, pt * c, bsz, NEW, NEW, nt, pt, pt -> c, bsz, n, nt, pt, pt
    x = semiring.mul(x.unsqueeze(4), ptpt[:, :, None, None])
    # c, bsz, n, nt, 1(id), (pt, pt) -> c, bsz, n, nt
    x = semiring.sum(x.flatten(5), dim=5)
    return x


def g_merge2(y, z, y_term, z_term, ptnt, ntpt, semiring):
    # y: c, bsz, n, 1(w), 1(id), nt
    # z: c, bsz, n, 1(w), 1(id), nt
    # y_term: c, bsz, n, 1(w), 1(id), pt
    # z_term: c, bsz, n, 1(w), 1(id), pt
    # ptnt: c, bsz, nt, pt, nt
    # ntpt: c, bsz, nt, nt, pt

    x1 = semiring.mul(y.unsqueeze(6), z_term.unsqueeze(5)).squeeze(3)
    x1 = semiring.mul(x1.unsqueeze(4), ntpt[:, :, None, None])
    x1 = semiring.sum(x1.flatten(5), dim=5)

    x3 = semiring.mul(y_term.unsqueeze(6), z.unsqueeze(5)).squeeze(3)
    x3 = semiring.mul(x3.unsqueeze(4), ptnt[:, :, None, None])
    x3 = semiring.sum(x3.flatten(5), dim=5)

    return semiring.add(x1, x3)


def g_merge3(y, z, y_term, z_term, ptnt, ntpt, ntnt, semiring):
    # y: c, bsz, n, 1, nt
    # z: c, bsz, n, 1, nt
    # y_term: c, bsz, n, 1, pt
    # z_term: c, bsz, n, 1, pt
    # ptnt: c, bsz, nt, pt, nt
    # ntpt: c, bsz, nt, nt, pt
    # ntnt: c, bsz, nt, nt, nt
    x1 = semiring.mul(y[:, :, :, -1:].unsqueeze(6), z_term.unsqueeze(5)).squeeze(3)
    x1 = semiring.mul(x1.unsqueeze(4), ntpt[:, :, None, None])
    x1 = semiring.sum(x1.flatten(5), dim=5)

    x2 = semiring.mul(y[:, :, :, :-1].unsqueeze(6), z[:, :, :, 1:].unsqueeze(5))
    x2 = semiring.sum(x2, dim=3)
    x2 = semiring.mul(x2.unsqueeze(4), ntnt[:, :, None, None])
    x2 = semiring.sum(x2.flatten(5), dim=5)

    x3 = semiring.mul(y_term.unsqueeze(6), z[:, :, :, :1].unsqueeze(5)).squeeze(3)
    x3 = semiring.mul(x3.unsqueeze(4), ptnt[:, :, None, None])
    x3 = semiring.sum(x3.flatten(5), dim=5)
    return semiring.add(x1, x2, x3)


def diagonal_copy_with_id_(x: torch.Tensor, y: torch.Tensor, b: int, w: int):
    assert x.is_contiguous()
    seq_len, n_pos = x.size(3), x.size(4)
    stride = list(x.stride())
    new_stride = [stride[0], stride[1]]
    new_stride.append(stride[2] + stride[3])
    new_stride.extend(stride[4:])
    num = n_pos - w + 1
    x.as_strided(
        size=(x.shape[0], b, seq_len - w, num, *list(x.shape[5:])),
        stride=new_stride,
        storage_offset=w * stride[3] + (w - 1) * stride[4],
    ).copy_(y)


def stripe_left(x: torch.Tensor, b: int, n: int, w: int, tgt_w: int, offset=(0, 1)):
    # x: semiring, batch, n, n, n, ...
    assert x.is_contiguous()
    seq_len, n_pos = x.size(3), x.size(4)
    stride = list(x.stride())
    numel, numel2 = stride[3], stride[4]
    stride[2] = (seq_len + 1) * numel
    stride[3] = numel + numel2
    num = n_pos - tgt_w + 1
    return x.as_strided(
        size=(x.shape[0], b, n, w, num, *list(x.shape[5:])),
        stride=stride,
        storage_offset=(offset[0] * seq_len + offset[1]) * numel + numel2 * (offset[1] - 1),
    )


def stripe_right(x: torch.Tensor, b: int, n: int, w: int, tgt_w: int, offset=(0, 0)):
    assert x.is_contiguous()
    seq_len, n_pos = x.size(3), x.size(4)
    stride = list(x.stride())
    numel, numel2 = stride[3], stride[4]
    stride[2] = (seq_len + 1) * numel
    stride[3] = seq_len * numel
    num = n_pos - tgt_w + 1
    return x.as_strided(
        size=(x.shape[0], b, n, w, num, *list(x.shape[5:])),
        stride=stride,
        storage_offset=(offset[0] * seq_len + offset[1]) * numel + numel2 * (tgt_w - 2),
    )


if __name__ == "__main__":

    import pytorch_lightning as pl
    from torch.autograd import grad

    from src.models.tgt_parser.struct3.semiring import GumbelCRFSemiring, MaxSemiring
    from src.models.tgt_parser.struct.pcfg import PCFG

    pl.seed_everything(1)

    # B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 1, 4, 2, 2, 2, 2
    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 2, 6, 2, 5, 3, 7
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    r = 3
    lens = [max(2, N - i) for i in range(B)]
    params = NoDecompDepth.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }

    pcfg = NoDecompDepth(params, lens, **meta)
    pcfg_ref = PCFG()

    print("test nll")
    nll = pcfg.nll
    nll_ref = pcfg_ref(params, lens)
    assert torch.allclose(nll, nll_ref), (nll, nll_ref)

    print("test marginal")
    m1 = pcfg.marginal

    print("test gumbel")
    max_logp, trace = pcfg.inside(pcfg.params, GumbelCRFSemiring(1.0), True, use_reentrant=False)
    mtrace = grad(max_logp.sum(), [trace], create_graph=True)[0]
    mtrace[0][0, 2, 1].sum().backward()

    print("test argmax")
    max_logp, trace = pcfg.inside(pcfg.params, MaxSemiring, True)
    max_logp.sum().backward()
    print(trace.grad[1].nonzero())
    print([(item[0], item[1] + 1) for item in pcfg_ref(params, lens, decode=True)[1] if item[1] > item[0]])
