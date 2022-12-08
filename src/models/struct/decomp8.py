import logging
from collections import Counter
from copy import copy
from enum import IntEnum
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import torch
from numba import jit
from torch import Tensor

from ._fn import diagonal_copy_, stripe
from ._utils import (
    check_full_marginal,
    checkpoint,
    compare_marginal,
    compute_unnormalized_prob,
    enumerate_seq,
    weighted_random_v2,
)
from .base import _COPY_NT, _COPY_PT, _OK, _REACHLIMIT, _SONMASK, _VOCAB, DecompBase, DecompSamplerBase

log = logging.getLogger(__file__)


class TokenType(IntEnum):
    VOCAB = _VOCAB
    COPY_NT = _COPY_NT
    COPY_PT = _COPY_PT


class Decomp8(DecompBase):
    # A[i] -> B[j], C[k]
    # ----------------
    # A -> BC
    # B,i -> j  Folded
    # C,i -> k  Folded
    # ================
    # D[i] -> w
    # ----------------
    # D,i -> w
    # ================
    # S -> A[i]
    # ----------------
    # S -> A
    # S -> i, NOTE that this is a trival dist because `i` must be the root span

    KEYS = ("term", "rule", "align_left", "align_right", "root")
    LOGSPACE = (True, True, True, True, True, True)

    def inside(self, params, semiring, trace=False, use_reentrant=True) -> Tuple[Tensor, Tensor]:
        params = self.preprocess(params, semiring)
        merge_h = checkpoint(partial(g_merge, semiring=semiring), use_reentrant=use_reentrant)
        merge_h2 = checkpoint(partial(g_merge2, semiring=semiring), use_reentrant=use_reentrant)
        merge_h3 = checkpoint(partial(g_merge3, semiring=semiring), use_reentrant=use_reentrant)

        rule: Tensor = params["rule"]  # (batch, tgt_nt, tgt_nt+pt, tgt_nt+pt)
        align_left: Tensor = params["align_left"]  # (batch, tgt_nt+pt, src_nt, src_nt/pt) B i j
        align_right: Tensor = params["align_right"]  # (batch, tgt_nt+pt, src_nt, src_nt/pt) C i k
        term: Tensor = params["term"]  # (batch, seq_len, PT)
        root: Tensor = params["root"]  # (batch, NT)

        constraint = params.get("constraint")

        _, batch, N, _ = term.shape
        N += 1

        ntnt = rule[..., : self.nt_states, : self.nt_states]
        ntpt = rule[..., : self.nt_states, self.nt_states :]
        ptnt = rule[..., self.nt_states :, : self.nt_states]
        ptpt = rule[..., self.nt_states :, self.nt_states :]
        align_left_nt, align_left_pt = align_left.split([self.nt_states, self.pt_states], dim=2)
        align_right_nt, align_right_pt = align_right.split([self.nt_states, self.pt_states], dim=2)
        align_left_pt = align_left_pt[..., : self.pt_num_nodes]
        align_left_nt = align_left_nt[..., : self.nt_num_nodes]
        align_right_pt = align_right_pt[..., : self.pt_num_nodes]
        align_right_nt = align_right_nt[..., : self.nt_num_nodes]

        if trace:
            _span_indicator = term.new_zeros(batch, N, N, self.nt_states, self.nt_num_nodes, requires_grad=True)
            span_indicator = _span_indicator.view(1, batch, N, N, self.nt_states, self.nt_num_nodes)
            span_indicator_running = span_indicator
        else:
            _span_indicator = None
            span_indicator = None
            span_indicator_running = None

        term = term.view(term.shape[0], batch, term.shape[2], self.pt_states, self.pt_num_nodes)
        # qnbj,qbij->qnbi
        left_term = semiring.sum(semiring.mul(term.unsqueeze(4), align_left_pt.unsqueeze(2)), dim=5)
        right_term = semiring.sum(semiring.mul(term.unsqueeze(4), align_right_pt.unsqueeze(2)), dim=5)
        left_s = semiring.new_zeros((batch, N, N, self.nt_states, self.nt_num_nodes))
        right_s = semiring.new_zeros((batch, N, N, self.nt_states, self.nt_num_nodes))

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (torch.arange(2, N + 1).unsqueeze(1) <= self.lens.cpu().unsqueeze(0)).sum(1)

        # w: span width
        final = []
        for step, w in enumerate(range(2, N)):
            # n: the number of spans of width w.
            n = N - w
            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            # s, b, n, 1, pt
            y_term = left_term[:, :, : N - w].unsqueeze(3)
            z_term = right_term[:, :, w - 1 :].unsqueeze(3)

            if w == 2:
                x = merge_h(y_term, z_term, ptpt)
            else:
                y = stripe(left_s, current_bsz, n, w - 2, (0, 2)).clone()
                z = stripe(right_s, current_bsz, n, w - 2, (1, w), 0).clone()
                if w == 3:
                    x = merge_h2(y, z, y_term, z_term, ptnt, ntpt)
                else:
                    x = merge_h3(y, z, y_term, z_term, ptnt, ntpt, ntnt)

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
                    ntnt = ntnt[:, :unfinished]
                    ntpt = ntpt[:, :unfinished]
                    ptnt = ptnt[:, :unfinished]
                    ptpt = ptpt[:, :unfinished]
                    align_left_nt = align_left_nt[:, :unfinished]
                    align_right_nt = align_right_nt[:, :unfinished]
                    if trace:
                        span_indicator_running = span_indicator_running[:, :unfinished]

                # qnbj,qbij->qnbi
                left_x = semiring.sum(semiring.mul(x.unsqueeze(4), align_left_nt.unsqueeze(2)), dim=5)
                right_x = semiring.sum(semiring.mul(x.unsqueeze(4), align_right_nt.unsqueeze(2)), dim=5)
                diagonal_copy_(left_s, left_x, unfinished, w)
                diagonal_copy_(right_s, right_x, unfinished, w)

            if unfinished == 0:
                break

        final = torch.cat(final, dim=1)
        final = semiring.mul(final.squeeze(2).flatten(-2), root)
        logZ = semiring.sum(final, dim=-1)
        logZ = semiring.unconvert(logZ)
        return logZ, _span_indicator

    @staticmethod
    def random(bsz, max_len, tgt_pt, src_pt, tgt_nt, src_nt, r):
        nt = tgt_nt * src_nt
        pt = tgt_pt * src_pt

        left = torch.randn(bsz, tgt_nt + tgt_pt, src_nt, max(src_nt, src_pt))
        left[:, :tgt_nt, :, src_nt:] = -1e9
        left[:, tgt_nt:, :, src_pt:] = -1e9
        right = torch.randn(bsz, tgt_nt + tgt_pt, src_nt, max(src_nt, src_pt))
        right[:, :tgt_nt, :, src_nt:] = -1e9
        right[:, tgt_nt:, :, src_pt:] = -1e9
        return {
            "term": torch.randn(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "rule": torch.randn(bsz, tgt_nt, (tgt_nt + tgt_pt) ** 2)
            .softmax(-1)
            .view(bsz, tgt_nt, tgt_nt + tgt_pt, tgt_nt + tgt_pt)
            .requires_grad_(True),
            "align_left": left.log_softmax(-1).requires_grad_(True),
            "align_right": right.log_softmax(-1).requires_grad_(True),
        }


class Decomp8Sampler(DecompSamplerBase):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(2).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        rule = params["rule"].exp().flatten(2).cumsum(2).cpu().numpy()
        align_left = params["align_left"].exp().cumsum(3).cpu().numpy()
        align_right = params["align_right"].exp().cumsum(3).cpu().numpy()

        return terms, rule, align_left, align_right, roots

    @staticmethod
    @jit(nopython=True)
    def sample_impl(
        terms: np.ndarray,  # seqlen x pt, in normal space
        rules: np.ndarray,  # tgt_nt x tgt_nt+pt x tgt_nt+pt, in normal space
        rules_left: np.ndarray,  # (tgt_nt+pt, src_nt, src_nt/pt), in normal space
        rules_right: np.ndarray,  # (tgt_nt+pt, src_nt, src_nt/pt), in normal space
        roots: np.ndarray,
        nt_num_nodes: int,
        nt_states: int,
        pt_num_nodes: int,
        pt_states: int,
        use_copy=True,
        num_samples=1,
        max_length=100,
        max_actions=100,
        unk=1,
    ):
        COPY_NT = nt_states - 1
        COPY_PT = pt_states - 1
        S = nt_states + pt_states
        samples = [[0] for _ in range(num_samples)]
        types = [[0] for _ in range(num_samples)]
        status = [_OK for _ in range(num_samples)]

        for sidx in range(num_samples):
            nonterminals: List[Tuple[int, int]] = []
            preterminals: List[Tuple[int, int, bool]] = []
            actions = 0
            try:
                s = weighted_random_v2(roots)
                state, i = divmod(s, nt_num_nodes)
                nonterminals.append((state, i))
                while len(nonterminals) > 0 and len(preterminals) < max_length and actions < max_actions:
                    actions += 1
                    s, i = nonterminals.pop()

                    if s >= nt_states:
                        preterminals.append((s - nt_states, i, False))
                        continue
                    if use_copy and s == COPY_NT:
                        preterminals.append((s, i, True))
                        continue

                    sample = weighted_random_v2(rules[s])
                    left, right = divmod(sample, S)

                    j = weighted_random_v2(rules_left[left, i])
                    k = weighted_random_v2(rules_right[right, i])

                    nonterminals.extend([(right, k), (left, j)])

            except Exception:
                status[sidx] = _SONMASK

            if actions == max_actions or (len(preterminals) == max_length and len(nonterminals) > 0):
                status[sidx] = _REACHLIMIT

            terminals: List[int] = []
            terminal_type: List[int] = []
            try:
                for s, i, flag in preterminals:
                    if flag:
                        terminals.append(i)
                        terminal_type.append(_COPY_NT)
                        continue
                    if use_copy and s == COPY_PT:
                        terminals.append(i)
                        terminal_type.append(_COPY_PT)
                    else:
                        sample = weighted_random_v2(terms[s * pt_num_nodes + i])
                        if use_copy and sample == unk:
                            # force <unk> tokens to copy
                            terminals.append(i)
                            terminal_type.append(_COPY_PT)
                        else:
                            terminals.append(sample)
                            terminal_type.append(_VOCAB)
            except Exception:
                status[sidx] = _SONMASK

            samples[sidx] = terminals
            types[sidx] = terminal_type
        return samples, types, status


def convert_decomp8_to_pcfg(params, nt_states):
    term = params["term"]
    rule = params["rule"].exp()
    left = params["align_left"].exp()
    right = params["align_right"].exp()
    bsz, tgt_nt, tgt_nt_pt, _ = rule.shape
    tgt_pt = tgt_nt_pt - tgt_nt
    src_nt = left.shape[2]
    src_pt = term.shape[2] // tgt_pt
    align_left_nt, align_left_pt = left.split([tgt_nt, tgt_pt], dim=1)
    align_right_nt, align_right_pt = right.split([tgt_nt, tgt_pt], dim=1)

    rule11 = torch.einsum(
        "qabc,qbij,qcik->qaibjck",
        rule[:, :, :tgt_nt, :tgt_nt],
        align_left_nt[..., :src_nt],
        align_right_nt[..., :src_nt],
    )
    rule12 = torch.einsum(
        "qabc,qbij,qcik->qaibjck",
        rule[:, :, :tgt_nt, tgt_nt:],
        align_left_nt[..., :src_nt],
        align_right_pt[..., :src_pt],
    )
    rule21 = torch.einsum(
        "qabc,qbij,qcik->qaibjck",
        rule[:, :, tgt_nt:, :tgt_nt],
        align_left_pt[..., :src_pt],
        align_right_nt[..., :src_nt],
    )
    rule22 = torch.einsum(
        "qabc,qbij,qcik->qaibjck",
        rule[:, :, tgt_nt:, tgt_nt:],
        align_left_pt[..., :src_pt],
        align_right_pt[..., :src_pt],
    )
    rule = rule11.new_zeros(
        bsz,
        nt_states * src_nt,
        (nt_states * src_nt) + (tgt_pt * src_pt),
        (nt_states * src_nt) + (tgt_pt * src_pt),
    )
    shape = rule11.shape
    rule[:, :, : nt_states * src_nt, : nt_states * src_nt] = (
        rule11.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]) + 1e-9
    ).log()
    shape = rule12.shape
    rule[:, :, : nt_states * src_nt, nt_states * src_nt :] = (
        rule12.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]) + 1e-9
    ).log()
    shape = rule21.shape
    rule[:, :, nt_states * src_nt :, : nt_states * src_nt] = (
        rule21.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]) + 1e-9
    ).log()
    shape = rule22.shape
    rule[:, :, nt_states * src_nt :, nt_states * src_nt :] = (
        rule22.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]) + 1e-9
    ).log()
    output = {"term": term, "rule": rule, "root": params["root"]}
    if "constraint" in params:
        output["constraint"] = params["constraint"]
    if "add" in params:
        output["add"] = params["add"]
    if "lse" in params:
        output["lse"] = params["lse"]
    return output


# oe.contract_path('qabc,qbij,qcik,qnwbj,qnwck->qnai',
#     rule, align_left, align_right, Y, Z, optimize='optimal')
# >>>
# Complete contraction:  qabc,qbij,qcik,qnwbj,qnwck->qnai
#           Naive scaling:  9
#       Optimized scaling:  6
#        Naive FLOP count:  2.560e+9
#    Optimized FLOP count:  5.760e+6
#     Theoretical speedup:  4.444e+2
#    Largest intermediate:  8.000e+4 elements
#  --------------------------------------------------------------------------------
#  scaling        BLAS                current                             remaining
#  --------------------------------------------------------------------------------
#     6              0      qnwbj,qbij->qnwbi           qabc,qcik,qnwck,qnwbi->qnai
#     6              0      qnwck,qcik->qnwci                qabc,qnwbi,qnwci->qnai
#     6              0     qnwci,qnwbi->qncib                      qabc,qncib->qnai
#     6              0       qncib,qabc->qnai                            qnai->qnai


def eq_qnkrj(v1, v2, semiring):
    # qnwbi,qnwci->qncib
    v = semiring.mul(v1.unsqueeze(-1), v2.transpose(-1, -2).unsqueeze(-3))
    v = semiring.sum(v, dim=3)
    return v


def g_merge(y, z, ptpt, semiring):
    # y: c, bsz, n, 1, pt
    # z: c, bsz, n, 1, pt
    # ptpt: c, bsz, nt, pt, pt
    x = eq_qnkrj(y, z, semiring)
    x = semiring.sum(semiring.mul(x.transpose(3, 4).unsqueeze(3), ptpt[:, :, None, :, None]).flatten(-2), dim=-1)
    return x


def g_merge2(y, z, y_term, z_term, ptnt, ntpt, semiring):
    x1 = eq_qnkrj(y_term, z, semiring)
    x1 = semiring.sum(semiring.mul(x1.transpose(3, 4).unsqueeze(3), ptnt[:, :, None, :, None]).flatten(-2), dim=-1)
    x3 = eq_qnkrj(y, z_term, semiring)
    x3 = semiring.sum(semiring.mul(x3.transpose(3, 4).unsqueeze(3), ntpt[:, :, None, :, None]).flatten(-2), dim=-1)
    return semiring.add(x1, x3)


def g_merge3(y, z, y_term, z_term, ptnt, ntpt, ntnt, semiring):
    x1 = eq_qnkrj(y_term, z[:, :, :, :1], semiring)
    x1 = semiring.sum(semiring.mul(x1.transpose(3, 4).unsqueeze(3), ptnt[:, :, None, :, None]).flatten(-2), dim=-1)
    x2 = eq_qnkrj(y[:, :, :, :-1], z[:, :, :, 1:], semiring)
    x2 = semiring.sum(semiring.mul(x2.transpose(3, 4).unsqueeze(3), ntnt[:, :, None, :, None]).flatten(-2), dim=-1)
    x3 = eq_qnkrj(y[:, :, :, -1:], z_term, semiring)
    x3 = semiring.sum(semiring.mul(x3.transpose(3, 4).unsqueeze(3), ntpt[:, :, None, :, None]).flatten(-2), dim=-1)
    return semiring.add(x1, x2, x3)


if __name__ == "__main__":
    from torch_struct import SentCFG

    from src.datamodules.scan_datamodule import SCANDataModule
    from src.models.tgt_parser.neural_decomp8 import NeuralDecomp8TgtParser

    datamodule = SCANDataModule(
        "data/scan_debug.txt", "data/scan_debug.txt", "data/scan_debug.txt", enable_cache=False
    )
    datamodule.setup()

    # B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 2, 4, 2, 5, 3, 7
    # NT = TGT_NT * SRC_NT
    # PT = TGT_PT * SRC_PT
    # VOCAB = 2
    # NUM_SAMPLE = 10000
    # MAX_LENGTH = 4
    # r = 3
    # lens = [max(2, N - i) for i in range(B)]
    # params = Decomp8.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    # meta = {
    #     "batch_size": B,
    #     "nt_states": TGT_NT,
    #     "nt_num_nodes": SRC_NT,
    #     "pt_states": TGT_PT,
    #     "pt_num_nodes": SRC_PT,
    #     "batch_size": B,
    # }

    # pcfg = Decomp8(params, lens, **meta)
    # params_ref = convert_decomp8_to_pcfg(params, TGT_NT)
    # pcfg_ref = SentCFG((params_ref["term"], params_ref["rule"], params_ref["root"]), lens)

    # print("test nll")
    # nll = pcfg.nll
    # nll_ref = -pcfg_ref.partition
    # assert torch.allclose(nll, nll_ref), (nll, nll_ref)

    # print("test marginal")
    # m1 = pcfg.marginal
    # check_full_marginal(m1["term"], m1["trace"], lens)

    # m2 = pcfg_ref.marginals[-1]
    # compare_marginal(m1["trace"], m2)

    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 2, 4, 2, 1, 1, 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    VOCAB = 4
    NUM_SAMPLE = 50000
    MAX_LENGTH = 4
    # r = 1
    # lens = [max(2, N - i) for i in range(B)]
    # params = Decomp8.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    # meta = {
    #     "batch_size": B,
    #     "nt_states": TGT_NT,
    #     "nt_num_nodes": SRC_NT,
    #     "pt_states": TGT_PT,
    #     "pt_num_nodes": SRC_PT,
    #     "batch_size": B,
    # }

    # pcfg = Decomp8(params, lens, **meta)

    # print("test sample tree")
    # output = pcfg.sample_one(need_span=True, need_event=True)
    # with torch.no_grad():
    #     prob = (pcfg.score(output["event"]) - pcfg.partition).exp()
    # target = output["span"]

    # cnt = [0 for i in range(B)]
    # for _ in range(1000):
    #     output = pcfg.sample_one(need_span=True)["span"]
    #     for b in range(B):
    #         t = target[b]
    #         p = output[b]
    #         if t == p:
    #             cnt[b] += 1

    # cnt = torch.tensor(cnt, dtype=torch.float) / 1000
    # print(prob, cnt)
    # assert torch.allclose(cnt, prob, rtol=0.01, atol=0.1), (prob, cnt)

    print("test sample seq")
    batch = next(iter(datamodule.train_dataloader()))
    MAX_LENGTH = 4
    B = 1
    VOCAB = len(datamodule.tgt_vocab)
    spans = [[(i, i + 1, 0) for i in range(l)] + [(0, i + 1, 0) for i in range(1, l)] for l in batch["src_lens"]]
    node_features = [torch.randn(len(s), 4) for s in spans]
    parser = NeuralDecomp8TgtParser(
        pt_states=TGT_PT,
        nt_states=TGT_NT,
        dim=4,
        src_dim=4,
        num_layers=1,
        vocab=VOCAB,
        datamodule=datamodule,
        use_copy=False,
        generation_max_length=MAX_LENGTH,
        generation_num_samples=NUM_SAMPLE,
        generation_strict=True,
    )
    pred = parser(node_features, spans)
    pred = parser.prepare_sampler(pred, batch["src"], batch["src_ids"])
    samples = pred.sampler()
    for bidx in range(B):
        count = Counter(tuple(item[0]) for item in samples[bidx])
        total = len(samples[bidx])

        sub_pred = copy(pred)
        params = pred.params
        params = {key: value[bidx, None] for key, value in params.items()}
        sub_pred.params = params
        sub_pred.batch_size = 1

        probs_with_seq = []
        for seq in enumerate_seq(MAX_LENGTH, VOCAB):
            prob = compute_unnormalized_prob(seq, parser, sub_pred)
            probs_with_seq.append((prob, seq))

        partition = sum(item[0] for item in probs_with_seq)

        sub_pred = parser.observe_x(sub_pred, batch["tgt_ids"], batch["tgt_lens"])
        partition_ref = sum(
            sub_pred.dist.partition_at_length(sub_pred.params, l).exp() for l in range(2, MAX_LENGTH + 1)
        )

        probs_with_seq.sort(reverse=True)

        errors = []
        empirical_cdf = 0
        theoratical_cdf = 0
        for prob, seq in probs_with_seq[:5]:
            empirical_cdf += count[tuple(seq)] / total
            theoratical_cdf += prob / partition
            errors.append(abs(empirical_cdf - theoratical_cdf) / theoratical_cdf)

        assert max(errors) < 0.1, errors
        print(empirical_cdf, theoratical_cdf, errors)
    print("pass")
