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


class Decomp7(DecompBase):
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

    def inside(self, params, semiring, trace=False, use_reentrant=True) -> Tuple[Tensor, Tensor]:
        params = self.preprocess(params, semiring)
        merge_h = checkpoint(partial(g_merge_h, semiring=semiring), use_reentrant=use_reentrant)
        merge_h2 = checkpoint(partial(g_merge_h2, semiring=semiring), use_reentrant=use_reentrant)
        merge_h3 = checkpoint(partial(g_merge_h3, semiring=semiring), use_reentrant=use_reentrant)

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
            _span_indicator = term.new_zeros(batch, N, N, self.nt_states, nt_spans, requires_grad=True)
            span_indicator = _span_indicator.view(1, batch, N, N, self.nt_states, nt_spans)
            span_indicator_running = span_indicator
        else:
            _span_indicator = None
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
        return logZ, _span_indicator

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


class Decomp7Sampler(DecompSamplerBase):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(2).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        H = params["head"].cumsum(2).cpu().numpy()
        L = params["left"].exp().cumsum(3).cpu().numpy()
        R = params["right"].exp().cumsum(3).cpu().numpy()

        SLR = params["slr"].flatten(3).cumsum(3).cpu().numpy()
        return terms, H, L, R, SLR, roots

    @staticmethod
    @jit(nopython=True)
    def sample_impl(
        terms: np.ndarray,  # seqlen x pt, in normal space
        rules_head: np.ndarray,  # nt x r, in normal space
        rules_left: np.ndarray,  # (nt+pt) x r, in normal space
        rules_right: np.ndarray,  # (nt+pt) x r, in normal space
        rules_slr: np.ndarray,
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

                    r = weighted_random_v2(rules_head[s * nt_num_nodes + i])
                    jk = weighted_random_v2(rules_slr[r, i])
                    j, k = divmod(jk, nt_num_nodes + pt_num_nodes)

                    left = weighted_random_v2(rules_left[r, j])
                    right = weighted_random_v2(rules_right[r, k])

                    if j >= nt_num_nodes:
                        left += nt_states
                        j -= nt_num_nodes
                    if k >= nt_num_nodes:
                        right += nt_states
                        k -= nt_num_nodes

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


def convert_decomp7_to_pcfg(params, nt_states):
    slr = params["slr"]
    term = params["term"]
    left = params["left"].exp()
    right = params["right"].exp()
    bsz, r, src_nt, src, _ = slr.shape
    src_pt = src - src_nt
    pt_states = term.shape[2] // src_pt
    head = params["head"].view(bsz, nt_states, src_nt, r)
    rule11 = torch.einsum(
        "xair,xrjb,xrkc,xrijk->xaibjck",
        head,
        left[:, :, :src_nt, :nt_states],
        right[:, :, :src_nt, :nt_states],
        params["slr"][:, :, :, :src_nt, :src_nt],
    )
    rule12 = torch.einsum(
        "xair,xrjb,xrkc,xrijk->xaibjck",
        head,
        left[:, :, :src_nt, :nt_states],
        right[:, :, src_nt:, :pt_states],
        params["slr"][:, :, :, :src_nt, src_nt:],
    )
    rule21 = torch.einsum(
        "xair,xrjb,xrkc,xrijk->xaibjck",
        head,
        left[:, :, src_nt:, :pt_states],
        right[:, :, :src_nt, :nt_states],
        params["slr"][:, :, :, src_nt:, :src_nt],
    )
    rule22 = torch.einsum(
        "xair,xrjb,xrkc,xrijk->xaibjck",
        head,
        left[:, :, src_nt:, :pt_states],
        right[:, :, src_nt:, :pt_states],
        params["slr"][:, :, :, src_nt:, src_nt:],
    )
    rule = rule11.new_zeros(
        bsz,
        nt_states * src_nt,
        (nt_states * src_nt) + (pt_states * src_pt),
        (nt_states * src_nt) + (pt_states * src_pt),
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


if __name__ == "__main__":
    from torch_struct import SentCFG

    from src.datamodules.scan_datamodule import SCANDataModule
    from src.models.tgt_parser.neural_decomp7 import NeuralDecomp7TgtParser

    datamodule = SCANDataModule(
        "data/scan_debug.txt", "data/scan_debug.txt", "data/scan_debug.txt", enable_cache=False
    )
    datamodule.setup()

    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 2, 4, 2, 5, 3, 7
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    r = 3
    lens = [max(2, N - i) for i in range(B)]
    params = Decomp7.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }

    pcfg = Decomp7(params, lens, **meta)
    params_ref = convert_decomp7_to_pcfg(params, TGT_NT)
    pcfg_ref = SentCFG((params_ref["term"], params_ref["rule"], params_ref["root"]), lens)

    print("test nll")
    nll = pcfg.nll
    nll_ref = -pcfg_ref.partition
    assert torch.allclose(nll, nll_ref), (nll, nll_ref)

    print("test marginal")
    m1 = pcfg.marginal
    check_full_marginal(m1["term"], m1["trace"], lens)

    m2 = pcfg_ref.marginals[-1]
    compare_marginal(m1["trace"], m2)

    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 2, 3, 1, 1, 1, 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    VOCAB = 4
    NUM_SAMPLE = 50000
    MAX_LENGTH = 4
    r = 1
    lens = [max(2, N - i) for i in range(B)]
    params = Decomp7.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }

    pcfg = Decomp7(params, lens, **meta)

    print("test sample tree")
    output = pcfg.sample_one(need_span=True, need_event=True)
    with torch.no_grad():
        prob = (pcfg.score(output["event"]) - pcfg.partition).exp()
    target = output["span"]

    cnt = [0 for i in range(B)]
    for _ in range(1000):
        output = pcfg.sample_one(need_span=True)["span"]
        for b in range(B):
            t = target[b]
            p = output[b]
            if t == p:
                cnt[b] += 1

    cnt = torch.tensor(cnt, dtype=torch.float) / 1000
    print(prob, cnt)
    assert torch.allclose(cnt, prob, rtol=0.01, atol=0.1), (prob, cnt)

    print("test sample seq")
    batch = next(iter(datamodule.train_dataloader()))
    MAX_LENGTH = 2
    B = 1
    VOCAB = len(datamodule.tgt_vocab)
    spans = [[(i, i + 1, 0) for i in range(l)] + [(0, i + 1, 0) for i in range(1, l)] for l in batch["src_lens"]]
    node_features = [torch.randn(len(s), 4) for s in spans]
    parser = NeuralDecomp7TgtParser(
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
