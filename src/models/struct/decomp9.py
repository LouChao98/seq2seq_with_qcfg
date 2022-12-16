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
from torch_struct.semirings.sample import SampledSemiring

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


class Decomp9(DecompBase):
    # A[i] -> B[j], C[k]
    # ----------------
    # A -> r
    # r -> B
    # r -> C
    # r,i -> j,k  Folded
    # ================
    # D[i] -> w
    # ----------------
    # D,i -> w
    # ================
    # S -> A[i]
    # ----------------
    # S -> A
    # S -> i, NOTE that this is a trival dist because `i` must be the root span

    KEYS = ("term", "head", "left", "right", "align_left", "align_right", "root")
    LOGSPACE = (True, True, True, True, True, True)

    def inside(self, params, semiring, trace=False, use_reentrant=True) -> Tuple[Tensor, Tensor]:
        params = self.preprocess(params, semiring)
        merge = checkpoint(partial(g_merge, semiring=semiring), use_reentrant=use_reentrant)

        head: Tensor = params["head"]  # (batch, tgt_nt, r)
        L: Tensor = params["left"]  # (batch, r, tgt_nt+pt), r->L
        R: Tensor = params["right"]  # (batch, r, tgt_nt+pt), r->R

        align_left: Tensor = params["align_left"]  # (batch, r, src_nt, src_nt+pt) B i j
        align_right: Tensor = params["align_right"]  # (batch, r, src_nt, src_nt+pt) C i k
        term: Tensor = params["term"]  # (batch, seq_len, PT)
        root: Tensor = params["root"]  # (batch, NT)

        constraint = params.get("constraint")
        assert constraint is None, "this do not support constraint. e.g., copy"

        _, batch, N, _ = term.shape
        N += 1
        r = head.shape[-1]

        lnt = L[..., : self.nt_states]
        lpt = L[..., self.nt_states :]
        rnt = R[..., : self.nt_states]
        rpt = R[..., self.nt_states :]
        # c, b, r, nt * c, b, nt, r = batch, parentr, childr
        hl = semiring.sum(semiring.mul(lnt.unsqueeze(4), head.unsqueeze(2)), dim=3)
        hr = semiring.sum(semiring.mul(rnt.unsqueeze(4), head.unsqueeze(2)), dim=3)

        # b, tgt_nt, src_nt * b, tgt_nt, r -> b, r, src_nt
        root = semiring.sum(
            semiring.mul(
                root.view(-1, batch, self.nt_states, 1, self.nt_num_nodes), head.view(-1, batch, self.nt_states, r, 1)
            ),
            dim=2,
        )

        align_left_nt, align_left_pt = align_left.split([self.nt_num_nodes, self.pt_num_nodes], dim=4)
        align_right_nt, align_right_pt = align_right.split([self.nt_num_nodes, self.pt_num_nodes], dim=4)

        if trace:
            _span_indicator = term.new_zeros(batch, N, N, requires_grad=True)
            span_indicator = _span_indicator.view(1, batch, N, N)
            span_indicator_running = span_indicator
        else:
            _span_indicator = None
            span_indicator = None
            span_indicator_running = None

        term = term.view(term.shape[0], batch, term.shape[2], self.pt_states, self.pt_num_nodes)

        def transform(_x, _r_nt, _align):
            # TODO normal-space should be faster
            # qnbj,qrb -> qnrj
            _a = semiring.mul(_x.unsqueeze(3), _r_nt[:, :, None, :, :, None])
            _a = semiring.sum(_a, 4)
            # qnrj,qrij -> qnri
            _a = semiring.mul(_a.unsqueeze(4), _align[:, :, None])
            _a = semiring.sum(_a, dim=5)
            return _a

        left_term = transform(term, lpt, align_left_pt)
        right_term = transform(term, rpt, align_right_pt)

        left_s = semiring.new_zeros((batch, N, N, r, self.nt_num_nodes))
        right_s = semiring.new_zeros((batch, N, N, r, self.nt_num_nodes))
        diagonal_copy_(left_s, left_term, batch, 1)
        diagonal_copy_(right_s, right_term, batch, 1)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (torch.arange(2, N + 1).unsqueeze(1) <= self.lens.cpu().unsqueeze(0)).sum(1)

        # w: span width
        final = []
        for step, w in enumerate(range(2, N)):
            # n: the number of spans of width w.
            n = N - w
            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            y = stripe(left_s, current_bsz, n, w - 1, (0, 1)).clone()
            z = stripe(right_s, current_bsz, n, w - 1, (1, w), 0).clone()
            x = merge(y, z)

            if trace:
                indicator = span_indicator_running.diagonal(w, 2, 3).movedim(-1, 2)
                x = x + indicator[..., None, None]

            if current_bsz - unfinished > 0:
                final.insert(0, x[:, unfinished:current_bsz, :1])

            if unfinished > 0:
                if current_bsz > unfinished:
                    x = x[:, :unfinished]
                    left_s = left_s[:, :unfinished]
                    right_s = right_s[:, :unfinished]
                    hl = hl[:, :unfinished]
                    hr = hr[:, :unfinished]
                    align_left_nt = align_left_nt[:, :unfinished]
                    align_right_nt = align_right_nt[:, :unfinished]
                    if trace:
                        span_indicator_running = span_indicator_running[:, :unfinished]

                # qnbj,qbij->qnbi
                left_x = transform(x, hl, align_left_nt)
                right_x = transform(x, hr, align_right_nt)
                diagonal_copy_(left_s, left_x, unfinished, w)
                diagonal_copy_(right_s, right_x, unfinished, w)

            if unfinished == 0:
                break

        final = torch.cat(final, dim=1)
        final = semiring.mul(final.squeeze(2), root)
        logZ = semiring.sum(final.flatten(-2), dim=-1)
        logZ = semiring.unconvert(logZ)
        return logZ, _span_indicator

    def sample_one(self, need_event=False, need_span=True):
        params = {
            k: v.detach().requires_grad_() if isinstance(v, torch.Tensor) else v for k, v in self.params.items()
        }
        logZ, trace = self.inside(params, SampledSemiring, True)
        logZ.sum().backward()

        output = {"logZ": logZ}

        if need_span:
            spans = [[] for _ in range(self.batch_size)]
            for b, i, j in trace.grad.nonzero().tolist():
                spans[b].append((i, j, None, None))
            for b, i, state_node in params["term"].grad.nonzero().tolist():
                state, node = divmod(state_node, self.pt_num_nodes)
                spans[b].append((i, i + 1, state, node))
            output["span"] = spans

        if need_event:
            output["event"] = {k: params[k].grad for k in self.KEYS} | {"trace": trace.grad}

        return output

    @staticmethod
    def random(bsz, max_len, tgt_pt, src_pt, tgt_nt, src_nt, r):
        nt = tgt_nt * src_nt
        pt = tgt_pt * src_pt

        left = torch.randn(bsz, r, src_nt, src_nt + src_pt)
        right = torch.randn(bsz, r, src_nt, src_nt + src_pt)
        return {
            "term": torch.randn(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "head": torch.randn(bsz, tgt_nt, r).log_softmax(-1).requires_grad_(True),
            "left": torch.randn(bsz, r, tgt_nt + tgt_pt).log_softmax(-1).requires_grad_(True),
            "right": torch.randn(bsz, r, tgt_nt + tgt_pt).log_softmax(-1).requires_grad_(True),
            "align_left": left.log_softmax(-1).requires_grad_(True),
            "align_right": right.log_softmax(-1).requires_grad_(True),
        }


class Decomp9Sampler(DecompSamplerBase):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(2).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        head = params["head"].exp().cumsum(2).cpu().numpy()
        left = params["left"].exp().cumsum(2).cpu().numpy()
        right = params["right"].exp().cumsum(2).cpu().numpy()
        align_left = params["align_left"].exp().cumsum(3).cpu().numpy()
        align_right = params["align_right"].exp().cumsum(3).cpu().numpy()

        return terms, head, left, right, align_left, align_right, roots

    @staticmethod
    @jit(nopython=True)
    def sample_impl(
        terms: np.ndarray,  # seqlen x pt
        rules_head: np.ndarray,  # tgt_nt xr
        rules_left: np.ndarray,  # r x tgt_nt+pt
        rules_right: np.ndarray,  # r x tgt_nt+pt
        rules_align_left: np.ndarray,  # (tgt_nt+pt, src_nt, src_nt/pt)
        rules_align_right: np.ndarray,  # (tgt_nt+pt, src_nt, src_nt/pt)
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

                    r = weighted_random_v2(rules_head[s])
                    left = weighted_random_v2(rules_left[r])
                    right = weighted_random_v2(rules_right[r])

                    j = weighted_random_v2(rules_align_left[r, i])
                    k = weighted_random_v2(rules_align_right[r, i])

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


def convert_decomp9_to_pcfg(params, nt_states):
    head = params["head"].exp()
    left = params["left"].exp()
    right = params["right"].exp()
    align_left = params["align_left"].exp()
    align_right = params["align_right"].exp()

    bsz, tgt_nt, r = head.shape
    _, _, tgt_nt_pt = left.shape
    tgt_pt = tgt_nt_pt - tgt_nt
    src_nt = align_left.shape[2]
    src_pt = align_left.shape[3] - align_left.shape[2]

    rule11 = torch.einsum(
        "qar,qrb,qrc,qrij,qrik->qaibjck",
        head,
        left[..., :tgt_nt],
        right[..., :tgt_nt],
        align_left[..., :src_nt],
        align_right[..., :src_nt],
    )
    rule12 = torch.einsum(
        "qar,qrb,qrc,qrij,qrik->qaibjck",
        head,
        left[..., :tgt_nt],
        right[..., tgt_nt:],
        align_left[..., :src_nt],
        align_right[..., src_nt:],
    )
    rule21 = torch.einsum(
        "qar,qrb,qrc,qrij,qrik->qaibjck",
        head,
        left[..., tgt_nt:],
        right[..., :tgt_nt],
        align_left[..., src_nt:],
        align_right[..., :src_nt],
    )
    rule22 = torch.einsum(
        "qar,qrb,qrc,qrij,qrik->qaibjck",
        head,
        left[..., tgt_nt:],
        right[..., tgt_nt:],
        align_left[..., src_nt:],
        align_right[..., src_nt:],
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
    output = {"term": params["term"], "rule": rule, "root": params["root"]}
    if "constraint" in params:
        output["constraint"] = params["constraint"]
    if "add" in params:
        output["add"] = params["add"]
    if "lse" in params:
        output["lse"] = params["lse"]
    return output


def g_merge(y, z, semiring):
    # y: c, bsz, n, 1, r, j
    # z: c, bsz, n, 1, r, j
    return semiring.sum(semiring.mul(y, z), dim=3)


if __name__ == "__main__":
    from torch_struct import SentCFG

    from src.datamodules.scan_datamodule import SCANDataModule
    from src.models.tgt_parser.neural_decomp9 import NeuralDecomp9TgtParser

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
    params = Decomp9.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }

    pcfg = Decomp9(params, lens, **meta)
    params_ref = convert_decomp9_to_pcfg(params, TGT_NT)
    pcfg_ref = SentCFG((params_ref["term"], params_ref["rule"], params_ref["root"]), lens)

    print("test nll")
    nll = pcfg.nll
    nll_ref = -pcfg_ref.partition
    assert torch.allclose(nll, nll_ref), (nll, nll_ref)

    print("test marginal")
    m1 = pcfg.marginal

    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 2, 4, 2, 1, 1, 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    VOCAB = 4
    NUM_SAMPLE = 50000
    MAX_LENGTH = 4
    r = 1
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
    parser = NeuralDecomp9TgtParser(
        pt_states=TGT_PT,
        nt_states=TGT_NT,
        dim=4,
        src_dim=4,
        num_layers=1,
        cpd_rank=r,
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
