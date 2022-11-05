import logging
from collections import Counter
from copy import copy
from functools import partial
from typing import Dict, List

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


class NoDecomp(DecompBase):
    """
    term: b n tgt_pt src_pt
    rule: b nt nt+pt nt+pt
    root: b tgt_ntt src_nt
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

        term = term.view(term.shape[0], bsz, term.shape[2], self.pt_states * self.pt_num_nodes)

        if trace:
            _span_indicator = term.new_zeros(bsz, N, N, self.nt_states, self.nt_num_nodes, requires_grad=True)
            span_indicator = _span_indicator.view(1, bsz, N, N, NT)
            span_indicator_running = span_indicator
        else:
            _span_indicator = None
            span_indicator = None
            span_indicator_running = None

        s = semiring.new_zeros((bsz, N, N, NT))

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
            y_term = term[:, :, : N - w].unsqueeze(3)
            z_term = term[:, :, w - 1 :].unsqueeze(3)

            if w == 2:
                x = merge(y_term, z_term, PTPT)
            else:
                y = stripe(s, current_bsz, n, w - 2, (0, 2)).clone()
                z = stripe(s, current_bsz, n, w - 2, (1, w), 0).clone()
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
                indicator = span_indicator_running.diagonal(w, 2, 3).movedim(-1, 2)
                x = x + indicator

            if current_bsz - unfinished > 0:
                final.insert(0, x[:, unfinished:current_bsz, :1])

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
                diagonal_copy_(s, x, unfinished, w)

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
            "term": torch.randn(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "rule": torch.randn(bsz, nt, nt + pt, nt + pt)
            .flatten(2)
            .log_softmax(-1)
            .view(bsz, nt, nt + pt, nt + pt)
            .requires_grad_(True),
        }


class NoDecompSampler(DecompSamplerBase):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(2).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        rule = self.threshold(params["rule"].exp()).flatten(2).cumsum(2).cpu().numpy()
        return terms, rule, roots

    @staticmethod
    @jit(nopython=True)
    def sample_impl(
        terms: np.ndarray,
        rules: np.ndarray,
        roots: np.ndarray,
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
        NT = rules.shape[0]
        PT = terms.shape[0]
        S = NT + PT
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
                    s = nonterminals.pop()  # get the last element
                    if s < NT:
                        if use_copy:
                            nt_state = s // nt_num_nodes
                            if nt_state == COPY_NT:
                                nt_node = s % nt_num_nodes
                                preterminals.append(nt_node)
                                is_copy_pt.append(True)
                                continue
                        actions += 1
                        sample = weighted_random_v2(rules[s])
                        left, right = divmod(sample, S)
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


def g_merge(y, z, ptpt, semiring):
    # y: c, bsz, n, 1, pt
    # z: c, bsz, n, 1, pt
    # ptpt: c, bsz, nt, pt, pt

    # c, bsz, n, 1, pt, NEW * c, bsz, n, 1, NEW, pt -> c, bsz, n, pt, pt
    x = semiring.mul(y.unsqueeze(5), z.unsqueeze(4)).squeeze(3)
    # c, bsz, n, NEW, pt, pt * c, bsz, NEW nt, pt, pt -> c, bsz, n, nt, pt, pt
    x = semiring.mul(x.unsqueeze(3), ptpt[:, :, None])
    # c, bsz, n, nt, (pt, pt) -> c, bsz, n, nt
    x = semiring.sum(x.flatten(4), dim=4)
    return x


def g_merge2(y, z, y_term, z_term, ptnt, ntpt, semiring):
    # y: c, bsz, n, 1, nt
    # z: c, bsz, n, 1, nt
    # y_term: c, bsz, n, 1, pt
    # z_term: c, bsz, n, 1, pt
    # ptnt: c, bsz, nt, pt, nt
    # ntpt: c, bsz, nt, nt, pt

    x1 = semiring.mul(y.unsqueeze(5), z_term.unsqueeze(4)).squeeze(3)
    x1 = semiring.mul(x1.unsqueeze(3), ntpt[:, :, None])
    x1 = semiring.sum(x1.flatten(4), dim=4)

    x3 = semiring.mul(y_term.unsqueeze(5), z.unsqueeze(4)).squeeze(3)
    x3 = semiring.mul(x3.unsqueeze(3), ptnt[:, :, None])
    x3 = semiring.sum(x3.flatten(4), dim=4)

    return semiring.add(x1, x3)


def g_merge3(y, z, y_term, z_term, ptnt, ntpt, ntnt, semiring):
    # y: c, bsz, n, 1, nt
    # z: c, bsz, n, 1, nt
    # y_term: c, bsz, n, 1, pt
    # z_term: c, bsz, n, 1, pt
    # ptnt: c, bsz, nt, pt, nt
    # ntpt: c, bsz, nt, nt, pt
    # ntnt: c, bsz, nt, nt, nt
    x1 = semiring.mul(y[:, :, :, -1:].unsqueeze(5), z_term.unsqueeze(4)).squeeze(3)
    x1 = semiring.mul(x1.unsqueeze(3), ntpt[:, :, None])
    x1 = semiring.sum(x1.flatten(4), dim=4)

    x2 = semiring.mul(y[:, :, :, :-1].unsqueeze(5), z[:, :, :, 1:].unsqueeze(4))
    x2 = semiring.sum(x2, dim=3)
    x2 = semiring.mul(x2.unsqueeze(3), ntnt[:, :, None])
    x2 = semiring.sum(x2.flatten(4), dim=4)

    x3 = semiring.mul(y_term.unsqueeze(5), z[:, :, :, :1].unsqueeze(4)).squeeze(3)
    x3 = semiring.mul(x3.unsqueeze(3), ptnt[:, :, None])
    x3 = semiring.sum(x3.flatten(4), dim=4)
    return semiring.add(x1, x2, x3)


if __name__ == "__main__":
    from src.models.tgt_parser.neural_nodecomp import NeuralNoDecompTgtParser
    from src.models.tgt_parser.struct.pcfg import PCFG

    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 2, 4, 2, 5, 3, 7
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    r = 3
    lens = [max(2, N - i) for i in range(B)]
    params = NoDecomp.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }

    pcfg = NoDecomp(params, lens, **meta)
    pcfg_ref = PCFG()

    print("test nll")
    nll = pcfg.nll
    nll_ref = pcfg_ref(params, lens)
    assert torch.allclose(nll, nll_ref), (nll, nll_ref)

    print("test marginal")
    m1 = pcfg.marginal
    check_full_marginal(m1["term"], m1["trace"], lens)

    m2 = pcfg_ref(params, lens, marginal=True)[-1]
    compare_marginal(m1["trace"], m2)

    # print('test mbr decoding')
    # decoded = pcfg.mbr_decoded
    # decoded_ref = pcfg_ref(params, lens, decode=True)

    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 2, 4, 1, 1, 1, 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    r = 3
    lens = [max(2, N - i) for i in range(B)]
    params = NoDecomp.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }

    pcfg = NoDecomp(params, lens, **meta)

    print("test sample tree")
    output = pcfg.sample_one(dtype="full")
    prob = (pcfg.score(output) - pcfg.partition).exp()
    target = output["span"]

    cnt = [0 for i in range(B)]
    for _ in range(1000):
        output = pcfg.sample_one(dtype="tuple")
        for b in range(B):
            t = target[b]
            p = output[b]
            if t == p:
                cnt[b] += 1

    cnt = torch.tensor(cnt, dtype=torch.float) / 1000
    print(prob, cnt)
    assert torch.allclose(cnt, prob, rtol=0.01, atol=10), (prob, cnt)

    print("test sample seq")
    spans = [[(i, i, 0) for i in range(l)] + [(0, i, 0) for i in range(1, l)] for l in lens]
    node_features = [torch.randn(l * 2 - 1, 8) for l in lens]
    parser = NeuralNoDecompTgtParser(
        TGT_PT,
        TGT_NT,
        dim=8,
        src_dim=8,
        num_layers=1,
        vocab=VOCAB,
        use_copy=False,
        generation_max_length=MAX_LENGTH,
        generation_num_samples=NUM_SAMPLE,
        generation_strict=True,
    )
    pred = parser(node_features, spans)
    pred = parser.prepare_sampler(pred, None, None)
    samples = pred.sampler()
    samples = parser.expand_preds_not_using_copy(samples)[0]
    for bidx in range(B):
        count = Counter(tuple(item) for item in samples[bidx])
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
