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


class Decomp1(DecompBase):
    # A,i->r r->B,j r->C,k

    KEYS = ("term", "head", "left", "right", "root")
    LOGSPACE = (True, True, True, True, True)

    def inside(self, params, semiring, trace=False, use_reentrant=True):
        params = self.preprocess(params, semiring)
        merge = checkpoint(partial(g_merge, semiring=semiring), use_reentrant=use_reentrant)
        merge2 = checkpoint(partial(g_merge2, semiring=semiring), use_reentrant=use_reentrant)
        merge3 = checkpoint(partial(g_merge3, semiring=semiring), use_reentrant=use_reentrant)

        head: Tensor = params["head"]  # (batch, NT, r), A[i] -> R
        term: Tensor = params["term"]  # (batch, seq_len, PT)
        root: Tensor = params["root"]  # (batch, NT)
        L: Tensor = params["left"]  # (batch, r, NT + T), r->L
        R: Tensor = params["right"]  # (batch, r, NT + T), r->R
        constraint = params.get("constraint")

        bsz = self.batch_size
        N = term.shape[2] + 1
        NT = self.nt_states * self.nt_num_nodes
        rank = L.shape[2]

        TLNT = L[..., :NT]
        TLPT = L[..., NT:]
        TRNT = R[..., :NT]
        TRPT = R[..., NT:]

        if trace:
            _span_indicator = term.new_zeros(bsz, N, N, self.nt_states, self.nt_num_nodes, requires_grad=True)
            span_indicator = _span_indicator.view(1, bsz, N, N, NT)
            span_indicator_running = span_indicator
        else:
            _span_indicator = None
            span_indicator = None
            span_indicator_running = None

        s = semiring.new_zeros((bsz, N, N, rank))

        # torch.einsum("xlp,xrp->xlr", term, TLPT)
        left_term = semiring.mul(term.unsqueeze(3), TLPT.unsqueeze(2))
        left_term = semiring.sum(left_term, dim=4)
        right_term = semiring.mul(term.unsqueeze(3), TRPT.unsqueeze(2))
        right_term = semiring.sum(right_term, dim=4)

        head = head.transpose(2, 3)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (torch.arange(2, N + 1).unsqueeze(1) <= self.lens.cpu().unsqueeze(0)).sum(1)

        # w: span width
        final = []
        for step, w in enumerate(range(2, N)):

            # n: the number of spans of width w.
            n = N - w
            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            y_term = left_term[:, :, :n].unsqueeze(3)
            z_term = right_term[:, :, w - 1 :].unsqueeze(3)

            if w == 2:
                x = merge(y_term, z_term, head)
            else:
                y = stripe(s, current_bsz, n, w - 2, (0, 2)).clone()
                z = stripe(s, current_bsz, n, w - 2, (w, 1)).clone()
                if w == 3:
                    x = merge2(y, z, y_term, z_term, head)
                else:
                    x = merge3(y, z, y_term, z_term, head)

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
                    head = head[:, :unfinished]
                    TLNT = TLNT[:, :unfinished]
                    TRNT = TRNT[:, :unfinished]
                    if trace:
                        span_indicator_running = span_indicator_running[:, :unfinished]
                # c b 1 n nt, c b r 1 nt
                left_x = semiring.sum(semiring.mul(x.unsqueeze(2), TLNT.unsqueeze(3)), dim=4)
                right_x = semiring.sum(semiring.mul(x.unsqueeze(2), TRNT.unsqueeze(3)), dim=4)

                s[:, :unfinished].diagonal(w, 2, 3).copy_(left_x)
                s[:, :unfinished].diagonal(-w, 2, 3).copy_(right_x)
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
        return {
            "term": torch.randn(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "head": torch.randn(bsz, nt, r).log_softmax(-1).requires_grad_(True),
            "left": torch.randn(bsz, r, nt + pt).log_softmax(-1).requires_grad_(True),
            "right": torch.randn(bsz, r, nt + pt).log_softmax(-1).requires_grad_(True),
        }


class Decomp1Sampler(DecompSamplerBase):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        terms = params["term"].exp().cumsum(2).cpu().numpy()
        roots = params["root"].exp().cumsum(1).cpu().numpy()
        H = params["head"].exp().cumsum(2).cpu().numpy()
        L = params["left"].exp().cumsum(2).cpu().numpy()
        R = params["right"].exp().cumsum(2).cpu().numpy()
        return terms, H, L, R, roots

    @staticmethod
    @jit(nopython=True)
    def sample_impl(
        terms: np.ndarray,  # seqlen x pt, in normal space
        rules_head: np.ndarray,  # nt x r, in normal space
        rules_left: np.ndarray,  # (nt+pt) x r, in normal space
        rules_right: np.ndarray,  # (nt+pt) x r, in normal space
        roots: np.ndarray,  # nt, in normal space
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
        NT = rules_head.shape[0]
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
                        head = weighted_random_v2(rules_head[s])
                        left = weighted_random_v2(rules_left[head])
                        right = weighted_random_v2(rules_right[head])
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


def convert_decomp1_to_pcfg(p, tgt_nt):
    rule = torch.einsum("xar,xrb,xrc->xabc", p["head"].exp(), p["left"].exp(), p["right"].exp())
    rule = (rule + 1e-9).log()
    output = {"term": p["term"], "rule": rule, "root": p["root"]}
    if "constraint" in p:
        output["constraint"] = p["constraint"]
    if "add" in p:
        output["add"] = p["add"]
    if "lse" in p:
        output["lse"] = p["lse"]
    return output


def g_merge(y, z, h, semiring):
    # y: c, bsz, n, 1, r
    # z: c, bsz, n, 1, r
    # h: c, bsz, r, nt

    # c, bsz, n, 1, r * c, bsz, n, 1, r -> c, bsz, n, r
    x = semiring.mul(y, z).squeeze(3)
    # c, bsz, n, r, 1 * c, bsz, 1, r, nt
    x = semiring.mul(x.unsqueeze(-1), h.unsqueeze(2))
    # c, bsz, n, r, nt -> c, bsz, n, nt
    x = semiring.sum(x, dim=3)
    return x


def g_merge2(y, z, y_term, z_term, h, semiring):
    # y: c, bsz, n, 1, r
    # z: c, bsz, n, 1, r
    # y_term: c, bsz, n, 1, r
    # z_term: c, bsz, n, 1, r
    # h: c, bsz, r, nt

    x1 = semiring.mul(y, z_term).squeeze(3)
    x3 = semiring.mul(y_term, z).squeeze(3)

    x = semiring.add(x1, x3)
    x = semiring.mul(x.unsqueeze(-1), h.unsqueeze(2))
    x = semiring.sum(x, dim=3)
    return x


def g_merge3(y, z, y_term, z_term, h, semiring):
    # y: c, bsz, n, 1, r
    # z: c, bsz, n, 1, r
    # y_term: c, bsz, n, 1, r
    # z_term: c, bsz, n, 1, r
    # h: c, bsz, r, nt

    x1 = semiring.mul(y[:, :, :, -1:], z_term).squeeze(3)
    x2 = semiring.sum(semiring.mul(y[:, :, :, :-1], z[:, :, :, 1:]), dim=3)
    x3 = semiring.mul(y_term, z[:, :, :, :1]).squeeze(3)

    x = semiring.add(x1, x2, x3)
    x = semiring.mul(x.unsqueeze(-1), h.unsqueeze(2))
    x = semiring.sum(x, dim=3)
    return x


if __name__ == "__main__":
    from src.models.tgt_parser.neural_decomp1 import NeuralDecomp1TgtParser
    from src.models.tgt_parser.struct.pcfg import PCFG

    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT = 2, 4, 2, 5, 3, 7
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    r = 3
    lens = [max(2, N - i) for i in range(B)]
    params = Decomp1.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    params_ref = convert_decomp1_to_pcfg(params, TGT_NT)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }

    pcfg = Decomp1(params, lens, **meta)
    pcfg_ref = PCFG()

    print("test nll")
    nll = pcfg.nll
    nll_ref = pcfg_ref(params_ref, lens)
    assert torch.allclose(nll, nll_ref), (nll, nll_ref)

    print("test marginal")
    m1 = pcfg.marginal
    check_full_marginal(m1["term"], m1["trace"], lens)

    m2 = pcfg_ref(params_ref, lens, marginal=True)[-1]
    compare_marginal(m1["trace"], m2)

    # print('test mbr decoding')
    # decoded = pcfg.mbr_decoded
    # decoded_ref = pcfg_ref(params, lens, decode=True)

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
    assert torch.allclose(cnt, prob, rtol=0.01, atol=10), (prob, cnt)

    print("test sample seq")
    spans = [[(i, i, 0) for i in range(l)] + [(0, i, 0) for i in range(1, l)] for l in lens]
    node_features = [torch.randn(l * 2 - 1, 8) for l in lens]
    parser = NeuralDecomp1TgtParser(
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
        theoratical_cfg = 0
        for prob, seq in probs_with_seq[:5]:
            empirical_cdf += count[tuple(seq)] / total
            theoratical_cfg += prob / partition
            errors.append(abs(empirical_cdf - theoratical_cfg) / theoratical_cfg)

        assert max(errors) < 0.1, errors

    print("pass")
