from enum import IntEnum
from typing import Dict, List

import numpy as np
import torch
from numba import jit, prange
from torch import Tensor
from torch_struct import SentCFG

from ._utils import process_param_for_marginal, weighted_random

# I don't know how to use IntEnum with numba's jit.
# So I use this workaround.
# TODO can we move this to _utils but not break numba? not tested.
_VOCAB, _COPY_NT, _COPY_PT = 0, 1, 2


class TokenType(IntEnum):
    VOCAB = _VOCAB
    COPY_NT = _COPY_NT
    COPY_PT = _COPY_PT


class PCFG:
    VOCAB = 0
    COPY_PT = 1
    COPY_NT = 2

    def __init__(self):
        self.threshold = torch.nn.Threshold(1e-3, 0)

    def __call__(self, params, lens, decode=False, marginal=False):
        # terms: bsz x seqlen x pt
        # rules: bsz x nt x (nt+pt) x (nt+pt)
        # roots: bsz x nt

        terms, rules, roots = params["term"], params["rule"], params["root"]
        params = (
            terms,
            rules,
            roots,
            params.get("copy_nt"),
            None,
            params.get("add_scores"),
        )

        if decode or marginal:
            grad_state = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            cm = torch.inference_mode(False)
            cm.__enter__()
            params = [process_param_for_marginal(item) for item in params]
        try:  # this try will not catch anything. just use finally.
            dist = SentCFG(params, lens)
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
        finally:
            if decode or marginal:
                torch.set_grad_enabled(grad_state)
                cm.__exit__(None, None, None)

    @torch.no_grad()
    def sampled_decoding(
        self,
        params: Dict[str, Tensor],
        nt_spans,
        nt_states,
        pt_spans,
        pt_states,
        use_copy=True,
        num_samples=10,
        max_length=100,
    ):
        terms = params["term"].detach()
        rules = params["rule"].detach()
        roots = params["root"].detach()

        terms = self.threshold(terms.softmax(2)).cumsum(2)
        rules = self.threshold(rules.flatten(2).softmax(2)).cumsum(2)
        roots = self.threshold(roots.softmax(1)).cumsum(1)
        terms = terms.cpu().numpy()
        rules = rules.cpu().numpy()
        roots = roots.cpu().numpy()

        max_nt_spans = max(len(item) for item in nt_spans)
        max_pt_spans = max(len(item) for item in pt_spans)

        preds = []
        for b in range(len(terms)):
            samples, types, scores = self.sample(
                terms[b],
                rules[b],
                roots[b],
                max_nt_spans,
                nt_states,
                max_pt_spans,
                pt_states,
                use_copy=use_copy,
                num_samples=num_samples,
                max_length=max_length,
            )
            sample_scores = [
                (sample, type_, score)
                for sample, type_, score in zip(samples, types, scores)
                if len(sample) > 0
            ]  # len=0 when max_actions is reached but no PT rules applied
            if len(sample_scores) == 0:
                sample_scores = [([0, 0], [TokenType.VOCAB, TokenType.VOCAB], 0)]
            preds.append(sample_scores)
        return preds

    @staticmethod
    @jit(nopython=True)
    def sample(
        terms: np.ndarray,  # pt x t, in normal space
        rules: np.ndarray,  # nt x (nt+pt) x (nt+pt), in normal space
        roots: np.ndarray,  # nt, in normal space
        nt_num_nodes: int,
        nt_states: int,
        pt_num_nodes: int,
        pt_states: int,
        use_copy=True,
        num_samples=1,
        max_length=100,
        max_actions=100,
        UNK=1,
    ):
        # Kim's impl derive the rightmost NT first. But I change it to left first.
        # My order (LL) should be more nature to combine with LM, as the generation order
        # of mine is left-to-right.
        # TODO inspect how often the sampling reaches max_length and max_actions.
        # TODO try bfs.
        #
        # Kim's impl seems to be wrong when deriving COPY. The order of left/right PT
        # appended to the buffer in his impl should be reversed.

        NT = rules.shape[0]
        PT = terms.shape[0]
        S = NT + PT
        COPY_NT = nt_states - 1
        COPY_PT = pt_states - 1
        samples = [[0] for _ in range(num_samples)]
        types = [[0] for _ in range(num_samples)]
        scores = [0.0 for _ in range(num_samples)]

        for i in range(num_samples):
            actions = 0
            sample = weighted_random(roots)
            # score = roots[sample]
            nonterminals: List[int] = [sample]
            preterminals: List[int] = []
            is_copy_pt: List[bool] = []

            while (
                len(nonterminals) > 0
                and len(preterminals) < max_length
                and actions < max_actions
            ):
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
                    sample = weighted_random(rules[s])
                    # score += rules[s, sample]
                    left, right = divmod(sample, S)
                    nonterminals.extend([right, left])
                else:
                    preterminals.append(s - NT)
                    is_copy_pt.append(False)

            terminals: List[int] = []
            terminal_type: List[int] = []  # 0=vocab, 1=nt span, 2=pt span
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
                        sample = weighted_random(terms[s])
                        # score += terms[s, sample]
                        if use_copy and sample == UNK:
                            # force <unk> tokens to copy
                            src_node = s % pt_num_nodes
                            terminals.append(src_node)
                            terminal_type.append(_COPY_PT)
                        else:
                            terminals.append(sample)
                            terminal_type.append(_VOCAB)
            samples[i] = terminals
            types[i] = terminal_type
            # scores[i] = score / (len(terminals) + 1e-9)
        return samples, types, scores

    @staticmethod
    def sample_inspect(
        terms: np.ndarray,  # pt x t, in normal space
        rules: np.ndarray,  # nt x (nt+pt) x (nt+pt), in normal space
        roots: np.ndarray,  # nt, in normal space
        nt_num_nodes: int,
        nt_states: int,
        pt_num_nodes: int,
        pt_states: int,
        use_copy=True,
        num_samples=1,
        max_length=100,
        max_actions=100,
        UNK=1,
    ):
        # Kim's impl derive the rightmost NT first. But I change it to left first.
        # My order (LL) should be more nature to combine with LM, as the generation order
        # of mine is left-to-right.
        # TODO inspect how often the sampling reaches max_length and max_actions.
        # TODO try bfs.
        #
        # Kim's impl seems to be wrong when deriving COPY. The order of left/right PT
        # appended to the buffer in his impl should be reversed.

        NT = rules.shape[0]
        PT = terms.shape[0]
        S = NT + PT
        COPY_NT = nt_states - 1
        COPY_PT = pt_states - 1
        samples = [[0] for _ in range(num_samples)]
        types = [[0] for _ in range(num_samples)]
        scores = [0.0 for _ in range(num_samples)]
        trajectories = [[] for _ in range(num_samples)]

        for i in range(num_samples):
            trajectory = []
            actions = 0
            sample = weighted_random(roots)
            trajectory.append(("r", sample))
            # score = roots[sample]
            nonterminals: List[int] = [sample]
            preterminals: List[int] = []
            is_copy_pt: List[bool] = []

            while (
                len(nonterminals) > 0
                and len(preterminals) < max_length
                and actions < max_actions
            ):
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
                    sample = weighted_random(rules[s])
                    trajectory.append(("r", sample))
                    # score += rules[s, sample]
                    left, right = divmod(sample, S)
                    nonterminals.extend([right, left])
                else:
                    preterminals.append(s - NT)
                    is_copy_pt.append(False)

            terminals: List[int] = []
            terminal_type: List[int] = []  # 0=vocab, 1=nt span, 2=pt span
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
                        sample = weighted_random(terms[s])
                        trajectory.append(("t", sample))
                        # score += terms[s, sample]
                        if use_copy and sample == UNK:
                            # force <unk> tokens to copy
                            src_node = s % pt_num_nodes
                            terminals.append(src_node)
                            terminal_type.append(_COPY_PT)
                        else:
                            terminals.append(sample)
                            terminal_type.append(_VOCAB)
            samples[i] = terminals
            types[i] = terminal_type
            trajectories[i] = trajectory
            # scores[i] = score / (len(terminals) + 1e-9)
        return samples, types, scores, trajectories

    @staticmethod
    @jit(nopython=True)
    def sample_right_first(
        terms: np.ndarray,  # pt x t, in normal space
        rules: np.ndarray,  # nt x (nt+pt) x (nt+pt), in normal space
        roots: np.ndarray,  # nt, in normal space
        nt_num_nodes: int,
        nt_states: int,
        pt_num_nodes: int,
        pt_states: int,
        use_copy=True,
        num_samples=1,
        max_length=100,
        max_actions=100,
        UNK=1,
    ):
        NT = rules.shape[0]
        PT = terms.shape[0]
        S = NT + PT
        COPY_NT = nt_states - 1
        COPY_PT = pt_states - 1
        samples = [[0] for _ in range(num_samples)]
        types = [[0] for _ in range(num_samples)]
        scores = [0.0 for _ in range(num_samples)]

        for i in range(num_samples):
            actions = 0
            sample = weighted_random(roots)
            # score = roots[sample]
            nonterminals: List[int] = [sample]
            preterminals: List[int] = []
            is_copy_pt: List[bool] = []

            while (
                len(nonterminals) > 0
                and len(preterminals) < max_length
                and actions < max_actions
            ):
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
                    sample = weighted_random(rules[s])
                    # score += rules[s, sample]
                    left, right = divmod(sample, S)
                    nonterminals.extend([left, right])
                else:
                    preterminals.append(s - NT)
                    is_copy_pt.append(False)

            preterminals = preterminals[::-1]
            is_copy_pt = is_copy_pt[::-1]

            terminals: List[int] = []
            terminal_type: List[int] = []  # 0=vocab, 1=nt span, 2=pt span
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
                        sample = weighted_random(terms[s])
                        # score += terms[s, sample]
                        if use_copy and sample == UNK:
                            # force <unk> tokens to copy
                            src_node = s % pt_num_nodes
                            terminals.append(src_node)
                            terminal_type.append(_COPY_PT)
                        else:
                            terminals.append(sample)
                            terminal_type.append(_VOCAB)
            samples[i] = terminals
            types[i] = terminal_type
            # scores[i] = score / len(terminals)
        return samples, types, scores


if __name__ == "__main__":
    torch.random.manual_seed(1)

    B, N, T, NT = 2, 5, 3, 7
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
    lens = torch.tensor([N, N - 3], dtype=torch.long, device=device)
    pcfg = PCFG()

    params2 = (params["term"], params["rule"], params["root"])
    dist = SentCFG(params2, lens)
    print(dist.gumbel_crf()[-1].nonzero())

    import torch_struct

    samples = dist._struct(torch_struct.SampledSemiring).marginals(
        dist.log_potentials, lengths=dist.lengths
    )
    print(samples[-1].nonzero())
