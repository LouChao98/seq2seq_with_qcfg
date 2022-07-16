from typing import List

import numpy as np
import torch
from numba import jit
from torch_struct import SentCFG

from ._utils import weighted_random

from enum import IntEnum


# I don't know how to use IntEnum with numba's jit.
# So I use this workaround.
_VOCAB, _COPY_NT, _COPY_PT = 0, 1, 2

class TokenType(IntEnum):
    VOCAB = _VOCAB
    COPY_NT = _COPY_NT
    COPY_PT = _COPY_PT


class PCFG:
    VOCAB = 0
    COPY_PT = 1
    COPY_NT = 2

    def __call__(self, params, lens, decode=False, marginal=False):
        # terms: bsz x seqlen x pt
        # rules: bsz x nt x (nt+pt) x (nt+pt)
        # roots: bsz x nt
        if "rule" not in params:
            params["rule"] = (
                torch.einsum(
                    "bxr,byr,bzr->bxyz", params["head"], params["left"], params["right"]
                )
                .clamp(1e-6)
                .log()
            )
            assert not params["rule"].isnan().any()

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

    @torch.no_grad()
    def sampled_decoding(
        self,
        params,
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

        terms: torch.Tensor
        zero = terms.new_full((1,), -1e9)
        # TODO: check clamp(1e-2) is enough to filter masked rules, and do not harm the sampling too much.
        threshold = terms.new_full((1,), np.log(1e-2))
        terms = torch.where(terms > threshold, terms, zero).softmax(2).cumsum(2)
        rules = (
            torch.where(rules > threshold, rules, zero)
            .view(*rules.shape[:2], -1)
            .softmax(2)
            .cumsum(2)
        )
        roots = torch.where(roots > threshold, roots, zero).softmax(1).cumsum(1)
        terms[:, :, -1] += 1  # avoid out of bound
        rules[:, :, -1] += 1
        roots[:, -1] += 1

        terms = terms.cpu().numpy()
        rules = rules.cpu().numpy()
        roots = roots.cpu().numpy()

        preds = []
        for b in range(len(terms)):
            samples, types, scores = self.sample(
                terms[b],
                rules[b],
                roots[b],
                len(nt_spans[b]),
                nt_states,
                len(pt_spans[b]),
                pt_states,
                use_copy=use_copy,
                num_samples=num_samples,
                max_length=max_length,
            )
            sample_scores = [
                (sample, type_, score)
                for sample, type_, score in zip(samples, types, scores)
            ]
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
        # TODO: fix scores, rules/terms/roots are cumsum. so current impl is wrong.
        # NOTE: this bug has no effect if check_ppl=True.
        #
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
            score = roots[sample]
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
                    score += rules[s, sample]
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
                        score += terms[s, sample]
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
            scores[i] = score / len(terminals)
        return samples, types, scores

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
            score = roots[sample]
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
                    score += rules[s, sample]
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
                        score += terms[s, sample]
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
            scores[i] = score / len(terminals)
        return samples, types, scores


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

