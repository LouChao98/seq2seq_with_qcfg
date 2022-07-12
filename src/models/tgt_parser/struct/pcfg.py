from typing import Dict, Union, List
import torch
from torch_struct import SentCFG

import numpy as np
from ._utils import weighted_random


class PCFG:
    def __call__(self, params, lens, decode=False, marginal=False):
        # terms: bsz x seqlen x pt
        # rules: bsz x nt x (nt+pt) x (nt+pt)
        # roots: bsz x nt
        if "rule" not in params:
            params["rule"] = torch.einsum(
                "bxr,byr,bzr->bxyz", params["head"], params["left"], params["right"]
            ).clamp(1e-6).log()
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
        # TODO: benchmark parallelization. on batch or on sample?
        # TODO: check clamp(1e-2) is enough to filter masked rules, and do not harm the sampling too much.
        terms = params["term"].detach()
        rules = params["rule"].detach()
        roots = params["root"].detach()

        terms: torch.Tensor
        zero = terms.new_full((1,), -1e9)
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
            samples, scores = self.sample(
                terms[b],
                rules[b],
                roots[b],
                nt_spans[b],
                nt_states,
                pt_spans[b],
                pt_states,
                use_copy=use_copy,
                num_samples=num_samples,
                max_length=max_length,
            )
            sample_scores = [(sample, score) for sample, score in zip(samples, scores)]
            sample_scores.sort(key=lambda x: x[1], reverse=True)
            preds.append(sample_scores)
        return preds

    @staticmethod
    # @jit(nopython=True, parallel=True)  # TODO solve error
    def sample(
        terms: np.ndarray,  # pt x t, in normal space
        rules: np.ndarray,  # nt x (nt+pt) x (nt+pt), in normal space
        roots: np.ndarray,  # nt, in normal space
        nt_spans: List[List[str]],
        nt_states: int,
        pt_spans: List[List[str]],
        pt_states: int,
        use_copy=True,
        num_samples=1,
        max_length=100,
        max_actions=100,
        UNK=1,
    ):
        # TODO: fix scores, rules/terms/roots are cumsum. so current impl is wrong.
        # NOTE: this debug has no effect if check_ppl=True.
        NT = rules.shape[0]
        PT = terms.shape[0]
        S = NT + PT
        COPY_NT = nt_states - 1
        COPY_PT = pt_states - 1
        samples = [None for _ in range(num_samples)]
        scores = [None for _ in range(num_samples)]
        nt_num_nodes = len(nt_spans)
        pt_num_nodes = len(pt_spans)

        for i in range(num_samples):
            actions = 0
            sample = weighted_random(roots)
            score = roots[sample]
            nonterminals: List[int] = [sample]
            preterminals: List[Union[List[str], int]] = []

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
                            preterminals.append(nt_spans[nt_node])
                            continue
                    actions += 1
                    sample = weighted_random(rules[s])
                    score += rules[s, sample]
                    left = sample // S
                    right = sample % S
                    nonterminals.extend([left, right])
                else:
                    preterminals.append(s - NT)

            preterminals = preterminals[::-1]
            terminals: List[Union[str, int]] = []
            for s in preterminals:
                if isinstance(s, list):
                    # copy in NT
                    terminals.extend(s)
                else:
                    src_pt_state = s // pt_num_nodes
                    if use_copy and src_pt_state == COPY_PT:
                        src_node = s % pt_num_nodes
                        terminals.extend(pt_spans[src_node])
                    else:
                        sample = weighted_random(terms[s])
                        score += terms[s, sample]
                        if use_copy and sample == UNK:
                            # force <unk> tokens to copy
                            src_node = s % pt_num_nodes
                            terminals.extend(pt_spans[src_node])
                        else:
                            terminals.append(sample)
            samples[i] = terminals
            scores[i] = score / len(terminals)
        return samples, scores


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

