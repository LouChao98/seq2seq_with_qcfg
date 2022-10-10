import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from numba import jit
from torch import Tensor

from ._fn import diagonal_copy_, stripe
from ._utils import checkpoint, process_param_for_trace, weighted_random_v2
from .decomp_base import _COPY_NT, _COPY_PT, _OK, _REACHLIMIT, _SONMASK, _VOCAB, DecompBase, DecompSamplerBase

log = logging.getLogger(__file__)


class NoDecomp(DecompBase):
    KEYS = ("term", "rule", "root")
    LOGSPACE = (True, True, True)

    def inside(self, trace) -> Tuple[Tensor, Tensor]:
        if trace:
            params = {k: process_param_for_trace(v) for k, v in self.params.items()}
        else:
            params = self.params

        term: Tensor = params["term"]
        rule: Tensor = params["rule"]
        root: Tensor = params["root"]
        constraint = params.get("constraint")
        lse_scores = params.get("lse")
        add_scores = params.get("add")

        batch, N, PT = term.shape
        NT = root.shape[1]
        MT = max(NT, PT)
        N += 1

        NTNT = rule[..., :NT, :NT]
        NTPT = rule[..., :NT, NT:]
        PTNT = rule[..., NT:, :NT]
        PTPT = rule[..., NT:, NT:]

        if trace:
            span_indicator = term.new_zeros(batch, N, N, MT, requires_grad=True)
            span_indicator_running = span_indicator[..., :NT]
        else:
            span_indicator = None
            span_indicator_running = None

        s = term.new_full((batch, N, N, MT), -1e9)
        if trace:
            indicator = span_indicator[..., :PT].diagonal(1, 1, 2).movedim(-1, 1)
            term = term + indicator
        diagonal_copy_(s, term, w=1, s3=PT)

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (torch.arange(2, N + 1).unsqueeze(1) <= self.lens.cpu().unsqueeze(0)).sum(1)

        # w: span width
        final = []
        for step, w in enumerate(range(2, N)):

            # n: the number of spans of width w.
            n = N - w
            y = stripe(s, n, w - 1, (0, 1))
            z = stripe(s, n, w - 1, (1, w), 0)

            if w == 2:
                x = merge(y, z, PTPT)
            elif w == 3:
                x = merge2(y, z, PTNT, NTPT)
            else:
                x = merge3(y, z, PTNT, NTPT, NTNT)
            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            if constraint is not None:
                value, mask = constraint[step]
                if value.ndim > 0:
                    value = value[:current_bsz]
                mask = mask[:current_bsz]
                x = torch.where(mask, value, x)
            if add_scores is not None:
                x = x + add_scores[step]
            if lse_scores is not None:
                x = torch.logaddexp(x, lse_scores[step])

            if trace:
                indicator = span_indicator_running.diagonal(w, 1, 2).movedim(-1, 1)
                x = x + indicator

            if current_bsz - unfinished > 0:
                final.insert(0, x[unfinished:current_bsz, :1])

            if unfinished > 0:
                if current_bsz > unfinished:
                    x = x[:unfinished]
                    s = s[:unfinished]
                    PTPT = PTPT[:unfinished]
                    PTNT = PTNT[:unfinished]
                    NTPT = NTPT[:unfinished]
                    NTNT = NTNT[:unfinished]
                    if trace:
                        span_indicator_running = span_indicator_running[:unfinished]
                diagonal_copy_(s, x, w, s3=NT)

            if unfinished == 0:
                break

        final = torch.cat(final, dim=0).squeeze(1) + root
        logZ = final.logsumexp(-1)
        return logZ, span_indicator


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


@checkpoint
@torch.jit.script
def merge(y, z, ptpt):
    pt = ptpt.shape[2]
    x = y[..., 0, :pt, None] + z[..., 0, None, :pt]
    x = (x.unsqueeze(2) + ptpt[:, None]).flatten(3).logsumexp(-1)
    return x


@checkpoint
@torch.jit.script
def merge2(y, z, ptnt, ntpt):
    pt, nt = ptnt.shape[-2:]
    x1 = y[:, :, 0, :pt, None] + z[:, :, 0, None, :nt]
    x3 = y[:, :, 1, :nt, None] + z[:, :, 1, None, :pt]
    x = torch.stack(
        [
            (x1.unsqueeze(2) + ptnt[:, None]).flatten(3).logsumexp(-1),
            (x3.unsqueeze(2) + ntpt[:, None]).flatten(3).logsumexp(-1),
        ],
        dim=-1,
    )
    return x.logsumexp(-1)


@checkpoint
@torch.jit.script
def merge3(y, z, ptnt, ntpt, ntnt):
    pt, nt = ptnt.shape[-2:]
    x1 = y[:, :, 0, :pt, None] + z[:, :, 0, None, :nt]
    x2 = (y[:, :, 1:-1, :nt, None] + z[:, :, 1:-1, None, :nt]).logsumexp(2)
    x3 = y[:, :, -1, :nt, None] + z[:, :, -1, None, :pt]
    x = torch.stack(
        [
            (x1.unsqueeze(2) + ptnt[:, None]).flatten(3).logsumexp(-1),
            (x2.unsqueeze(2) + ntnt[:, None]).flatten(3).logsumexp(-1),
            (x3.unsqueeze(2) + ntpt[:, None]).flatten(3).logsumexp(-1),
        ],
        dim=-1,
    )
    return x.logsumexp(-1)
