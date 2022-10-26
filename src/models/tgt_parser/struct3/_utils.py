from itertools import product

import numpy as np
import torch
from numba import jit
from torch.utils.checkpoint import checkpoint as torch_ckpt


def checkpoint(func):
    def wrapper(*args, **kwargs):
        # If not all argument tensors are requires_grad=True, use checkpoint will raise some errors.
        # The only case is marginal=True and the one need grad is span_indicator.
        # We do not need checkpoint in this case because no big mid tensors need to be traced.
        if any(v.requires_grad for v in args):
            return torch_ckpt(func, *args, **kwargs)
        else:
            return func(*args)

    return wrapper


@jit(nopython=True)
def weighted_random(cumsum):
    # cumsum = np.cumsum(w)

    rdm_unif = np.random.rand() * (cumsum[-1] - 1e-8)
    ind = np.searchsorted(cumsum, rdm_unif, side="right").item()
    return ind


@jit(nopython=True)
def weighted_random_v2(cumsum):
    if cumsum[-1] < 1e-6:
        raise ValueError("Sampling on masked NT.")
    rdm_unif = np.random.rand() * (cumsum[-1] - 1e-8)
    ind = np.searchsorted(cumsum, rdm_unif, side="right").item()
    return ind


def reorder(func):
    def wrapper(self, params, lens, *args, **kwargs):
        is_ordered = True
        for i in range(1, len(lens)):
            if lens[i - 1] < lens[i]:
                is_ordered = False
                break
        if is_ordered:
            return func(self, params, lens, *args, **kwargs)

        argsort = list(range(len(lens)))
        argsort.sort(key=lambda i: lens[i], reverse=True)

        reordered_params = {}
        for key, value in params.items():
            if key == "copy_nt":
                if value is None:
                    continue
                reordered_value = []
                for item in value:
                    if item[0].ndim > 0:
                        v = item[0][argsort]
                    else:
                        v = item[0]
                    m = item[1][argsort]
                    reordered_value.append((v, m))
                reordered_params["copy_nt"] = reordered_value
            else:
                reordered_params[key] = value[argsort]

        reordered_lens = [lens[i] for i in argsort]
        output = func(self, reordered_params, reordered_lens, *args, **kwargs)

        recovery_order = [None for _ in lens]
        for i, v in enumerate(argsort):
            recovery_order[v] = i

        if isinstance(output, list):
            output = [output[i] for i in recovery_order]
        else:
            output = output[recovery_order]
        return output

    return wrapper


def process_param_for_trace(item):
    if item is None:
        return item
    elif isinstance(item, torch.Tensor):
        if item.is_inference():
            item = item.clone()
        # else:
        #     item = item.detach()
        if torch.is_floating_point(item):
            if not item.requires_grad:
                return item.requires_grad_()
            item.retain_grad()
            return item
        else:
            return item
    elif isinstance(item, (list, tuple)):
        return [process_param_for_trace(i) for i in item]
    raise NotImplementedError


def compare_marginal(m1, m2):
    # m1: (l, r, t)
    # m2: (w, l, t)
    m1 = m1.contiguous().flatten(3)
    for i in range(m1.shape[1] - 2):
        if not torch.allclose(
            m1.diagonal(2 + i, dim1=1, dim2=2).transpose(1, 2), m2[:, i, : -i - 1], rtol=1e-4, atol=1e-6
        ):
            assert False


def check_full_marginal(term_m, trace_m, lens):
    if term_m.ndim == 3:
        assert torch.allclose(term_m.flatten(1).sum(1), torch.tensor(lens, dtype=torch.float))
        assert torch.allclose(trace_m.flatten(1).sum(1), torch.tensor(lens, dtype=torch.float) - 1)
    else:
        raise NotImplementedError


def compute_unnormalized_prob(seq, parser, pred):
    x = torch.tensor([seq])
    pred = parser.observe_x(pred, x, [len(seq)], inplace=False)
    return pred.dist.partition.exp().item()


def enumerate_seq(length, vocab):
    v = list(range(vocab))
    for i in range(2, length + 1):
        for x in product(*([v] * i)):
            yield x
