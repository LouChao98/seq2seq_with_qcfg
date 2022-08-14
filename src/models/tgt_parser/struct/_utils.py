import numpy as np
from numba import jit
from torch.utils.checkpoint import checkpoint as torch_ckpt


def checkpoint(func):
    def wrapper(*args, **kwargs):
        # If not all argument tensors are requires_grad=True, use checkpoint will raise some errors.
        # The only case is marginal=True and the one need grad is span_indicator.
        # We do not need checkpoint in this case because no big mid tensors need to be traced.
        if all(v.requires_grad for v in args):
            return torch_ckpt(func, *args, **kwargs)
        else:
            return func(*args)

    return wrapper


@jit(nopython=True)
def weighted_random(cumsum):
    # cumsum = np.cumsum(w)
    
    rdm_unif = np.random.rand(1)
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
            if key == 'copy_nt':
                reordered_value = []
                for item in value:
                    v = item[0][argsort]
                    m = item[1][argsort]
                    reordered_value.append((v, m))
                reordered_params['copy_nt'] = reordered_value
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
