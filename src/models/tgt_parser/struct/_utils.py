import numpy as np
from numba import jit
from torch.utils.checkpoint import checkpoint as torch_ckpt


def checkpoint(func):
    def wrapper(*args, **kwargs):
        # If not all argument tensors are requires_grad=True, use checkpoint will raise some errors.
        # The only case is marginal=True and the one need grad is span_indicator.
        # We do not need checkpoint in this case because no big mid tensors need to be traced.
        if all(v.requires_grad for v in args):
            return torch_ckpt(func, *args, use_reentrant=False, **kwargs)
        else:
            return func(*args)

    return wrapper


@jit(nopython=True)
def weighted_random(cumsum):
    # cumsum = np.cumsum(w)
    rdm_unif = np.random.rand(1)
    return np.searchsorted(cumsum, rdm_unif, side="right").item()
