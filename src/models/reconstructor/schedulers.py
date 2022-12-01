import numpy as np


def get_frange_cycle_linear_sheduler(**kwargs):
    L = frange_cycle_linear(**kwargs)
    while True:
        i = 0
        for i in range(len(L)):
            yield L[i]


def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L
