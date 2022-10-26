from collections import Counter
from copy import copy
from itertools import product

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models.tgt_parser.struct2.decomp7 import Decomp7Impl4
from src.models.tgt_parser.struct2.semiring import LogSemiring, SampledSemiring


def test_decomp7_sampling_given_x():
    B = 2
    N = 4
    TGT_PT = 2
    SRC_PT = 1
    TGT_NT = 2
    SRC_NT = 1
    r = 1

    lens = [max(2, N - i) for i in range(B)]
    params = Decomp7Impl4.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }
    pcfg = Decomp7Impl4(params, lens, **meta)
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
