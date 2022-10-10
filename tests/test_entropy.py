from collections import defaultdict
from itertools import product

import pytest
import torch
from torch_struct import SentCFG

from src.models.tgt_parser.struct.decomp1 import Decomp1, convert_decomp1_to_pcfg
from src.models.tgt_parser.struct.decomp2 import Decomp2, convert_decomp2_to_pcfg
from src.models.tgt_parser.struct.decomp3 import Decomp3, convert_decomp3_to_pcfg
from src.models.tgt_parser.struct.decomp4 import Decomp4, convert_decomp4_to_pcfg
from src.models.tgt_parser.struct.decomp5 import Decomp5, convert_decomp5_to_pcfg
from src.models.tgt_parser.struct.decomp7 import Decomp7, convert_decomp7_to_pcfg
from src.models.tgt_parser.struct.no_decomp import NoDecomp
from src.utils.fn import spans2tree


def test_nodecomp_entropy():
    B = 2
    N = 4
    TGT_PT = 2
    SRC_PT = 1
    TGT_NT = 2
    SRC_NT = 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT

    lens = [max(2, N - i) for i in range(B)]
    params1 = {
        "term": torch.randn(B, N, PT).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "rule": torch.randn(B, NT, NT + PT, NT + PT)
        .flatten(2)
        .log_softmax(-1)
        .view(B, NT, NT + PT, NT + PT)
        .requires_grad_(True),
    }
    params2 = {
        "term": torch.randn(B, N, PT).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "rule": torch.randn(B, NT, NT + PT, NT + PT)
        .flatten(2)
        .log_softmax(-1)
        .view(B, NT, NT + PT, NT + PT)
        .requires_grad_(True),
    }
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }
    pcfg1 = NoDecomp(params1, lens, **meta)
    pcfg2 = NoDecomp(params2, lens, **meta)
    ent1 = pcfg1.entropy
    ent2 = pcfg2.entropy
    ce12 = pcfg1.cross_entropy(pcfg2)
    ce21 = pcfg2.cross_entropy(pcfg1)
    kl12 = pcfg1.kl(pcfg2)
    kl21 = pcfg2.kl(pcfg1)

    pcfg_ref1 = SentCFG((params1["term"], params1["rule"], params1["root"]), lens)
    pcfg_ref2 = SentCFG((params2["term"], params2["rule"], params2["root"]), lens)
    ent_ref1 = pcfg_ref1.entropy
    ent_ref2 = pcfg_ref2.entropy
    ce_ref12 = batchify(params1, params2, lens, NT, PT, ce)
    ce_ref21 = batchify(params2, params1, lens, NT, PT, ce)
    kl_ref12 = batchify(params1, params2, lens, NT, PT, kl)
    kl_ref21 = batchify(params2, params1, lens, NT, PT, kl)

    assert torch.allclose(ent1, ent_ref1, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ent2, ent_ref2, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce12, ce_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce21, ce_ref21, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl12, kl_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl21, kl_ref21, rtol=1e-3, atol=1e-6)


def test_decomp1_entropy():
    B = 2
    N = 2
    TGT_PT = 2
    SRC_PT = 1
    TGT_NT = 2
    SRC_NT = 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    r = 1

    lens = [max(2, N - i) for i in range(B)]
    params1 = Decomp1.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    params2 = Decomp1.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }
    pcfg1 = Decomp1(params1, lens, **meta)
    pcfg2 = Decomp1(params2, lens, **meta)
    ent1 = pcfg1.entropy
    ent2 = pcfg2.entropy
    ce12 = pcfg1.cross_entropy(pcfg2)
    ce21 = pcfg2.cross_entropy(pcfg1)
    kl12 = pcfg1.kl(pcfg2)
    kl21 = pcfg2.kl(pcfg1)

    params1 = convert_decomp1_to_pcfg(params1, TGT_NT)
    params2 = convert_decomp1_to_pcfg(params2, TGT_NT)
    pcfg_ref1 = SentCFG((params1["term"], params1["rule"], params1["root"]), lens)
    pcfg_ref2 = SentCFG((params2["term"], params2["rule"], params2["root"]), lens)
    ent_ref1 = pcfg_ref1.entropy
    ent_ref2 = pcfg_ref2.entropy
    ce_ref12 = batchify(params1, params2, lens, NT, PT, ce)
    ce_ref21 = batchify(params2, params1, lens, NT, PT, ce)
    kl_ref12 = batchify(params1, params2, lens, NT, PT, kl)
    kl_ref21 = batchify(params2, params1, lens, NT, PT, kl)

    assert torch.allclose(ent1, ent_ref1, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ent2, ent_ref2, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce12, ce_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce21, ce_ref21, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl12, kl_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl21, kl_ref21, rtol=1e-3, atol=1e-6)


def test_decomp2_entropy():
    B = 2
    N = 2
    TGT_PT = 2
    SRC_PT = 1
    TGT_NT = 2
    SRC_NT = 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    r = 1

    lens = [max(2, N - i) for i in range(B)]
    params1 = Decomp2.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    params2 = Decomp2.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }
    pcfg1 = Decomp2(params1, lens, **meta)
    pcfg2 = Decomp2(params2, lens, **meta)
    ent1 = pcfg1.entropy
    ent2 = pcfg2.entropy
    ce12 = pcfg1.cross_entropy(pcfg2)
    ce21 = pcfg2.cross_entropy(pcfg1)
    kl12 = pcfg1.kl(pcfg2)
    kl21 = pcfg2.kl(pcfg1)

    params1 = convert_decomp2_to_pcfg(params1, TGT_NT)
    params2 = convert_decomp2_to_pcfg(params2, TGT_NT)
    pcfg_ref1 = SentCFG((params1["term"], params1["rule"], params1["root"]), lens)
    pcfg_ref2 = SentCFG((params2["term"], params2["rule"], params2["root"]), lens)
    ent_ref1 = pcfg_ref1.entropy
    ent_ref2 = pcfg_ref2.entropy
    ce_ref12 = batchify(params1, params2, lens, NT, PT, ce)
    ce_ref21 = batchify(params2, params1, lens, NT, PT, ce)
    kl_ref12 = batchify(params1, params2, lens, NT, PT, kl)
    kl_ref21 = batchify(params2, params1, lens, NT, PT, kl)

    assert torch.allclose(ent1, ent_ref1, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ent2, ent_ref2, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce12, ce_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce21, ce_ref21, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl12, kl_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl21, kl_ref21, rtol=1e-3, atol=1e-6)


def test_decomp3_dir0_entropy():
    B = 2
    N = 2
    TGT_PT = 2
    SRC_PT = 1
    TGT_NT = 2
    SRC_NT = 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    r = 1

    lens = [max(2, N - i) for i in range(B)]
    params1 = Decomp3.random_dir0(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    params2 = Decomp3.random_dir0(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
        "direction": 0,
    }
    pcfg1 = Decomp3(params1, lens, **meta)
    pcfg2 = Decomp3(params2, lens, **meta)
    ent1 = pcfg1.entropy
    ent2 = pcfg2.entropy
    ce12 = pcfg1.cross_entropy(pcfg2)
    ce21 = pcfg2.cross_entropy(pcfg1)
    kl12 = pcfg1.kl(pcfg2)
    kl21 = pcfg2.kl(pcfg1)

    params1 = convert_decomp3_to_pcfg(params1, TGT_NT)
    params2 = convert_decomp3_to_pcfg(params2, TGT_NT)
    pcfg_ref1 = SentCFG((params1["term"], params1["rule"], params1["root"]), lens)
    pcfg_ref2 = SentCFG((params2["term"], params2["rule"], params2["root"]), lens)
    ent_ref1 = pcfg_ref1.entropy
    ent_ref2 = pcfg_ref2.entropy
    ce_ref12 = batchify(params1, params2, lens, NT, PT, ce)
    ce_ref21 = batchify(params2, params1, lens, NT, PT, ce)
    kl_ref12 = batchify(params1, params2, lens, NT, PT, kl)
    kl_ref21 = batchify(params2, params1, lens, NT, PT, kl)

    assert torch.allclose(ent1, ent_ref1, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ent2, ent_ref2, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce12, ce_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce21, ce_ref21, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl12, kl_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl21, kl_ref21, rtol=1e-3, atol=1e-6)


def test_decomp3_dir1_entropy():
    B = 2
    N = 2
    TGT_PT = 2
    SRC_PT = 1
    TGT_NT = 2
    SRC_NT = 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    r = 1

    lens = [max(2, N - i) for i in range(B)]
    params1 = Decomp3.random_dir1(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    params2 = Decomp3.random_dir1(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
        "direction": 1,
    }
    pcfg1 = Decomp3(params1, lens, **meta)
    pcfg2 = Decomp3(params2, lens, **meta)
    ent1 = pcfg1.entropy
    ent2 = pcfg2.entropy
    ce12 = pcfg1.cross_entropy(pcfg2)
    ce21 = pcfg2.cross_entropy(pcfg1)
    kl12 = pcfg1.kl(pcfg2)
    kl21 = pcfg2.kl(pcfg1)

    params1 = convert_decomp3_to_pcfg(params1, TGT_NT)
    params2 = convert_decomp3_to_pcfg(params2, TGT_NT)
    pcfg_ref1 = SentCFG((params1["term"], params1["rule"], params1["root"]), lens)
    pcfg_ref2 = SentCFG((params2["term"], params2["rule"], params2["root"]), lens)
    ent_ref1 = pcfg_ref1.entropy
    ent_ref2 = pcfg_ref2.entropy
    ce_ref12 = batchify(params1, params2, lens, NT, PT, ce)
    ce_ref21 = batchify(params2, params1, lens, NT, PT, ce)
    kl_ref12 = batchify(params1, params2, lens, NT, PT, kl)
    kl_ref21 = batchify(params2, params1, lens, NT, PT, kl)

    assert torch.allclose(ent1, ent_ref1, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ent2, ent_ref2, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce12, ce_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce21, ce_ref21, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl12, kl_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl21, kl_ref21, rtol=1e-3, atol=1e-6)


def test_decomp4_dir0_entropy():
    B = 2
    N = 2
    TGT_PT = 2
    SRC_PT = 1
    TGT_NT = 2
    SRC_NT = 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    r = 1

    lens = [max(2, N - i) for i in range(B)]
    params1 = Decomp4.random_dir0(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    params2 = Decomp4.random_dir0(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
        "direction": 0,
    }
    pcfg1 = Decomp4(params1, lens, **meta)
    pcfg2 = Decomp4(params2, lens, **meta)
    ent1 = pcfg1.entropy
    ent2 = pcfg2.entropy
    ce12 = pcfg1.cross_entropy(pcfg2)
    ce21 = pcfg2.cross_entropy(pcfg1)
    kl12 = pcfg1.kl(pcfg2)
    kl21 = pcfg2.kl(pcfg1)

    params1 = convert_decomp4_to_pcfg(params1, TGT_NT)
    params2 = convert_decomp4_to_pcfg(params2, TGT_NT)
    pcfg_ref1 = SentCFG((params1["term"], params1["rule"], params1["root"]), lens)
    pcfg_ref2 = SentCFG((params2["term"], params2["rule"], params2["root"]), lens)
    ent_ref1 = pcfg_ref1.entropy
    ent_ref2 = pcfg_ref2.entropy
    ce_ref12 = batchify(params1, params2, lens, NT, PT, ce)
    ce_ref21 = batchify(params2, params1, lens, NT, PT, ce)
    kl_ref12 = batchify(params1, params2, lens, NT, PT, kl)
    kl_ref21 = batchify(params2, params1, lens, NT, PT, kl)

    assert torch.allclose(ent1, ent_ref1, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ent2, ent_ref2, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce12, ce_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce21, ce_ref21, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl12, kl_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl21, kl_ref21, rtol=1e-3, atol=1e-6)


def test_decomp4_dir1_entropy():
    B = 2
    N = 2
    TGT_PT = 2
    SRC_PT = 1
    TGT_NT = 2
    SRC_NT = 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    r = 1

    lens = [max(2, N - i) for i in range(B)]
    params1 = Decomp4.random_dir1(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    params2 = Decomp4.random_dir1(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
        "direction": 1,
    }
    pcfg1 = Decomp4(params1, lens, **meta)
    pcfg2 = Decomp4(params2, lens, **meta)
    ent1 = pcfg1.entropy
    ent2 = pcfg2.entropy
    ce12 = pcfg1.cross_entropy(pcfg2)
    ce21 = pcfg2.cross_entropy(pcfg1)
    kl12 = pcfg1.kl(pcfg2)
    kl21 = pcfg2.kl(pcfg1)

    params1 = convert_decomp4_to_pcfg(params1, TGT_NT)
    params2 = convert_decomp4_to_pcfg(params2, TGT_NT)
    pcfg_ref1 = SentCFG((params1["term"], params1["rule"], params1["root"]), lens)
    pcfg_ref2 = SentCFG((params2["term"], params2["rule"], params2["root"]), lens)
    ent_ref1 = pcfg_ref1.entropy
    ent_ref2 = pcfg_ref2.entropy
    ce_ref12 = batchify(params1, params2, lens, NT, PT, ce)
    ce_ref21 = batchify(params2, params1, lens, NT, PT, ce)
    kl_ref12 = batchify(params1, params2, lens, NT, PT, kl)
    kl_ref21 = batchify(params2, params1, lens, NT, PT, kl)

    assert torch.allclose(ent1, ent_ref1, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ent2, ent_ref2, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce12, ce_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce21, ce_ref21, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl12, kl_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl21, kl_ref21, rtol=1e-3, atol=1e-6)


def test_decomp5_dir0_entropy():
    B = 2
    N = 2
    TGT_PT = 2
    SRC_PT = 1
    TGT_NT = 2
    SRC_NT = 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    r = 1

    lens = [max(2, N - i) for i in range(B)]
    params1 = Decomp5.random_dir0(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    params2 = Decomp5.random_dir0(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
        "direction": 0,
    }
    pcfg1 = Decomp5(params1, lens, **meta)
    pcfg2 = Decomp5(params2, lens, **meta)
    ent1 = pcfg1.entropy
    ent2 = pcfg2.entropy
    ce12 = pcfg1.cross_entropy(pcfg2)
    ce21 = pcfg2.cross_entropy(pcfg1)
    kl12 = pcfg1.kl(pcfg2)
    kl21 = pcfg2.kl(pcfg1)

    params1 = convert_decomp5_to_pcfg(params1, TGT_NT)
    params2 = convert_decomp5_to_pcfg(params2, TGT_NT)
    pcfg_ref1 = SentCFG((params1["term"], params1["rule"], params1["root"]), lens)
    pcfg_ref2 = SentCFG((params2["term"], params2["rule"], params2["root"]), lens)
    ent_ref1 = pcfg_ref1.entropy
    ent_ref2 = pcfg_ref2.entropy
    ce_ref12 = batchify(params1, params2, lens, NT, PT, ce)
    ce_ref21 = batchify(params2, params1, lens, NT, PT, ce)
    kl_ref12 = batchify(params1, params2, lens, NT, PT, kl)
    kl_ref21 = batchify(params2, params1, lens, NT, PT, kl)

    assert torch.allclose(ent1, ent_ref1, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ent2, ent_ref2, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce12, ce_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce21, ce_ref21, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl12, kl_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl21, kl_ref21, rtol=1e-3, atol=1e-6)


def test_decomp5_dir1_entropy():
    B = 2
    N = 2
    TGT_PT = 2
    SRC_PT = 1
    TGT_NT = 2
    SRC_NT = 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    r = 1

    lens = [max(2, N - i) for i in range(B)]
    params1 = Decomp5.random_dir1(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    params2 = Decomp5.random_dir1(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
        "direction": 1,
    }
    pcfg1 = Decomp5(params1, lens, **meta)
    pcfg2 = Decomp5(params2, lens, **meta)
    ent1 = pcfg1.entropy
    ent2 = pcfg2.entropy
    ce12 = pcfg1.cross_entropy(pcfg2)
    ce21 = pcfg2.cross_entropy(pcfg1)
    kl12 = pcfg1.kl(pcfg2)
    kl21 = pcfg2.kl(pcfg1)

    params1 = convert_decomp5_to_pcfg(params1, TGT_NT)
    params2 = convert_decomp5_to_pcfg(params2, TGT_NT)
    pcfg_ref1 = SentCFG((params1["term"], params1["rule"], params1["root"]), lens)
    pcfg_ref2 = SentCFG((params2["term"], params2["rule"], params2["root"]), lens)
    ent_ref1 = pcfg_ref1.entropy
    ent_ref2 = pcfg_ref2.entropy
    ce_ref12 = batchify(params1, params2, lens, NT, PT, ce)
    ce_ref21 = batchify(params2, params1, lens, NT, PT, ce)
    kl_ref12 = batchify(params1, params2, lens, NT, PT, kl)
    kl_ref21 = batchify(params2, params1, lens, NT, PT, kl)

    assert torch.allclose(ent1, ent_ref1, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ent2, ent_ref2, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce12, ce_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce21, ce_ref21, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl12, kl_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl21, kl_ref21, rtol=1e-3, atol=1e-6)


def test_decomp7_entropy():
    B = 2
    N = 2
    TGT_PT = 2
    SRC_PT = 1
    TGT_NT = 2
    SRC_NT = 1
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT
    r = 1

    lens = [max(2, N - i) for i in range(B)]
    params1 = Decomp7.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    params2 = Decomp7.random(B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r)
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }
    pcfg1 = Decomp7(params1, lens, **meta)
    pcfg2 = Decomp7(params2, lens, **meta)
    ent1 = pcfg1.entropy
    ent2 = pcfg2.entropy
    ce12 = pcfg1.cross_entropy(pcfg2)
    ce21 = pcfg2.cross_entropy(pcfg1)
    kl12 = pcfg1.kl(pcfg2)
    kl21 = pcfg2.kl(pcfg1)

    params1 = convert_decomp7_to_pcfg(params1, TGT_NT)
    params2 = convert_decomp7_to_pcfg(params2, TGT_NT)
    pcfg_ref1 = SentCFG((params1["term"], params1["rule"], params1["root"]), lens)
    pcfg_ref2 = SentCFG((params2["term"], params2["rule"], params2["root"]), lens)
    ent_ref1 = pcfg_ref1.entropy
    ent_ref2 = pcfg_ref2.entropy
    ce_ref12 = batchify(params1, params2, lens, NT, PT, ce)
    ce_ref21 = batchify(params2, params1, lens, NT, PT, ce)
    kl_ref12 = batchify(params1, params2, lens, NT, PT, kl)
    kl_ref21 = batchify(params2, params1, lens, NT, PT, kl)

    assert torch.allclose(ent1, ent_ref1, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ent2, ent_ref2, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce12, ce_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(ce21, ce_ref21, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl12, kl_ref12, rtol=1e-3, atol=1e-6)
    assert torch.allclose(kl21, kl_ref21, rtol=1e-3, atol=1e-6)


# PCFG utilities


def enumerate_pcfg_tree(i, j, nt, pt):
    if i + 1 == j:
        for t in range(pt):
            yield [(i, j, t)]

    for t in range(nt):
        spans = [(i, j, t)]
        for k in range(i + 1, j):
            for l, r in product(enumerate_pcfg_tree(i, k, nt, pt), enumerate_pcfg_tree(k, j, nt, pt)):
                yield spans + l + r


def score2(params, length, span, nt):
    s, p = spans2tree(span)
    parent2children = defaultdict(list)
    for i, pj in enumerate(p):
        parent2children[pj].append(i)

    _score = 0
    for pj, children in parent2children.items():
        if len(children) == 1:
            assert pj == -1
            _score += params["root"][0, s[children[0]][2]]
        else:
            assert len(children) == 2
            l = s[children[0]]
            lt = l[2] + (0 if l[0] != l[1] else nt)
            r = s[children[1]]
            rt = r[2] + (0 if r[0] != r[1] else nt)
            pt = s[pj][2]
            _score += params["rule"][0, pt, lt, rt]

    pts = [None for _ in range(length)]
    for sj in s:
        if sj[0] == sj[1]:
            pts[sj[0]] = sj[2]

    for i, pt in enumerate(pts):
        _score += params["term"][0, i, pt]
    return _score


def ce(params, params2, n, nt, pt):
    ll = []
    ll2 = []
    for tree in enumerate_pcfg_tree(0, n, nt, pt):
        tree = [(l, r - 1, t) for l, r, t in tree]
        stree = score2(params, n, tree, nt)
        ll.append(stree)
        ll2.append(score2(params2, n, tree, nt))

    ll = torch.tensor(ll)
    ll = ll.log_softmax(0)
    ll2 = torch.tensor(ll2)
    ll2 = ll2.log_softmax(0)
    return -(ll2 * ll.exp()).sum()


def kl(params, params2, n, nt, pt):
    ll = []
    ll2 = []
    for tree in enumerate_pcfg_tree(0, n, nt, pt):
        tree = [(l, r - 1, t) for l, r, t in tree]
        stree = score2(params, n, tree, nt)
        ll.append(stree)
        ll2.append(score2(params2, n, tree, nt))

    ll = torch.tensor(ll)
    ll = ll.log_softmax(0)
    ll2 = torch.tensor(ll2)
    ll2 = ll2.log_softmax(0)
    return (ll.exp() * (ll - ll2)).sum()


def convert(params):
    return params["term"], params["rule"], params["root"]


def batchify(params1, params2, lengths, NT, PT, func):
    result = []
    for bidx in range(len(lengths)):
        p1 = {k: v[bidx, None] for k, v in params1.items()}
        p2 = {k: v[bidx, None] for k, v in params2.items()}
        result.append(func(p1, p2, lengths[bidx], NT, PT))
    return torch.tensor(result)
