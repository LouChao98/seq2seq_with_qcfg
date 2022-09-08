import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models.tgt_parser.struct.d1_pcfg import D1PCFG
from src.models.tgt_parser.struct.d1_pcfg_flex import D1PCFGFlex
from src.models.tgt_parser.struct.pcfg import PCFG
from src.models.tgt_parser.struct.pcfg_rdp import PCFGRandomizedDP


@given(st.data())
@settings(max_examples=10, deadline=None)
def test_d1_pcfg_flex(data):
    B = data.draw(st.integers(min_value=1, max_value=3), label="b")
    N = data.draw(st.integers(min_value=2, max_value=4), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_pt")
    SRC_PT = data.draw(st.integers(min_value=1, max_value=4), label="src_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_nt")
    SRC_NT = data.draw(st.integers(min_value=1, max_value=4), label="src_nt")
    r = data.draw(st.integers(min_value=1, max_value=4), label="r")

    NT = TGT_NT * SRC_NT
    T = TGT_PT * SRC_PT
    lens = [max(2, N - i) for i in range(B)]

    slr = torch.rand(B, r, SRC_NT, SRC_NT + SRC_PT, SRC_NT + SRC_PT)
    slr[..., :SRC_NT, :SRC_NT] /= slr[..., :SRC_NT, :SRC_NT].sum((3, 4), keepdim=True)
    slr[..., SRC_NT:, :SRC_NT] /= slr[..., SRC_NT:, :SRC_NT].sum((3, 4), keepdim=True)
    slr[..., :SRC_NT, SRC_NT:] /= slr[..., :SRC_NT, SRC_NT:].sum((3, 4), keepdim=True)
    slr[..., SRC_NT:, SRC_NT:] /= slr[..., SRC_NT:, SRC_NT:].sum((3, 4), keepdim=True)

    params = {
        "term": torch.randn(B, N, T).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
        "left": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
        "right": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
        "slr": slr,
    }

    pcfg = D1PCFGFlex(TGT_NT, TGT_PT)
    nll = pcfg(params, lens)

    pcfg_ref = PCFG()
    nll_ref = pcfg_ref(D1PCFGFlex.get_pcfg_rules(params, TGT_NT), lens)

    assert torch.allclose(nll, nll_ref)

    m1 = pcfg(params, lens, marginal=True)
    assert torch.allclose(
        m1.sum((1, 2, 3, 4)),
        torch.tensor([item * 2 - 1 for item in lens], dtype=torch.float),
    )

    m2 = pcfg_ref(D1PCFGFlex.get_pcfg_rules(params, TGT_NT), lens, marginal=True)[-1]
    assert torch.allclose(
        m1.diagonal(2, dim1=1, dim2=2).sum((1, 2)), m2[:, 0, :-1].sum(-1)
    )


@given(st.data())
@settings(max_examples=10, deadline=None)
def test_d1_pcfg(data):
    B = data.draw(st.integers(min_value=1, max_value=3), label="b")
    N = data.draw(st.integers(min_value=2, max_value=4), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_pt")
    SRC_PT = data.draw(st.integers(min_value=1, max_value=4), label="src_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_nt")
    SRC_NT = SRC_PT
    r = data.draw(st.integers(min_value=1, max_value=4), label="r")

    NT = TGT_NT * SRC_NT
    T = TGT_PT * SRC_PT
    lens = [max(2, N - i) for i in range(B)]

    slr = (
        torch.randn(B, r, SRC_NT, SRC_NT, SRC_NT)
        .view(B, r * SRC_NT, -1)
        .softmax(-1)
        .view(B, r, SRC_NT, SRC_NT, SRC_NT)
    )

    params = {
        "term": torch.randn(B, N, T).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
        "left": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
        "right": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
        "slr": slr,
    }

    pcfg = D1PCFG(TGT_NT, TGT_PT)
    nll = pcfg(params, lens)

    pcfg_ref = PCFG()
    nll_ref = pcfg_ref(D1PCFG.get_pcfg_rules(params, TGT_NT), lens)

    assert torch.allclose(nll, nll_ref)

    m1 = pcfg(params, lens, marginal=True)
    assert torch.allclose(
        m1.sum((1, 2, 3, 4)),
        torch.tensor([item * 2 - 1 for item in lens], dtype=torch.float),
    )

    m2 = pcfg_ref(D1PCFG.get_pcfg_rules(params, TGT_NT), lens, marginal=True)[-1]
    assert torch.allclose(
        m1.diagonal(2, dim1=1, dim2=2).sum((1, 2)), m2[:, 0, :-1].sum(-1)
    )


@given(st.data())
@settings(max_examples=2, deadline=None)
def test_rdp(data):
    B = data.draw(st.integers(min_value=1, max_value=3), label="b")
    N = data.draw(st.integers(min_value=2, max_value=4), label="n")
    T = data.draw(st.integers(min_value=1, max_value=4), label="t")
    NT = data.draw(st.integers(min_value=1, max_value=4), label="nt")
    TOPK = data.draw(st.integers(min_value=1, max_value=4), label="topk")
    SAMPLE = data.draw(st.integers(min_value=1, max_value=4), label="sample")
    SMOOTH = data.draw(st.floats(min_value=1e-4, max_value=1), label="smooth")
    lens = [max(2, N - i) for i in range(B)]

    params = {
        "term": torch.randn(B, N, T).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "rule": torch.randn(B, NT, (NT + T) ** 2)
        .log_softmax(-1)
        .view(B, NT, NT + T, NT + T)
        .requires_grad_(True),
    }
    pcfg = PCFGRandomizedDP(TOPK, SAMPLE, SMOOTH)

    with torch.no_grad():
        nll = torch.zeros(B)
        for _ in range(100):
            nll += pcfg(params, lens)
        nll /= 100

        pcfg_ref = PCFG()
        nll_ref = pcfg_ref(params, lens)

    assert ((nll - nll_ref).abs() < 0.05).all()
