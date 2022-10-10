import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models.tgt_parser.struct.decomp1 import Decomp1, convert_decomp1_to_pcfg
from src.models.tgt_parser.struct.decomp2 import Decomp2, convert_decomp2_to_pcfg
from src.models.tgt_parser.struct.decomp3 import Decomp3, convert_decomp3_to_pcfg
from src.models.tgt_parser.struct.decomp4 import Decomp4, convert_decomp4_to_pcfg
from src.models.tgt_parser.struct.decomp5 import Decomp5, convert_decomp5_to_pcfg
from src.models.tgt_parser.struct.decomp7 import Decomp7, convert_decomp7_to_pcfg
from src.models.tgt_parser.struct.no_decomp import NoDecomp
from src.models.tgt_parser.struct.pcfg import PCFG
from src.models.tgt_parser.struct.pcfg_rdp import PCFGRandomizedDP

# from src.models.tgt_parser.struct.decomp5 import Decomp5, convert_decomp5_to_pcfg
# from src.models.tgt_parser.struct.decomp6 import Decomp6, convert_decomp6_to_pcfg

# NOTE term rules should be gather from B x T x VOCAB
#   but I just use a B x N x T with a normalization.
#   It should be ok for testing.


# @given(st.data())
# @settings(max_examples=100, deadline=None)
# def test_d1_pcfg_flex_dir0(data):
#     B = data.draw(st.integers(min_value=1, max_value=3), label="b")
#     N = data.draw(st.integers(min_value=2, max_value=4), label="n")
#     TGT_PT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_pt")
#     SRC_PT = data.draw(st.integers(min_value=1, max_value=4), label="src_pt")
#     TGT_NT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_nt")
#     SRC_NT = data.draw(st.integers(min_value=1, max_value=4), label="src_nt")
#     r = data.draw(st.integers(min_value=1, max_value=4), label="r")

#     NT = TGT_NT * SRC_NT
#     T = TGT_PT * SRC_PT
#     lens = [max(2, N - i) for i in range(B)]

#     slr = torch.rand(B, r, SRC_NT, SRC_NT + SRC_PT, SRC_NT + SRC_PT)
#     slr[..., :SRC_NT, :SRC_NT] /= slr[..., :SRC_NT, :SRC_NT].sum((3, 4), keepdim=True)
#     slr[..., SRC_NT:, :SRC_NT] /= slr[..., SRC_NT:, :SRC_NT].sum((3, 4), keepdim=True)
#     slr[..., :SRC_NT, SRC_NT:] /= slr[..., :SRC_NT, SRC_NT:].sum((3, 4), keepdim=True)
#     slr[..., SRC_NT:, SRC_NT:] /= slr[..., SRC_NT:, SRC_NT:].sum((3, 4), keepdim=True)

#     params = {
#         "term": torch.randn(B, N, T).log_softmax(-1).requires_grad_(True),
#         "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
#         "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
#         "left": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
#         "right": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
#         "slr": slr,
#     }

#     pcfg = D1PCFGFlex(TGT_NT, TGT_PT)
#     nll = pcfg(params, lens)

#     pcfg_ref = PCFG()
#     nll_ref = pcfg_ref(D1PCFGFlex.get_pcfg_rules(params, TGT_NT), lens)

#     assert torch.allclose(nll, nll_ref)

#     m1 = pcfg(params, lens, marginal=True)
#     assert torch.allclose(
#         m1.sum((1, 2, 3, 4)),
#         torch.tensor([item * 2 - 1 for item in lens], dtype=torch.float),
#     )

#     m2 = pcfg_ref(D1PCFGFlex.get_pcfg_rules(params, TGT_NT), lens, marginal=True)[-1]
#     assert torch.allclose(
#         m1.diagonal(2, dim1=1, dim2=2).sum((1, 2)), m2[:, 0, :-1].sum(-1)
#     )


# @given(st.data())
# @settings(max_examples=100, deadline=None)
# def test_d1_pcfg_flex_dir1(data):
#     B = data.draw(st.integers(min_value=1, max_value=3), label="b")
#     N = data.draw(st.integers(min_value=2, max_value=4), label="n")
#     TGT_PT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_pt")
#     SRC_PT = data.draw(st.integers(min_value=1, max_value=4), label="src_pt")
#     TGT_NT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_nt")
#     SRC_NT = data.draw(st.integers(min_value=1, max_value=4), label="src_nt")
#     r = data.draw(st.integers(min_value=1, max_value=4), label="r")

#     NT = TGT_NT * SRC_NT
#     T = TGT_PT * SRC_PT
#     lens = [max(2, N - i) for i in range(B)]

#     slr = torch.rand(B, r, SRC_NT, SRC_NT + SRC_PT, SRC_NT + SRC_PT)
#     slr /= slr.sum((3, 4), keepdim=True)

#     left = torch.rand(B, r, TGT_NT + TGT_PT)
#     left[..., :TGT_NT] /= left[..., :TGT_NT].sum(-1, keepdim=True)
#     left[..., TGT_NT:] /= left[..., TGT_NT:].sum(-1, keepdim=True)

#     right = torch.rand(B, r, TGT_NT + TGT_PT)
#     right[..., :TGT_NT] /= right[..., :TGT_NT].sum(-1, keepdim=True)
#     right[..., TGT_NT:] /= right[..., TGT_NT:].sum(-1, keepdim=True)

#     params = {
#         "term": torch.randn(B, N, T).log_softmax(-1).requires_grad_(True),
#         "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
#         "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
#         "left": left.requires_grad_(True),
#         "right": right.requires_grad_(True),
#         "slr": slr.requires_grad_(True),
#     }

#     pcfg = D1PCFGFlex(TGT_NT, TGT_PT)
#     nll = pcfg(params, lens)

#     pcfg_ref = PCFG()
#     nll_ref = pcfg_ref(D1PCFGFlex.get_pcfg_rules(params, TGT_NT), lens)

#     assert torch.allclose(nll, nll_ref)

#     m1 = pcfg(params, lens, marginal=True)
#     assert torch.allclose(
#         m1.sum((1, 2, 3, 4)),
#         torch.tensor([item * 2 - 1 for item in lens], dtype=torch.float),
#     )

#     m2 = pcfg_ref(D1PCFGFlex.get_pcfg_rules(params, TGT_NT), lens, marginal=True)[-1]
#     assert torch.allclose(
#         m1.diagonal(2, dim1=1, dim2=2).sum((1, 2)), m2[:, 0, :-1].sum(-1)
#     )


# @given(st.data())
# @settings(max_examples=100, deadline=None)
# def test_d1_pcfg(data):
#     B = data.draw(st.integers(min_value=1, max_value=3), label="b")
#     N = data.draw(st.integers(min_value=2, max_value=4), label="n")
#     TGT_PT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_pt")
#     SRC_PT = data.draw(st.integers(min_value=1, max_value=4), label="src_pt")
#     TGT_NT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_nt")
#     SRC_NT = SRC_PT
#     r = data.draw(st.integers(min_value=1, max_value=4), label="r")

#     NT = TGT_NT * SRC_NT
#     T = TGT_PT * SRC_PT
#     lens = [max(2, N - i) for i in range(B)]

#     slr = (
#         torch.randn(B, r, SRC_NT, SRC_NT, SRC_NT)
#         .view(B, r * SRC_NT, -1)
#         .softmax(-1)
#         .view(B, r, SRC_NT, SRC_NT, SRC_NT)
#     )

#     params = {
#         "term": torch.randn(B, N, T).log_softmax(-1).requires_grad_(True),
#         "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
#         "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
#         "left": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
#         "right": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
#         "slr": slr,
#     }

#     pcfg = D1PCFG(TGT_NT, TGT_PT)
#     nll = pcfg(params, lens)

#     pcfg_ref = PCFG()
#     nll_ref = pcfg_ref(D1PCFG.get_pcfg_rules(params, TGT_NT), lens)

#     assert torch.allclose(nll, nll_ref)

#     m1 = pcfg(params, lens, marginal=True)
#     assert torch.allclose(
#         m1.sum((1, 2, 3, 4)),
#         torch.tensor([item * 2 - 1 for item in lens], dtype=torch.float),
#     )

#     m2 = pcfg_ref(D1PCFG.get_pcfg_rules(params, TGT_NT), lens, marginal=True)[-1]
#     assert torch.allclose(
#         m1.diagonal(2, dim1=1, dim2=2).sum((1, 2)), m2[:, 0, :-1].sum(-1)
#     )


# @given(st.data())
# @settings(max_examples=5, deadline=None)
# def test_rdp(data):
#     B = data.draw(st.integers(min_value=1, max_value=3), label="b")
#     N = data.draw(st.integers(min_value=2, max_value=4), label="n")
#     T = data.draw(st.integers(min_value=1, max_value=4), label="t")
#     NT = data.draw(st.integers(min_value=1, max_value=4), label="nt")
#     TOPK = data.draw(st.integers(min_value=1, max_value=4), label="topk")
#     SAMPLE = data.draw(st.integers(min_value=1, max_value=4), label="sample")
#     SMOOTH = data.draw(st.floats(min_value=1e-4, max_value=1), label="smooth")
#     lens = [max(2, N - i) for i in range(B)]

#     params = {
#         "term": torch.randn(B, N, T).log_softmax(-1).requires_grad_(True),
#         "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
#         "rule": torch.randn(B, NT, (NT + T) ** 2).log_softmax(-1).view(B, NT, NT + T, NT + T).requires_grad_(True),
#     }
#     pcfg = PCFGRandomizedDP(TOPK, SAMPLE, SMOOTH)

#     with torch.no_grad():
#         nll = torch.zeros(B)
#         for _ in range(100):
#             nll += pcfg(params, lens)
#         nll /= 100

#         pcfg_ref = PCFG()
#         nll_ref = pcfg_ref(params, lens)

#     assert ((nll - nll_ref).abs() < 0.05).all()


def compare_marginal(m1, m2):
    # m1: (l, r, t)
    # m2: (w, l, t)
    m1 = m1.contiguous().flatten(3)
    for i in range(m1.shape[1] - 2):
        if not torch.allclose(
            m1.diagonal(2 + i, dim1=1, dim2=2).transpose(1, 2), m2[:, i, : -i - 1], rtol=1e-4, atol=1e-6
        ):
            return False
    return True


def check_full_marginal(m, l):
    return torch.allclose(
        m.flatten(1).sum(1),
        torch.tensor(l, dtype=torch.float) * 2 - 1,
    )


@given(st.data())
@settings(max_examples=100, deadline=None)
def test_nodecomp(data):
    B = data.draw(st.integers(min_value=1, max_value=3), label="b")
    N = data.draw(st.integers(min_value=2, max_value=4), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_pt")
    SRC_PT = data.draw(st.integers(min_value=1, max_value=4), label="src_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_nt")
    SRC_NT = data.draw(st.integers(min_value=1, max_value=4), label="src_nt")
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT

    lens = [max(2, N - i) for i in range(B)]
    params = {
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
    pcfg = NoDecomp(params, lens, **meta)
    nll = pcfg.nll

    pcfg_ref = PCFG()
    nll_ref = pcfg_ref(params, lens)
    assert torch.allclose(nll, nll_ref)

    m1 = pcfg.marginal
    assert check_full_marginal(m1, lens)

    m2 = pcfg_ref(params, lens, marginal=True)[-1]
    assert compare_marginal(m1[..., :NT], m2)


@given(st.data())
@settings(max_examples=100, deadline=None)
def test_decomp1(data):
    B = data.draw(st.integers(min_value=1, max_value=3), label="b")
    N = data.draw(st.integers(min_value=2, max_value=4), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_pt")
    SRC_PT = data.draw(st.integers(min_value=1, max_value=4), label="src_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_nt")
    SRC_NT = data.draw(st.integers(min_value=1, max_value=4), label="src_nt")
    r = data.draw(st.integers(min_value=1, max_value=4), label="r")
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT

    lens = [max(2, N - i) for i in range(B)]
    params = {
        "term": torch.randn(B, N, PT).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
        "left": torch.randn(B, r, NT + PT).softmax(-1).requires_grad_(True),
        "right": torch.randn(B, r, NT + PT).softmax(-1).requires_grad_(True),
    }
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }

    pcfg = Decomp1(params, lens, **meta)
    nll = pcfg.nll

    pcfg_ref = PCFG()
    params_ref = convert_decomp1_to_pcfg(params, NT)
    nll_ref = pcfg_ref(params_ref, lens)
    assert torch.allclose(nll, nll_ref)

    m1 = pcfg.marginal
    assert check_full_marginal(m1, lens)

    m2 = pcfg_ref(params_ref, lens, marginal=True)[-1]
    assert compare_marginal(m1[..., :NT], m2)


@given(st.data())
@settings(max_examples=100, deadline=None)
def test_decomp2(data):
    B = data.draw(st.integers(min_value=1, max_value=3), label="b")
    N = data.draw(st.integers(min_value=2, max_value=4), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_pt")
    SRC_PT = data.draw(st.integers(min_value=1, max_value=4), label="src_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_nt")
    SRC_NT = data.draw(st.integers(min_value=1, max_value=4), label="src_nt")
    r = data.draw(st.integers(min_value=1, max_value=4), label="r")
    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT

    lens = [max(2, N - i) for i in range(B)]
    params = {
        "term": torch.randn(B, N, PT).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
        "sl": torch.randn(B, r, SRC_NT + SRC_PT).softmax(-1).requires_grad_(True),
        "sr": torch.randn(B, r, SRC_NT + SRC_PT).softmax(-1).requires_grad_(True),
        "left": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
        "right": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
    }
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }

    pcfg = Decomp2(params, lens, **meta)
    nll = pcfg.nll

    pcfg_ref = PCFG()
    params_ref = convert_decomp2_to_pcfg(params, TGT_NT)
    nll_ref = pcfg_ref(params_ref, lens)
    assert torch.allclose(nll, nll_ref)

    m1 = pcfg.marginal
    assert check_full_marginal(m1, lens)

    m2 = pcfg_ref(params_ref, lens, marginal=True)[-1]
    assert compare_marginal(m1[..., :NT], m2)


@given(st.data())
@settings(max_examples=100, deadline=None)
def test_decomp3_dir0(data):
    B = data.draw(st.integers(min_value=1, max_value=3), label="b")
    N = data.draw(st.integers(min_value=2, max_value=4), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_pt")
    SRC_PT = data.draw(st.integers(min_value=1, max_value=4), label="src_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_nt")
    SRC_NT = data.draw(st.integers(min_value=1, max_value=4), label="src_nt")
    r = data.draw(st.integers(min_value=1, max_value=4), label="r")

    NT = TGT_NT * SRC_NT
    PT = TGT_PT * SRC_PT

    lens = [max(2, N - i) for i in range(B)]

    slr = torch.rand(B, r, SRC_NT, SRC_NT + SRC_PT, SRC_NT + SRC_PT)
    slr[..., :SRC_NT, :SRC_NT] /= slr[..., :SRC_NT, :SRC_NT].sum((3, 4), keepdim=True)
    slr[..., SRC_NT:, :SRC_NT] /= slr[..., SRC_NT:, :SRC_NT].sum((3, 4), keepdim=True)
    slr[..., :SRC_NT, SRC_NT:] /= slr[..., :SRC_NT, SRC_NT:].sum((3, 4), keepdim=True)
    slr[..., SRC_NT:, SRC_NT:] /= slr[..., SRC_NT:, SRC_NT:].sum((3, 4), keepdim=True)

    params = {
        "term": torch.randn(B, N, PT).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
        "left": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
        "right": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
        "slr": slr,
    }
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
        "direction": 0,
    }

    pcfg = Decomp3(params, lens, **meta)
    nll = pcfg.nll

    pcfg_ref = PCFG()
    params_ref = convert_decomp3_to_pcfg(params, TGT_NT)
    nll_ref = pcfg_ref(convert_decomp3_to_pcfg(params, TGT_NT), lens)
    assert torch.allclose(nll, nll_ref)

    m1 = pcfg.marginal
    assert check_full_marginal(m1, lens)

    m2 = pcfg_ref(params_ref, lens, marginal=True)[-1]
    assert compare_marginal(m1[..., :TGT_NT, :SRC_NT], m2)


@given(st.data())
@settings(max_examples=100, deadline=None)
def test_decomp3_dir1(data):
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
    slr /= slr.sum((3, 4), keepdim=True)

    left = torch.rand(B, r, TGT_NT + TGT_PT)
    left[..., :TGT_NT] /= left[..., :TGT_NT].sum(-1, keepdim=True)
    left[..., TGT_NT:] /= left[..., TGT_NT:].sum(-1, keepdim=True)

    right = torch.rand(B, r, TGT_NT + TGT_PT)
    right[..., :TGT_NT] /= right[..., :TGT_NT].sum(-1, keepdim=True)
    right[..., TGT_NT:] /= right[..., TGT_NT:].sum(-1, keepdim=True)

    params = {
        "term": torch.randn(B, N, T).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
        "left": left.requires_grad_(True),
        "right": right.requires_grad_(True),
        "slr": slr.requires_grad_(True),
    }
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
        "direction": 1,
    }

    pcfg = Decomp3(params, lens, **meta)
    nll = pcfg.nll

    pcfg_ref = PCFG()
    params_ref = convert_decomp3_to_pcfg(params, TGT_NT)
    nll_ref = pcfg_ref(params_ref, lens)
    assert torch.allclose(nll, nll_ref)

    m1 = pcfg.marginal
    assert check_full_marginal(m1, lens)

    m2 = pcfg_ref(params_ref, lens, marginal=True)[-1]
    assert compare_marginal(m1[..., :TGT_NT, :SRC_NT], m2)


@given(st.data())
@settings(max_examples=100, deadline=None)
def test_decomp4_dir0(data):
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

    sl = torch.rand(B, r, SRC_NT, SRC_NT + SRC_PT)
    sl[..., :SRC_NT] /= sl[..., :SRC_NT].sum(3, keepdim=True)
    sl[..., SRC_NT:] /= sl[..., SRC_NT:].sum(3, keepdim=True)

    params = {
        "term": torch.randn(B, N, T).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
        "left": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
        "right": torch.randn(B, r, TGT_NT + TGT_PT).softmax(-1).requires_grad_(True),
        "sl": sl,
        "sr": sl,
    }
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
        "direction": 0,
    }

    pcfg = Decomp4(params, lens, **meta)
    nll = pcfg.nll

    pcfg_ref = PCFG()
    params_ref = convert_decomp4_to_pcfg(params, TGT_NT)
    nll_ref = pcfg_ref(params_ref, lens)
    assert torch.allclose(nll, nll_ref)

    m1 = pcfg.marginal
    assert check_full_marginal(m1, lens)

    m2 = pcfg_ref(params_ref, lens, marginal=True)[-1]
    assert compare_marginal(m1[..., :TGT_NT, :SRC_NT], m2)


@given(st.data())
@settings(max_examples=100, deadline=None)
def test_decomp4_dir1(data):
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

    sl = torch.rand(B, r, SRC_NT, SRC_NT + SRC_PT)
    sl /= sl.sum(3, keepdim=True)

    left = torch.rand(B, r, TGT_NT + TGT_PT)
    left[..., :TGT_NT] /= left[..., :TGT_NT].sum(-1, keepdim=True)
    left[..., TGT_NT:] /= left[..., TGT_NT:].sum(-1, keepdim=True)

    right = torch.rand(B, r, TGT_NT + TGT_PT)
    right[..., :TGT_NT] /= right[..., :TGT_NT].sum(-1, keepdim=True)
    right[..., TGT_NT:] /= right[..., TGT_NT:].sum(-1, keepdim=True)

    params = {
        "term": torch.randn(B, N, T).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
        "left": left.requires_grad_(True),
        "right": right.requires_grad_(True),
        "sl": sl.requires_grad_(True),
        "sr": sl.requires_grad_(True),
    }
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
        "direction": 1,
    }

    pcfg = Decomp4(params, lens, **meta)
    nll = pcfg.nll

    pcfg_ref = PCFG()
    params_ref = convert_decomp4_to_pcfg(params, TGT_NT)
    nll_ref = pcfg_ref(params_ref, lens)
    assert torch.allclose(nll, nll_ref)

    m1 = pcfg.marginal
    assert check_full_marginal(m1, lens)

    m2 = pcfg_ref(params_ref, lens, marginal=True)[-1]
    assert compare_marginal(m1[..., :TGT_NT, :SRC_NT], m2)


@given(st.data())
@settings(max_examples=100, deadline=None)
def test_decomp5_dir0(data):
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

    slr = torch.rand(B, TGT_NT, SRC_NT, SRC_NT + SRC_PT, SRC_NT + SRC_PT)
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
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
        "direction": 0,
    }

    pcfg = Decomp5(params, lens, **meta)
    nll = pcfg.nll

    pcfg_ref = PCFG()
    params_ref = convert_decomp5_to_pcfg(params, TGT_NT)
    nll_ref = pcfg_ref(params_ref, lens)
    assert torch.allclose(nll, nll_ref)

    m1 = pcfg.marginal
    assert check_full_marginal(m1, lens)

    m2 = pcfg_ref(params_ref, lens, marginal=True)[-1]
    assert compare_marginal(m1[..., :TGT_NT, :SRC_NT], m2)


@given(st.data())
@settings(max_examples=100, deadline=None)
def test_decomp5_dir1(data):
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

    slr = torch.rand(B, TGT_NT, SRC_NT, SRC_NT + SRC_PT, SRC_NT + SRC_PT)
    slr /= slr.sum((3, 4), keepdim=True)

    left = torch.rand(B, r, TGT_NT + TGT_PT)
    left[..., :TGT_NT] /= left[..., :TGT_NT].sum(-1, keepdim=True)
    left[..., TGT_NT:] /= left[..., TGT_NT:].sum(-1, keepdim=True)

    right = torch.rand(B, r, TGT_NT + TGT_PT)
    right[..., :TGT_NT] /= right[..., :TGT_NT].sum(-1, keepdim=True)
    right[..., TGT_NT:] /= right[..., TGT_NT:].sum(-1, keepdim=True)

    params = {
        "term": torch.randn(B, N, T).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
        "left": left.requires_grad_(True),
        "right": right.requires_grad_(True),
        "slr": slr.requires_grad_(True),
    }
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
        "direction": 1,
    }

    pcfg = Decomp5(params, lens, **meta)
    nll = pcfg.nll

    pcfg_ref = PCFG()
    params_ref = convert_decomp5_to_pcfg(params, TGT_NT)
    nll_ref = pcfg_ref(params_ref, lens)
    assert torch.allclose(nll, nll_ref)

    m1 = pcfg.marginal
    assert check_full_marginal(m1, lens)

    m2 = pcfg_ref(params_ref, lens, marginal=True)[-1]
    assert compare_marginal(m1[..., :TGT_NT, :SRC_NT], m2)


@given(st.data())
@settings(max_examples=100, deadline=None)
def test_decomp7(data):
    B = data.draw(st.integers(min_value=1, max_value=3), label="b")
    N = data.draw(st.integers(min_value=2, max_value=4), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_pt")
    SRC_PT = data.draw(st.integers(min_value=1, max_value=4), label="src_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=4), label="tgt_nt")
    SRC_NT = data.draw(st.integers(min_value=1, max_value=4), label="src_nt")
    r = data.draw(st.integers(min_value=1, max_value=4), label="r")

    SRC = SRC_NT + SRC_PT
    NT = TGT_NT * SRC_NT
    T = TGT_PT * SRC_PT
    lens = [max(2, N - i) for i in range(B)]

    slr = (
        torch.randn(B, r, SRC_NT, SRC_NT + SRC_PT, SRC_NT + SRC_PT)
        .flatten(3)
        .softmax(-1)
        .view(B, r, SRC_NT, SRC_NT + SRC_PT, SRC_NT + SRC_PT)
    )

    params = {
        "term": torch.randn(B, N, T).log_softmax(-1).requires_grad_(True),
        "root": torch.randn(B, NT).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r).softmax(-1).requires_grad_(True),
        "left": torch.randn(B, r, SRC, max(TGT_NT, TGT_PT)).softmax(-1).requires_grad_(True),
        "right": torch.randn(B, r, SRC, max(TGT_NT, TGT_PT)).softmax(-1).requires_grad_(True),
        "slr": slr,
    }
    meta = {
        "batch_size": B,
        "nt_states": TGT_NT,
        "nt_num_nodes": SRC_NT,
        "pt_states": TGT_PT,
        "pt_num_nodes": SRC_PT,
        "batch_size": B,
    }

    pcfg = Decomp7(params, lens, **meta)
    nll = pcfg.nll

    pcfg_ref = PCFG()
    params_ref = convert_decomp7_to_pcfg(params, TGT_NT)
    nll_ref = pcfg_ref(params_ref, lens)
    assert torch.allclose(nll, nll_ref)

    m1 = pcfg.marginal
    assert check_full_marginal(m1, lens)

    m2 = pcfg_ref(params_ref, lens, marginal=True)[-1]
    assert compare_marginal(m1[..., :TGT_NT, :SRC_NT], m2)
