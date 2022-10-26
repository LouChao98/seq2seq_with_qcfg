import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# from src.models.tgt_parser.struct.decomp1 import Decomp1, convert_decomp1_to_pcfg
# from src.models.tgt_parser.struct.decomp1_copy_as_t import Decomp1CopyPT, convert_decomp1copypt_to_pcfg
# from src.models.tgt_parser.struct.decomp2 import Decomp2, convert_decomp2_to_pcfg
# from src.models.tgt_parser.struct.decomp3 import Decomp3, convert_decomp3_to_pcfg
# from src.models.tgt_parser.struct.decomp4 import Decomp4, convert_decomp4_to_pcfg
# from src.models.tgt_parser.struct.decomp5 import Decomp5, convert_decomp5_to_pcfg
# from src.models.tgt_parser.struct.decomp7 import Decomp7, convert_decomp7_to_pcfg
# from src.models.tgt_parser.struct.decomp7_impl2 import Decomp7Impl2, convert_decomp7impl2_to_pcfg
# from src.models.tgt_parser.struct.decomp7_impl3 import Decomp7Impl3, convert_decomp7impl3_to_pcfg
from src.models.tgt_parser.struct3.no_decomp import NoDecomp
from src.models.tgt_parser.struct.pcfg import PCFG


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
@settings(max_examples=100, deadline=None, use_coverage=False)
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
