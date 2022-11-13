import logging
from typing import Dict

import torch
from torch import Tensor

from .decomp1 import Decomp1, Decomp1Sampler, convert_decomp1_to_pcfg

log = logging.getLogger(__file__)


class Decomp2(Decomp1):
    def __init__(self, params: Dict[str, Tensor], lens, **kwargs):
        params = convert_decomp2_to_decomp1(params, kwargs["nt_states"])
        super().__init__(params, lens, **kwargs)

    @staticmethod
    def random(bsz, max_len, tgt_pt, src_pt, tgt_nt, src_nt, r):
        nt = tgt_nt * src_nt
        pt = tgt_pt * src_pt
        return {
            "term": torch.randn(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "head": torch.randn(bsz, nt, r).softmax(-1).requires_grad_(True),
            "sl": torch.randn(bsz, r, src_nt + src_pt).softmax(-1).requires_grad_(True),
            "sr": torch.randn(bsz, r, src_nt + src_pt).softmax(-1).requires_grad_(True),
            "left": torch.randn(bsz, r, tgt_nt + tgt_pt).softmax(-1).requires_grad_(True),
            "right": torch.randn(bsz, r, tgt_nt + tgt_pt).softmax(-1).requires_grad_(True),
        }


class Decomp2Sampler(Decomp1Sampler):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        params = convert_decomp2_to_decomp1(params, self.nt_states)
        return super().process_params(params)


def compose(t, s):
    return (t.unsqueeze(-1) * s.unsqueeze(-2)).flatten(2)


def convert_decomp2_to_decomp1(p, tgt_nt):
    L, SL = p["left"], p["sl"]
    R, SR = p["right"], p["sr"]
    src_nt = p["root"].shape[1] // tgt_nt
    LNT, LPT = L[..., :tgt_nt], L[..., tgt_nt:]
    RNT, RPT = R[..., :tgt_nt], R[..., tgt_nt:]
    SLNT, SLPT = SL[..., :src_nt], SL[..., src_nt:]
    SRNT, SRPT = SR[..., :src_nt], SR[..., src_nt:]
    LNT = compose(LNT, SLNT)
    LPT = compose(LPT, SLPT)
    RNT = compose(RNT, SRNT)
    RPT = compose(RPT, SRPT)

    output = {
        "term": p["term"],
        "head": p["head"],
        "root": p["root"],
        "left": torch.cat([LNT, LPT], dim=-1),
        "right": torch.cat([RNT, RPT], dim=-1),
    }
    if "constraint" in p:
        output["constraint"] = p["constraint"]
    if "add" in p:
        output["add"] = p["add"]
    if "lse" in p:
        output["lse"] = p["lse"]
    return output


def convert_decomp2_to_pcfg(p, tgt_nt):
    return convert_decomp1_to_pcfg(convert_decomp2_to_decomp1(p, tgt_nt), tgt_nt)
