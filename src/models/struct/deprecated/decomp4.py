import logging
from typing import Dict

import torch
from torch import Tensor

from .decomp3 import Decomp3, Decomp3Dir0Sampler, Decomp3Dir1Sampler, convert_decomp3_to_pcfg

log = logging.getLogger(__file__)


class Decomp4(Decomp3):
    # A[i] -> B[j], C[k]
    # ================
    # A[i] -> R
    # R -> B
    # R -> C
    # R, i -> j
    # R, i -> k
    # ================
    # Time complexity: 6
    # Flex

    def __init__(self, params, lens, **kwargs):
        params = convert_decomp4_to_decomp3(params)
        super().__init__(params, lens, **kwargs)

    @staticmethod
    def random_dir0(bsz, max_len, tgt_pt, src_pt, tgt_nt, src_nt, r):
        nt = tgt_nt * src_nt
        pt = tgt_pt * src_pt
        sl = torch.rand(bsz, r, src_nt, src_nt + src_pt)
        sl[..., :src_nt] /= sl[..., :src_nt].sum(3, keepdim=True)
        sl[..., src_nt:] /= sl[..., src_nt:].sum(3, keepdim=True)
        sr = torch.rand(bsz, r, src_nt, src_nt + src_pt)
        sr[..., :src_nt] /= sr[..., :src_nt].sum(3, keepdim=True)
        sr[..., src_nt:] /= sr[..., src_nt:].sum(3, keepdim=True)
        return {
            "term": torch.randn(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "head": torch.randn(bsz, nt, r).softmax(-1).requires_grad_(True),
            "left": torch.randn(bsz, r, tgt_nt + tgt_pt).softmax(-1).requires_grad_(True),
            "right": torch.randn(bsz, r, tgt_nt + tgt_pt).softmax(-1).requires_grad_(True),
            "sl": sl,
            "sr": sr,
        }

    @staticmethod
    def random_dir1(bsz, max_len, tgt_pt, src_pt, tgt_nt, src_nt, r):
        nt = tgt_nt * src_nt
        pt = tgt_pt * src_pt
        sl = torch.rand(bsz, r, src_nt, src_nt + src_pt)
        sl /= sl.sum(3, keepdim=True)

        left = torch.rand(bsz, r, tgt_nt + tgt_pt)
        left[..., :tgt_nt] /= left[..., :tgt_nt].sum(-1, keepdim=True)
        left[..., tgt_nt:] /= left[..., tgt_nt:].sum(-1, keepdim=True)

        right = torch.rand(bsz, r, tgt_nt + tgt_pt)
        right[..., :tgt_nt] /= right[..., :tgt_nt].sum(-1, keepdim=True)
        right[..., tgt_nt:] /= right[..., tgt_nt:].sum(-1, keepdim=True)
        return {
            "term": torch.randn(bsz, max_len, pt).log_softmax(-1).requires_grad_(True),
            "root": torch.randn(bsz, nt).log_softmax(-1).requires_grad_(True),
            "head": torch.randn(bsz, nt, r).softmax(-1).requires_grad_(True),
            "left": left.requires_grad_(True),
            "right": right.requires_grad_(True),
            "sl": sl.requires_grad_(True),
            "sr": sl.requires_grad_(True),
        }


class Decomp4Sampler:
    def __new__(cls, *args, direction, **kwargs):
        if direction == 0:
            return Decomp4Dir0Sampler(*args, **kwargs)
        elif direction == 1:
            return Decomp4Dir1Sampler(*args, **kwargs)
        raise ValueError


class Decomp4Dir0Sampler(Decomp3Dir0Sampler):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        params = convert_decomp4_to_decomp3(params)
        return super().process_params(params)


class Decomp4Dir1Sampler(Decomp3Dir1Sampler):
    @torch.no_grad()
    def process_params(self, params: Dict[str, Tensor]):
        params = convert_decomp4_to_decomp3(params)
        return super().process_params(params)


def convert_decomp4_to_decomp3(p):
    output = {**p}
    output["slr"] = output.pop("sl").unsqueeze(-1) + output.pop("sr").unsqueeze(-2)
    if "constraint" in p:
        output["constraint"] = p["constraint"]
    if "add" in p:
        output["add"] = p["add"]
    if "lse" in p:
        output["lse"] = p["lse"]
    return output


def convert_decomp4_to_pcfg(p, tgt_nt):
    return convert_decomp3_to_pcfg(convert_decomp4_to_decomp3(p), tgt_nt)
