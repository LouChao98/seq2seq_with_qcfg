from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit

from ..components.common import MultiResidualLayer
from .base import SrcParserBase


class GoldTreeProcessor(SrcParserBase):
    # read src_tree and convert them to spans
    def __init__(self, *args, **kwargs):
        super(GoldTreeProcessor, self).__init__()

    def forward(self, x, lengths):
        return self

    @property
    def partition(self):
        # simulate dist
        return torch.tensor(0.0, requires_grad=True)

    def get_spans(self, trees):
        spans = []
        for tree in trees:
            tree_str = tree._pformat_flat("", "()", False)
            spans.append(tree2span(tree_str))
        return spans

    def sample(self, x, lengths, **kwargs):
        raise NotImplementedError("To event matric")
        trees = kwargs["src_tree"]
        samples = []
        for tree in trees:
            tree_str = tree._pformat_flat("", "()", False)
            samples.append(tree2span(tree_str))
        return [None, None, None, samples], 0

    def argmax(self, x, lengths, **kwargs):
        return self.sample(x, lengths, **kwargs)


@njit
def tree2span(tree_str: str):
    stack = []
    pos = 0
    spans = []
    for i, c in enumerate(tree_str):
        if c == "(":
            stack.append(pos)
        elif c == ")":
            if tree_str[i - 1] != ")":
                pos += 1
            start = stack.pop()
            spans.append((start, pos, 0))
    return spans
