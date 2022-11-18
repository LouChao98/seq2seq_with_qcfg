from functools import reduce

import torch
from nltk.tree import Tree
from numba import njit

from .base import SrcParserBase


class GoldTreeProcessor(SrcParserBase):
    # read src_tree and convert them to spans
    def __init__(self, binarize=False, *args, **kwargs):
        super(GoldTreeProcessor, self).__init__()
        self.binarize = binarize

    def forward(self, x, lengths):
        return self

    @property
    def partition(self):
        return torch.tensor(0.0)

    @property
    def nll(self):
        return torch.tensor(0.0)

    def get_spans(self, trees):
        spans = []
        for tree in trees:
            if self.binarize:
                tree.chomsky_normal_form()
            tree_str = tree._pformat_flat("", "()", False)
            spans.append(tree2span(tree_str))
        return spans

    def sample(self, x, lengths, **kwargs):
        raise NotImplementedError

    def argmax(self, x, lengths, **kwargs):
        raise NotImplementedError


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
            if pos > start + 1:
                spans.append((start, pos, 0))
    return spans
