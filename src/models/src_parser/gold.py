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
        # simulate dist
        return torch.tensor(0.0, requires_grad=True)

    def get_spans(self, batch):
        spans = []
        for tree in batch["src_tree"]:
            if self.binarize:
                tree.chomsky_normal_form()
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

    def entropy(self, x, lengths, **kwargs):
        return x.new_zeros(len(lengths))


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
                spans.append((start, pos - 1, 0))
    return spans
