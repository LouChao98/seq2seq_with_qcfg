from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.common import MultiResidualLayer
from .base import SrcParserBase


class NaiveSrcParser(SrcParserBase):
    # generate a two-level tree
    #    ROOT
    #  /  | .. \
    # w1  w2 ..  wn

    def __init__(self, *args, **kwargs):
        super(NaiveSrcParser, self).__init__()

    def forward(self, x, lengths):
        return self

    @property
    def partition(self):
        # simulate dist
        return torch.tensor(0.0, requires_grad=True)

    def sample(self, x, lengths, **kwargs):

        b, n = x.shape
        samples = torch.zeros(b, n, n, 1, dtype=torch.long, device=x.device)
        samples[torch.arange(b), torch.tensor(lengths) - 2, 0] = 1.0
        return [None, None, None, samples], 0

    def argmax(self, x, lengths, **kwargs):
        return self.sample(x, lengths, **kwargs)
