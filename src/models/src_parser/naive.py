from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.common import MultiResidualLayer
from .base import SrcParserBase


class NaiveTreeProcessor(SrcParserBase):
    # generate a two-level tree
    #    ROOT
    #  /  | .. \
    # w1  w2 ..  wn

    def __init__(self, mode="depth1", *args, **kwargs):
        super(NaiveTreeProcessor, self).__init__()
        self.mode = mode
        assert mode in ("depth1", "leftbranching", "rightbranching")

    def forward(self, x, lengths):
        return self

    @property
    def partition(self):
        # simulate dist
        return torch.tensor(0.0, requires_grad=True)

    def get_spans(self, batch):
        spans = []
        if self.mode == "depth1":
            for l in batch["src_lens"]:
                spans.append([(0, l - 1, 0)])
        elif self.mode == "leftbranching":
            for l in batch["src_lens"]:
                spans.append([(0, i, 0) for i in range(1, l)])
        elif self.mode == "rightbranching":
            for l in batch["src_lens"]:
                spans.append([(i, l - 1, 0) for i in range(l - 2, -1, -1)])
        else:
            raise ValueError
        return spans

    def sample(self, x, lengths, **kwargs):
        b, n = x.shape
        samples = torch.zeros(b, n, n, 1, dtype=torch.long, device=x.device)
        samples[torch.arange(b), torch.tensor(lengths) - 2, 0] = 1.0
        return [None, None, None, samples], 0

    def argmax(self, x, lengths, **kwargs):
        return self.sample(x, lengths, **kwargs)
