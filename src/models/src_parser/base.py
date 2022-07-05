import torch
import torch.nn as nn


class SrcParserBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, lengths):
        raise NotImplementedError

    def sample(self, x, lengths, **kwargs):
        raise NotImplementedError

    def argmax(self, x, lengths, **kwargs):
        raise NotImplementedError