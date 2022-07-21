import torch
import torch.nn as nn


class TreeEncoderBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, lengths, spans=None):
        raise NotImplementedError

    def get_output_dim(self):
        return 0