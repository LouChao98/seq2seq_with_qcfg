import torch
import torch.nn as nn


class ReconstructorBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, lengths, node_event):
        raise NotImplementedError
