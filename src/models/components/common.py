import torch
from torch import nn
import torch.nn.functional as F


class ResidualLayer(nn.Module):
    def __init__(self, dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.out_dim = dim

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x


class MultiResidualLayer(nn.Module):
    def __init__(self, in_dim=100, res_dim=100, out_dim=None, num_layers=3):
        super(MultiResidualLayer, self).__init__()
        self.num_layers = num_layers
        if in_dim is not None:
            self.in_linear = nn.Linear(in_dim, res_dim)
        else:
            self.in_linear = None
        if out_dim is not None:
            self.out_linear = nn.Linear(res_dim, out_dim)
        else:
            self.out_linear = None
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(res_dim) for _ in range(num_layers)]
        )
        self.out_dim = res_dim if out_dim is None else out_dim

    def forward(self, x):
        if self.in_linear is not None:
            out = self.in_linear(x)
        else:
            out = x
        for i in range(self.num_layers):
            out = self.res_blocks[i](out)
        if self.out_linear is not None:
            out = self.out_linear(out)
        return out
