import torch.nn as nn


class CombinarySpanEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout) -> None:
        super().__init__()
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, in_dim)
        self.bilinear = nn.Bilinear(in_dim, in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x1, x2):
        x1 = self.act(self.linear1(self.dropout(x1)))
        x2 = self.act(self.linear2(self.dropout(x2)))
        out = self.bilinear(x1, x2)
        return out

    def get_output_dim(self):
        return self.out_dim
