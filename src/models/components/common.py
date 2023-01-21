import torch
import torch.nn.functional as F
from torch import nn


class ResidualLayer(nn.Module):
    def __init__(self, dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.out_dim = dim

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x
        return F.leaky_relu(self.lin2(F.leaky_relu(self.lin1(x)))) + x


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
        self.res_blocks = nn.ModuleList([ResidualLayer(res_dim) for _ in range(num_layers)])
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


class SharedDropout(nn.Module):
    r"""
    :class:`SharedDropout` differs from the vanilla dropout strategy in that the dropout mask is shared across one dimension.
    Args:
        p (float):
            The probability of an element to be zeroed. Default: 0.5.
        batch_first (bool):
            If ``True``, the input and output tensors are provided as ``[batch_size, seq_len, *]``.
            Default: ``True``.
    Examples:
        >>> batch_size, seq_len, hidden_size = 1, 3, 5
        >>> x = torch.ones(batch_size, seq_len, hidden_size)
        >>> nn.Dropout()(x)
        tensor([[[0., 2., 2., 0., 0.],
                 [2., 2., 0., 2., 2.],
                 [2., 2., 2., 2., 0.]]])
        >>> SharedDropout()(x)
        tensor([[[2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.]]])
    """

    def __init__(self, p: float = 0.5, batch_first: bool = True):
        super().__init__()

        self.p = p
        self.batch_first = batch_first

    def __repr__(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (~torch.Tensor):
                A tensor of any shape.
        Returns:
            A tensor with the same shape as `x`.
        """

        if not self.training:
            return x
        return x * self.get_mask(x[:, 0], self.p).unsqueeze(1) if self.batch_first else self.get_mask(x[0], self.p)

    @staticmethod
    def get_mask(x: torch.Tensor, p: float) -> torch.FloatTensor:
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2norm(t)
    cosine_sim = torch.einsum("i d, j d -> i j", normed_codes, normed_codes)
    return (cosine_sim**2).sum() / (n**2) - (1 / n)
