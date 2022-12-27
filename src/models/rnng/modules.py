from cgitb import reset
from turtle import forward

import torch


class HackedLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.weight = None

    def setup_weight(self, weight):
        self.weight = weight

    # def forward(self, input):
    #     breakpoint()
    #     return torch.einsum("b...x,bxy->b...y", input, self.weight)

    def forward(self, input):
        input = input.view(len(self.weight), -1, input.shape[-1])
        result = torch.einsum("bax,byx->bay", input, self.weight)
        return result.view(-1, result.shape[-1])


class HackedEmbedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = None

    def setup_weight(self, weight):
        self.weight = weight

    def forward(self, input, input2=None):
        if input2 is not None:
            return self.weight[input, input2]
        else:
            shape = list(input.shape)
            input = input.flatten(1)
            result = self.weight.gather(1, input.unsqueeze(-1).expand(-1, -1, self.weight.shape[-1]))
            return result.view(shape + [-1])

    # def forward(self, input):
    #     result = self.weight.gather(1, input[:, None, None].expand(-1, -1, self.weight.shape[-1]))
    #     return result.squeeze(1)
