import torch
from torch import nn
import torch.nn.functional as F


class BinaryTreeLSTMLayer(nn.Module):
    def __init__(self, dim=200):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim * 2, dim * 5)

    def forward(self, x1, x2, e=None):
        # x = (h, c). h, c = b x dim. hidden/cell states of children
        # e = b x e_dim. external information vector
        if not isinstance(x1, tuple):
            x1 = (x1, None)
        h1, c1 = x1
        if x2 is None:
            x2 = (torch.zeros_like(h1), torch.zeros_like(h1))
        elif not isinstance(x2, tuple):
            x2 = (x2, None)
        h2, c2 = x2
        if c1 is None:
            c1 = torch.zeros_like(h1)
        if c2 is None:
            c2 = torch.zeros_like(h2)
        concat = torch.cat([h1, h2], 1)
        all_sum = self.linear(concat)
        i, f1, f2, o, g = all_sum.split(self.dim, 1)
        c = (
            torch.sigmoid(f1) * c1
            + torch.sigmoid(f2) * c2
            + torch.sigmoid(i) * torch.tanh(g)
        )
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class BinaryTreeLSTM(nn.Module):
    def __init__(
        self,
        dim=16,
    ):
        super(BinaryTreeLSTM, self).__init__()
        self.dim = dim
        self.tree_rnn = BinaryTreeLSTMLayer(dim)
        self.SHIFT = 0
        self.REDUCE = 1

    def get_actions(self, spans, l):
        spans_set = set([(s[0], s[1]) for s in spans if s[0] < s[1]])
        actions = [self.SHIFT, self.SHIFT]
        stack = [(0, 0), (1, 1)]
        ptr = 2
        num_reduce = 0
        while ptr < l:
            if len(stack) >= 2:
                cand_span = (stack[-2][0], stack[-1][1])
            else:
                cand_span = (-1, -1)
            if cand_span in spans_set:
                actions.append(self.REDUCE)
                stack.pop()
                stack.pop()
                stack.append(cand_span)
                num_reduce += 1
            else:
                actions.append(self.SHIFT)
                stack.append((ptr, ptr))
                ptr += 1
        while len(actions) < 2 * l - 1:
            actions.append(self.REDUCE)
        return actions

    def forward(self, x, lengths, spans=None, token_type=None):
        x = x.unsqueeze(2)
        node_features = []
        all_spans = []
        for b in range(len(x)):
            len_b = lengths[b]
            spans_b = [(i, i, -1) for i in range(len_b)]
            node_features_b = [x[b][i] for i in range(len_b)]
            stack = []
            if len_b == 1:
                actions = []
            else:
                actions = self.get_actions(spans[b], len_b)
            ptr = 0
            for action in actions:
                if action == self.SHIFT:
                    # [(h, c), (left_boundry, right_boundry, ?)]
                    stack.append([(x[b][ptr], None), (ptr, ptr, -1)])
                    ptr += 1
                else:
                    right = stack.pop()
                    left = stack.pop()
                    new = self.tree_rnn(left[0], right[0])
                    new_span = (left[1][0], right[1][1], -1)
                    spans_b.append(new_span)
                    node_features_b.append(new[0])
                    stack.append([new, new_span])
            node_features.append(torch.cat(node_features_b, 0))
            all_spans.append(spans_b)
        return node_features, all_spans

