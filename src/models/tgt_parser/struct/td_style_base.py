import torch
from torch.autograd import grad

from ._fn import diagonal, diagonal_copy_, stripe


class TDStyleBase:
    def get_prediction(self, logZ, span_indicator, lens):
        batch, seq_len = span_indicator.shape[:2]
        # to avoid some trivial corner cases.
        if seq_len >= 3:
            marginals = grad(logZ.sum(), [span_indicator])[0].detach()
            # return self._cky_zero_order(marginals.detach(), lens)
            scores = marginals.sum((3, 4))
            spans = self._cky_zero_order(scores.detach(), lens)
        else:
            # minimal length is 2
            spans = [[(0, 0), (1, 1), (0, 1)] for _ in range(batch)]
        spans_ = []
        labels = marginals.flatten(3).argmax(-1).cpu().numpy()
        for i, spans_inst in enumerate(spans):
            labels_spans_inst = []
            for span in spans_inst:
                labels_spans_inst.append(
                    (span[0], span[1] - 1, labels[i, span[0], span[1]])
                )
            spans_.append(labels_spans_inst)
        return spans_

    @torch.no_grad()
    def _cky_zero_order(self, marginals, lens):
        N = marginals.shape[-1]
        s = marginals.new_zeros(*marginals.shape).fill_(-1e9)
        p = marginals.new_zeros(*marginals.shape).long()
        diagonal_copy_(s, diagonal(marginals, 1), 1)
        for w in range(2, N):
            n = N - w
            starts = p.new_tensor(range(n))
            if w != 2:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            else:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            X, split = (Y + Z).max(2)
            x = X + diagonal(marginals, w)
            diagonal_copy_(s, x, w)
            diagonal_copy_(p, split + starts.unsqueeze(0) + 1, w)

        def backtrack(p, i, j):
            if j == i + 1:
                return [(i, j)]
            split = p[i][j]
            ltree = backtrack(p, i, split)
            rtree = backtrack(p, split, j)
            return [(i, j)] + ltree + rtree

        p = p.tolist()
        lens = lens.tolist()
        spans = [backtrack(p[i], 0, length) for i, length in enumerate(lens)]
        for spans_inst in spans:
            spans_inst.sort(key=lambda x: x[1] - x[0])
        return spans

    def convert_to_tree(self, spans, length):
        tree = [(i, str(i)) for i in range(length)]
        tree = dict(tree)
        for l, r, _ in spans:
            if l != r:
                span = "({} {})".format(tree[l], tree[r])
                tree[r] = tree[l] = span
        return tree[0]
