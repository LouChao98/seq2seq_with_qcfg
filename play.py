import math
from collections import defaultdict
from itertools import product

import torch

from src.models.posterior_regularization.amr import AMRNeqQCFGPrTask
from src.models.tgt_parser.struct.pcfg import PCFG
from src.utils.fn import spans2tree


def enumerate_tree(i, j, nt, pt):
    if i + 1 == j:
        for t in range(pt):
            yield [(i, j, t)]

    for t in range(nt):
        spans = [(i, j, t)]
        for k in range(i + 1, j):
            for l, r in product(
                enumerate_tree(i, k, nt, pt), enumerate_tree(k, j, nt, pt)
            ):
                yield spans + l + r


def score(params, seq, span, nt):

    s, p = spans2tree(span)
    parent2children = defaultdict(list)
    for i, pj in enumerate(p):
        parent2children[pj].append(i)

    score = 0
    for pj, children in parent2children.items():
        if len(children) == 1:
            assert pj == -1
            score += params["root"][0, s[children[0]][2]]
        else:
            assert len(children) == 2
            l = s[children[0]]
            lt = l[2] + (0 if l[0] != l[1] else nt)
            r = s[children[1]]
            rt = r[2] + (0 if r[0] != r[1] else nt)
            pt = s[pj][2]
            score += params["rule"][0, pt, lt, rt]

    pts = [None for _ in seq]
    for sj in s:
        if sj[0] == sj[1]:
            pts[sj[0]] = sj[2]

    for pt, t in zip(pts, seq):
        score += params["term"][0, pt, t]

    return score


B = 1
N = 3
VOCAB = 3
TGT_PT = 2
SRC_PT = 2
TGT_NT = 2
SRC_NT = 2
r = 2
NT = TGT_NT * SRC_NT
PT = TGT_PT * SRC_PT


params = {
    "term": torch.randn(B, PT, VOCAB).log_softmax(-1),
    "root": torch.randn(B, NT).log_softmax(-1),
    "rule": torch.randn(B, NT, (NT + PT) ** 2)
    .log_softmax(-1)
    .view(B, NT, NT + PT, NT + PT),
}
lens = torch.tensor([N] * B)

seq = torch.randint(0, VOCAB, (B, N))
seq_inst = seq[0].tolist()
ll = []

for tree in enumerate_tree(0, N, NT, PT):
    tree = [(l, r - 1, t) for l, r, t in tree]
    stree = score(params, seq_inst, tree, NT)
    ll.append(stree)
print(torch.tensor(ll).logsumexp(0))


n = seq.size(1)
terms = params["term"].unsqueeze(1).expand(B, n, PT, params["term"].size(2))
x_expand = seq.unsqueeze(2).expand(B, n, PT).unsqueeze(3)
terms = torch.gather(terms, 3, x_expand).squeeze(3)
params["term"] = terms

pcfg = PCFG()
print((-pcfg(params, lens))[0].detach())


pr_task = AMRNeqQCFGPrTask()
