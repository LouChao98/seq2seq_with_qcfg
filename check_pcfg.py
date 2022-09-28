import math
from collections import defaultdict
from itertools import product
from pprint import pprint

import pytorch_lightning as pl
import torch
from torch_struct.distributions import SentCFG

from src.models.posterior_regularization.amr import AMRNeqQCFGPrTask
from src.models.posterior_regularization.pr import compute_pr
from src.models.tgt_parser.struct.pcfg import PCFG
from src.utils.fn import spans2tree

pl.seed_everything(1)


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

    _score = 0
    for pj, children in parent2children.items():
        if len(children) == 1:
            assert pj == -1
            _score += params["root"][0, s[children[0]][2]]
        else:
            assert len(children) == 2
            l = s[children[0]]
            lt = l[2] + (0 if l[0] != l[1] else nt)
            r = s[children[1]]
            rt = r[2] + (0 if r[0] != r[1] else nt)
            pt = s[pj][2]
            _score += params["rule"][0, pt, lt, rt]

    pts = [None for _ in seq]
    for sj in s:
        if sj[0] == sj[1]:
            pts[sj[0]] = sj[2]

    for pt, t in zip(pts, seq):
        _score += params["term"][0, pt, t]

    return _score


def score2(params, seq, span, nt):
    s, p = spans2tree(span)
    parent2children = defaultdict(list)
    for i, pj in enumerate(p):
        parent2children[pj].append(i)

    _score = 0
    for pj, children in parent2children.items():
        if len(children) == 1:
            assert pj == -1
            _score += params["root"][0, s[children[0]][2]]
        else:
            assert len(children) == 2
            l = s[children[0]]
            lt = l[2] + (0 if l[0] != l[1] else nt)
            r = s[children[1]]
            rt = r[2] + (0 if r[0] != r[1] else nt)
            pt = s[pj][2]
            _score += params["rule"][0, pt, lt, rt]

    pts = [None for _ in seq]
    for sj in s:
        if sj[0] == sj[1]:
            pts[sj[0]] = sj[2]

    for i, (pt, t) in enumerate(zip(pts, seq)):
        _score += params["term"][0, i, pt]

    return _score


def log_likelihood(params, x, n, nt, pt):
    ll = []

    for tree in enumerate_tree(0, n, nt, pt):
        tree = [(l, r - 1, t) for l, r, t in tree]
        stree = score(params, x, tree, nt)
        ll.append(stree)

    return torch.tensor(ll).logsumexp(0)


def log_likelihood2(params, x, n, nt, pt):
    ll = []

    for tree in enumerate_tree(0, n, nt, pt):
        tree = [(l, r - 1, t) for l, r, t in tree]
        stree = score2(params, x, tree, nt)
        ll.append(stree)

    return torch.tensor(ll).logsumexp(0)


def entropy(params, x, n, nt, pt):
    ll = []

    for tree in enumerate_tree(0, n, nt, pt):
        tree = [(l, r - 1, t) for l, r, t in tree]
        stree = score2(params, x, tree, nt)
        ll.append(stree)

    ll = torch.tensor(ll)
    ll = ll.log_softmax(0)
    return -(ll * ll.exp()).sum()


def ce_entropy(params, params2, x, n, nt, pt):
    ll = []
    ll2 = []
    for tree in enumerate_tree(0, n, nt, pt):
        tree = [(l, r - 1, t) for l, r, t in tree]
        stree = score2(params, x, tree, nt)
        ll.append(stree)
        ll2.append(score2(params2, x, tree, nt))

    ll = torch.tensor(ll)
    ll = ll.log_softmax(0)
    ll2 = torch.tensor(ll2)
    ll2 = ll2.log_softmax(0)
    return -(ll2 * ll.exp()).sum()


def kl(params, params2, x, n, nt, pt):
    ll = []
    ll2 = []
    for tree in enumerate_tree(0, n, nt, pt):
        tree = [(l, r - 1, t) for l, r, t in tree]
        stree = score2(params, x, tree, nt)
        ll.append(stree)
        ll2.append(score2(params2, x, tree, nt))

    ll = torch.tensor(ll)
    ll = ll.log_softmax(0)
    ll2 = torch.tensor(ll2)
    ll2 = ll2.log_softmax(0)
    return (ll.exp() * (ll - ll2)).sum()


def convert(params):
    return params["term"], params["rule"], params["root"]


B = 1
N = 2
VOCAB = 3
TGT_PT = 2
SRC_PT = 4
TGT_NT = 2
SRC_NT = 2
r = 2
NT = TGT_NT * SRC_NT
PT = TGT_PT * SRC_PT

seq = torch.randint(0, VOCAB, (B, N))
seq_inst = seq[0].tolist()
n = seq.size(1)
x_expand = seq.unsqueeze(2).expand(B, n, PT).unsqueeze(3)

lens = torch.tensor([N] * B)

params = {
    "term": torch.randn(B, PT, VOCAB).log_softmax(-1),
    "root": torch.randn(B, NT).log_softmax(-1),
    "rule": torch.randn(B, NT, (NT + PT) ** 2)
    .log_softmax(-1)
    .view(B, NT, NT + PT, NT + PT),
}
terms = params["term"].unsqueeze(1).expand(B, n, PT, params["term"].size(2))
terms = torch.gather(terms, 3, x_expand).squeeze(3)
params["term"] = terms

params2 = {
    "term": torch.randn(B, PT, VOCAB).log_softmax(-1),
    "root": torch.randn(B, NT).log_softmax(-1),
    "rule": torch.randn(B, NT, (NT + PT) ** 2)
    .log_softmax(-1)
    .view(B, NT, NT + PT, NT + PT),
}
terms = params2["term"].unsqueeze(1).expand(B, n, PT, params2["term"].size(2))
terms = torch.gather(terms, 3, x_expand).squeeze(3)
params2["term"] = terms

pcfg = PCFG()

print("Brute force ll", log_likelihood2(params, seq_inst, N, NT, PT))
print(pcfg(params, lens))

print("Brute force kl", kl(params, params2, seq_inst, N, NT, PT))
print(pcfg.kl(params, params2, lens))

print("Brute force ce", ce_entropy(params, params2, seq_inst, N, NT, PT))
print(pcfg.ce(params, params2, lens))
