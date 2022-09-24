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


def calculate_e(params, x, n, nt, pt, src_pt):
    ll = log_likelihood2(params, x, n, nt, pt)
    src_pt_inst = torch.zeros(src_pt, dtype=torch.float64)
    records = []
    for tree in enumerate_tree(0, n, nt, pt):
        tree = [(l, r - 1, t) for l, r, t in tree]
        prob = math.exp(score2(params, x, tree, nt) - ll)
        arr = torch.zeros(src_pt, dtype=torch.float64)
        for l, r, t in tree:
            if l == r:
                arr[t % src_pt] += 1
        records.append((prob, arr))
        src_pt_inst += arr * prob

    records.sort(key=lambda x: x[0], reverse=True)
    pprint(records[:10])

    cnt_by_phi = defaultdict(float)
    for prob, arr in records:
        arr = "".join(map(str, arr.long().tolist()))
        cnt_by_phi[arr] += prob
    cnt_by_phi = list(cnt_by_phi.items())
    cnt_by_phi.sort(key=lambda x: x[1], reverse=True)
    pprint(cnt_by_phi[:4])
    return src_pt_inst


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


def convert(params):
    return params["term"], params["rule"], params["root"]


B = 1
N = 3
VOCAB = 3
TGT_PT = 2
SRC_PT = 4
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
print("ll1", log_likelihood(params, seq_inst, N, NT, PT))

n = seq.size(1)
terms = params["term"].unsqueeze(1).expand(B, n, PT, params["term"].size(2))
x_expand = seq.unsqueeze(2).expand(B, n, PT).unsqueeze(3)
terms = torch.gather(terms, 3, x_expand).squeeze(3)
params_x = {**params}
params_x["term"] = terms
params_x["copy_nt"] = None

pcfg = PCFG()
print("ref", (-pcfg(params_x, lens))[0].detach())

print("ll2", log_likelihood2(params_x, seq_inst, N, NT, PT))

pr_task = AMRNeqQCFGPrTask(SRC_PT, SRC_NT, TGT_PT, TGT_NT, "cpu")
constraint = torch.ones_like(seq)
cc = pr_task.process_constraint(constraint)

ec = calculate_e(params_x, seq_inst, N, NT, PT, SRC_PT)
print("Before PR", ec)

if (ec < pr_task.get_b(len(lens))).all():
    exit(0)

# print(pr_task.calc_e(SentCFG(convert(params_x), lens), cc))

print("======")

# cparams = compute_pr(
#     params_x, lens, torch.ones_like(seq), pr_task, get_param=True, num_iter=3
# )
# ec = calculate_e(cparams, seq_inst, N, NT, PT, SRC_PT)
# print("After PR", ec)

# print("======")

cparams = compute_pr(
    params_x, lens, torch.ones_like(seq), pr_task, get_param=True, num_iter=100
)
ec = calculate_e(cparams, seq_inst, N, NT, PT, SRC_PT)
print("After PR", ec)
print("Entropy", entropy(cparams, seq_inst, N, NT, PT))
print("======")

cparams = compute_pr(
    params_x,
    lens,
    torch.ones_like(seq),
    pr_task,
    get_param=True,
    num_iter=100,
    entropy_reg=0.5,
)
ec = calculate_e(cparams, seq_inst, N, NT, PT, SRC_PT)
print("After PR", ec)
print("Entropy", entropy(cparams, seq_inst, N, NT, PT))
print("======")

cparams = compute_pr(
    params_x,
    lens,
    torch.ones_like(seq),
    pr_task,
    get_param=True,
    num_iter=100,
    entropy_reg=0.1,
)
ec = calculate_e(cparams, seq_inst, N, NT, PT, SRC_PT)
print("After PR", ec)
print("Entropy", entropy(cparams, seq_inst, N, NT, PT))

# print("======")

# cparams = compute_pr(
#     params_x, lens, torch.ones_like(seq), pr_task, get_param=True, num_iter=200, entropy_reg=0.5
# )
# ec = calculate_e(cparams, seq_inst, N, NT, PT, SRC_PT)
# print("After PR", ec)
# print('Entropy', entropy(cparams, seq_inst, N, NT, PT))

# print(pr_task.calc_e(SentCFG(convert(cparams), lens), cc))
