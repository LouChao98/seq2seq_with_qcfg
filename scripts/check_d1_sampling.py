from collections import Counter

import torch

from src.models.tgt_parser.neural_qcfg_d1 import NeuralQCFGD1TgtParser
from src.models.tgt_parser.struct.d1_pcfg import D1PCFG
from src.models.tgt_parser.struct.pcfg import PCFG

VOCAB = 2
NUM_SAMPLE = 100000
PROBE_ID = 1


def compute_unnormalized_prob(p, seq, model):
    x = torch.tensor([seq])
    n = x.size(1)
    B = 1
    T = 2 * 9
    terms = p["term"].unsqueeze(1).expand(B, n, T, p["term"].size(2))
    x_expand = x.unsqueeze(2).expand(B, n, T).unsqueeze(3)
    terms = torch.gather(terms, 3, x_expand).squeeze(3)
    params5 = {**p}
    params5["term"] = terms
    nll = model(params5, [x.shape[1]])
    return (-nll).exp().item()


tgt_parser = NeuralQCFGD1TgtParser(
    pt_states=2,
    nt_states=2,
    pt_span_range=[1, 1000],
    nt_span_range=[1, 1000],
    use_copy=False,
    dim=8,
    cpd_rank=3,
    src_dim=8,
    num_layers=1,
    vocab=VOCAB,
)

params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = tgt_parser.get_params(
    [torch.randn(9, 8), torch.randn(7, 8)],
    [
        [
            (0, 0, -1),
            (1, 1, -1),
            (2, 2, -1),
            (3, 3, -1),
            (4, 4, -1),
            (0, 1, -1),
            (2, 3, -1),
            (0, 3, -1),
            (0, 4, -1),
        ],
        [
            (0, 0, -1),
            (1, 1, -1),
            (2, 2, -1),
            (3, 3, -1),
            (1, 2, -1),
            (0, 2, -1),
            (0, 3, -1),
        ],
    ],
)
print("D1")

pcfg = D1PCFG(2, 2)
pred = pcfg.sampled_decoding(params, nt_spans, 2, pt_spans, 2, False, NUM_SAMPLE, max_length=4, strict=True)

new_pred = []

for inst in pred:
    new_pred_inst = []
    for item, type in inst:
        new_pred_inst.append(item)
    new_pred.append(new_pred_inst)
pred = new_pred

count = Counter("".join(map(str, item)) for item in pred[PROBE_ID])
print(count.most_common(15), sum(count.values()), len(count))

params = {key: value[PROBE_ID, None] if isinstance(value, torch.Tensor) else value for key, value in params.items()}

probs = []
probs_with_seq = []
for a in range(VOCAB):
    for b in range(VOCAB):
        prob = compute_unnormalized_prob(params, [a, b], pcfg)
        probs.append(prob)
        probs_with_seq.append((prob, [a, b]))
for a in range(VOCAB):
    for b in range(VOCAB):
        for c in range(VOCAB):
            prob = compute_unnormalized_prob(params, [a, b, c], pcfg)
            probs.append(prob)
            probs_with_seq.append((prob, [a, b, c]))
for a in range(VOCAB):
    for b in range(VOCAB):
        for c in range(VOCAB):
            for d in range(VOCAB):
                prob = compute_unnormalized_prob(params, [a, b, c, d], pcfg)
                probs.append(prob)
                probs_with_seq.append((prob, [a, b, c, d]))
partition = sum(probs)

probs_with_seq.sort(reverse=True)
num_sample = len(pred[PROBE_ID])
print(num_sample)
for prob, seq in probs_with_seq[:15]:
    print(seq, prob / partition * num_sample, prob / partition)


print("=" * 79)

print("Ref")


pcfg_ref = PCFG()
params2 = D1PCFG.get_pcfg_rules(params, 2)
pred_ref = pcfg_ref.sampled_decoding(params2, nt_spans, 2, pt_spans, 2, False, NUM_SAMPLE, max_length=4, strict=True)

new_pred = []

for inst in pred_ref:
    new_pred_inst = []
    for item, *_ in inst:
        new_pred_inst.append(item)
    new_pred.append(new_pred_inst)
pred_ref = new_pred

count_ref = Counter("".join(map(str, item)) for item in pred_ref[0])
print(count_ref.most_common(15), sum(count_ref.values()), len(count_ref))

probs = []
probs_with_seq = []
for a in range(VOCAB):
    for b in range(VOCAB):
        prob = compute_unnormalized_prob(params2, [a, b], pcfg_ref)
        probs.append(prob)
        probs_with_seq.append((prob, [a, b]))
for a in range(VOCAB):
    for b in range(VOCAB):
        for c in range(VOCAB):
            prob = compute_unnormalized_prob(params2, [a, b, c], pcfg_ref)
            probs.append(prob)
            probs_with_seq.append((prob, [a, b, c]))
for a in range(VOCAB):
    for b in range(VOCAB):
        for c in range(VOCAB):
            for d in range(VOCAB):
                prob = compute_unnormalized_prob(params2, [a, b, c, d], pcfg_ref)
                probs.append(prob)
                probs_with_seq.append((prob, [a, b, c, d]))
partition = sum(probs)

probs_with_seq.sort(reverse=True)
num_sample = len(pred_ref[0])
print(num_sample)
for prob, seq in probs_with_seq[:15]:
    print(seq, prob / partition * num_sample, prob / partition)
