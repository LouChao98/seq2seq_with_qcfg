from collections import Counter
from copy import copy
from itertools import product

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models.tgt_parser.base import TgtParserBase
from src.models.tgt_parser.neural_decomp1 import NeuralDecomp1TgtParser
from src.models.tgt_parser.neural_decomp2 import NeuralDecomp2TgtParser
from src.models.tgt_parser.neural_decomp3 import NeuralDecomp3TgtParser
from src.models.tgt_parser.neural_decomp5 import NeuralDecomp5TgtParser
from src.models.tgt_parser.neural_decomp7 import NeuralDecomp7TgtParser
from src.models.tgt_parser.neural_nodecomp import NeuralNoDecompTgtParser


def compute_unnormalized_prob(seq, parser: TgtParserBase, pred):
    x = torch.tensor([seq])
    pred = parser.observe_x(pred, x, [len(seq)], inplace=False)
    return pred.dist.partition.exp().item()


def enumerate(length, vocab):
    v = list(range(vocab))
    for i in range(2, length + 1):
        for x in product(*([v] * i)):
            yield x


@given(st.data())
@settings(max_examples=5, deadline=None)
def test_nodecomp_sampling(data):
    B = 2
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    SRC_N = data.draw(st.integers(min_value=2, max_value=3), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=3), label="tgt_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=3), label="tgt_nt")

    lens = [max(2, SRC_N - i) for i in range(B)]

    # left-branching
    spans = [[(i, i, 0) for i in range(l)] + [(0, i, 0) for i in range(1, l)] for l in lens]
    node_features = [torch.randn(l * 2 - 1, 8) for l in lens]

    parser = NeuralNoDecompTgtParser(
        TGT_PT,
        TGT_NT,
        dim=8,
        src_dim=8,
        num_layers=1,
        vocab=VOCAB,
        use_copy=False,
        generation_max_length=MAX_LENGTH,
        generation_num_samples=NUM_SAMPLE,
        generation_strict=True,
    )
    pred = parser(node_features, spans)
    pred = parser.prepare_sampler(pred, None, None)
    samples = pred.sampler()
    samples = parser.expand_preds_not_using_copy(samples)[0]

    for bidx in range(B):
        count = Counter(tuple(item) for item in samples[bidx])
        total = len(samples[bidx])

        sub_pred = copy(pred)
        params = pred.params
        params = {key: value[bidx, None] for key, value in params.items()}
        sub_pred.params = params
        sub_pred.batch_size = 1

        probs_with_seq = []
        for seq in enumerate(MAX_LENGTH, VOCAB):
            prob = compute_unnormalized_prob(seq, parser, sub_pred)
            probs_with_seq.append((prob, seq))

        partition = sum(item[0] for item in probs_with_seq)
        probs_with_seq.sort(reverse=True)

        errors = []
        empirical_cdf = 0
        theoratical_cfg = 0
        for prob, seq in probs_with_seq[:5]:
            empirical_cdf += count[tuple(seq)] / total
            theoratical_cfg += prob / partition
            errors.append(abs(empirical_cdf - theoratical_cfg) / theoratical_cfg)

        assert max(errors) < 0.1, errors


@given(st.data())
@settings(max_examples=5, deadline=None)
def test_decomp1_sampling(data):
    B = 2
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    SRC_N = data.draw(st.integers(min_value=2, max_value=3), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=3), label="tgt_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=3), label="tgt_nt")

    lens = [max(2, SRC_N - i) for i in range(B)]

    # left-branching
    spans = [[(i, i, 0) for i in range(l)] + [(0, i, 0) for i in range(1, l)] for l in lens]
    node_features = [torch.randn(l * 2 - 1, 8) for l in lens]

    parser = NeuralDecomp1TgtParser(
        TGT_PT,
        TGT_NT,
        dim=8,
        src_dim=8,
        num_layers=1,
        vocab=VOCAB,
        use_copy=False,
        generation_max_length=MAX_LENGTH,
        generation_num_samples=NUM_SAMPLE,
        generation_strict=True,
    )
    pred = parser(node_features, spans)
    pred = parser.prepare_sampler(pred, None, None)
    samples = pred.sampler()
    samples = parser.expand_preds_not_using_copy(samples)[0]

    for bidx in range(B):
        count = Counter(tuple(item) for item in samples[bidx])
        total = len(samples[bidx])

        sub_pred = copy(pred)
        params = pred.params
        params = {key: value[bidx, None] for key, value in params.items()}
        sub_pred.params = params
        sub_pred.batch_size = 1

        probs_with_seq = []
        for seq in enumerate(MAX_LENGTH, VOCAB):
            prob = compute_unnormalized_prob(seq, parser, sub_pred)
            probs_with_seq.append((prob, seq))

        partition = sum(item[0] for item in probs_with_seq)
        probs_with_seq.sort(reverse=True)

        errors = []
        empirical_cdf = 0
        theoratical_cfg = 0
        for prob, seq in probs_with_seq[:5]:
            empirical_cdf += count[tuple(seq)] / total
            theoratical_cfg += prob / partition
            errors.append(abs(empirical_cdf - theoratical_cfg) / theoratical_cfg)

        assert max(errors) < 0.1, errors


@given(st.data())
@settings(max_examples=5, deadline=None)
def test_decomp2_sampling(data):
    B = 2
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    SRC_N = data.draw(st.integers(min_value=2, max_value=3), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=3), label="tgt_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=3), label="tgt_nt")

    lens = [max(2, SRC_N - i) for i in range(B)]

    # left-branching
    spans = [[(i, i, 0) for i in range(l)] + [(0, i, 0) for i in range(1, l)] for l in lens]
    node_features = [torch.randn(l * 2 - 1, 8) for l in lens]

    parser = NeuralDecomp2TgtParser(
        TGT_PT,
        TGT_NT,
        dim=8,
        src_dim=8,
        num_layers=1,
        vocab=VOCAB,
        use_copy=False,
        generation_max_length=MAX_LENGTH,
        generation_num_samples=NUM_SAMPLE,
        generation_strict=True,
        direction=0,
    )
    pred = parser(node_features, spans)
    pred = parser.prepare_sampler(pred, None, None)
    samples = pred.sampler()
    samples = parser.expand_preds_not_using_copy(samples)[0]

    for bidx in range(B):
        count = Counter(tuple(item) for item in samples[bidx])
        total = len(samples[bidx])

        sub_pred = copy(pred)
        params = pred.params
        params = {key: value[bidx, None] for key, value in params.items()}
        sub_pred.params = params
        sub_pred.batch_size = 1

        probs_with_seq = []
        for seq in enumerate(MAX_LENGTH, VOCAB):
            prob = compute_unnormalized_prob(seq, parser, sub_pred)
            probs_with_seq.append((prob, seq))

        partition = sum(item[0] for item in probs_with_seq)
        probs_with_seq.sort(reverse=True)

        errors = []
        empirical_cdf = 0
        theoratical_cfg = 0
        for prob, seq in probs_with_seq[:5]:
            empirical_cdf += count[tuple(seq)] / total
            theoratical_cfg += prob / partition
            errors.append(abs(empirical_cdf - theoratical_cfg) / theoratical_cfg)

        assert max(errors) < 0.1, errors


@given(st.data())
@settings(max_examples=5, deadline=None)
def test_decomp3_sampling(data):
    B = 2
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    SRC_N = data.draw(st.integers(min_value=2, max_value=3), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=3), label="tgt_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=3), label="tgt_nt")

    lens = [max(2, SRC_N - i) for i in range(B)]

    # left-branching
    spans = [[(i, i, 0) for i in range(l)] + [(0, i, 0) for i in range(1, l)] for l in lens]
    node_features = [torch.randn(l * 2 - 1, 8) for l in lens]

    parser = NeuralDecomp3TgtParser(
        TGT_PT,
        TGT_NT,
        dim=8,
        src_dim=8,
        num_layers=1,
        vocab=VOCAB,
        use_copy=False,
        direction=0,
        generation_max_length=MAX_LENGTH,
        generation_num_samples=NUM_SAMPLE,
        generation_strict=True,
    )
    pred = parser(node_features, spans)
    pred = parser.prepare_sampler(pred, None, None)
    samples = pred.sampler()
    samples = parser.expand_preds_not_using_copy(samples)[0]

    for bidx in range(B):
        count = Counter(tuple(item) for item in samples[bidx])
        total = len(samples[bidx])

        sub_pred = copy(pred)
        params = pred.params
        params = {key: value[bidx, None] for key, value in params.items()}
        sub_pred.params = params
        sub_pred.batch_size = 1

        probs_with_seq = []
        for seq in enumerate(MAX_LENGTH, VOCAB):
            prob = compute_unnormalized_prob(seq, parser, sub_pred)
            probs_with_seq.append((prob, seq))

        partition = sum(item[0] for item in probs_with_seq)
        probs_with_seq.sort(reverse=True)

        errors = []
        empirical_cdf = 0
        theoratical_cfg = 0
        for prob, seq in probs_with_seq[:5]:
            empirical_cdf += count[tuple(seq)] / total
            theoratical_cfg += prob / partition
            errors.append(abs(empirical_cdf - theoratical_cfg) / theoratical_cfg)

        assert max(errors) < 0.1, errors


@given(st.data())
@settings(max_examples=5, deadline=None)
def test_decomp5_sampling(data):
    B = 2
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    SRC_N = data.draw(st.integers(min_value=2, max_value=3), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=3), label="tgt_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=3), label="tgt_nt")

    lens = [max(2, SRC_N - i) for i in range(B)]

    # left-branching
    spans = [[(i, i, 0) for i in range(l)] + [(0, i, 0) for i in range(1, l)] for l in lens]
    node_features = [torch.randn(l * 2 - 1, 8) for l in lens]

    parser = NeuralDecomp5TgtParser(
        TGT_PT,
        TGT_NT,
        dim=8,
        src_dim=8,
        num_layers=1,
        vocab=VOCAB,
        use_copy=False,
        direction=0,
        generation_max_length=MAX_LENGTH,
        generation_num_samples=NUM_SAMPLE,
        generation_strict=True,
    )
    pred = parser(node_features, spans)
    pred = parser.prepare_sampler(pred, None, None)
    samples = pred.sampler()
    samples = parser.expand_preds_not_using_copy(samples)[0]

    for bidx in range(B):
        count = Counter(tuple(item) for item in samples[bidx])
        total = len(samples[bidx])

        sub_pred = copy(pred)
        params = pred.params
        params = {key: value[bidx, None] for key, value in params.items()}
        sub_pred.params = params
        sub_pred.batch_size = 1

        probs_with_seq = []
        for seq in enumerate(MAX_LENGTH, VOCAB):
            prob = compute_unnormalized_prob(seq, parser, sub_pred)
            probs_with_seq.append((prob, seq))

        partition = sum(item[0] for item in probs_with_seq)
        probs_with_seq.sort(reverse=True)

        errors = []
        empirical_cdf = 0
        theoratical_cfg = 0
        for prob, seq in probs_with_seq[:5]:
            empirical_cdf += count[tuple(seq)] / total
            theoratical_cfg += prob / partition
            errors.append(abs(empirical_cdf - theoratical_cfg) / theoratical_cfg)

        assert max(errors) < 0.1, errors


@given(st.data())
@settings(max_examples=5, deadline=None)
def test_decomp7_sampling(data):
    B = 2
    VOCAB = 2
    NUM_SAMPLE = 10000
    MAX_LENGTH = 4
    SRC_N = data.draw(st.integers(min_value=2, max_value=3), label="n")
    TGT_PT = data.draw(st.integers(min_value=1, max_value=3), label="tgt_pt")
    TGT_NT = data.draw(st.integers(min_value=1, max_value=3), label="tgt_nt")

    lens = [max(2, SRC_N - i) for i in range(B)]

    # left-branching
    spans = [[(i, i, 0) for i in range(l)] + [(0, i, 0) for i in range(1, l)] for l in lens]
    node_features = [torch.randn(l * 2 - 1, 8) for l in lens]

    parser = NeuralDecomp7TgtParser(
        TGT_PT,
        TGT_NT,
        dim=8,
        src_dim=8,
        num_layers=1,
        vocab=VOCAB,
        use_copy=False,
        generation_max_length=MAX_LENGTH,
        generation_num_samples=NUM_SAMPLE,
        generation_strict=True,
    )
    pred = parser(node_features, spans)
    pred = parser.prepare_sampler(pred, None, None)
    samples = pred.sampler()
    samples = parser.expand_preds_not_using_copy(samples)[0]

    for bidx in range(B):
        count = Counter(tuple(item) for item in samples[bidx])
        total = len(samples[bidx])

        sub_pred = copy(pred)
        params = pred.params
        params = {key: value[bidx, None] for key, value in params.items()}
        sub_pred.params = params
        sub_pred.batch_size = 1

        probs_with_seq = []
        for seq in enumerate(MAX_LENGTH, VOCAB):
            prob = compute_unnormalized_prob(seq, parser, sub_pred)
            probs_with_seq.append((prob, seq))

        partition = sum(item[0] for item in probs_with_seq)
        probs_with_seq.sort(reverse=True)

        errors = []
        empirical_cdf = 0
        theoratical_cfg = 0
        for prob, seq in probs_with_seq[:5]:
            empirical_cdf += count[tuple(seq)] / total
            theoratical_cfg += prob / partition
            errors.append(abs(empirical_cdf - theoratical_cfg) / theoratical_cfg)

        assert max(errors) < 0.1, errors
