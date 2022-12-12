import math
from collections import Counter
from logging import getLogger
from turtle import forward

import numpy as np
import torch
import torch.nn as nn
from allennlp.modules.augmented_lstm import BiAugmentedLstm as _allennlp_BiAugmentedLstm
from allennlp.modules.span_extractors.self_attentive_span_extractor import SelfAttentiveSpanExtractor
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from hydra.utils import instantiate
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

logger = getLogger(__file__)


def get_span_extractor_module(name, *args, **kwargs):
    # http://docs.allennlp.org/main/api/modules/span_extractors/span_extractor/
    return SpanExtractor.by_name(name)(*args, **kwargs)


class _CachedModule(torch.nn.Module):
    def __init__(self, origin_module) -> None:
        super().__init__()
        self.origin_module = origin_module
        self.__dict__["cache"] = None

    def forward(self, *args, **kwargs):
        if (result := self.__dict__["cache"]) is not None:
            return result
        result = self.origin_module(*args, **kwargs)
        self.__dict__["cache"] = result
        return result

    def reset(self):
        self.__dict__["cache"] = None


class BucketedSpanExtractor(nn.Module):
    # allennlp will pad all spans to max_span_width in some extractors (e.g., self-attentive)
    # this wrapper class split spans into buckets by length to save memory
    # assume span_indices are sorted according to span width(1 to n)

    def __init__(self, module, n_bucket, input_dim):
        super().__init__()
        self.module: SpanExtractor = instantiate(module, input_dim=input_dim)

        if isinstance(self.module, SelfAttentiveSpanExtractor):
            self.module._global_attention = _CachedModule(self.module._global_attention)

        self.n_bucket = n_bucket
        self.output_dim = self.module.get_output_dim()

        logger.warning("BucketedSpanExtractor assume input span_indices are sorted by width.")

    def forward(
        self,
        sequence_tensor,
        span_indices: torch.Tensor,
        sequence_mask=None,
        span_indices_mask=None,
    ):
        # the two masks seem to be useless
        if self.n_bucket == "all":
            counter = Counter((span_indices[0, ..., 1] - span_indices[0, ..., 0]).tolist())
            counter = counter.most_common()
            _segment = [c[1] for c in counter]
            segment = [_segment[0]]
            for s in _segment[1:]:
                if segment[-1] < 5:
                    segment[-1] += s
                else:
                    segment.append(s)
        else:
            if isinstance(self.n_bucket, int):
                n_bucket = self.n_bucket
            elif self.n_bucket == "log":
                n_bucket = math.ceil(math.log2(sequence_tensor.shape[1]))
            elif self.n_bucket == "sqrt":
                n_bucket = math.ceil(math.sqrt(sequence_tensor.shape[1]))
            else:
                raise ValueError(f"Bad n_bucket: {self.n_bucket}")
            # a segment such that in each block, padding size is similar
            segment = get_simple_seg((span_indices[0, :, 1] - span_indices[0, :, 0]).cpu().numpy(), n_bucket)
        assert sum(segment) == span_indices.shape[1], f"{sum(segment)}, {span_indices.shape[1]}"
        span_indices = span_indices.split(segment, dim=1)
        output = []
        for piece in span_indices:
            output.append(self.module(sequence_tensor, piece))
        if isinstance(self.module, SelfAttentiveSpanExtractor):
            self.module._global_attention.reset()
        return torch.cat(output, dim=1)

    def get_output_dim(self):
        return self.module.get_output_dim()


class BiAugmentedLstm(_allennlp_BiAugmentedLstm):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        recurrent_dropout_probability: float = 0,
        bidirectional: bool = False,
        padding_value: float = 0,
        use_highway: bool = True,
    ) -> None:
        super().__init__(
            input_dim,
            hidden_size,
            num_layers,
            bias,
            recurrent_dropout_probability,
            bidirectional,
            padding_value,
            use_highway,
        )

    def forward(self, hidden, seq_len):
        total_length = hidden.shape[1]
        hidden = pack_padded_sequence(hidden, seq_len, batch_first=True)
        hidden = super().forward(hidden)[0]
        hidden = pad_packed_sequence(hidden, batch_first=True, total_length=total_length)[0]
        hidden = hidden.contiguous()
        return hidden

    def get_output_dim(self):
        return self.hidden_size * (2 if self.bidirectional else 1)


def get_simple_seg(x, n_bucket):
    from sklearn.cluster import KMeans

    x = np.expand_dims(x, 1)
    init = np.expand_dims(np.arange(n_bucket) * 1.0 / n_bucket * x[0, 0], 1)
    output = KMeans(n_bucket, init=init).fit(x).predict(x)
    c = [(output == i).sum() for i in range(n_bucket)]
    return c


if __name__ == "__main__":
    # Note: module.get_output_dim()

    get_span_extractor_module(
        "endpoint",
        input_dim=32,
        combination="x,y",  # x  y  x*y  x+y  x-y  x/y
        num_width_embeddings=10,
        span_width_embedding_dim=32,
        bucket_widths=True,  # bucket span widths into log-space buckets
    )

    get_span_extractor_module(
        "max_pooling",
        input_dim=32,
        num_width_embeddings=10,
        span_width_embedding_dim=32,
        bucket_widths=True,  # bucket span widths into log-space buckets
    )

    get_span_extractor_module(
        "self_attentive",
        input_dim=32,
        num_width_embeddings=10,
        span_width_embedding_dim=32,
        bucket_widths=True,  # bucket span widths into log-space buckets
    )
