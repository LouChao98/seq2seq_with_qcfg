from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from supar.modules import MLP, Biaffine
from supar.structs import ConstituencyCRF

from .base import SrcParserBase


class NeuralCRFSrcParser(SrcParserBase):
    def __init__(
        self,
        n_span_mlp,
        n_encoder_hidden,
        mlp_dropout,
    ):
        super(NeuralCRFSrcParser, self).__init__()
        self.span_mlp_l = MLP(n_in=n_encoder_hidden, n_out=n_span_mlp, dropout=mlp_dropout)
        self.span_mlp_r = MLP(n_in=n_encoder_hidden, n_out=n_span_mlp, dropout=mlp_dropout)
        self.span_attn = Biaffine(n_in=n_span_mlp, bias_x=True, bias_y=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, encoded, lengths, extra_scores=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.
        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible constituents.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each constituent.
        """

        x_f, x_b = encoded.chunk(2, -1)
        x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)

        span_l = self.span_mlp_l(x)
        span_r = self.span_mlp_r(x)

        # [batch_size, seq_len, seq_len]
        s_span = self.span_attn(span_l, span_r)

        return ConstituencyCRF(s_span, torch.tensor(lengths, device=s_span.device))

    def marginals(self, x, lengths, dist: Optional[ConstituencyCRF] = None, extra_scores=None):
        raise NotImplementedError

    @torch.enable_grad()
    def sample(self, x, lengths, dist: Optional[ConstituencyCRF] = None, extra_scores=None):
        if dist is None:
            dist = self(x, lengths, extra_scores)

        output = dist.sample()
        logprobs = dist.score(output > 0.99) - dist.log_partition
        span_preds = [sorted(torch.nonzero(i).tolist(), key=lambda x: (x[0], -x[1])) for i in output]
        return {"event": output, "span": span_preds}, logprobs

    def gumbel_sample(self, x, lengths, temperature, dist: Optional[ConstituencyCRF] = None, extra_scores=None):
        raise NotImplementedError

    @torch.enable_grad()
    def argmax(self, x, lengths, dist: Optional[ConstituencyCRF] = None, extra_scores=None):
        if dist is None:
            dist = self(x, lengths, extra_scores)

        return dist.argmax

    @torch.enable_grad()
    def entropy(self, x, lengths, dist: Optional[ConstituencyCRF] = None, extra_scores=None):
        if dist is None:
            dist = self(x, lengths, extra_scores)

        return dist.entropy
