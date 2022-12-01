from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CombinarySpanEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout) -> None:
        super().__init__()
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, in_dim)
        self.bilinear = nn.Bilinear(in_dim, in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x1, x2):
        x1 = self.act(self.linear1(self.dropout(x1)))
        x2 = self.act(self.linear2(self.dropout(x2)))
        out = self.bilinear(x1, x2)
        return out

    def get_output_dim(self):
        return self.out_dim


# >>>>>>>>>>>>> From HTNN >>>>>>>>>>>>>


def get_span_mask(start_ids, end_ids, max_len, device):
    tmp = torch.arange(max_len).unsqueeze(0).expand(start_ids.shape[0], -1)
    batch_start_ids = start_ids.unsqueeze(1).expand_as(tmp)
    batch_end_ids = end_ids.unsqueeze(1).expand_as(tmp)

    tmp = tmp.to(device)
    batch_start_ids = batch_start_ids.to(device)
    batch_end_ids = batch_end_ids.to(device)

    mask = (
        (tmp >= batch_start_ids).float() * (tmp <= batch_end_ids).float()
    ).unsqueeze(2)
    return mask


class SpanRepr(ABC, nn.Module):
    """Abstract class describing span representation."""

    def __init__(self, input_dim, use_proj=False, proj_dim=256):
        super(SpanRepr, self).__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Linear(self.input_dim, self.proj_dim)

    @abstractmethod
    def forward(self, encoded_input, start_ids, end_ids):
        raise NotImplementedError

    def get_input_dim(self):
        return self.input_dim

    @abstractmethod
    def get_output_dim(self):
        raise NotImplementedError


class AvgSpanRepr(SpanRepr, nn.Module):
    """Class implementing the avg span representation."""

    def forward(self, encoded_input, start_ids, end_ids):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)

        span_lengths = (end_ids - start_ids + 1).unsqueeze(1).cuda()
        span_masks = get_span_mask(
            start_ids, end_ids, encoded_input.shape[1], encoded_input.device
        )
        span_repr = torch.sum(encoded_input * span_masks, dim=1) / span_lengths.float()
        return span_repr

    def get_output_dim(self):
        if self.use_proj:
            return self.proj_dim
        else:
            return self.input_dim


class AttnSpanRepr(SpanRepr, nn.Module):
    """Class implementing the attention-based span representation."""

    def __init__(self, input_dim, use_proj=False, proj_dim=256, use_endpoints=False):
        """If use_endpoints is true then concatenate the end points to attention-pooled span repr.
        Otherwise just return the attention pooled term.
        """
        super(AttnSpanRepr, self).__init__(
            input_dim, use_proj=use_proj, proj_dim=proj_dim
        )
        self.use_endpoints = use_endpoints
        if use_proj:
            input_dim = proj_dim
        self.attention_params = nn.Linear(input_dim, 1)
        # Initialize weight to zero weight
        # self.attention_params.weight.data.fill_(0)
        # self.attention_params.bias.data.fill_(0)

    def forward(self, encoded_input, start_ids, end_ids):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)

        span_mask = get_span_mask(
            start_ids, end_ids, encoded_input.shape[1], encoded_input.device
        )
        attn_mask = (1 - span_mask) * (-1e10)
        attn_logits = self.attention_params(encoded_input) + attn_mask
        attention_wts = nn.functional.softmax(attn_logits, dim=1)
        attention_term = torch.sum(attention_wts * encoded_input, dim=1)
        if self.use_endpoints:
            batch_size = encoded_input.shape[0]
            h_start = encoded_input[torch.arange(batch_size), start_ids, :]
            h_end = encoded_input[torch.arange(batch_size), end_ids, :]
            return torch.cat([h_start, h_end, attention_term], dim=1)
        else:
            return attention_term

    def get_output_dim(self):
        if not self.use_endpoints:
            if self.use_proj:
                return self.proj_dim
            else:
                return self.input_dim
        else:
            if self.use_proj:
                return 3 * self.proj_dim
            else:
                return 3 * self.input_dim


# <<<<<<<<<< From HTNN. End. <<<<<<<<<<
