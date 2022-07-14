import torch.nn as nn
from allennlp.modules.augmented_lstm import BiAugmentedLstm as _allennlp_BiAugmentedLstm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        hidden = pad_packed_sequence(
            hidden, batch_first=True, total_length=total_length
        )[0]
        hidden = hidden.contiguous()
        return hidden

    def get_output_dim(self):
        return self.hidden_size * (2 if self.bidirectional else 1)


class EmptyModule(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x):
        return x

    def get_output_dim(self):
        return self.input_dim

