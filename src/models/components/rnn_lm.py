# Copy from https://github.com/kamigaito/rnnlm-pytorch with some minor modification

import torch
import torch.nn as nn
import torch.nn.functional as F


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class ResRNNBase(nn.Module):
    """
    RNN with residual connections
    """

    def __init__(self, rnn_type, ninp, nhid, nlayers, nonlinearity="tanh", dropout=0.2, bidirectional=False):
        super(ResRNNBase, self).__init__()
        self.bidirectional = bidirectional
        self.drop = nn.Dropout(dropout)
        if self.bidirectional:
            if rnn_type in ["LSTM", "GRU"]:
                self.forward_rnns = nn.ModuleList(
                    [getattr(nn, rnn_type)(ninp, nhid, 1, dropout=0.0, bidirectional=False) for _ in range(nlayers)]
                )
                self.backward_rnns = nn.ModuleList(
                    [getattr(nn, rnn_type)(ninp, nhid, 1, dropout=0.0, bidirectional=False) for _ in range(nlayers)]
                )
            elif rnn_type == "RNN":
                self.forward_rnns = nn.ModuleList(
                    [
                        getattr(nn, rnn_type)(
                            ninp, nhid, 1, nonlinearity=nonlinearity, dropout=0.0, bidirectional=False
                        )
                        for _ in range(nlayers)
                    ]
                )
                self.backward_rnns = nn.ModuleList(
                    [
                        getattr(nn, rnn_type)(
                            ninp, nhid, 1, nonlinearity=nonlinearity, dropout=0.0, bidirectional=False
                        )
                        for _ in range(nlayers)
                    ]
                )
        else:
            if rnn_type in ["LSTM", "GRU"]:
                self.rnns = nn.ModuleList(
                    [
                        getattr(nn, rnn_type)(ninp, nhid, 1, dropout=0.0, bidirectional=bidirectional)
                        for _ in range(nlayers)
                    ]
                )
            elif rnn_type == "RNN":
                self.rnns = nn.ModuleList(
                    [
                        getattr(nn, rnn_type)(
                            ninp, nhid, 1, nonlinearity=nonlinearity, dropout=0.0, bidirectional=bidirectional
                        )
                        for _ in range(nlayers)
                    ]
                )

    def forward(self, emb, hidden):
        if self.bidirectional:
            return self.forward_both(emb, hidden)
        else:
            return self.forward_one(emb, hidden)

    def forward_one(self, emb, init_hidden_list):
        """
        Inputs
        ----------
        emb: [seq_len - 1, nbatch, emb]
        init_hidden_list: tuple

        Returns
        ----------
        rnn_out: [seq_len - 1, nbatch, tok_hid]
        hidden_list: tuple
        """

        """ The number of layers should be same. """
        assert len(self.rnns) == len(init_hidden_list)

        emb = self.drop(emb)
        rnn_out = emb
        hidden_list = []
        for layer_id in range(len(self.rnns)):
            rnn = self.rnns[layer_id]
            init_hidden = init_hidden_list[layer_id]
            res_out = rnn_out
            # output: [seq_len, nbatch, nhid], hidden: [1, nbatch, nhid] or ([1, nbatch, nhid], [1, nbatch, nhid])
            rnn_out, hidden = rnn(rnn_out, init_hidden)
            # residual connection
            rnn_out = rnn_out + res_out
            # dropout
            if layer_id < len(self.rnns) - 1:
                rnn_out = self.drop(rnn_out)
            # store hidden states
            hidden_list.append(hidden)
        return rnn_out, hidden_list

    def forward_both(self, emb, init_hidden_list):
        """
        Inputs
        ----------
        emb: [seq_len, nbatch, emb]
        init_hidden_list: tuple

        Returns
        ----------
        rnn_out: [seq_len, nbatch, tok_hid]
        hidden_list: tuple
        """

        """ The number of layers should be same. """
        assert len(self.forward_rnns) == len(init_hidden_list)
        assert len(self.backward_rnns) == len(init_hidden_list)

        emb = self.drop(emb)
        forward_rnn_out = emb
        """ Reverse the order of token embeddings for the backward rnn. """
        backward_rnn_out = torch.flip(emb, [0])
        forward_hidden_list = []
        backward_hidden_list = []

        """ forward """
        for layer_id in range(len(init_hidden_list)):
            forward_res_out = forward_rnn_out
            forward_init_hidden = init_hidden_list[layer_id][0]
            forward_rnn = self.forward_rnns[layer_id]
            # output: [seq_len, nbatch, nhid], hidden: [1, nbatch, nhid] or ([1, nbatch, nhid], [1, nbatch, nhid])
            forward_rnn_out, forward_rnn_hidden = forward_rnn(forward_rnn_out, forward_init_hidden)
            forward_hidden_list.append(forward_rnn_hidden)
            forward_rnn_out = self.drop(forward_rnn_out) + forward_res_out
        """ Shift the forward hidden states. """
        forward_rnn_out = torch.cat(
            [
                torch.zeros(1, forward_rnn_out.shape[1], forward_rnn_out.shape[2], device=forward_rnn_out.device),
                forward_rnn_out[:-1],
            ],
            0,
        )

        """ backward """
        for layer_id in range(len(init_hidden_list)):
            backward_res_out = backward_rnn_out
            backward_init_hidden = init_hidden_list[layer_id][1]
            backward_rnn = self.backward_rnns[layer_id]
            # output: [seq_len, nbatch, nhid], hidden: [1, nbatch, nhid] or ([1, nbatch, nhid], [1, nbatch, nhid])
            backward_rnn_out, backward_rnn_hidden = backward_rnn(backward_rnn_out, backward_init_hidden)
            backward_rnn_out = self.drop(backward_rnn_out) + backward_res_out
        """ Reverse the order of hidden states """
        backward_rnn_out = torch.flip(backward_rnn_out, [0])
        """ Shift the backward hidden states """
        backward_rnn_out = torch.cat(
            [
                backward_rnn_out[1:],
                torch.zeros(1, backward_rnn_out.shape[1], backward_rnn_out.shape[2], device=backward_rnn_out.device),
            ],
            0,
        )

        """ concatenate output states """
        rnn_out = torch.cat([forward_rnn_out, backward_rnn_out], 2)
        hidden_list = zip(forward_hidden_list, backward_hidden_list)

        return rnn_out, hidden_list


class ResLSTM(ResRNNBase):
    """
    LSTM with residual connections
    """

    def __init__(self, ninp, nhid, nlayers, dropout, bidirectional):
        super(ResLSTM, self).__init__("LSTM", ninp, nhid, nlayers, dropout=dropout, bidirectional=bidirectional)


class ResGRU(ResRNNBase):
    """
    GRU with residual connections
    """

    def __init__(self, ninp, nhid, nlayers, dropout, bidirectional):
        super(ResGRU, self).__init__("GRU", ninp, nhid, nlayers, dropout=dropout, bidirectional=bidirectional)


class ResRNN(ResRNNBase):
    """
    RNN with residual connections
    """

    def __init__(self, ninp, nhid, nlayers, nonlinearity="tanh", dropout=0.0, bidirectional=False):
        super(ResRNN, self).__init__(
            "RNN", ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout, bidirectional=bidirectional
        )


class RNNModel(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    """

    def __init__(self, **prm):
        prm.setdefault("rnn_type", "LSTM")
        prm.setdefault("direction", "both")
        prm.setdefault("tok_emb", 64)
        prm.setdefault("tok_hid", 64)
        prm.setdefault("nlayers", 1)
        prm.setdefault("dropout", 0.2)
        prm.setdefault("init_range", 0.1)
        prm.setdefault("tied", False)

        super(RNNModel, self).__init__()
        self.prm = prm
        self.drop = nn.Dropout(prm["dropout"])
        self.emb_dim = prm["tok_emb"]
        self.word_encoder = nn.Embedding(prm["tok_len"], prm["tok_emb"])

        if prm["direction"] == "both":
            bidirectional = True
            self.decoder = nn.Linear(prm["tok_hid"] * 2, prm["tok_len"])
        elif prm["direction"] == "left2right" or prm["direction"] == "right2left":
            bidirectional = False
            self.decoder = nn.Linear(prm["tok_hid"], prm["tok_len"])
        else:
            assert False
        if prm["rnn_type"] in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, prm["rnn_type"])(
                self.emb_dim, prm["tok_hid"], prm["nlayers"], dropout=prm["dropout"], bidirectional=bidirectional
            )
        elif prm["rnn_type"] in ["ResLSTM", "ResGRU"]:
            self.rnn = globals()[prm["rnn_type"]](
                self.emb_dim, prm["tok_hid"], prm["nlayers"], dropout=prm["dropout"], bidirectional=bidirectional
            )
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu", "ResRNN_TANH": "tanh", "ResRNN_RELU": "relu"}[
                    prm["rnn_type"]
                ]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--rnn_type` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            if prm["rnn_type"] in ["ResRNN_TANH", "ResRNN_RELU"]:
                self.rnn = ResRNN(
                    self.emb_dim,
                    prm["tok_hid"],
                    prm["nlayers"],
                    nonlinearity=nonlinearity,
                    dropout=prm["dropout"],
                    bidirectional=bidirectional,
                )
            else:
                self.rnn = nn.RNN(
                    self.emb_dim,
                    prm["tok_hid"],
                    prm["nlayers"],
                    nonlinearity=nonlinearity,
                    dropout=prm["dropout"],
                    bidirectional=bidirectional,
                )

        """
        Optionally tie weights as in:
        "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        https://arxiv.org/abs/1608.05859
        and
        "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        https://arxiv.org/abs/1611.01462
        """
        if prm["tied"]:
            if prm["tok_hid"] != prm["tok_emb"]:
                raise ValueError("When using the tied flag, nhid must be equal to emsize")
            self.decoder.weight = self.word_encoder.weight

        self.init_weights()

    def init_weights(self):
        """
        Initialize the model weights
        """
        init_range = self.prm["init_range"]
        self.word_encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def embed(self, input):
        """
        Calculate embeddings for each token

        Inputs
        ----------
        input: [seq_len, nbatch]

        Returns
        ----------
        emb: The shape of this value depends on self.prm["wo_tok"] and self.prm["wo_char"]
        """
        emb = self.drop(self.word_encoder(input["word"]))
        return emb

    def freeze_emb(self):
        self.word_encoder.weight.requires_grad = False
        self.decoder.weight.requires_grad = False

    def forward(self, batch, hidden):
        """
        Inputs
        ----------
            input : torch.Tensor [seq_len, nbatch]
            hidden : torch.Tensor [seq_len, nbatch, nhid]
        Returns
        ----------
            decoded: torch.Tensor [seq_len, nbatch, ntoken]
            hidden: torch.Tensor [seq_len, nbatch, nhid]

        """
        # for mono-directional language models
        if self.prm["direction"] in ["left2right", "right2left"]:
            decoded, word_repre, last_hidden = self.forward_one(batch, hidden)
        # for bi-directional language models
        elif self.prm["direction"] == "both":
            decoded, word_repre, last_hidden = self.forward_both(batch, hidden)
        else:
            assert False
        return decoded, word_repre, last_hidden

    def forward_one(self, input, hidden):
        """
        For mono-directional language models

        Inputs
        ----------
            input : torch.Tensor [seq_len, nbatch]
            hidden : torch.Tensor [seq_len, nbatch, nhid]
        Returns
        ----------
            decoded: torch.Tensor [seq_len, nbatch, ntoken]
            hidden: torch.Tensor [seq_len, nbatch, nhid]

        """
        # emb: [seq_len, nbatch, ninp]
        emb = self.embed(input)
        # output: [seq_len, nbatch, nhid], hidden: [nlayer, nbatch, nhid] or ([nlayer, nbatch, nhid], [nlayer, nbatch, nhid])
        output, hidden = self.rnn(emb, hidden)
        # output: [seq_len, nbatch, nhid]
        output = self.drop(output)
        # output.view: [seq_len*nbatch, nhid], decoded: [seq_len*nbatch, ntoken]
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        # decoded.view: [seq_len, nbatch, ntoken], hidden: [nlayer, nbatch, nhid]
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), output, hidden

    def forward_both(self, input, hidden):
        """
        For bi-directional language models

        Inputs
        ----------
            input : torch.Tensor [seq_len, nbatch]
            hidden : torch.Tensor [seq_len, nbatch, nhid]
        Returns
        ----------
            decoded: torch.Tensor [seq_len, nbatch, ntoken]
            hidden: torch.Tensor [seq_len, nbatch, nhid]

        """
        # emb: [seq_len, nbatch, ninp]
        emb = self.embed(input)
        # output: [seq_len, nbatch, 2*nhid], hidden: [nlayer, nbatch, 2*nhid] or ([nlayer, nbatch, 2*nhid], [nlayer, nbatch, 2*nhid])
        output, hidden = self.rnn(emb, hidden)
        # output: [seq_len, nbatch, 2*nhid]
        output = self.drop(output)
        # output_splitted: [seq_len, nbatch, 2, nhid]
        output_splitted = output.reshape(output.shape[0], output.shape[1], 2, int(output.shape[2] / 2))
        # output_forward: [seq_len, nbatch, nhid]
        output_forward = output_splitted[:, :, 0, :].squeeze(2)
        # output_backward: [seq_len, nbatch, nhid]
        output_backward = output_splitted[:, :, 1, :].squeeze(2)
        """
        Shift the forward and backward hidden states

        # Before
        Forward:  A B C D
        Backward: A B C D
        Output:   A B C D

        # After
        Forward:  0 A B C
        Backward: B C D 0
        Output:   A B C D
        """
        # output_forward: [seq_len, nbatch, nhid]
        output_forward = torch.cat(
            [
                torch.zeros(1, output_forward.shape[1], output_forward.shape[2], device=output_forward.device),
                output_forward[:-1],
            ],
            0,
        )
        # output_backward: [seq_len, nbatch, nhid]
        output_backward = torch.cat(
            [
                output_backward[1:],
                torch.zeros(1, output_backward.shape[1], output_backward.shape[2], device=output_backward.device),
            ],
            0,
        )
        """ concatenate the forward and backward hidden states """
        # output: [seq_len, nbatch, nhid*2]
        output = torch.cat([output_forward, output_backward], 2)
        # output.view: [seq_len*nbatch, nhid*2], decoded: [seq_len*nbatch, ntoken]
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        # decoded.view: [seq_len, nbatch, ntoken], hidden: [nlayer, nbatch, nhid]
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), output, hidden

    def batch2input(self, batch, device):
        """
        Convert a batch into the input of the model
        Left most or right most words are removed for training.
        Inputs
        ----------
            batch : Dict
            device : A device that has the model
        Return
        ----------
            Dict
        """
        seq_len = batch["seq_len"]

        if seq_len != batch["word"]["index"].shape[1]:
            assert seq_len == batch["word"]["index"].shape[1]
        if self.prm["direction"] == "left2right":
            # input_word: [seq_len - 1, nbatch]
            input_word = batch["word"]["index"][:, :-1].t().to(device)
        elif self.prm["direction"] == "right2left":
            # input_word: [seq_len - 1, nbatch]
            input_word = torch.flip(batch["word"]["index"], [0])[:, :-1].t().to(device)
        elif self.prm["direction"] == "both":
            # input_word: [seq_len, nbatch]
            input_word = batch["word"]["index"].t().to(device)
        else:
            assert False
        # print(input_word.t())
        return {"word": input_word}

    def word2input(self, batch, device):
        """
        Convert a batch into the input of the model

        Inputs
        ----------
            batch : Dict
            device : A device that has the model
        Return
        ----------
            Dict
        """
        seq_len = batch["seq_len"]
        if seq_len != batch["word"]["index"].shape[1]:
            assert seq_len == batch["word"]["index"].shape[1]
        # input_word: [seq_len, nbatch]
        input_word = batch["word"]["index"].t().to(device)
        return {"word": input_word}

    def batch2flat(self, batch, device):
        """
        Convert a batch into the output of the model

        Inputs
        ----------
            batch : Dict
            device : A device that has the model
        Return
        ----------
            Dict
        """
        if self.prm["direction"] == "left2right":
            # target_flat: [(seq_len - 1) * nbatch]
            target_flat = batch["word"]["index"].t()[1:].contiguous().view(-1).to(device)
        elif self.prm["direction"] == "right2left":
            # target_flat: [(seq_len - 1) * nbatch]
            target_flat = torch.flip(batch["word"]["index"].t(), [0])[1:].contiguous().view(-1).to(device)
        elif self.prm["direction"] == "both":
            # target_flat: [seq_len * nbatch]
            target_flat = batch["word"]["index"].t().contiguous().view(-1).to(device)
        else:
            assert False
        # print(target_flat.reshape(-1, batch["word"]["index"].shape[0]).t())
        return target_flat

    def init_hidden(self, batch):
        """
        Initialize the weights of the model

        Inputs
        ----------
            batch : Dict
        Return
        ----------
            Tuple
        """
        bsz = batch["word"]["index"].shape[0]
        # make a tensor whose type is similar to current model weights
        weight = next(self.parameters())
        # Joointly initialize the memory and hidden states for each layer
        if self.prm["rnn_type"] == "LSTM":
            if self.prm["direction"] == "both":
                return (
                    weight.new_zeros(self.prm["nlayers"] * 2, bsz, self.prm["tok_hid"]),
                    weight.new_zeros(self.prm["nlayers"] * 2, bsz, self.prm["tok_hid"]),
                )
            elif self.prm["direction"] == "left2right" or self.prm["direction"] == "right2left":
                return (
                    weight.new_zeros(self.prm["nlayers"], bsz, self.prm["tok_hid"]),
                    weight.new_zeros(self.prm["nlayers"], bsz, self.prm["tok_hid"]),
                )
            else:
                assert False
        # Separately initialize the memory and hidden states for each layer
        elif self.prm["rnn_type"] == "ResLSTM":
            if self.prm["direction"] == "both":
                return (
                    (
                        (weight.new_zeros(1, bsz, self.prm["tok_hid"]), weight.new_zeros(1, bsz, self.prm["tok_hid"]))
                        for i in range(self.prm["nlayers"])
                    ),
                    (
                        (weight.new_zeros(1, bsz, self.prm["tok_hid"]), weight.new_zeros(1, bsz, self.prm["tok_hid"]))
                        for i in range(self.prm["nlayers"])
                    ),
                )
            elif self.prm["direction"] == "left2right" or self.prm["direction"] == "right2left":
                return (
                    (weight.new_zeros(1, bsz, self.prm["tok_hid"]), weight.new_zeros(1, bsz, self.prm["tok_hid"]))
                    for i in range(self.prm["nlayers"])
                )
            else:
                assert False
        # Separately initialize the hidden states for each layer
        elif self.prm["rnn_type"] in ["ResGRU", "ResRNN", "ResRNN_TANH", "ResRNN_RELU"]:
            if self.prm["direction"] == "both":
                return (
                    (weight.new_zeros(1, bsz, self.prm["tok_hid"]) for i in range(self.prm["nlayers"])),
                    (weight.new_zeros(1, bsz, self.prm["tok_hid"]) for i in range(self.prm["nlayers"])),
                )
            elif self.prm["direction"] == "left2right" or self.prm["direction"] == "right2left":
                return (weight.new_zeros(1, bsz, self.prm["tok_hid"]) for i in range(self.prm["nlayers"]))
            else:
                assert False
        # Jointly initialize the hidden states for each layer
        else:
            if self.prm["direction"] == "both":
                return weight.new_zeros(self.prm["nlayers"] * 2, bsz, self.prm["tok_hid"])
            elif self.prm["direction"] == "left2right" or self.prm["direction"] == "right2left":
                return weight.new_zeros(self.prm["nlayers"], bsz, self.prm["tok_hid"])
            else:
                assert False
