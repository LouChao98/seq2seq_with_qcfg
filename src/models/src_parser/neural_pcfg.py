from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_struct
from torch_struct import SentCFG

from ..components.common import MultiResidualLayer
from .base import SrcParserBase


class NeuralPCFGSrcParser(SrcParserBase):
    def __init__(
        self,
        vocab=100,
        dim=256,
        pt_states=20,
        nt_states=20,
        num_layers=2,
        vocab_out=None,
    ):
        super(NeuralPCFGSrcParser, self).__init__()
        self.neg_huge = -1e5
        self.vocab = vocab
        self.nt_states = nt_states
        self.pt_states = pt_states
        self.all_states = nt_states + pt_states
        self.pt_emb = nn.Parameter(torch.randn(pt_states, dim))
        self.nt_emb = nn.Parameter(torch.randn(nt_states, dim))
        self.root_emb = nn.Parameter(torch.randn(1, dim))
        self.rule_mlp = nn.Sequential(nn.Linear(dim, self.all_states ** 2))
        self.root_mlp = MultiResidualLayer(
            in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=nt_states
        )
        if vocab_out is None:
            self.vocab_out = MultiResidualLayer(
                in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=vocab
            )
        else:
            self.vocab_out = vocab_out
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.pt_emb)
        nn.init.xavier_uniform_(self.nt_emb)
        nn.init.xavier_uniform_(self.root_emb)

    def get_rules(self, x: torch.Tensor):
        batch_size, n = x.size()
        root_emb = self.root_emb.expand(batch_size, -1)
        roots = self.root_mlp(root_emb)
        roots = F.log_softmax(roots, 1)
        nt_emb = self.nt_emb.unsqueeze(0).expand(batch_size, -1, -1)
        pt_emb = self.pt_emb.unsqueeze(0).expand(batch_size, -1, -1)
        nt = nt_emb.size(1)
        pt = pt_emb.size(1)
        rules = self.rule_mlp(nt_emb)
        rules = F.log_softmax(rules, 2)
        rules = rules.view(batch_size, nt, nt + pt, nt + pt)
        terms = F.log_softmax(self.vocab_out(pt_emb), 2)
        terms = terms.unsqueeze(1).expand(batch_size, n, pt, self.vocab)
        x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
        terms = torch.gather(terms, 3, x_expand).squeeze(3)
        return terms, rules, roots

    def forward(self, x, lengths):
        params = self.get_rules(x)
        dist = SentCFG(params, lengths)
        return dist

    def marginals(self, x, lengths):
        params = self.get_rules(x)
        dist = SentCFG(params, lengths)
        log_Z = dist.partition
        marginals = dist.marginals[-1]
        return -log_Z, marginals.sum(-1)

    @torch.enable_grad()
    def sample(self, x, lengths, dist: Optional[SentCFG] = None):
        if dist is None:
            dist = self(x, lengths)
        samples = dist._struct(torch_struct.SampledSemiring).marginals(
            dist.log_potentials, lengths=dist.lengths
        )
        log_Z = dist.partition
        logprobs = dist._struct().score(dist.log_potentials, samples) - log_Z
        return samples, logprobs

    def argmax(self, x, lengths, dist: Optional[SentCFG] = None):
        if dist is None:
            dist = self(x, lengths)

        spans_onehot = dist.argmax
        log_Z = dist.partition
        logprobs = dist._struct().score(dist.log_potentials, spans_onehot) - log_Z
        return spans_onehot, logprobs
