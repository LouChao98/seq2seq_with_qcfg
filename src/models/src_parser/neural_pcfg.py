from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.common import MultiResidualLayer
from ..struct.no_decomp import NoDecomp
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
        self.root_mlp = MultiResidualLayer(in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=nt_states)
        self.rule_mlp = nn.Sequential(nn.Linear(dim, self.all_states**2))
        if vocab_out is None:
            self.vocab_out = MultiResidualLayer(in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=vocab)
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
        nt = self.nt_states
        pt = self.pt_states
        rules = self.rule_mlp(nt_emb)
        rules = F.log_softmax(rules, 2)
        rules = rules.view(batch_size, nt, nt + pt, nt + pt)
        terms = F.log_softmax(self.vocab_out(pt_emb), 2)
        terms = terms.unsqueeze(1).expand(batch_size, n, pt, self.vocab)
        x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
        terms = torch.gather(terms, 3, x_expand).squeeze(3)
        return terms, rules, roots

    def forward(self, x, lengths, extra_scores=None):
        params = self.get_rules(x)
        params = {"term": params[0], "rule": params[1], "root": params[2]}
        if extra_scores is not None:
            if (constraint := extra_scores.get("observed_mask")) is not None:
                params["constraint"] = [
                    (
                        torch.full(list(mask.shape) + [self.nt_states], -1e9, device=mask.device),
                        mask.unsqueeze(-1).expand(-1, -1, self.nt_states),
                    )
                    for mask in constraint
                ]
        dist = NoDecomp(
            params,
            lengths,
            self.nt_states,
            1,
            self.pt_states,
            1,
            len(lengths),
        )
        return dist

    def marginals(self, x, lengths, dist: Optional[NoDecomp] = None, extra_scores=None):
        raise NotImplementedError

    @torch.enable_grad()
    def sample(self, x, lengths, dist: Optional[NoDecomp] = None, extra_scores=None):
        if dist is None:
            dist = self(x, lengths, extra_scores)

        output = dist.sample_one(need_span=True, need_event=True)
        logprobs = dist.score(output["event"])
        return output, logprobs - dist.partition

    @torch.enable_grad()
    def gumbel_sample(self, x, lengths, temperature, dist: Optional[NoDecomp] = None, extra_scores=None):
        if dist is None:
            dist = self(x, lengths, extra_scores)

        mtrace = dist.gumbel_sample_one(temperature)
        return mtrace

    @torch.enable_grad()
    def argmax(self, x, lengths, dist: Optional[NoDecomp] = None, extra_scores=None):
        if dist is None:
            dist = self(x, lengths, extra_scores)

        return dist.decoded

    @torch.enable_grad()
    def entropy(self, x, lengths, dist: Optional[NoDecomp] = None, extra_scores=None):
        if dist is None:
            dist = self(x, lengths, extra_scores)

        return dist.entropy
