from typing import Optional

import torch
import torch.nn as nn

from ..components.common import MultiResidualLayer
from ..struct.decomp1_fast import Decomp1Fast
from .base import SrcParserBase


class NeuralFastTNPCFGSrcParser(SrcParserBase):
    # This cannot be used in reinforce estimator because sampling is not implemented.

    def __init__(
        self,
        vocab=100,
        dim=256,
        pt_states=20,
        nt_states=20,
        num_layers=2,
        cpd_rank=32,
        vocab_out=None,
    ):
        super(NeuralFastTNPCFGSrcParser, self).__init__()
        self.neg_huge = -1e5
        self.vocab = vocab
        self.nt_states = nt_states
        self.pt_states = pt_states
        self.all_states = nt_states + pt_states

        ## root
        self.root_emb = nn.Parameter(torch.randn(1, dim))
        self.root_mlp = MultiResidualLayer(in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=nt_states)

        # terms
        if vocab_out is None:
            self.term_mlp = MultiResidualLayer(in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=vocab)
        else:
            self.term_mlp = vocab_out

        self.rule_state_emb = nn.Parameter(torch.randn(self.all_states, dim))

        self.parent_mlp = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.left_mlp = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.right_mlp = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)

        _w, _b = self.parent_mlp.out_linear.weight, self.parent_mlp.out_linear.bias
        self.left_mlp.out_linear.weight = _w
        self.left_mlp.out_linear.bias = _b
        self.right_mlp.out_linear.weight = _w
        self.right_mlp.out_linear.bias = _b

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rule_state_emb)
        nn.init.xavier_uniform_(self.root_emb)

    def get_rules(self, x: torch.Tensor):
        b, n = x.shape[:2]

        roots = self.root_mlp(self.root_emb)
        roots = roots.log_softmax(-1)
        roots = roots.expand(b, roots.shape[-1])

        term_emb = self.rule_state_emb[self.nt_states :]
        term_prob = self.term_mlp(term_emb).log_softmax(-1)
        term = term_prob[torch.arange(self.pt_states)[None, None], x[:, :, None]]

        rule_state_emb = self.rule_state_emb
        nonterm_emb = rule_state_emb[: self.nt_states]
        head = self.parent_mlp(nonterm_emb)
        left = self.left_mlp(rule_state_emb)
        right = self.right_mlp(rule_state_emb)
        head = head.softmax(-1)
        left = left.T.softmax(-1)
        right = right.T.softmax(-1)
        head = head.unsqueeze(0).expand(b, *head.shape)
        left = left.unsqueeze(0).expand(b, *left.shape)
        right = right.unsqueeze(0).expand(b, *right.shape)
        # breakpoint()
        return {"term": term, "root": roots, "left": left, "right": right, "head": head}

    def forward(self, x, lengths, extra_scores=None):
        params = self.get_rules(x)

        if extra_scores is not None:
            if (constraint := extra_scores.get("observed_mask")) is not None:
                params["constraint"] = [
                    (
                        torch.full(list(mask.shape) + [self.nt_states], -1e9, device=mask.device),
                        mask.unsqueeze(-1).expand(-1, -1, self.nt_states),
                    )
                    for mask in constraint
                ]
        dist = Decomp1Fast(
            params,
            lengths,
            self.nt_states,
            1,
            self.pt_states,
            1,
            len(lengths),
        )
        return dist

    def marginals(self, x, lengths, dist: Optional[Decomp1Fast] = None, extra_scores=None):
        raise NotImplementedError

    def sample(self, x, lengths, dist: Optional[Decomp1Fast] = None, extra_scores=None):
        raise NotImplementedError

    def gumbel_sample(self, x, lengths, temperature, dist: Optional[Decomp1Fast] = None, extra_scores=None):
        raise NotImplementedError

    @torch.enable_grad()
    def argmax(self, x, lengths, dist: Optional[Decomp1Fast] = None, extra_scores=None):
        if dist is None:
            dist = self(x, lengths, extra_scores)

        return dist.decoded

    @torch.enable_grad()
    def entropy(self, x, lengths, dist: Optional[Decomp1Fast] = None, extra_scores=None):
        if dist is None:
            dist = self(x, lengths, extra_scores)

        return dist.entropy
