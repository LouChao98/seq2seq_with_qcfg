import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.common import MultiResidualLayer
from .base import SrcParserBase
from .neural_pcfg import NeuralPCFGSrcParser


class NeuralPCFGRefSrcParser(NeuralPCFGSrcParser):
    def __init__(
        self,
        vocab=100,
        dim=256,
        pt_states=20,
        nt_states=20,
        num_layers=2,
        vocab_out=None,
    ):
        SrcParserBase.__init__(self)
        self.neg_huge = -1e5
        self.vocab = vocab
        self.nt_states = nt_states
        self.pt_states = pt_states
        self.all_states = nt_states + pt_states
        self.dim = dim

        self.src_pt_emb = nn.Parameter(torch.randn(pt_states, dim))
        self.src_nt_emb = nn.Parameter(torch.randn(nt_states, dim))

        self.root_mlp_child = MultiResidualLayer(dim, dim, out_dim=1, num_layers=num_layers)

        self.rule_mlp_parent = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_mlp_left = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_mlp_right = MultiResidualLayer(dim, dim, num_layers=num_layers)

        if vocab_out is None:
            self.vocab_out = MultiResidualLayer(in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=vocab)
        else:
            self.vocab_out = vocab_out

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.src_pt_emb)
        nn.init.xavier_uniform_(self.src_nt_emb)

    def get_rules(self, x: torch.Tensor):
        batch_size, n = x.size()
        nt_emb = self.src_nt_emb.unsqueeze(0).expand(batch_size, -1, -1)
        pt_emb = self.src_pt_emb.unsqueeze(0).expand(batch_size, -1, -1)
        nt = self.nt_states
        pt = self.pt_states
        all_states = nt + pt

        roots = self.root_mlp_child(nt_emb)
        roots = roots.view(batch_size, self.nt_states)
        roots = F.log_softmax(roots, 1)

        all_state_emb = torch.cat([nt_emb, pt_emb], 1)
        rule_emb_parent = self.rule_mlp_parent(nt_emb)  # b x nt_all x dm
        rule_emb_left = self.rule_mlp_left(all_state_emb)
        rule_emb_right = self.rule_mlp_right(all_state_emb)

        rule_emb_child = rule_emb_left[:, :, None, :] + rule_emb_right[:, None, :, :]
        rule_emb_child = rule_emb_child.view(batch_size, (all_states) ** 2, self.dim)
        rules = torch.matmul(rule_emb_parent, rule_emb_child.transpose(1, 2))
        rules = F.log_softmax(rules, 2)
        rules = rules.view(batch_size, nt, nt + pt, nt + pt)

        terms = F.log_softmax(self.vocab_out(pt_emb), 2)
        terms = terms.unsqueeze(1).expand(batch_size, n, pt, self.vocab)
        x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
        terms = torch.gather(terms, 3, x_expand).squeeze(3)
        return terms, rules, roots
