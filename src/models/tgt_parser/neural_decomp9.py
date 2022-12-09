import logging
from copy import copy, deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.common import MultiResidualLayer
from ..struct.decomp1 import Decomp1
from ..struct.decomp9 import Decomp9, Decomp9Sampler
from .base import NO_COPY_SPAN, TgtParserBase, TgtParserPrediction

log = logging.getLogger(__file__)


def normalize(t: torch.Tensor):
    shape = t.shape
    return t.flatten(-2).softmax(-1).view(shape)


class NeuralDecomp9TgtParser(TgtParserBase):
    def __init__(self, vocab=100, dim=256, num_layers=3, src_dim=256, cpd_rank=64, tie_r=False, **kwargs):
        super().__init__(**kwargs)

        self.vocab = vocab
        self.dim = dim
        self.src_dim = src_dim
        self.num_layers = num_layers
        self.cpd_rank = cpd_rank
        self.tie_r = tie_r

        self.src_nt_emb = nn.Parameter(torch.randn(self.nt_states, dim))
        self.src_nt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.src_pt_emb = nn.Parameter(torch.randn(self.pt_states, dim))
        self.src_pt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)

        self.root_mlp_child = MultiResidualLayer(dim, dim, out_dim=1, num_layers=num_layers)
        self.vocab_out = MultiResidualLayer(dim, dim, out_dim=vocab, num_layers=num_layers)
        self.vocab_out2 = deepcopy(self.vocab_out)

        self.rule_mlp_parent = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rule_mlp_left = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rule_mlp_right = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)

        self.rule_align_head_mlp = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_align_left_mlp = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_align_right_mlp = MultiResidualLayer(dim, dim, num_layers=num_layers)

        self.src_nt_shared_emb = nn.Parameter(torch.randn(1, 1, dim))
        self.src_pt_shared_emb = nn.Parameter(torch.randn(1, 1, dim))

        self.r_emb = nn.Embedding(cpd_rank, dim)

        if tie_r:
            _w, _b = self.rule_mlp_parent.out_linear.weight, self.rule_mlp_parent.out_linear.bias
            self.rule_mlp_left.out_linear.weight = _w
            self.rule_mlp_left.out_linear.bias = _b
            self.rule_mlp_right.out_linear.weight = _w
            self.rule_mlp_right.out_linear.bias = _b
            self.r_emb.weight = _w

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.src_nt_emb.data)
        nn.init.xavier_uniform_(self.src_pt_emb.data)
        nn.init.xavier_uniform_(self.src_nt_shared_emb.data)
        nn.init.xavier_uniform_(self.src_pt_shared_emb.data)

    def forward(self, node_features, spans, **kwargs):
        batch_size = len(spans)
        device = node_features[0].device

        (
            nt_spans,
            nt_num_nodes_list,
            nt_num_nodes,
            nt_node_features,
            pt_spans,
            pt_num_nodes_list,
            pt_num_nodes,
            pt_node_features,
        ) = self.build_src_features(spans, node_features)

        nt = self.nt_states * nt_num_nodes
        pt = self.pt_states * pt_num_nodes

        # e = u + h
        nt_node_emb = self.src_nt_node_mlp(nt_node_features)
        nt_state_emb = self.src_nt_emb.expand(batch_size, self.nt_states, self.dim)

        pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        pt_state_emb = self.src_pt_emb.expand(batch_size, self.pt_states, self.dim)
        pt_emb = pt_state_emb.unsqueeze(2) + pt_node_emb.unsqueeze(1)
        pt_emb = pt_emb.view(batch_size, pt, -1)

        # S->A
        # nt_state_emb_for_root = self.src_nt_emb_for_root.expand(batch_size, self.nt_states, self.dim)
        # roots = self.root_mlp_child(nt_state_emb_for_root.unsqueeze(2).repeat(1, 1, nt_num_nodes, 1))
        roots = self.root_mlp_child(nt_state_emb.unsqueeze(2).repeat(1, 1, nt_num_nodes, 1))
        roots = roots.view(batch_size, self.nt_states, nt_num_nodes)
        mask = torch.arange(nt_num_nodes, device=device).view(1, 1, -1).expand(batch_size, 1, -1)
        allowed = (torch.tensor(nt_num_nodes_list, device=device) - 1).view(-1, 1, 1)
        roots = torch.where(mask == allowed, roots, roots.new_tensor(self.neg_huge))
        roots = roots.view(batch_size, -1)
        roots = F.log_softmax(roots, 1)

        # A->BC
        all_state_emb = torch.cat([nt_state_emb, pt_state_emb], 1)
        rule_head = self.rule_mlp_parent(nt_state_emb).log_softmax(-1)  # b x nt_all x dm
        rule_left = self.rule_mlp_left(all_state_emb).transpose(1, 2).log_softmax(-1)
        rule_right = self.rule_mlp_right(all_state_emb).transpose(1, 2).log_softmax(-1)

        r_i = self.rule_align_head_mlp(self.r_emb.weight[None, :, None] + nt_node_emb.unsqueeze(1))
        nt_node_emb_for_align = nt_node_emb + self.src_nt_shared_emb
        pt_node_emb_for_align = pt_node_emb + self.src_pt_shared_emb
        node_emb_for_align = torch.cat([nt_node_emb_for_align, pt_node_emb_for_align], dim=1)

        j = self.rule_align_left_mlp(node_emb_for_align)
        k = self.rule_align_right_mlp(node_emb_for_align)

        rule_align_left = self.score_normalize(torch.einsum("xbih,xjh->xbij", r_i, j), 3)
        rule_align_right = self.score_normalize(torch.einsum("xbih,xjh->xbij", r_i, k), 3)

        # fmt: off
        nt_mask = torch.arange(nt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=device).unsqueeze(1)
        # fmt: on
        _mask = torch.cat([nt_mask, pt_mask], dim=1)[:, None, None].expand(-1, self.cpd_rank, nt_num_nodes, -1)

        rule_align_left[~_mask] = self.neg_huge
        rule_align_right[~_mask] = self.neg_huge

        if self.rule_hard_constraint is not None:
            raise NotImplementedError

        rule_align_left = rule_align_left.log_softmax(-1).clone()
        rule_align_right = rule_align_right.log_softmax(-1).clone()

        _mask = (~nt_mask)[:, None, :, None].expand(-1, self.cpd_rank, -1, nt_num_nodes + pt_num_nodes)
        rule_align_left[_mask] = self.neg_huge
        rule_align_right[_mask] = self.neg_huge

        # A->a
        terms = F.log_softmax(self.score_normalize(self.vocab_out(pt_emb), 2), 2)
        terms_tgt_only = F.log_softmax(self.score_normalize(self.vocab_out2(pt_state_emb), 2), 2)

        params = {
            "term": terms,
            "root": roots,
            "align_left": rule_align_left,
            "align_right": rule_align_right,
            "left": rule_left,
            "right": rule_right,
            "head": rule_head,
            "_term_tgt_only": terms_tgt_only,
            "_root_tgt_only": roots.view(batch_size, self.nt_states, -1)[
                torch.arange(batch_size), :, torch.tensor(nt_num_nodes_list) - 1
            ],
        }

        # from ..struct.decomp8 import convert_decomp8_to_pcfg
        # pref = convert_decomp8_to_pcfg(params, self.nt_states)
        # r = pref['rule'].flatten(2).exp().sum()

        pred = TgtParserPrediction(
            batch_size=batch_size,
            nt=nt,
            nt_states=self.nt_states,
            nt_nodes=nt_spans,
            nt_num_nodes=nt_num_nodes,
            pt=pt,
            pt_states=self.pt_states,
            pt_nodes=pt_spans,
            pt_num_nodes=pt_num_nodes,
            params=params,
            device=device,
        )
        return pred

    def observe_x(self, pred: TgtParserPrediction, x, lengths, inplace=True, **kwargs) -> TgtParserPrediction:
        pred = super().observe_x(pred, x, lengths, inplace, **kwargs)
        pred.dist = Decomp9(pred.posterior_params, pred.lengths, **pred.common())
        return pred

    def get_mini(self, pred: TgtParserPrediction, x):
        pred = copy(pred)
        term = pred.posterior_params["_term_tgt_only"]
        batch_size, n = x.shape[:2]
        term = term.unsqueeze(1).expand(batch_size, n, pred.pt_states, term.size(2))
        tgt_expand = x.unsqueeze(2).expand(batch_size, n, pred.pt_states).unsqueeze(3)
        term = torch.gather(term, 3, tgt_expand).squeeze(3)
        pred.params = {
            "term": pred.posterior_params["_term_tgt_only"],
            "head": pred.posterior_params["head"],
            "left": pred.posterior_params["left"],
            "right": pred.posterior_params["right"],
            "root": pred.posterior_params["_root_tgt_only"],
        }
        pred.posterior_params = {
            "term": term,
            "head": pred.posterior_params["head"],
            "left": pred.posterior_params["left"],
            "right": pred.posterior_params["right"],
            "root": pred.posterior_params["_root_tgt_only"],
        }
        pred.nt_num_nodes = 1
        pred.pt_num_nodes = 1
        pred.dist = Decomp1(
            pred.posterior_params,
            pred.lengths,
            **pred.common(),
        )
        pred.pt_nodes = [[(0, 1, NO_COPY_SPAN)] for _ in range(pred.batch_size)]
        pred.nt_nodes = [[(0, l, NO_COPY_SPAN)] for l in pred.lengths]
        return pred

    def prepare_sampler(self, pred: TgtParserPrediction, src, src_ids, inplace=True) -> TgtParserPrediction:
        pred = super().prepare_sampler(pred, src, src_ids, inplace)
        pred.sampler = Decomp9Sampler(pred.params, **pred.common(), **self.sampler_common())
        return pred

    def post_process_nt_constraint(self, constraint, device):
        constraint_ = []
        for value, mask in constraint:
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
                mask = torch.from_numpy(mask)
            else:
                value = value.flatten(2)
                mask = mask.flatten(2)
            constraint_.append((value.to(device), mask.to(device)))
        return constraint_
