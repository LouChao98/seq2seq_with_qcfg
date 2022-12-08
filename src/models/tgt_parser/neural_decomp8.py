import logging
from copy import copy, deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.common import MultiResidualLayer
from ..struct.decomp8 import Decomp8, Decomp8Sampler
from ..struct.no_decomp import NoDecomp
from .base import NO_COPY_SPAN, TgtParserBase, TgtParserPrediction

log = logging.getLogger(__file__)


def normalize(t: torch.Tensor):
    shape = t.shape
    return t.flatten(-2).softmax(-1).view(shape)


class NeuralDecomp8TgtParser(TgtParserBase):
    def __init__(self, vocab=100, dim=256, num_layers=3, src_dim=256, **kwargs):
        super().__init__(**kwargs)

        self.vocab = vocab
        self.dim = dim
        self.src_dim = src_dim
        self.num_layers = num_layers

        self.src_nt_emb = nn.Parameter(torch.randn(self.nt_states, dim))
        # self.src_nt_emb_for_root = nn.Parameter(torch.randn(self.nt_states, dim))
        self.src_nt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.src_pt_emb = nn.Parameter(torch.randn(self.pt_states, dim))
        self.src_pt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)

        self.root_mlp_child = MultiResidualLayer(dim, dim, out_dim=1, num_layers=num_layers)
        self.vocab_out = MultiResidualLayer(dim, dim, out_dim=vocab, num_layers=num_layers)
        # self.vocab_out2 = MultiResidualLayer(dim, dim, out_dim=vocab, num_layers=num_layers)
        self.vocab_out2 = deepcopy(self.vocab_out)

        self.rule_mlp_parent = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_mlp_left = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_mlp_right = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_weight = nn.Parameter(torch.randn(2 * dim, dim))

        self.rule_align_head_mlp = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_align_left_mlp = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_align_right_mlp = MultiResidualLayer(dim, dim, num_layers=num_layers)

        self.src_nt_shared_emb = nn.Parameter(torch.randn(1, 1, dim))
        self.src_pt_shared_emb = nn.Parameter(torch.randn(1, 1, dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.src_nt_emb.data)
        # nn.init.xavier_uniform_(self.src_nt_emb_for_root.data)
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
        all_states = self.nt_states + self.pt_states

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
        rule_emb_parent = self.rule_mlp_parent(nt_state_emb)  # b x nt_all x dm
        rule_emb_left = self.rule_mlp_left(all_state_emb)
        rule_emb_right = self.rule_mlp_right(all_state_emb)

        rule_emb_child = torch.empty(
            batch_size,
            rule_emb_left.shape[1],
            rule_emb_right.shape[1],
            2 * rule_emb_left.shape[2],
            device=rule_emb_left.device,
        )
        rule_emb_child[..., : rule_emb_left.shape[2]] = rule_emb_left.unsqueeze(2)
        rule_emb_child[..., rule_emb_left.shape[2] :] = rule_emb_right.unsqueeze(1)

        rules = torch.einsum("blrx,xy,bpy->bplr", rule_emb_child, self.rule_weight, rule_emb_parent)

        # rule_emb_child = rule_emb_left[:, :, None, :] + rule_emb_right[:, None, :, :]
        # rule_emb_child = rule_emb_child.view(batch_size, (all_states) ** 2, self.dim)
        # rules = torch.matmul(rule_emb_parent, rule_emb_child.transpose(1, 2))

        rules = rules.view(batch_size, self.nt_states, -1)
        rules = self.score_normalize(rules, 2).log_softmax(-1)
        rules = rules.view(batch_size, self.nt_states, all_states, all_states)

        nt_i = self.rule_align_head_mlp(nt_state_emb.unsqueeze(2) + nt_node_emb.unsqueeze(1))
        pt_i = self.rule_align_head_mlp(pt_state_emb.unsqueeze(2) + nt_node_emb.unsqueeze(1))
        j_nt = self.rule_align_left_mlp(nt_node_emb + self.src_nt_shared_emb)
        k_nt = self.rule_align_right_mlp(nt_node_emb + self.src_nt_shared_emb)
        j_pt = self.rule_align_left_mlp(pt_node_emb + self.src_pt_shared_emb)
        k_pt = self.rule_align_right_mlp(pt_node_emb + self.src_pt_shared_emb)

        rule_align_left_nt = self.score_normalize(torch.einsum("xbih,xjh->xbij", nt_i, j_nt), 3)
        rule_align_left_pt = self.score_normalize(torch.einsum("xbih,xjh->xbij", pt_i, j_pt), 3)
        rule_align_right_nt = self.score_normalize(torch.einsum("xbih,xjh->xbij", nt_i, k_nt), 3)
        rule_align_right_pt = self.score_normalize(torch.einsum("xbih,xjh->xbij", pt_i, k_pt), 3)

        # fmt: off
        nt_mask = torch.arange(nt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=device).unsqueeze(1)
        # fmt: on
        _nt_mask = (~nt_mask)[:, None, None].expand(-1, self.nt_states, nt_num_nodes, -1)
        _pt_mask = (~pt_mask)[:, None, None].expand(-1, self.pt_states, nt_num_nodes, -1)
        rule_align_left_nt[_nt_mask] = self.neg_huge
        rule_align_left_pt[_pt_mask] = self.neg_huge
        rule_align_right_nt[_nt_mask] = self.neg_huge
        rule_align_right_pt[_pt_mask] = self.neg_huge

        if self.rule_hard_constraint is not None:
            raise NotImplementedError

        rule_align_left_nt = rule_align_left_nt.log_softmax(-1)
        rule_align_left_pt = rule_align_left_pt.log_softmax(-1)
        rule_align_right_nt = rule_align_right_nt.log_softmax(-1)
        rule_align_right_pt = rule_align_right_pt.log_softmax(-1)

        _mask = (~nt_mask)[:, None, :, None].expand(-1, all_states, -1, max(nt_num_nodes, pt_num_nodes))
        rule_align_left = torch.full_like(_mask, self.neg_huge, device=device, dtype=torch.float32)
        rule_align_left[:, : self.nt_states, :, :nt_num_nodes] = rule_align_left_nt
        rule_align_left[:, self.nt_states :, :, :pt_num_nodes] = rule_align_left_pt
        rule_align_left[_mask] = self.neg_huge

        rule_align_right = torch.full_like(_mask, self.neg_huge, device=device, dtype=torch.float32)
        rule_align_right[:, : self.nt_states, :, :nt_num_nodes] = rule_align_right_nt
        rule_align_right[:, self.nt_states :, :, :pt_num_nodes] = rule_align_right_pt
        rule_align_right[_mask] = self.neg_huge

        # A->a
        terms = F.log_softmax(self.score_normalize(self.vocab_out(pt_emb), 2), 2)
        terms_tgt_only = F.log_softmax(self.score_normalize(self.vocab_out2(pt_state_emb), 2), 2)

        params = {
            "term": terms,
            "root": roots,
            "align_left": rule_align_left,
            "align_right": rule_align_right,
            "rule": rules,
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
        pred.dist = Decomp8(pred.posterior_params, pred.lengths, **pred.common())
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
            "rule": pred.posterior_params["rule"],
            "root": pred.posterior_params["_root_tgt_only"],
        }
        pred.posterior_params = {
            "term": term,
            "rule": pred.posterior_params["rule"],
            "root": pred.posterior_params["_root_tgt_only"],
        }
        pred.nt_num_nodes = 1
        pred.pt_num_nodes = 1
        pred.dist = NoDecomp(
            pred.posterior_params,
            pred.lengths,
            **pred.common(),
        )
        pred.pt_nodes = [[(0, 1, NO_COPY_SPAN)] for _ in range(pred.batch_size)]
        pred.nt_nodes = [[(0, l, NO_COPY_SPAN)] for l in pred.lengths]
        return pred

    def prepare_sampler(self, pred: TgtParserPrediction, src, src_ids, inplace=True) -> TgtParserPrediction:
        pred = super().prepare_sampler(pred, src, src_ids, inplace)
        pred.sampler = Decomp8Sampler(pred.params, **pred.common(), **self.sampler_common())
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
