import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from torch_struct import SentCFG

from ..components.common import MultiResidualLayer
from ..struct.decomp1 import Decomp1, Decomp1Sampler
from ..struct.decomp1_fast import Decomp1Fast, Decomp1FastSampler, convert_decomp1_to_pcfg
from .base import TgtParserBase, TgtParserPrediction

log = logging.getLogger(__file__)


class NeuralDecomp1TgtParser(TgtParserBase):
    def __init__(
        self,
        cpd_rank=32,
        vocab=100,
        dim=256,
        num_layers=3,
        src_dim=256,
        tie_r=False,
        use_fast=False,
        vector_quantize=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert self.rule_hard_constraint is None, "Do not support any constraint."
        # assert self.rule_soft_constraint is None, "Do not support any constraint."  # need folded, use mask

        self.vocab = vocab
        self.dim = dim
        self.src_dim = src_dim
        self.num_layers = num_layers
        self.cpd_rank = cpd_rank
        self.tie_r = tie_r
        self.use_fast = use_fast

        self.src_nt_emb = nn.Parameter(torch.randn(self.nt_states, dim))
        self.src_nt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.src_pt_emb = nn.Parameter(torch.randn(self.pt_states, dim))
        self.src_pt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)

        self.root_mlp_child = nn.Linear(dim, 1, bias=False)
        self.vocab_out = MultiResidualLayer(dim, dim, out_dim=vocab, num_layers=num_layers)

        self.rule_mlp_parent = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rule_mlp_left = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)
        self.rule_mlp_right = MultiResidualLayer(dim, dim, out_dim=cpd_rank, num_layers=num_layers)

        if tie_r:
            _w, _b = self.rule_mlp_parent.out_linear.weight, self.rule_mlp_parent.out_linear.bias
            self.rule_mlp_left.out_linear.weight = _w
            self.rule_mlp_left.out_linear.bias = _b
            self.rule_mlp_right.out_linear.weight = _w
            self.rule_mlp_right.out_linear.bias = _b

        self.vector_quantizer = instantiate(vector_quantize, dim=src_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.src_nt_emb.data)
        nn.init.xavier_uniform_(self.src_pt_emb.data)

    def forward(self, node_features, spans, weight=None, **kwargs):
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

        commit_loss = 0
        if self.vector_quantizer is not None:
            nt_node_features, _indices, _loss = self.vector_quantizer(nt_node_features)
            commit_loss += _loss

        nt = self.nt_states * nt_num_nodes
        pt = self.pt_states * pt_num_nodes

        # e = u + h
        nt_node_emb = self.src_nt_node_mlp(nt_node_features)
        nt_state_emb = self.src_nt_emb.expand(batch_size, self.nt_states, self.dim)
        nt_emb = nt_state_emb.unsqueeze(2) + nt_node_emb.unsqueeze(1)
        nt_emb = nt_emb.view(batch_size, nt, -1)

        pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        pt_state_emb = self.src_pt_emb.expand(batch_size, self.pt_states, self.dim)
        pt_emb = pt_state_emb.unsqueeze(2) + pt_node_emb.unsqueeze(1)
        pt_emb = pt_emb.view(batch_size, pt, -1)

        # S->A
        roots = self.root_mlp_child(nt_emb)
        roots = roots.view(batch_size, self.nt_states, nt_num_nodes)
        mask = torch.arange(nt_num_nodes, device=device).view(1, 1, -1).expand(batch_size, 1, -1)
        allowed = (torch.tensor(nt_num_nodes_list, device=device) - 1).view(-1, 1, 1)
        roots = torch.where(mask == allowed, roots, roots.new_tensor(self.neg_huge))
        roots = roots.view(batch_size, -1)
        roots = F.log_softmax(roots, 1)

        # A->BC
        all_emb = torch.cat([nt_emb, pt_emb], 1)
        rule_head = self.rule_mlp_parent(nt_emb)
        rule_left = self.rule_mlp_left(all_emb)
        rule_right = self.rule_mlp_right(all_emb)

        if self.score_regularziation > 0:
            reg = torch.norm(rule_head) + torch.norm(rule_left) + torch.norm(rule_right)
            reg = reg * self.score_regularziation
            rule_left = rule_left.clone()
            rule_right = rule_right.clone()

        # fmt: off
        device = all_emb.device
        nt_mask = torch.arange(nt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=device).unsqueeze(1)
        nt_mask = nt_mask.unsqueeze(1).expand(-1, self.nt_states, -1).reshape(batch_size, -1)
        pt_mask = pt_mask.unsqueeze(1).expand(-1, self.pt_states, -1).reshape(batch_size, -1)
        # fmt: on
        mask = torch.cat([nt_mask, pt_mask], dim=1)
        rule_left[~mask] = self.neg_huge
        rule_right[~mask] = self.neg_huge

        if self.use_fast:
            rule_head = rule_head.softmax(-1)
            rule_left = rule_left.transpose(1, 2).softmax(-1)
            rule_right = rule_right.transpose(1, 2).softmax(-1)
        else:
            rule_head = rule_head.log_softmax(-1)
            rule_left = rule_left.transpose(1, 2).log_softmax(-1)
            rule_right = rule_right.transpose(1, 2).log_softmax(-1)

        # A->a
        terms = F.log_softmax(self.vocab_out(pt_emb), 2)

        params = {
            "term": terms,
            "root": roots,
            "left": rule_left,
            "right": rule_right,
            "head": rule_head,
        }
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

        if self.vector_quantizer is not None:
            pred.vq_commit_loss = commit_loss

        if self.score_regularziation > 0:
            pred.score_reg_loss = reg

        return pred

    def observe_x(self, pred: TgtParserPrediction, x, lengths, inplace=True, **kwargs) -> TgtParserPrediction:
        pred = super().observe_x(pred, x, lengths, inplace, **kwargs)
        pred.dist = (Decomp1Fast if self.use_fast else Decomp1)(pred.posterior_params, pred.lengths, **pred.common())
        return pred

    def prepare_sampler(self, pred: TgtParserPrediction, src, src_ids, inplace=True) -> TgtParserPrediction:
        pred = super().prepare_sampler(pred, src, src_ids, inplace)
        pred.sampler = (Decomp1FastSampler if self.use_fast else Decomp1Sampler)(
            pred.params, **pred.common(), **self.sampler_common()
        )
        return pred

    # def get_soft_constraint_loss(self, pred: TgtParserPrediction):
    #     assert self.use_fast
    #     mask = self.rule_soft_constraint.get_mask_from_pred(pred)
    #     head = pred.params["head"].view(pred.batch_size, pred.nt_states, pred.nt_num_nodes, -1).sum(1)
    #     left_nt, left_pt = pred.params["left"].split([pred.nt, pred.pt], 2)
    #     right_nt, right_pt = pred.params["right"].split([pred.nt, pred.pt], 2)
    #     left_nt = left_nt.view(pred.batch_size, -1, pred.nt_states, pred.nt_num_nodes).sum(2)
    #     left_pt = left_pt.view(pred.batch_size, -1, pred.pt_states, pred.pt_num_nodes).sum(2)
    #     right_nt = right_nt.view(pred.batch_size, -1, pred.nt_states, pred.nt_num_nodes).sum(2)
    #     right_pt = right_pt.view(pred.batch_size, -1, pred.pt_states, pred.pt_num_nodes).sum(2)
    #     x1 = torch.einsum(
    #         "qar,qrb,qrc->qrabc", head, torch.cat([left_nt, left_pt], 2), torch.cat([right_nt, right_pt], 2)
    #     )

    #     m_head = (
    #         pred.dist.rule_marginal_with_grad["head"]
    #         .view(pred.batch_size, pred.nt_states, pred.nt_num_nodes, -1)
    #         .sum(1)
    #     )
    #     breakpoint()
    #     x1_inv = torch.linalg.inv(x1 + torch.randn_like(x1) * 0.001)
    #     score = torch.einsum("qrabc,qar->qabc", x1_inv, m_head)
    #     breakpoint()

    #     nt_mask = (
    #         torch.arange(pred.nt_num_nodes).unsqueeze(0)
    #         < torch.tensor([len([i for i in item if i[2] != -999]) for item in pred.nt_nodes]).unsqueeze(1)
    #     ).to(pred.device)
    #     pt_mask = (
    #         torch.arange(pred.pt_num_nodes).unsqueeze(0)
    #         < torch.tensor([len([i for i in item if i[2] != -999]) for item in pred.pt_nodes]).unsqueeze(1)
    #     ).to(pred.device)
    #     nt_pt_mask = torch.cat([nt_mask, pt_mask], 1)
    #     valid_mask = nt_mask[:, :, None, None] & nt_pt_mask[:, None, :, None] & nt_pt_mask[:, None, None, :]
    #     score[~valid_mask] = 0.0

    #     # params_ref = convert_decomp1_to_pcfg(pred.dist.params, pred.nt_states)
    #     # pcfg_ref = SentCFG((params_ref["term"], params_ref["rule"], params_ref["root"]), pred.lengths)
    #     # SRC_NT = pred.nt_num_nodes
    #     # SRC_PT = pred.pt_num_nodes
    #     # NT = pred.nt
    #     # B = pred.batch_size
    #     # TGT_NT = pred.nt_states
    #     # TGT_PT = pred.pt_states
    #     # trace = torch.empty(B, SRC_NT, SRC_NT + SRC_PT, SRC_NT + SRC_PT)
    #     # t = pcfg_ref.marginals[1]
    #     # trace[:, :, :SRC_NT, :SRC_NT] = t[:, :, :NT, :NT].reshape(B, TGT_NT, SRC_NT, TGT_NT, SRC_NT, TGT_NT, SRC_NT).sum((1,3,5))
    #     # trace[:, :, :SRC_NT, SRC_NT:] = t[:, :, :NT, NT:].reshape(B, TGT_NT, SRC_NT, TGT_NT, SRC_NT, TGT_PT, SRC_PT).sum((1,3,5))
    #     # trace[:, :, SRC_NT:, :SRC_NT] = t[:, :, NT:, :NT].reshape(B, TGT_NT, SRC_NT, TGT_PT, SRC_PT, TGT_NT, SRC_NT).sum((1,3,5))
    #     # trace[:, :, SRC_NT:, SRC_NT:] = t[:, :, NT:, NT:].reshape(B, TGT_NT, SRC_NT, TGT_PT, SRC_PT, TGT_PT, SRC_PT).sum((1,3,5))
    #     # obj = trace[~mask].sum()
    #     # print(score.max(), merge.max())
    #     # print(obj, trace[mask].sum())

    #     return -(score * (~mask).float()).clamp(max=100).flatten(1).sum(1) * 0.01

    # def get_soft_constraint_loss(self, pred: TgtParserPrediction):
    #     assert self.use_fast
    #     mask = self.rule_soft_constraint.get_mask_from_pred(pred)
    #     head = pred.params["head"].view(pred.batch_size, pred.nt_states, pred.nt_num_nodes, -1).sum(1)
    #     left_nt, left_pt = pred.params["left"].split([pred.nt, pred.pt], 2)
    #     right_nt, right_pt = pred.params["right"].split([pred.nt, pred.pt], 2)
    #     left_nt = left_nt.view(pred.batch_size, -1, pred.nt_states, pred.nt_num_nodes).sum(2)
    #     left_pt = left_pt.view(pred.batch_size, -1, pred.pt_states, pred.pt_num_nodes).sum(2)
    #     right_nt = right_nt.view(pred.batch_size, -1, pred.nt_states, pred.nt_num_nodes).sum(2)
    #     right_pt = right_pt.view(pred.batch_size, -1, pred.pt_states, pred.pt_num_nodes).sum(2)
    #     x1 = torch.einsum(
    #         "qar,qrb,qrc->qrabc", head, torch.cat([left_nt, left_pt], 2), torch.cat([right_nt, right_pt], 2)
    #     )
    #     x2 = torch.einsum(
    #         "qar,qrb,qrc->qabc", head, torch.cat([left_nt, left_pt], 2), torch.cat([right_nt, right_pt], 2)
    #     )
    #     merge = x1 / x2.unsqueeze(1).clamp(1e-6)
    #     m_head = (
    #         pred.dist.rule_marginal_with_grad["head"]
    #         .view(pred.batch_size, pred.nt_states, pred.nt_num_nodes, -1)
    #         .sum(1)
    #     )
    #     m_left_nt, m_left_pt = pred.dist.rule_marginal_with_grad["left"].split([pred.nt, pred.pt], 2)
    #     m_right_nt, m_right_pt = pred.dist.rule_marginal_with_grad["right"].split([pred.nt, pred.pt], 2)
    #     m_left_nt = m_left_nt.view(pred.batch_size, -1, pred.nt_states, pred.nt_num_nodes).sum(2)
    #     m_left_pt = m_left_pt.view(pred.batch_size, -1, pred.pt_states, pred.pt_num_nodes).sum(2)
    #     m_right_nt = m_right_nt.view(pred.batch_size, -1, pred.nt_states, pred.nt_num_nodes).sum(2)
    #     m_right_pt = m_right_pt.view(pred.batch_size, -1, pred.pt_states, pred.pt_num_nodes).sum(2)
    #     m_left = torch.cat([m_left_nt, m_left_pt], dim=2)
    #     m_right = torch.cat([m_right_nt, m_right_pt], dim=2)
    #     score = (merge / m_head.transpose(1, 2)[..., None, None].clamp(1e-6)).sum(1) + \
    #         (merge / m_left[:, :, None, :, None].clamp(1e-6)).sum(1) + \
    #         (merge / m_right[:, :, None, None, :].clamp(1e-6)).sum(1)

    #     nt_mask = (torch.arange(pred.nt_num_nodes).unsqueeze(0) < torch.tensor([len([i for i in item if i[2] != -999]) for item in pred.nt_nodes]).unsqueeze(1)).to(pred.device)
    #     pt_mask = (torch.arange(pred.pt_num_nodes).unsqueeze(0) < torch.tensor([len([i for i in item if i[2] != -999]) for item in pred.pt_nodes]).unsqueeze(1)).to(pred.device)
    #     nt_pt_mask = torch.cat([nt_mask, pt_mask], 1)
    #     valid_mask = nt_mask[:, :, None, None] & nt_pt_mask[:, None, :, None] & nt_pt_mask[:, None, None, :]
    #     score[~valid_mask] = 0.

    #     # params_ref = convert_decomp1_to_pcfg(pred.dist.params, pred.nt_states)
    #     # pcfg_ref = SentCFG((params_ref["term"], params_ref["rule"], params_ref["root"]), pred.lengths)
    #     # SRC_NT = pred.nt_num_nodes
    #     # SRC_PT = pred.pt_num_nodes
    #     # NT = pred.nt
    #     # B = pred.batch_size
    #     # TGT_NT = pred.nt_states
    #     # TGT_PT = pred.pt_states
    #     # trace = torch.empty(B, SRC_NT, SRC_NT + SRC_PT, SRC_NT + SRC_PT)
    #     # t = pcfg_ref.marginals[1]
    #     # trace[:, :, :SRC_NT, :SRC_NT] = t[:, :, :NT, :NT].reshape(B, TGT_NT, SRC_NT, TGT_NT, SRC_NT, TGT_NT, SRC_NT).sum((1,3,5))
    #     # trace[:, :, :SRC_NT, SRC_NT:] = t[:, :, :NT, NT:].reshape(B, TGT_NT, SRC_NT, TGT_NT, SRC_NT, TGT_PT, SRC_PT).sum((1,3,5))
    #     # trace[:, :, SRC_NT:, :SRC_NT] = t[:, :, NT:, :NT].reshape(B, TGT_NT, SRC_NT, TGT_PT, SRC_PT, TGT_NT, SRC_NT).sum((1,3,5))
    #     # trace[:, :, SRC_NT:, SRC_NT:] = t[:, :, NT:, NT:].reshape(B, TGT_NT, SRC_NT, TGT_PT, SRC_PT, TGT_PT, SRC_PT).sum((1,3,5))
    #     # obj = trace[~mask].sum()
    #     # print(score.max(), merge.max())
    #     # print(obj, trace[mask].sum())

    #     return -(score * (~mask).float()).clamp(max=100).flatten(1).sum(1) * 0.01
