import logging

import torch
import torch.nn.functional as F

from ..components.common import MultiResidualLayer
from ..components.sparse_activations import entmax15, sparsemax
from .base import TgtParserPrediction
from .neural_decomp1 import NeuralDecomp1TgtParser

log = logging.getLogger(__file__)

# SHOULD ALWAYS USE softmax FOR REAL TRAINING
normalize_func = {
    "softmax": torch.log_softmax,
    "sparsemax": lambda x, dim: sparsemax(x, dim=dim).clamp(1e-32).log(),
    "entmax": lambda x, dim: entmax15(x, dim=dim).clamp(1e-32).log(),
}


class NeuralDecomp1SkeletonTgtParser(NeuralDecomp1TgtParser):
    # This inspired by Bayes Recursive Bayesian Networks: Generalising and
    #       Unifying Probabilistic Context-Free Grammars and Dynamic Bayesian Networks
    # A -> BC and A -> w is seperated by a structure random variable z.
    # Here we use z to choose ->NT ->PT
    def __init__(self, normalize_mode="softmax", use_nn=True, **kwargs):
        super().__init__(**kwargs)
        self.normalize_mode = normalize_mode
        self.normalizer = normalize_func[normalize_mode]

        assert not use_nn or self.tie_r
        self.use_nn = use_nn
        if self.use_nn:
            self.skeleton_mlp = MultiResidualLayer(self.rule_mlp_parent.out_linear.weight.shape[1], self.dim, 1)
        else:
            rule_skeleton = (torch.tensor([self.nt_states, self.pt_states]) / (self.nt_states + self.pt_states)).log()
            rule_skeleton = rule_skeleton[None, :, None].repeat(2, 1, self.cpd_rank)  # 2(direction) x 2(nt/pt) x r
            self.register_parameter("rule_skeleton", torch.nn.Parameter(rule_skeleton, requires_grad=True))

    def forward(self, node_features, spans):
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

        rule_head = self.rule_mlp_parent(nt_emb)
        rule_left_nt = self.rule_mlp_left(nt_emb)
        rule_left_pt = self.rule_mlp_left(pt_emb)
        rule_right_nt = self.rule_mlp_right(nt_emb)
        rule_right_pt = self.rule_mlp_right(pt_emb)

        if self.use_nn:
            # -> r x 4
            rule_skeleton = self.skeleton_mlp(self.rule_mlp_parent.out_linear.weight)
            rule_skeleton = rule_skeleton.transpose(0, 1).view(2, 2, -1)
            rule_skeleton = rule_skeleton.log_softmax(1)
        else:
            rule_skeleton = self.rule_skeleton.log_softmax(1)

        # fmt: off
        device = roots.device
        nt_mask = torch.arange(nt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=device).unsqueeze(1)
        nt_mask = nt_mask.unsqueeze(1).expand(-1, self.nt_states, -1).reshape(batch_size, -1)
        pt_mask = pt_mask.unsqueeze(1).expand(-1, self.pt_states, -1).reshape(batch_size, -1)
        # fmt: on
        mask = torch.cat([nt_mask, pt_mask], dim=1)
        rule_left_nt[~nt_mask] = self.neg_huge
        rule_left_pt[~pt_mask] = self.neg_huge
        rule_right_nt[~nt_mask] = self.neg_huge
        rule_right_pt[~pt_mask] = self.neg_huge
        rule_left_nt = self.normalizer(rule_left_nt, dim=1) + rule_skeleton[None, None, 0, 0]
        rule_left_pt = torch.log_softmax(rule_left_pt, dim=1) + rule_skeleton[None, None, 0, 1]
        rule_right_nt = self.normalizer(rule_right_nt, dim=1) + rule_skeleton[None, None, 1, 0]
        rule_right_pt = torch.log_softmax(rule_right_pt, dim=1) + rule_skeleton[None, None, 1, 1]

        rule_head = rule_head.log_softmax(-1)
        rule_left = torch.cat([rule_left_nt, rule_left_pt], dim=1).transpose(1, 2)
        rule_right = torch.cat([rule_right_nt, rule_right_pt], dim=1).transpose(1, 2)

        # A->a
        terms = F.log_softmax(self.vocab_out(pt_emb), dim=2)

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
        return pred
