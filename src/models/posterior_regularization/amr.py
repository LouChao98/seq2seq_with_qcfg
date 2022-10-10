from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_struct import SentCFG

from src.utils.fn import apply_to_nested_tensor

from ..tgt_parser.neural_decomp3 import NeuralDecomp3TgtParser
from ..tgt_parser.neural_nodecomp import NeuralNoDecompTgtParser
from .pr import PrTask

if TYPE_CHECKING:
    from src.models.tgt_parser.base import TgtParserPrediction
    from src.models.tgt_parser.struct.decomp_base import DecompBase


class AMRNeqPrTask(PrTask):
    def get_b(self, pred: TgtParserPrediction):
        return torch.ones(pred.batch_size, pred.pt_num_nodes, device=pred.device) * 0

    def get_init_lambdas(self, pred: TgtParserPrediction):
        return torch.full((pred.batch_size, pred.pt_num_nodes), 2.0, device=pred.device, requires_grad=True)

    def process_constraint(self, pred: TgtParserPrediction, constraints: torch.Tensor):
        return constraints.unsqueeze(-1).expand(-1, -1, pred.pt_num_nodes)

    def build_constrained_dist(self, pred: TgtParserPrediction, lambdas, constraints, entropy_reg=None):
        dist = pred.dist
        cparams = {**dist.params}
        cparams["term"] = (
            cparams["term"].view(*cparams["term"].shape[:2], dist.pt_states, dist.pt_num_nodes)
            - (lambdas[:, None] * constraints).unsqueeze(2)
        ).flatten(2)
        if entropy_reg is not None and entropy_reg > 0:
            factor = 1 / (1 - entropy_reg)
            for key, is_log_param in zip(dist.KEYS, dist.LOGSPACE):
                if is_log_param:
                    cparams[key] = cparams[key] * factor
                else:
                    cparams[key] = torch.pow(cparams[key], factor)
        return dist.spawn(params=cparams, no_trace=True)

    def calc_e(self, pred: TgtParserPrediction, constraints):
        m = pred.dist.marginal_rule["term"]
        return (m.view(*m.shape[:2], pred.pt_states, pred.pt_num_nodes) * constraints.unsqueeze(2)).sum((1, 2))


# class AMREqQCFGPrTask(PrTask):
#     def __init__(self, num_src_pt, num_src_nt, num_tgt_pt, num_tgt_nt, device):
#         self.num_src_pt = num_src_pt
#         self.num_src_nt = num_src_nt
#         self.num_tgt_pt = num_tgt_pt
#         self.num_tgt_nt = num_tgt_nt
#         self.device = device

#     def get_b(self, batch_size):
#         return torch.zeros(batch_size, self.num_src_pt, device=self.device)

#     def get_init_lambdas(self, batch_size):
#         return torch.full((batch_size, self.num_src_pt), 5.0, device=self.device, requires_grad=True)

#     def process_constraint(self, constraints):
#         return constraints.unsqueeze(-1).expand(-1, -1, self.num_src_pt)

#     def build_constrained_params(self, params, lambdas, constraints):
#         cparams = {**params}
#         cparams["term"] = (
#             params["term"].view(*params["term"].shape[:2], self.num_tgt_pt, self.num_src_pt)
#             - (lambdas[:, None] * constraints).unsqueeze(2)
#         ).flatten(2)
#         return cparams

#     def nll(self, params, lens):
#         params = (
#             params["term"],
#             params["rule"],
#             params["root"],
#             params["copy_nt"],
#         )
#         return -SentCFG(params, lens).partition

#     def ce(self, q_params, p_params, lens):
#         q_params = (
#             q_params["term"].detach(),
#             q_params["rule"].detach(),
#             q_params["root"].detach(),
#             q_params["copy_nt"],
#         )
#         q = SentCFG(q_params, lens)
#         q_margin = q.marginals
#         p_params = (
#             p_params["term"],
#             p_params["rule"],
#             p_params["root"],
#             p_params["copy_nt"],
#         )
#         p = SentCFG(p_params, lens)
#         ce = (
#             p.partition
#             - (q_margin[0].detach() * p_params[0]).sum((1, 2))
#             - (q_margin[1].detach() * p_params[1]).sum((1, 2, 3))
#             - (q_margin[2].detach() * p_params[2]).sum(1)
#         )
#         return ce

#     def calc_e(self, dist, constraints):
#         m = dist.marginals[0]
#         return (m.view(*m.shape[:2], self.num_tgt_pt, self.num_src_pt) * constraints.unsqueeze(2)).sum((1, 2))
