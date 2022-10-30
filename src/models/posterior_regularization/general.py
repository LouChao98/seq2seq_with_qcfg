from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from ..tgt_parser.neural_decomp3 import NeuralDecomp3TgtParser
from ..tgt_parser.neural_nodecomp import NeuralNoDecompTgtParser
from .pr import PrTask

if TYPE_CHECKING:
    from src.models.tgt_parser.base import TgtParserPrediction
    from src.models.tgt_parser.struct.decomp_base import DecompBase


class NeqPT(PrTask):
    # this pushes count of alignment to 0.
    # problematic. cannot distingsuish 0 5 0 0 0 vs 1 1 1 1 1
    def __init__(self, strength=None) -> None:
        super().__init__()
        self.strength = strength

    # this encourage y_pt < 1
    def get_b(self, pred: TgtParserPrediction):
        if self.strength is not None:
            return None
        return torch.ones(pred.batch_size, pred.pt_num_nodes, device=pred.device)

    def get_init_lambdas(self, pred: TgtParserPrediction):
        return torch.full((pred.batch_size, pred.pt_num_nodes), 0.5, device=pred.device, requires_grad=True)

    def process_constraint(self, pred: TgtParserPrediction, constraints: Optional[torch.Tensor] = None):
        # constraints: bsz x n or None
        if constraints is None:
            return torch.ones(
                (pred.batch_size, pred.posterior_params["term"].shape[1], pred.pt_num_nodes), device=pred.device
            )
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
        return dist.spawn(params=cparams)

    def calc_e(self, pred: TgtParserPrediction, constraints):
        m = pred.dist.marginal["term"]
        return (m.view(*m.shape[:2], pred.pt_states, pred.pt_num_nodes) * constraints.unsqueeze(2)).sum((1, 2))

    def lambda_simplex_constraint(self):
        return self.strength


class NeqPTImpl2(PrTask):
    # this pushes count of alignment >= 1

    # this encourage y_pt < 1
    def get_b(self, pred: TgtParserPrediction):
        return -torch.ones(pred.batch_size, pred.pt_num_nodes, device=pred.device)

    def get_init_lambdas(self, pred: TgtParserPrediction):
        return torch.full((pred.batch_size, pred.pt_num_nodes), 0.5, device=pred.device, requires_grad=True)

    def process_constraint(self, pred: TgtParserPrediction, constraints: Optional[torch.Tensor] = None):
        # constraints: bsz x n or None
        if constraints is None:
            return -torch.ones(
                (pred.batch_size, pred.posterior_params["term"].shape[1], pred.pt_num_nodes), device=pred.device
            )
        return -constraints.unsqueeze(-1).expand(-1, -1, pred.pt_num_nodes)

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
        return dist.spawn(params=cparams)

    def calc_e(self, pred: TgtParserPrediction, constraints):
        m = pred.dist.marginal["term"]
        return (m.view(*m.shape[:2], pred.pt_states, pred.pt_num_nodes) * constraints.unsqueeze(2)).sum((1, 2))


class NeqNTImpl2(PrTask):
    def __init__(self, mode) -> None:
        super().__init__()
        # mode 1 = rule with shape bsz x nt x (nt + pt) x (nt + pt)
        # mode 2 = head with shape bsz x nt x r
        assert mode in (1, 2)
        self.mode = mode

    def get_b(self, pred: TgtParserPrediction):
        return -torch.ones(pred.batch_size, pred.nt_num_nodes, device=pred.device)

    def get_init_lambdas(self, pred: TgtParserPrediction):
        return torch.full((pred.batch_size, pred.nt_num_nodes), 0.5, device=pred.device, requires_grad=True)

    def process_constraint(self, pred: TgtParserPrediction, constraints: Optional[torch.Tensor] = None):
        assert constraints is None
        return -torch.ones((pred.batch_size, pred.nt_num_nodes), device=pred.device)

    def build_constrained_dist(self, pred: TgtParserPrediction, lambdas, constraints, entropy_reg=None):
        dist = pred.dist
        cparams = {**dist.params}
        field = "rule" if self.model == 1 else "head"
        cparams[field] = (
            cparams[field].view(pred.batch_size, pred.nt_states, pred.nt_num_nodes, -1)
            - (lambdas * constraints)[:, None, :, None]
        ).view(cparams[field].shape)
        if entropy_reg is not None and entropy_reg > 0:
            factor = 1 / (1 - entropy_reg)
            for key, is_log_param in zip(dist.KEYS, dist.LOGSPACE):
                if is_log_param:
                    cparams[key] = cparams[key] * factor
                else:
                    cparams[key] = torch.pow(cparams[key], factor)
        return dist.spawn(params=cparams)
