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


class NoSharedAlignment(PrTask):
    def get_b(self, pred: TgtParserPrediction):
        return torch.ones(pred.batch_size, pred.nt_num_nodes, device=pred.device)

    def get_init_lambdas(self, pred: TgtParserPrediction):
        return torch.full((pred.batch_size, pred.nt_num_nodes), 0.5, device=pred.device, requires_grad=True)

    def process_constraint(self, pred: TgtParserPrediction, constraints: torch.Tensor):
        return constraints.unsqueeze(-1).expand(-1, -1, pred.pt_num_nodes)

    def build_constrained_dist(self, pred: TgtParserPrediction, lambdas, constraints, entropy_reg=None):
        dist = pred.dist
        cparams = {**dist.params}
        cparams
        if entropy_reg is not None and entropy_reg > 0:
            factor = 1 / (1 - entropy_reg)
            for key, is_log_param in zip(dist.KEYS, dist.LOGSPACE):
                if is_log_param:
                    cparams[key] = cparams[key] * factor
                else:
                    cparams[key] = torch.pow(cparams[key], factor)
        return dist.spawn(params=cparams)
