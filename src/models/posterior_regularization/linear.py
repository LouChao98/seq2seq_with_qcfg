from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.utils.fn import apply_to_nested_tensor

if TYPE_CHECKING:
    from src.models.tgt_parser.base import TgtParserPrediction


log = logging.getLogger(__file__)


def neg_log_length():
    return lambda x: -torch.tensor(x.lengths, dtype=torch.float, device=x.device).log_()


def neg_log2_length():
    return lambda x: -torch.tensor(x.lengths, dtype=torch.float, device=x.device).log2_()


@dataclass
class UngroundedPRLineSearchSolver:
    # solve E[phi] <= b, where b is a scalar

    field: str = "rule"
    b: float = 2.0
    lbound: float = 1e-4
    rbound: float = 1e3
    num_point: int = 16
    num_iter: int = 3
    ignore_failure: bool = True

    def __call__(self, pred: TgtParserPrediction, constraint_feature, get_dist=False, shortcut=True):
        cparams = apply_to_nested_tensor(pred.posterior_params, lambda x: x.detach())
        if cparams[self.field].ndim != constraint_feature.ndim:
            # TODO: REMOVE DANGEROUS ASSUMPTION
            assert cparams[self.field].shape[2:] == constraint_feature.shape[1:]
            constraint_feature = constraint_feature.unsqueeze(1)
        if self.b == 0 and shortcut:
            p = cparams[self.field].clone()
            p[constraint_feature > 0.1] = -1e9
            cparams[self.field] = p
            dist = pred.dist.spawn(params=cparams)
        else:
            lambdas = self.solve(pred, constraint_feature)
            cparams[self.field] = self.make_constrained_params(
                cparams[self.field], constraint_feature, lambdas, pred.dist.is_log_param(self.field)
            )
            dist = pred.dist.spawn(params=cparams)
        if get_dist:
            return dist
        return dist.cross_entropy(pred.dist, fix_left=True)

    @torch.no_grad()
    def solve(self, pred: TgtParserPrediction, constraint_feature):
        lambdas = []
        for bidx in range(pred.batch_size):
            sub_pred = pred.get_and_expand(bidx, self.num_point)
            constraint_feature_item = constraint_feature[bidx, None]
            lambdas.append(self.solve_one_instance(sub_pred, constraint_feature_item))
        return torch.tensor(lambdas, device=pred.device, dtype=torch.float32)

    def solve_one_instance(self, pred: TgtParserPrediction, constraint_feature):
        lb, rb = self.lbound, self.rbound
        lt, rt, max_t, max_l = None, None, None, None
        input_params = pred.posterior_params
        b = self.b if isinstance(self.b, (int, float)) else self.b(pred)
        for itidx in range(self.num_iter):
            if itidx > 0:  # skip lb rb
                lgrid_np = np.geomspace(lb, rb, self.num_point + 2, dtype=np.float32)
                lgrid = torch.from_numpy(lgrid_np[1:-1]).to(pred.device)
            else:
                lgrid_np = np.geomspace(lb, rb, self.num_point, dtype=np.float32)
                lgrid = torch.from_numpy(lgrid_np).to(pred.device)
            # potential * exp(-lambda * constraint)
            params = {**input_params}
            params[self.field] = self.make_constrained_params(
                params[self.field], constraint_feature, lgrid, pred.dist.is_log_param(self.field)
            )
            dist = pred.dist.spawn(params=params, batch_size=self.num_point)
            target = -lgrid * b + dist.nll
            target = target.cpu().numpy()
            if itidx > 0:  # take back lb, rb and argmax
                target = [lt, *target.tolist(), rt]
                max_insert = lgrid_np.searchsorted(max_l)
                lgrid_np = lgrid_np.tolist()
                lgrid_np.insert(max_insert, max_l)
                target.insert(max_insert, max_t)
                target = np.asarray(target)
            argmax_i = np.argmax(target)
            if argmax_i == 0 or argmax_i == len(target) - 1:
                if itidx == 0 and argmax_i != 0 and not self.ignore_failure:
                    # A very small i (argmax=0) is acceptable as it means
                    # we can satisfy the constraint without effort
                    #
                    # This can also be understood as the slack-penalty version
                    # of PR. The dual problem is
                    # max -b\lambda - log Z s.t. 0 <= \lambda, dual_norm(\lambda) < \sigma
                    log.warning("Line search fails.")
                return lgrid_np[argmax_i]
            lt, rt = target[argmax_i - 1], target[argmax_i + 1]
            lb, rb = lgrid_np[argmax_i - 1], lgrid_np[argmax_i + 1]
            max_t = target[argmax_i]
            max_l = lgrid_np[argmax_i]
            if rb - lb < 1e-2 or abs(lt - rt) < 1e-2:
                return lgrid_np[argmax_i]
        else:
            return lgrid_np[argmax_i]

    def make_constrained_params(self, t, constraint_feature, lambdas, log_input):
        if log_input:
            return t - constraint_feature * lambdas.view([-1] + [1] * (constraint_feature.ndim - 1))
        else:
            return (t.log() - constraint_feature * lambdas.view([-1] + [1] * (constraint_feature.ndim - 1))).exp()
