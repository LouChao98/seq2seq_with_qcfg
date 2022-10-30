from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import torch

from src.utils.fn import apply_to_nested_tensor

from .project_simplex import project_simplex

if TYPE_CHECKING:
    from src.models.tgt_parser.base import TgtParserPrediction


class PrTask:
    def get_b(self, pred, batch_size):
        ...

    def get_init_lambdas(self, pred, batch_size):
        ...

    def process_constraint(self, pred, constraints):
        ...

    def build_constrained_dist(self, pred, lambdas, constraints, entropy_reg=None):
        ...

    def calc_e(self, pred, constraints):
        ...

    def lambda_simplex_constraint(self):
        ...


@torch.enable_grad()
def compute_pr(pred: TgtParserPrediction, constraints, task: PrTask, get_dist=False, **kwargs):
    constraints = task.process_constraint(pred, constraints)
    entropy_reg = kwargs.get("entropy_reg", 0.0)

    b = task.get_b(pred)
    if b is not None:
        e = task.calc_e(pred, constraints)
        if (e < task.get_b(pred)).all():
            if get_dist:  # do nothing
                return copy(pred)
            return torch.zeros(pred, device=pred.device)

    lambdas = pgd_solver(pred, constraints, task, **kwargs)
    cdist = task.build_constrained_dist(pred, lambdas, constraints, entropy_reg)

    pred_debug = copy(pred)
    pred_debug.dist = cdist
    pe = task.calc_e(pred_debug, constraints)

    if get_dist:
        return cdist
    ce = cdist.cross_entropy(pred.dist, fix_left=True)
    return ce


def pgd_solver(pred: TgtParserPrediction, constraints, task: PrTask, **kwargs):
    num_iter = kwargs.get("num_iter", 10)
    entropy_reg = kwargs.get("entropy_reg", 0.0)

    lambdas = task.get_init_lambdas(pred)
    lambda_simplex_constraint = task.lambda_simplex_constraint()
    b = task.get_b(pred)
    # b is for constraint set, lambda_simplex_constraint is for slack penalty
    assert (b is None) != (lambda_simplex_constraint is None), "Bad task"
    pred = copy(pred)
    pred.posterior_params = apply_to_nested_tensor(pred.posterior_params, lambda x: x.detach())
    pred.dist = pred.dist.spawn(params=pred.posterior_params)
    for itidx in range(num_iter):
        # constrained_params = {**params}
        # constrained_params["add_scores"] = [
        #     -(lambdas[:, None, None, None] * item).sum(-1)
        #     for item in factorized_constraint
        # ]
        cdist = task.build_constrained_dist(pred, lambdas, constraints, entropy_reg)
        if b is None:
            target = ((1 - entropy_reg) * cdist.nll).sum()
        else:
            target = (-(lambdas * b).sum(-1) + (1 - entropy_reg) * cdist.nll).sum()
        target.backward()
        if (lambdas.grad.abs() < 1e-4).all():
            break
        with torch.no_grad():
            # maximize target
            step_size = 1.0  # 1 looks pretty good

            # step_size = 2 / (2 + itidx)

            # step_size = torch.ones(batch_size, device=target.device)
            # _l = lambdas + lambdas.grad
            # cparams = pr_task.build_constrained_dist(params, lambdas, constraints)
            # factor = (lambdas.grad ** 2).sum(-1)
            # condition = (-(-(_l * b).sum(-1) + self.pcfg(cparams, lens)).sum() + target) * 2 / factor
            # while (condition > - step_size).any():
            #     step_size[condition > -step_size] *= 0.8
            # step_size = step_size.unsqueeze(-1).to(lambdas.device)

            lambdas += step_size * lambdas.grad

            # projection
            if lambda_simplex_constraint is not None:
                lambdas.data = project_simplex(lambdas.data, lambda_simplex_constraint, axis=1)
            else:
                lambdas.clamp_(0)

            lambdas.grad.zero_()
    # print(itidx)
    return lambdas.detach()
