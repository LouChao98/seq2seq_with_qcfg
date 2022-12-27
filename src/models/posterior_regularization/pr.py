from __future__ import annotations

import logging
from copy import copy
from typing import TYPE_CHECKING

import torch

from src.utils.fn import apply_to_nested_tensor

from .project_simplex import project_simplex

if TYPE_CHECKING:
    from src.models.tgt_parser.base import TgtParserPrediction

logger = logging.getLogger(__file__)


class PrTask:
    def get_b(self, pred):
        ...

    def get_init_lambdas(self, pred):
        ...

    def process_constraint(self, pred, constraints):
        ...

    def build_constrained_dist(self, pred, lambdas, constraints, entropy_reg=None):
        ...

    def calc_e(self, pred, constraints):
        ...

    def lambda_simplex_constraint(self):
        ...


class MultiTask(PrTask):
    def __init__(self, *tasks):
        self.tasks = tasks
        self.sizes = None

    def get_b(self, pred):
        b = [task.get_b(pred) for task in self.tasks]
        sizes = [item.shape[1] for item in b]
        if self.sizes is None:
            self.sizes = sizes
        else:
            assert self.sizes == sizes
        return torch.cat(b, dim=1)

    def get_init_lambdas(self, pred):
        lambdas = [task.get_b(pred) for task in self.tasks]
        sizes = [item.shape[1] for item in lambdas]
        if self.sizes is None:
            self.sizes = sizes
        else:
            assert self.sizes == sizes
        return torch.cat(lambdas, dim=1)

    def process_constraint(self, pred, constraints):
        output = []
        assert len(self.tasks) == len(constraints)
        for task, constraint in zip(self.tasks, constraints):
            output.append(task.process_constraint(pred, constraint))
        return output

    def build_constrained_dist(self, pred, lambdas, constraints, entropy_reg=None):
        pred = copy(pred)
        offset = 0
        for task, size in zip(self.tasks, self.sizes):
            dist = task.build_constrained_dist(
                pred, lambdas[:, offset : offset + size], constraints[:, offset : offset + size], None
            )
            pred.dist = dist
            offset += size
        if entropy_reg is not None and entropy_reg > 0:
            cparams = pred.dist.params
            factor = 1 / (1 - entropy_reg)
            for key, is_log_param in zip(dist.KEYS, dist.LOGSPACE):
                if is_log_param:
                    cparams[key] = cparams[key] * factor
                else:
                    cparams[key] = torch.pow(cparams[key], factor)
            dist = dist.spawn(params=cparams)
        return dist

    def calc_e(self, pred, constraints):
        offset = 0
        e = []
        for task, size in zip(self.tasks, self.sizes):
            e.append(task.calc_e(pred, constraints[:, offset : offset + size]))
            offset += size
        return torch.cat(e, dim=1)

    def lambda_simplex_constraint(self):
        c = [task.lambda_simplex_constraint for task in self.tasks]
        assert all(item is None for item in c)


@torch.enable_grad()
def compute_pr(pred: TgtParserPrediction, constraints, task: PrTask, get_dist=False, **kwargs):
    constraints = task.process_constraint(pred, constraints)
    entropy_reg = kwargs.get("entropy_reg", 0.0)

    b = task.get_b(pred)
    if b is not None:
        e = task.calc_e(pred, constraints)
        if (e < b).all():
            logger.warning("Skipping PR.")
            if get_dist:  # do nothing
                return pred.dist
            return torch.zeros(pred.batch_size, device=pred.device)

    lambdas = pgd_solver(pred, constraints, task, **kwargs)
    cdist = task.build_constrained_dist(pred, lambdas, constraints, entropy_reg)

    # pred_debug = copy(pred)
    # pred_debug.dist = cdist
    # pe = task.calc_e(pred_debug, constraints)

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
