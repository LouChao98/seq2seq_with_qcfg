import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.utils.fn import apply_to_nested_tensor

log = logging.getLogger(__file__)


@dataclass
class UngroundedPRLineSearchSolver:
    # solve E[phi] <= b, where b is a scalar

    on: str = "rule"
    b: float = 2.0
    lbound: float = 1e-4
    rbound: float = 1e3
    num_point: int = 16
    num_iter: int = 3
    pcfg: Any = None

    def __call__(self, params, lens, constraint_feature):
        lambdas = self.solve(params, lens, constraint_feature)
        cparams = apply_to_nested_tensor(params, lambda x: x.detach())
        cparams[self.on] = cparams[self.on] - constraint_feature * lambdas.view(
            [-1] * (constraint_feature.ndim - 1)
        )
        return self.pcfg.ce(cparams, params, lens)

    @torch.no_grad()
    def solve(self, params, lens, constraint_feature):
        batch_size = len(lens)
        n = self.num_point
        lambdas = []
        for bidx in range(batch_size):
            params_item = apply_to_nested_tensor(
                params,
                lambda x: x[bidx, None].detach().expand([n] + [-1] * (x.ndim - 1)),
            )
            constraint_feature_item = constraint_feature[bidx, None]
            lens_item = lens[bidx]
            lambdas.append(
                self.solve_one_instance(params_item, lens_item, constraint_feature_item)
            )
        return torch.tensor(lambdas, device=params[self.on].device, dtype=torch.float32)

    def solve_one_instance(self, params, lens, constraint_feature):
        lb, rb = self.lbound, self.rbound
        lt, rt, max_t, max_l = None, None, None, None
        device = constraint_feature.device
        for itidx in range(self.num_iter):
            if itidx > 0:  # skip lb rb
                lgrid_np = np.geomspace(lb, rb, self.num_point + 2, dtype=np.float32)
                lgrid = torch.from_numpy(lgrid_np[1:-1]).to(device)
            else:
                lgrid_np = np.geomspace(lb, rb, self.num_point, dtype=np.float32)
                lgrid = torch.from_numpy(lgrid_np).to(device)
            # potential * exp(-lambda * constraint)
            params[self.on] = params[self.on] - constraint_feature * lgrid.view(
                -1, *[1] * (constraint_feature.ndim - 1)
            )
            target = -lgrid * self.b + self.pcfg(params, [lens] * self.num_point)
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
                if itidx == 0 and argmax_i != 0:
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
            if rb - lb < 1e-3:
                return lgrid_np[argmax_i]
        else:
            return lgrid_np[argmax_i]
