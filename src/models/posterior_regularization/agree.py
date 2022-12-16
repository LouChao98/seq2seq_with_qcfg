from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.models.struct._inside_observed import cross_entropy, inside_with_fully_observed_tree, marginal
from src.models.struct.base import DecompBase
from src.utils.fn import apply_to_nested_tensor

if TYPE_CHECKING:
    from src.models.tgt_parser.base import TgtParserPrediction


@dataclass
class PTAlignmentAgree:
    b: float = 0.0
    rbound: float = 10
    num_iter: int = 3
    entropy_reg: float = 0.0

    def __call__(self, pred1: TgtParserPrediction, pred2: TgtParserPrediction, get_dist=False):
        mask = (
            pred1.posterior_params["term"].view(pred1.batch_size, -1, pred1.pt_states, pred1.pt_num_nodes)[:, :, 0]
            > -1e6
        )

        cparams1 = apply_to_nested_tensor(pred1.posterior_params, lambda x: x.detach())
        cparams2 = apply_to_nested_tensor(pred2.posterior_params, lambda x: x.detach())

        # If phrase copy, p(pt align to some node) = 0. So PTAgree does not like phrase copy.
        cparams1["constraint"] = None
        cparams2["constraint"] = None

        dist1 = pred1.dist.spawn(params=cparams1)
        dist2 = pred2.dist.spawn(params=cparams2)

        lambdas = self.solve(dist1, dist2, mask).detach()

        cdist1, cdist2 = self.make_constrained_dist(dist1, dist2, lambdas, mask)

        # def show(_dist, pt_nodes, explain):
        #     print(f"---[{explain}]---")
        #     decoded = _dist.decoded
        #     for span in decoded[0]:
        #         if span[0] + 1 == span[1]:
        #             aligned = pt_nodes[0][span[4]]
        #             print(f"{span[0], span[1]} -- {aligned[0], aligned[1]}")

        # show(cdist1, pred1.pt_nodes, "After PR. Dist1")
        # show(cdist2, pred2.pt_nodes, "After PR. Dist2")
        # show(dist1, pred1.pt_nodes, "Before PR. Dist1")
        # show(dist2, pred2.pt_nodes, "Before PR. Dist2")

        # mterm1 = cdist1.marginal["term"].view(pred1.batch_size, -1, pred1.pt_states, pred1.pt_num_nodes).sum(2)
        # mterm2 = cdist2.marginal["term"].view(pred2.batch_size, -1, pred2.pt_states, pred2.pt_num_nodes).sum(2)
        # diff1 = mterm1 - mterm2.transpose(1, 2)

        # mterm1_orig = dist1.marginal["term"].view(pred1.batch_size, -1, pred1.pt_states, pred1.pt_num_nodes).sum(2)
        # mterm2_orig = dist2.marginal["term"].view(pred2.batch_size, -1, pred2.pt_states, pred2.pt_num_nodes).sum(2)
        # diff2 = mterm1_orig - mterm2_orig.transpose(1, 2)

        if get_dist:
            return dist1, dist2

        return cdist1.cross_entropy(pred1.dist) + cdist2.cross_entropy(pred2.dist)

    def get_init_lambdas(self, dist1, dist2):
        return torch.zeros(
            dist1.batch_size,
            dist1.pt_num_nodes,
            dist2.pt_num_nodes,
            device=dist1.params["term"].device,
            requires_grad=True,
        )

    def solve(self, dist1, dist2, mask):
        lambdas = self.get_init_lambdas(dist1, dist2)

        for itidx in range(self.num_iter):
            cdist1, cdist2 = self.make_constrained_dist(dist1, dist2, lambdas, mask)
            target = -self.b * lambdas.sum() - (1 - self.entropy_reg) * (cdist1.partition + cdist2.partition).sum()
            target.backward()

            if (lambdas.grad.abs() < 1e-4).all():
                break
            # else:
            #     print(lambdas.grad.norm())
            with torch.no_grad():
                step_size = 1.0  # 10 / (10 + itidx)
                lambdas += step_size * lambdas.grad
                lambdas.clamp_(min=-self.rbound, max=self.rbound)
                lambdas.grad.zero_()

        return lambdas.detach()

    def make_constrained_dist(self, dist1, dist2, lambdas, mask):
        factor = 1 / (1 - self.entropy_reg)

        new_param = dist1.params["term"].view(dist1.batch_size, -1, dist1.pt_states, dist1.pt_num_nodes)
        new_param = (new_param + (lambdas.transpose(1, 2) * mask).unsqueeze(2)).flatten(2)
        cdist1 = dist1.spawn(params={"term": new_param * factor})

        new_param = dist2.params["term"].view(dist2.batch_size, -1, dist2.pt_states, dist2.pt_num_nodes)
        new_param = (new_param - (lambdas * mask.transpose(1, 2)).unsqueeze(2)).flatten(2)
        cdist2 = dist2.spawn(params={"term": new_param * factor})
        return cdist1, cdist2


@dataclass
class TreeAgree:
    b: float = 0.0
    rbound: float = 10
    num_iter: int = 3
    entropy_reg: float = 0.0

    def __call__(self, dist1: DecompBase, dist2: DecompBase, get_dist=False):

        cparams1 = apply_to_nested_tensor(dist1.params, lambda x: x.detach())
        cparams2 = apply_to_nested_tensor(dist2.params, lambda x: x.detach())

        cparams1["constraint"] = None
        cparams2["constraint"] = None

        cdist1 = dist1.spawn(params=cparams1)
        cdist2 = dist2.spawn(params=cparams2)

        lambdas = self.solve(cdist1, cdist2).detach()

        cdist1, cdist2 = self.make_constrained_dist(cdist1, cdist2, lambdas)

        # mterm1 = cdist1.marginal["trace"].flatten(3).sum(3)
        # mterm2 = cdist2.marginal["trace"].flatten(3).sum(3)
        # diff1 = mterm1 - mterm2

        # mterm1_orig = dist1.marginal["trace"].flatten(3).sum(3)
        # mterm2_orig = dist2.marginal["trace"].flatten(3).sum(3)
        # diff2 = mterm1_orig - mterm2_orig

        if get_dist:
            return cdist1, cdist2

        return cdist1.cross_entropy(dist1) + cdist2.cross_entropy(dist2)

    def get_init_lambdas(self, dist1: DecompBase, dist2: DecompBase):
        N = dist1.params["term"].shape[1] + 1
        assert dist1.params["term"].shape[1] == dist2.params["term"].shape[1]
        return torch.zeros(dist1.batch_size, N, N, device=dist1.params["term"].device, requires_grad=True)

    def solve(self, dist1: DecompBase, dist2: DecompBase):
        lambdas = self.get_init_lambdas(dist1, dist2)

        for itidx in range(self.num_iter):
            cdist1, cdist2 = self.make_constrained_dist(dist1, dist2, lambdas)
            target = -self.b * lambdas.sum() - (1 - self.entropy_reg) * (cdist1.partition + cdist2.partition).sum()
            target.backward()

            if (lambdas.grad.abs() < 1e-4).all():
                break
            # else:
            #     print(lambdas.grad.norm())
            with torch.no_grad():
                step_size = 1.0  # 10 / (10 + itidx)
                lambdas += step_size * lambdas.grad
                lambdas.clamp_(min=-self.rbound, max=self.rbound)
                lambdas.grad.zero_()

        return lambdas.detach()

    def make_constrained_dist(self, dist1: DecompBase, dist2: DecompBase, lambdas):
        factor = 1 / (1 - self.entropy_reg)
        lambdas = lambdas * factor

        add_scores = [lambdas.diagonal(w, dim1=1, dim2=2).unsqueeze(-1) for w in range(2, lambdas.shape[1])]
        cdist1 = dist1.spawn(params={"add": add_scores})

        lambdas = -lambdas
        add_scores = [lambdas.diagonal(w, dim1=1, dim2=2).unsqueeze(-1) for w in range(2, lambdas.shape[1])]
        cdist2 = dist2.spawn(params={"add": add_scores})

        return cdist1, cdist2


@dataclass
class NTAlignmentAgree:
    b: float = 0.0
    rbound: float = 10
    num_iter: int = 3
    entropy_reg: float = 0.0

    def __call__(self, dist1: DecompBase, dist2: DecompBase, spans1, spans2, get_dist=False):

        cparams1 = apply_to_nested_tensor(dist1.params, lambda x: x.detach())
        cparams2 = apply_to_nested_tensor(dist2.params, lambda x: x.detach())

        cparams1["constraint"] = None
        cparams2["constraint"] = None

        cdist1 = dist1.spawn(params=cparams1)
        cdist2 = dist2.spawn(params=cparams2)

        lambdas = self.solve(cdist1, cdist2, spans1, spans2).detach()

        add1, add2 = self.make_add_scores(cdist1, cdist2, lambdas)

        # m1 = marginal(cdist1.params, cdist1.lens, spans2, add1, constrained=True).view(dist1.batch_size, -1, dist1.nt_states, dist1.nt_num_nodes).sum(2)
        # m2 = marginal(cdist2.params, cdist2.lens, spans1, add2, constrained=True).view(dist2.batch_size, -1, dist2.nt_states, dist2.nt_num_nodes).sum(2)
        # diff1 = m1 - m2.transpose(1, 2)
        # m1_orig = marginal(cdist1.params, cdist1.lens, spans2, add1, constrained=False).view(dist1.batch_size, -1, dist1.nt_states, dist1.nt_num_nodes).sum(2)
        # m2_orig = marginal(cdist2.params, cdist2.lens, spans1, add2, constrained=False).view(dist2.batch_size, -1, dist2.nt_states, dist2.nt_num_nodes).sum(2)
        # diff2 = m1_orig - m2_orig.transpose(1, 2)

        if get_dist:
            return dist1, dist2

        return cross_entropy(dist1.params, dist1.lens, spans2, add1) + cross_entropy(
            dist2.params, dist2.lens, spans1, add2
        )

    def get_init_lambdas(self, dist1: DecompBase, dist2: DecompBase):
        return torch.zeros(
            dist1.batch_size,
            dist1.nt_num_nodes,
            dist2.nt_num_nodes,
            device=dist1.params["term"].device,
            requires_grad=True,
        )

    def solve(self, dist1: DecompBase, dist2: DecompBase, spans1, spans2):
        lambdas = self.get_init_lambdas(dist1, dist2)

        for itidx in range(self.num_iter):
            add1, add2 = self.make_add_scores(dist1, dist2, lambdas)
            target = (
                -self.b * lambdas.sum()
                - (1 - self.entropy_reg)
                * (
                    inside_with_fully_observed_tree(dist1.params, dist1.lens, spans2, add1)
                    + inside_with_fully_observed_tree(dist2.params, dist2.lens, spans1, add2)
                ).sum()
            )
            target.backward()

            if (lambdas.grad.abs() < 1e-4).all():
                break
            # else:
            #     print(lambdas.grad.norm())
            with torch.no_grad():
                step_size = 1.0  # 10 / (10 + itidx)
                lambdas += step_size * lambdas.grad
                lambdas.clamp_(min=-self.rbound, max=self.rbound)
                lambdas.grad.zero_()

        return lambdas.detach()

    def make_add_scores(self, dist1: DecompBase, dist2: DecompBase, lambdas):
        factor = 1 / (1 - self.entropy_reg)
        lambdas = lambdas * factor

        add_scores1 = lambdas.transpose(1, 2).unsqueeze(2).expand(-1, -1, dist1.nt_states, -1).contiguous().flatten(2)

        lambdas = -lambdas
        add_scores2 = lambdas.unsqueeze(2).expand(-1, -1, dist2.nt_states, -1).contiguous().flatten(2)

        return add_scores1, add_scores2
