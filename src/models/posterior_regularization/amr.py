import torch
from torch_struct import SentCFG

from ..tgt_parser.neural_qcfg import NeuralQCFGTgtParser
from .pr import PrTask


class AMRNeqQCFGPrTask(PrTask):
    def __init__(self, num_src_pt, num_tgt_pt, device):
        self.num_src_pt = num_src_pt
        self.num_tgt_pt = num_tgt_pt
        self.device = device

    def get_b(self, batch_size):
        return torch.ones(batch_size, self.num_src_pt, device=self.device)

    def get_init_lambdas(self, batch_size):
        return torch.full(
            (batch_size, self.num_src_pt), 5.0, device=self.device, requires_grad=True
        )

    def process_constraint(self, constraints):
        return constraints.unsqueeze(-1).expand(-1, -1, self.num_src_pt)

    def build_constrained_params(self, params, lambdas, constraints):
        cparams = {**params}
        cparams["term"] = (
            params["term"].view(
                *params["term"].shape[:2], self.num_tgt_pt, self.num_src_pt
            )
            - (lambdas[:, None] * constraints).unsqueeze(2)
        ).flatten(2)
        return cparams

    def nll(self, params, lens):
        params = (
            params["term"],
            params["rule"],
            params["root"],
            params["copy_nt"],
        )
        return -SentCFG(params, lens).partition

    def ce(self, q_params, p_params, lens):
        q_params = (
            q_params["term"].detach(),
            q_params["rule"].detach(),
            q_params["root"].detach(),
            q_params["copy_nt"],
        )
        q = SentCFG(q_params, lens)
        q_margin = q.marginals
        p_params = (
            p_params["term"],
            p_params["rule"],
            p_params["root"],
            p_params["copy_nt"],
        )
        p = SentCFG(p_params, lens)
        ce = (
            p.partition
            - (q_margin[0].detach() * p_params[0]).sum((1, 2))
            - (q_margin[1].detach() * p_params[1]).sum((1, 2, 3))
            - (q_margin[2].detach() * p_params[2]).sum(1)
        )
        return ce

    def calc_e(self, dist, constraints):
        m = dist.marginals[0]
        return (
            m.view(*m.shape[:2], self.num_tgt_pt, self.num_src_pt)
            * constraints.unsqueeze(2)
        ).sum((1, 2))


AMRNeqQCFGTasks = {NeuralQCFGTgtParser: AMRNeqQCFGPrTask}
