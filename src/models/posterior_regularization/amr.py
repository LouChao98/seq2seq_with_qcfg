import torch
from torch_struct import SentCFG

from ..tgt_parser.neural_qcfg import NeuralQCFGTgtParser
from ..tgt_parser.neural_qcfg_d1 import NeuralQCFGD1TgtParser
from ..tgt_parser.struct.d1_pcfg import D1PCFG
from .pr import PrTask


class AMRNeqQCFGPrTask(PrTask):
    def __init__(self, num_src_pt, num_src_nt, num_tgt_pt, num_tgt_nt, device):
        self.num_src_pt = num_src_pt
        self.num_src_nt = num_src_nt
        self.num_tgt_pt = num_tgt_pt
        self.num_tgt_nt = num_tgt_nt
        self.device = device

    def get_b(self, batch_size):
        return torch.ones(batch_size, self.num_src_pt, device=self.device)

    def get_init_lambdas(self, batch_size):
        return torch.full(
            (batch_size, self.num_src_pt), 2.0, device=self.device, requires_grad=True
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

    def calc_e(self, params, lens, constraints):
        params = (
            params["term"],
            params["rule"],
            params["root"],
            params["copy_nt"],
        )
        dist = SentCFG(params, lens)

        m = dist.marginals[0]
        return (
            m.view(*m.shape[:2], self.num_tgt_pt, self.num_src_pt)
            * constraints.unsqueeze(2)
        ).sum((1, 2))


class AMRNeqD1PrTask(AMRNeqQCFGPrTask):
    params = ["term", "root", "left", "right", "head", "slr"]

    def __init__(self, num_src_pt, num_src_nt, num_tgt_pt, num_tgt_nt, device):
        super().__init__(num_src_pt, num_src_nt, num_tgt_pt, num_tgt_nt, device)
        self.pcfg = D1PCFG(self.num_tgt_nt, self.num_tgt_pt)

    def nll(self, params, lens):
        return self.pcfg(params, lens)

    @torch.enable_grad()
    def ce(self, q_params, p_params, lens):
        q_params = {
            k: v.detach().requires_grad_() if isinstance(v, torch.Tensor) else v
            for k, v in q_params.items()
        }
        q_ll = -self.pcfg(q_params, lens)
        q_margin = torch.autograd.grad(q_ll.sum(), [q_params[k] for k in self.params])
        ce = -self.pcfg(p_params, lens)
        for i, k in enumerate(self.params):
            ce = ce - (q_margin[i].detach() * p_params[k]).flatten(1).sum(1)
        return ce

    @torch.enable_grad()
    def calc_e(self, params, lengths, constraints):
        if not params["term"].requires_grad:
            params = {
                k: v.requires_grad_() if isinstance(v, torch.Tensor) else v
                for k, v in params.items()
            }
        q_ll = -self.pcfg(params, lengths)
        m = torch.autograd.grad(q_ll.sum(), [params["term"]])[0]
        # print("entropy of label dist", - (m * (m + 1e-9).log()).sum(-1))
        return (
            m.view(*m.shape[:2], self.num_tgt_pt, self.num_src_pt)
            * constraints.unsqueeze(2)
        ).sum((1, 2))


AMRNeqTasks = {
    NeuralQCFGTgtParser: AMRNeqQCFGPrTask,
    NeuralQCFGD1TgtParser: AMRNeqD1PrTask,
}


class AMREqQCFGPrTask(PrTask):
    def __init__(self, num_src_pt, num_src_nt, num_tgt_pt, num_tgt_nt, device):
        self.num_src_pt = num_src_pt
        self.num_src_nt = num_src_nt
        self.num_tgt_pt = num_tgt_pt
        self.num_tgt_nt = num_tgt_nt
        self.device = device

    def get_b(self, batch_size):
        return torch.zeros(batch_size, self.num_src_pt, device=self.device)

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
