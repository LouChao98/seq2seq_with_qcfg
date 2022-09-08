import torch


class PrTask:
    def get_b(self, batch_size):
        ...

    def get_init_lambdas(self, batch_size):
        ...

    def process_constraint(self, constraints):
        ...

    def build_constrained_params(self, params, lambdas, constraints):
        ...

    def nll(self, params, lens):
        ...

    def calc_e(self, dist, constraints):
        ...

    def ce(self, q_params, p_params, lens):
        ...


def compute_pr(params, lens, constraints, task: PrTask, get_param=False, **kwargs):
    constraints = task.process_constraint(constraints)
    lambdas = pgd_solver(params, lens, constraints, task, **kwargs).detach()
    cparams = task.build_constrained_params(params, lambdas, constraints)
    if get_param:
        return cparams
    return task.ce(cparams, params, lens)


def pgd_solver(params, lens, constraints, task: PrTask, **kwargs):
    num_iter = kwargs.get("num_iter", 3)

    batch_size = len(lens)
    lambdas = task.get_init_lambdas(batch_size)
    b = task.get_b(batch_size)
    params = {
        k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in params.items()
    }
    for itidx in range(num_iter):
        # constrained_params = {**params}
        # constrained_params["add_scores"] = [
        #     -(lambdas[:, None, None, None] * item).sum(-1)
        #     for item in factorized_constraint
        # ]
        cparams = task.build_constrained_params(params, lambdas, constraints)
        # TODO we can cache grads in BP and reuse them
        target = (-(lambdas * b).sum(-1) + task.nll(cparams, lens)).sum()
        target.backward()
        if (lambdas.grad.abs() < 1e-5).all():
            break
        with torch.no_grad():
            # maximize target
            step_size = 1.0  # 1 looks pretty good

            # step_size = 2 / (2 + itidx)

            # step_size = torch.ones(batch_size, device=target.device)
            # _l = lambdas + lambdas.grad
            # cparams = pr_task.build_constrained_params(params, lambdas, constraints)
            # factor = (lambdas.grad ** 2).sum(-1)
            # condition = (-(-(_l * b).sum(-1) + self.pcfg(cparams, lens)).sum() + target) * 2 / factor
            # while (condition > - step_size).any():
            #     step_size[condition > -step_size] *= 0.8
            # step_size = step_size.unsqueeze(-1).to(lambdas.device)

            lambdas += step_size * lambdas.grad
            lambdas.clamp_(0)
            lambdas.grad.zero_()

    return lambdas
