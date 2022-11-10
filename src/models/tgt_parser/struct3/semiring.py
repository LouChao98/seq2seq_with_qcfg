from typing import List

import torch


class SemiringBase:
    zero: torch.Tensor
    one: torch.Tensor
    size: int

    @classmethod
    def set_device(cls, device):
        cls.zero = cls.zero.to(device)
        cls.one = cls.one.to(device)

    @classmethod
    def new_zeros(cls, shape):
        out = cls.zero.clone().view([cls.size] + [1] * len(shape))
        out = out.repeat(1, *shape)
        return out

    @classmethod
    def new_ones(cls, shape):
        out = cls.one.clone().view([cls.size] + [1] * len(shape))
        out = out.repeat(1, *shape)
        return out

    @classmethod
    def add(cls, *t):
        x = torch.stack(t, dim=-1)
        return cls.sum(x, dim=-1)

    @classmethod
    def normal_space_add(cls, *t):
        x = torch.stack(t, dim=-1)
        return cls.normal_space_sum(x, dim=-1)


class LogSemiring(SemiringBase):
    zero = torch.tensor(-1e9)
    one = torch.tensor(0.0)
    size = 1

    @staticmethod
    def convert(a):
        return a.unsqueeze(0)

    @staticmethod
    def unconvert(a):
        return a.squeeze(0)

    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def normal_space_mul(a, b):
        return a * b

    @staticmethod
    def sum(a, dim):
        return torch.logsumexp(a, dim)

    @staticmethod
    def normal_space_sum(a, dim):
        return a.sum(dim)

    @staticmethod
    def to_normal_space(tlist, dims):
        max_val = [t.amax(dims) for t in tlist]
        normalizer = torch.stack(max_val, dim=-1).max(-1)[0]
        shape = list(normalizer.shape) + [1] * len(dims)
        tlist = [(t - normalizer.view(shape)).exp() for t in tlist]
        return tlist, normalizer

    @staticmethod
    def to_log_space(x, xn):
        return (x + 1e-9).log() + xn.view(list(xn.shape) + [1] * (x.ndim - xn.ndim))


class MaxSemiring(LogSemiring):
    @staticmethod
    def sum(a, dim):
        return torch.max(a, dim)[0]

    @staticmethod
    def normal_space_sum(a, dim):
        return torch.max(a, dim)[0]

    # # TODO these should be also transparent like SampledSemiring, but this should
    # # not be used in practice because cpd-ed pcfgs do not support this.
    # @staticmethod
    # def to_normal_space(tlist, dims):
    #     max_val = [t.amax(dims) for t in tlist]
    #     normalizer = torch.stack(max_val, dim=-1).max(-1)[0]
    #     shape = list(normalizer.shape) + [1] * len(dims)
    #     tlist = [(t - normalizer.view(shape)).exp() for t in tlist]
    #     return tlist, normalizer

    # @staticmethod
    # def to_log_space(x, xn):
    #     return (x + 1e-9).log() + xn.view(list(xn.shape) + [1] * (x.ndim - xn.ndim))


class EntropySemiring(SemiringBase):
    zero = torch.tensor([-1e9, 0.0])
    one = torch.tensor([0.0, 0.0])
    size = 2

    @staticmethod
    def convert(a):
        values = a.new_zeros((2,) + a.shape)
        values[0] = a
        return values

    @staticmethod
    def unconvert(a):
        return a[1]

    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def normal_space_mul(a, b):
        # return torch.stack((a[0] * b[0], a[1] + b[1]), dim=0)
        out = torch.empty_like(a)
        torch.mul(a[0], b[0], out[0])
        torch.add(a[1], b[1], out[1])
        return out

    @staticmethod
    def sum(a, dim):
        dim = dim - 1 if dim > 0 else dim
        part = torch.logsumexp(a[0], dim=dim)
        log_sm = a[0] - part.unsqueeze(dim)
        sm = log_sm.exp()
        return torch.stack((part, torch.sum(a[1].mul(sm) - log_sm.mul(sm), dim=dim)))

    @staticmethod
    def normal_space_sum(a, dim):
        dim = dim - 1 if dim > 0 else dim
        part = torch.sum(a[0], dim=dim)
        sm = a[0] / (part.unsqueeze(dim) + 1e-9)
        return torch.stack((part, torch.sum(a[1].mul(sm) - sm.log().mul(sm), dim=dim)))

    @staticmethod
    def to_normal_space(tlist: List[torch.Tensor], dims: List[int]):
        max_val = [t[0, None].amax(dims) for t in tlist]
        normalizer = torch.stack(max_val, dim=-1).max(-1)[0].squeeze(0)  # remove dim of semiring
        shape = list(normalizer.shape) + [1] * len(dims)
        output = []
        for t in tlist:
            t = t.clone()
            t[0] = (t[0] - normalizer.view(shape)).exp()
            output.append(t)
        return output, normalizer

    @staticmethod
    def to_log_space(x, xn):
        x0 = (x[0] + 1e-9).log() + xn.view(list(xn.shape) + [1] * (x.ndim - xn.ndim - 1))
        return torch.stack([x0, x[1]], dim=0)


class CrossEntropySemiring(SemiringBase):
    zero = torch.tensor([-1e9, -1e9, 0.0])
    one = torch.tensor([0.0, 0.0, 0.0])
    size = 3

    @staticmethod
    def convert(a, b):
        values = a.new_zeros((3,) + a.shape)
        values[0] = a
        values[1] = b
        return values

    @staticmethod
    def unconvert(a):
        return a[2]

    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def normal_space_mul(a, b):
        # return torch.stack((a[0] * b[0], a[1] + b[1]), dim=0)
        out = torch.empty_like(a)
        torch.mul(a[0], b[0], out[0])
        torch.mul(a[1], b[1], out[1])
        torch.add(a[2], b[2], out[2])
        return out

    @staticmethod
    def sum(a, dim):
        dim = dim - 1 if dim > 0 else dim
        part_p = torch.logsumexp(a[0], dim=dim)
        part_q = torch.logsumexp(a[1], dim=dim)
        log_sm_p = a[0] - part_p.unsqueeze(dim)
        log_sm_q = a[1] - part_q.unsqueeze(dim)
        sm_p = log_sm_p.exp()
        return torch.stack((part_p, part_q, torch.sum(a[2].mul(sm_p) - log_sm_q.mul(sm_p), dim=dim)))

    @staticmethod
    def normal_space_sum(a, dim):
        dim = dim - 1 if dim > 0 else dim
        part_p = torch.sum(a[0], dim=dim)
        part_q = torch.sum(a[1], dim=dim)
        sm_p = a[0] / (part_p.unsqueeze(dim) + 1e-9)
        sm_q = a[1] / (part_q.unsqueeze(dim) + 1e-9)
        return torch.stack((part_p, part_q, torch.sum(a[2].mul(sm_p) - sm_q.log().mul(sm_p), dim=dim)))

    @staticmethod
    def to_normal_space(tlist: List[torch.Tensor], dims: List[int]):
        raise NotImplementedError
        max_val = [t[0, None].amax(dims) for t in tlist]
        normalizer = torch.stack(max_val, dim=-1).max(-1)[0].squeeze(0)  # remove dim of semiring
        shape = list(normalizer.shape) + [1] * len(dims)
        output = []
        for t in tlist:
            t = t.clone()
            t[0] = (t[0] - normalizer.view(shape)).exp()
            output.append(t)
        return output, normalizer

    @staticmethod
    def to_log_space(x, xn):
        raise NotImplementedError
        x0 = (x[0] + 1e-9).log() + xn.view(list(xn.shape) + [1] * (x.ndim - xn.ndim - 1))
        return torch.stack([x0, x[1]], dim=0)


class _SampledLogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.save_for_backward(input, torch.tensor(dim))
        return torch.logsumexp(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        logits, dim = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:

            def sample(ls):
                pre_shape = ls.shape
                draws = torch.multinomial(ls.softmax(-1).view(-1, pre_shape[-1]), 1, True)
                draws.squeeze(1)
                return torch.nn.functional.one_hot(draws, pre_shape[-1]).view(*pre_shape).type_as(ls)

            if dim == -1:
                s = sample(logits)
            else:
                dim = dim if dim >= 0 else logits.dim() + dim
                perm = [i for i in range(logits.dim()) if i != dim] + [dim]
                rev_perm = [a for a, b in sorted(enumerate(perm), key=lambda a: a[1])]
                s = sample(logits.permute(perm)).permute(rev_perm)

            grad_input = grad_output.unsqueeze(dim).mul(s)
        return grad_input, None


class _SampledSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.save_for_backward(input, torch.tensor(dim))
        return torch.sum(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        logits, dim = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:

            def sample(ls):
                pre_shape = ls.shape
                # no need to normalize
                draws = torch.multinomial(ls.view(-1, pre_shape[-1]), 1, True)
                draws.squeeze(1)
                return torch.nn.functional.one_hot(draws, pre_shape[-1]).view(*pre_shape).type_as(ls)

            if dim == -1:
                s = sample(logits)
            else:
                dim = dim if dim >= 0 else logits.dim() + dim
                perm = [i for i in range(logits.dim()) if i != dim] + [dim]
                rev_perm = [a for a, b in sorted(enumerate(perm), key=lambda a: a[1])]
                s = sample(logits.permute(perm)).permute(rev_perm)

            grad_input = grad_output.unsqueeze(dim).mul(s)
        return grad_input, None


class _SampledToNormalSpace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dims, *tlist):
        max_val = [t.amax(dims) for t in tlist]
        normalizer = torch.stack(max_val, dim=-1).max(-1)[0]
        shape = list(normalizer.shape) + [1] * len(dims)
        tlist = [(t - normalizer.view(shape)).exp() for t in tlist]
        ctx.shapes = [t.shape for t in tlist]
        return (*tlist, normalizer)

    @staticmethod
    def backward(ctx, *grad_output):
        return (None, *grad_output[:-1])


class _SampledToLogSpace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xn):
        return (x + 1e-9).log() + xn.view(list(xn.shape) + [1] * (x.ndim - xn.ndim))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _SampledMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a_inp_size = [s if s != t else -1 for s, t in zip(grad_output.shape, ctx.a_shape)]
        b_inp_size = [s if s != t else -1 for s, t in zip(grad_output.shape, ctx.b_shape)]
        return grad_output.expand(a_inp_size), grad_output.expand(b_inp_size)


class SampledSemiring(LogSemiring):
    @staticmethod
    def sum(xs, dim=-1):
        return _SampledLogSumExp.apply(xs, dim)

    @staticmethod
    def normal_space_sum(xs, dim=-1):
        return _SampledSum.apply(xs, dim)

    @staticmethod
    def normal_space_mul(a, b):
        return _SampledMul.apply(a, b)

    @staticmethod
    def to_normal_space(tlist: List[torch.Tensor], dims: List[int]):
        *tlist, normalizer = _SampledToNormalSpace.apply(dims, *tlist)
        return tlist, normalizer

    @staticmethod
    def to_log_space(x, xn):
        return _SampledToLogSpace.apply(x, xn)


def GumbelMaxSemiring(temp):
    class _GumbelMaxLogSumExp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, dim):
            ctx.save_for_backward(input, torch.tensor(dim))
            return torch.logsumexp(input, dim=dim)

        @staticmethod
        def backward(ctx, grad_output):
            logits, dim = ctx.saved_tensors
            grad_input = None
            if ctx.needs_input_grad[0]:

                def sample(ls):
                    pre_shape = ls.shape
                    update = (ls + torch.distributions.Gumbel(0, 1).sample((ls.shape[-1],))) / temp
                    out = torch.nn.functional.one_hot(update.max(-1)[1], pre_shape[-1])
                    return out

                if dim == -1:
                    s = sample(logits)
                else:
                    dim = dim if dim >= 0 else logits.dim() + dim
                    perm = [i for i in range(logits.dim()) if i != dim] + [dim]
                    rev_perm = [a for a, b in sorted(enumerate(perm), key=lambda a: a[1])]
                    s = sample(logits.permute(perm)).permute(rev_perm)

                grad_input = grad_output.unsqueeze(dim).mul(s)
            return grad_input, None

    class _GumbelMaxSemiring(LogSemiring):
        @staticmethod
        def sum(xs, dim=-1):
            return _GumbelMaxLogSumExp.apply(xs, dim)

    return _GumbelMaxSemiring


def GumbelCRFSemiring(temp):
    class ST(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, dim):
            out = torch.nn.functional.one_hot(logits.max(-1)[1], dim)
            out = out.type_as(logits)
            ctx.save_for_backward(logits, out)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            logits, out = ctx.saved_tensors
            with torch.enable_grad():
                ret = torch.autograd.grad(logits.softmax(-1), logits, out * grad_output)[0]
            return ret, None

    class _GumbelCRFLogSumExp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, dim):
            ctx.save_for_backward(input, torch.tensor(dim))
            return torch.logsumexp(input, dim=dim)

        @staticmethod
        def backward(ctx, grad_output):
            logits, dim = ctx.saved_tensors
            grad_input = None
            if ctx.needs_input_grad[0]:

                def sample(ls):
                    update = (ls + torch.distributions.Gumbel(0, 1).sample((ls.shape[-1],))) / temp
                    out = ST.apply(update, ls.shape[-1])
                    return out

                if dim == -1:
                    s = sample(logits)
                else:
                    dim = dim if dim >= 0 else logits.dim() + dim
                    perm = [i for i in range(logits.dim()) if i != dim] + [dim]
                    rev_perm = [a for a, b in sorted(enumerate(perm), key=lambda a: a[1])]
                    s = sample(logits.permute(perm)).permute(rev_perm)

                grad_input = grad_output.unsqueeze(dim).mul(s)
            return grad_input, None

    class _GumbelCRFSemiring(LogSemiring):
        @staticmethod
        def sum(xs, dim=-1):
            return _GumbelCRFLogSumExp.apply(xs, dim)

    return _GumbelCRFSemiring
