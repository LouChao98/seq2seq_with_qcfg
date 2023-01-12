import torch


class _UniDirDifferentiableClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min=None, max=None):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        ctx.min = min if min is not None else -1e12
        ctx.max = max if max is not None else 1e12
        return input.clamp(min=min, max=max)

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (input,) = ctx.saved_tensors

        lower = input <= ctx.min
        upper = input >= ctx.max

        grad_output = torch.where(lower, grad_output.clamp(max=0), grad_output)
        grad_output = torch.where(upper, grad_output.clamp(0), grad_output)
        return grad_output, None, None


def uni_dir_differentiable_clamp(a: torch.Tensor, min, max):
    return a.detach().clamp(min=min, max=max) + a - a.detach()
    return _UniDirDifferentiableClamp.apply(a, min, max)


if __name__ == "__main__":
    a = torch.randn(5, requires_grad=True)
    out = _UniDirDifferentiableClamp.apply(a, -0.1, 0.1)
    out.sum().backward()

    print(a)
    print(a.grad)
