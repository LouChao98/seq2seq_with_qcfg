import logging
import math
from dataclasses import dataclass

from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

log = logging.getLogger(__name__)


@dataclass
class _CosineSchedulerLambda:
    num_warmup_steps: int
    num_training_steps: int
    num_cycles: int

    def __call__(self, current_step):

        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)),
        )


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    return LambdaLR(
        optimizer,
        _CosineSchedulerLambda(num_warmup_steps, num_training_steps, num_cycles),
        last_epoch,
    )


def get_exponential_lr_scheduler(optimizer, gamma, **kwargs):
    if isinstance(gamma, str):
        gamma = eval(gamma)
        log.debug(f"gamma is converted to {gamma} {type(gamma)}")
    kwargs["gamma"] = gamma
    return lr_scheduler.ExponentialLR(optimizer, **kwargs)


def get_reduce_lr_on_plateau_scheduler(optimizer, **kwargs):
    return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)


def get_lr_lambda_scheduler(optimizer, lr_lambda, **kwargs):
    if isinstance(lr_lambda, str):
        lr_lambda = eval(lr_lambda)
        log.debug(f"lr_lambda is converted to {lr_lambda} {type(lr_lambda)}")
    kwargs["lr_lambda"] = lr_lambda
    return lr_scheduler.LambdaLR(optimizer, **kwargs)
