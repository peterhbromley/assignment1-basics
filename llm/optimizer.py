from collections.abc import Callable, Iterable
from typing import Optional
import torch
from torch import Tensor
import math


# Copied from assignment
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if betas[0] < 0 or betas[0] >= 1 or betas[1] < 0 or betas[1] >= 1:
            raise ValueError(f"Invalid betas: {betas}")

        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Get or initialize first and second moment vectors (m and v),
                # and timestep t. zeros_like inherits dtype and device.
                # Note that t starts at 1 for consistency with pseudocode from
                # paper.
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                # Get gradient wrt params
                grad = p.grad.data

                # Calculate adjusted learning rate for timestep t
                lr_t = lr * (math.sqrt(1 - b2**t) / (1 - b1**t))

                # Apply weight decay in place
                p.data -= lr * weight_decay * p.data

                # Update first and second moment estimates
                new_m = b1 * m + (1 - b1) * grad
                new_v = b2 * v + (1 - b2) * (grad**2)

                p.data -= lr_t * (new_m / (torch.sqrt(new_v) + eps))

                state["m"] = new_m
                state["v"] = new_v
                state["t"] = t + 1
        return loss


def gradient_clipping(params: Iterable[Tensor], max_l2_norm: float, eps: float=1e-6):
    # Convert to list in case caller uses a generator, because we need to iterate
    # over params twice.
    params = list(params)

    sum_squared_gradients = 0.0
    for p in params:
        if p.grad is None:
            continue

        sum_squared_gradients += torch.sum(torch.square(p.grad))

    l2_norm = torch.sqrt(sum_squared_gradients)
    
    if l2_norm <= max_l2_norm:
        return

    scale_factor = max_l2_norm / (l2_norm + eps)
    for p in params:
        if p.grad is None:
            continue

        p.grad *= scale_factor

    return
