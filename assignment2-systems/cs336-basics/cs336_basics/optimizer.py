
import torch.nn as nn
import einops
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

from sympy.physics.quantum.identitysearch import lr_op


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

                state = self.state[p]          # Get state associated with p.
                t = state.get("t", 0)           # Get iteration number from the state.
                grad = p.grad.data              # Gradient of loss w.r.t. p.

                # Parameter update
                p.data -= lr / math.sqrt(t + 1) * grad

                # Update state
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Hyperparameters
            alpha = group['lr']
            beta1, beta2 = group['betas']
            epsilon = group['eps']
            lambda_val = group['weight_decay']

            for theta in group['params']:
                if theta.grad is None:
                    continue

                # g: Gradient
                g = theta.grad

                # Initialize state if this is the first step
                state = self.state[theta]
                if len(state) == 0:
                    state['step'] = 0
                    # m = 0 (Initial value of first moment)
                    state['m'] = torch.zeros_like(theta, memory_format=torch.preserve_format)
                    # v = 0 (Initial value of second moment)
                    state['v'] = torch.zeros_like(theta, memory_format=torch.preserve_format)

                # Load m and v from state
                m = state['m']
                v = state['v']
                state['step'] += 1
                t = state['step']

                # --- 1. Update the first moment estimate m ---
                # m <- beta1 * m + (1 - beta1) * g
                m = beta1 * m + (1 - beta1) * g

                # --- 2. Update the second moment estimate v ---
                # v <- beta2 * v + (1 - beta2) * g^2
                v = beta2 * v + (1 - beta2) * (g * g)

                # Save updated m and v back to state for next iteration
                state['m'] = m
                state['v'] = v

                # --- 3. Compute adjusted alpha for iteration t ---
                # alpha_t <- alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
                bias_correction2 = 1 - beta2 ** t
                bias_correction1 = 1 - beta1 ** t
                alpha_t = alpha * (bias_correction2 ** 0.5) / bias_correction1

                # --- 4. Update the parameters ---
                # theta <- theta - alpha_t * m / (sqrt(v) + epsilon)
                update = alpha_t * m / (torch.sqrt(v) + epsilon)
                theta.data = theta.data - update

                # --- 5. Apply weight decay ---
                # theta <- theta - alpha * lambda * theta
                if lambda_val > 0:
                    theta.data = theta.data - (alpha * lambda_val * theta.data)

        return loss

def lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
) -> float:
    lr_cosine = 0.0
    if it < warmup_iters:
        # Linear warmup
        lr_cosine = max_learning_rate * (it / warmup_iters)
    if warmup_iters <= it & it <= cosine_cycle_iters:
        # Cosine decay
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        lr_cosine = min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    if it > cosine_cycle_iters:
        lr_cosine = min_learning_rate
    return lr_cosine


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """
    Clips gradients of an iterable of parameters at max_l2_norm.
    The gradients are modified in-place.
    """
    # 1. Filter parameters that have gradients
    # We convert to a list to iterate multiple times (once for norm, once for scaling)
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return

    # 2. Compute the global L2 norm
    # formula: sqrt( sum( |param.grad|^2 ) )
    # We iterate over all params, compute the sum of squares for each, sum those up, then take sqrt.
    total_norm_sq = torch.sum(torch.stack([torch.sum(p.grad ** 2) for p in params]))
    total_norm = torch.sqrt(total_norm_sq)

    # 3. Clip gradients if necessary
    # "If this norm is less than M, then we leave g as is; otherwise..."
    if total_norm > max_l2_norm:
        # Scale factor = M / (||g|| + epsilon)
        scale_factor = max_l2_norm / (total_norm + eps)

        # Apply scaling in-place
        for p in params:
            p.grad.detach().mul_(scale_factor)