import math
import torch
import torch.nn as nn
import einops


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-entropy loss between logits and targets.

    Args:
        inputs (torch.Tensor): The predicted logits with shape (batch_size, vocab_size).
        targets (torch.Tensor): The ground truth labels with shape (batch_size,).

    Returns:
        torch.Tensor: The computed cross-entropy loss.
    """
    # 1. Numerical Stability: Subtract max value
    row_max = einops.reduce(inputs, 'b v -> b 1', 'max')
    logits_shifted = inputs - row_max

    # 2. Compute log-sum-exp
    exp_logits = torch.exp(logits_shifted)
    sum_exp = einops.reduce(exp_logits, 'b v -> b 1', 'sum')
    log_sum_exp = torch.log(sum_exp)

    # 3. Gather the logits corresponding to the target classes
    target_logits = logits_shifted.gather(1, targets.unsqueeze(1))

    # 4. Compute Loss
    # Loss = -log(P(target))
    #      = -log(exp(logit_target) / sum(exp(logits)))
    #      = -(logit_target - logsumexp)
    #      = logsumexp - logit_target
    loss = log_sum_exp - target_logits

    return loss.mean()  # Return mean loss over the batch

def perplexity_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the perplexity loss between logits and targets.

    Args:
        inputs (torch.Tensor): The predicted logits with shape (batch_size, vocab_size).
        targets (torch.Tensor): The ground truth labels with shape (batch_size,).

    Returns:
        torch.Tensor: The computed perplexity loss.
    """
    ce_loss = cross_entropy_loss(inputs, targets)
    perplexity = torch.exp(ce_loss)
    return perplexity

