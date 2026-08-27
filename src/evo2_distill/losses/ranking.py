from __future__ import annotations

import torch


def pairwise_logistic_ranking_loss(
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
) -> torch.Tensor:
    direction = torch.sign(target_a - target_b)
    valid = direction.ne(0)
    if not torch.any(valid):
        return prediction_a.new_zeros(())
    margin = direction[valid] * (prediction_a[valid] - prediction_b[valid])
    return torch.nn.functional.softplus(-margin).mean()


def gap_weighted_pairwise_ranking_loss(
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    gap_weight: float = 2.0,
    maximum_weight: float = 6.0,
) -> torch.Tensor:
    """Pairwise logistic loss emphasizing consequential target rank gaps."""
    difference = target_a - target_b
    valid = difference.ne(0)
    if not torch.any(valid):
        return prediction_a.new_zeros(())
    difference = difference[valid]
    margin = torch.sign(difference) * (prediction_a[valid] - prediction_b[valid])
    gaps = difference.abs().detach()
    scale = gaps.median().clamp_min(torch.finfo(gaps.dtype).eps)
    weights = (1.0 + float(gap_weight) * gaps / scale).clamp_max(float(maximum_weight))
    losses = torch.nn.functional.softplus(-margin)
    return (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
