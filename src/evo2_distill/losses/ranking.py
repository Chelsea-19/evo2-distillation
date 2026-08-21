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

