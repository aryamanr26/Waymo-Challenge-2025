"""Trajectory supervision (ADE / FDE) for future waypoints."""

import torch


def ade_fde_loss(predictions, targets, mask=None, alpha=1.0, beta=0.8):
    """
    Args:
        predictions: (B, T, 3)
        targets: (B, T, 3)
        mask: optional (B, T) bool
    """
    l2_dist = torch.norm(predictions - targets, dim=-1)
    if mask is not None:
        ade = (l2_dist * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    else:
        ade = l2_dist.mean(dim=-1)
    final_pred = predictions[:, -1]
    final_target = targets[:, -1]
    fde = torch.norm(final_pred - final_target, dim=-1)
    return (alpha * ade.mean()) + (beta * fde.mean())
