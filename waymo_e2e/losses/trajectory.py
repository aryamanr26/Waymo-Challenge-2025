"""Trajectory supervision losses for future waypoints."""

import torch
import torch.nn.functional as F


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


def past_path_proximity_loss(
    pred_xy: torch.Tensor,
    past_xy: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Soft collision proxy: penalise predicted waypoints that pass within ``margin`` (m) of the
    ego past polyline in XY (dataset frame). Invalid padded past rows are ignored.
    """
    # pred_xy: (B, T, 2), past_xy: (B, P, 2)
    d = torch.norm(pred_xy.unsqueeze(2) - past_xy.unsqueeze(1), dim=-1)
    valid_past = past_xy.norm(dim=-1) > 1e-3
    d = d.masked_fill(~valid_past.unsqueeze(1), float("inf"))
    min_d, _ = d.min(dim=-1)
    finite = torch.isfinite(min_d)
    pen = F.relu(margin - min_d) * finite.float()
    return pen.sum() / finite.float().sum().clamp(min=1.0)


def trajectory_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    past_xy: torch.Tensor | None = None,
    ade_weight: float = 1.0,
    fde_weight: float = 0.8,
    heading_weight: float = 0.3,
    smooth_weight: float = 0.1,
    collision_weight: float = 0.15,
) -> torch.Tensor:
    """
    Comprehensive trajectory loss combining:
      - Weighted ADE (L2 per waypoint, progressive near-future emphasis)
      - FDE (final endpoint error)
      - Heading consistency (penalise abrupt direction changes in XY plane)
      - Velocity smoothness (penalise acceleration jerk)

    Args:
        predictions: (B, T, 3)
        targets:     (B, T, 3)
        mask:        optional (B, T) bool — True where waypoint is valid
    """
    B, T, _ = predictions.shape

    # --- ADE with progressive temporal weights (near future matters more) ---
    # Weight decays linearly: t=0 gets weight T, t=T-1 gets weight 1
    time_weights = torch.linspace(T, 1, T, device=predictions.device)  # (T,)
    time_weights = time_weights / time_weights.sum()
    l2_dist = torch.norm(predictions - targets, dim=-1)  # (B, T)
    if mask is not None:
        ade = (l2_dist * time_weights * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    else:
        ade = (l2_dist * time_weights).sum(dim=-1)
    ade = ade.mean()

    # --- FDE ---
    fde = torch.norm(predictions[:, -1] - targets[:, -1], dim=-1).mean()

    # --- Heading consistency (XY plane only) ---
    # Displacement vectors between consecutive predicted waypoints
    disp = predictions[:, 1:, :2] - predictions[:, :-1, :2]   # (B, T-1, 2)
    disp_norm = F.normalize(disp, dim=-1, eps=1e-6)            # (B, T-1, 2)
    # Cosine similarity between consecutive direction vectors
    cos_sim = (disp_norm[:, :-1] * disp_norm[:, 1:]).sum(-1)   # (B, T-2)
    # Only penalise where the vehicle is actually moving (avoid instability at near-zero disp)
    moving = (disp[:, :-1].norm(dim=-1) > 0.01) & (disp[:, 1:].norm(dim=-1) > 0.01)
    heading_loss = ((1.0 - cos_sim) * moving.float()).sum() / moving.float().sum().clamp(min=1.0)

    # --- Velocity smoothness (acceleration magnitude) ---
    vel = predictions[:, 1:] - predictions[:, :-1]   # (B, T-1, 3)
    acc = vel[:, 1:] - vel[:, :-1]                   # (B, T-2, 3)
    smooth_loss = acc.norm(dim=-1).mean()

    total = (
        ade_weight * ade
        + fde_weight * fde
        + heading_weight * heading_loss
        + smooth_weight * smooth_loss
    )
    if past_xy is not None and collision_weight > 0:
        col = past_path_proximity_loss(predictions[..., :2], past_xy)
        total = total + collision_weight * col
    return total
