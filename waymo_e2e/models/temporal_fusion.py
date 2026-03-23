import torch
import torch.nn as nn


class TemporalFusion(nn.Module):
    """Learned additive embedding per (camera × token) slot (positional / weak temporal prior)."""

    def __init__(self, num_positions: int, d_model: int):
        super().__init__()
        self.time_embeddings = nn.Parameter(torch.zeros(num_positions, d_model))
        nn.init.trunc_normal_(self.time_embeddings, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 2:
            return tokens + self.time_embeddings
        if tokens.dim() == 3:
            return tokens + self.time_embeddings.unsqueeze(0)
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tokens.shape}")
