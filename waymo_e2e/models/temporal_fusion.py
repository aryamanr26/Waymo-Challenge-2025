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


class TemporalCrossAttention(nn.Module):
    """
    Fuse a short history of Q-Former outputs: last timestep queries attend to all timesteps × slots.
    Adds per-slot bias (same role as the old ``TemporalFusion`` embedding for the planner branch).
    """

    def __init__(
        self,
        num_token_slots: int,
        d_model: int,
        num_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.slot_emb = nn.Parameter(torch.zeros(num_token_slots, d_model))
        nn.init.trunc_normal_(self.slot_emb, std=0.02)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, D) single frame, or (B, T, N, D) with T history frames (current = last).
        """
        if tokens.dim() == 3:
            return tokens + self.slot_emb.unsqueeze(0)
        if tokens.dim() != 4:
            raise ValueError(f"Expected (B,N,D) or (B,T,N,D), got {tokens.shape}")
        B, T, N, D = tokens.shape
        q = tokens[:, -1]
        kv = tokens.reshape(B, T * N, D)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        out = self.norm(q + self.dropout(attn_out))
        return out + self.slot_emb.unsqueeze(0)
