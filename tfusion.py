# tfusion.py

import torch
import torch.nn as nn

class TemporalFusion(nn.Module):
    """
    Temporal Fusion Module

    Adds a learned time embedding to each visual token position so that
    self-attention can recover temporal order and motion cues.

    Args:
        num_positions (int): Total number of token positions per sample
                             (e.g., N_cameras × M_tokens_per_camera).
        d_model (int):       Dimensionality of token embeddings.
    """
    def __init__(self, num_positions: int, d_model: int):
        super(TemporalFusion, self).__init__()
        # One learnable embedding per (camera×token) position across time
        self.time_embeddings = nn.Parameter(torch.zeros(num_positions, d_model))
        # Initialize like ViT: truncated normal
        nn.init.trunc_normal_(self.time_embeddings, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: torch.Tensor of shape (B, num_positions, d_model) or
                    (num_positions, d_model) if no batch dimension.

        Returns:
            torch.Tensor of the same shape, with time embeddings added.
        """
        if tokens.dim() == 2:
            # No batch dim: (num_positions, d_model)
            return tokens + self.time_embeddings
        elif tokens.dim() == 3:
            # Batched: (B, num_positions, d_model)
            return tokens + self.time_embeddings.unsqueeze(0)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tokens.shape}")

if __name__ == "__main__":
    # Example usage & shape printing
    
    # Suppose after your vision_encoder you have:
    # final_tokens: torch.Tensor of shape (B, N·M, D_t)
    B, NM, D_t = 4, 128, 1024
    final_tokens = torch.randn(B, NM, D_t)

    # Instantiate TemporalFusion
    tf_module = TemporalFusion(num_positions=NM, d_model=D_t)

    # Stage 1: Input
    print("5. Temporal Fusion Module")
    print("Input:")
    print(f"  Visual tokens projected to text dim: {final_tokens.shape}")  
    # e.g. torch.Size([4, 128, 1024])

    # Stage 2: Internal
    time_embedded = tf_module(final_tokens)
    print("Internal:")
    print(f"  Add learned time embeddings: {time_embedded.shape}")  
    # torch.Size([4, 128, 1024])

    # Stage 3: Output
    print("Output:")
    print(f"  Time-embedded tokens: {time_embedded.shape}")  
    # torch.Size([4, 128, 1024])
