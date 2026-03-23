"""Single forward through a vision backbone + optional thin depth fusion on patches (no second ViT)."""

from __future__ import annotations

import torch
import torch.nn as nn

# --- ImageNet normalization (DINOv2) ---
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

# OpenAI CLIP / many SigLIP checkpoints
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def _move_stats(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)


def normalize_images(images_bvchw: torch.Tensor, backbone: str) -> torch.Tensor:
    """
    images_bvchw: (B, V, 3, H, W) in [0, 1].
    Returns tensor ready for ``pixel_values`` (normalized).
    """
    b = backbone.lower()
    x = images_bvchw
    if b in ("clip", "siglip"):
        mean = _move_stats(_CLIP_MEAN, x)
        std = _move_stats(_CLIP_STD, x)
        return (x - mean) / std
    if b == "dinov2":
        mean = _move_stats(_IMAGENET_MEAN, x)
        std = _move_stats(_IMAGENET_STD, x)
        return (x - mean) / std
    raise ValueError(f"Unknown backbone '{backbone}' for normalization")


class ThinDepthFusion(nn.Module):
    """
    Lightweight head on patch tokens: predicts depth-style features and adds them (late fusion).
    Single MLP path — no extra ViT / DPT forward.
    """

    def __init__(self, d_model: int, bottleneck: int | None = None):
        super().__init__()
        b = bottleneck or max(d_model // 2, 64)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, b),
            nn.GELU(),
            nn.Linear(b, d_model),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: (B, V, P, D)
        return patches + self.scale * self.mlp(patches)


def _load_backbone(backbone: str, model_name: str) -> tuple[nn.Module, int]:
    b = backbone.lower()
    if b == "clip":
        from transformers import CLIPVisionModel

        m = CLIPVisionModel.from_pretrained(model_name)
        return m, int(m.config.hidden_size)
    if b == "dinov2":
        from transformers import Dinov2Model

        m = Dinov2Model.from_pretrained(model_name)
        return m, int(m.config.hidden_size)
    if b == "siglip":
        from transformers import SiglipVisionModel

        m = SiglipVisionModel.from_pretrained(model_name)
        return m, int(m.config.hidden_size)
    raise ValueError(f"vision_backbone must be clip|dinov2|siglip, got {backbone!r}")


def _backbone_forward(module: nn.Module, backbone: str, pixel_values: torch.Tensor) -> torch.Tensor:
    """Returns patch tokens (B*V, P, H) without CLS."""
    b = backbone.lower()
    if b == "clip":
        out = module(pixel_values=pixel_values).last_hidden_state
    elif b == "dinov2":
        out = module(pixel_values=pixel_values).last_hidden_state
    elif b == "siglip":
        out = module(pixel_values=pixel_values).last_hidden_state
    else:
        raise ValueError(backbone)
    return out[:, 1:, :]


class MultiViewQFormer(nn.Module):
    """
    One frozen (by default) vision encoder + optional projection to ``d_model`` +
    optional thin depth fusion on patches + per-camera embeddings +
    learnable queries with TransformerDecoder over multi-view memory.
    """

    def __init__(
        self,
        num_views: int = 3,
        num_queries_per_view: int = 16,
        vision_backbone: str = "clip",
        vision_model_name: str = "openai/clip-vit-base-patch16",
        d_model: int = 1024,
        num_layers: int = 2,
        num_heads: int = 16,
        d_ff: int | None = None,
        freeze_backbone: bool = True,
        use_thin_depth_fusion: bool = True,
    ):
        super().__init__()
        self.num_views = num_views
        self.vision_backbone = vision_backbone
        self.d_model = d_model

        self.backbone, backbone_dim = _load_backbone(vision_backbone, vision_model_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.patch_proj = nn.Linear(backbone_dim, d_model) if backbone_dim != d_model else nn.Identity()
        self.thin_depth_fusion = ThinDepthFusion(d_model) if use_thin_depth_fusion else None

        if d_ff is None:
            d_ff = 4 * d_model
        self.camera_id_emb = nn.Embedding(num_views, d_model)
        self.num_queries = num_views * num_queries_per_view
        self.queries = nn.Parameter(torch.randn(self.num_queries, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            activation="gelu",
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, V, 3, H, W) in [0, 1]
        B, V, C, H, W = images.shape
        x = normalize_images(images, self.vision_backbone)
        x = x.view(B * V, C, H, W)
        patches = _backbone_forward(self.backbone, self.vision_backbone, x)
        patches = self.patch_proj(patches)
        P = patches.size(1)
        patches = patches.view(B, V, P, self.d_model)
        if self.thin_depth_fusion is not None:
            patches = self.thin_depth_fusion(patches)
        cam_ids = torch.arange(V, device=images.device)
        cam_emb = self.camera_id_emb(cam_ids)
        patches = patches + cam_emb.unsqueeze(0).unsqueeze(2)
        memory = patches.view(B, V * P, self.d_model).permute(1, 0, 2)
        queries = self.queries.unsqueeze(1).expand(-1, B, -1)
        q_out = self.decoder(tgt=queries, memory=memory)
        return q_out.permute(1, 0, 2)
