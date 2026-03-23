"""Single module wrapping the full perception → planning stack."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from waymo_e2e.models.bev_encoder import BEVFeatureEncoder
from waymo_e2e.models.depth_estimator import DepthEstimator
from waymo_e2e.models.embeddings import PoseTokenEncoder, RouteTokenEncoder
from waymo_e2e.models.planner_head import LightTrajectoryHead, PlannerHead4D
from waymo_e2e.models.temporal_fusion import TemporalFusion
from waymo_e2e.models.vision_qformer import MultiViewQFormer


@dataclass
class E2EConfig:
    """Hyperparameters for ``E2EVisualPlanner`` (match ``num_views`` to dataset cameras)."""

    batch_size: int = 8
    num_views: int = 3
    num_queries_per_view: int = 16
    d_model: int = 1024
    bev_dim: int = 64
    bev_h: int = 16
    bev_w: int = 16
    image_h: int = 224
    image_w: int = 224
    # Single vision encoder: clip | dinov2 | siglip
    vision_backbone: str = "clip"
    vision_model_name: str = "openai/clip-vit-base-patch16"
    freeze_backbone: bool = True
    use_thin_depth_fusion: bool = True
    qformer_layers: int = 2
    qformer_heads: int = 16
    # Planner: "light" (default) = small Transformer + MLP; "t5" = legacy Flan-T5 encoder
    planner_type: str = "light"
    planner_model_name: str = "google/flan-t5-large"
    planner_hidden_dim: int = 1024
    planner_tf_layers: int = 2
    planner_tf_heads: int = 16
    planner_dropout: float = 0.1
    num_waypoints: int = 20
    pose_dim: int = 64
    route_dim: int = 16
    # Legacy: run frozen DPT and second ViT+BEV pass (heavy). Prefer thin fusion above.
    use_legacy_dpt_bev_branch: bool = False


class E2EVisualPlanner(nn.Module):
    """
    End-to-end stack: one vision forward per view → optional thin depth fusion on patches →
    Q-Former decoder → BEV + temporal tokens → pose/route → light trajectory head (or legacy T5).

    If ``use_legacy_dpt_bev_branch``, also runs DPT + a second full vision pass into an extra
    BEV stream (expensive; not recommended for training at scale).
    """

    def __init__(self, config: E2EConfig | None = None):
        super().__init__()
        self.config = config or E2EConfig()
        c = self.config
        nm = c.num_views * c.num_queries_per_view

        self.vision = MultiViewQFormer(
            num_views=c.num_views,
            num_queries_per_view=c.num_queries_per_view,
            vision_backbone=c.vision_backbone,
            vision_model_name=c.vision_model_name,
            d_model=c.d_model,
            num_layers=c.qformer_layers,
            num_heads=c.qformer_heads,
            freeze_backbone=c.freeze_backbone,
            use_thin_depth_fusion=c.use_thin_depth_fusion,
        )
        self.bev_encoder = BEVFeatureEncoder(
            num_views=c.num_views,
            num_queries_per_view=c.num_queries_per_view,
            query_dim=c.d_model,
            bev_dim=c.bev_dim,
            bev_h=c.bev_h,
            bev_w=c.bev_w,
        )
        self.temporal = TemporalFusion(num_positions=nm, d_model=c.d_model)
        self.pose_encoder = PoseTokenEncoder(input_dim=c.pose_dim, d_model=c.d_model)
        self.route_encoder = RouteTokenEncoder(input_dim=c.route_dim, d_model=c.d_model)
        if c.planner_type == "light":
            self.planner = LightTrajectoryHead(
                d_model=c.d_model,
                num_waypoints=c.num_waypoints,
                hidden_dim=c.planner_hidden_dim,
                num_encoder_layers=c.planner_tf_layers,
                nhead=c.planner_tf_heads,
                dropout=c.planner_dropout,
            )
        elif c.planner_type == "t5":
            self.planner = PlannerHead4D(
                model_name=c.planner_model_name,
                d_model=c.d_model,
                num_waypoints=c.num_waypoints,
                dropout_rate=c.planner_dropout,
            )
        else:
            raise ValueError(f"planner_type must be 'light' or 't5', got {c.planner_type!r}")
        self.use_legacy_dpt_bev_branch = c.use_legacy_dpt_bev_branch
        self._depth = None

    @property
    def depth_estimator(self) -> DepthEstimator | None:
        if not self.use_legacy_dpt_bev_branch:
            return None
        if self._depth is None:
            self._depth = DepthEstimator()
        return self._depth

    def forward(
        self,
        images_bvchw: torch.Tensor,
        pose_token: torch.Tensor,
        routing_token: torch.Tensor,
    ):
        """
        Args:
            images_bvchw: (B, V, 3, H, W) in [0, 1] (per-backbone normalization applied inside vision).
        """
        visual_tokens = self.vision(images_bvchw)
        bev_emb = self.bev_encoder(visual_tokens)
        time_embedded = self.temporal(visual_tokens)
        pose_emb = self.pose_encoder(pose_token)
        route_emb = self.route_encoder(routing_token)

        depth_emb = None
        if self.use_legacy_dpt_bev_branch and self.depth_estimator is not None:
            with torch.no_grad():
                depth_maps = self.depth_estimator.estimate_depth_batch(images_bvchw)
            depth_maps = depth_maps.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            depth_tokens = self.vision(depth_maps)
            depth_emb = self.bev_encoder(depth_tokens)

        return self.planner(bev_emb, time_embedded, pose_emb, route_emb, depth_emb)
