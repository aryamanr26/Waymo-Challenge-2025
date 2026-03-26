from importlib import import_module

__all__ = [
    "BEVFeatureEncoder",
    "DepthEstimator",
    "MultiViewQFormer",
    "TemporalFusion",
    "TemporalCrossAttention",
    "PoseEmbedding",
    "PoseTokenEncoder",
    "RouteEmbedding",
    "RouteTokenEncoder",
    "LightTrajectoryHead",
    "PlannerHead3D",
    "PlannerHead4D",
    "ThinDepthFusion",
    "normalize_images",
]


_EXPORT_TO_MODULE = {
    "BEVFeatureEncoder": ("waymo_e2e.models.bev_encoder", "BEVFeatureEncoder"),
    "DepthEstimator": ("waymo_e2e.models.depth_estimator", "DepthEstimator"),
    "PoseEmbedding": ("waymo_e2e.models.embeddings", "PoseEmbedding"),
    "PoseTokenEncoder": ("waymo_e2e.models.embeddings", "PoseTokenEncoder"),
    "RouteEmbedding": ("waymo_e2e.models.embeddings", "RouteEmbedding"),
    "RouteTokenEncoder": ("waymo_e2e.models.embeddings", "RouteTokenEncoder"),
    "LightTrajectoryHead": ("waymo_e2e.models.planner_head", "LightTrajectoryHead"),
    "PlannerHead3D": ("waymo_e2e.models.planner_head", "PlannerHead3D"),
    "PlannerHead4D": ("waymo_e2e.models.planner_head", "PlannerHead4D"),
    "TemporalCrossAttention": ("waymo_e2e.models.temporal_fusion", "TemporalCrossAttention"),
    "TemporalFusion": ("waymo_e2e.models.temporal_fusion", "TemporalFusion"),
    "MultiViewQFormer": ("waymo_e2e.models.vision_qformer", "MultiViewQFormer"),
    "ThinDepthFusion": ("waymo_e2e.models.vision_qformer", "ThinDepthFusion"),
    "normalize_images": ("waymo_e2e.models.vision_qformer", "normalize_images"),
}


def __getattr__(name):
    if name not in _EXPORT_TO_MODULE:
        raise AttributeError(f"module 'waymo_e2e.models' has no attribute {name!r}")
    module_name, symbol = _EXPORT_TO_MODULE[name]
    return getattr(import_module(module_name), symbol)
