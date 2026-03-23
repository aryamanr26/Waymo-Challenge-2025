"""Compatibility shim — use ``waymo_e2e.models.depth_estimator``."""

from waymo_e2e.models.depth_estimator import DepthEstimator, load_waymo_batch, plot_front_3_camera_images

__all__ = ["DepthEstimator", "load_waymo_batch", "plot_front_3_camera_images"]
