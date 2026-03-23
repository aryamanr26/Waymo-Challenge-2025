"""Compatibility shim — use ``waymo_e2e.models.planner_head``."""

from waymo_e2e.models.planner_head import LightTrajectoryHead, PlannerHead3D, PlannerHead4D

__all__ = ["LightTrajectoryHead", "PlannerHead3D", "PlannerHead4D"]
