"""
Waymo end-to-end visual planning experiments (CLIP multi-view → BEV → temporal → planner).

Public API: use ``from waymo_e2e import E2EVisualPlanner`` or import submodules under
``waymo_e2e.models`` and ``waymo_e2e.data``.
"""

from waymo_e2e.pipeline import E2EVisualPlanner

__all__ = ["E2EVisualPlanner", "__version__"]
__version__ = "0.1.0"
