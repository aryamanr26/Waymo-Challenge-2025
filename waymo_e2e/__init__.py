"""
Waymo end-to-end visual planning experiments (CLIP multi-view → BEV → temporal → planner).

Public API: use ``from waymo_e2e import E2EVisualPlanner`` or import submodules under
``waymo_e2e.models`` and ``waymo_e2e.data``.
"""

from importlib import import_module

__all__ = ["E2EVisualPlanner", "baseline", "autovla", "__version__"]
__version__ = "0.1.0"


def __getattr__(name):
    # Lazy exports avoid importing heavy vision/depth deps on lightweight paths
    # like ``from waymo_e2e.data import ...``.
    if name == "E2EVisualPlanner":
        return import_module("waymo_e2e.pipeline").E2EVisualPlanner
    if name == "baseline":
        return import_module("waymo_e2e.architectures.baseline")
    if name == "autovla":
        return import_module("waymo_e2e.architectures.autovla")
    raise AttributeError(f"module 'waymo_e2e' has no attribute {name!r}")
