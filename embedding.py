"""Compatibility shim — use ``waymo_e2e.models.embeddings``."""

from waymo_e2e.models.embeddings import (
    PoseEmbedding,
    PoseTokenEncoder,
    RouteEmbedding,
    RouteTokenEncoder,
)

__all__ = ["PoseEmbedding", "PoseTokenEncoder", "RouteEmbedding", "RouteTokenEncoder"]
