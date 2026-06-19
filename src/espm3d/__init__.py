"""ESPM-3D: generalized 3-D spatial pattern matching.

The public API exposes the matcher/index classes and the synthetic data tools.
"""

from .matcher import (
    Box3D,
    EdgeSign,
    ESPM3DMatcher,
    InvertedOctreeIndex,
    MatchStats,
    PatternEdge,
    Point3D,
    SpatialObject,
    SpatialPattern,
    brute_force_match,
    max_distance_sq_boxes,
    min_distance_sq_boxes,
    min_distance_sq_point_box,
    squared_distance_points,
)

__all__ = [
    "Box3D",
    "EdgeSign",
    "ESPM3DMatcher",
    "InvertedOctreeIndex",
    "MatchStats",
    "PatternEdge",
    "Point3D",
    "SpatialObject",
    "SpatialPattern",
    "brute_force_match",
    "max_distance_sq_boxes",
    "min_distance_sq_boxes",
    "min_distance_sq_point_box",
    "squared_distance_points",
]
