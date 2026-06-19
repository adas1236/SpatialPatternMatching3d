"""Synthetic 3-D data and pattern generation for ESPM-3D.

This module is intentionally compatible with ``espm3d.py``.  It can:

* build a controllable synthetic 3-D spatio-textual object set;
* return a suite of named 3-D spatial patterns useful for benchmarking;
* optionally plant one or more guaranteed matches for those patterns; and
* save/load objects and patterns as simple JSON files.

The generator is useful in two modes:

1. ``ensure_patterns=True``: plant known matches for correctness tests and
   demos.  Exclusion constraints are protected by keyword-specific avoid zones.
2. ``ensure_patterns=False``: generate background-only data for selectivity and
   scalability experiments where some patterns may have zero matches.

The implementation uses only the Python standard library plus the local
``espm3d`` module.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .matcher import (
    Box3D,
    EdgeSign,
    ESPM3DMatcher,
    InvertedOctreeIndex,
    PatternEdge,
    Point3D,
    SpatialObject,
    SpatialPattern,
    VertexId,
    squared_distance_points,
)

EPS = 1e-12
DEFAULT_BOUNDS = Box3D((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0))


@dataclass(frozen=True)
class NamedPattern:
    """A spatial pattern plus a name, description, and optional planting layout.

    ``template_points`` are local 3-D coordinates for the pattern vertices.  They
    are used only to plant guaranteed matches.  Rotation and translation preserve
    distances, so a template that satisfies the pattern remains valid after it is
    placed in the global data volume.
    """

    name: str
    pattern: SpatialPattern
    description: str = ""
    template_points: Mapping[VertexId, Point3D] = field(default_factory=dict)


@dataclass(frozen=True)
class PlantedMatch:
    """The known vertex-to-object assignment for a planted match."""

    pattern_name: str
    match_index: int
    assignment: Mapping[VertexId, str]


@dataclass(frozen=True)
class SyntheticDataset:
    """Result returned by ``generate_synthetic_3d_dataset``."""

    objects: List[SpatialObject]
    patterns: List[NamedPattern]
    planted_matches: List[PlantedMatch]
    bounds: Box3D
    seed: int
    metadata: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Pattern suite
# ---------------------------------------------------------------------------


def _dist(a: Point3D, b: Point3D) -> float:
    return math.sqrt(squared_distance_points(a, b))


def _edge(
    points: Mapping[VertexId, Point3D],
    source: VertexId,
    target: VertexId,
    lower: float,
    upper: float,
    sign: str = "--",
) -> PatternEdge:
    d = _dist(points[source], points[target])
    if not (lower - 1e-9 <= d <= upper + 1e-9):
        raise ValueError(
            f"template edge {source!r}-{target!r} has distance {d:.3f}, "
            f"outside [{lower}, {upper}]"
        )
    return PatternEdge(source, target, lower, upper, sign)


def _named(
    name: str,
    description: str,
    vertices: Mapping[VertexId, str],
    points: Mapping[VertexId, Point3D],
    edge_specs: Sequence[Tuple[VertexId, VertexId, float, float, str]],
) -> NamedPattern:
    edges = tuple(_edge(points, s, t, lo, hi, sign) for s, t, lo, hi, sign in edge_specs)
    pattern = SpatialPattern(dict(vertices), edges)
    _assert_template_satisfies(pattern, points)
    return NamedPattern(name=name, description=description, pattern=pattern, template_points=dict(points))


def default_pattern_suite() -> List[NamedPattern]:
    """Return 20 named 3-D spatial patterns for testing ESPM-3D.

    The patterns mix chains, stars, triangles, cycles, diamonds, tetrahedra,
    vertical layouts, and edges with exclusion semantics.  Units are arbitrary;
    the default generator uses a 1000 x 1000 x 1000 box.
    """

    patterns: List[NamedPattern] = []

    patterns.append(
        _named(
            "p01_residential_area",
            "House near a park, with a station close but not too close to the house.",
            {"H": "house", "P": "park", "S": "station"},
            {"H": (0, 0, 0), "P": (12, 3, 0), "S": (42, 0, 6)},
            [
                ("H", "P", 0, 20, "--"),
                ("H", "S", 30, 65, "->"),
            ],
        )
    )

    patterns.append(
        _named(
            "p02_trip_planning_loop",
            "Hotel, airport, museum, gallery, and beach arranged as a small itinerary loop.",
            {"H": "hotel", "A": "airport", "M": "museum", "G": "gallery", "B": "beach"},
            {"H": (0, 0, 0), "A": (120, 20, 10), "M": (25, 0, 0), "G": (50, 5, 8), "B": (74, 10, 5)},
            [
                ("H", "A", 80, 160, "--"),
                ("H", "M", 0, 40, "--"),
                ("M", "G", 0, 35, "--"),
                ("G", "B", 0, 35, "--"),
                ("B", "H", 60, 95, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p03_human_settlement_triangle",
            "Office, house, and waterworks in three separated 3-D regions.",
            {"O": "office", "H": "house", "W": "waterworks"},
            {"O": (0, 0, 20), "H": (90, 10, 0), "W": (50, 90, 10)},
            [
                ("O", "H", 70, 130, "--"),
                ("O", "W", 70, 140, "--"),
                ("H", "W", 60, 130, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p04_logistics_chain",
            "Warehouse, rail, port, customs, and airport along a freight corridor.",
            {"W": "warehouse", "R": "rail", "P": "port", "A": "airport", "C": "customs"},
            {"W": (0, 0, 0), "R": (35, 5, 0), "P": (80, 15, -5), "A": (135, 35, 30), "C": (95, 40, 5)},
            [
                ("W", "R", 20, 50, "--"),
                ("R", "P", 30, 70, "--"),
                ("P", "A", 50, 95, "--"),
                ("P", "C", 20, 55, "--"),
                ("C", "A", 35, 70, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p05_emergency_star",
            "Hospital hub with nearby services and emergency facilities kept at a safer distance.",
            {"H": "hospital", "A": "ambulance", "P": "pharmacy", "F": "fire_station", "L": "police"},
            {"H": (0, 0, 0), "A": (8, 4, 0), "P": (20, -5, 0), "F": (60, 0, 0), "L": (42, 30, 6)},
            [
                ("H", "A", 0, 15, "--"),
                ("H", "P", 0, 30, "--"),
                ("H", "F", 45, 80, "->"),
                ("H", "L", 40, 70, "<->"),
            ],
        )
    )

    patterns.append(
        _named(
            "p06_campus_hub",
            "Dorm-centered campus amenities with parking not directly adjacent to the dorm.",
            {"D": "dorm", "C": "cafe", "L": "library", "B": "lab", "G": "gym", "P": "parking"},
            {"D": (0, 0, 0), "C": (10, 4, 0), "L": (22, 0, 4), "B": (35, 12, 8), "G": (15, 25, 10), "P": (55, 0, 0)},
            [
                ("D", "C", 0, 18, "--"),
                ("C", "L", 0, 20, "--"),
                ("L", "B", 10, 25, "--"),
                ("D", "G", 10, 35, "--"),
                ("D", "P", 35, 75, "->"),
            ],
        )
    )

    patterns.append(
        _named(
            "p07_retail_diamond",
            "Mall, restaurant, cinema, parking, and transit in a diamond-like layout.",
            {"M": "mall", "R": "restaurant", "C": "cinema", "P": "parking", "T": "transit"},
            {"M": (0, 0, 0), "R": (15, 0, 0), "C": (0, 20, 5), "P": (35, 5, 0), "T": (60, 0, 5)},
            [
                ("M", "R", 0, 25, "--"),
                ("M", "C", 0, 30, "--"),
                ("R", "P", 10, 35, "--"),
                ("C", "P", 20, 45, "--"),
                ("P", "T", 15, 40, "--"),
                ("M", "T", 45, 75, "->"),
            ],
        )
    )

    patterns.append(
        _named(
            "p08_sensor_tetrahedron",
            "Three sensors and a relay forming a compact tetrahedral monitoring cell.",
            {"S1": "sensor_a", "S2": "sensor_b", "S3": "sensor_c", "R": "relay"},
            {"S1": (0, 0, 0), "S2": (30, 0, 0), "S3": (15, 25, 0), "R": (15, 10, 28)},
            [
                ("S1", "S2", 20, 45, "--"),
                ("S1", "S3", 20, 45, "--"),
                ("S2", "S3", 20, 45, "--"),
                ("S1", "R", 20, 45, "--"),
                ("S2", "R", 20, 45, "--"),
                ("S3", "R", 20, 45, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p09_cave_vertical_chain",
            "Subsurface entrance, drill, sensor, pump, and vent with vertical separation.",
            {"E": "cave_entrance", "D": "drill", "S": "seismic_sensor", "P": "pump", "V": "vent"},
            {"E": (0, 0, 0), "D": (20, 0, -20), "S": (35, 10, -45), "P": (55, 5, -60), "V": (10, 30, -30)},
            [
                ("E", "D", 20, 40, "--"),
                ("D", "S", 20, 40, "--"),
                ("S", "P", 20, 35, "--"),
                ("D", "V", 25, 50, "--"),
                ("V", "S", 25, 45, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p10_drone_corridor",
            "Depot-to-target corridor through two waypoints and a charger, with a no-fly marker not too close.",
            {"D": "depot", "W1": "waypoint_a", "W2": "waypoint_b", "C": "charger", "T": "target", "N": "no_fly"},
            {"D": (0, 0, 0), "W1": (35, 10, 20), "W2": (70, 5, 45), "C": (95, 30, 60), "T": (130, 40, 70), "N": (60, 35, 40)},
            [
                ("D", "W1", 35, 50, "--"),
                ("W1", "W2", 35, 55, "--"),
                ("W2", "C", 30, 50, "--"),
                ("C", "T", 30, 55, "--"),
                ("W2", "N", 20, 60, "->"),
            ],
        )
    )

    patterns.append(
        _named(
            "p11_utility_vertical",
            "Power infrastructure with transformer, substation, tower, battery, and control room.",
            {"T": "transformer", "S": "substation", "W": "tower", "B": "battery", "C": "control_room"},
            {"T": (0, 0, 0), "S": (0, 0, 35), "W": (30, 0, 80), "B": (-25, 10, 55), "C": (5, 40, 30)},
            [
                ("T", "S", 25, 50, "--"),
                ("S", "W", 45, 65, "--"),
                ("S", "B", 25, 45, "--"),
                ("S", "C", 30, 55, "--"),
                ("T", "C", 35, 60, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p12_habitat_cycle",
            "Ecological habitat cycle linking nest, water, tree, food, and hiding cover.",
            {"N": "nest", "W": "water", "T": "tree", "F": "food", "H": "hide"},
            {"N": (0, 0, 10), "W": (25, 0, 0), "T": (10, 20, 25), "F": (35, 25, 10), "H": (-15, 15, 15)},
            [
                ("N", "W", 20, 40, "--"),
                ("N", "T", 15, 35, "--"),
                ("W", "F", 25, 45, "--"),
                ("T", "F", 20, 40, "--"),
                ("N", "H", 10, 30, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p13_highrise_vertical_mix",
            "Multi-floor building pattern with lobby, elevator, office, cafe, gym, and apartment.",
            {"L": "lobby", "E": "elevator", "O": "office", "C": "cafe", "G": "gym", "A": "apartment"},
            {"L": (0, 0, 0), "E": (0, 0, 10), "O": (0, 0, 60), "C": (15, 0, 35), "G": (-15, 5, 90), "A": (5, 5, 130)},
            [
                ("L", "E", 0, 15, "--"),
                ("E", "O", 40, 70, "--"),
                ("O", "C", 20, 40, "--"),
                ("O", "G", 25, 45, "--"),
                ("G", "A", 30, 55, "--"),
                ("C", "A", 80, 110, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p14_datacenter_safety",
            "Data-center services with generator and security constraints around the server hall.",
            {"S": "server_hall", "C": "coolant", "G": "generator", "T": "transformer", "U": "security", "N": "network"},
            {"S": (0, 0, 0), "C": (12, 0, 0), "G": (65, 0, 0), "T": (90, 15, 0), "U": (30, 35, 10), "N": (10, 20, 0)},
            [
                ("S", "C", 0, 20, "--"),
                ("S", "N", 15, 35, "--"),
                ("S", "G", 50, 85, "->"),
                ("G", "T", 20, 45, "--"),
                ("S", "U", 25, 55, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p15_geologic_fault",
            "Geologic survey around a fault with seismic, well, spring, vent, and field lab.",
            {"F": "fault", "S": "seismic", "W": "well", "P": "spring", "V": "gas_vent", "L": "field_lab"},
            {"F": (0, 0, 0), "S": (20, 10, 5), "W": (55, 20, -15), "P": (80, 45, -10), "V": (40, -30, 20), "L": (110, 0, 0)},
            [
                ("F", "S", 15, 30, "--"),
                ("S", "W", 30, 55, "--"),
                ("W", "P", 25, 50, "--"),
                ("F", "V", 35, 60, "--"),
                ("W", "V", 45, 80, "--"),
                ("W", "L", 50, 80, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p16_marine_volume",
            "Marine volume with reef, buoy, dock, sonar, fishery, and beacon.",
            {"R": "reef", "B": "buoy", "D": "dock", "S": "sonar", "F": "fishery", "C": "beacon"},
            {"R": (0, 0, -20), "B": (0, 0, 5), "D": (75, 10, 0), "S": (25, 15, -30), "F": (45, -20, -15), "C": (90, -10, 10)},
            [
                ("R", "B", 20, 35, "--"),
                ("R", "S", 20, 40, "--"),
                ("R", "F", 35, 60, "--"),
                ("B", "D", 60, 95, "--"),
                ("D", "C", 15, 35, "--"),
                ("F", "C", 45, 75, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p17_agriculture_supply",
            "Agricultural production cluster with field, silo, irrigation, road, market, and barn.",
            {"F": "field", "S": "silo", "I": "irrigation", "R": "road", "M": "market", "B": "barn"},
            {"F": (0, 0, 0), "S": (30, 0, 0), "I": (0, 35, -5), "R": (55, 15, 0), "M": (120, 35, 5), "B": (20, 25, 5)},
            [
                ("F", "S", 20, 45, "--"),
                ("F", "I", 25, 45, "--"),
                ("F", "B", 20, 40, "--"),
                ("S", "R", 20, 45, "--"),
                ("R", "M", 55, 85, "--"),
                ("I", "R", 45, 75, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p18_school_safety",
            "School near a park and clinic, while highway and bar are not too close.",
            {"S": "school", "P": "park", "C": "clinic", "H": "highway", "B": "bar"},
            {"S": (0, 0, 0), "P": (15, 10, 0), "C": (30, -5, 0), "H": (95, 0, 0), "B": (70, 35, 0)},
            [
                ("S", "P", 0, 30, "--"),
                ("S", "C", 20, 45, "--"),
                ("S", "H", 75, 130, "->"),
                ("S", "B", 60, 100, "<->"),
            ],
        )
    )

    patterns.append(
        _named(
            "p19_playground_factory_buffer",
            "Playground with clinic and school nearby, but factory mutually buffered.",
            {"P": "playground", "C": "clinic", "F": "factory", "W": "warehouse", "S": "school"},
            {"P": (0, 0, 0), "C": (20, 0, 0), "F": (120, 0, 0), "W": (150, 20, 0), "S": (-25, 10, 0)},
            [
                ("P", "C", 0, 30, "--"),
                ("P", "S", 0, 40, "--"),
                ("P", "F", 90, 160, "<->"),
                ("F", "W", 20, 45, "--"),
            ],
        )
    )

    patterns.append(
        _named(
            "p20_connected_diamond_skip_edges",
            "Five inclusion edges forming a diamond/cycle, useful for skip-edge behavior.",
            {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            {"A": (0, 0, 0), "B": (30, 0, 0), "C": (15, 30, 0), "D": (45, 30, 10)},
            [
                ("A", "B", 20, 40, "--"),
                ("A", "C", 25, 45, "--"),
                ("B", "D", 25, 45, "--"),
                ("C", "D", 25, 45, "--"),
                ("B", "C", 25, 45, "--"),
                ("A", "D", 45, 65, "--"),
            ],
        )
    )

    return patterns


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_synthetic_3d_dataset(
    n_objects: int,
    *,
    seed: int = 0,
    bounds: Box3D = DEFAULT_BOUNDS,
    patterns: Optional[Sequence[NamedPattern | SpatialPattern]] = None,
    ensure_patterns: bool = True,
    planted_matches_per_pattern: int = 1,
    n_noise_keywords: int = 128,
    clustered_fraction: float = 0.65,
    cluster_count: Optional[int] = None,
    cluster_std_fraction: float = 0.035,
    extra_keyword_probability: float = 0.05,
    max_keywords_per_object: int = 3,
    pattern_keyword_weight: float = 3.0,
    noise_keyword_weight: float = 1.0,
    rotate_templates: bool = True,
    max_attempts_per_plant: int = 2000,
    max_attempts_per_background_object: int = 200,
    validate_planted: bool = False,
) -> SyntheticDataset:
    """Generate a synthetic 3-D spatio-textual dataset.

    Parameters
    ----------
    n_objects:
        Exact number of objects in the returned dataset, including planted
        objects.
    seed:
        Random seed for reproducibility.
    bounds:
        Axis-aligned 3-D data volume.
    patterns:
        Optional patterns to return and, when ``ensure_patterns`` is true, plant.
        If omitted, the 20-pattern default suite is used.
    ensure_patterns:
        If true, plant ``planted_matches_per_pattern`` known matches for each
        supplied pattern.  If false, generate only background objects.
    planted_matches_per_pattern:
        Number of known matches to plant per pattern.
    n_noise_keywords:
        Number of non-pattern keywords to include in the background keyword
        vocabulary.
    clustered_fraction:
        Probability that a background object is sampled from a Gaussian cluster
        rather than uniformly from the full volume.
    cluster_count:
        Number of Gaussian cluster centers.  Defaults to a moderate value based
        on ``n_objects``.
    cluster_std_fraction:
        Cluster standard deviation as a fraction of the largest box side.
    extra_keyword_probability:
        Probability of adding each extra keyword after the primary keyword.
    max_keywords_per_object:
        Maximum number of keywords per generated background object.
    pattern_keyword_weight, noise_keyword_weight:
        Relative keyword sampling weights.  Larger pattern keyword weight makes
        accidental/non-planted matches more common.
    rotate_templates:
        Randomly rotate planted templates before translation.
    validate_planted:
        If true, do a full post-generation scan to verify every planted match.
        This is helpful in tests, but can be expensive for very large datasets.
    """

    if n_objects < 0:
        raise ValueError("n_objects must be non-negative")
    if planted_matches_per_pattern < 0:
        raise ValueError("planted_matches_per_pattern must be non-negative")
    if n_noise_keywords < 0:
        raise ValueError("n_noise_keywords must be non-negative")
    if not (0.0 <= clustered_fraction <= 1.0):
        raise ValueError("clustered_fraction must be in [0, 1]")
    if not (0.0 <= extra_keyword_probability <= 1.0):
        raise ValueError("extra_keyword_probability must be in [0, 1]")
    if max_keywords_per_object < 1:
        raise ValueError("max_keywords_per_object must be at least 1")

    rng = random.Random(seed)
    named_patterns = _normalize_patterns(patterns)

    planted_objects: List[SpatialObject] = []
    planted_matches: List[PlantedMatch] = []
    avoid_zones: DefaultDict[str, List[Tuple[Point3D, float]]] = defaultdict(list)

    if ensure_patterns and planted_matches_per_pattern > 0:
        for pattern_idx, named_pattern in enumerate(named_patterns):
            for match_idx in range(planted_matches_per_pattern):
                objects_for_match, assignment, zones = _plant_one_match(
                    named_pattern,
                    pattern_idx=pattern_idx,
                    match_idx=match_idx,
                    rng=rng,
                    bounds=bounds,
                    existing_objects=planted_objects,
                    existing_avoid_zones=avoid_zones,
                    rotate_template=rotate_templates,
                    max_attempts=max_attempts_per_plant,
                )
                planted_objects.extend(objects_for_match)
                planted_matches.append(
                    PlantedMatch(
                        pattern_name=named_pattern.name,
                        match_index=match_idx,
                        assignment=dict(assignment),
                    )
                )
                for keyword, keyword_zones in zones.items():
                    avoid_zones[keyword].extend(keyword_zones)

    if len(planted_objects) > n_objects:
        raise ValueError(
            f"n_objects={n_objects} is too small for the requested planted matches; "
            f"need at least {len(planted_objects)} objects"
        )

    pattern_keywords = sorted({kw for np in named_patterns for kw in np.pattern.vertices.values()})
    noise_keywords = [f"noise_kw_{i:04d}" for i in range(n_noise_keywords)]
    keyword_pool = pattern_keywords + noise_keywords
    if not keyword_pool:
        keyword_pool = ["noise_kw_0000"]

    keyword_weights = _keyword_weights(
        pattern_keywords,
        noise_keywords,
        pattern_keyword_weight=pattern_keyword_weight,
        noise_keyword_weight=noise_keyword_weight,
    )

    fill_count = n_objects - len(planted_objects)
    centers = _cluster_centers(rng, bounds, n_objects, cluster_count)
    cluster_std = max(_side_lengths(bounds)) * cluster_std_fraction

    objects: List[SpatialObject] = list(planted_objects)
    for i in range(fill_count):
        object_id = f"bg_{i:08d}"
        obj = _sample_background_object(
            object_id,
            rng=rng,
            bounds=bounds,
            centers=centers,
            cluster_std=cluster_std,
            clustered_fraction=clustered_fraction,
            keyword_pool=keyword_pool,
            keyword_weights=keyword_weights,
            extra_keyword_probability=extra_keyword_probability,
            max_keywords_per_object=max_keywords_per_object,
            avoid_zones=avoid_zones,
            max_attempts=max_attempts_per_background_object,
        )
        objects.append(obj)

    if validate_planted:
        for planted in planted_matches:
            pattern = _pattern_by_name(named_patterns, planted.pattern_name).pattern
            if not assignment_is_valid(objects, pattern, planted.assignment):
                raise AssertionError(f"planted match failed validation: {planted}")

    metadata = {
        "n_objects": n_objects,
        "n_planted_objects": len(planted_objects),
        "n_background_objects": fill_count,
        "n_patterns": len(named_patterns),
        "ensure_patterns": ensure_patterns,
        "planted_matches_per_pattern": planted_matches_per_pattern,
        "n_noise_keywords": n_noise_keywords,
        "clustered_fraction": clustered_fraction,
        "cluster_count": len(centers),
        "cluster_std_fraction": cluster_std_fraction,
        "extra_keyword_probability": extra_keyword_probability,
        "max_keywords_per_object": max_keywords_per_object,
        "pattern_keyword_weight": pattern_keyword_weight,
        "noise_keyword_weight": noise_keyword_weight,
        "bounds": {"mins": bounds.mins, "maxs": bounds.maxs},
        "volume": _volume(bounds),
        "object_density": (n_objects / _volume(bounds)) if _volume(bounds) > 0 else None,
        "expected_spacing": (_volume(bounds) / n_objects) ** (1.0 / 3.0) if n_objects > 0 and _volume(bounds) > 0 else None,
    }

    return SyntheticDataset(
        objects=objects,
        patterns=named_patterns,
        planted_matches=planted_matches,
        bounds=bounds,
        seed=seed,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Planting and validation
# ---------------------------------------------------------------------------


def _plant_one_match(
    named_pattern: NamedPattern,
    *,
    pattern_idx: int,
    match_idx: int,
    rng: random.Random,
    bounds: Box3D,
    existing_objects: Sequence[SpatialObject],
    existing_avoid_zones: Mapping[str, List[Tuple[Point3D, float]]],
    rotate_template: bool,
    max_attempts: int,
) -> Tuple[List[SpatialObject], Dict[VertexId, str], Dict[str, List[Tuple[Point3D, float]]]]:
    pattern = named_pattern.pattern

    for _ in range(max_attempts):
        if named_pattern.template_points:
            points = _place_template(
                named_pattern.template_points,
                rng=rng,
                bounds=bounds,
                rotate_template=rotate_template,
            )
        else:
            local = _sample_pattern_embedding(pattern, rng)
            points = _place_template(local, rng=rng, bounds=bounds, rotate_template=rotate_template)

        slug = _slugify(named_pattern.name)
        objects: List[SpatialObject] = []
        assignment: Dict[VertexId, str] = {}
        for vertex, keyword in pattern.vertices.items():
            point = points[vertex]
            object_id = f"plant_{pattern_idx:02d}_{slug}_{match_idx:02d}_{_slugify(str(vertex))}"
            obj = SpatialObject(object_id, point[0], point[1], point[2], frozenset({keyword}))
            objects.append(obj)
            assignment[vertex] = object_id

        if not assignment_is_valid(objects, pattern, assignment):
            continue

        new_zones = _avoid_zones_for_assignment(pattern, objects, assignment)
        if _objects_violate_avoid_zones(objects, existing_avoid_zones):
            continue
        if _objects_violate_avoid_zones(existing_objects, new_zones):
            continue
        if _objects_violate_avoid_zones(objects, new_zones, ignore_assignment=assignment):
            continue

        return objects, assignment, new_zones

    raise RuntimeError(
        f"could not plant {named_pattern.name!r} after {max_attempts} attempts. "
        "Try larger bounds, fewer planted matches, smaller exclusion lower bounds, "
        "or ensure_patterns=False."
    )


def _avoid_zones_for_assignment(
    pattern: SpatialPattern,
    objects: Sequence[SpatialObject],
    assignment: Mapping[VertexId, str],
) -> Dict[str, List[Tuple[Point3D, float]]]:
    by_id = {str(obj.id): obj for obj in objects}
    zones: DefaultDict[str, List[Tuple[Point3D, float]]] = defaultdict(list)
    for edge in pattern.edges:
        if edge.lower <= 0:
            continue
        src = by_id[str(assignment[edge.source])]
        tgt = by_id[str(assignment[edge.target])]
        if edge.sign in (EdgeSign.SOURCE_EXCLUDES_TARGET, EdgeSign.MUTUAL_EXCLUSION):
            zones[pattern.vertices[edge.target]].append((src.point, edge.lower))
        if edge.sign in (EdgeSign.TARGET_EXCLUDES_SOURCE, EdgeSign.MUTUAL_EXCLUSION):
            zones[pattern.vertices[edge.source]].append((tgt.point, edge.lower))
    return {k: list(v) for k, v in zones.items()}


def _objects_violate_avoid_zones(
    objects: Sequence[SpatialObject],
    avoid_zones: Mapping[str, List[Tuple[Point3D, float]]],
    *,
    ignore_assignment: Optional[Mapping[VertexId, str]] = None,
) -> bool:
    # ``ignore_assignment`` is intentionally conservative: the assigned object is
    # not ignored globally because an exclusion lower bound also applies to the
    # matched object.  Since templates place matched endpoints at distance >=
    # lower, this remains valid and catches accidental same-keyword duplicates.
    del ignore_assignment
    for obj in objects:
        if _point_violates_avoid_zones(obj.point, obj.keywords, avoid_zones):
            return True
    return False


def _point_violates_avoid_zones(
    point: Point3D,
    keywords: Iterable[str],
    avoid_zones: Mapping[str, List[Tuple[Point3D, float]]],
) -> bool:
    for keyword in keywords:
        for center, radius in avoid_zones.get(keyword, ()):
            if radius > 0 and squared_distance_points(point, center) < radius * radius - EPS:
                return True
    return False


def assignment_is_valid(
    objects: Iterable[SpatialObject],
    pattern: SpatialPattern,
    assignment: Mapping[VertexId, str],
) -> bool:
    """Return true if a known assignment satisfies a pattern on the object set."""

    by_id = {str(obj.id): obj for obj in objects}
    by_keyword: DefaultDict[str, List[SpatialObject]] = defaultdict(list)
    for obj in by_id.values():
        for keyword in obj.keywords:
            by_keyword[keyword].append(obj)

    for vertex in pattern.vertices:
        if vertex not in assignment or str(assignment[vertex]) not in by_id:
            return False
        if pattern.vertices[vertex] not in by_id[str(assignment[vertex])].keywords:
            return False

    for edge in pattern.edges:
        src = by_id[str(assignment[edge.source])]
        tgt = by_id[str(assignment[edge.target])]
        d2 = squared_distance_points(src.point, tgt.point)
        if d2 < edge.lower * edge.lower - EPS or d2 > edge.upper * edge.upper + EPS:
            return False

        if edge.sign in (EdgeSign.SOURCE_EXCLUDES_TARGET, EdgeSign.MUTUAL_EXCLUSION):
            target_keyword = pattern.vertices[edge.target]
            for obj in by_keyword.get(target_keyword, ()):
                if squared_distance_points(src.point, obj.point) < edge.lower * edge.lower - EPS:
                    return False

        if edge.sign in (EdgeSign.TARGET_EXCLUDES_SOURCE, EdgeSign.MUTUAL_EXCLUSION):
            source_keyword = pattern.vertices[edge.source]
            for obj in by_keyword.get(source_keyword, ()):
                if squared_distance_points(tgt.point, obj.point) < edge.lower * edge.lower - EPS:
                    return False

    return True


def _assert_template_satisfies(pattern: SpatialPattern, points: Mapping[VertexId, Point3D]) -> None:
    objects = [SpatialObject(str(v), p[0], p[1], p[2], frozenset({kw})) for v, (kw, p) in _vertex_items(pattern, points)]
    assignment = {v: str(v) for v in pattern.vertices}
    if not assignment_is_valid(objects, pattern, assignment):
        raise ValueError("default template does not satisfy its pattern")


def _vertex_items(pattern: SpatialPattern, points: Mapping[VertexId, Point3D]) -> Iterable[Tuple[VertexId, Tuple[str, Point3D]]]:
    for vertex, keyword in pattern.vertices.items():
        if vertex not in points:
            raise ValueError(f"missing template point for vertex {vertex!r}")
        yield vertex, (keyword, points[vertex])


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _sample_background_object(
    object_id: str,
    *,
    rng: random.Random,
    bounds: Box3D,
    centers: Sequence[Point3D],
    cluster_std: float,
    clustered_fraction: float,
    keyword_pool: Sequence[str],
    keyword_weights: Sequence[float],
    extra_keyword_probability: float,
    max_keywords_per_object: int,
    avoid_zones: Mapping[str, List[Tuple[Point3D, float]]],
    max_attempts: int,
) -> SpatialObject:
    for _ in range(max_attempts):
        point = _sample_point(rng, bounds, centers, cluster_std, clustered_fraction)
        keywords = _sample_keywords(
            rng,
            keyword_pool,
            keyword_weights,
            extra_keyword_probability=extra_keyword_probability,
            max_keywords_per_object=max_keywords_per_object,
        )
        if not _point_violates_avoid_zones(point, keywords, avoid_zones):
            return SpatialObject(object_id, point[0], point[1], point[2], frozenset(keywords))

    # Last resort: use a noise keyword that has no avoid zones if one exists.
    safe_keywords = [kw for kw in keyword_pool if not avoid_zones.get(kw)]
    if safe_keywords:
        for _ in range(max_attempts):
            point = _sample_point(rng, bounds, centers, cluster_std, clustered_fraction)
            keyword = rng.choice(safe_keywords)
            if not _point_violates_avoid_zones(point, [keyword], avoid_zones):
                return SpatialObject(object_id, point[0], point[1], point[2], frozenset({keyword}))

    raise RuntimeError(
        f"could not place background object {object_id!r} without violating planted exclusion zones"
    )


def _sample_point(
    rng: random.Random,
    bounds: Box3D,
    centers: Sequence[Point3D],
    cluster_std: float,
    clustered_fraction: float,
) -> Point3D:
    if centers and rng.random() < clustered_fraction:
        center = rng.choice(centers)
        for _ in range(50):
            p = tuple(rng.gauss(center[d], cluster_std) for d in range(3))
            if _contains(bounds, p):
                return p  # type: ignore[return-value]
    return tuple(rng.uniform(bounds.mins[d], bounds.maxs[d]) for d in range(3))  # type: ignore[return-value]


def _sample_keywords(
    rng: random.Random,
    keyword_pool: Sequence[str],
    keyword_weights: Sequence[float],
    *,
    extra_keyword_probability: float,
    max_keywords_per_object: int,
) -> List[str]:
    keywords = {rng.choices(keyword_pool, weights=keyword_weights, k=1)[0]}
    while len(keywords) < max_keywords_per_object and rng.random() < extra_keyword_probability:
        keywords.add(rng.choices(keyword_pool, weights=keyword_weights, k=1)[0])
    return sorted(keywords)


def _keyword_weights(
    pattern_keywords: Sequence[str],
    noise_keywords: Sequence[str],
    *,
    pattern_keyword_weight: float,
    noise_keyword_weight: float,
) -> List[float]:
    if pattern_keyword_weight <= 0 or noise_keyword_weight <= 0:
        raise ValueError("keyword weights must be positive")
    weights = [float(pattern_keyword_weight) for _ in pattern_keywords]
    # Mild Zipf-like noise popularity creates more realistic skew without making
    # every object share the same noise keyword.
    weights.extend(float(noise_keyword_weight) / math.sqrt(i + 1) for i in range(len(noise_keywords)))
    return weights


def _cluster_centers(
    rng: random.Random,
    bounds: Box3D,
    n_objects: int,
    cluster_count: Optional[int],
) -> List[Point3D]:
    if cluster_count is None:
        cluster_count = max(4, min(128, int(math.sqrt(max(n_objects, 1)) / 4)))
    if cluster_count < 0:
        raise ValueError("cluster_count must be non-negative")
    return [tuple(rng.uniform(bounds.mins[d], bounds.maxs[d]) for d in range(3)) for _ in range(cluster_count)]  # type: ignore[list-item]


def _place_template(
    points: Mapping[VertexId, Point3D],
    *,
    rng: random.Random,
    bounds: Box3D,
    rotate_template: bool,
) -> Dict[VertexId, Point3D]:
    vertices = list(points.keys())
    centroid = tuple(sum(points[v][d] for v in vertices) / len(vertices) for d in range(3))
    centered = {v: tuple(points[v][d] - centroid[d] for d in range(3)) for v in vertices}
    rotation = _random_rotation_matrix(rng) if rotate_template else ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    rotated = {v: _mat_vec(rotation, centered[v]) for v in vertices}

    mins = tuple(min(rotated[v][d] for v in vertices) for d in range(3))
    maxs = tuple(max(rotated[v][d] for v in vertices) for d in range(3))

    side = max(_side_lengths(bounds))
    margin = max(1e-6 * side, 1e-6)
    translation: List[float] = []
    for d in range(3):
        low = bounds.mins[d] + margin - mins[d]
        high = bounds.maxs[d] - margin - maxs[d]
        if low > high:
            raise RuntimeError("pattern template is too large for the requested bounds")
        translation.append(rng.uniform(low, high))

    out: Dict[VertexId, Point3D] = {}
    for v, p in rotated.items():
        out[v] = tuple(p[d] + translation[d] for d in range(3))  # type: ignore[assignment]
    return out


def _sample_pattern_embedding(pattern: SpatialPattern, rng: random.Random, max_attempts: int = 2000) -> Dict[VertexId, Point3D]:
    """Heuristic local embedding for custom patterns without templates."""

    vertices = list(pattern.vertices.keys())
    if not pattern.edges:
        return {v: (float(i * 10), 0.0, 0.0) for i, v in enumerate(vertices)}

    max_upper = max(edge.upper for edge in pattern.edges)
    component_gap = max(10.0, 3.0 * max_upper)

    adjacency: DefaultDict[VertexId, List[PatternEdge]] = defaultdict(list)
    for edge in pattern.edges:
        adjacency[edge.source].append(edge)
        adjacency[edge.target].append(edge)

    for _ in range(max_attempts):
        points: Dict[VertexId, Point3D] = {}
        component = 0
        for root in vertices:
            if root in points:
                continue
            points[root] = (component * component_gap, 0.0, 0.0)
            component += 1
            queue = [root]
            while queue:
                current = queue.pop(0)
                for edge in adjacency[current]:
                    if edge.source == current:
                        other = edge.target
                    else:
                        other = edge.source
                    if other in points:
                        continue
                    direction = _random_unit_vector(rng)
                    lo, hi = edge.lower, edge.upper
                    if hi <= lo:
                        distance = lo
                    else:
                        distance = rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo))
                    base = points[current]
                    points[other] = tuple(base[d] + distance * direction[d] for d in range(3))  # type: ignore[assignment]
                    queue.append(other)

        objects = [SpatialObject(str(v), *points[v], frozenset({pattern.vertices[v]})) for v in vertices]
        assignment = {v: str(v) for v in vertices}
        if assignment_is_valid(objects, pattern, assignment):
            return points

    raise RuntimeError("could not find a valid local embedding for a custom pattern")


def _random_unit_vector(rng: random.Random) -> Point3D:
    z = rng.uniform(-1.0, 1.0)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return (r * math.cos(theta), r * math.sin(theta), z)


def _random_rotation_matrix(rng: random.Random) -> Tuple[Point3D, Point3D, Point3D]:
    # Uniform random unit quaternion method.
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _mat_vec(matrix: Tuple[Point3D, Point3D, Point3D], vector: Point3D) -> Point3D:
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _contains(bounds: Box3D, point: Point3D) -> bool:
    return all(bounds.mins[d] <= point[d] <= bounds.maxs[d] for d in range(3))


def _side_lengths(bounds: Box3D) -> Tuple[float, float, float]:
    return tuple(bounds.maxs[d] - bounds.mins[d] for d in range(3))  # type: ignore[return-value]


def _volume(bounds: Box3D) -> float:
    x, y, z = _side_lengths(bounds)
    return x * y * z


def _normalize_patterns(patterns: Optional[Sequence[NamedPattern | SpatialPattern]]) -> List[NamedPattern]:
    if patterns is None:
        return default_pattern_suite()
    out: List[NamedPattern] = []
    for i, item in enumerate(patterns):
        if isinstance(item, NamedPattern):
            out.append(item)
        elif isinstance(item, SpatialPattern):
            out.append(NamedPattern(name=f"custom_pattern_{i:02d}", pattern=item))
        else:
            raise TypeError(f"expected NamedPattern or SpatialPattern, got {type(item)!r}")
    return out


def _pattern_by_name(patterns: Sequence[NamedPattern], name: str) -> NamedPattern:
    for pattern in patterns:
        if pattern.name == name:
            return pattern
    raise KeyError(name)


def _slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "x"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_objects_jsonl(objects: Iterable[SpatialObject], path: str | Path) -> None:
    """Write objects as JSON Lines."""

    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for obj in objects:
            row = {"id": obj.id, "x": obj.x, "y": obj.y, "z": obj.z, "keywords": sorted(obj.keywords)}
            f.write(json.dumps(row, sort_keys=True) + "\n")


def load_objects_jsonl(path: str | Path) -> List[SpatialObject]:
    """Load objects written by ``save_objects_jsonl``."""

    objects: List[SpatialObject] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                objects.append(SpatialObject(row["id"], row["x"], row["y"], row["z"], frozenset(row["keywords"])))
            except KeyError as exc:
                raise ValueError(f"missing key {exc} on line {line_number}") from exc
    return objects


def save_patterns_json(patterns: Iterable[NamedPattern], path: str | Path) -> None:
    """Write named patterns as JSON."""

    rows = []
    for named in patterns:
        rows.append(
            {
                "name": named.name,
                "description": named.description,
                "vertices": dict(named.pattern.vertices),
                "edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "lower": edge.lower,
                        "upper": edge.upper,
                        "sign": edge.sign.value,
                    }
                    for edge in named.pattern.edges
                ],
                "template_points": {str(k): list(v) for k, v in named.template_points.items()},
            }
        )
    Path(path).write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def load_patterns_json(path: str | Path) -> List[NamedPattern]:
    """Load patterns written by ``save_patterns_json``."""

    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out: List[NamedPattern] = []
    for row in rows:
        vertices = dict(row["vertices"])
        edges = [PatternEdge(e["source"], e["target"], e["lower"], e["upper"], e["sign"]) for e in row["edges"]]
        template_points = {k: tuple(v) for k, v in row.get("template_points", {}).items()}
        out.append(
            NamedPattern(
                name=row["name"],
                description=row.get("description", ""),
                pattern=SpatialPattern(vertices, edges),
                template_points=template_points,
            )
        )
    return out


def save_planted_matches_json(planted_matches: Iterable[PlantedMatch], path: str | Path) -> None:
    rows = [
        {"pattern_name": pm.pattern_name, "match_index": pm.match_index, "assignment": dict(pm.assignment)}
        for pm in planted_matches
    ]
    Path(path).write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _bounds_from_args(values: Optional[Sequence[float]]) -> Box3D:
    if values is None:
        return DEFAULT_BOUNDS
    if len(values) != 6:
        raise ValueError("--bounds expects xmin ymin zmin xmax ymax zmax")
    return Box3D((values[0], values[1], values[2]), (values[3], values[4], values[5]))


def bounds_from_sparsity_controls(
    n_objects: int,
    *,
    bounds_values: Optional[Sequence[float]] = None,
    domain_side: Optional[float] = None,
    target_density: Optional[float] = None,
) -> Box3D:
    """Resolve global spatial sparsity controls into a 3-D bounding box.

    Precedence is: explicit ``bounds_values`` > ``domain_side`` >
    ``target_density`` > ``DEFAULT_BOUNDS``.

    Global object density is ``n_objects / volume``.  For a fixed ``n_objects``,
    increasing the domain side or decreasing target density makes the dataset
    spatially sparser.
    """

    specified = [bounds_values is not None, domain_side is not None, target_density is not None]
    if sum(specified) > 1:
        raise ValueError("use only one of --bounds, --domain-side, or --target-density")
    if bounds_values is not None:
        return _bounds_from_args(bounds_values)
    if domain_side is not None:
        if domain_side <= 0:
            raise ValueError("--domain-side must be positive")
        return Box3D((0.0, 0.0, 0.0), (domain_side, domain_side, domain_side))
    if target_density is not None:
        if target_density <= 0:
            raise ValueError("--target-density must be positive")
        side = (max(n_objects, 1) / target_density) ** (1.0 / 3.0)
        return Box3D((0.0, 0.0, 0.0), (side, side, side))
    return DEFAULT_BOUNDS


def _cmd_list_patterns(_: argparse.Namespace) -> None:
    for i, named in enumerate(default_pattern_suite(), start=1):
        print(f"{i:02d}. {named.name}: {named.description}")
        print(f"    vertices={dict(named.pattern.vertices)}")
        print(f"    edges={[ (e.source, e.target, e.lower, e.upper, e.sign.value) for e in named.pattern.edges ]}")


def _cmd_generate(args: argparse.Namespace) -> None:
    bounds = bounds_from_sparsity_controls(
        args.n_objects,
        bounds_values=args.bounds,
        domain_side=args.domain_side,
        target_density=args.target_density,
    )
    patterns = default_pattern_suite()[: args.pattern_count]
    dataset = generate_synthetic_3d_dataset(
        args.n_objects,
        seed=args.seed,
        bounds=bounds,
        patterns=patterns,
        ensure_patterns=not args.no_plant,
        planted_matches_per_pattern=args.matches_per_pattern,
        n_noise_keywords=args.noise_keywords,
        clustered_fraction=args.clustered_fraction,
        cluster_count=args.cluster_count,
        cluster_std_fraction=args.cluster_std_fraction,
        extra_keyword_probability=args.extra_keyword_probability,
        max_keywords_per_object=args.max_keywords_per_object,
        pattern_keyword_weight=args.pattern_keyword_weight,
        noise_keyword_weight=args.noise_keyword_weight,
        validate_planted=args.validate_planted,
    )
    save_objects_jsonl(dataset.objects, args.objects_out)
    save_patterns_json(dataset.patterns, args.patterns_out)
    save_planted_matches_json(dataset.planted_matches, args.planted_out)
    Path(args.metadata_out).write_text(json.dumps(dataset.metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(dataset.objects)} objects to {args.objects_out}")
    print(f"wrote {len(dataset.patterns)} patterns to {args.patterns_out}")
    print(f"wrote {len(dataset.planted_matches)} planted assignments to {args.planted_out}")


def _cmd_smoke_test(args: argparse.Namespace) -> None:
    patterns = default_pattern_suite()[: args.pattern_count]
    dataset = generate_synthetic_3d_dataset(
        args.n_objects,
        seed=args.seed,
        patterns=patterns,
        ensure_patterns=True,
        planted_matches_per_pattern=1,
        n_noise_keywords=args.noise_keywords,
        validate_planted=True,
    )
    index = InvertedOctreeIndex(dataset.objects, capacity=args.capacity, min_level=args.min_level, max_level=args.max_level)
    matcher = ESPM3DMatcher(index, require_distinct_objects=True)
    for named in dataset.patterns:
        matches = matcher.match(named.pattern)
        status = "OK" if matches else "NO MATCH"
        print(f"{status:8s} {named.name:35s} matches_found={len(matches)}")
        if not matches:
            raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic 3-D generator for ESPM-3D")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-patterns", help="print the default 20-pattern suite")
    list_parser.set_defaults(func=_cmd_list_patterns)

    gen = sub.add_parser("generate", help="generate objects, patterns, planted assignments, and metadata")
    gen.add_argument("--n-objects", type=int, default=10000)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--pattern-count", type=int, default=20, help="use the first K default patterns")
    gen.add_argument("--matches-per-pattern", type=int, default=1)
    gen.add_argument("--no-plant", action="store_true", help="do not plant guaranteed matches")
    gen.add_argument("--noise-keywords", type=int, default=128)
    gen.add_argument("--clustered-fraction", type=float, default=0.65)
    gen.add_argument("--cluster-count", type=int, default=None)
    gen.add_argument("--cluster-std-fraction", type=float, default=0.035)
    gen.add_argument("--extra-keyword-probability", type=float, default=0.05)
    gen.add_argument("--max-keywords-per-object", type=int, default=3)
    gen.add_argument("--pattern-keyword-weight", type=float, default=3.0)
    gen.add_argument("--noise-keyword-weight", type=float, default=1.0)
    gen.add_argument("--bounds", type=float, nargs=6, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
    gen.add_argument("--domain-side", type=float, default=None, help="use a cubic domain [0, side]^3; larger side means lower spatial density")
    gen.add_argument("--target-density", type=float, default=None, help="objects per cubic unit; overrides default bounds by choosing a cube side")
    gen.add_argument("--validate-planted", action="store_true")
    gen.add_argument("--objects-out", default="synthetic_objects.jsonl")
    gen.add_argument("--patterns-out", default="synthetic_patterns.json")
    gen.add_argument("--planted-out", default="synthetic_planted_matches.json")
    gen.add_argument("--metadata-out", default="synthetic_metadata.json")
    gen.set_defaults(func=_cmd_generate)

    smoke = sub.add_parser("smoke-test", help="generate a small dataset and run ESPM-3D on each pattern")
    smoke.add_argument("--n-objects", type=int, default=2000)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--pattern-count", type=int, default=5)
    smoke.add_argument("--noise-keywords", type=int, default=64)
    smoke.add_argument("--capacity", type=int, default=64)
    smoke.add_argument("--min-level", type=int, default=1)
    smoke.add_argument("--max-level", type=int, default=8)
    smoke.set_defaults(func=_cmd_smoke_test)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
