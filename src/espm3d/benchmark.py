"""Scalability benchmark runner for ESPM-3D.

The benchmark runner records one detailed raw result row per
``(dataset size, sparsity profile, pattern variant)`` and one aggregate summary
row per ``(dataset size, sparsity profile)`` scenario.

The built-in profiles now vary *candidate graph sparsity*: roughly, how many
object-pair edges survive between the keyword postings that participate in a
pattern.  Spatial density, clustering, and keyword frequency all affect that
candidate graph.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised only when psutil is installed
    import psutil  # type: ignore
except Exception:  # pragma: no cover - fallback path
    psutil = None  # type: ignore

from .generate_synthetic_data import (
    NamedPattern,
    bounds_from_sparsity_controls,
    default_pattern_suite,
    generate_synthetic_3d_dataset,
)
from .matcher import Box3D, ESPM3DMatcher, InvertedOctreeIndex, MatchStats, PatternEdge, SpatialPattern


PROFILE_SCALES: Mapping[str, Tuple[int, ...]] = {
    # Quick local sanity check.
    "smoke": (2_000,),
    # Useful for laptops and CI while still exercising the benchmark harness.
    "standard": (10_000, 100_000),
    # Includes the paper-inspired large scales.  Expect long runtimes and high RAM.
    "full": (10_000, 100_000, 1_000_000, 10_000_000),
}


@dataclass(frozen=True)
class SparsityProfile:
    """Settings that influence candidate/e-match graph sparsity.

    These are not properties of the query pattern graph.  They change the
    generated data distribution and therefore the induced graph of possible
    object pairs between pattern keywords.
    """

    name: str
    description: str
    domain_side: Optional[float]
    target_density: Optional[float]
    n_noise_keywords: int
    clustered_fraction: float
    cluster_count: Optional[int]
    cluster_std_fraction: float
    extra_keyword_probability: float
    max_keywords_per_object: int
    pattern_keyword_weight: float
    noise_keyword_weight: float


SPARSITY_PROFILES: Mapping[str, SparsityProfile] = {
    "manual": SparsityProfile(
        name="manual",
        description=(
            "Use the explicit command-line/config generator controls instead of a built-in "
            "graph-sparsity preset. This reproduces the pre-profile behavior."
        ),
        domain_side=None,
        target_density=None,
        n_noise_keywords=128,
        clustered_fraction=0.65,
        cluster_count=None,
        cluster_std_fraction=0.035,
        extra_keyword_probability=0.05,
        max_keywords_per_object=3,
        pattern_keyword_weight=3.0,
        noise_keyword_weight=1.0,
    ),
    "very_sparse_graph": SparsityProfile(
        name="very_sparse_graph",
        description=(
            "Very few candidate object-pair edges: large volume, nearly uniform points, "
            "many noise keywords, and rare pattern keywords."
        ),
        domain_side=5_000.0,
        target_density=None,
        n_noise_keywords=5_000,
        clustered_fraction=0.02,
        cluster_count=None,
        cluster_std_fraction=0.05,
        extra_keyword_probability=0.005,
        max_keywords_per_object=2,
        pattern_keyword_weight=0.25,
        noise_keyword_weight=4.0,
    ),
    "sparse_graph": SparsityProfile(
        name="sparse_graph",
        description=(
            "Low candidate graph density: large volume, weak clustering, many noise keywords, "
            "and comparatively uncommon pattern keywords."
        ),
        domain_side=2_500.0,
        target_density=None,
        n_noise_keywords=2_048,
        clustered_fraction=0.15,
        cluster_count=None,
        cluster_std_fraction=0.045,
        extra_keyword_probability=0.02,
        max_keywords_per_object=2,
        pattern_keyword_weight=0.75,
        noise_keyword_weight=3.0,
    ),
    "medium_graph": SparsityProfile(
        name="medium_graph",
        description=(
            "Balanced default: moderate volume, moderate clustering, and moderate pattern-keyword "
            "frequency. This is closest to the original benchmark defaults."
        ),
        domain_side=1_000.0,
        target_density=None,
        n_noise_keywords=512,
        clustered_fraction=0.50,
        cluster_count=None,
        cluster_std_fraction=0.035,
        extra_keyword_probability=0.04,
        max_keywords_per_object=3,
        pattern_keyword_weight=2.0,
        noise_keyword_weight=1.5,
    ),
    "dense_graph": SparsityProfile(
        name="dense_graph",
        description=(
            "Many candidate object-pair edges: smaller volume, strong local clustering, fewer noise "
            "keywords, and common pattern keywords."
        ),
        domain_side=600.0,
        target_density=None,
        n_noise_keywords=128,
        clustered_fraction=0.80,
        cluster_count=64,
        cluster_std_fraction=0.025,
        extra_keyword_probability=0.08,
        max_keywords_per_object=4,
        pattern_keyword_weight=4.0,
        noise_keyword_weight=0.75,
    ),
    "very_dense_graph": SparsityProfile(
        name="very_dense_graph",
        description=(
            "Stress setting with extremely many candidate edges. Use carefully with small scales or "
            "tight match limits."
        ),
        domain_side=350.0,
        target_density=None,
        n_noise_keywords=64,
        clustered_fraction=0.90,
        cluster_count=32,
        cluster_std_fraction=0.018,
        extra_keyword_probability=0.15,
        max_keywords_per_object=5,
        pattern_keyword_weight=8.0,
        noise_keyword_weight=0.50,
    ),
}

PROFILE_SPARSITY_SWEEPS: Mapping[str, Tuple[str, ...]] = {
    # Small enough to exercise all three default graph-sparsity regimes quickly.
    "smoke": ("sparse_graph", "medium_graph", "dense_graph"),
    # The normal laptop/CI sweep: scale x graph sparsity.
    "standard": ("sparse_graph", "medium_graph", "dense_graph"),
    # Avoid the densest preset by default at 10M objects; it can still be requested.
    "full": ("very_sparse_graph", "sparse_graph", "medium_graph"),
}

RESULT_FIELDS = [
    "run_id",
    "scenario_id",
    "status",
    "error_type",
    "error_message",
    "n_objects",
    "sparsity_profile",
    "sparsity_description",
    "pattern_index",
    "pattern_name",
    "pattern_description",
    "pattern_vertices",
    "pattern_edges",
    "exclusive_edges",
    "interval_scale",
    "matches_returned",
    "match_limit",
    "limit_reached",
    "generation_time_s",
    "index_build_time_s",
    "match_time_s",
    "scenario_total_time_s",
    "rss_before_mb",
    "rss_after_generation_mb",
    "rss_after_index_mb",
    "rss_after_match_mb",
    "rss_peak_scenario_mb",
    "rss_peak_match_mb",
    "distinct_keywords",
    "keyword_postings_total",
    "keyword_postings_mean",
    "keyword_postings_max",
    "keyword_postings_by_vertex",
    "keyword_postings_min_by_vertex",
    "keyword_postings_mean_by_vertex",
    "keyword_postings_max_by_vertex",
    "candidate_pair_space_total",
    "candidate_pair_space_non_skip",
    "candidate_pair_space_max_edge",
    "candidate_pair_space_mean_edge",
    "candidate_graph_edges_measured",
    "candidate_graph_density_measured",
    "candidate_graph_edges_by_edge",
    "index_tree_count",
    "index_node_count",
    "index_leaf_count",
    "index_max_depth",
    "nmatch_levels",
    "nmatch_total_all_levels",
    "nmatch_total_final_level",
    "ematch_total",
    "skip_edge_count",
    "skip_edges",
    "domain_volume",
    "object_density",
    "expected_spacing",
    "n_noise_keywords",
    "clustered_fraction",
    "cluster_count",
    "cluster_std_fraction",
    "extra_keyword_probability",
    "max_keywords_per_object",
    "pattern_keyword_weight",
    "noise_keyword_weight",
    "capacity",
    "min_level",
    "max_level",
    "require_distinct_objects",
    "seed",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a scalability sweep."""

    scales: Tuple[int, ...]
    sparsity_profiles: Tuple[str, ...] = ("manual",)
    pattern_count: int = 20
    interval_scales: Tuple[float, ...] = (1.0,)
    seed: int = 42
    output_dir: str = "results"
    match_limit: Optional[int] = 1000
    ensure_patterns: bool = True
    matches_per_pattern: int = 1
    n_noise_keywords: int = 128
    clustered_fraction: float = 0.65
    cluster_count: Optional[int] = None
    cluster_std_fraction: float = 0.035
    extra_keyword_probability: float = 0.05
    max_keywords_per_object: int = 3
    pattern_keyword_weight: float = 3.0
    noise_keyword_weight: float = 1.0
    bounds: Optional[Tuple[float, float, float, float, float, float]] = None
    domain_side: Optional[float] = None
    target_density: Optional[float] = None
    capacity: int = 64
    min_level: int = 1
    max_level: int = 10
    require_distinct_objects: bool = True
    write_dataset: bool = False


class MemorySampler:
    """Sample process RSS in a background thread.

    ``resource.getrusage`` only reports maximum RSS since process start on many
    platforms.  Sampling current RSS gives a more useful peak for a benchmark
    phase.  If psutil is unavailable, the sampler reports ``None``.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self.start_rss_mb: Optional[float] = current_rss_mb()
        self.peak_rss_mb: Optional[float] = self.start_rss_mb
        self.end_rss_mb: Optional[float] = None

    def __enter__(self) -> "MemorySampler":
        if self._process is not None:
            self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, self.interval_s * 4))
        self.end_rss_mb = current_rss_mb()
        if self.end_rss_mb is not None:
            if self.peak_rss_mb is None:
                self.peak_rss_mb = self.end_rss_mb
            else:
                self.peak_rss_mb = max(self.peak_rss_mb, self.end_rss_mb)

    def _run(self) -> None:
        assert self._process is not None
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss / (1024.0 * 1024.0)
                if self.peak_rss_mb is None:
                    self.peak_rss_mb = rss
                else:
                    self.peak_rss_mb = max(self.peak_rss_mb, rss)
            except Exception:
                pass
            self._stop.wait(self.interval_s)


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.end = self.start
        self.elapsed_s = 0.0
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.end = time.perf_counter()
        self.elapsed_s = self.end - self.start


def current_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def _json(value: Any) -> str:
    """Serialize nested fields for CSV cells."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _pattern_summary(named: NamedPattern, interval_scale: float) -> Dict[str, Any]:
    pattern = named.pattern
    edges = list(pattern.edges)
    exclusive_edges = sum(1 for edge in edges if edge.sign.is_exclusive)
    return {
        "pattern_name": named.name,
        "pattern_description": named.description,
        "pattern_vertices": len(pattern.vertices),
        "pattern_edges": len(edges),
        "exclusive_edges": exclusive_edges,
        "interval_scale": interval_scale,
    }


def _stats_summary(stats: Optional[MatchStats]) -> Dict[str, Any]:
    if stats is None:
        return {
            "nmatch_levels": 0,
            "nmatch_total_all_levels": None,
            "nmatch_total_final_level": None,
            "ematch_total": None,
            "skip_edge_count": None,
            "skip_edges": [],
        }
    nmatch_total_all = sum(sum(level_counts.values()) for level_counts in stats.nmatch_counts_by_level)
    final_level_counts = stats.nmatch_counts_by_level[-1] if stats.nmatch_counts_by_level else {}
    return {
        "nmatch_levels": len(stats.nmatch_counts_by_level),
        "nmatch_total_all_levels": nmatch_total_all,
        "nmatch_total_final_level": sum(final_level_counts.values()),
        "ematch_total": sum(stats.ematch_counts.values()),
        "skip_edge_count": len(stats.skip_edges),
        "skip_edges": sorted(stats.skip_edges),
    }


def _empty_candidate_graph_summary() -> Dict[str, Any]:
    return {
        "keyword_postings_by_vertex": {},
        "keyword_postings_min_by_vertex": None,
        "keyword_postings_mean_by_vertex": None,
        "keyword_postings_max_by_vertex": None,
        "candidate_pair_space_total": None,
        "candidate_pair_space_non_skip": None,
        "candidate_pair_space_max_edge": None,
        "candidate_pair_space_mean_edge": None,
        "candidate_graph_edges_measured": None,
        "candidate_graph_density_measured": None,
        "candidate_graph_edges_by_edge": [],
    }


def _candidate_pair_space(
    index: InvertedOctreeIndex,
    source_keyword: str,
    target_keyword: str,
    *,
    require_distinct_objects: bool,
) -> int:
    source_count = len(index.by_keyword.get(source_keyword, ()))
    target_count = len(index.by_keyword.get(target_keyword, ()))
    if require_distinct_objects and source_keyword == target_keyword:
        return source_count * max(target_count - 1, 0)
    return source_count * target_count


def _candidate_graph_summary(
    pattern: SpatialPattern,
    index: Optional[InvertedOctreeIndex],
    stats: Optional[MatchStats],
    *,
    require_distinct_objects: bool,
) -> Dict[str, Any]:
    """Summarize the induced object-pair graph for one pattern.

    ``candidate_pair_space_*`` counts the possible object pairs implied by the
    relevant keyword postings before distance/sign pruning. ``ematch_total`` from
    the matcher is the measured count of surviving object-pair edges for
    non-skip edges.  Skip edges are excluded from the density denominator because
    ESPM intentionally does not materialize their e-match lists.
    """

    if index is None:
        return _empty_candidate_graph_summary()

    postings_by_vertex = {
        str(vertex): len(index.by_keyword.get(keyword, ())) for vertex, keyword in pattern.vertices.items()
    }
    posting_values = list(postings_by_vertex.values())
    per_edge: List[Dict[str, Any]] = []
    skip_edges = stats.skip_edges if stats is not None else set()
    ematch_counts = stats.ematch_counts if stats is not None else {}

    total_pair_space = 0
    non_skip_pair_space = 0
    for eid, edge in enumerate(pattern.edges):
        source_keyword = pattern.vertices[edge.source]
        target_keyword = pattern.vertices[edge.target]
        pair_space = _candidate_pair_space(
            index,
            source_keyword,
            target_keyword,
            require_distinct_objects=require_distinct_objects,
        )
        is_skip = eid in skip_edges
        ematches = ematch_counts.get(eid)
        total_pair_space += pair_space
        if not is_skip:
            non_skip_pair_space += pair_space
        per_edge.append(
            {
                "edge_id": eid,
                "source": str(edge.source),
                "target": str(edge.target),
                "source_keyword": source_keyword,
                "target_keyword": target_keyword,
                "sign": edge.sign.value,
                "lower": edge.lower,
                "upper": edge.upper,
                "pair_space": pair_space,
                "skip_edge": is_skip,
                "ematches": ematches,
                "density": (ematches / pair_space) if ematches is not None and pair_space > 0 else None,
            }
        )

    measured_edges = sum(ematch_counts.values()) if stats is not None else None
    return {
        "keyword_postings_by_vertex": postings_by_vertex,
        "keyword_postings_min_by_vertex": min(posting_values) if posting_values else 0,
        "keyword_postings_mean_by_vertex": statistics.fmean(posting_values) if posting_values else 0.0,
        "keyword_postings_max_by_vertex": max(posting_values) if posting_values else 0,
        "candidate_pair_space_total": total_pair_space,
        "candidate_pair_space_non_skip": non_skip_pair_space,
        "candidate_pair_space_max_edge": max((edge["pair_space"] for edge in per_edge), default=0),
        "candidate_pair_space_mean_edge": statistics.fmean(edge["pair_space"] for edge in per_edge) if per_edge else 0.0,
        "candidate_graph_edges_measured": measured_edges,
        "candidate_graph_density_measured": (
            measured_edges / non_skip_pair_space
            if measured_edges is not None and non_skip_pair_space > 0
            else None
        ),
        "candidate_graph_edges_by_edge": per_edge,
    }


def _count_tree_nodes(root: Any) -> Tuple[int, int, int]:
    """Return (nodes, leaves, max_depth) for one octree."""

    nodes = 0
    leaves = 0
    max_depth = 0
    stack = [root]
    while stack:
        node = stack.pop()
        nodes += 1
        max_depth = max(max_depth, node.level)
        if getattr(node, "children", None):
            stack.extend(node.children)
        else:
            leaves += 1
    return nodes, leaves, max_depth


def index_summary(index: InvertedOctreeIndex) -> Dict[str, Any]:
    postings = [len(items) for items in index.by_keyword.values()]
    node_count = 0
    leaf_count = 0
    max_depth = 0
    for tree in index.trees.values():
        nodes, leaves, depth = _count_tree_nodes(tree.root)
        node_count += nodes
        leaf_count += leaves
        max_depth = max(max_depth, depth)
    return {
        "distinct_keywords": len(index.by_keyword),
        "keyword_postings_total": sum(postings),
        "keyword_postings_mean": statistics.fmean(postings) if postings else 0.0,
        "keyword_postings_max": max(postings) if postings else 0,
        "index_tree_count": len(index.trees),
        "index_node_count": node_count,
        "index_leaf_count": leaf_count,
        "index_max_depth": max_depth,
    }


def _scale_pattern(named: NamedPattern, interval_scale: float) -> NamedPattern:
    if interval_scale == 1.0:
        return named
    if interval_scale < 1.0:
        raise ValueError("interval scales must be >= 1.0 so planted templates remain valid")
    edges: List[PatternEdge] = []
    for edge in named.pattern.edges:
        width = edge.upper - edge.lower
        upper = edge.lower + width * interval_scale
        edges.append(PatternEdge(edge.source, edge.target, edge.lower, upper, edge.sign))
    label = str(interval_scale).replace(".", "p")
    return NamedPattern(
        name=f"{named.name}__interval_x{label}",
        description=f"{named.description} Interval upper bounds scaled by {interval_scale:g}.",
        pattern=SpatialPattern(dict(named.pattern.vertices), tuple(edges)),
        template_points=dict(named.template_points),
    )


def build_benchmark_patterns(pattern_count: int, interval_scales: Sequence[float]) -> List[Tuple[NamedPattern, float]]:
    base = default_pattern_suite()[:pattern_count]
    out: List[Tuple[NamedPattern, float]] = []
    for scale in interval_scales:
        for pattern in base:
            out.append((_scale_pattern(pattern, float(scale)), float(scale)))
    return out


def _domain_metrics(bounds: Box3D, n_objects: int) -> Dict[str, Optional[float]]:
    side_lengths = [bounds.maxs[d] - bounds.mins[d] for d in range(3)]
    volume = side_lengths[0] * side_lengths[1] * side_lengths[2]
    density = n_objects / volume if volume > 0 else None
    expected_spacing = (volume / n_objects) ** (1.0 / 3.0) if volume > 0 and n_objects > 0 else None
    return {"domain_volume": volume, "object_density": density, "expected_spacing": expected_spacing}


def _write_csv_row(writer: csv.DictWriter, row: Mapping[str, Any]) -> None:
    converted: Dict[str, Any] = {}
    for field in RESULT_FIELDS:
        value = row.get(field)
        if isinstance(value, (dict, list, tuple, set)):
            converted[field] = _json(sorted(value) if isinstance(value, set) else value)
        else:
            converted[field] = value
    writer.writerow(converted)


def _write_jsonl_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()


def _make_output_paths(output_dir: str | Path, run_id: str) -> Dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "settings": out / f"{run_id}_settings.json",
        "results_jsonl": out / f"{run_id}_results.jsonl",
        "results_csv": out / f"{run_id}_results.csv",
        "summary_json": out / f"{run_id}_summary.json",
        "summary_csv": out / f"{run_id}_summary.csv",
    }


def system_info() -> Dict[str, Any]:
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "psutil_available": psutil is not None,
    }
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            info["system_memory_total_mb"] = vm.total / (1024.0 * 1024.0)
        except Exception:
            pass
    return info


def _resolve_sparsity_controls(config: BenchmarkConfig, profile_name: str) -> Dict[str, Any]:
    """Return concrete generator controls for one sparsity profile.

    ``manual`` uses the flat BenchmarkConfig fields.  Built-in profiles use the
    preset values so the profile name has stable meaning across runs.
    """

    if profile_name == "manual":
        return {
            "sparsity_profile": "manual",
            "sparsity_description": SPARSITY_PROFILES["manual"].description,
            "bounds": config.bounds,
            "domain_side": config.domain_side,
            "target_density": config.target_density,
            "n_noise_keywords": config.n_noise_keywords,
            "clustered_fraction": config.clustered_fraction,
            "cluster_count": config.cluster_count,
            "cluster_std_fraction": config.cluster_std_fraction,
            "extra_keyword_probability": config.extra_keyword_probability,
            "max_keywords_per_object": config.max_keywords_per_object,
            "pattern_keyword_weight": config.pattern_keyword_weight,
            "noise_keyword_weight": config.noise_keyword_weight,
        }

    profile = SPARSITY_PROFILES[profile_name]
    return {
        "sparsity_profile": profile.name,
        "sparsity_description": profile.description,
        "bounds": None,
        "domain_side": profile.domain_side,
        "target_density": profile.target_density,
        "n_noise_keywords": profile.n_noise_keywords,
        "clustered_fraction": profile.clustered_fraction,
        "cluster_count": profile.cluster_count,
        "cluster_std_fraction": profile.cluster_std_fraction,
        "extra_keyword_probability": profile.extra_keyword_probability,
        "max_keywords_per_object": profile.max_keywords_per_object,
        "pattern_keyword_weight": profile.pattern_keyword_weight,
        "noise_keyword_weight": profile.noise_keyword_weight,
    }


def _iter_scenarios(config: BenchmarkConfig) -> List[Tuple[int, int, str, Dict[str, Any]]]:
    scenarios: List[Tuple[int, int, str, Dict[str, Any]]] = []
    scenario_index = 0
    for n_objects in config.scales:
        for profile_name in config.sparsity_profiles:
            if profile_name not in SPARSITY_PROFILES:
                raise ValueError(f"unknown sparsity profile: {profile_name!r}")
            scenario_index += 1
            scenarios.append((scenario_index, n_objects, profile_name, _resolve_sparsity_controls(config, profile_name)))
    return scenarios


def _numeric_values(rows: Iterable[Mapping[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
            out.append(float(value))
    return out


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _safe_median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _safe_max(values: Sequence[float]) -> Optional[float]:
    return max(values) if values else None


def _scenario_summary(
    *,
    run_id: str,
    scenario_id: str,
    n_objects: int,
    controls: Mapping[str, Any],
    generation_time_s: Optional[float],
    index_build_time_s: Optional[float],
    scenario_total_time_s: float,
    rss_peak_scenario_mb: Optional[float],
    idx_summary: Mapping[str, Any],
    domain_metrics: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") not in (None, "ok")]

    match_times = _numeric_values(ok_rows, "match_time_s")
    matches = _numeric_values(ok_rows, "matches_returned")
    ematches = _numeric_values(ok_rows, "ematch_total")
    nmatch_final = _numeric_values(ok_rows, "nmatch_total_final_level")
    graph_density = _numeric_values(ok_rows, "candidate_graph_density_measured")
    pair_space_total = _numeric_values(ok_rows, "candidate_pair_space_total")
    pair_space_non_skip = _numeric_values(ok_rows, "candidate_pair_space_non_skip")
    skip_counts = _numeric_values(ok_rows, "skip_edge_count")
    limit_reached_count = sum(1 for row in ok_rows if row.get("limit_reached"))

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "n_objects": n_objects,
        "sparsity_profile": controls["sparsity_profile"],
        "sparsity_description": controls["sparsity_description"],
        "generation_time_s": generation_time_s,
        "index_build_time_s": index_build_time_s,
        "scenario_total_time_s": scenario_total_time_s,
        "rss_peak_scenario_mb": rss_peak_scenario_mb,
        "patterns_attempted": len(rows),
        "patterns_ok": len(ok_rows),
        "patterns_error": len(error_rows),
        "limit_reached_count": limit_reached_count,
        "match_time_total_s": sum(match_times) if match_times else None,
        "match_time_mean_s": _safe_mean(match_times),
        "match_time_median_s": _safe_median(match_times),
        "match_time_max_s": _safe_max(match_times),
        "matches_returned_total": int(sum(matches)) if matches else None,
        "matches_returned_mean": _safe_mean(matches),
        "ematch_total_mean": _safe_mean(ematches),
        "nmatch_total_final_level_mean": _safe_mean(nmatch_final),
        "skip_edge_count_mean": _safe_mean(skip_counts),
        "candidate_pair_space_total_mean": _safe_mean(pair_space_total),
        "candidate_pair_space_non_skip_mean": _safe_mean(pair_space_non_skip),
        "candidate_graph_density_mean": _safe_mean(graph_density),
        **idx_summary,
        **domain_metrics,
        "n_noise_keywords": controls["n_noise_keywords"],
        "clustered_fraction": controls["clustered_fraction"],
        "cluster_count": controls["cluster_count"],
        "cluster_std_fraction": controls["cluster_std_fraction"],
        "extra_keyword_probability": controls["extra_keyword_probability"],
        "max_keywords_per_object": controls["max_keywords_per_object"],
        "pattern_keyword_weight": controls["pattern_keyword_weight"],
        "noise_keyword_weight": controls["noise_keyword_weight"],
    }


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Path]:
    """Run the benchmark and return paths of written result files."""

    run_id = datetime.now(timezone.utc).strftime("espm3d_%Y%m%dT%H%M%SZ")
    paths = _make_output_paths(config.output_dir, run_id)
    patterns_with_scale = build_benchmark_patterns(config.pattern_count, config.interval_scales)
    scenarios = _iter_scenarios(config)

    settings = {
        "run_id": run_id,
        "config": asdict(config),
        "system": system_info(),
        "pattern_names": [p.name for p, _ in patterns_with_scale],
        "sparsity_profile_definitions": {name: asdict(profile) for name, profile in SPARSITY_PROFILES.items()},
        "profile_sparsity_sweeps": dict(PROFILE_SPARSITY_SWEEPS),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    paths["settings"].write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")

    summary_rows: List[Dict[str, Any]] = []
    with paths["results_jsonl"].open("w", encoding="utf-8") as jsonl, paths["results_csv"].open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        csv_writer.writeheader()

        for scenario_index, n_objects, profile_name, controls in scenarios:
            scenario_id = f"n{n_objects}__{profile_name}"
            print(
                f"\n=== Scenario {scenario_index}/{len(scenarios)}: "
                f"n_objects={n_objects:,}, sparsity={profile_name} ===",
                flush=True,
            )
            scenario_started = time.perf_counter()
            bounds = bounds_from_sparsity_controls(
                n_objects,
                bounds_values=controls["bounds"],
                domain_side=controls["domain_side"],
                target_density=controls["target_density"],
            )
            domain_metrics = _domain_metrics(bounds, n_objects)
            scenario_memory = MemorySampler()
            dataset = None
            index = None
            idx_summary: Dict[str, Any] = {}
            metadata: Dict[str, Any] = {}
            generation_time_s: Optional[float] = None
            index_build_time_s: Optional[float] = None
            rss_before = current_rss_mb()
            rss_after_generation = None
            rss_after_index = None
            scenario_rows: List[Dict[str, Any]] = []
            scenario_seed = config.seed + scenario_index - 1

            with scenario_memory:
                try:
                    with Timer() as generation_timer:
                        dataset = generate_synthetic_3d_dataset(
                            n_objects,
                            seed=scenario_seed,
                            bounds=bounds,
                            patterns=[p for p, _ in patterns_with_scale],
                            ensure_patterns=config.ensure_patterns,
                            planted_matches_per_pattern=config.matches_per_pattern,
                            n_noise_keywords=controls["n_noise_keywords"],
                            clustered_fraction=controls["clustered_fraction"],
                            cluster_count=controls["cluster_count"],
                            cluster_std_fraction=controls["cluster_std_fraction"],
                            extra_keyword_probability=controls["extra_keyword_probability"],
                            max_keywords_per_object=controls["max_keywords_per_object"],
                            pattern_keyword_weight=controls["pattern_keyword_weight"],
                            noise_keyword_weight=controls["noise_keyword_weight"],
                            validate_planted=False,
                        )
                    generation_time_s = generation_timer.elapsed_s
                    rss_after_generation = current_rss_mb()
                    metadata = dict(dataset.metadata)
                    if config.write_dataset:
                        from .generate_synthetic_data import save_objects_jsonl, save_patterns_json, save_planted_matches_json

                        scenario_dir = Path(config.output_dir) / f"{run_id}_{scenario_id}_dataset"
                        scenario_dir.mkdir(parents=True, exist_ok=True)
                        save_objects_jsonl(dataset.objects, scenario_dir / "objects.jsonl")
                        save_patterns_json(dataset.patterns, scenario_dir / "patterns.json")
                        save_planted_matches_json(dataset.planted_matches, scenario_dir / "planted_matches.json")
                        (scenario_dir / "metadata.json").write_text(
                            json.dumps(dataset.metadata, indent=2, sort_keys=True), encoding="utf-8"
                        )

                    with Timer() as index_timer:
                        index = InvertedOctreeIndex(
                            dataset.objects,
                            capacity=config.capacity,
                            min_level=config.min_level,
                            max_level=config.max_level,
                        )
                    index_build_time_s = index_timer.elapsed_s
                    rss_after_index = current_rss_mb()
                    idx_summary = index_summary(index)
                    matcher = ESPM3DMatcher(index, require_distinct_objects=config.require_distinct_objects)

                    for pattern_index, (named_pattern, interval_scale) in enumerate(patterns_with_scale, start=1):
                        print(
                            f"  [{pattern_index:03d}/{len(patterns_with_scale):03d}] {named_pattern.name}",
                            flush=True,
                        )
                        match_memory = MemorySampler()
                        row: Dict[str, Any] = {
                            "run_id": run_id,
                            "scenario_id": scenario_id,
                            "status": "ok",
                            "error_type": None,
                            "error_message": None,
                            "n_objects": n_objects,
                            "sparsity_profile": controls["sparsity_profile"],
                            "sparsity_description": controls["sparsity_description"],
                            "pattern_index": pattern_index,
                            **_pattern_summary(named_pattern, interval_scale),
                            "match_limit": config.match_limit,
                            "generation_time_s": generation_time_s,
                            "index_build_time_s": index_build_time_s,
                            "rss_before_mb": rss_before,
                            "rss_after_generation_mb": rss_after_generation,
                            "rss_after_index_mb": rss_after_index,
                            **idx_summary,
                            **domain_metrics,
                            "n_noise_keywords": controls["n_noise_keywords"],
                            "clustered_fraction": controls["clustered_fraction"],
                            "cluster_count": metadata.get("cluster_count", controls["cluster_count"]),
                            "cluster_std_fraction": controls["cluster_std_fraction"],
                            "extra_keyword_probability": controls["extra_keyword_probability"],
                            "max_keywords_per_object": controls["max_keywords_per_object"],
                            "pattern_keyword_weight": controls["pattern_keyword_weight"],
                            "noise_keyword_weight": controls["noise_keyword_weight"],
                            "capacity": config.capacity,
                            "min_level": config.min_level,
                            "max_level": config.max_level,
                            "require_distinct_objects": config.require_distinct_objects,
                            "seed": scenario_seed,
                        }
                        try:
                            with match_memory, Timer() as match_timer:
                                matches = matcher.match(named_pattern.pattern, limit=config.match_limit)
                            stats = matcher.last_stats
                            row.update(
                                {
                                    "matches_returned": len(matches),
                                    "limit_reached": bool(config.match_limit and len(matches) >= config.match_limit),
                                    "match_time_s": match_timer.elapsed_s,
                                    "rss_after_match_mb": current_rss_mb(),
                                    "rss_peak_match_mb": match_memory.peak_rss_mb,
                                    **_stats_summary(stats),
                                    **_candidate_graph_summary(
                                        named_pattern.pattern,
                                        index,
                                        stats,
                                        require_distinct_objects=config.require_distinct_objects,
                                    ),
                                }
                            )
                        except Exception as exc:  # keep the benchmark log readable even when one pattern fails
                            stats = matcher.last_stats
                            row.update(
                                {
                                    "status": "error",
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc),
                                    "matches_returned": None,
                                    "limit_reached": None,
                                    "match_time_s": None,
                                    "rss_after_match_mb": current_rss_mb(),
                                    "rss_peak_match_mb": match_memory.peak_rss_mb,
                                    **_stats_summary(stats),
                                    **_candidate_graph_summary(
                                        named_pattern.pattern,
                                        index,
                                        stats,
                                        require_distinct_objects=config.require_distinct_objects,
                                    ),
                                    "traceback": traceback.format_exc(),
                                }
                            )
                        row["scenario_total_time_s"] = time.perf_counter() - scenario_started
                        row["rss_peak_scenario_mb"] = scenario_memory.peak_rss_mb
                        scenario_rows.append(dict(row))
                        _write_jsonl_row(jsonl, row)
                        _write_csv_row(csv_writer, row)
                        csv_file.flush()

                except BaseException as exc:
                    # Generation/index errors are scenario-level failures.  Write one row and continue when possible.
                    row = {
                        "run_id": run_id,
                        "scenario_id": scenario_id,
                        "status": "scenario_error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "n_objects": n_objects,
                        "sparsity_profile": controls["sparsity_profile"],
                        "sparsity_description": controls["sparsity_description"],
                        "pattern_index": None,
                        "pattern_name": None,
                        "pattern_description": None,
                        "pattern_vertices": None,
                        "pattern_edges": None,
                        "exclusive_edges": None,
                        "interval_scale": None,
                        "matches_returned": None,
                        "match_limit": config.match_limit,
                        "limit_reached": None,
                        "generation_time_s": generation_time_s,
                        "index_build_time_s": index_build_time_s,
                        "match_time_s": None,
                        "scenario_total_time_s": time.perf_counter() - scenario_started,
                        "rss_before_mb": rss_before,
                        "rss_after_generation_mb": rss_after_generation,
                        "rss_after_index_mb": rss_after_index,
                        "rss_after_match_mb": current_rss_mb(),
                        "rss_peak_scenario_mb": scenario_memory.peak_rss_mb,
                        "rss_peak_match_mb": None,
                        **idx_summary,
                        **_stats_summary(None),
                        **_empty_candidate_graph_summary(),
                        **domain_metrics,
                        "n_noise_keywords": controls["n_noise_keywords"],
                        "clustered_fraction": controls["clustered_fraction"],
                        "cluster_count": controls["cluster_count"],
                        "cluster_std_fraction": controls["cluster_std_fraction"],
                        "extra_keyword_probability": controls["extra_keyword_probability"],
                        "max_keywords_per_object": controls["max_keywords_per_object"],
                        "pattern_keyword_weight": controls["pattern_keyword_weight"],
                        "noise_keyword_weight": controls["noise_keyword_weight"],
                        "capacity": config.capacity,
                        "min_level": config.min_level,
                        "max_level": config.max_level,
                        "require_distinct_objects": config.require_distinct_objects,
                        "seed": scenario_seed,
                        "traceback": traceback.format_exc(),
                    }
                    scenario_rows.append(dict(row))
                    _write_jsonl_row(jsonl, row)
                    _write_csv_row(csv_writer, row)
                    csv_file.flush()
                    print(f"Scenario failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                finally:
                    scenario_total = time.perf_counter() - scenario_started
                    summary_rows.append(
                        _scenario_summary(
                            run_id=run_id,
                            scenario_id=scenario_id,
                            n_objects=n_objects,
                            controls={**controls, "cluster_count": metadata.get("cluster_count", controls["cluster_count"])},
                            generation_time_s=generation_time_s,
                            index_build_time_s=index_build_time_s,
                            scenario_total_time_s=scenario_total,
                            rss_peak_scenario_mb=scenario_memory.peak_rss_mb,
                            idx_summary=idx_summary,
                            domain_metrics=domain_metrics,
                            rows=scenario_rows,
                        )
                    )
                    del index
                    del dataset
                    gc.collect()

    paths["summary_json"].write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    with paths["summary_csv"].open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in summary_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nWrote benchmark outputs:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return paths


def _parse_scales(values: Optional[Sequence[str]], profile: str) -> Tuple[int, ...]:
    if not values:
        return PROFILE_SCALES[profile]
    out: List[int] = []
    for value in values:
        value = value.replace(",", "").strip().lower()
        multiplier = 1
        if value.endswith("k"):
            multiplier = 1_000
            value = value[:-1]
        elif value.endswith("m"):
            multiplier = 1_000_000
            value = value[:-1]
        out.append(int(float(value) * multiplier))
    return tuple(out)


def _print_sparsity_profiles() -> None:
    print("Available sparsity profiles:\n")
    for name in sorted(SPARSITY_PROFILES):
        profile = SPARSITY_PROFILES[name]
        print(f"{name}")
        print(f"  {profile.description}")
        if name != "manual":
            print(
                "  controls: "
                f"domain_side={profile.domain_side}, "
                f"noise_keywords={profile.n_noise_keywords}, "
                f"clustered_fraction={profile.clustered_fraction}, "
                f"cluster_count={profile.cluster_count}, "
                f"cluster_std_fraction={profile.cluster_std_fraction}, "
                f"extra_keyword_probability={profile.extra_keyword_probability}, "
                f"max_keywords_per_object={profile.max_keywords_per_object}, "
                f"pattern_keyword_weight={profile.pattern_keyword_weight}, "
                f"noise_keyword_weight={profile.noise_keyword_weight}"
            )
        print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ESPM-3D scalability benchmarks on synthetic 3-D data."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_SCALES),
        default="standard",
        help="choose default scales and default sparsity-profile sweep",
    )
    parser.add_argument(
        "--scales",
        nargs="*",
        help="object counts; accepts forms like 10000, 100k, 1m. Overrides the scale portion of --profile.",
    )
    parser.add_argument(
        "--sparsity-profiles",
        nargs="+",
        choices=sorted(SPARSITY_PROFILES),
        default=None,
        help=(
            "candidate-graph sparsity presets to sweep. Defaults are profile-specific: "
            "smoke/standard use sparse_graph, medium_graph, dense_graph; full uses "
            "very_sparse_graph, sparse_graph, medium_graph. Use 'manual' to use the "
            "explicit generator flags instead of a preset."
        ),
    )
    parser.add_argument("--list-sparsity-profiles", action="store_true", help="print sparsity profile definitions and exit")
    parser.add_argument("--pattern-count", type=int, default=20)
    parser.add_argument(
        "--interval-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="run pattern variants with enlarged upper distance bounds; values must be >= 1.0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--match-limit", type=int, default=1000, help="cap returned matches per pattern; use 0 for no cap")
    parser.add_argument("--no-plant", action="store_true", help="do not plant guaranteed pattern matches")
    parser.add_argument("--matches-per-pattern", type=int, default=1)

    # Synthetic data controls.  These are used directly only with --sparsity-profiles manual.
    parser.add_argument("--noise-keywords", type=int, default=128)
    parser.add_argument("--clustered-fraction", type=float, default=0.65)
    parser.add_argument("--cluster-count", type=int, default=None)
    parser.add_argument("--cluster-std-fraction", type=float, default=0.035)
    parser.add_argument("--extra-keyword-probability", type=float, default=0.05)
    parser.add_argument("--max-keywords-per-object", type=int, default=3)
    parser.add_argument("--pattern-keyword-weight", type=float, default=3.0)
    parser.add_argument("--noise-keyword-weight", type=float, default=1.0)
    parser.add_argument("--bounds", type=float, nargs=6, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
    parser.add_argument("--domain-side", type=float, default=None, help="use cubic domain [0, side]^3")
    parser.add_argument("--target-density", type=float, default=None, help="objects per cubic unit")

    # Index/matcher controls.
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--min-level", type=int, default=1)
    parser.add_argument("--max-level", type=int, default=10)
    parser.add_argument("--allow-same-object", action="store_true", help="allow one object to satisfy multiple pattern vertices")
    parser.add_argument("--write-dataset", action="store_true", help="also write generated objects/patterns for each scenario")
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    match_limit = None if args.match_limit == 0 else args.match_limit
    sparsity_profiles = tuple(args.sparsity_profiles or PROFILE_SPARSITY_SWEEPS[args.profile])
    return BenchmarkConfig(
        scales=_parse_scales(args.scales, args.profile),
        sparsity_profiles=sparsity_profiles,
        pattern_count=args.pattern_count,
        interval_scales=tuple(args.interval_scales),
        seed=args.seed,
        output_dir=args.output_dir,
        match_limit=match_limit,
        ensure_patterns=not args.no_plant,
        matches_per_pattern=args.matches_per_pattern,
        n_noise_keywords=args.noise_keywords,
        clustered_fraction=args.clustered_fraction,
        cluster_count=args.cluster_count,
        cluster_std_fraction=args.cluster_std_fraction,
        extra_keyword_probability=args.extra_keyword_probability,
        max_keywords_per_object=args.max_keywords_per_object,
        pattern_keyword_weight=args.pattern_keyword_weight,
        noise_keyword_weight=args.noise_keyword_weight,
        bounds=tuple(args.bounds) if args.bounds else None,  # type: ignore[arg-type]
        domain_side=args.domain_side,
        target_density=args.target_density,
        capacity=args.capacity,
        min_level=args.min_level,
        max_level=args.max_level,
        require_distinct_objects=not args.allow_same_object,
        write_dataset=args.write_dataset,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.list_sparsity_profiles:
        _print_sparsity_profiles()
        return
    config = config_from_args(args)
    if any(scale < 1 for scale in config.scales):
        parser.error("all scales must be positive")
    if config.pattern_count < 1:
        parser.error("--pattern-count must be positive")
    if any(scale < 1.0 for scale in config.interval_scales):
        parser.error("--interval-scales values must be >= 1.0")
    if any(name not in SPARSITY_PROFILES for name in config.sparsity_profiles):
        parser.error(f"unknown sparsity profile in {config.sparsity_profiles!r}")
    run_benchmark(config)


if __name__ == "__main__":
    main()
