"""Convert the Hamburg buildings/facades/amenities/trees CSV to ESPM-3D JSONL.

The Hamburg file is not a point dataset.  Most rows describe an axis-aligned
rectangular prism or a 2-D area-like feature through ``lowerCorner_*`` and
``upperCorner_*`` columns.  This converter turns each such row into point-like
spatio-textual objects that the ESPM-3D matcher can index:

* a bottom-center object, e.g. ``<gml_id>:bottom``;
* a top-center object, e.g. ``<gml_id>:top``;
* optionally a centroid object; and
* optional facade opening objects for rows with ``opening_type``.

The output object JSONL is compatible with
``espm3d.generate_synthetic_data.load_objects_jsonl``.  The pattern JSON is
compatible with ``load_patterns_json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .generate_synthetic_data import NamedPattern, load_patterns_json, save_patterns_json
from .matcher import ESPM3DMatcher, InvertedOctreeIndex, PatternEdge, SpatialObject, SpatialPattern

Point3D = Tuple[float, float, float]

_REQUIRED_COLUMNS = {
    "gml_id",
    "feature_type",
    "upperCorner_x",
    "upperCorner_y",
    "upperCorner_z",
    "lowerCorner_x",
    "lowerCorner_y",
    "lowerCorner_z",
}

_OPTIONAL_COLUMNS = {
    "material",
    "roofType",
    "measured_height",
    "storeysAboveGround",
    "opening_type",
    "opening_x1",
    "opening_x2",
    "opening_y1",
    "opening_y2",
    "opening_z",
    "osm_building_id",
    "osm_building_tags",
    "amenities",
    "poi_count",
    "building_use",
    "name",
}

_ROOF_TYPE_NAMES = {
    "1000": "roof_flat",
    "1010": "roof_monopitch",
    "1020": "roof_duopitch",
    "1030": "roof_hipped",
    "1040": "roof_half_hipped",
    "1070": "roof_mansard",
    "2100": "roof_pent",
    "3100": "roof_gabled",
    "3200": "roof_hipped",
    "3500": "roof_mansard",
    "4000": "roof_arch",
    "9999": "roof_unknown",
}


@dataclass(frozen=True)
class ConvertStats:
    rows_read: int
    base_features_seen: int
    base_features_written: int
    objects_written: int
    opening_objects_written: int
    centroid_objects_written: int
    duplicate_base_rows_skipped: int
    rows_missing_required_geometry: int
    rows_with_missing_z_imputed: int
    keyword_counts: Counter[str]
    feature_type_counts: Counter[str]
    role_counts: Counter[str]
    bounds_before_origin_shift: Dict[str, List[float]]
    origin: Point3D


def hamburg_pattern_suite() -> List[NamedPattern]:
    """Return five suggested patterns for converted Hamburg data.

    Distances are in the same projected units as the CSV coordinates, which are
    effectively metres for the intended use of this file.  The patterns use
    moderately selective keywords so they are more suitable as a real-data
    baseline than very broad patterns like ``building_bottom`` joined with all
    ``building_top`` objects.
    """

    return [
        NamedPattern(
            name="hamburg_p01_highrise_vertical_extent",
            description=(
                "A high-rise bottom and high-rise top separated by a plausible vertical height. "
                "This checks that prism endpoints preserve vertical information without joining every building."
            ),
            pattern=SpatialPattern(
                {"B0": "highrise_bottom", "BT": "highrise_top"},
                [PatternEdge("B0", "BT", 30.0, 180.0, "--")],
            ),
            template_points={"B0": (0.0, 0.0, 0.0), "BT": (0.0, 0.0, 60.0)},
        ),
        NamedPattern(
            name="hamburg_p02_facade_door_window_stack",
            description=(
                "A facade door lower endpoint, facade window upper endpoint, and building top. "
                "Useful for rows that contain Door/Window facade annotations."
            ),
            pattern=SpatialPattern(
                {"D": "door_bottom", "W": "window_top", "BT": "building_top"},
                [
                    PatternEdge("D", "W", 1.0, 45.0, "--"),
                    PatternEdge("W", "BT", 0.0, 120.0, "--"),
                    PatternEdge("D", "BT", 2.0, 140.0, "--"),
                ],
            ),
            template_points={"D": (0.0, 0.0, 1.0), "W": (0.0, 0.0, 15.0), "BT": (0.0, 0.0, 30.0)},
        ),
        NamedPattern(
            name="hamburg_p03_hospital_green_buffer",
            description="Hospital near a park and nearby trees.",
            pattern=SpatialPattern(
                {"H": "hospital_bottom", "P": "park_bottom", "T": "tree_bottom"},
                [
                    PatternEdge("H", "P", 0.0, 800.0, "--"),
                    PatternEdge("H", "T", 0.0, 120.0, "--"),
                    PatternEdge("P", "T", 0.0, 300.0, "--"),
                ],
            ),
            template_points={"H": (0.0, 0.0, 0.0), "P": (350.0, 0.0, 0.0), "T": (60.0, 0.0, 0.0)},
        ),
        NamedPattern(
            name="hamburg_p04_education_play_cluster",
            description="School and kindergarten near the same playground.",
            pattern=SpatialPattern(
                {"S": "school_bottom", "K": "kindergarten_bottom", "G": "playground_bottom"},
                [
                    PatternEdge("S", "G", 0.0, 500.0, "--"),
                    PatternEdge("K", "G", 0.0, 500.0, "--"),
                    PatternEdge("S", "K", 0.0, 1000.0, "--"),
                ],
            ),
            template_points={"S": (0.0, 0.0, 0.0), "K": (250.0, 0.0, 0.0), "G": (150.0, 0.0, 0.0)},
        ),
        NamedPattern(
            name="hamburg_p05_transit_retail_parking",
            description="Train station, retail building, and parking feature in a local-access pattern.",
            pattern=SpatialPattern(
                {"T": "train_station_bottom", "R": "retail_bottom", "P": "parking_bottom"},
                [
                    PatternEdge("T", "R", 0.0, 1200.0, "--"),
                    PatternEdge("R", "P", 0.0, 300.0, "--"),
                    PatternEdge("T", "P", 0.0, 1200.0, "--"),
                ],
            ),
            template_points={"T": (0.0, 0.0, 0.0), "R": (500.0, 0.0, 0.0), "P": (650.0, 0.0, 0.0)},
        ),
    ]


def convert_hamburg_csv(
    input_csv: str | Path,
    objects_out: str | Path,
    *,
    metadata_out: str | Path | None = None,
    patterns_out: str | Path | None = None,
    include_centroid: bool = False,
    opening_policy: str = "endpoints",
    flat_z: float = 0.0,
    flat_thickness: float = 1.0,
    normalize_origin: bool = True,
    max_rows: int | None = None,
    dedupe_base_features: bool = True,
    max_keywords_per_object: int = 128,
) -> ConvertStats:
    """Convert Hamburg CSV rows to ESPM-3D object JSONL.

    Parameters
    ----------
    input_csv:
        Source Hamburg CSV.
    objects_out:
        Destination JSONL path.  Each line contains ``id``, ``x``, ``y``, ``z``
        and ``keywords``.
    metadata_out:
        Optional JSON file with conversion statistics and coordinate metadata.
    patterns_out:
        Optional JSON file containing five suggested Hamburg patterns.
    include_centroid:
        If true, also emits one center point per base feature.  The default is
        false because bottom/top endpoints preserve vertical extent better.
    opening_policy:
        ``endpoints`` emits bottom and top objects for each facade opening;
        ``point`` emits one opening-center object; ``skip`` ignores openings.
    flat_z, flat_thickness:
        Used for area features with missing z, such as parks and water.  They
        receive bottom z = ``flat_z`` and top z = ``flat_z + flat_thickness``.
    normalize_origin:
        If true, subtract the minimum generated x/y/z from all points.  This
        keeps distances unchanged but makes coordinates much smaller.
    max_rows:
        Optional debugging limit on input rows.
    dedupe_base_features:
        If true, emit bottom/top/centroid once per ``gml_id``.  Opening rows are
        still processed individually.
    max_keywords_per_object:
        Hard cap on keyword count per emitted object.  High-priority structural
        keywords are added first.
    """

    input_csv = Path(input_csv)
    objects_out = Path(objects_out)
    if patterns_out is not None:
        save_patterns_json(hamburg_pattern_suite(), patterns_out)

    if opening_policy not in {"endpoints", "point", "skip"}:
        raise ValueError("opening_policy must be one of: endpoints, point, skip")
    if flat_thickness < 0:
        raise ValueError("flat_thickness must be non-negative")
    if max_keywords_per_object < 1:
        raise ValueError("max_keywords_per_object must be positive")

    header = _read_header(input_csv)
    missing = sorted(_REQUIRED_COLUMNS - set(header))
    if missing:
        raise ValueError(f"input CSV is missing required columns: {missing}")

    origin, bounds, first_pass_counts = _infer_origin_and_bounds(
        input_csv,
        opening_policy=opening_policy,
        flat_z=flat_z,
        flat_thickness=flat_thickness,
        normalize_origin=normalize_origin,
        max_rows=max_rows,
        dedupe_base_features=dedupe_base_features,
    )

    objects_out.parent.mkdir(parents=True, exist_ok=True)
    seen_base: set[str] = set()
    rows_read = 0
    base_seen = 0
    base_written = 0
    objects_written = 0
    opening_objects_written = 0
    centroid_objects_written = 0
    duplicate_skipped = 0
    missing_geom = 0
    missing_z_imputed = 0
    keyword_counts: Counter[str] = Counter()
    feature_type_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()

    with objects_out.open("w", encoding="utf-8") as out:
        for row_number, row in _iter_rows(input_csv, max_rows=max_rows):
            rows_read += 1
            gml_id = _safe_id(row.get("gml_id"), fallback=f"row_{row_number}")
            feature_type = _clean_keyword(row.get("feature_type")) or "feature"
            feature_type_counts[feature_type] += 1

            base_points = _base_points(row, flat_z=flat_z, flat_thickness=flat_thickness)
            if base_points is None:
                missing_geom += 1
            else:
                base_seen += 1
                _, _, z_was_imputed = base_points
                if z_was_imputed:
                    missing_z_imputed += 1
                if (not dedupe_base_features) or gml_id not in seen_base:
                    seen_base.add(gml_id)
                    semantic_keywords = _base_semantic_keywords(row)
                    bottom, top, _ = base_points
                    base_written += 1
                    for role, point in (("bottom", bottom), ("top", top)):
                        keywords = _role_keywords(semantic_keywords, role, max_keywords_per_object)
                        object_id = f"{gml_id}:{role}"
                        _write_object(
                            out,
                            object_id,
                            _shift(point, origin),
                            keywords,
                            source={"gml_id": gml_id, "row_number": row_number, "entity": role, "feature_type": feature_type},
                        )
                        objects_written += 1
                        role_counts[role] += 1
                        keyword_counts.update(keywords)
                    if include_centroid:
                        center = tuple((bottom[i] + top[i]) * 0.5 for i in range(3))  # type: ignore[assignment]
                        keywords = _role_keywords(semantic_keywords, "centroid", max_keywords_per_object)
                        _write_object(
                            out,
                            f"{gml_id}:centroid",
                            _shift(center, origin),
                            keywords,
                            source={"gml_id": gml_id, "row_number": row_number, "entity": "centroid", "feature_type": feature_type},
                        )
                        objects_written += 1
                        centroid_objects_written += 1
                        role_counts["centroid"] += 1
                        keyword_counts.update(keywords)
                else:
                    duplicate_skipped += 1

            if opening_policy != "skip":
                for object_id, role, point, keywords in _opening_objects(
                    row,
                    row_number=row_number,
                    gml_id=gml_id,
                    opening_policy=opening_policy,
                    max_keywords_per_object=max_keywords_per_object,
                ):
                    _write_object(
                        out,
                        object_id,
                        _shift(point, origin),
                        keywords,
                        source={"gml_id": gml_id, "row_number": row_number, "entity": role, "feature_type": feature_type},
                    )
                    objects_written += 1
                    opening_objects_written += 1
                    role_counts[role] += 1
                    keyword_counts.update(keywords)

    stats = ConvertStats(
        rows_read=rows_read,
        base_features_seen=base_seen,
        base_features_written=base_written,
        objects_written=objects_written,
        opening_objects_written=opening_objects_written,
        centroid_objects_written=centroid_objects_written,
        duplicate_base_rows_skipped=duplicate_skipped,
        rows_missing_required_geometry=missing_geom,
        rows_with_missing_z_imputed=missing_z_imputed,
        keyword_counts=keyword_counts,
        feature_type_counts=feature_type_counts,
        role_counts=role_counts,
        bounds_before_origin_shift=bounds,
        origin=origin,
    )

    if metadata_out is not None:
        _write_metadata(metadata_out, stats, input_csv=input_csv, objects_out=objects_out, config={
            "include_centroid": include_centroid,
            "opening_policy": opening_policy,
            "flat_z": flat_z,
            "flat_thickness": flat_thickness,
            "normalize_origin": normalize_origin,
            "max_rows": max_rows,
            "dedupe_base_features": dedupe_base_features,
            "max_keywords_per_object": max_keywords_per_object,
        })

    return stats


def _read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty CSV: {path}") from exc


def _iter_rows(path: Path, *, max_rows: int | None) -> Iterator[Tuple[int, Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=1):
            if max_rows is not None and row_number > max_rows:
                break
            yield row_number, row


def _infer_origin_and_bounds(
    path: Path,
    *,
    opening_policy: str,
    flat_z: float,
    flat_thickness: float,
    normalize_origin: bool,
    max_rows: int | None,
    dedupe_base_features: bool,
) -> Tuple[Point3D, Dict[str, List[float]], Dict[str, int]]:
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    seen_base: set[str] = set()
    counts = {"points_seen": 0, "rows_seen": 0}

    def add(point: Point3D) -> None:
        counts["points_seen"] += 1
        for i, value in enumerate(point):
            mins[i] = min(mins[i], value)
            maxs[i] = max(maxs[i], value)

    for row_number, row in _iter_rows(path, max_rows=max_rows):
        counts["rows_seen"] += 1
        gml_id = _safe_id(row.get("gml_id"), fallback=f"row_{row_number}")
        base_points = _base_points(row, flat_z=flat_z, flat_thickness=flat_thickness)
        if base_points is not None and ((not dedupe_base_features) or gml_id not in seen_base):
            seen_base.add(gml_id)
            bottom, top, _ = base_points
            add(bottom)
            add(top)
            center = tuple((bottom[i] + top[i]) * 0.5 for i in range(3))
            add(center)  # include optional centroid in bounds even if not emitted
        if opening_policy != "skip":
            for _object_id, _role, point, _keywords in _opening_objects(
                row,
                row_number=row_number,
                gml_id=gml_id,
                opening_policy=opening_policy,
                max_keywords_per_object=128,
            ):
                add(point)

    if counts["points_seen"] == 0:
        raise ValueError("no convertible points found in input CSV")
    origin = (mins[0], mins[1], mins[2]) if normalize_origin else (0.0, 0.0, 0.0)
    return origin, {"mins": mins, "maxs": maxs}, counts


def _base_points(row: Mapping[str, str], *, flat_z: float, flat_thickness: float) -> Optional[Tuple[Point3D, Point3D, bool]]:
    lx = _to_float(row.get("lowerCorner_x"))
    ux = _to_float(row.get("upperCorner_x"))
    ly = _to_float(row.get("lowerCorner_y"))
    uy = _to_float(row.get("upperCorner_y"))
    if None in (lx, ux, ly, uy):
        return None
    x = (lx + ux) * 0.5  # type: ignore[operator]
    y = (ly + uy) * 0.5  # type: ignore[operator]
    lz = _to_float(row.get("lowerCorner_z"))
    uz = _to_float(row.get("upperCorner_z"))
    z_was_imputed = False
    if lz is None or uz is None:
        lz = flat_z
        uz = flat_z + flat_thickness
        z_was_imputed = True
    z0 = min(lz, uz)
    z1 = max(lz, uz)
    return (x, y, z0), (x, y, z1), z_was_imputed


def _opening_objects(
    row: Mapping[str, str],
    *,
    row_number: int,
    gml_id: str,
    opening_policy: str,
    max_keywords_per_object: int,
) -> Iterator[Tuple[str, str, Point3D, List[str]]]:
    opening_type = _clean_keyword(row.get("opening_type"))
    if not opening_type:
        return
    x1 = _to_float(row.get("opening_x1"))
    x2 = _to_float(row.get("opening_x2"))
    z1 = _to_float(row.get("opening_y1"))
    z2 = _to_float(row.get("opening_y2"))
    y = _to_float(row.get("opening_z"))
    if None in (x1, x2, z1, z2, y):
        return
    x = (x1 + x2) * 0.5  # type: ignore[operator]
    low_z = min(z1, z2)  # type: ignore[arg-type]
    high_z = max(z1, z2)  # type: ignore[arg-type]
    semantic = _base_semantic_keywords(row)
    semantic = _ordered_unique(["opening", opening_type, f"opening_type:{opening_type}"] + semantic)
    slug = _safe_id(opening_type, fallback="opening")
    if opening_policy == "point":
        point = (x, y, (low_z + high_z) * 0.5)  # type: ignore[arg-type]
        role = f"{opening_type}_center"
        keywords = _role_keywords(semantic, "center", max_keywords_per_object)
        yield f"{gml_id}:opening:{row_number}:{slug}:center", role, point, keywords
    elif opening_policy == "endpoints":
        for role, z in (("bottom", low_z), ("top", high_z)):
            point = (x, y, z)  # type: ignore[arg-type]
            keywords = _role_keywords(semantic, role, max_keywords_per_object)
            yield f"{gml_id}:opening:{row_number}:{slug}:{role}", f"{opening_type}_{role}", point, keywords
    elif opening_policy == "skip":
        return
    else:
        raise ValueError(f"unknown opening policy: {opening_policy}")


def _base_semantic_keywords(row: Mapping[str, str]) -> List[str]:
    keywords: List[str] = []
    feature_type = _clean_keyword(row.get("feature_type"))
    if feature_type:
        keywords.extend([feature_type, f"feature:{feature_type}"])

    building_use = _clean_keyword(row.get("building_use"))
    if building_use:
        keywords.extend([building_use, f"building_use:{building_use}", f"use:{building_use}"])
        if building_use in {"apartments", "house", "detached", "semidetached_house", "terrace", "residential", "dormitory"}:
            keywords.append("residential")
        if building_use in {"retail", "commercial", "supermarket", "shop"}:
            keywords.append("commerce")
        if building_use in {"school", "kindergarten", "university", "college"}:
            keywords.append("education")

    material = _clean_keyword(row.get("material"))
    if material:
        keywords.extend([material, f"material:{material}"])

    roof_code = _roof_code(row.get("roofType"))
    if roof_code:
        keywords.append(f"roof_type:{roof_code}")
        if roof_code in _ROOF_TYPE_NAMES:
            keywords.append(_ROOF_TYPE_NAMES[roof_code])

    height = _to_float(row.get("measured_height"))
    if height is not None:
        keywords.append("has_height")
        if height < 8:
            keywords.extend(["height:low", "low_building"])
        elif height < 20:
            keywords.extend(["height:mid", "midrise"])
        else:
            keywords.extend(["height:tall", "tall_building"])
        if height >= 35:
            keywords.append("highrise")

    storeys = _to_float(row.get("storeysAboveGround"))
    if storeys is not None:
        keywords.append("has_storeys")
        if storeys >= 5:
            keywords.append("multi_storey")

    poi_count = _to_float(row.get("poi_count"))
    if poi_count is not None and poi_count > 0:
        keywords.extend(["poi", "has_poi"])
        if poi_count >= 5:
            keywords.append("many_poi")

    if _clean_text(row.get("name")):
        keywords.append("named")

    keywords.extend(_keywords_from_tag_string(row.get("osm_building_tags")))
    keywords.extend(_keywords_from_tag_string(row.get("amenities")))
    return _ordered_unique(keywords)


def _keywords_from_tag_string(value: Optional[str]) -> List[str]:
    text = _clean_text(value)
    if not text:
        return []
    keywords: List[str] = []
    for raw_part in text.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        # Input often looks like "amenity=cafe(Name)".  The label inside
        # parentheses is a human-readable name; the SPM keyword should be the
        # stable OSM-like key/value pair.
        part_without_name = re.sub(r"\([^)]*\)", "", part).strip()
        if "=" in part_without_name:
            key, value_part = part_without_name.split("=", 1)
            key_kw = _clean_keyword(key)
            val_kw = _clean_keyword(value_part)
            if key_kw:
                keywords.append(key_kw)
            if val_kw:
                keywords.append(val_kw)
            if key_kw and val_kw:
                keywords.append(f"{key_kw}:{val_kw}")
        else:
            kw = _clean_keyword(part_without_name)
            if kw:
                keywords.append(kw)
    return _ordered_unique(keywords)


def _role_keywords(semantic_keywords: Sequence[str], role: str, max_keywords_per_object: int) -> List[str]:
    role = _clean_keyword(role) or "point"
    out: List[str] = [role]
    # Direct semantic keywords make this behave like a normal spatio-textual
    # dataset.  Role-specific keywords let patterns select bottom/top/opening
    # endpoints when that distinction matters.
    out.extend(semantic_keywords)
    for kw in semantic_keywords:
        role_kw = _role_suffix(kw, role)
        if role_kw:
            out.append(role_kw)
    if max_keywords_per_object:
        return _ordered_unique(out)[:max_keywords_per_object]
    return _ordered_unique(out)


def _role_suffix(keyword: str, role: str) -> str:
    base = re.sub(r"[^a-z0-9_]+", "_", keyword.lower()).strip("_")
    if not base:
        return ""
    return f"{base}_{role}"


def _write_object(out: Any, object_id: str, point: Point3D, keywords: Sequence[str], *, source: Mapping[str, Any]) -> None:
    row = {
        "id": object_id,
        "x": point[0],
        "y": point[1],
        "z": point[2],
        "keywords": list(_ordered_unique(keywords)),
        "source": dict(source),
    }
    out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_metadata(path: str | Path, stats: ConvertStats, *, input_csv: Path, objects_out: Path, config: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_csv": str(input_csv),
        "objects_out": str(objects_out),
        "rows_read": stats.rows_read,
        "base_features_seen": stats.base_features_seen,
        "base_features_written": stats.base_features_written,
        "objects_written": stats.objects_written,
        "opening_objects_written": stats.opening_objects_written,
        "centroid_objects_written": stats.centroid_objects_written,
        "duplicate_base_rows_skipped": stats.duplicate_base_rows_skipped,
        "rows_missing_required_geometry": stats.rows_missing_required_geometry,
        "rows_with_missing_z_imputed": stats.rows_with_missing_z_imputed,
        "origin_subtracted": list(stats.origin),
        "bounds_before_origin_shift": stats.bounds_before_origin_shift,
        "bounds_after_origin_shift": {
            "mins": [stats.bounds_before_origin_shift["mins"][i] - stats.origin[i] for i in range(3)],
            "maxs": [stats.bounds_before_origin_shift["maxs"][i] - stats.origin[i] for i in range(3)],
        },
        "feature_type_counts": dict(stats.feature_type_counts.most_common()),
        "role_counts": dict(stats.role_counts.most_common()),
        "top_keywords": dict(stats.keyword_counts.most_common(200)),
        "config": dict(config),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _roof_code(value: Optional[str]) -> str:
    number = _to_float(value)
    if number is None:
        return ""
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return _clean_keyword(value)


def _clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _clean_keyword(value: Optional[str]) -> str:
    text = _clean_text(value).lower()
    if not text:
        return ""
    text = text.replace("/", "_")
    text = re.sub(r"[^a-z0-9:_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_:-")
    return text


def _safe_id(value: Optional[str], *, fallback: str) -> str:
    text = _clean_text(value)
    if not text:
        return fallback
    # Keep IDs readable but avoid whitespace and path-like delimiters.
    text = re.sub(r"\s+", "_", text.strip())
    text = text.replace("/", "_")
    return text


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _shift(point: Point3D, origin: Point3D) -> Point3D:
    return (point[0] - origin[0], point[1] - origin[1], point[2] - origin[2])



def run_converted_patterns(
    objects_jsonl: str | Path,
    patterns_json: str | Path,
    *,
    results_out: str | Path,
    summary_out: str | Path | None = None,
    match_limit: int | None = 1000,
    capacity: int = 64,
    min_level: int = 1,
    max_level: int = 10,
    require_distinct_objects: bool = True,
) -> Dict[str, Any]:
    """Run a pattern file against an already converted object JSONL file.

    This is a lightweight real-data baseline runner.  It is intentionally smaller
    than the synthetic scalability benchmark, but it records the core per-pattern
    runtime and ESPM statistics in JSONL.
    """

    objects_jsonl = Path(objects_jsonl)
    patterns_json = Path(patterns_json)
    results_out = Path(results_out)
    results_out.parent.mkdir(parents=True, exist_ok=True)
    if summary_out is not None:
        summary_out = Path(summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)

    process = _current_process_or_none()
    start_rss = _rss_mb(process)

    t0 = time.perf_counter()
    objects = _load_objects_jsonl_with_source_ignored(objects_jsonl)
    load_time_s = time.perf_counter() - t0
    after_load_rss = _rss_mb(process)

    t1 = time.perf_counter()
    index = InvertedOctreeIndex(objects, capacity=capacity, min_level=min_level, max_level=max_level)
    index_build_time_s = time.perf_counter() - t1
    after_index_rss = _rss_mb(process)

    patterns = load_patterns_json(patterns_json)
    matcher = ESPM3DMatcher(index, require_distinct_objects=require_distinct_objects)

    rows: List[Dict[str, Any]] = []
    match_time_total_s = 0.0
    peak_rss_mb = max(v for v in [start_rss, after_load_rss, after_index_rss] if v is not None) if any(v is not None for v in [start_rss, after_load_rss, after_index_rss]) else None

    with results_out.open("w", encoding="utf-8") as f:
        for pattern_index, named in enumerate(patterns, start=1):
            before_match_rss = _rss_mb(process)
            t = time.perf_counter()
            status = "ok"
            error = None
            try:
                matches = matcher.match(named.pattern, limit=match_limit)
            except Exception as exc:  # pragma: no cover - defensive baseline runner
                matches = []
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
            match_time_s = time.perf_counter() - t
            match_time_total_s += match_time_s
            after_match_rss = _rss_mb(process)
            if after_match_rss is not None:
                peak_rss_mb = max(peak_rss_mb or after_match_rss, after_match_rss)
            stats = matcher.last_stats
            keyword_postings = {
                str(vertex): len(index.objects_for_keyword(keyword))
                for vertex, keyword in named.pattern.vertices.items()
            }
            row = {
                "pattern_index": pattern_index,
                "pattern_name": named.name,
                "status": status,
                "error": error,
                "pattern_vertices": len(named.pattern.vertices),
                "pattern_edges": len(named.pattern.edges),
                "match_time_s": match_time_s,
                "matches_returned": len(matches),
                "limit_reached": bool(match_limit is not None and len(matches) >= match_limit),
                "match_limit": match_limit,
                "rss_before_match_mb": before_match_rss,
                "rss_after_match_mb": after_match_rss,
                "ematch_total": sum(stats.ematch_counts.values()) if stats else None,
                "nmatch_total_final_level": sum(stats.nmatch_counts_by_level[-1].values()) if stats and stats.nmatch_counts_by_level else None,
                "skip_edge_count": len(stats.skip_edges) if stats else None,
                "skip_edges": sorted(stats.skip_edges) if stats else [],
                "keyword_postings_by_vertex": keyword_postings,
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
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()

    summary = {
        "objects_jsonl": str(objects_jsonl),
        "patterns_json": str(patterns_json),
        "results_out": str(results_out),
        "n_objects": len(objects),
        "n_patterns": len(patterns),
        "capacity": capacity,
        "min_level": min_level,
        "max_level": max_level,
        "require_distinct_objects": require_distinct_objects,
        "match_limit": match_limit,
        "load_time_s": load_time_s,
        "index_build_time_s": index_build_time_s,
        "match_time_total_s": match_time_total_s,
        "scenario_total_time_s": load_time_s + index_build_time_s + match_time_total_s,
        "rss_start_mb": start_rss,
        "rss_after_load_mb": after_load_rss,
        "rss_after_index_mb": after_index_rss,
        "rss_peak_observed_mb": peak_rss_mb,
        "patterns_ok": sum(1 for row in rows if row["status"] == "ok"),
        "patterns_error": sum(1 for row in rows if row["status"] != "ok"),
        "matches_returned_total": sum(row["matches_returned"] for row in rows),
    }
    if summary_out is not None:
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _current_process_or_none() -> Any:
    try:
        import psutil  # type: ignore

        return psutil.Process()
    except Exception:
        return None


def _rss_mb(process: Any) -> Optional[float]:
    if process is None:
        return None
    try:
        return float(process.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def run_smoke_test(
    input_csv: str | Path,
    *,
    max_rows: int = 5000,
    match_limit: int = 10,
    capacity: int = 64,
    min_level: int = 1,
    max_level: int = 10,
) -> List[Dict[str, Any]]:
    """Convert a small slice and run the five Hamburg patterns.

    This is meant as a sanity check; it writes temporary files and returns one
    summary dict per pattern.
    """

    with tempfile.TemporaryDirectory(prefix="hamburg_espm3d_") as tmp:
        tmp_path = Path(tmp)
        objects_path = tmp_path / "objects.jsonl"
        patterns_path = tmp_path / "patterns.json"
        convert_hamburg_csv(
            input_csv,
            objects_path,
            patterns_out=patterns_path,
            max_rows=max_rows,
            opening_policy="endpoints",
        )
        objects = _load_objects_jsonl_with_source_ignored(objects_path)
        patterns = load_patterns_json(patterns_path)
        index = InvertedOctreeIndex(objects, capacity=capacity, min_level=min_level, max_level=max_level)
        matcher = ESPM3DMatcher(index, require_distinct_objects=True)
        rows: List[Dict[str, Any]] = []
        for named in patterns:
            matches = matcher.match(named.pattern, limit=match_limit)
            rows.append(
                {
                    "pattern_name": named.name,
                    "matches_returned": len(matches),
                    "limit_reached": matcher.last_stats.limit_reached if matcher.last_stats else False,
                    "ematch_total": matcher.last_stats.ematch_total if matcher.last_stats else None,
                }
            )
        return rows


def _load_objects_jsonl_with_source_ignored(path: str | Path) -> List[SpatialObject]:
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Hamburg real-data CSV to ESPM-3D JSONL.")
    sub = parser.add_subparsers(dest="command")

    convert = sub.add_parser("convert", help="Convert the CSV to object JSONL and write Hamburg patterns.")
    convert.add_argument("input_csv", help="Path to hamburg_buildings_facade_amenities_trees.csv")
    convert.add_argument("--objects-out", default="hamburg_objects.jsonl", help="Output ESPM-3D object JSONL.")
    convert.add_argument("--patterns-out", default="hamburg_patterns_5.json", help="Output five suggested Hamburg patterns.")
    convert.add_argument("--metadata-out", default="hamburg_conversion_metadata.json", help="Output conversion metadata JSON.")
    convert.add_argument("--include-centroid", action="store_true", help="Also emit one centroid object per base feature.")
    convert.add_argument("--opening-policy", choices=["endpoints", "point", "skip"], default="endpoints", help="How to convert facade openings.")
    convert.add_argument("--flat-z", type=float, default=0.0, help="Bottom z for features with missing z, such as parks/water.")
    convert.add_argument("--flat-thickness", type=float, default=1.0, help="Top-bottom thickness for features with missing z.")
    convert.add_argument("--no-normalize-origin", action="store_true", help="Keep original large projected coordinates instead of subtracting min x/y/z.")
    convert.add_argument("--max-rows", type=int, default=None, help="Optional debug limit on input rows.")
    convert.add_argument("--no-dedupe-base-features", action="store_true", help="Do not deduplicate base bottom/top entities by gml_id.")
    convert.add_argument("--max-keywords-per-object", type=int, default=128, help="Cap keyword count per emitted object.")

    list_patterns = sub.add_parser("list-patterns", help="Print the five suggested Hamburg patterns as JSON.")
    list_patterns.add_argument("--output", default=None, help="Optional path to write the pattern JSON.")

    smoke = sub.add_parser("smoke-test", help="Convert a small slice and run the five Hamburg patterns.")
    smoke.add_argument("input_csv", help="Path to hamburg_buildings_facade_amenities_trees.csv")
    smoke.add_argument("--max-rows", type=int, default=5000, help="Rows to convert for the smoke test.")
    smoke.add_argument("--match-limit", type=int, default=10, help="Per-pattern match limit.")
    smoke.add_argument("--capacity", type=int, default=64)
    smoke.add_argument("--min-level", type=int, default=1)
    smoke.add_argument("--max-level", type=int, default=10)

    run = sub.add_parser("run-patterns", help="Run pattern JSON against converted object JSONL.")
    run.add_argument("--objects", required=True, help="Converted ESPM-3D object JSONL.")
    run.add_argument("--patterns", required=True, help="Pattern JSON, e.g. hamburg_patterns_5.json.")
    run.add_argument("--results-out", default="hamburg_pattern_results.jsonl", help="Per-pattern result JSONL.")
    run.add_argument("--summary-out", default="hamburg_pattern_summary.json", help="Summary JSON.")
    run.add_argument("--match-limit", type=int, default=1000, help="Per-pattern match limit. Use 0 for no limit.")
    run.add_argument("--capacity", type=int, default=64)
    run.add_argument("--min-level", type=int, default=1)
    run.add_argument("--max-level", type=int, default=10)
    run.add_argument("--allow-same-object", action="store_true", help="Allow one object to satisfy multiple pattern vertices.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return

    if args.command == "convert":
        stats = convert_hamburg_csv(
            args.input_csv,
            args.objects_out,
            metadata_out=args.metadata_out,
            patterns_out=args.patterns_out,
            include_centroid=args.include_centroid,
            opening_policy=args.opening_policy,
            flat_z=args.flat_z,
            flat_thickness=args.flat_thickness,
            normalize_origin=not args.no_normalize_origin,
            max_rows=args.max_rows,
            dedupe_base_features=not args.no_dedupe_base_features,
            max_keywords_per_object=args.max_keywords_per_object,
        )
        print(
            json.dumps(
                {
                    "rows_read": stats.rows_read,
                    "objects_written": stats.objects_written,
                    "base_features_written": stats.base_features_written,
                    "opening_objects_written": stats.opening_objects_written,
                    "objects_out": args.objects_out,
                    "patterns_out": args.patterns_out,
                    "metadata_out": args.metadata_out,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "list-patterns":
        patterns = hamburg_pattern_suite()
        if args.output:
            save_patterns_json(patterns, args.output)
            print(f"wrote {len(patterns)} patterns to {args.output}")
        else:
            rows = [
                {
                    "name": p.name,
                    "description": p.description,
                    "vertices": dict(p.pattern.vertices),
                    "edges": [
                        {
                            "source": e.source,
                            "target": e.target,
                            "lower": e.lower,
                            "upper": e.upper,
                            "sign": e.sign.value,
                        }
                        for e in p.pattern.edges
                    ],
                }
                for p in patterns
            ]
            print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.command == "smoke-test":
        rows = run_smoke_test(
            args.input_csv,
            max_rows=args.max_rows,
            match_limit=args.match_limit,
            capacity=args.capacity,
            min_level=args.min_level,
            max_level=args.max_level,
        )
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.command == "run-patterns":
        limit = None if args.match_limit == 0 else args.match_limit
        summary = run_converted_patterns(
            args.objects,
            args.patterns,
            results_out=args.results_out,
            summary_out=args.summary_out,
            match_limit=limit,
            capacity=args.capacity,
            min_level=args.min_level,
            max_level=args.max_level,
            require_distinct_objects=not args.allow_same_object,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:  # pragma: no cover - argparse prevents this
        raise AssertionError(args.command)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
