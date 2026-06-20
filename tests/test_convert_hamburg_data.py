import csv
import json
from pathlib import Path

from espm3d.convert_hamburg_data import convert_hamburg_csv, hamburg_pattern_suite, run_converted_patterns
from espm3d.generate_synthetic_data import load_objects_jsonl, load_patterns_json


HEADER = [
    "gml_id",
    "material",
    "roofType",
    "measured_height",
    "storeysAboveGround",
    "upperCorner_x",
    "upperCorner_y",
    "upperCorner_z",
    "lowerCorner_x",
    "lowerCorner_y",
    "lowerCorner_z",
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
    "feature_type",
    "building_use",
    "name",
]


def _write_fixture(path: Path) -> None:
    rows = [
        {
            "gml_id": "b1",
            "material": "concrete",
            "roofType": "1000",
            "measured_height": "20",
            "storeysAboveGround": "5",
            "upperCorner_x": "10",
            "upperCorner_y": "20",
            "upperCorner_z": "30",
            "lowerCorner_x": "0",
            "lowerCorner_y": "0",
            "lowerCorner_z": "0",
            "opening_type": "Door",
            "opening_x1": "2",
            "opening_x2": "4",
            "opening_y1": "0.5",
            "opening_y2": "3.0",
            "opening_z": "1",
            "osm_building_id": "w1",
            "osm_building_tags": "building=apartments",
            "amenities": "amenity=cafe(Test Cafe); shop=books(Book Shop)",
            "poi_count": "2",
            "feature_type": "building",
            "building_use": "apartments",
            "name": "Example Building",
        },
        # Duplicate base building row; it should not duplicate b1:bottom/top,
        # but it should create a separate window opening.
        {
            "gml_id": "b1",
            "material": "concrete",
            "roofType": "1000",
            "measured_height": "20",
            "storeysAboveGround": "5",
            "upperCorner_x": "10",
            "upperCorner_y": "20",
            "upperCorner_z": "30",
            "lowerCorner_x": "0",
            "lowerCorner_y": "0",
            "lowerCorner_z": "0",
            "opening_type": "Window",
            "opening_x1": "2",
            "opening_x2": "4",
            "opening_y1": "8",
            "opening_y2": "10",
            "opening_z": "1",
            "osm_building_id": "w1",
            "osm_building_tags": "building=apartments",
            "amenities": "amenity=cafe(Test Cafe)",
            "poi_count": "1",
            "feature_type": "building",
            "building_use": "apartments",
            "name": "Example Building",
        },
        {
            "gml_id": "park1",
            "material": "",
            "roofType": "",
            "measured_height": "",
            "storeysAboveGround": "",
            "upperCorner_x": "110",
            "upperCorner_y": "120",
            "upperCorner_z": "",
            "lowerCorner_x": "100",
            "lowerCorner_y": "100",
            "lowerCorner_z": "",
            "opening_type": "",
            "opening_x1": "",
            "opening_x2": "",
            "opening_y1": "",
            "opening_y2": "",
            "opening_z": "",
            "osm_building_id": "w2",
            "osm_building_tags": "leisure=park",
            "amenities": "amenity=bench",
            "poi_count": "1",
            "feature_type": "park",
            "building_use": "",
            "name": "Pocket Park",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_convert_hamburg_csv_emits_endpoint_objects_and_patterns(tmp_path: Path) -> None:
    csv_path = tmp_path / "hamburg.csv"
    objects_path = tmp_path / "objects.jsonl"
    patterns_path = tmp_path / "patterns.json"
    metadata_path = tmp_path / "metadata.json"
    _write_fixture(csv_path)

    stats = convert_hamburg_csv(
        csv_path,
        objects_path,
        patterns_out=patterns_path,
        metadata_out=metadata_path,
        flat_z=0.0,
        flat_thickness=1.0,
    )

    assert stats.rows_read == 3
    assert stats.base_features_written == 2
    assert stats.opening_objects_written == 4
    assert stats.duplicate_base_rows_skipped == 1

    objects = load_objects_jsonl(objects_path)
    by_id = {str(obj.id): obj for obj in objects}
    assert set(by_id) >= {
        "b1:bottom",
        "b1:top",
        "b1:opening:1:door:bottom",
        "b1:opening:2:window:top",
        "park1:bottom",
        "park1:top",
    }

    assert "building_bottom" in by_id["b1:bottom"].keywords
    assert "building_top" in by_id["b1:top"].keywords
    assert "apartments_bottom" in by_id["b1:bottom"].keywords
    assert "door_bottom" in by_id["b1:opening:1:door:bottom"].keywords
    assert "window_top" in by_id["b1:opening:2:window:top"].keywords
    assert "park_bottom" in by_id["park1:bottom"].keywords

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["objects_written"] == len(objects)
    assert metadata["config"]["opening_policy"] == "endpoints"

    patterns = load_patterns_json(patterns_path)
    assert [p.name for p in patterns] == [p.name for p in hamburg_pattern_suite()]


def test_hamburg_pattern_suite_has_expected_keywords() -> None:
    patterns = hamburg_pattern_suite()
    assert len(patterns) == 5
    keywords = {kw for p in patterns for kw in p.pattern.vertices.values()}
    assert "highrise_bottom" in keywords
    assert "highrise_top" in keywords
    assert "building_top" in keywords
    assert "door_bottom" in keywords
    assert "window_top" in keywords
    assert "hospital_bottom" in keywords
    assert "playground_bottom" in keywords
    assert "train_station_bottom" in keywords


def test_run_converted_patterns_writes_jsonl_and_summary(tmp_path: Path) -> None:
    csv_path = tmp_path / "hamburg.csv"
    objects_path = tmp_path / "objects.jsonl"
    patterns_path = tmp_path / "patterns.json"
    results_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_fixture(csv_path)
    convert_hamburg_csv(csv_path, objects_path, patterns_out=patterns_path)

    summary = run_converted_patterns(
        objects_path,
        patterns_path,
        results_out=results_path,
        summary_out=summary_path,
        match_limit=5,
        max_level=4,
    )

    assert summary["n_objects"] > 0
    assert summary["n_patterns"] == 5
    assert results_path.exists()
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 5
    assert all("match_time_s" in row for row in rows)
    assert summary_path.exists()
