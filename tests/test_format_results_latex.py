from __future__ import annotations

import json
from pathlib import Path

from espm3d.format_results_latex import (
    build_table_from_results,
    compact_size_label,
    parse_size_token,
)


def _write_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([row], indent=2), encoding="utf-8")


def test_parse_size_token_and_label() -> None:
    assert parse_size_token("10k") == 10_000
    assert parse_size_token("100K") == 100_000
    assert parse_size_token("1m") == 1_000_000
    assert compact_size_label(10_000) == "10K"
    assert compact_size_label(1_000_000) == "1M"
    assert compact_size_label(123_456) == "123456"


def test_builds_scenario_latex_table_with_default_scenario_time_and_gb(tmp_path: Path) -> None:
    _write_summary(
        tmp_path / "10k" / "sparse_graph" / "run_a_summary.json",
        {
            "run_id": "run_a",
            "scenario_id": "n10000__sparse_graph",
            "n_objects": 10_000,
            "sparsity_profile": "sparse_graph",
            "patterns_attempted": 20,
            "patterns_ok": 20,
            "patterns_error": 0,
            "scenario_total_time_s": 1234.5678,
            "match_time_total_s": 12.3456,
            "rss_peak_scenario_mb": 4096.0,
        },
    )
    _write_summary(
        tmp_path / "10k" / "dense_graph" / "run_b_summary.json",
        {
            "run_id": "run_b",
            "scenario_id": "n10000__dense_graph",
            "n_objects": 10_000,
            "sparsity_profile": "dense_graph",
            "patterns_attempted": 20,
            "patterns_ok": 20,
            "patterns_error": 0,
            "scenario_total_time_s": 9876.5432,
            "match_time_total_s": 98.7654,
            "rss_peak_scenario_mb": 8192.0,
        },
    )

    latex = build_table_from_results(
        tmp_path,
        sizes=[10_000],
        sparsity_profiles=["sparse_graph", "dense_graph"],
        time_precision=2,
        ram_precision=1,
    )

    assert r"\multicolumn{2}{c}{\textbf{Sparse}}" in latex
    assert r"\multicolumn{2}{c}{\textbf{Dense}}" in latex
    assert r"\textbf{Time (s)}" in latex
    assert r"\textbf{Peak RAM (GB)}" in latex
    assert "GiB" not in latex
    assert "10K & 1234.57 & 4.0 & 9876.54 & 8.0" in latex
    assert "1,234" not in latex
    assert "9,876" not in latex
    assert r"\bottomrule" in latex


def test_can_request_match_time_metric_explicitly(tmp_path: Path) -> None:
    _write_summary(
        tmp_path / "10k" / "sparse_graph" / "run_a_summary.json",
        {
            "run_id": "run_a",
            "scenario_id": "n10000__sparse_graph",
            "n_objects": 10_000,
            "sparsity_profile": "sparse_graph",
            "patterns_attempted": 20,
            "patterns_ok": 20,
            "patterns_error": 0,
            "scenario_total_time_s": 100.0,
            "match_time_total_s": 12.3456,
            "rss_peak_scenario_mb": 1024.0,
        },
    )

    latex = build_table_from_results(
        tmp_path,
        sizes=[10_000],
        sparsity_profiles=["sparse_graph"],
        time_metric="match_time_total_s",
        time_precision=2,
        ram_precision=1,
    )

    assert "10K & 12.35 & 1.0" in latex


def test_trace_fallback_marks_incomplete_ram(tmp_path: Path) -> None:
    trace = tmp_path / "10m" / "dense_graph" / "run_memory_trace.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "n_objects": 10_000_000,
                        "sparsity_profile": "dense_graph",
                        "elapsed_s": 10.0,
                        "rss_mb": 100_000.0,
                        "peak_rss_mb": 100_000.0,
                    }
                ),
                json.dumps(
                    {
                        "n_objects": 10_000_000,
                        "sparsity_profile": "dense_graph",
                        "elapsed_s": 20.0,
                        "rss_mb": 200_000.0,
                        "peak_rss_mb": 200_000.0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    latex = build_table_from_results(
        tmp_path,
        sizes=[10_000_000],
        sparsity_profiles=["dense_graph"],
        ram_precision=1,
    )

    assert r"\textsc{fail}" in latex
    assert r"195.3$^{\dagger}$" in latex
    assert "memory trace" in latex


def test_pattern_source_requires_filter_and_builds_table(tmp_path: Path) -> None:
    _write_summary(
        tmp_path / "100k" / "sparse_graph" / "run_summary_by_pattern.json",
        {
            "run_id": "run",
            "scenario_id": "n100000__sparse_graph",
            "n_objects": 100_000,
            "sparsity_profile": "sparse_graph",
            "pattern_name": "p01_residential_area",
            "interval_scale": 1.0,
            "pattern_runs_attempted": 1,
            "pattern_runs_ok": 1,
            "pattern_runs_error": 0,
            "scenario_total_time_s": 3.5,
            "match_time_total_s": 1.25,
            "rss_peak_scenario_mb": 1024.0,
        },
    )

    latex = build_table_from_results(
        tmp_path,
        source="pattern",
        pattern_name="p01_residential_area",
        sizes=[100_000],
        sparsity_profiles=["sparse_graph"],
    )

    assert "100K & 3.50 & 1.0" in latex
