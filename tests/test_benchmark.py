import json
from pathlib import Path

from espm3d.benchmark import (
    BenchmarkConfig,
    PROFILE_SPARSITY_SWEEPS,
    SPARSITY_PROFILES,
    build_arg_parser,
    build_benchmark_patterns,
    config_from_args,
    run_benchmark,
)


def test_pattern_variants_expand_default_suite():
    patterns = build_benchmark_patterns(pattern_count=2, interval_scales=[1.0, 1.5])
    assert len(patterns) == 4
    assert patterns[2][0].name.endswith("interval_x1p5")


def test_cli_profile_expands_to_default_sparsity_sweep():
    parser = build_arg_parser()
    args = parser.parse_args(["--profile", "smoke"])
    config = config_from_args(args)
    assert config.sparsity_profiles == PROFILE_SPARSITY_SWEEPS["smoke"]
    assert "sparse_graph" in config.sparsity_profiles
    assert "dense_graph" in config.sparsity_profiles


def test_manual_sparsity_profile_preserves_explicit_generator_controls():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--profile",
            "smoke",
            "--sparsity-profiles",
            "manual",
            "--noise-keywords",
            "17",
            "--clustered-fraction",
            "0.25",
        ]
    )
    config = config_from_args(args)
    assert config.sparsity_profiles == ("manual",)
    assert config.n_noise_keywords == 17
    assert config.clustered_fraction == 0.25
    assert "very_sparse_graph" in SPARSITY_PROFILES


def test_smoke_benchmark_writes_readable_results(tmp_path: Path):
    config = BenchmarkConfig(
        scales=(300,),
        sparsity_profiles=("manual",),
        pattern_count=2,
        seed=123,
        output_dir=str(tmp_path),
        match_limit=10,
        n_noise_keywords=16,
        max_level=6,
    )
    paths = run_benchmark(config)
    assert paths["results_csv"].exists()
    assert paths["results_jsonl"].exists()
    assert paths["summary_csv"].exists()
    assert paths["summary_json"].exists()
    assert paths["summary_by_pattern_csv"].exists()
    assert paths["summary_by_pattern_json"].exists()

    csv_text = paths["results_csv"].read_text(encoding="utf-8")
    assert "pattern_name" in csv_text
    assert "sparsity_profile" in csv_text
    assert "candidate_graph_density_measured" in csv_text

    raw_lines = [json.loads(line) for line in paths["results_jsonl"].read_text(encoding="utf-8").splitlines() if line]
    assert raw_lines
    assert raw_lines[0]["sparsity_profile"] == "manual"
    assert "candidate_pair_space_total" in raw_lines[0]
    assert "candidate_graph_edges_by_edge" in raw_lines[0]

    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert len(summary) == 1
    assert summary[0]["sparsity_profile"] == "manual"
    assert "match_time_mean_s" in summary[0]
    assert "candidate_graph_density_mean" in summary[0]

    summary_by_pattern = json.loads(paths["summary_by_pattern_json"].read_text(encoding="utf-8"))
    assert len(summary_by_pattern) == 2
    assert {row["pattern_name"] for row in summary_by_pattern}
    assert all(row["scenario_id"] == "n300__manual" for row in summary_by_pattern)
    assert all("match_time_mean_s" in row for row in summary_by_pattern)
    assert all("candidate_graph_density_mean" in row for row in summary_by_pattern)

    summary_by_pattern_csv = paths["summary_by_pattern_csv"].read_text(encoding="utf-8")
    assert "pattern_runs_attempted" in summary_by_pattern_csv
    assert "rss_peak_match_max_mb" in summary_by_pattern_csv


def test_scenario_seed_is_stable_when_sweep_changes():
    from espm3d.benchmark import _scenario_seed

    seed_sparse_only = _scenario_seed(42, 10000, "sparse_graph")
    seed_with_dense_also_present = _scenario_seed(42, 10000, "sparse_graph")
    assert seed_sparse_only == seed_with_dense_also_present
    assert seed_sparse_only != _scenario_seed(42, 10000, "dense_graph")


def test_isolated_benchmark_writes_aggregate_results(tmp_path: Path):
    config = BenchmarkConfig(
        scales=(120,),
        sparsity_profiles=("manual",),
        pattern_count=1,
        seed=321,
        output_dir=str(tmp_path),
        match_limit=5,
        n_noise_keywords=8,
        max_level=4,
        isolate_scenarios=True,
        cleanup_between_patterns=True,
    )
    paths = run_benchmark(config)
    assert paths["results_jsonl"].exists()
    rows = [json.loads(line) for line in paths["results_jsonl"].read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["run_id"] in paths["results_jsonl"].name
    assert rows[0]["status"] == "ok"
    assert paths["summary_by_pattern_json"].exists()
