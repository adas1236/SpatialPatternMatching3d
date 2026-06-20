# ESPM-3D

## Overview

ESPM-3D is a Python implementation of generalized 3-D spatial pattern
matching for spatio-textual objects. It provides:

- an in-memory inverted octree index, with one octree per keyword;
- a 3-D ESPM-style matcher using minimum bounding rectangular prisms;
- synthetic 3-D data and pattern generation;
- benchmark runners for synthetic graph-sparsity experiments;
- conversion and baseline matching tools for the Hamburg buildings/facades/
  amenities/trees dataset; and
- utilities for formatting benchmark summaries as LaTeX tables.

This project builds on the ideas from the ESPM work. Thanks to the ESPM
authors for the original spatial pattern matching formulation. Their paper is:
[Efficient Spatial Pattern Matching](https://ieeexplore.ieee.org/document/8869793/).

## File Structure

```text
SpatialPatternMatching3d/
├── main.py                         # Source-checkout wrapper for synthetic benchmarks
├── generate_synthetic_data.py      # Source-checkout wrapper for synthetic data tools
├── convert_hamburg_data.py         # Source-checkout wrapper for Hamburg conversion/run tools
├── format_results_latex.py         # Source-checkout wrapper for LaTeX table formatting
├── pyproject.toml                  # Package metadata, dependencies, console scripts
├── uv.lock                         # uv lockfile
├── README.md
├── examples/
│   ├── default_patterns_20.json    # JSON copy of the built-in synthetic pattern suite
│   └── hamburg_patterns_5.json     # Suggested real-data patterns for Hamburg data
├── data/                           # Generated/converted data outputs, ignored by git
├── results/                        # Benchmark and table outputs, ignored by git
├── logs/                           # Run logs, ignored by git
├── src/
│   └── espm3d/
│       ├── __init__.py
│       ├── matcher.py              # Core objects, patterns, octree index, matcher
│       ├── generate_synthetic_data.py
│       ├── benchmark.py            # Synthetic scalability and sparsity benchmark runner
│       ├── convert_hamburg_data.py # Hamburg conversion, patterns, and baseline runner
│       └── format_results_latex.py
└── tests/
    ├── test_matcher.py
    ├── test_generate_synthetic_data.py
    ├── test_benchmark.py
    ├── test_convert_hamburg_data.py
    └── test_format_results_latex.py
```

The root-level Python files add `src/` to `sys.path`, so they work from a
source checkout without installing the package first. After installation, the
same tools are also available as console scripts:

```bash
espm3d-benchmark
generate-synthetic-data
convert-hamburg-data
format-results-latex
```

## Project Setup

Using `uv`:

```bash
uv sync --extra dev
uv run pytest -q
```

Using standard `venv` and `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The only runtime dependency is `psutil`; tests use `pytest`.

## Synthetic Experiments

### Quick Smoke Run

Run a small benchmark locally:

```bash
uv run main.py --profile smoke --output-dir results/smoke
```

Equivalent installed-package form:

```bash
uv run espm3d-benchmark --profile smoke --output-dir results/smoke
```

This writes timestamped files under `results/smoke/`, including:

- `<run_id>_settings.json`
- `<run_id>_results.jsonl`
- `<run_id>_results.csv`
- `<run_id>_summary.json`
- `<run_id>_summary.csv`
- `<run_id>_summary_by_pattern.json`
- `<run_id>_summary_by_pattern.csv`

### Standard Local Run

The standard profile runs 10K and 100K objects over sparse, medium, and dense
graph profiles:

```bash
uv run main.py \
  --profile standard \
  --isolate-scenarios \
  --malloc-trim-between-patterns \
  --output-dir results/standard
```

`--isolate-scenarios` runs each object-count/sparsity scenario in a fresh child
process. This is slower, but safer for memory-heavy runs. If a worker is killed,
the parent records a scenario failure and continues.

### Full Synthetic Sweep

The full synthetic experiment sweeps these object counts:

```bash
10k 100k 1m 10m
```

For each object count, run all five sparsity profiles:

```bash
very_sparse_graph
sparse_graph
medium_graph
dense_graph
very_dense_graph
```

Each run writes to:

```text
results/<scale>/<sparsity_profile>/
```

For example, to run the 10K scale:

```bash
uv run main.py \
  --scales 10k \
  --sparsity-profiles very_sparse_graph \
  --pattern-count 20 \
  --isolate-scenarios \
  --malloc-trim-between-patterns \
  --output-dir results/10k/very_sparse_graph

uv run main.py \
  --scales 10k \
  --sparsity-profiles sparse_graph \
  --pattern-count 20 \
  --isolate-scenarios \
  --malloc-trim-between-patterns \
  --output-dir results/10k/sparse_graph

uv run main.py \
  --scales 10k \
  --sparsity-profiles medium_graph \
  --pattern-count 20 \
  --isolate-scenarios \
  --malloc-trim-between-patterns \
  --output-dir results/10k/medium_graph

uv run main.py \
  --scales 10k \
  --sparsity-profiles dense_graph \
  --pattern-count 20 \
  --isolate-scenarios \
  --malloc-trim-between-patterns \
  --output-dir results/10k/dense_graph

uv run main.py \
  --scales 10k \
  --sparsity-profiles very_dense_graph \
  --pattern-count 20 \
  --isolate-scenarios \
  --malloc-trim-between-patterns \
  --output-dir results/10k/very_dense_graph
```

Repeat the same command set with `--scales 100k`, `--scales 1m`, and
`--scales 10m` for the larger object counts. Change `--scales`,
`--sparsity-profiles`, or `--pattern-count` to run a smaller subset. For very
large jobs, keep `--isolate-scenarios` enabled.

### Useful Synthetic Options

List available sparsity profiles:

```bash
uv run main.py --list-sparsity-profiles
```

Run one scale and one sparsity profile:

```bash
uv run main.py \
  --scales 100k \
  --sparsity-profiles sparse_graph \
  --pattern-count 20 \
  --output-dir results/100k/sparse_graph
```

Disable the per-pattern match cap:

```bash
uv run main.py --profile smoke --match-limit 0
```

Generate synthetic data files directly:

```bash
uv run python generate_synthetic_data.py generate \
  --n-objects 50000 \
  --seed 42 \
  --pattern-count 20 \
  --matches-per-pattern 1 \
  --objects-out synthetic_objects.jsonl \
  --patterns-out synthetic_patterns.json \
  --planted-out synthetic_planted_matches.json \
  --metadata-out synthetic_metadata.json
```

## Hamburg Experiments

### Download the Data

Download the Hamburg CSV from:

https://cloud.hcu-hamburg.de/nextcloud/s/JaakBq8R7WBteD3

Place the CSV at the repository root with this filename:

```text
hamburg_buildings_facade_amenities_trees.csv
```

The file is ignored by git.

### Convert the CSV

Convert the Hamburg data to ESPM-3D object JSONL and write the five suggested
Hamburg patterns:

```bash
mkdir -p data examples results

uv run python convert_hamburg_data.py convert \
  hamburg_buildings_facade_amenities_trees.csv \
  --objects-out data/hamburg_objects.jsonl \
  --patterns-out examples/hamburg_patterns_5.json \
  --metadata-out data/hamburg_conversion_metadata.json
```

The converter turns base prism/area features into bottom-center and top-center
objects. By default, facade `Door` and `Window` rows are converted into bottom
and top endpoint objects. The metadata JSON records row counts, object counts,
origin shift, bounds, feature-type counts, role counts, and common keywords.

Useful converter options:

```bash
# Keep original projected coordinates instead of subtracting min x/y/z.
uv run python convert_hamburg_data.py convert input.csv --no-normalize-origin

# Also emit a centroid point for each base feature.
uv run python convert_hamburg_data.py convert input.csv --include-centroid

# Ignore facade openings, or emit one center point per opening.
uv run python convert_hamburg_data.py convert input.csv --opening-policy skip
uv run python convert_hamburg_data.py convert input.csv --opening-policy point

# Debug on a prefix of the CSV.
uv run python convert_hamburg_data.py convert input.csv --max-rows 50000
```

### Run Hamburg Patterns

Run the five baseline Hamburg patterns against the converted object file:

```bash
uv run python convert_hamburg_data.py run-patterns \
  --objects data/hamburg_objects.jsonl \
  --patterns examples/hamburg_patterns_5.json \
  --results-out results/hamburg_baseline_results.jsonl \
  --summary-out results/hamburg_baseline_summary.json \
  --match-limit 1000
```

The output JSONL contains one row per pattern with timing, returned matches,
limit status, n-match/e-match counts, skip-edge information, and keyword
postings by vertex.

To inspect the suggested Hamburg patterns:

```bash
uv run python convert_hamburg_data.py list-patterns
```

## Formatting Synthetic Results Into Tables

After running synthetic experiments, you can format the scenario summaries into a LaTeX
table:

```bash
uv run python format_results_latex.py \
  --results-root results \
  --sizes 10k 100k 1m 10m \
  --sparsity-profiles very_sparse_graph sparse_graph medium_graph dense_graph very_dense_graph \
  --output results/scalability_table.tex
```

By default, the table uses:

- `scenario_total_time_s` for time, which includes data generation, index
  construction, and all pattern matching for that scenario;
- `rss_peak_scenario_mb` for RAM, converted to GB in the table.

Use total matching time only:

```bash
uv run python format_results_latex.py \
  --results-root results \
  --time-metric match_time_total_s \
  --output results/match_time_table.tex
```

Make a table for one specific synthetic pattern from
`*_summary_by_pattern.json` files:

```bash
uv run python format_results_latex.py \
  --results-root results \
  --source pattern \
  --pattern-name p01_residential_area \
  --sizes 10k 100k 1m 10m \
  --sparsity-profiles very_sparse_graph sparse_graph medium_graph dense_graph very_dense_graph \
  --output results/p01_scalability_table.tex
```

If a run enabled `--memory-trace-interval` and a scenario was killed before
writing a normal summary, the formatter can recover the best available peak RAM
from `*_memory_trace.jsonl`. Such values are marked with a LaTeX dagger.

The generated table uses `booktabs`; add this to your preamble:

```latex
\usepackage{booktabs}
```

If you pass `--resize-to-textwidth`, also add:

```latex
\usepackage{graphicx}
```

## Development Checks

Run the full test suite:

```bash
uv run pytest -q
```

Run a generator and matcher smoke test:

```bash
uv run python generate_synthetic_data.py smoke-test \
  --n-objects 2000 \
  --pattern-count 5
```

Run a small benchmark that also writes a memory trace:

```bash
uv run main.py \
  --scales 300 \
  --sparsity-profiles manual \
  --pattern-count 2 \
  --match-limit 10 \
  --memory-trace-interval 0.01 \
  --output-dir results/dev_smoke
```
