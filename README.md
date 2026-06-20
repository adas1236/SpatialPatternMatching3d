# ESPM-3D

A clean Python repository for generalized **3-D spatial pattern matching**. It contains:

- an in-memory inverted octree index;
- a 3-D ESPM-style matcher using minimum bounding rectangular prisms;
- a synthetic 3-D spatio-textual data generator;
- a 20-pattern default benchmark suite; and
- a scalability benchmark runner that writes timing, memory, index, and implicit candidate-graph sparsity metrics to CSV, JSONL, and summary JSON.

The implementation favors clarity and reproducibility. It is not a disk-backed production index; very large runs, especially 10 million objects, can require substantial RAM and time.

## Project structure

```text
espm3d_repo/
├── main.py                         # Benchmark entry point
├── generate_synthetic_data.py      # Thin CLI wrapper for the synthetic data generator
├── convert_hamburg_data.py         # Thin CLI wrapper for the Hamburg CSV converter
├── format_results_latex.py         # Thin CLI wrapper for LaTeX table formatting
├── pyproject.toml                  # Package metadata, dependencies, console scripts
├── README.md
├── LICENSE
├── examples/
│   ├── default_patterns_20.json    # JSON copy of the built-in synthetic pattern suite
│   └── hamburg_patterns_5.json     # Suggested real-data patterns for the Hamburg CSV
├── results/
│   └── .gitkeep                    # Benchmark outputs are written here by default
├── src/
│   └── espm3d/
│       ├── __init__.py
│       ├── matcher.py              # Index, matcher, pattern/object classes
│       ├── generate_synthetic_data.py
│       ├── convert_hamburg_data.py # Hamburg real-data conversion and patterns
│       ├── format_results_latex.py
│       └── benchmark.py            # Scalability and graph-sparsity benchmark runner
└── tests/
    ├── test_matcher.py
    ├── test_generate_synthetic_data.py
    └── test_benchmark.py
```

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q
```

You can also run the scripts from the repository root without installing; `main.py` and `generate_synthetic_data.py` add `src/` to `sys.path` automatically.


## Convert the Hamburg real-data CSV

The repository includes a converter for the Hamburg buildings/facades/amenities/
trees CSV.  That CSV is not purely a point dataset: most rows describe an
axis-aligned rectangular prism or a 2-D area through `lowerCorner_*` and
`upperCorner_*`.  The converter turns each base feature into ESPM-3D point
objects by using the **bottom-center** and **top-center** of that prism.  This
preserves vertical information better than a single centroid.  By default it
also converts facade `Door`/`Window` rows into bottom and top endpoint objects.

```bash
uv run python convert_hamburg_data.py convert \
  hamburg_buildings_facade_amenities_trees.csv \
  --objects-out data/hamburg_objects.jsonl \
  --patterns-out data/hamburg_patterns_5.json \
  --metadata-out data/hamburg_conversion_metadata.json
```

The object JSONL is compatible with `load_objects_jsonl`, and the pattern JSON
is compatible with `load_patterns_json`.

Useful converter options:

```bash
# Keep the original large projected coordinates instead of subtracting min x/y/z.
uv run python convert_hamburg_data.py convert input.csv --no-normalize-origin

# Also emit a single centroid object per base prism/area.
uv run python convert_hamburg_data.py convert input.csv --include-centroid

# Ignore facade openings, or emit one center point per opening instead of endpoints.
uv run python convert_hamburg_data.py convert input.csv --opening-policy skip
uv run python convert_hamburg_data.py convert input.csv --opening-policy point

# For flat OSM area features with missing z, set the imputed bottom/top z values.
uv run python convert_hamburg_data.py convert input.csv --flat-z 0 --flat-thickness 1

# Debug on a small prefix of the CSV.
uv run python convert_hamburg_data.py convert input.csv --max-rows 50000
```

The converter writes `hamburg_conversion_metadata.json` with row counts, object
counts, origin shift, before/after bounds, feature-type counts, role counts, and
the most common emitted keywords.  Keyword examples include `building_bottom`,
`building_top`, `tree_bottom`, `park_bottom`, `apartments_bottom`,
`school_bottom`, `door_bottom`, and `window_top`.

List the suggested real-data patterns:

```bash
uv run python convert_hamburg_data.py list-patterns
```

Smoke-test a small slice of the CSV:

```bash
uv run python convert_hamburg_data.py smoke-test \
  hamburg_buildings_facade_amenities_trees.csv \
  --max-rows 50000 \
  --match-limit 10
```

Run the five patterns against the converted full object file:

```bash
uv run python convert_hamburg_data.py run-patterns \
  --objects data/hamburg_objects.jsonl \
  --patterns data/hamburg_patterns_5.json \
  --results-out data/hamburg_pattern_results.jsonl \
  --summary-out data/hamburg_pattern_summary.json \
  --match-limit 1000
```

`run-patterns` builds the same inverted octree index used by the matcher, runs
each pattern, and writes per-pattern timing, match counts, n-match/e-match
counts, keyword postings, skip-edge information, and rough RSS memory readings.
Use `--match-limit 0` only when you really want to enumerate every match.

The five suggested Hamburg patterns are:

1. `hamburg_p01_highrise_vertical_extent`: `highrise_bottom` near `highrise_top`.
2. `hamburg_p02_facade_door_window_stack`: `door_bottom`, `window_top`, and `building_top`.
3. `hamburg_p03_hospital_green_buffer`: `hospital_bottom`, `park_bottom`, and `tree_bottom`.
4. `hamburg_p04_education_play_cluster`: `school_bottom`, `kindergarten_bottom`, and `playground_bottom`.
5. `hamburg_p05_transit_retail_parking`: `train_station_bottom`, `retail_bottom`, and `parking_bottom`.

These are intentionally baseline patterns, not claims about urban planning.  You
should tune distance intervals after inspecting match counts and selectivity on
the converted dataset.

## Run scalability and graph-sparsity benchmarks

Quick smoke benchmark:

```bash
python main.py --profile smoke
```

Laptop-friendly benchmark:

```bash
python main.py --profile standard
```

Paper-inspired large sweep, including 10K, 100K, 1M, and 10M objects:

```bash
python main.py --profile full --match-limit 1000
```

Run exactly the scales you want:

```bash
python main.py --scales 10000 100000 1m --pattern-count 20 --output-dir results/my_run
```

The runner accepts `k` and `m` suffixes, so `100k`, `1m`, and `10m` are valid.

For memory-heavy runs, especially `dense_graph` or `full`, prefer isolated scenarios:

```bash
python main.py --profile standard --isolate-scenarios --malloc-trim-between-patterns
```

`--isolate-scenarios` runs each object-count/sparsity scenario in a fresh child Python process and aggregates the child outputs into the normal result files.  It is a little slower, but it is much safer when one dense scenario uses a lot of memory.  If a child is killed by the OS, the parent benchmark process records a `scenario_process_error` row and continues with the remaining scenarios.

`--malloc-trim-between-patterns` is Linux/WSL-specific best-effort cleanup.  It runs garbage collection and then asks glibc to return freed heap pages to the OS after each pattern run.  Use it when RSS keeps rising across patterns or scenarios.

This update also fixes a scenario-cleanup issue in the previous benchmark runner: the matcher object retained a reference to the scenario index, and the index retained the scenario objects. The cleanup now clears matcher, index, dataset, matches, and stats in dependency order before moving to the next scenario.

## Reproducibility

The benchmark has a reproducibility seed.  If you do not pass one, it uses:

```bash
--seed 42
```

Every scenario receives a deterministic derived seed based on:

```text
base seed + object count + sparsity profile name
```

That means `n100000__sparse_graph` gets the same generated data whether you run only `sparse_graph` or run `sparse_graph medium_graph dense_graph` together.  The actual scenario seed is written in every raw result row as `seed`, and the base config is written to `<run_id>_settings.json`.

Example:

```bash
python main.py --profile standard --seed 12345 --output-dir results/seed_12345
```

## Profiles now include graph-sparsity sweeps

The original packaged benchmark swept object count only. The updated benchmark sweeps object count **and** named `--sparsity-profiles`.

```text
profile    object counts                         default sparsity profiles
smoke      2,000                                 sparse_graph, medium_graph, dense_graph
standard   10,000; 100,000                       sparse_graph, medium_graph, dense_graph
full       10,000; 100,000; 1,000,000; 10,000,000 very_sparse_graph, sparse_graph, medium_graph
```

The built-in sparsity profiles are:

```text
manual             use the explicit generator flags exactly
very_sparse_graph  very large domain, almost uniform points, many noise keywords, rare pattern keywords
sparse_graph       large domain, weak clustering, many noise keywords, uncommon pattern keywords
medium_graph       balanced default setting
dense_graph        smaller domain, strong local clustering, fewer noise keywords, common pattern keywords
very_dense_graph   stress setting; use carefully with small scales or tight match limits
```

List profile definitions:

```bash
python main.py --list-sparsity-profiles
```

Run only selected sparsity profiles:

```bash
python main.py \
  --scales 10000 100000 \
  --sparsity-profiles sparse_graph dense_graph \
  --pattern-count 20
```

Use your own manual sparsity settings:

```bash
python main.py \
  --scales 100000 \
  --sparsity-profiles manual \
  --target-density 1e-6 \
  --noise-keywords 1000 \
  --pattern-keyword-weight 1.0 \
  --clustered-fraction 0.2
```

## What “sparsity” means here

The pattern itself is a small graph, but these sparsity profiles are **not** changing the pattern graph unless you separately change `--pattern-count` or `--interval-scales`.

Here, sparsity means sparsity of the **implicit candidate/e-match graph** induced by a dataset and a pattern:

- candidate graph vertices are spatial objects with pattern keywords;
- candidate graph edges are object pairs that are keyword-compatible for a pattern edge;
- e-match edges are candidate edges that survive distance and sign constraints.

A sparse implicit graph has few e-matches relative to the number of keyword-compatible object pairs. A dense implicit graph has many e-matches, which usually creates more join work.

The generator controls this graph sparsity through three mechanisms.

### 1. Global spatial density

For a fixed number of objects, a larger 3-D volume means objects are farther apart on average. Fewer object pairs fall inside the upper distance bounds, so the implicit graph tends to be sparser.

Manual example:

```bash
python main.py --sparsity-profiles manual --scales 100000 --target-density 1e-6
```

Recorded metrics:

- `domain_volume`
- `object_density = n_objects / domain_volume`
- `expected_spacing = (domain_volume / n_objects) ** (1/3)`

### 2. Local clustering

A dataset can be globally sparse but locally dense. If many objects are sampled from tight clusters, a cluster can contain many objects within the same pattern edge radius. That creates dense candidate-graph components even when the full domain is large.

Manual example:

```bash
python main.py \
  --sparsity-profiles manual \
  --scales 100000 \
  --target-density 1e-6 \
  --clustered-fraction 0.85 \
  --cluster-count 64 \
  --cluster-std-fraction 0.018
```

### 3. Keyword prevalence

ESPM is keyword-first: it builds one octree per keyword, and each pattern vertex searches only the octree for its keyword. If a pattern keyword appears on many objects, the candidate-pair universe is large. If pattern keywords are rare, the implicit graph can remain sparse even in a spatially dense volume.

Manual example:

```bash
python main.py \
  --sparsity-profiles manual \
  --scales 100000 \
  --noise-keywords 1000 \
  --pattern-keyword-weight 1.0 \
  --noise-keyword-weight 3.0 \
  --extra-keyword-probability 0.02
```

Recorded metrics include `keyword_postings_by_vertex`, `keyword_postings_mean_by_vertex`, `keyword_postings_max_by_vertex`, `candidate_pair_space_total`, and `candidate_graph_density_measured`.

## Output files

Each benchmark run writes timestamped files under `results/` or your chosen `--output-dir`:

```text
<run_id>_settings.json    # full benchmark config, profile definitions, selected patterns, and system info
<run_id>_results.csv      # one row per size/profile/pattern run; easiest to inspect manually
<run_id>_results.jsonl    # detailed machine-readable event log, one JSON object per pattern run
<run_id>_summary.csv      # one row per dataset-size/sparsity-profile scenario
<run_id>_summary.json     # JSON version of the scenario summary
<run_id>_summary_by_pattern.csv  # one row per dataset-size/sparsity-profile/pattern summary
<run_id>_summary_by_pattern.json # JSON version of the pattern-split summary
<run_id>_memory_trace.jsonl      # optional periodic RSS trace, written only when --memory-trace-interval > 0
```

### Reading `<run_id>_summary.json`

`summary.json` is a JSON array. Each object summarizes one scenario: one object count plus one graph-sparsity profile. With `--profile standard`, you should see 6 summary objects because there are 2 scales × 3 sparsity profiles.

Important fields:

- `scenario_id`: combines object count and profile, such as `n100000__sparse_graph`.
- `n_objects`: number of generated objects.
- `sparsity_profile`: named graph-sparsity setting.
- `object_density`: actual global object density.
- `expected_spacing`: cube-root average spacing; larger means more spatially sparse.
- `generation_time_s`: time to generate synthetic objects.
- `index_build_time_s`: time to build the inverted octree index.
- `match_time_total_s`: total matching time over all patterns in that scenario.
- `match_time_mean_s`, `match_time_median_s`, `match_time_max_s`: aggregate per-pattern match times.
- `rss_peak_scenario_mb`: peak resident memory while generating, indexing, and matching the scenario.
- `rss_peak_scenario_delta_mb`: peak scenario RSS minus the scenario-start RSS.
- `rss_peak_match_max_mb`: largest per-pattern match-phase RSS peak in this scenario.
- `rss_peak_match_delta_max_mb`: largest per-pattern match-phase RSS increase above that pattern's starting RSS.
- `patterns_attempted`, `patterns_ok`, `patterns_error`: number of pattern runs and failures.
- `limit_reached_count`: how many patterns hit `--match-limit`; when this is nonzero, returned match counts are lower bounds.
- `matches_returned_total`: total matches returned across all patterns, subject to `--match-limit`.
- `ematch_total_mean`: average non-skip e-match count across successful patterns.
- `candidate_graph_density_mean`: average measured implicit graph density over non-skip edges.
- `candidate_pair_space_total_mean`: average keyword-compatible pair space before distance/sign pruning.
- `index_node_count`, `index_leaf_count`, `index_max_depth`: index size/shape metrics.

Quick inspection:

```bash
python - <<'PY'
import json
from pathlib import Path
summary_path = sorted(Path('results').glob('*_summary.json'))[-1]
summary = json.loads(summary_path.read_text())
for row in summary:
    print(row['scenario_id'], row['match_time_mean_s'], row['candidate_graph_density_mean'])
PY
```

### Reading `<run_id>_summary_by_pattern.json`

`summary_by_pattern.json` is also a JSON array, but it does **not** average all
patterns together.  Each object summarizes one pattern within one benchmark
scenario:

```text
one row = one object count × one sparsity profile × one pattern variant
```

This is the easiest file to use when you want to compare scalability pattern by
pattern without digging through the raw JSONL.  For example, if you run 2 scales,
3 sparsity profiles, and 20 patterns, this file will contain `2 × 3 × 20 = 120`
rows.

Important fields:

- `scenario_id`: object-count/profile combination, such as `n100000__sparse_graph`.
- `pattern_name`: the pattern being summarized.
- `interval_scale`: the interval-widening factor for this pattern variant.
- `pattern_vertices`, `pattern_edges`, `exclusive_edges`: pattern complexity.
- `pattern_runs_attempted`, `pattern_runs_ok`, `pattern_runs_error`: useful if repeated runs are added later.
- `match_time_mean_s`, `match_time_median_s`, `match_time_max_s`: timing for this pattern in this scenario.
- `matches_returned_total`, `matches_returned_mean`: returned matches, subject to `match_limit`.
- `limit_reached_count`: nonzero means match counts were capped.
- `candidate_graph_density_mean`: measured implicit graph density for this pattern.
- `candidate_pair_space_total_mean`, `candidate_pair_space_non_skip_mean`: keyword-compatible pair-space size for this pattern.
- `ematch_total_mean`, `nmatch_total_final_level_mean`, `skip_edge_count_mean`: matcher-internal work indicators.
- `rss_before_match_mean_mb`: RSS immediately before the pattern match began.
- `rss_peak_match_max_mb`: peak memory sampled during this pattern match. This is absolute process RSS, so it includes the generated dataset and index already resident in memory.
- `rss_peak_match_delta_mean_mb`, `rss_peak_match_delta_max_mb`: extra RSS consumed during the pattern match, measured as peak match RSS minus match-start RSS.

Example: print a compact pattern-by-pattern table from the latest run.

```bash
python - <<'PY'
import json
from pathlib import Path
summary_path = sorted(Path('results').glob('*_summary_by_pattern.json'))[-1]
rows = json.loads(summary_path.read_text())
for row in rows:
    print(
        row['scenario_id'],
        row['pattern_name'],
        'time=', row['match_time_mean_s'],
        'density=', row['candidate_graph_density_mean'],
        'matches=', row['matches_returned_total'],
    )
PY
```

Example: find the slowest pattern for each scenario.

```bash
python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

summary_path = sorted(Path('results').glob('*_summary_by_pattern.json'))[-1]
rows = [r for r in json.loads(summary_path.read_text()) if r['status'] == 'ok']

by_scenario = defaultdict(list)
for row in rows:
    by_scenario[row['scenario_id']].append(row)

for scenario_id, scenario_rows in sorted(by_scenario.items()):
    slowest = max(scenario_rows, key=lambda r: r['match_time_mean_s'] or 0.0)
    print(scenario_id, slowest['pattern_name'], slowest['match_time_mean_s'])
PY
```

### Reading `<run_id>_results.jsonl`

`results.jsonl` is JSON Lines: each line is one complete pattern run. This is the best file for detailed analysis because it preserves per-pattern metrics and nested per-edge data.

Useful fields:

- `scenario_id`, `sparsity_profile`, `n_objects`: identify the benchmark scenario.
- `pattern_name`, `pattern_vertices`, `pattern_edges`, `exclusive_edges`, `interval_scale`: identify the pattern and its complexity.
- `status`: `ok`, `error`, `scenario_error`, or `scenario_process_error`. The last one means an isolated worker process exited or was killed before writing normal pattern rows.
- `matches_returned`: number of matches returned for this pattern, subject to `match_limit`.
- `limit_reached`: true if the match limit was reached.
- `match_time_s`: matching time for this one pattern.
- `rss_before_match_mb`: RSS immediately before this pattern match began.
- `rss_peak_match_mb`: peak absolute RSS during this one pattern match. This includes the dataset and index that are already resident.
- `rss_peak_match_delta_mb`: additional RSS used during this pattern match, computed as `rss_peak_match_mb - rss_before_match_mb`. This is usually the best per-pattern memory-pressure number.
- `memory_sampler_interval_s`, `memory_trace_interval_s`: memory instrumentation settings used for the run.
- `nmatch_total_all_levels`: total node-pair matches across octree levels.
- `nmatch_total_final_level`: node-pair matches at the final level.
- `ematch_total`: total e-matches for non-skip edges.
- `skip_edge_count`, `skip_edges`: edges whose e-match materialization was skipped and checked during joining.
- `keyword_postings_by_vertex`: how many objects have each pattern vertex keyword.
- `candidate_pair_space_total`: total keyword-compatible object-pair count across all pattern edges.
- `candidate_pair_space_non_skip`: denominator used for measured density; skip edges are excluded because their e-match lists are not materialized.
- `candidate_graph_density_measured`: `candidate_graph_edges_measured / candidate_pair_space_non_skip`; lower means sparser implicit graph.
- `candidate_graph_edges_by_edge`: per-edge pair-space and e-match details.

Example analysis:

```bash
python - <<'PY'
import json
from pathlib import Path
results_path = sorted(Path('results').glob('*_results.jsonl'))[-1]
rows = [json.loads(line) for line in results_path.read_text().splitlines() if line]
slowest = sorted((r for r in rows if r['status'] == 'ok'), key=lambda r: r['match_time_s'], reverse=True)[:10]
for r in slowest:
    print(r['scenario_id'], r['pattern_name'], r['match_time_s'], r['candidate_graph_density_measured'])
PY
```

### Reading `<run_id>_memory_trace.jsonl`

This file is written only when you pass `--memory-trace-interval` with a positive value. It is a periodic RSS trace during pattern matching. The normal raw result row is written after a pattern finishes, but if a worker is OOM-killed mid-pattern, that final row may never be written. The trace file is flushed incrementally, so it can preserve the last observed RSS before the kill.

Example command:

```bash
python main.py \
  --scales 100000 \
  --sparsity-profiles dense_graph \
  --isolate-scenarios \
  --memory-trace-interval 1.0 \
  --output-dir results/dense_trace
```

Each line is a JSON object with fields such as:

- `scenario_id`, `n_objects`, `sparsity_profile`, `pattern_index`, `pattern_name`: the running pattern.
- `sample_kind`: `start`, `sample`, or `end`.
- `elapsed_s`: seconds since this pattern's memory sampler started.
- `rss_mb`: sampled resident memory.
- `peak_rss_mb`: highest sampled RSS so far for this pattern.
- `peak_delta_mb`: highest sampled RSS so far minus pattern-start RSS.
- `pid`: useful when `--isolate-scenarios` is enabled.

Example: print the highest sampled RSS per pattern from the latest trace.

```bash
python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

trace_path = sorted(Path('results').glob('*_memory_trace.jsonl'))[-1]
peaks = defaultdict(float)
for line in trace_path.read_text().splitlines():
    row = json.loads(line)
    key = (row['scenario_id'], row['pattern_name'])
    peaks[key] = max(peaks[key], row.get('peak_rss_mb') or 0.0)

for (scenario_id, pattern_name), peak in sorted(peaks.items(), key=lambda kv: kv[1], reverse=True)[:20]:
    print(scenario_id, pattern_name, peak)
PY
```

## Pattern controls

By default, the benchmark uses the built-in 20 patterns.

Use fewer patterns:

```bash
python main.py --profile smoke --pattern-count 5
```

Create additional pattern variants by widening edge intervals. For example, this runs 20 original patterns plus 20 widened variants:

```bash
python main.py --scales 10000 --sparsity-profiles medium_graph --pattern-count 20 --interval-scales 1.0 1.5
```

Interval scale values must be at least `1.0` so planted pattern templates remain valid.

## Synthetic data generation

List the built-in patterns:

```bash
python generate_synthetic_data.py list-patterns
```

Generate 50,000 objects, 20 patterns, planted matches, and metadata:

```bash
python generate_synthetic_data.py generate \
  --n-objects 50000 \
  --seed 42 \
  --pattern-count 20 \
  --matches-per-pattern 1 \
  --objects-out synthetic_objects.jsonl \
  --patterns-out synthetic_patterns.json \
  --planted-out synthetic_planted_matches.json \
  --metadata-out synthetic_metadata.json
```

Generate background-only data, where patterns may or may not occur naturally:

```bash
python generate_synthetic_data.py generate --n-objects 50000 --no-plant
```

Run a generator + matcher smoke test:

```bash
python generate_synthetic_data.py smoke-test --n-objects 2000 --pattern-count 5
```

## CLI flags

### Benchmark selection

- `--profile {smoke,standard,full}` selects default object counts and default sparsity-profile sweep.
- `--scales ...` overrides profile object counts. Accepts integers plus `k`/`m` suffixes.
- `--sparsity-profiles ...` overrides profile sparsity profiles. Use `manual` to use the individual sparsity flags exactly.
- `--list-sparsity-profiles` prints built-in profile definitions and exits.
- `--pattern-count N` selects the first `N` built-in patterns.
- `--interval-scales ...` creates widened pattern variants.
- `--seed N` controls random generation. Defaults to `42`; scenario-specific seeds are derived from the base seed, object count, and sparsity profile name.
- `--output-dir PATH` controls where result files go.
- `--memory-sampler-interval SECONDS` controls how often RSS is sampled for peak-memory estimates. The default is `0.05`. Smaller values are more precise but add a little overhead.
- `--memory-trace-interval SECONDS` writes periodic per-pattern RSS samples to `<run_id>_memory_trace.jsonl`. The default is `0`, meaning disabled. Use values like `1.0` or `5.0` for large jobs where you want useful memory breadcrumbs if a worker is killed.
- `--isolate-scenarios` runs each object-count/sparsity scenario in a fresh Python process and then aggregates the outputs. This is the safest mode for WSL and dense scenarios because memory is fully released when each child exits.
- `--cleanup-between-patterns` runs Python garbage collection after every pattern run.
- `--malloc-trim-between-patterns` also calls `malloc_trim(0)` on Linux/WSL after every pattern run. This can return freed glibc heap pages to the OS and implies pattern cleanup.

### Match enumeration

- `--match-limit N` caps returned matches per pattern. Use `0` for no cap.
- `--no-plant` disables guaranteed planted matches.
- `--matches-per-pattern N` controls how many planted matches are inserted for each pattern.
- `--allow-same-object` lets one object satisfy multiple pattern vertices.

### Manual sparsity controls

These are used directly only when `--sparsity-profiles manual` is selected.

- `--bounds XMIN YMIN ZMIN XMAX YMAX ZMAX` sets the exact 3-D domain.
- `--domain-side S` sets a cubic domain `[0,S]^3`.
- `--target-density D` chooses a cubic domain with volume `n_objects / D`.
- `--clustered-fraction P` controls the probability that a background object comes from a Gaussian cluster instead of the uniform background.
- `--cluster-count N` controls the number of Gaussian cluster centers.
- `--cluster-std-fraction F` controls cluster tightness as a fraction of the domain side.
- `--noise-keywords N` sets how many non-pattern keywords exist.
- `--pattern-keyword-weight W` controls how common pattern keywords are compared with noise keywords.
- `--noise-keyword-weight W` controls how common noise keywords are compared with pattern keywords.
- `--extra-keyword-probability P` controls whether objects get additional keywords after their primary keyword.
- `--max-keywords-per-object N` caps object keyword count.

### Index and output controls

- `--capacity N` is the octree leaf split threshold.
- `--min-level N` forces objects to at least this octree level.
- `--max-level N` caps octree depth.
- `--write-dataset` writes generated objects, patterns, planted matches, and metadata for every scenario.
- `--isolate-scenarios` runs each dataset-size/sparsity scenario in a fresh worker process. Use this for large WSL runs where you want the parent process to survive a worker OOM kill and continue to later scenarios.
- `--cleanup-between-patterns` runs garbage collection after each pattern. This is usually unnecessary for small runs but can help dense runs.
- `--malloc-trim-between-patterns` also calls Linux/glibc `malloc_trim(0)` after each pattern, which can return freed heap pages to the OS on WSL/Linux.

### Memory-safety controls

- `--isolate-scenarios` is the main reliability switch. It uses a fresh child process per scenario, so Python heap fragmentation or unreleased memory from previous scenarios cannot accumulate in the parent process. If a child dies, the parent writes a `scenario_process_error` raw row, keeps the partial outputs, and moves on.
- `--cleanup-between-patterns` is lighter weight. It calls `gc.collect()` after each pattern. This can help if there is cyclic garbage, but it usually will not reduce RSS by itself.
- `--malloc-trim-between-patterns` is more useful on WSL/Linux. It runs `gc.collect()` and then tries `malloc_trim(0)`, which can return freed heap pages to the OS. This may help after large e-match lists are freed.
- `--memory-trace-interval SECONDS` does not reduce memory, but it makes OOM diagnosis much easier. For HPC jobs, `--memory-trace-interval 5.0` is a low-overhead choice; for a short dense debugging run, `1.0` gives more detail.

A robust WSL command is:

```bash
python main.py \
  --profile standard \
  --isolate-scenarios \
  --malloc-trim-between-patterns \
  --output-dir results/standard_isolated
```

## A note on 10 million objects

The `full` profile includes 10 million objects and three sparsity profiles. This is intentionally ambitious. A practical ramp-up is:

```bash
python main.py --scales 10000 --sparsity-profiles medium_graph --pattern-count 20
python main.py --scales 100000 --sparsity-profiles sparse_graph dense_graph --pattern-count 20
python main.py --scales 1m --sparsity-profiles sparse_graph --match-limit 1000 --isolate-scenarios
python main.py --scales 10m --sparsity-profiles very_sparse_graph --match-limit 1000 --max-level 10 --isolate-scenarios
```

## Formatting benchmark results as a LaTeX table

After running benchmark jobs, you can convert the generated summaries into a
LaTeX table with dataset size as rows and sparsity profile as columns. Each
sparsity profile receives two subcolumns: `Time (s)` and `Peak RAM (GB)`.

For your Slurm output layout, where results are written under paths like
`results/10m/sparse_graph`, run:

```bash
uv run python format_results_latex.py \
  --results-root results \
  --sizes 10k 100k 1m 10m \
  --sparsity-profiles very_sparse_graph sparse_graph medium_graph dense_graph very_dense_graph \
  --output results/scalability_table.tex
```

The default table now uses:

- `scenario_total_time_s` for the time column. This is the end-to-end scenario
  wall time, including synthetic data generation, index construction, and all
  pattern matching in that scenario.
- `rss_peak_scenario_mb` for the RAM column. This is the maximum total process
  RSS observed during the scenario. In other words, it includes the loaded
  synthetic dataset, octree index, Python runtime, and the temporary memory used
  by the matching method. The table converts the stored RSS value to GB and
  prints the header as `Peak RAM (GB)`.

The formatter intentionally prints numeric cells without thousands separators,
so a value such as `12345.68` is emitted instead of `12,345.68`.

Useful alternatives:

```bash
# Use total matching time only, excluding generation and index construction.
uv run python format_results_latex.py \
  --results-root results \
  --time-metric match_time_total_s \
  --output results/match_time_table.tex

# Use absolute peak RSS observed during pattern matching only. This still means
# total process memory, not a delta; it includes the already-loaded index.
uv run python format_results_latex.py \
  --results-root results \
  --ram-metric rss_peak_match_max_mb \
  --output results/match_peak_ram_table.tex

# Make a table for one specific pattern from *_summary_by_pattern.json files.
uv run python format_results_latex.py \
  --results-root results \
  --source pattern \
  --pattern-name p01_residential_area \
  --sizes 10k 100k 1m 10m \
  --sparsity-profiles very_sparse_graph sparse_graph medium_graph dense_graph very_dense_graph \
  --output results/p01_scalability_table.tex
```

If you enabled `--memory-trace-interval` during benchmarking and a scenario was
killed before writing a normal summary, the formatter will use the trace file to
recover the best available peak RAM estimate. Such values are marked with a
LaTeX dagger (`$^\dagger$`) and should be treated as partial.

The generated table uses `booktabs`; add this to your paper preamble:

```latex
\usepackage{booktabs}
```

If you use `--resize-to-textwidth`, also add:

```latex
\usepackage{graphicx}
```
