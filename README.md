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
├── pyproject.toml                  # Package metadata, dependencies, console scripts
├── README.md
├── LICENSE
├── examples/
│   └── default_patterns_20.json    # JSON copy of the built-in pattern suite
├── results/
│   └── .gitkeep                    # Benchmark outputs are written here by default
├── src/
│   └── espm3d/
│       ├── __init__.py
│       ├── matcher.py              # Index, matcher, pattern/object classes
│       ├── generate_synthetic_data.py
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

### Reading `<run_id>_results.jsonl`

`results.jsonl` is JSON Lines: each line is one complete pattern run. This is the best file for detailed analysis because it preserves per-pattern metrics and nested per-edge data.

Useful fields:

- `scenario_id`, `sparsity_profile`, `n_objects`: identify the benchmark scenario.
- `pattern_name`, `pattern_vertices`, `pattern_edges`, `exclusive_edges`, `interval_scale`: identify the pattern and its complexity.
- `status`: `ok`, `error`, or `scenario_error`.
- `matches_returned`: number of matches returned for this pattern, subject to `match_limit`.
- `limit_reached`: true if the match limit was reached.
- `match_time_s`: matching time for this one pattern.
- `rss_peak_match_mb`: peak RSS during this one pattern match.
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
- `--seed N` controls random generation.
- `--output-dir PATH` controls where result files go.

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

## A note on 10 million objects

The `full` profile includes 10 million objects and three sparsity profiles. This is intentionally ambitious. A practical ramp-up is:

```bash
python main.py --scales 10000 --sparsity-profiles medium_graph --pattern-count 20
python main.py --scales 100000 --sparsity-profiles sparse_graph dense_graph --pattern-count 20
python main.py --scales 1m --sparsity-profiles sparse_graph --match-limit 1000
python main.py --scales 10m --sparsity-profiles very_sparse_graph --match-limit 1000 --max-level 10
```
