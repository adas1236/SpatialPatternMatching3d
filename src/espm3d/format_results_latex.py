"""Create LaTeX tables from ESPM-3D benchmark results.

The benchmark runner writes one scenario-level summary JSON per run:

    <run_id>_summary.json

and, in newer versions, a pattern-level summary:

    <run_id>_summary_by_pattern.json

This module scans a results directory recursively, loads those summaries, and
pivots them into a table whose rows are dataset sizes and whose columns are
sparsity profiles.  Each sparsity profile receives two subcolumns: one for a
selected time metric and one for a selected RAM metric.  By default, the table reports scenario total time in seconds and maximum total RAM in GB.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_SPARSITY_ORDER = (
    "very_sparse_graph",
    "sparse_graph",
    "medium_graph",
    "dense_graph",
    "very_dense_graph",
)

SPARSITY_LABELS: Mapping[str, str] = {
    "very_sparse_graph": "Very sparse",
    "sparse_graph": "Sparse",
    "medium_graph": "Medium",
    "dense_graph": "Dense",
    "very_dense_graph": "Very dense",
    "manual": "Manual",
}

TIME_METRIC_LABELS: Mapping[str, str] = {
    "scenario_total_time_s": "Scenario total time",
    "match_time_total_s": "Total match time",
    "match_time_mean_s": "Mean match time",
    "match_time_median_s": "Median match time",
    "match_time_max_s": "Max match time",
    "generation_time_s": "Generation time",
    "index_build_time_s": "Index build time",
}

RAM_METRIC_LABELS: Mapping[str, str] = {
    "rss_peak_scenario_mb": "Maximum total RAM",
    "rss_peak_scenario_delta_mb": "Peak scenario RSS delta",
    "rss_peak_match_max_mb": "Peak match RSS",
    "rss_peak_match_delta_max_mb": "Peak match RSS delta",
    "rss_peak_match_delta_mean_mb": "Mean match RSS delta",
}

TIME_UNITS: Mapping[str, Tuple[str, float]] = {
    "s": ("s", 1.0),
    "min": ("min", 60.0),
    "h": ("h", 3600.0),
}

RAM_UNITS: Mapping[str, Tuple[str, float]] = {
    # Benchmark RSS fields are computed as bytes / 1024**2.
    # For report-style labels, GB/TB use the corresponding binary conversion
    # from these stored values while displaying the requested unit label.
    "MB": ("MB", 1.0),
    "GB": ("GB", 1024.0),
    "TB": ("TB", 1024.0 * 1024.0),
    # Backward-compatible explicit binary unit names.
    "MiB": ("MiB", 1.0),
    "GiB": ("GiB", 1024.0),
    "TiB": ("TiB", 1024.0 * 1024.0),
}


@dataclass(frozen=True)
class LoadedRow:
    """A summary row plus enough metadata to resolve duplicates."""

    key: Tuple[int, str]
    row: Dict[str, Any]
    source_path: Path
    source_mtime: float
    source_kind: str


@dataclass(frozen=True)
class TraceFallback:
    """Peak-memory information recovered from *_memory_trace.jsonl."""

    key: Tuple[int, str]
    peak_rss_mb: Optional[float]
    elapsed_s: Optional[float]
    source_path: Path
    source_mtime: float


def parse_size_token(value: str) -> int:
    """Parse sizes like 10000, 10k, 1m, and 10m."""

    text = value.strip().lower().replace(",", "")
    multiplier = 1
    if text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("b"):
        multiplier = 1_000_000_000
        text = text[:-1]
    number = float(text)
    if number <= 0:
        raise ValueError(f"dataset size must be positive: {value!r}")
    result = int(number * multiplier)
    if not math.isclose(result, number * multiplier):
        raise ValueError(f"dataset size must resolve to an integer: {value!r}")
    return result


def compact_size_label(n_objects: int) -> str:
    """Return compact labels such as 10K, 100K, 1M, and 10M."""

    if n_objects >= 1_000_000 and n_objects % 1_000_000 == 0:
        return f"{n_objects // 1_000_000}M"
    if n_objects >= 1_000 and n_objects % 1_000 == 0:
        return f"{n_objects // 1_000}K"
    return f"{n_objects}"


def latex_escape(text: object) -> str:
    """Escape text for normal LaTeX text mode."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse JSON file {path}: {exc}") from exc
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"expected JSON object or list in {path}")
    rows: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            rows.append(dict(item))
    return rows


def _summary_glob(results_root: Path, source: str) -> Iterable[Path]:
    if source == "scenario":
        yield from results_root.rglob("*_summary.json")
    elif source == "pattern":
        yield from results_root.rglob("*_summary_by_pattern.json")
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(source)


def load_summary_rows(
    results_root: Path,
    *,
    source: str = "scenario",
    pattern_name: Optional[str] = None,
    interval_scale: Optional[float] = None,
) -> List[LoadedRow]:
    """Load benchmark summary rows recursively from ``results_root``."""

    loaded: List[LoadedRow] = []
    for path in _summary_glob(results_root, source):
        # Avoid accidentally reading pattern summaries when using the broader
        # scenario glob on unusual filesystems or shells.
        if source == "scenario" and "summary_by_pattern" in path.name:
            continue
        mtime = path.stat().st_mtime
        for row in _load_json_array(path):
            if source == "pattern":
                if pattern_name is not None and row.get("pattern_name") != pattern_name:
                    continue
                if interval_scale is not None:
                    row_scale = _as_number(row.get("interval_scale"))
                    if row_scale is None or not math.isclose(row_scale, interval_scale):
                        continue
            n_objects = _as_number(row.get("n_objects"))
            profile = row.get("sparsity_profile")
            if n_objects is None or profile is None:
                continue
            key = (int(n_objects), str(profile))
            loaded.append(
                LoadedRow(
                    key=key,
                    row=row,
                    source_path=path,
                    source_mtime=mtime,
                    source_kind=source,
                )
            )
    return loaded


def load_trace_fallbacks(results_root: Path) -> Dict[Tuple[int, str], TraceFallback]:
    """Load best-effort peak RSS data from memory trace sidecar files."""

    best: Dict[Tuple[int, str], TraceFallback] = {}
    working: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for path in results_root.rglob("*_memory_trace.jsonl"):
        mtime = path.stat().st_mtime
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_objects = _as_number(row.get("n_objects"))
                profile = row.get("sparsity_profile")
                if n_objects is None or profile is None:
                    continue
                key = (int(n_objects), str(profile))
                peak = _as_number(row.get("peak_rss_mb"))
                if peak is None:
                    peak = _as_number(row.get("rss_mb"))
                elapsed = _as_number(row.get("elapsed_s"))
                state = working.setdefault(
                    key,
                    {
                        "peak_rss_mb": None,
                        "elapsed_s": None,
                        "source_path": path,
                        "source_mtime": mtime,
                    },
                )
                if peak is not None:
                    current_peak = state["peak_rss_mb"]
                    state["peak_rss_mb"] = peak if current_peak is None else max(current_peak, peak)
                if elapsed is not None:
                    current_elapsed = state["elapsed_s"]
                    state["elapsed_s"] = elapsed if current_elapsed is None else max(current_elapsed, elapsed)
                if mtime >= state["source_mtime"]:
                    state["source_path"] = path
                    state["source_mtime"] = mtime

    for key, state in working.items():
        existing = best.get(key)
        candidate = TraceFallback(
            key=key,
            peak_rss_mb=state["peak_rss_mb"],
            elapsed_s=state["elapsed_s"],
            source_path=state["source_path"],
            source_mtime=state["source_mtime"],
        )
        if existing is None or candidate.source_mtime > existing.source_mtime:
            best[key] = candidate
    return best


def select_rows(
    rows: Sequence[LoadedRow],
    *,
    duplicate_policy: str = "latest",
) -> Dict[Tuple[int, str], LoadedRow]:
    """Resolve duplicate rows for the same size/profile pair."""

    grouped: Dict[Tuple[int, str], List[LoadedRow]] = {}
    for row in rows:
        grouped.setdefault(row.key, []).append(row)

    selected: Dict[Tuple[int, str], LoadedRow] = {}
    duplicates = {key: values for key, values in grouped.items() if len(values) > 1}
    if duplicates and duplicate_policy == "error":
        details = ", ".join(f"{key} has {len(values)} rows" for key, values in sorted(duplicates.items()))
        raise ValueError(f"duplicate summary rows found; use --duplicate-policy latest to choose one: {details}")

    for key, values in grouped.items():
        if duplicate_policy == "latest":
            selected[key] = max(values, key=lambda item: (item.source_mtime, str(item.source_path)))
        elif duplicate_policy == "first":
            selected[key] = values[0]
        elif duplicate_policy == "error":
            selected[key] = values[0]
        else:  # pragma: no cover - argparse prevents this
            raise ValueError(duplicate_policy)
    return selected


def _row_is_incomplete(row: Mapping[str, Any]) -> bool:
    status = row.get("status")
    if status not in (None, "", "ok"):
        return True

    patterns_error = _as_number(row.get("patterns_error"))
    if patterns_error is not None and patterns_error > 0:
        return True

    pattern_runs_error = _as_number(row.get("pattern_runs_error"))
    if pattern_runs_error is not None and pattern_runs_error > 0:
        return True

    attempted = _as_number(row.get("patterns_attempted"))
    ok = _as_number(row.get("patterns_ok"))
    if attempted is not None and ok is not None and ok < attempted:
        return True

    attempted = _as_number(row.get("pattern_runs_attempted"))
    ok = _as_number(row.get("pattern_runs_ok"))
    if attempted is not None and ok is not None and ok < attempted:
        return True

    return bool(row.get("trace_only"))


def _format_number(value: float, precision: int) -> str:
    """Format a numeric table cell without thousands separators."""

    return f"{value:.{precision}f}"


def _metric_value(
    row: Mapping[str, Any],
    metric: str,
    *,
    trace: Optional[TraceFallback] = None,
    allow_trace_ram_fallback: bool = True,
) -> Tuple[Optional[float], bool]:
    """Return ``(value, is_partial)`` for a requested metric."""

    value = _as_number(row.get(metric))
    if value is not None:
        return value, False

    if allow_trace_ram_fallback and metric in RAM_METRIC_LABELS and trace is not None:
        if trace.peak_rss_mb is not None:
            return trace.peak_rss_mb, True

    return None, False


def _format_metric_cell(
    row: Optional[Mapping[str, Any]],
    metric: str,
    *,
    trace: Optional[TraceFallback],
    scale_factor: float,
    precision: int,
    missing: str,
    fail_text: str,
    mark_incomplete: bool,
    allow_trace_ram_fallback: bool,
) -> str:
    if row is None:
        # A trace without a summary usually means the scenario did not complete
        # before the worker/job was killed.  Show failure for time-like cells but
        # still use the trace to recover the best available peak RAM estimate.
        if trace is not None and metric in RAM_METRIC_LABELS and allow_trace_ram_fallback:
            if trace.peak_rss_mb is not None:
                return _format_number(trace.peak_rss_mb / scale_factor, precision) + (r"$^{\dagger}$" if mark_incomplete else "")
        if trace is not None:
            return fail_text
        return missing

    value, partial = _metric_value(row, metric, trace=trace, allow_trace_ram_fallback=allow_trace_ram_fallback)
    incomplete = _row_is_incomplete(row) or partial
    if value is None:
        return fail_text if incomplete else missing

    rendered = _format_number(value / scale_factor, precision)
    if mark_incomplete and incomplete:
        rendered += r"$^{\dagger}$"
    return rendered


def _default_profiles(keys: Iterable[Tuple[int, str]]) -> List[str]:
    present = {profile for _, profile in keys}
    ordered = [profile for profile in DEFAULT_SPARSITY_ORDER if profile in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _default_sizes(keys: Iterable[Tuple[int, str]]) -> List[int]:
    return sorted({n_objects for n_objects, _ in keys})


def build_latex_table(
    rows: Mapping[Tuple[int, str], LoadedRow],
    *,
    traces: Optional[Mapping[Tuple[int, str], TraceFallback]] = None,
    sizes: Optional[Sequence[int]] = None,
    sparsity_profiles: Optional[Sequence[str]] = None,
    time_metric: str = "scenario_total_time_s",
    ram_metric: str = "rss_peak_scenario_mb",
    time_unit: str = "s",
    ram_unit: str = "GB",
    time_precision: int = 2,
    ram_precision: int = 1,
    caption: str = "ESPM-3D scalability results.",
    label: str = "tab:espm3d_scalability",
    missing: str = "---",
    fail_text: str = r"\textsc{fail}",
    mark_incomplete: bool = True,
    allow_trace_ram_fallback: bool = True,
    include_table_environment: bool = True,
    use_small: bool = True,
    resize_to_textwidth: bool = False,
) -> str:
    """Build a LaTeX table from selected summary rows."""

    traces = dict(traces or {})
    all_keys = set(rows.keys()) | set(traces.keys())
    if not all_keys:
        raise ValueError("no benchmark summary or memory-trace rows were found")

    if time_metric not in TIME_METRIC_LABELS:
        raise ValueError(f"unknown time metric: {time_metric}")
    if ram_metric not in RAM_METRIC_LABELS:
        raise ValueError(f"unknown RAM metric: {ram_metric}")
    if time_unit not in TIME_UNITS:
        raise ValueError(f"unknown time unit: {time_unit}")
    if ram_unit not in RAM_UNITS:
        raise ValueError(f"unknown RAM unit: {ram_unit}")

    selected_sizes = list(sizes) if sizes is not None else _default_sizes(all_keys)
    selected_profiles = list(sparsity_profiles) if sparsity_profiles is not None else _default_profiles(all_keys)
    time_unit_label, time_scale = TIME_UNITS[time_unit]
    ram_unit_label, ram_scale = RAM_UNITS[ram_unit]

    time_label = f"Time ({time_unit_label})"
    ram_label = f"Peak RAM ({ram_unit_label})"

    any_incomplete = False
    for key in all_keys:
        loaded = rows.get(key)
        if loaded is None:
            any_incomplete = True
        else:
            any_incomplete = any_incomplete or _row_is_incomplete(loaded.row)

    lines: List[str] = []
    lines.append(r"% Requires \usepackage{booktabs}" + (r" and \usepackage{graphicx}" if resize_to_textwidth else ""))
    lines.append(f"% Time metric: {time_metric} ({TIME_METRIC_LABELS[time_metric]})")
    lines.append(f"% RAM metric: {ram_metric} ({RAM_METRIC_LABELS[ram_metric]})")
    if include_table_environment:
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{latex_escape(caption)}}}")
        lines.append(f"\\label{{{latex_escape(label)}}}")
        if use_small:
            lines.append(r"\small")
    if resize_to_textwidth:
        lines.append(r"\resizebox{\textwidth}{!}{%")

    col_spec = "l" + "rr" * len(selected_profiles)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header1 = [r"\textbf{Dataset size}"]
    for profile in selected_profiles:
        label_text = SPARSITY_LABELS.get(profile, profile.replace("_", " ").title())
        header1.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{latex_escape(label_text)}}}}}")
    lines.append(" & ".join(header1) + r" \\")

    cmidrules = []
    for idx in range(len(selected_profiles)):
        start = 2 + 2 * idx
        end = start + 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append("".join(cmidrules))

    header2 = [""]
    for _profile in selected_profiles:
        header2.extend([f"\\textbf{{{latex_escape(time_label)}}}", f"\\textbf{{{latex_escape(ram_label)}}}"])
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for n_objects in selected_sizes:
        cells = [latex_escape(compact_size_label(n_objects))]
        for profile in selected_profiles:
            key = (n_objects, profile)
            loaded = rows.get(key)
            trace = traces.get(key)
            row = loaded.row if loaded is not None else None
            cells.append(
                _format_metric_cell(
                    row,
                    time_metric,
                    trace=trace,
                    scale_factor=time_scale,
                    precision=time_precision,
                    missing=missing,
                    fail_text=fail_text,
                    mark_incomplete=mark_incomplete,
                    allow_trace_ram_fallback=allow_trace_ram_fallback,
                )
            )
            cells.append(
                _format_metric_cell(
                    row,
                    ram_metric,
                    trace=trace,
                    scale_factor=ram_scale,
                    precision=ram_precision,
                    missing=missing,
                    fail_text=fail_text,
                    mark_incomplete=mark_incomplete,
                    allow_trace_ram_fallback=allow_trace_ram_fallback,
                )
            )
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize_to_textwidth:
        lines.append(r"}%")
    if include_table_environment and any_incomplete and mark_incomplete:
        lines.append(r"\vspace{0.25em}")
        lines.append(
            r"\footnotesize{$^{\dagger}$ Scenario did not complete successfully, "
            r"or the value was recovered from the memory trace; treat it as partial.}"
        )
    if include_table_environment:
        lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def build_table_from_results(
    results_root: Path,
    *,
    source: str = "scenario",
    pattern_name: Optional[str] = None,
    interval_scale: Optional[float] = None,
    duplicate_policy: str = "latest",
    sizes: Optional[Sequence[int]] = None,
    sparsity_profiles: Optional[Sequence[str]] = None,
    time_metric: str = "scenario_total_time_s",
    ram_metric: str = "rss_peak_scenario_mb",
    time_unit: str = "s",
    ram_unit: str = "GB",
    time_precision: int = 2,
    ram_precision: int = 1,
    caption: str = "ESPM-3D scalability results.",
    label: str = "tab:espm3d_scalability",
    missing: str = "---",
    fail_text: str = r"\textsc{fail}",
    mark_incomplete: bool = True,
    allow_trace_ram_fallback: bool = True,
    include_table_environment: bool = True,
    use_small: bool = True,
    resize_to_textwidth: bool = False,
) -> str:
    loaded = load_summary_rows(
        results_root,
        source=source,
        pattern_name=pattern_name,
        interval_scale=interval_scale,
    )
    selected = select_rows(loaded, duplicate_policy=duplicate_policy)
    traces = load_trace_fallbacks(results_root) if allow_trace_ram_fallback else {}
    return build_latex_table(
        selected,
        traces=traces,
        sizes=sizes,
        sparsity_profiles=sparsity_profiles,
        time_metric=time_metric,
        ram_metric=ram_metric,
        time_unit=time_unit,
        ram_unit=ram_unit,
        time_precision=time_precision,
        ram_precision=ram_precision,
        caption=caption,
        label=label,
        missing=missing,
        fail_text=fail_text,
        mark_incomplete=mark_incomplete,
        allow_trace_ram_fallback=allow_trace_ram_fallback,
        include_table_environment=include_table_environment,
        use_small=use_small,
        resize_to_textwidth=resize_to_textwidth,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a LaTeX table from ESPM-3D benchmark summaries. The default table "
            "uses dataset size as rows and sparsity profile as columns, with Time (s) and "
            "Peak RAM (GB) subcolumns for each sparsity profile."
        )
    )
    parser.add_argument("--results-root", default="results", help="Root directory containing benchmark output files.")
    parser.add_argument("--output", "--out", default=None, help="Write LaTeX to this path instead of stdout.")
    parser.add_argument(
        "--source",
        choices=("scenario", "pattern"),
        default="scenario",
        help="Use scenario-level *_summary.json files or pattern-level *_summary_by_pattern.json files.",
    )
    parser.add_argument(
        "--pattern-name",
        default=None,
        help="Pattern name to use when --source pattern is selected, e.g. p01_residential_area.",
    )
    parser.add_argument(
        "--interval-scale",
        type=float,
        default=None,
        help="Optional interval scale filter when --source pattern is selected.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("latest", "first", "error"),
        default="latest",
        help="How to handle multiple summary rows for the same size/profile pair.",
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        default=None,
        help="Optional row order, e.g. --sizes 10k 100k 1m 10m. Defaults to all sizes found.",
    )
    parser.add_argument(
        "--sparsity-profiles",
        nargs="*",
        default=None,
        help=(
            "Optional column order, e.g. --sparsity-profiles very_sparse_graph sparse_graph "
            "medium_graph dense_graph. Defaults to known sparsity profiles found in the results."
        ),
    )
    parser.add_argument(
        "--time-metric",
        choices=tuple(TIME_METRIC_LABELS),
        default="scenario_total_time_s",
        help="Time metric to put in the Time subcolumns. Defaults to end-to-end scenario time.",
    )
    parser.add_argument(
        "--ram-metric",
        choices=tuple(RAM_METRIC_LABELS),
        default="rss_peak_scenario_mb",
        help=(
            "RAM metric to put in the Peak RAM subcolumns. Default is rss_peak_scenario_mb, "
            "the maximum total process RSS during the scenario."
        ),
    )
    parser.add_argument("--time-unit", choices=tuple(TIME_UNITS), default="s")
    parser.add_argument("--ram-unit", choices=tuple(RAM_UNITS), default="GB")
    parser.add_argument("--time-precision", type=int, default=2)
    parser.add_argument("--ram-precision", type=int, default=1)
    parser.add_argument("--caption", default="ESPM-3D scalability results.")
    parser.add_argument("--label", default="tab:espm3d_scalability")
    parser.add_argument("--missing", default="---", help="Cell text for missing results.")
    parser.add_argument("--fail-text", default=r"\textsc{fail}", help="Cell text for failed results with no metric value.")
    parser.add_argument("--no-mark-incomplete", action="store_true", help="Do not append dagger markers to incomplete rows.")
    parser.add_argument(
        "--no-trace-fallback",
        action="store_true",
        help="Do not use *_memory_trace.jsonl to recover RAM values for failed/incomplete scenarios.",
    )
    parser.add_argument("--fragment", action="store_true", help="Only emit the tabular environment, not a full table float.")
    parser.add_argument("--no-small", action="store_true", help="Do not insert \\small before the table.")
    parser.add_argument(
        "--resize-to-textwidth",
        action="store_true",
        help="Wrap the tabular in \\resizebox{\\textwidth}{!}{...}. Requires graphicx.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    results_root = Path(args.results_root)
    if not results_root.exists():
        print(f"results root does not exist: {results_root}", file=sys.stderr)
        return 2

    if args.source == "pattern" and not args.pattern_name:
        print("--pattern-name is required when --source pattern is used", file=sys.stderr)
        return 2

    sizes = [parse_size_token(item) for item in args.sizes] if args.sizes else None
    profiles = list(args.sparsity_profiles) if args.sparsity_profiles else None

    try:
        latex = build_table_from_results(
            results_root,
            source=args.source,
            pattern_name=args.pattern_name,
            interval_scale=args.interval_scale,
            duplicate_policy=args.duplicate_policy,
            sizes=sizes,
            sparsity_profiles=profiles,
            time_metric=args.time_metric,
            ram_metric=args.ram_metric,
            time_unit=args.time_unit,
            ram_unit=args.ram_unit,
            time_precision=args.time_precision,
            ram_precision=args.ram_precision,
            caption=args.caption,
            label=args.label,
            missing=args.missing,
            fail_text=args.fail_text,
            mark_incomplete=not args.no_mark_incomplete,
            allow_trace_ram_fallback=not args.no_trace_fallback,
            include_table_environment=not args.fragment,
            use_small=not args.no_small,
            resize_to_textwidth=args.resize_to_textwidth,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(latex, encoding="utf-8")
        print(f"wrote {output}", file=sys.stderr)
    else:
        sys.stdout.write(latex)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
