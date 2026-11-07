#!/usr/bin/env python3
"""Extract CSV/JSON diagnostics from deterministic-control traces."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common.trace_io import (
    find_lambda_entry,
    first_y,
    hit_query,
    lambda_label,
    load_control_traces,
    merge_entries,
    vi_display_curve,
    waterfall_display_curve,
)
from common.waterfall_diagnostics import infer_waterfall_h_events, residual_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="Path to experiment-local figure_spec.json.")
    return parser.parse_args()


def _resolve(exp_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else exp_dir / path


def _trace_group_name(raw_path: str | Path) -> str:
    parts = Path(raw_path).parts
    for idx, part in enumerate(parts):
        if part == "raw" and idx > 0 and parts[idx - 1] == "data" and idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(raw_path).parent.name


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract(spec_path: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    exp_dir = spec_path.parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    derived_dir = _resolve(exp_dir, spec.get("derived_dir", "derived"))
    x_max = float(spec.get("x_max", 100))
    selected_lambdas = [float(value) for value in spec["selected_lambdas"]]

    baseline_entries = load_control_traces(_resolve(exp_dir, spec["baseline_trace_file"]))
    groups = [baseline_entries]
    trace_group_names: dict[str, str] = {}
    for entry in baseline_entries:
        trace_group_names[lambda_label(float(entry["lambda"]))] = "baseline"
    for raw_path in spec.get("extra_trace_files", []):
        group = load_control_traces(_resolve(exp_dir, raw_path))
        groups.append(group)
        group_name = _trace_group_name(raw_path)
        for entry in group:
            trace_group_names.setdefault(lambda_label(float(entry["lambda"])), group_name)

    entries = merge_entries(groups)
    baseline_entry = find_lambda_entry(baseline_entries, float(spec.get("vi_reference_lambda", 0.0)))
    vi_points = vi_display_curve(baseline_entry, x_max=x_max)
    initial_policy_error = first_y(vi_points)

    curve_rows: list[dict[str, Any]] = []
    curve_rows.extend(
        {"series": "VI", "algorithm": "vi", "lambda": "", "x": x, "y": y}
        for x, y in vi_points
    )
    for lambda_value in selected_lambdas:
        entry = find_lambda_entry(entries, lambda_value)
        for x, y in waterfall_display_curve(
            entry,
            initial_error=initial_policy_error,
            x_max=x_max,
            truncate_at_hit=bool(spec.get("truncate_waterfall_at_hit", True)),
        ):
            curve_rows.append(
                {
                    "series": f"waterfall_lambda_{lambda_label(lambda_value)}",
                    "algorithm": "waterfall",
                    "lambda": lambda_value,
                    "x": x,
                    "y": y,
                }
            )

    hit_rows: list[dict[str, Any]] = []
    residual_trace_rows: list[dict[str, Any]] = []
    adaptation_events: dict[str, Any] = {}
    vi_reference_hit = hit_query(baseline_entry, "vi")
    for entry in entries:
        lambda_value = float(entry["lambda"])
        label = lambda_label(lambda_value)
        results = entry.get("results", {})
        hit_rows.append(
            {
                "lambda": lambda_value,
                "trace_group": trace_group_names.get(label, ""),
                "vi_reference_hit": vi_reference_hit,
                "waterfall_hit": results.get("waterfall"),
                "osvi_hit": results.get("osvi"),
                "geo_waterfall_hit": results.get("geo_waterfall"),
                "dd_waterfall_hit": results.get("dd_waterfall"),
            }
        )
        if "waterfall" in entry.get("traces", {}):
            residual_trace_rows.extend(residual_rows(entry, lambda_value))
            adaptation_events[label] = infer_waterfall_h_events(entry, x_max=x_max)

    selected_hit_summary = {
        "vi": vi_reference_hit,
        "waterfall": {
            lambda_label(value): find_lambda_entry(entries, value)["results"]["waterfall"]
            for value in selected_lambdas
        },
    }
    summary = {
        "experiment": spec.get("experiment"),
        "initial_policy_error": initial_policy_error,
        "x_max": x_max,
        "selected_lambdas": selected_lambdas,
        "selected_hit_queries": selected_hit_summary,
        "adaptation_lambda": spec.get("adaptation_lambda"),
        "note": "Adaptive IE-VI h events are inferred from residual traces; h is not persisted directly.",
    }

    _write_csv(
        derived_dir / "selected_display_curves.csv",
        curve_rows,
        ["series", "algorithm", "lambda", "x", "y"],
    )
    _write_csv(
        derived_dir / "all_lambda_hit_times.csv",
        hit_rows,
        [
            "lambda",
            "trace_group",
            "vi_reference_hit",
            "waterfall_hit",
            "osvi_hit",
            "geo_waterfall_hit",
            "dd_waterfall_hit",
        ],
    )
    _write_csv(
        derived_dir / "waterfall_residual_traces.csv",
        residual_trace_rows,
        ["lambda", "query", "residual", "policy_value_error", "value_error"],
    )
    (derived_dir / "waterfall_adaptation_events.json").write_text(
        json.dumps(adaptation_events, indent=2) + "\n",
        encoding="utf-8",
    )
    (derived_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = _parse_args()
    print(json.dumps(extract(args.spec), indent=2))


if __name__ == "__main__":
    main()
