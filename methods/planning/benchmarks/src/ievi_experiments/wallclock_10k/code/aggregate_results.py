#!/usr/bin/env python3
"""Aggregate benchmark records and derive summary tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
config = json.load((ROOT / "config.json").open("r", encoding="utf-8"))
files = sorted((ROOT / "data" / "raw").glob("benchmark_n*_rep*.csv"))
expected = sum(int(v) for v in config["benchmark_repetitions"].values())
if len(files) != expected:
    raise RuntimeError(f"Expected {expected} benchmark files, found {len(files)}")
raw = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
raw.to_csv(ROOT / "data" / "raw" / "benchmark_runs.csv", index=False)

summary = raw.groupby(["grid_side", "state_count", "method"], as_index=False).agg(
    repetitions=("runtime_seconds", "count"),
    median_runtime_seconds=("runtime_seconds", "median"),
    runtime_q25=("runtime_seconds", lambda s: float(np.quantile(s, 0.25))),
    runtime_q75=("runtime_seconds", lambda s: float(np.quantile(s, 0.75))),
    all_converged=("converged", "all"),
    median_relative_residual=("final_relative_true_residual", "median"),
    median_sup_error=("sup_error_to_reference", "median"),
    median_true_calls=("true_operator_calls", "median"),
    median_true_microsteps=("true_microsteps", "median"),
    median_surrogate_calls=("surrogate_operator_calls", "median"),
    median_surrogate_microsteps=("surrogate_microsteps", "median"),
    median_total_microsteps=("true_microsteps", lambda s: 0.0),
    median_outer_iterations=("outer_iterations", "median"),
    median_inner_vi_iterations=("inner_vi_iterations", "median"),
)
# Use grouped medians for total microsteps.
total_micro = raw.assign(total_microsteps=raw.true_microsteps + raw.surrogate_microsteps).groupby(
    ["grid_side", "state_count", "method"], as_index=False
).agg(median_total_microsteps=("total_microsteps", "median"))
summary = summary.drop(columns=["median_total_microsteps"]).merge(
    total_micro, on=["grid_side", "state_count", "method"], how="left"
)
summary.to_csv(ROOT / "data" / "derived" / "runtime_summary.csv", index=False)

for n in [int(x) for x in config["benchmark_grid_sides"]]:
    subset = summary[summary.grid_side == n].copy()
    vi_time = float(subset.loc[subset.method == "VI", "median_runtime_seconds"].iloc[0])
    vi_micro = float(subset.loc[subset.method == "VI", "median_total_microsteps"].iloc[0])
    subset["runtime_speedup_over_VI"] = vi_time / subset["median_runtime_seconds"]
    subset["microstep_reduction_over_VI"] = vi_micro / subset["median_total_microsteps"]
    subset.to_csv(ROOT / "data" / "derived" / f"headline_n{n}.csv", index=False)

status = {
    "completed": True,
    "benchmark_file_count": len(files),
    "benchmark_row_count": int(len(raw)),
    "all_methods_converged": bool(raw["converged"].all()),
    "maximum_relative_true_residual": float(raw["final_relative_true_residual"].max()),
    "gmres_rows": int(raw["method"].str.contains("GMRES", case=False).sum()),
}
with (ROOT / "data" / "derived" / "aggregate_study-summary.json").open("w", encoding="utf-8") as handle:
    json.dump(status, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(status)
