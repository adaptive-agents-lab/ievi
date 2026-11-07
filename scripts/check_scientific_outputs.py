"""Independently compare full recomputed final-study tables and archived input arrays."""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_planning import find_verified
from scripts.compare_results import compare_pair, json_safe
from scripts.reproduction_manifest import load_tasks, sha256, utc_now, write_json

REFERENCE = ROOT / "methods/planning/results/benchmarks"


def compare_arrays(actual, reference):
    rows = []
    with np.load(actual, allow_pickle=False) as left, np.load(reference, allow_pickle=False) as right:
        keys_equal = set(left.files) == set(right.files)
        for key in sorted(set(left.files) | set(right.files)):
            if key not in left.files or key not in right.files:
                rows.append({"array": key, "passed": False, "error": "missing array"})
                continue
            a, b = left[key], right[key]
            shape_equal, dtype_equal = a.shape == b.shape, a.dtype == b.dtype
            # These archives are deterministic inputs, including their float64
            # surrogate matrices and targets. Require elementwise equality.
            passed = shape_equal and dtype_equal and np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else shape_equal and dtype_equal and np.array_equal(a, b)
            maximum = float(np.max(np.abs(a - b))) if shape_equal and a.dtype.kind in "fc" and a.size else None
            rows.append({"array": key, "passed": bool(passed), "actual_shape": list(a.shape), "reference_shape": list(b.shape),
                         "actual_dtype": str(a.dtype), "reference_dtype": str(b.dtype), "maximum_absolute_difference": maximum})
        surrogates = left["surrogate_models"] if "surrogate_models" in left.files else None
        models = int(np.prod(surrogates.shape[:-2])) if surrogates is not None else None
        dimensions = {"surrogate_models": models, "seeds": len(left["seeds"]), "horizons": len(left["H_grid"]), "schedules": len(left["schedules"])}
    return {"passed": keys_equal and all(row["passed"] for row in rows) and dimensions == {"surrogate_models": 420, "seeds": 30, "horizons": 7, "schedules": 2},
            "actual": str(actual), "reference": str(reference), "actual_sha256": sha256(actual), "reference_sha256": sha256(reference),
            "keys_equal": keys_equal, "dimensions": dimensions, "arrays": rows, "equality": "exact dtype, shape and element values; NPZ container bytes need not match"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def timing_invariants(rows, calibration, selection):
    methods = {"VI", "IEVI_VI_Ks4", "IEVI_VI_Ks8", "IEVI_VI_Ks16", "IEVI_VI_Ks32"}
    adaptive = methods - {"VI"}
    candidates = {128, 256, 512, 1024, 2048, 4096}
    keys = [(int(row["grid_side"]), row["method"], int(row["repetition"])) for row in rows]
    expected = {(100, method, repetition) for method in methods for repetition in range(10)}
    pilot_keys = [(row["method"], float(row["h"])) for row in calibration]
    expected_pilot = {(method, float(h)) for method in adaptive for h in candidates}
    eligible = [h for h in sorted(candidates) if all(
        any(row["method"] == method and float(row["h"]) == h and row["converged"] == "True" and int(row["outer_iterations"]) <= 6
            for row in calibration) for method in adaptive)]
    return {"measurements_50": len(rows) == 50,
            "exact_50_unique_method_repetition_keys": len(keys) == len(set(keys)) == 50 and set(keys) == expected,
            "grid100_state10000_every_measurement": all(int(row["grid_side"]) == 100 and int(row["state_count"]) == 10000 for row in rows),
            "all_converged": all(row["converged"] == "True" for row in rows),
            "all_true_residuals_within_target": all(float(row["final_relative_true_residual"]) <= 1e-5 for row in rows),
            "all_adaptive_inner_solvers_converged": all(row["inner_solver_converged"] == "True" for row in rows if row["method"] in adaptive),
            "method_order_per_repetition_complete": all({int(row["method_order_position"]) for row in rows if int(row["repetition"]) == rep} == set(range(5)) for rep in range(10)),
            "order_seed_per_repetition_exact": all(int(row["timing_order_seed"]) == 21260729 + int(row["repetition"]) for row in rows),
            "calibration_candidates_24": len(calibration) == 24,
            "exact_24_unique_pilot_keys": len(pilot_keys) == len(set(pilot_keys)) == 24 and set(pilot_keys) == expected_pilot,
            "pilot_grid24_state576_every_case": all(int(row["grid_side"]) == 24 and int(row["state_count"]) == 576 for row in calibration),
            "selected_h_1024": selection["selected_h"] == 1024,
            "pilot_selection_rule_recomputed": bool(eligible) and min(eligible) == selection["selected_h"],
            "selected_h_used_every_measurement": all(float(row["selected_h_from_calibration"]) == selection["selected_h"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--study", choices=["control", "horizon", "timing", "all"], default="all")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new report filename; existing reports are preserved")
    tasks = load_tasks(ROOT)
    selected = ["control", "horizon", "timing"] if args.study == "all" else [args.study]
    report = {"schema_version": 1, "checked_at": utc_now(), "checker_sha256": sha256(__file__), "argv": sys.argv, "studies": {}}
    for study in selected:
        path, record = find_verified(tasks, args.run_root.resolve(), "final-" + study)
        source = path.parent
        comparisons = []
        invariants = {}
        if study == "control":
            for name in ["control_accuracy_summary.csv", "control_accuracy_trajectories.csv", "control_accuracy_ievi_attempts.csv"]:
                comparisons.append(compare_pair(source / "control_accuracy/raw" / name, REFERENCE / "control_accuracy/raw" / name,
                                                {"mode": "csv", "ignore_columns": ["wall_seconds_full_run"]}))
            summary = read_csv(source / "control_accuracy/raw/control_accuracy_summary.csv")
            invariants = {"summary_rows_95": len(summary) == 95, "lambda_settings_19": len({row["lambda"] for row in summary}) == 19,
                          "all_completed_budget": all(row["status"] == "completed_budget" for row in summary)}
        elif study == "horizon":
            arrays = compare_arrays(source / "horizon_scaling/data/raw/experiment_inputs.npz", REFERENCE / "horizon_scaling/data/raw/experiment_inputs.npz")
            comparisons.append(arrays)
            for name, ignore in [("raw/records.csv", ["wall_time_median_seconds", "wall_time_mad_seconds", "timing_batch_repeats"]),
                                 ("derived/summary_by_H.csv", ["median_wall_time_seconds", "q25_wall_time_seconds", "q75_wall_time_seconds"]),
                                 ("derived/slope_summary.csv", [])]:
                spec = {"mode": "csv", "ignore_columns": ignore}
                if name == "derived/slope_summary.csv":
                    spec["columns"] = list(read_csv(REFERENCE / "horizon_scaling/data" / name)[0])
                comparisons.append(compare_pair(source / "horizon_scaling/data" / name, REFERENCE / "horizon_scaling/data" / name, spec))
            records = read_csv(source / "horizon_scaling/data/raw/records.csv")
            timings = read_csv(source / "horizon_scaling/data/raw/timing_repetitions.csv")
            slopes = read_csv(source / "horizon_scaling/data/derived/slope_summary.csv")
            groups = {}
            for row in records:
                key = (row["schedule"], row["H"], row["algorithm"])
                groups[key] = groups.get(key, 0) + 1
            invariants = {"records_1680": len(records) == 1680, "seeds_30": len({row["seed"] for row in records}) == 30,
                          "conditions_56_each_30": len(groups) == 56 and all(n == 30 for n in groups.values()),
                          "all_hits": all(row["hit"] == "True" for row in records),
                          "all_stable_tail_errors_within_target": all(float(row["error_after_stable_tail"]) <= .001 * (1 + 1e-10) for row in records),
                          "supporting_timing_repetitions_8400": len(timings) == 8400,
                          "supporting_timing_positive_finite": all(math.isfinite(float(row["seconds"])) and float(row["seconds"]) > 0 for row in timings),
                          "additional_slope_status_available": all(row["status"] == "available" for row in slopes),
                          "additional_slope_bootstrap_resamples_4000": all(int(row["bootstrap_requested_resamples"]) == 4000 and int(row["bootstrap_valid_resamples"]) == 4000 for row in slopes)}
        else:
            for pair in tasks["planning-timing"]["comparator"]["pairs"]:
                comparisons.append(compare_pair(source / pair["actual"], ROOT / pair["reference"], pair))
            rows = read_csv(source / "wallclock/data/raw/benchmark_runs.csv")
            calibration = read_csv(source / "wallclock/data/raw/calibration_runs.csv")
            selection = json.loads((source / "wallclock/data/derived/calibration_selection.json").read_text())
            invariants = timing_invariants(rows, calibration, selection)
        report["studies"][study] = {"passed": all(item["passed"] for item in comparisons) and all(invariants.values()),
                                   "execution_record": str(path), "execution_record_sha256": sha256(path), "identity_sha256": record["identity_sha256"],
                                   "comparisons": comparisons, "invariants": invariants}
    report["passed"] = all(study["passed"] for study in report["studies"].values())
    write_json(args.output, json_safe(report))
    print(json.dumps({"passed": report["passed"], "report": str(args.output), "studies": {name: {"passed": study["passed"], "invariants": study["invariants"]} for name, study in report["studies"].items()}}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
