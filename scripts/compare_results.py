"""Explicit file-pair comparison. Timing deviations are reported, not hidden."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import json
import math
import re
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path

try:
    from scripts.reproduction_manifest import sha256, write_json
    from scripts.reference_dataset import accessible, iter_files, safe_relative
except ModuleNotFoundError:  # Also support direct python scripts/compare_results.py.
    from reproduction_manifest import sha256, write_json
    from reference_dataset import accessible, iter_files, safe_relative


def equal_value(a, b, rtol, atol):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    try:
        x, y = float(a), float(b)
        if math.isnan(x) or math.isnan(y):
            return math.isnan(x) and math.isnan(y)
        return math.isclose(x, y, rel_tol=rtol, abs_tol=atol)
    except (ValueError, TypeError):
        return a == b


def discrete_field(name):
    """Study grids, outcomes, policies and accounting counters require equality."""
    return name in {"lambda", "lambda_value", "seed", "replication", "repetition", "H", "gamma", "delta_solve", "threshold", "schedule",
                    "attempt", "outer", "inner", "iteration", "direct_solves", "restorations", "incumbent_attempt", "h", "m", "gamma_h", "inner_vi_rtol", "surrogate_microsteps_per_call", "selected_h_from_calibration", "hit", "stable_hit", "first_hit", "hit_time", "queries", "policy_trace", "policy_steps", "sample_steps", "last_query", "lambda_values", "lambda_grid", "seed_values", "seed_grid", "row_seeds", "H_grid", "method_order", "state_count", "grid_side", "n", "replications",
                    "completed_outputs", "final_accepted", "final_rejected", "final_inner_failures", "restoration_events", "timing_order_seed",
                    "method_order_position", "selected_h", "n_per_policy_row", "offline_transition_samples"} or bool(
        re.search(r"(?:^N_|_N_P|_N_Phat|_(?:calls|applications|microsteps|iterations|repetitions|updates|count|query|output|attempts?|solves|restorations|sweeps|hits)$)", name))


def exact_value(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    try:
        x, y = Decimal(str(a)), Decimal(str(b))
        return (x.is_nan() and y.is_nan()) or x == y
    except (InvalidOperation, ValueError, TypeError):
        return a == b


def compare_pair(actual, reference, spec):
    actual, reference = accessible(actual), accessible(reference)
    mode = spec.get("mode", "exact")
    rtol, atol = spec.get("rtol", 1e-9), spec.get("atol", 1e-11)
    ignore = set(spec.get("ignore_columns", []))
    differences = []
    reported_variation = []
    ignored_provenance_keys = []
    if not actual.is_file() or not reference.is_file():
        return {"passed": False, "error": "Actual or reference file missing", "actual": str(actual), "reference": str(reference)}
    if mode == "exact":
        passed = sha256(actual) == sha256(reference)
    elif mode == "csv":
        with actual.open(newline="", encoding="utf-8-sig") as f:
            left = list(csv.DictReader(f))
        with reference.open(newline="", encoding="utf-8-sig") as f:
            right = list(csv.DictReader(f))
        keys = spec.get("key_columns")
        if keys:
            left.sort(key=lambda row: tuple(row[k] for k in keys))
            right.sort(key=lambda row: tuple(row[k] for k in keys))
        columns = spec.get("columns")
        if len(left) != len(right):
            differences.append({"row_count": [len(left), len(right)]})
        for index, (a, b) in enumerate(zip(left, right)):
            selected = set(columns) if columns else ((set(a) | set(b)) - ignore)
            for key in sorted(selected):
                exact = discrete_field(key) or key in spec.get("exact_columns", []) or key in (keys or [])
                equal = exact_value(a.get(key), b.get(key)) if exact else equal_value(a.get(key), b.get(key), rtol, atol)
                if key not in a or key not in b or not equal:
                    if len(differences) < 30:
                        differences.append({"row": index, "column": key, "actual": a.get(key), "reference": b.get(key)})
            for key in spec.get("report_columns", []):
                x, y = a.get(key), b.get(key)
                try:
                    ratio = float(x) / float(y) if float(y) else None
                except (ValueError, TypeError):
                    ratio = None
                reported_variation.append({"row": index, "column": key, "actual": x, "reference": y, "actual_to_reference_ratio": ratio})
        passed = not differences
    elif mode == "arrays":
        import numpy as np
        left_container, right_container = np.load(actual, allow_pickle=False), np.load(reference, allow_pickle=False)
        try:
            left = {k: left_container[k] for k in left_container.files} if hasattr(left_container, "files") else {"array": left_container}
            right = {k: right_container[k] for k in right_container.files} if hasattr(right_container, "files") else {"array": right_container}
            if set(left) != set(right):
                differences.append({"array_keys": [sorted(left), sorted(right)]})
            for key in sorted(set(left) & set(right)):
                a, b = left[key], right[key]
                exact = spec.get("exact_arrays", False) or a.dtype.kind not in "fc" or discrete_field(key)
                shape_equal = a.shape == b.shape and a.dtype == b.dtype
                equal = shape_equal and (np.array_equal(a, b, equal_nan=True) if exact and a.dtype.kind in "fc" else np.array_equal(a, b) if exact else np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
                if not equal:
                    differences.append({"array": key, "actual_shape": list(a.shape), "reference_shape": list(b.shape), "exact": exact})
            passed = not differences
        finally:
            if hasattr(left_container, "close"):
                left_container.close()
            if hasattr(right_container, "close"):
                right_container.close()
    elif mode == "json":
        left, right = json.loads(actual.read_text()), json.loads(reference.read_text())
        # Reconstructed targets retain the absolute destination used by their
        # own run. This provenance path is expected to differ between attempts.
        ignored_provenance_keys = ["target_manifest"]
        ignored = set(spec.get("ignore_keys", [])) | set(ignored_provenance_keys)
        def visit(a, b, path="", exact=False):
            if isinstance(a, dict) and isinstance(b, dict):
                for key in sorted((set(a) | set(b)) - ignored):
                    if key not in a or key not in b:
                        differences.append({"path": path + "/" + key, "error": "missing key"})
                    else:
                        visit(a[key], b[key], path + "/" + key, exact or discrete_field(key) or key in spec.get("exact_keys", []))
            elif isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    differences.append({"path": path, "lengths": [len(a), len(b)]})
                for index, (x, y) in enumerate(zip(a, b)):
                    visit(x, y, path + "/" + str(index), exact)
            elif not (exact_value(a, b) if exact or (type(a) is int or type(b) is int) else equal_value(a, b, rtol, atol)):
                if len(differences) < 30:
                    differences.append({"path": path, "actual": a, "reference": b})
        visit(left, right)
        passed = not differences
    else:
        raise ValueError(f"Unknown comparator mode: {mode}")
    return {"passed": passed, "mode": mode, "actual": str(actual), "reference": str(reference),
            "actual_sha256": sha256(actual), "reference_sha256": sha256(reference), "differences": differences,
            "specification": spec, "reported_variation": reported_variation,
            "exact_discrete_fields": True, "ignored_provenance_keys": ignored_provenance_keys}


def compare_task(task, root, output):
    spec = task.get("comparator")
    reference = task.get("scientific_checker", False)
    if (not spec or not spec.get("pairs")) and not reference:
        return {"status": "not_configured", "passed": None, "task_id": task["id"]}
    results = []
    for pair in (spec or {}).get("pairs", []):
        combined = {k: v for k, v in spec.items() if k != "pairs"}
        combined.update(pair)
        actual = Path(output) / pair["actual"]
        reference = Path(root) / pair["reference"]
        results.append(compare_pair(actual, reference, combined))
    if reference:
        results.append(compare_reference(task, root, output))
    passed = False if any(item.get("passed") is False for item in results) else (None if any(item.get("passed") is not True for item in results) else True)
    status = "comparison_pending" if passed is None else "compared"
    return json_safe({"status": status, "task_id": task["id"], "passed": passed, "comparisons": results})


def compare_reference(task, root, output):
    """Run scientific checks outside immutable attempts and retain limitations."""
    root, output = Path(root), Path(output)
    study = output / "study"
    report_root = root / "verification/reference-comparisons" / task["id"] / (output.name + "-" + uuid.uuid4().hex[:8])
    checker_path = root / "scripts/check_planning_reproduction.py"
    result = {"mode": "reference-scientific", "task_id": task["id"], "study_output": str(study),
              "report_root": str(report_root), "checker": str(checker_path), "passed": False}
    try:
        if not checker_path.is_file():
            raise ValueError("Reference scientific checker is missing")
        if not (study / "completed.json").is_file():
            raise ValueError("Reference completed.json is missing; no scientific pass can be established")
        completed = json.loads((study / "completed.json").read_text(encoding="utf-8"))
        if completed.get("completed") is not True or completed.get("prepare_only"):
            raise ValueError("Reference study is incomplete or preparation-only")
        if completed.get("id") != task["id"]:
            raise ValueError("Reference completed record belongs to a different task")
        checker, result["checker_sha256"] = load_reference_checker(checker_path)
        report = checker.check(study, report_root=report_root)
        report_path = report_root / "scientific-comparison.json"
        if not isinstance(report, dict) or not report_path.is_file():
            raise ValueError("Reference checker did not produce a scientific comparison report")
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        if json_safe(saved) != json_safe(report):
            raise ValueError("Reference saved report differs from the computed scientific result")
        if sha256(checker_path) != result["checker_sha256"]:
            raise ValueError("Reference checker changed during comparison")
        if report.get("comparison_pending"):
            result["passed"] = None
            result["comparison_pending"] = True
        elif type(report.get("passed")) is not bool:
            raise ValueError("Reference report has no explicit scientific pass/fail result")
        else:
            result["passed"] = report["passed"]
        result.update(report=report, report_path=str(report_path), report_sha256=sha256(report_path))
    except Exception as exc:
        result.update(passed=False, error=str(exc))
    return json_safe(result)


def load_reference_checker(path):
    """Compile the exact recorded source, avoiding stale imported-module caches."""
    source = Path(path).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    spec = importlib.util.spec_from_file_location("scripts._scientific_checker_" + digest[:12], path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, digest


def json_safe(value):
    """Keep invalid numerical evidence visible in strict JSON reports."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return [json_safe(x) for x in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


# These metadata fields record execution/platform facts; they are deliberately
# outside scientific equality. Missing scientific fields always remain failures.
PROVENANCE_KEYS = {"deterministic_target_manifest", "target_manifest", "output_root", "results_csv", "raw_results_json", "logs_dir", "offline_eval_dir", "control_traces_json", "pe_traces_json", "platform", "python_executable", "python_version", "created_at", "started_at", "finished_at", "generated_at", "checked_at"}


def runtime_field(name):
    return not discrete_field(name) and bool(re.search(r"(?:seconds|(?:^|_)wall(?:_|$)|(?:^|_)elapsed(?:_|$)|(?:^|_)runtime(?:_|$)|(?:^|_)duration(?:_|$)|(?:^|_)timing_batch_repeats$)", name))


def accepted_specs(task, reference, reference_files=None):
    """Use measured streams/arrays, grids and accounting; exclude runtime media."""
    reference = accessible(reference)
    specs = {pair["actual"]: {k: v for k, v in pair.items() if k != "reference"} for pair in task.get("comparator", {}).get("pairs", [])}
    # Accepted comparisons derive membership from the verified manifest. Direct
    # helper callers use the same long-path-safe, fail-closed enumeration as the
    # materialization verifier, rather than a glob that can shrink coverage.
    paths = ([reference / safe_relative(name) for name in sorted(reference_files)]
             if reference_files is not None else sorted(iter_files(reference)))
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Accepted reference file missing: {path}")
        name = path.relative_to(reference).as_posix()
        scientific = False
        if task["id"] == "planning-control":
            scientific = name.startswith("control_accuracy/raw/") and path.suffix == ".csv"
        elif task["id"] == "planning-horizon":
            scientific = name in {"horizon_scaling/data/raw/experiment_inputs.npz", "horizon_scaling/data/raw/records.csv", "horizon_scaling/data/derived/summary_by_H.csv", "horizon_scaling/data/derived/slope_summary.csv"}
        elif task["id"] == "planning-timing":
            scientific = name in {"wallclock/data/raw/reference_n100.npz", "wallclock/data/raw/solutions_n100_rep0.npz"}
        elif task["id"] == "restoration-common":
            scientific = path.suffix == ".csv"
        elif task["id"] == "two-state-stiffness":
            scientific = name.startswith("study/package/data/") and path.suffix == ".json"
        elif task["id"].startswith(("planning-", "restoration-", "selected-")):
            scientific = path.suffix in {".npy", ".npz"} or name.endswith(("raw-traces.json", "inner-diagnostics.json", "outer-diagnostics.json", "control_traces.json", "pe_traces.json", "raw/results.json", "results.csv"))
            # Practical summaries are science; selected baseline summary.json
            # only contains runtime/path bookkeeping and is not a metric table.
            scientific = scientific or name in {"practical/summary.json", "study/solver/summary.json"} and isinstance(json.loads(path.read_text()), list)
        if scientific and name not in specs:
            specs[name] = {"actual": name, "mode": "csv" if path.suffix == ".csv" else "arrays" if path.suffix in {".npz", ".npy"} else "json"}
    for name, spec in specs.items():
        # Timing residual/error estimates are continuous scientific outputs.
        # Preserve exact protocol and accounting fields while allowing the same
        # established numerical tolerance used for other continuous streams.
        spec["exact_columns"] = [key for key in spec.get("exact_columns", []) if key not in {"final_relative_true_residual", "sup_error_to_reference"}]
        spec.setdefault("rtol", 1e-9)
        spec.setdefault("atol", 1e-11)
        if spec["mode"] == "csv" and (reference / name).is_file():
            with (reference / name).open(newline="", encoding="utf-8-sig") as stream:
                headers = next(csv.reader(stream), [])
            timing = [key for key in headers if runtime_field(key)]
            spec["ignore_columns"] = sorted(set(spec.get("ignore_columns", [])) | set(timing))
            spec["report_columns"] = sorted(set(spec.get("report_columns", [])) | set(timing))
            # Continuous bootstrap summaries may contain added diagnostic
            # columns; the accepted file defines the supported scientific fields.
        elif spec["mode"] == "json":
            timing = set()
            def collect(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        if runtime_field(key):
                            timing.add(key)
                        collect(child)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)
            if (reference / name).is_file():
                collect(json.loads((reference / name).read_text()))
            spec["ignore_keys"] = sorted(set(spec.get("ignore_keys", [])) | PROVENANCE_KEYS | timing)
        elif spec["mode"] == "arrays":
            spec["exact_arrays"] = any(key in name for key in ("experiment_inputs", "environment-target", "environment-and-target", "target-arrays", "reference_n100", "replay/"))
    return list(specs.values())


def timing_measurement_invariants(rows, pilot, summary):
    """Timing may vary, but every measured value and derived sum stays valid."""
    import numpy as np
    def valid_values(records, names, positive=False):
        try:
            return bool(records) and all(math.isfinite(float(row[key])) and (float(row[key]) > 0 if positive else float(row[key]) >= 0) for row in records for key in names)
        except (ValueError, TypeError, KeyError):
            return False
    def equation(row, target, components):
        try:
            return math.isclose(float(row[target]), sum(float(row[key]) for key in components), rel_tol=1e-9, abs_tol=1e-11)
        except (ValueError, TypeError, KeyError):
            return False
    results = {
        "benchmark_runtime_solver_positive_finite": valid_values(rows, ["runtime_seconds", "solver_seconds"], True),
        "benchmark_setup_finite_nonnegative": valid_values(rows, ["true_setup_seconds", "surrogate_setup_seconds", "setup_plus_solver_seconds"]),
        "pilot_runtime_positive_finite": valid_values(pilot, ["runtime_seconds"], True),
        "summary_runtime_positive_finite": valid_values(summary, ["median_runtime_seconds", "runtime_q25", "runtime_q75"], True),
        "runtime_equals_solver": bool(rows) and all(equation(row, "runtime_seconds", ["solver_seconds"]) for row in rows),
        "setup_plus_solver_sum_consistent": bool(rows) and all(equation(row, "setup_plus_solver_seconds", ["true_setup_seconds", "surrogate_setup_seconds", "solver_seconds"]) for row in rows),
    }
    try:
        groups = {}
        for row in rows:
            groups.setdefault((row["grid_side"], row["method"]), []).append(float(row["runtime_seconds"]))
        keys = [(row["grid_side"], row["method"]) for row in summary]
        results["summary_has_all_unique_measurement_groups"] = len(keys) == len(set(keys)) and set(keys) == set(groups)
        results["summary_runtime_quantiles_recomputed"] = bool(summary) and all(
            math.isclose(float(row[key]), float(np.quantile(groups[(row["grid_side"], row["method"])], quantile)), rel_tol=1e-9, abs_tol=1e-11)
            for row in summary for key, quantile in (("runtime_q25", .25), ("median_runtime_seconds", .5), ("runtime_q75", .75)))
    except (ValueError, TypeError, KeyError):
        results["summary_has_all_unique_measurement_groups"] = False
        results["summary_runtime_quantiles_recomputed"] = False
    return results


def compare_accepted_task(task, root, output, dataset_path):
    """Compare a new strict verified run with accepted scientific streams."""
    from scripts.reference_dataset import load_dataset, resolve_attempt
    accepted = load_dataset(dataset_path)['tasks'][task['id']]
    execution, data_record = resolve_attempt(dataset_path, task["id"])
    reference, output = accessible(execution.parent), accessible(output)
    specs = accepted_specs(task, reference, reference_files=accepted['files'])
    results = [compare_pair(output / spec["actual"], reference / spec["actual"], spec) for spec in specs]
    invariants = {}
    if task["id"] == "planning-timing":
        from scripts.check_scientific_outputs import read_csv, timing_invariants
        rows = read_csv(Path(output) / "wallclock/data/raw/benchmark_runs.csv")
        pilot = read_csv(Path(output) / "wallclock/data/raw/calibration_runs.csv")
        selection = json.loads((Path(output) / "wallclock/data/derived/calibration_selection.json").read_text())
        invariants = timing_invariants(rows, pilot, selection)
        summary = read_csv(Path(output) / "wallclock/data/derived/runtime_summary.csv")
        invariants.update(timing_measurement_invariants(rows, pilot, summary))
    elif task["id"] == "planning-horizon":
        from scripts.check_scientific_outputs import read_csv
        timings = read_csv(Path(output) / "horizon_scaling/data/raw/timing_repetitions.csv")
        invariants["supporting_timing_repetitions_8400"] = len(timings) == 8400
        invariants["supporting_timing_positive_finite"] = all(math.isfinite(float(row["seconds"])) and float(row["seconds"]) > 0 for row in timings)
    return json_safe({"status": "compared", "task_id": task["id"], "passed": bool(results) and all(x["passed"] for x in results) and all(invariants.values()),
        "reference_role": "accepted-paper-reproduction", "dataset_manifest": str(dataset_path), "dataset_manifest_sha256": sha256(accessible(dataset_path)),
        "accepted_execution_sha256": sha256(accessible(execution)), "accepted_identity_sha256": data_record["identity_sha256"],
        "comparisons": results, "comparison_count": len(results), "invariants": invariants,
        "comparison_scope": "Exact discrete outcomes, seeds, grids, policies and accounting; tolerant continuous streams/arrays; timing measurements report variation and satisfy protocol invariants, never elapsed-time equality"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("actual", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--mode", choices=["exact", "csv", "json"], default="exact")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json_safe(compare_pair(args.actual, args.reference, {"mode": args.mode}))
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
