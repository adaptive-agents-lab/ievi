"""Reproduce Cliffwalk tuning selection, held-out results and figure data."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CW = DATA / "results/cliffwalk"
METHODS = ("q", "os", "ie")
DISPLAY_METHODS = METHODS
CELL = "pol|all|L1"
SELECTION_KEY = CELL + "|su|per_lam|{method}|0.8"
NOMINAL_SAMPLES = np.arange(0, 80001, 2000)
TUNING_SEEDS = np.arange(20, 30)
HELDOUT_SEEDS = np.arange(250, 310)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _new_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Output must be a new directory: {path}")
    path.mkdir(parents=True)
    return path


def _experiment(runtime=None):
    """Load the numerical driver without importing JAX or the solver."""
    os.environ["CLIFFWALK_AXIS"] = "nominal"
    source = (Path(runtime) if runtime else HERE / "implementation") / "experiment.py"
    spec = importlib.util.spec_from_file_location("_cliffwalk_experiment", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_settings(data_root=None):
    data_root = Path(data_root) if data_root else DATA
    selected = _json(data_root / "results/cliffwalk/selection.json")["settings"]
    return {method: selected[SELECTION_KEY.format(method=method)] for method in METHODS}


def raw_name(setting):
    s = setting
    return f"{s['method']}_{s['jobfam']}_l{s['lam']}_J{s['J']}_B{s['B']}_h0.0.npz"


def selected_curve(path, setting):
    with np.load(path, allow_pickle=False) as raw:
        pairs = list(zip(raw["scheds"].tolist(), raw["hs"].tolist()))
        combo = pairs.index((setting["sched"], setting["h"]))
        cell = raw["cell_names"].tolist().index(CELL)
        np.testing.assert_array_equal(raw["seeds"], HELDOUT_SEEDS)
        curve = raw["cells"][:, combo, :, cell].astype(float)
        used = raw["used"].copy()
    if curve.shape != (41, 60) or not np.isfinite(curve).all():
        raise AssertionError(f"Invalid selected curve: {path}")
    return curve, used


def area40(errors):
    """Per-seed normalized trapezoidal area on the nominal 0..40k axis."""
    return np.trapezoid(np.asarray(errors)[:21], NOMINAL_SAMPLES[:21], axis=0) / 40000


def load_panel_data(data_root=None):
    """Return raw curves and immutable reference bands for the renderer.

    errors (3,41,60), means (3,41), intervals (2,3,41), ordered
    q/os/ie. Statistics are in percent before the display log transform.
    """
    data_root = Path(data_root) if data_root else DATA
    settings = selected_settings(data_root)
    records, curves, actual = [], [], []
    for method in METHODS:
        path = data_root / "results/cliffwalk/heldout" / raw_name(settings[method])
        curve, used = selected_curve(path, settings[method])
        curves.append(curve)
        actual.append(used)
        records.append({"method": method, "source": path.relative_to(HERE).as_posix()
                        if path.is_relative_to(HERE) else str(path),
                        "sha256": sha256(path), "setting": settings[method],
                        "area40": float(area40(curve).mean()),
                        "final80": float(curve[-3:].mean()),
                        "mean_at_exactly_80000": float(curve[-1].mean()),
                        "actual_last_sample": int(used[-1])})
    errors = np.stack(curves)
    with np.load(data_root / "panel-data.npz", allow_pickle=False) as reference:
        np.testing.assert_array_equal(errors, reference["errors"])
        np.testing.assert_array_equal(reference["methods"], METHODS)
        intervals = reference["intervals"].copy()
        np.testing.assert_array_equal(errors.mean(-1), reference["means"])
    return {"errors": errors, "means": errors.mean(-1), "intervals": intervals,
            "nominal_samples": NOMINAL_SAMPLES.copy(), "seeds": HELDOUT_SEEDS.copy(),
            "actual_samples": np.stack(actual), "methods": np.asarray(METHODS),
            "source_data": {"metric": CELL, "normalization":
                "100 * sum_s |Vpi(s)-Vstar(s)| / sum_s |Vstar(s)|",
                "selection": "per_lam", "family": "support-uniform", "gamma": .99,
                "lambda": .8, "methods": records, "tuning_seeds": TUNING_SEEDS.tolist(),
                "test_seeds": HELDOUT_SEEDS.tolist(), "nominal_samples": NOMINAL_SAMPLES.tolist(),
                "n_bootstrap": 20000, "bootstrap_rng_seed": 20261106,
                "bootstrap_unit": "whole seed trajectory", "statistics_before_log_transform": True}}


def verify():
    """Check every immutable input hash, raw schema and seed range."""
    manifest = _json(HERE / "manifest.json")
    for row in manifest["files"]:
        path = HERE / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise AssertionError(f"Immutable input hash mismatch: {row['path']}")
    for name, digest in _json(DATA / "source-manifest.json").items():
        if sha256(HERE / "implementation" / name) != digest:
            raise AssertionError(f"Distributed numerical source hash mismatch: {name}")
    experiment = _experiment()
    expected = {experiment.fname("cw", "tune", j).name for j in experiment.jobs_for("cw", "tune")}
    files = sorted((CW / "tuning").glob("*.npz"))
    assert len(files) == 44 and {p.name for p in files} == expected
    nonfinite_files = 0
    for stage, seeds, count in (("tuning", TUNING_SEEDS, 44), ("heldout", HELDOUT_SEEDS, 3)):
        paths = sorted((CW / stage).glob("*.npz"))
        assert len(paths) == count
        for path in paths:
            with np.load(path, allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw["seeds"], seeds)
                metric_count = 16 if stage == "tuning" else 1
                assert raw["cells"].shape == (41, len(raw["scheds"]), len(seeds), metric_count)
                assert len(raw["cell_names"]) == metric_count
                if stage == "heldout":
                    assert len(raw["scheds"]) == 1 and raw["cell_names"].tolist() == [CELL]
                bad = int((~np.isfinite(raw["cells"])).sum())
                if stage == "tuning":
                    nonfinite_files += bool(bad)
                else:
                    assert bad == 0
    assert nonfinite_files == 0
    panel = load_panel_data()
    assert panel["actual_samples"][1, -1] == 79980
    return {"status": "verified", "tuning_files": 44, "heldout_files": 3,
            "nonfinite_tuning_files_retained": nonfinite_files, "selection_rows":
            len(_json(CW / "selection.json")["settings"]), "immutable_files": len(manifest["files"])}


def _selection_candidates():
    """Score the full candidate grid for the reported metric using tuning seeds only."""
    experiment = _experiment()
    experiment.OUTROOT = DATA / "results"
    reference_fname = experiment.fname
    experiment.fname = lambda env, stage, job: (CW / "tuning" / reference_fname(env, stage, job).name
                                         if stage == "tune" else reference_fname(env, stage, job))
    selected, rows = {}, []
    for method in METHODS:
        table, cells = experiment.tune_table("cw", method, "su", .8)
        keys = list(table)
        values = np.asarray([table[key][cells.index(CELL)] for key in keys])
        penalized = np.where(np.isfinite(values), values, np.inf)
        choice = int(penalized.argmin())
        sched, J, B, h = keys[choice]
        setting = {"method": method, "jobfam": "-" if method == "q" else "su",
                   "lam": 0.0 if method == "q" else .8, "sched": sched,
                   "J": J, "B": B, "h": h, "area40_tune": float(penalized[choice])}
        selected[SELECTION_KEY.format(method=method)] = setting
        order = np.argsort(penalized, kind="stable")
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        for index, ((sched, J, B, h), score, penalty) in enumerate(zip(keys, values, penalized)):
            name = experiment.fname("cw", "tune", (method, setting["jobfam"], setting["lam"], J, B, h)).name
            rows.append({"method": method, "source_file": name, "schedule": sched, "J": J,
                         "B": B, "h": h, "area40": float(score), "penalized_area40": float(penalty),
                         "nonfinite_penalized": not bool(np.isfinite(score)),
                         "rank": int(ranks[index]), "selected": index == choice})
    return {"cells": [CELL], "settings": selected}, rows


def select(output):
    """Reproduce the three reported settings and all 1,584 candidate scores."""
    output = _new_output(output)
    computed, rows = _selection_candidates()
    if computed != _json(CW / "selection.json"):
        raise AssertionError("Tuning-only selection differs from the reported settings")
    directory = output / "results/cliffwalk"
    directory.mkdir(parents=True)
    _write_json(directory / "selection.json", computed)
    np.savez_compressed(output / "tuning-candidate-scores.npz",
                        **{key: np.asarray([row[key] for row in rows]) for key in rows[0]})
    with (output / "tuning-ranking.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {"status": "exact_selection_match", "tuning_files": 44,
               "selection_rows": len(computed["settings"]), "candidate_rows": len(rows),
               "method_candidate_counts": {method: sum(row["method"] == method for row in rows) for method in METHODS},
               "nonfinite_candidates_retained": sum(row["nonfinite_penalized"] for row in rows),
               "selection_uses_heldout": False, "metric": CELL,
               "selection_sha256": sha256(directory / "selection.json")}
    _write_json(output / "selection-verification.json", receipt)
    return receipt


def _runtime(output):
    runtime = output / "runtime"
    runtime.mkdir()
    for path in (HERE / "implementation").glob("*.py"):
        shutil.copy2(path, runtime / path.name)
    reference = HERE.parents[1] / "methods/planning/implementation/src"
    if not (reference / "besp").is_dir():
        raise FileNotFoundError(f"Packaged reference source missing: {reference}")
    source = runtime / "environment.py"
    lines = source.read_text().splitlines(keepends=True)
    assert sum(line.startswith("SRC = ") for line in lines) == 1
    source.write_text("".join(f"SRC = {str(reference)!r}\n" if line.startswith("SRC = ") else line
                              for line in lines), encoding="utf-8")
    selected = runtime / "results/cliffwalk"
    selected.mkdir(parents=True)
    shutil.copy2(CW / "selection.json", selected / "selection.json")
    shutil.copy2(CW / "execution-groups.json", selected / "execution-groups.json")
    return runtime


def _project_heldout(path, setting):
    """Export only the reported configuration and metric after grouped computation."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as raw:
        pairs = list(zip(raw["scheds"].tolist(), raw["hs"].tolist()))
        combo = pairs.index((setting["sched"], setting["h"]))
        cell = raw["cell_names"].tolist().index(CELL)
        arrays = {"used": raw["used"].copy(), "cells": raw["cells"][:, combo:combo+1, :, cell:cell+1].copy(),
                  "cell_names": raw["cell_names"][cell:cell+1].copy(),
                  "scheds": raw["scheds"][combo:combo+1].copy(), "hs": raw["hs"][combo:combo+1].copy(),
                  "seeds": raw["seeds"].copy()}
    temporary = path.with_name(path.stem + ".projection.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _run_one(args):
    runtime, job = args
    runtime = Path(runtime)
    sys.path.insert(0, str(runtime))
    os.environ.update(CLIFFWALK_AXIS="nominal", JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    experiment = _experiment(runtime)
    key, seconds = experiment.run_job(("cw", "test", job))
    _project_heldout(experiment.fname("cw", "test", job), selected_settings()[job[0]])
    import jax
    import scipy
    return {"job": list(key), "seconds": seconds,
            "versions": {"python": sys.version, "numpy": np.__version__,
                         "jax": jax.__version__, "scipy": scipy.__version__},
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


def compare(output, methods=DISPLAY_METHODS):
    output = Path(output)
    rows = []
    for method in methods:
        setting = selected_settings()[method]
        baseline = CW / "heldout" / raw_name(setting)
        fresh = output / "runtime/results/cliffwalk/heldout" / raw_name(setting)
        expected, expected_used = selected_curve(baseline, setting)
        actual, actual_used = selected_curve(fresh, setting)
        np.testing.assert_array_equal(actual_used, expected_used)
        delta = np.abs(actual - expected)
        rows.append({"method": method, "array_equal": bool(np.array_equal(actual, expected)),
                     "max_absolute_error_percent": float(delta.max()),
                     "reference_area40": float(area40(expected).mean()),
                     "fresh_area40": float(area40(actual).mean()),
                     "reference_sha256": sha256(baseline), "fresh_sha256": sha256(fresh),
                     "allclose_rtol_1e-5_atol_1e-5": bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-5)),
                     "seeds": HELDOUT_SEEDS.tolist(), "setting": setting})
    receipt = {"status": "compared", "methods": rows,
               "all_array_equal": all(r["array_equal"] for r in rows),
               "all_within_tolerance": all(r["allclose_rtol_1e-5_atol_1e-5"] for r in rows)}
    _write_json(output / "comparison.json", receipt)
    return receipt


def run(output, methods=DISPLAY_METHODS, workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    output = _new_output(output)
    runtime = _runtime(output)
    experiment = _experiment(runtime)
    settings = selected_settings()
    names = {raw_name(settings[m]) for m in methods}
    jobs = [j for j in experiment.jobs_for("cw", "test") if experiment.fname("cw", "test", j).name in names]
    assert len(jobs) == len(set(methods))
    # Set thread limits before NumPy is imported in each spawned worker.
    os.environ.update(CLIFFWALK_AXIS="nominal", JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        receipts = list(pool.map(_run_one, [(str(runtime), job) for job in jobs]))
    _write_json(output / "execution.json", {"jobs": receipts, "methods": list(methods),
                "seeds": HELDOUT_SEEDS.tolist(), "budget": 80000, "evaluation_spacing": 2000,
                "selection_source_sha256": sha256(CW / "selection.json"),
                "reference_source": "methods/planning/implementation/src"})
    return compare(output, methods)


def render_data(output):
    output = _new_output(output)
    data = load_panel_data()
    np.savez_compressed(output / "panel-data.npz", **{k: v for k, v in data.items() if k != "source_data"})
    _write_json(output / "source_data.json", data["source_data"])
    return {"status": "exported", "output": str(output), "methods": list(METHODS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    for command in ("select", "run", "render-data", "compare"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path, required=True)
        if command in ("run", "compare"):
            p.add_argument("--methods", nargs="+", choices=METHODS, default=list(DISPLAY_METHODS))
        if command == "run":
            p.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "verify":
        receipt = verify()
    elif args.command == "select":
        receipt = select(args.output)
    elif args.command == "render-data":
        receipt = render_data(args.output)
    elif args.command == "compare":
        receipt = compare(args.output, args.methods)
    else:
        receipt = run(args.output, args.methods, args.workers)
    print(json.dumps(receipt, indent=2, allow_nan=False))
    if args.command in ("run", "compare") and not receipt["all_within_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
