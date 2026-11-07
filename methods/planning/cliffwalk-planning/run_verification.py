"""Bounded planning stable-hit comparison. Reads only bundled frozen source/config/evidence.

Run: python run_verification.py --output results
An output directory with existing files is rejected. Existing measurement files are protected.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runtime"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from besp.algorithms.jax import deterministic as alg
from besp.environments.jax_native import generate_modified_cliffwalk_jax


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stable_hit(queries, errors, threshold):
    # Strict hit criterion: error < threshold.
    last_failure = -1
    for i, error in enumerate(errors):
        if not np.isfinite(error) or error >= threshold:
            last_failure = i
    return int(queries[last_failure + 1]) if last_failure + 1 < len(queries) else None


def first_hit(queries, errors, threshold):
    return next((int(q) for q, e in zip(queries, errors, strict=True)
                 if np.isfinite(e) and e < threshold), None)


class Recorder:
    """Observe actual algorithm calls and inner-solver returns, without feedback."""
    def __init__(self, name, lam, model, handle):
        self.name, self.lam, self.handle = name, lam, handle
        models = [model] + ([model.isp] if hasattr(model, "isp") else [])
        self.true_ids = {id(m.p) for m in models}
        self.true_queries = 0
        self.query_kinds = Counter()
        self.residual_output_counts = []
        self.inner_stops = Counter()
        self.inner_iterations = Counter()
        self.inner_max_residual = 0.0
        self.inner_max_value_error_bound = 0.0
        self.inner_nonfinite = 0
        self.inner_solve_count = 0
        self.policy_improvement_steps = 0

    def __call__(self, event, local):
        if event in ("kernel", "true_correction"):
            if id(local["p"]) in self.true_ids:
                self.true_queries += 1
                self.query_kinds[event] += 1
                if self.name == "aisp" and event == "kernel":
                    self.residual_output_counts.append(self.true_queries)
            return
        solver = local["self"]
        if "r_tilde_arr" in local:
            rewards = np.asarray(local["r_tilde_arr"])
            discount = float(local["gamma_tilde"])
            model = np.asarray(solver.p_hat)
            fixed_steps = None
        else:
            rewards = np.asarray(local["r_arr"])
            discount = float(local["gamma"])
            model = np.asarray(local["p_arr"])
            fixed_steps = local.get("fixed_steps")
        value = np.asarray(local["v"])
        q = rewards + discount * np.einsum("sak,k->sa", model, value)
        residual = float(np.max(np.abs(np.max(q, axis=1) - value)))
        bound = residual / (1.0 - discount)
        reason = "policy_stable" if event == "inner_policy_stable" else (
            "fixed_iteration_count" if fixed_steps is not None else "iteration_cap")
        count = int(local["_"]) + 1
        self.inner_solve_count += 1
        self.policy_improvement_steps += count
        self.inner_stops[reason] += 1
        self.inner_iterations[str(count)] += 1
        finite = bool(np.isfinite(residual) and np.isfinite(bound))
        if finite:
            self.inner_max_residual = max(self.inner_max_residual, residual)
            self.inner_max_value_error_bound = max(self.inner_max_value_error_bound, bound)
        else:
            self.inner_nonfinite += 1
        self.handle.write(json.dumps({
            "algorithm": self.name, "lambda": self.lam,
            "solve_index": self.inner_solve_count,
            "actual_true_queries_at_return": self.true_queries,
            "policy_iteration_steps": count, "stop_reason": reason,
            "inner_discount": discount,
            "observed_bellman_residual_inf": residual if finite else None,
            "observed_value_error_bound": bound if finite else None,
            "finite": finite,
            "diagnostic_only": True,
        }, allow_nan=False) + "\n")

    def summary(self):
        return {
            "actual_true_queries": self.true_queries,
            "actual_true_query_kinds": dict(self.query_kinds),
            "inner_solves": self.inner_solve_count,
            "inner_stop_counts": dict(self.inner_stops),
            "inner_iterations_histogram": dict(self.inner_iterations),
            "policy_improvement_steps": self.policy_improvement_steps,
            "inner_max_observed_bellman_residual_inf": self.inner_max_residual,
            "inner_max_observed_value_error_bound": self.inner_max_value_error_bound,
            "inner_nonfinite_diagnostics": self.inner_nonfinite,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to replace existing verification output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((HERE / "verification-config.json").read_text(encoding="utf-8"))
    frozen_manifest = json.loads((HERE / "runtime-manifest.json").read_text(encoding="utf-8"))
    for entry in frozen_manifest:
        actual = hashlib.sha256((HERE / entry["path"]).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise RuntimeError(f"Frozen source changed: {entry['path']}")
    archived = json.loads((HERE / config["reference_trace"]).read_text(encoding="utf-8"))
    archive_by_lambda = {float(row["lambda"]): row for row in archived}
    package_names = ["jax", "jaxlib", "numpy", "scipy", "matplotlib", "PyYAML"]
    versions = {name: importlib.metadata.version(name) for name in package_names}
    start = time.perf_counter()
    mdp = generate_modified_cliffwalk_jax(gamma=config["gamma"])
    p, rewards = np.asarray(mdp.P), np.asarray(mdp.R)
    target = alg.certify_optimal_control_target(mdp, epsilon_target=config["target_value_error_budget"])
    if not target["certificate"]["certified"]:
        raise RuntimeError("Reference target certification failed")
    vstar = np.asarray(target["value"])
    np.savez(out / "environment-and-target.npz", P=p, R=rewards, V_star=vstar,
             pi_star=np.asarray(target["policy"]), gamma=config["gamma"])
    write_json(out / "target-certificate.json", target["certificate"])
    eye = np.eye(p.shape[0])
    rows = np.arange(p.shape[0])
    policy_cache = {}

    def policy_error(policy):
        key = tuple(int(a) for a in policy)
        if key not in policy_cache:
            pi = np.asarray(key)
            pv = np.linalg.solve(eye - config["gamma"] * p[rows, pi, :], rewards[rows, pi])
            policy_cache[key] = float(np.sum(np.abs(pv - vstar)) / (np.sum(np.abs(vstar)) + 1e-12))
        return policy_cache[key]

    def form_trace(name, queries, values, policies, residuals=None):
        values = np.asarray(values)
        policies = np.asarray(policies, dtype=np.int32)
        errors = [policy_error(pi) for pi in policies]
        # These true-model diagnostic products are OFFLINE and excluded from solver counts.
        true_q = rewards[None, :, :] + config["gamma"] * np.einsum("sak,bk->bsa", p, values)
        greedy = np.argmax(true_q, axis=2)
        if residuals is None:
            residuals = np.max(np.abs(np.max(true_q, axis=2) - values), axis=1)
        return {
            "queries": [int(q) for q in queries],
            "residuals": [float(x) for x in residuals],
            "value_errors": (np.sum(np.abs(values - vstar), axis=1) / (np.sum(np.abs(vstar)) + 1e-12)).tolist(),
            "policy_value_errors": errors, "policy_trace": policies.tolist(),
            "recorded_policy_differs_from_true_greedy_at_value_count": int(np.sum(np.any(greedy != policies, axis=1))),
            "metric_mode": "policy_value_error", "value_norm": "L1",
            "value_normalized": True, "threshold": config["threshold"],
            "first_hit": first_hit(queries, errors, config["threshold"]),
            "stable_hit": stable_hit(queries, errors, config["threshold"]),
            "policy_definition": "true-model greedy before VI update" if name == "vi" else
                "returned surrogate policy after OS-VI update" if name == "osvi" else
                "returned policy of final correction solve in completed AISP outer update",
        }

    records, comparisons, diagnostics = [], [], []
    with (out / "inner-solver-diagnostics.jsonl").open("w", encoding="utf-8") as inner:
        # VI does not depend on lambda. Execute its unchanged environment once and reuse its trace.
        vi = alg.VI_Control(mdp)
        rec = Recorder("vi", None, vi, inner)
        alg.VERIFICATION_HOOK = rec
        vi_values, vi_policies = vi.run(config["max_queries"])
        alg.VERIFICATION_HOOK = None
        assert rec.true_queries == config["max_queries"]
        vi_trace = form_trace("vi", range(1, rec.true_queries + 1), vi_values, vi_policies)
        vi_diagnostics = rec.summary()
        for lam in config["lambdas"]:
            support = (mdp.P > 0).astype(jnp.float64)
            uniform = support / jnp.sum(support, axis=2, keepdims=True)
            phat = (1.0 - lam) * mdp.P + lam * uniform
            traces = {"vi": vi_trace}
            one_diagnostics = {"lambda": lam, "vi": {**vi_diagnostics, "shared_lambda_independent_baseline": True}}
            for name in ["osvi", "aisp"]:
                model = alg.MSVI_Control(mdp, phat, surrogate_policy_iteration_steps=config["osvi_policy_iteration_steps"]) if name == "osvi" else alg.AISP_Control(mdp, phat)
                rec = Recorder(name, lam, model, inner)
                alg.VERIFICATION_HOOK = rec
                if name == "osvi":
                    values, policies = model.run(config["max_queries"])
                    queries, residuals = list(range(1, config["max_queries"] + 1)), None
                else:
                    queries, residuals, values, policies = model.run_with_policy_trace(
                        target_epsilon=config["target_epsilon"], max_avps=config["max_queries"])
                    assert rec.residual_output_counts[1:] == queries, (
                        "Actual counted applications disagree with recorded outputs", lam,
                        rec.residual_output_counts, queries)
                alg.VERIFICATION_HOOK = None
                assert rec.true_queries == queries[-1]
                assert rec.true_queries <= config["max_queries"]
                traces[name] = form_trace(name, queries, values, policies, residuals)
                one_diagnostics[name] = rec.summary()
                if name == "aisp":
                    b0 = float(np.max(np.abs(rewards.max(axis=1))))
                    stop = config["target_epsilon"] * (1.0 - config["gamma"]) / (1.0 + config["gamma"]) * b0
                    one_diagnostics[name]["outer_stop_reason"] = "residual_target" if residuals[-1] <= stop else "insufficient_budget_for_complete_update"
                    one_diagnostics[name]["residual_target"] = stop
            record = {"lambda": lam, "seed": config["seed"],
                      "results": {name: trace["stable_hit"] for name, trace in traces.items()},
                      "traces": traces}
            records.append(record)
            diagnostics.append(one_diagnostics)
            old = archive_by_lambda[float(lam)]
            comparison = {"lambda": lam}
            for name in ["vi", "osvi", "aisp"]:
                trace, history = traces[name], old["traces"][name]
                old_hit = stable_hit(history["queries"], history["policy_value_errors"], config["threshold"])
                comparison[name + "_reference_stable_hit"] = old_hit
                comparison[name + "_verified_stable_hit"] = trace["stable_hit"]
                comparison[name + "_stable_hit_equal"] = trace["stable_hit"] == old_hit
                same_queries = trace["queries"] == history["queries"]
                comparison[name + "_query_indices_equal"] = same_queries
                comparison[name + "_actual_total_true_queries"] = one_diagnostics[name]["actual_true_queries"]
                comparison[name + "_reference_last_recorded_query"] = history["queries"][-1]
                comparison[name + "_policy_trace_equal"] = trace["policy_trace"] == history["policy_trace"]
                comparison[name + "_max_policy_error_abs_difference"] = (
                    float(np.max(np.abs(np.asarray(trace["policy_value_errors"]) - np.asarray(history["policy_value_errors"]))))
                    if same_queries else None)
            comparisons.append(comparison)
            write_json(out / "raw" / (f"lambda-{lam:g}.json"), record)
            print(json.dumps({"lambda": lam, "verified": record["results"],
                "stable_hits_equal": all(comparison[name + "_stable_hit_equal"] for name in ["vi", "osvi", "aisp"]),
                "actual_true_queries": {n: one_diagnostics[n]["actual_true_queries"] for n in ["vi", "osvi", "aisp"]}}), flush=True)
    write_json(out / "control_traces.json", records)
    write_json(out / "all-19-comparison.json", comparisons)
    with (out / "all-19-comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    write_json(out / "run-diagnostics.json", diagnostics)
    summary = {
        "configuration_count": len(records), "algorithm_grid_points_compared": len(records) * 3,
        "all_stable_hits_equal_to_reference": all(row[name + "_stable_hit_equal"] for row in comparisons for name in ["vi", "osvi", "aisp"]),
        "all_query_indices_equal_to_reference": all(row[name + "_query_indices_equal"] for row in comparisons for name in ["vi", "osvi", "aisp"]),
        "all_policy_traces_equal_to_reference": all(row[name + "_policy_trace_equal"] for row in comparisons for name in ["vi", "osvi", "aisp"]),
        "aisp_inner_iteration_cap_hits": sum(row["aisp"]["inner_stop_counts"].get("iteration_cap", 0) for row in diagnostics),
        "aisp_inner_solves": sum(row["aisp"]["inner_solves"] for row in diagnostics),
        "osvi_inner_solves": sum(row["osvi"]["inner_solves"] for row in diagnostics),
        "aisp_max_observed_inner_residual_inf": max(row["aisp"]["inner_max_observed_bellman_residual_inf"] for row in diagnostics),
        "aisp_max_observed_inner_value_error_bound": max(row["aisp"]["inner_max_observed_value_error_bound"] for row in diagnostics),
        "inner_nonfinite_diagnostics": sum(row[name]["inner_nonfinite_diagnostics"] for row in diagnostics for name in ["osvi", "aisp"]),
        "reference_executed_source_identity_proven": False,
        "bundled_solver_run_identity_proven": True,
        "reference_target_certificate": target["certificate"],
        "offline_policy_evaluation_unique_policies": len(policy_cache),
        "elapsed_seconds": time.perf_counter() - start,
        "platform": platform.platform(), "python": sys.version, "packages": versions,
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "source_change": "bounded preflight guard plus observational hooks; no change to accepted-update formula, policy extraction, inner stopping, or incumbent behavior",
        "threshold_comparison": "strictly less than, matching the reference planning stable-hit renderer",
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({"complete": True, **summary}), flush=True)


if __name__ == "__main__":
    main()
