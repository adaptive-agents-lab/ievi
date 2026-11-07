"""Validate frozen-source deltas and the completed bounded run without rerunning solvers."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"


class RemoveObservations(ast.NodeTransformer):
    def visit_If(self, node):
        if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == "VERIFICATION_HOOK":
            return None
        if ast.unparse(node.test) == "true_term is None and VERIFICATION_HOOK is not None":
            return None
        if ast.unparse(node.test) == "total_queries + J + 1 > max_avps":
            node.test = ast.parse("total_queries >= max_avps", mode="eval").body
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == "VERIFICATION_HOOK" for t in node.targets):
            return None
        return self.generic_visit(node)


def main():
    original = ast.parse((HERE / "reference_source/besp/algorithms/jax/deterministic.py").read_text(encoding="utf-8"))
    instrumented = ast.parse((HERE / "runtime/besp/algorithms/jax/deterministic.py").read_text(encoding="utf-8"))
    stripped = RemoveObservations().visit(instrumented)
    ast_equal = ast.dump(original, include_attributes=False) == ast.dump(stripped, include_attributes=False)
    assert ast_equal, "Changes beyond observational hooks and budget preflight detected"
    config = json.loads((HERE / "verification-config.json").read_text(encoding="utf-8"))
    traces = json.loads((OUT / "control_traces.json").read_text(encoding="utf-8"))
    comparisons = json.loads((OUT / "all-19-comparison.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((OUT / "run-diagnostics.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert len(traces) == len(comparisons) == len(diagnostics) == 19
    assert [x["lambda"] for x in traces] == config["lambdas"]
    arrays = np.load(OUT / "environment-and-target.npz", allow_pickle=False)
    p, r, vstar, gamma = arrays["P"], arrays["R"], arrays["V_star"], float(arrays["gamma"])
    assert p.shape == (36, 4, 36) and r.shape == (36, 4)
    assert np.max(np.abs(p.sum(axis=2) - 1)) < 1e-14
    assert p.min() >= 0
    residual = np.max(np.abs(np.max(r + gamma * np.einsum("sak,k->sa", p, vstar), axis=1) - vstar))
    assert residual <= (1 - gamma) * config["target_value_error_budget"] * 1.01
    equal_threshold = 0
    policy_cache = {}
    maximum_replay_delta = 0.0
    rows = np.arange(36)
    for entry, diag in zip(traces, diagnostics, strict=True):
        for name, trace in entry["traces"].items():
            q = trace["queries"]
            assert len(q) == len(trace["policy_trace"]) == len(trace["policy_value_errors"])
            assert all(x < y for x, y in zip(q, q[1:]))
            assert q[-1] == diag[name]["actual_true_queries"] <= config["max_queries"]
            if name == "aisp":
                assert q == [1 + 11 * n for n in range(1, len(q) + 1)]
                assert diag[name]["inner_solves"] == 10 * len(q)
                assert diag[name]["actual_true_query_kinds"] == {"kernel": 1 + len(q), "true_correction": 10 * len(q)}
            for policy, saved in zip(trace["policy_trace"], trace["policy_value_errors"], strict=True):
                key = tuple(policy)
                if key not in policy_cache:
                    pi = np.asarray(policy)
                    vp = np.linalg.solve(np.eye(36) - gamma * p[rows, pi, :], r[rows, pi])
                    policy_cache[key] = float(np.sum(np.abs(vp - vstar)) / (np.sum(np.abs(vstar)) + 1e-12))
                maximum_replay_delta = max(maximum_replay_delta, abs(policy_cache[key] - saved))
                equal_threshold += saved == config["threshold"]
    assert maximum_replay_delta < 1e-13
    report = {
        "pass": True,
        "source_ast_equivalent_after_removing_only_declared_hooks_and_budget_preflight": ast_equal,
        "configuration_count": len(traces),
        "algorithm_grid_points": 3 * len(traces),
        "actual_query_counters_match_every_recorded_trajectory": True,
        "all_actual_run_counts_within_1000": True,
        "offline_policy_metric_replay_max_abs_difference": maximum_replay_delta,
        "exact_threshold_equalities": equal_threshold,
        "target_bellman_residual_independent_numpy_check": float(residual),
        "target_value_error_bound": float(residual / (1 - gamma)),
        "all_stable_hits_equal_to_reference": summary["all_stable_hits_equal_to_reference"],
        "all_query_indices_equal_to_reference": summary["all_query_indices_equal_to_reference"],
        "all_policy_traces_equal_to_reference": summary["all_policy_traces_equal_to_reference"],
        "aisp_inner_cap_hits": summary["aisp_inner_iteration_cap_hits"],
        "reference_solver_identity_remains_unproven": True,
    }
    (OUT / "verification-checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
