#!/usr/bin/env python3
"""Independent consistency checks for the value-iteration runtime comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from micro_simulator import DynamicsParameters, MicroSimulator, make_reward  # noqa: E402

config = json.load((ROOT / "config.json").open("r", encoding="utf-8"))
raw = pd.read_csv(ROOT / "data" / "raw" / "benchmark_runs.csv")
summary = pd.read_csv(ROOT / "data" / "derived" / "runtime_summary.csv")
discrepancy = pd.read_csv(ROOT / "data" / "raw" / "surrogate_discrepancy.csv")
selection = json.load((ROOT / "data" / "derived" / "calibration_selection.json").open("r", encoding="utf-8"))
environment = json.load((ROOT / "environment" / "environment.json").open("r", encoding="utf-8"))

checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str) -> None:
    checks.append((name, bool(condition), detail))


expected_files = sum(int(v) for v in config["benchmark_repetitions"].values())
expected_rows = expected_files * (1 + len(config["surrogate_microsteps"]))
check("raw row count", len(raw) == expected_rows, f"observed={len(raw)}, expected={expected_rows}")
check("all runs converged", bool(raw.converged.all()), f"failures={int((~raw.converged).sum())}")
target = float(config["target_relative_true_bellman_residual"])
check(
    "all final true residuals meet target",
    bool((raw.final_relative_true_residual <= target * 1.001).all()),
    f"max={raw.final_relative_true_residual.max():.6g}, target={target:.6g}",
)
check("no GMRES method rows", not raw.method.str.contains("GMRES", case=False).any(), "methods are VI and IEVI_VI only")
check("all IE-VI inner solves converged", bool(raw[raw.method.str.startswith("IEVI")].inner_solver_converged.all()), "checked every timed IE-VI run")

# Source-code check: algorithms and runners must not import or call GMRES.
executable_files = [ROOT / "code" / "algorithms.py", ROOT / "code" / "prepare_experiment.py", ROOT / "code" / "run_one_benchmark.py"]
code_text = "\n".join(path.read_text(encoding="utf-8") for path in executable_files)
check("no GMRES in executable source", "gmres(" not in code_text.lower() and "import gmres" not in code_text.lower(), "algorithm and runner source contain no GMRES call or import")

true_steps = int(config["true_microsteps"])
check(
    "true microstep accounting",
    bool((raw.true_microsteps == raw.true_operator_calls * true_steps).all()),
    "true_microsteps == true_calls × 128",
)
ie_rows = raw[raw.method.str.startswith("IEVI")].copy()
ie_steps = ie_rows.method.str.extract(r"Ks(\d+)")[0].astype(int)
check(
    "surrogate microstep accounting",
    bool((ie_rows.surrogate_microsteps.to_numpy() == ie_rows.surrogate_operator_calls.to_numpy() * ie_steps.to_numpy()).all()),
    "surrogate_microsteps == surrogate_calls × K_s",
)
check(
    "VI surrogate counts are zero",
    bool((raw[raw.method == "VI"].surrogate_operator_calls == 0).all()),
    "ordinary VI never calls the surrogate",
)

# Calibration rule.
calibration = pd.read_csv(ROOT / "data" / "raw" / "calibration_runs.csv")
selected_h = float(selection["selected_h"])
selected_rows = calibration[calibration.h == selected_h]
rule = int(config["calibration_max_outer_rounds_rule"])
check("selected h present", len(selected_rows) == len(config["surrogate_microsteps"]), f"selected_h={selected_h:g}")
check(
    "selected h satisfies outer-round rule",
    bool(selected_rows.converged.all() and selected_rows.outer_iterations.max() <= rule),
    f"max_outer={selected_rows.outer_iterations.max()}, rule={rule}",
)
smaller = calibration[calibration.h < selected_h].groupby("h").outer_iterations.max()
check(
    "selected h is smallest satisfying rule",
    bool((smaller > rule).all()),
    f"smaller-h maximum outer rounds={smaller.to_dict()}",
)

# Fidelity monotonicity.
for n, group in discrepancy.groupby("grid_side"):
    ordered = group.sort_values("surrogate_microsteps_per_call")
    values = ordered.max_probe_sup_discrepancy.to_numpy()
    check(
        f"probe discrepancy decreases with K_s at n={n}",
        bool(np.all(np.diff(values) < 0.0)),
        f"values={values.tolist()}",
    )

# Operator validity and saved-solution residuals.
dynamics = config["dynamics"]
params = DynamicsParameters(
    float(dynamics["diffusion_rate"]),
    float(dynamics["advection_amplitude"]),
    float(dynamics["secondary_amplitude"]),
    int(dynamics["phase_seed"]),
)
gamma = float(config["gamma"])
for n in [int(x) for x in config["benchmark_grid_sides"]]:
    true_operator = MicroSimulator(n, true_steps, params)
    operators = {"true": true_operator}
    operators.update({str(k): MicroSimulator(n, int(k), params) for k in config["surrogate_microsteps"]})
    one = np.ones(n * n)
    for label, operator in operators.items():
        constant_error = float(np.max(np.abs(operator.apply(one, count=False) - one)))
        check(f"constant preservation n={n}, op={label}", constant_error < 2e-12, f"error={constant_error:.3e}")
        check(
            f"microstep stochasticity n={n}, op={label}",
            operator.maximum_microstep_exit_probability <= 1.0 + 1e-12,
            f"max_exit_probability={operator.maximum_microstep_exit_probability:.6g}",
        )
    solutions = np.load(ROOT / "data" / "raw" / f"solutions_n{n}_rep0.npz")
    reward = make_reward(n)
    initial = float(np.max(np.abs(reward)))
    for method in solutions.files:
        value = solutions[method]
        residual = float(np.max(np.abs(reward + gamma * true_operator.apply(value, count=False) - value))) / initial
        check(f"saved solution residual n={n}, method={method}", residual <= target * 1.001, f"relative_residual={residual:.6g}")
    reference = np.load(ROOT / "data" / "raw" / f"reference_n{n}.npz")["value"]
    reference_residual = float(np.max(np.abs(reward + gamma * true_operator.apply(reference, count=False) - reference))) / initial
    check(f"reference residual n={n}", reference_residual <= float(config["reference_relative_true_bellman_residual"]) * 1.001, f"relative_residual={reference_residual:.3e}")

# Verify measurements at 10,000 states.
n_max = max(int(x) for x in config["benchmark_grid_sides"])
head = summary[summary.grid_side == n_max].set_index("method")
vi_time = float(head.loc["VI", "median_runtime_seconds"])
vi_micro = float(head.loc["VI", "median_total_microsteps"])
for ks in (4, 8, 16, 32):
    method = f"IEVI_VI_Ks{ks}"
    runtime = float(head.loc[method, "median_runtime_seconds"])
    micro = float(head.loc[method, "median_total_microsteps"])
    check(f"{method} faster than VI at n={n_max}", runtime < vi_time, f"{runtime:.6f} < {vi_time:.6f}")
    check(f"{method} uses fewer total microsteps than VI", micro < vi_micro, f"{micro:.0f} < {vi_micro:.0f}")

# Runtime and work should worsen monotonically as surrogate temporal resolution increases.
ie_head = head.loc[[f"IEVI_VI_Ks{k}" for k in (4, 8, 16, 32)]]
check(
    "IE-VI runtime increases with surrogate resolution",
    bool(np.all(np.diff(ie_head.median_runtime_seconds.to_numpy()) > 0.0)),
    f"times={ie_head.median_runtime_seconds.to_list()}",
)
check(
    "IE-VI total microsteps increase with surrogate resolution",
    bool(np.all(np.diff(ie_head.median_total_microsteps.to_numpy()) > 0.0)),
    f"microsteps={ie_head.median_total_microsteps.to_list()}",
)

# Thread environment.
for variable, expected_value in config["threading"].items():
    observed = environment["thread_environment"].get(variable)
    check(f"single-thread setting {variable}", observed == expected_value, f"observed={observed}")

passed = sum(1 for _, ok, _ in checks if ok)
failed = len(checks) - passed
lines = [
    "# Validation report",
    "",
    f"- Checks: {len(checks)}",
    f"- Passed: {passed}",
    f"- Failed: {failed}",
    "",
]
for name, ok, detail in checks:
    lines.append(f"- [{'PASS' if ok else 'FAIL'}] **{name}** — {detail}")
(ROOT / "validation").mkdir(parents=True, exist_ok=True)
(ROOT / "validation" / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Validation: {passed}/{len(checks)} passed")
if failed:
    raise SystemExit(1)
