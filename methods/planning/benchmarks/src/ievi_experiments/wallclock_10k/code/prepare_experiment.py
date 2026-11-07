#!/usr/bin/env python3
"""Prepare calibration, ordinary-VI references, diagnostics, and environment."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
import numba
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from algorithms import compute_reference_by_value_iteration, run_ievi  # noqa: E402
from micro_simulator import DynamicsParameters, MicroSimulator, make_reward  # noqa: E402


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_config() -> dict[str, Any]:
    with (ROOT / "config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parameters(config: dict[str, Any]) -> DynamicsParameters:
    values = config["dynamics"]
    return DynamicsParameters(
        diffusion_rate=float(values["diffusion_rate"]),
        advection_amplitude=float(values["advection_amplitude"]),
        secondary_amplitude=float(values["secondary_amplitude"]),
        phase_seed=int(values["phase_seed"]),
    )


def discrepancy_probes(
    true_operator: MicroSimulator,
    surrogate_operator: MicroSimulator,
    probe_count: int = 32,
) -> dict[str, float | int]:
    rng = np.random.default_rng(314159 + true_operator.grid_side)
    discrepancies: list[float] = []
    for _ in range(probe_count):
        probe = rng.choice(np.array([-1.0, 1.0]), size=true_operator.state_count)
        true_value = true_operator.apply(probe, count=False)
        surrogate_value = surrogate_operator.apply(probe, count=False)
        discrepancies.append(float(np.max(np.abs(true_value - surrogate_value))))
    return {
        "probe_count": probe_count,
        "mean_probe_sup_discrepancy": float(np.mean(discrepancies)),
        "median_probe_sup_discrepancy": float(np.median(discrepancies)),
        "max_probe_sup_discrepancy": float(np.max(discrepancies)),
    }


def collect_environment(config: dict[str, Any]) -> dict[str, Any]:
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
    except OSError:
        lscpu = ""
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "numba": numba.__version__,
        "hostname": platform.node(),
        "machine": platform.machine(),
        "thread_environment": {
            name: os.environ.get(name) for name in config["threading"]
        },
        "lscpu": lscpu,
    }


def main() -> None:
    config = load_config()
    params = parameters(config)
    gamma = float(config["gamma"])
    target = float(config["target_relative_true_bellman_residual"])
    inner_rtol = float(config["inner_vi_relative_bellman_residual"])
    reference_rtol = float(config["reference_relative_true_bellman_residual"])
    true_steps = int(config["true_microsteps"])
    surrogate_steps = [int(x) for x in config["surrogate_microsteps"]]

    for directory in (ROOT / "data" / "raw", ROOT / "data" / "derived", ROOT / "environment"):
        directory.mkdir(parents=True, exist_ok=True)
    save_json(ROOT / "environment" / "environment.json", collect_environment(config))

    # Calibrate on a disjoint small grid using the configured outer-round limit.
    # Select the smallest eligible h; if none qualifies, select the converged h
    # with the minimum median runtime.
    n_cal = int(config["calibration_grid_side"])
    reward = make_reward(n_cal)
    true_operator = MicroSimulator(n_cal, true_steps, params)
    reference, reference_meta = compute_reference_by_value_iteration(
        reward=reward,
        gamma=gamma,
        true_operator=true_operator,
        relative_tolerance=reference_rtol,
    )
    surrogates = {k: MicroSimulator(n_cal, k, params) for k in surrogate_steps}
    calibration_rows: list[dict[str, Any]] = []
    for h in [float(x) for x in config["calibration_h_grid"]]:
        for k in surrogate_steps:
            result, _ = run_ievi(
                reward=reward,
                gamma=gamma,
                true_operator=true_operator,
                surrogate_operator=surrogates[k],
                h=h,
                m=int(config["ievi_inner_depth"]),
                inner_vi_rtol=inner_rtol,
                target_relative_residual=target,
                reference_value=reference,
                grid_side=n_cal,
                repetition=0,
            )
            calibration_rows.append(result.as_dict())
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(ROOT / "data" / "raw" / "calibration_runs.csv", index=False)
    grouped = calibration.groupby("h", as_index=False).agg(
        all_converged=("converged", "all"),
        maximum_outer_rounds=("outer_iterations", "max"),
        median_runtime_seconds=("runtime_seconds", "median"),
        maximum_total_microsteps=("true_microsteps", "max"),
    )
    rule = int(config["calibration_max_outer_rounds_rule"])
    eligible = grouped[grouped["all_converged"] & (grouped["maximum_outer_rounds"] <= rule)]
    if eligible.empty:
        selected = grouped[grouped["all_converged"]].sort_values("median_runtime_seconds").iloc[0]
        selection_rule = "fallback: minimum median runtime among fully converged h values"
    else:
        selected = eligible.sort_values("h").iloc[0]
        selection_rule = f"smallest h converging for every surrogate in at most {rule} outer rounds"
    selected_h = float(selected["h"])
    save_json(
        ROOT / "data" / "derived" / "calibration_selection.json",
        {
            "selected_h": selected_h,
            "selection_rule": selection_rule,
            "calibration_grid_side": n_cal,
            "reference": reference_meta,
            "inner_solver": "ordinary value iteration",
        },
    )

    # Benchmark references, setup records, and fidelity diagnostics.
    setup: dict[str, Any] = {}
    discrepancy_rows: list[dict[str, Any]] = []
    for n in [int(x) for x in config["benchmark_grid_sides"]]:
        reward = make_reward(n)
        true_operator = MicroSimulator(n, true_steps, params)
        surrogate_operators = {k: MicroSimulator(n, k, params) for k in surrogate_steps}
        reference, reference_meta = compute_reference_by_value_iteration(
            reward=reward,
            gamma=gamma,
            true_operator=true_operator,
            relative_tolerance=reference_rtol,
        )
        np.savez_compressed(ROOT / "data" / "raw" / f"reference_n{n}.npz", value=reference, reward=reward)
        setup[str(n)] = {
            "true_operator": true_operator.metadata(),
            "surrogate_operators": {str(k): op.metadata() for k, op in surrogate_operators.items()},
            "reference": reference_meta,
        }
        for k, operator in surrogate_operators.items():
            discrepancy_rows.append(
                {
                    "grid_side": n,
                    "state_count": n * n,
                    "true_microsteps_per_call": true_steps,
                    "surrogate_microsteps_per_call": k,
                    "microstep_cost_ratio": k / true_steps,
                    **discrepancy_probes(true_operator, operator),
                }
            )
    pd.DataFrame(discrepancy_rows).to_csv(
        ROOT / "data" / "raw" / "surrogate_discrepancy.csv", index=False
    )
    save_json(ROOT / "data" / "raw" / "setup_and_reference.json", setup)
    save_json(
        ROOT / "data" / "raw" / "prepare_study-summary.json",
        {
            "completed": True,
            "selected_h": selected_h,
            "timestamp_seconds": perf_counter(),
            "gmres_used": False,
        },
    )
    print(f"Preparation complete; selected h={selected_h:g}")


if __name__ == "__main__":
    main()
