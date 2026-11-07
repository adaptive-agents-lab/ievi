#!/usr/bin/env python3
"""Run VI and IE-VI-with-VI-inner-solves for one size and repetition."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from algorithms import run_ievi, run_value_iteration  # noqa: E402
from micro_simulator import DynamicsParameters, MicroSimulator, make_reward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-side", type=int, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    args = parser.parse_args()

    config = json.load((ROOT / "config.json").open("r", encoding="utf-8"))
    selection = json.load((ROOT / "data" / "derived" / "calibration_selection.json").open("r", encoding="utf-8"))
    dynamics = config["dynamics"]
    params = DynamicsParameters(
        float(dynamics["diffusion_rate"]),
        float(dynamics["advection_amplitude"]),
        float(dynamics["secondary_amplitude"]),
        int(dynamics["phase_seed"]),
    )
    n = args.grid_side
    gamma = float(config["gamma"])
    target = float(config["target_relative_true_bellman_residual"])
    reward = make_reward(n)
    reference = np.load(ROOT / "data" / "raw" / f"reference_n{n}.npz")["value"]
    true_operator = MicroSimulator(n, int(config["true_microsteps"]), params)
    surrogate_steps = [int(x) for x in config["surrogate_microsteps"]]
    surrogates = {k: MicroSimulator(n, k, params) for k in surrogate_steps}

    methods = ["VI"] + [f"IEVI_VI_Ks{k}" for k in surrogate_steps]
    rng = random.Random(int(config["timing_order_seed"]) + 10_000 * n + args.repetition)
    rng.shuffle(methods)
    rows = []
    solutions = {}
    for position, method in enumerate(methods):
        if method == "VI":
            result, solution = run_value_iteration(
                reward=reward,
                gamma=gamma,
                true_operator=true_operator,
                target_relative_residual=target,
                reference_value=reference,
                grid_side=n,
                repetition=args.repetition,
            )
        else:
            k = int(method.split("Ks", maxsplit=1)[1])
            result, solution = run_ievi(
                reward=reward,
                gamma=gamma,
                true_operator=true_operator,
                surrogate_operator=surrogates[k],
                h=float(selection["selected_h"]),
                m=int(config["ievi_inner_depth"]),
                inner_vi_rtol=float(config["inner_vi_relative_bellman_residual"]),
                target_relative_residual=target,
                reference_value=reference,
                grid_side=n,
                repetition=args.repetition,
            )
        row = result.as_dict()
        row["solver_seconds"] = row["runtime_seconds"]
        row["true_setup_seconds"] = true_operator.setup_seconds
        row["surrogate_setup_seconds"] = 0.0 if method == "VI" else surrogates[k].setup_seconds
        row["setup_plus_solver_seconds"] = row["solver_seconds"] + row["true_setup_seconds"] + row["surrogate_setup_seconds"]
        row["timing_order_seed"] = int(config["timing_order_seed"]) + 10_000 * n + args.repetition
        row["method_order_position"] = position
        row["selected_h_from_calibration"] = float(selection["selected_h"])
        rows.append(row)
        if args.repetition == 0:
            solutions[method] = solution
        print(f"n={n} rep={args.repetition} {method}: {result.runtime_seconds:.6f}s")

    output = ROOT / "data" / "raw" / f"benchmark_n{n}_rep{args.repetition}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    if args.repetition == 0:
        np.savez_compressed(ROOT / "data" / "raw" / f"solutions_n{n}_rep0.npz", **solutions)


if __name__ == "__main__":
    main()

