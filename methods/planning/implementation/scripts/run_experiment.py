# ruff: noqa: E402

import argparse
import csv
from dataclasses import replace
import json
import logging
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from besp.core import (
    ExperimentConfig,
    RunnerOverrides,
    build_run_artifacts,
    save_json_payload,
    save_json_rows,
    to_jsonable,
)
from besp.runners.registry import build_runner_plan, execute_runner_plan, flatten_results








def _load_experiment_config(path: str) -> ExperimentConfig:
    config_path = Path(path).resolve()
    strict_maintained = config_path.is_relative_to((REPO_ROOT / "configs").resolve())
    return ExperimentConfig.from_yaml(config_path, strict_maintained=strict_maintained)


def _persist_optional_trace_artifacts(
    results: Sequence[object],
    artifacts,
    summary: dict[str, object],
) -> Path | None:
    trace_rows: list[dict[str, object]] = []
    artifact_name = "pe_traces.json"
    summary_key = "pe_traces_json"
    for result in results:
        traces = getattr(result, "traces", None)
        if not traces:
            continue
        if any(
            "policy_value_errors" in to_jsonable(trace)
            for trace in dict(traces).values()
            if isinstance(to_jsonable(trace), dict)
        ):
            artifact_name = "control_traces.json"
            summary_key = "control_traces_json"
        trace_rows.append(
            {
                "lambda": float(getattr(result, "lambda_value")),
                "seed": int(getattr(result, "seed")),
                "results": dict(getattr(result, "results")),
                "traces": dict(traces),
                **(
                    {"metadata": dict(getattr(result, "metadata"))}
                    if getattr(result, "metadata", None)
                    else {}
                ),
            }
        )

    if not trace_rows:
        return None

    trace_path = artifacts.raw_path(artifact_name)
    save_json_payload(trace_path, trace_rows)
    summary[summary_key] = str(trace_path)
    return trace_path


def main():
    parser = argparse.ArgumentParser(description="BESP experiment runner")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--platform",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Execution platform for maintained JAX workloads",
    )
    parser.add_argument("--backend", type=str, default="jax", choices=["jax"], help="GPU backend")
    parser.add_argument(
        "--cores", type=int, default=multiprocessing.cpu_count(), help="CPU worker count"
    )
    parser.add_argument("--env", type=str, help="Override environment")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    args = parser.parse_args()

    config = _load_experiment_config(args.config)
    overrides = RunnerOverrides.from_cli(
        platform=args.platform,
        backend=args.backend,
        cores=args.cores,
        env=args.env,
        output_dir=args.output_dir,
    )
    planned_platform = overrides.resolved_platform(uses_gpu_algorithms=config.uses_gpu_algorithms)
    if planned_platform == "cpu" and not config.uses_gpu_algorithms:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    exp_name = f"{config.name}_GPU" if planned_platform == "gpu" else config.name
    artifacts = build_run_artifacts(
        experiment_name=exp_name,
        launcher="run_experiment",
        output_dir=overrides.output_dir,
        category="runs",
    )
    artifacts.ensure_directories()
    execution_overrides = replace(overrides, output_dir=artifacts.run_dir)

    with artifacts.config_snapshot_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_mapping(), f, sort_keys=False)

    print(f"Running Experiment: {exp_name}")
    print(f"Output Directory: {artifacts.run_dir}")
    plan = build_runner_plan(config, execution_overrides)
    deterministic_task_count = (
        len(config.build_cpu_tasks(env_name=plan.env_name))
        if plan.mode == "deterministic_jax"
        else len(config.lambdas) * config.n_seeds
    )
    if plan.mode == "stochastic_jax":
        print("Mode: JAX stochastic baselines")
        print("Using JAX Dense Backend (O(S^2) memory).")
    else:
        print(f"Mode: JAX deterministic ({plan.platform})")
        if plan.platform == "gpu":
            print(
                f"Executing {deterministic_task_count} deterministic tasks through the grouped JAX deterministic GPU runner."
            )
        else:
            print(
                f"Submitting {deterministic_task_count} tasks to {overrides.cores} workers..."
            )
    results = execute_runner_plan(plan)

    rows = flatten_results(results)
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_csv(artifacts.results_csv_path, index=False)
        save_json_rows(artifacts.results_json_path, rows)
    else:
        fieldnames = list(rows[0].keys()) if rows else ["lambda", "seed"]
        with artifacts.results_csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        save_json_rows(artifacts.results_json_path, rows)
        df = None

    summary = {
        "name": exp_name,
        "mode": plan.mode,
        "platform": plan.platform,
        "env": plan.env_name,
        "lambda_values": list(config.lambdas),
        "n_rows": int(len(rows)),
        "results_csv": str(artifacts.results_csv_path),
        "raw_results_json": str(artifacts.results_json_path),
        "logs_dir": str(artifacts.logs_dir),
    }
    if artifacts.offline_eval_dir.exists() and any(artifacts.offline_eval_dir.iterdir()):
        summary["offline_eval_dir"] = str(artifacts.offline_eval_dir)
    trace_path = _persist_optional_trace_artifacts(results, artifacts, summary)
    with artifacts.summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Experiment Complete. Results saved to {artifacts.run_dir}")
    if trace_path is not None:
        trace_label = "PE traces" if "pe_traces_json" in summary else "Control traces"
        print(f"{trace_label} saved to {trace_path}")



if __name__ == "__main__":
    main()
