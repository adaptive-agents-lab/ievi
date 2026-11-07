"""Canonical runner selection and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import multiprocessing
from typing import Any, Iterable, Mapping

from ..core.config import ExperimentConfig, RunnerOverrides
from ..core.types import ExperimentExecutionRow
from .gpu_worker import (
    run_deterministic_jax_task,
    run_deterministic_jax_tasks_multi_device,
    run_gpu_batch_task,
)


@dataclass(frozen=True)
class RunnerPlan:
    mode: str
    platform: str
    env_name: str
    config: ExperimentConfig
    overrides: RunnerOverrides
    offline_eval_root: Path | None = None


def build_runner_plan(config: ExperimentConfig, overrides: RunnerOverrides) -> RunnerPlan:
    env_name = config.resolved_env(overrides)
    platform = overrides.resolved_platform(uses_gpu_algorithms=config.uses_gpu_algorithms)
    if overrides.backend != "jax":
        raise ValueError("Active maintained execution is JAX-only.")

    if config.uses_gpu_algorithms and platform != "gpu":
        raise ValueError("Stochastic baseline algorithms are active only through the JAX GPU path.")

    mode = "stochastic_jax" if config.uses_gpu_algorithms else "deterministic_jax"
    return RunnerPlan(
        mode=mode,
        platform=platform,
        env_name=env_name,
        config=config,
        overrides=overrides,
        offline_eval_root=(
            overrides.output_dir / "raw" / "offline_eval"
            if overrides.output_dir is not None
            else None
        ),
    )


def execute_runner_plan(plan: RunnerPlan) -> list[ExperimentExecutionRow | dict[str, Any]]:
    if plan.mode == "stochastic_jax":
        results: list[ExperimentExecutionRow | dict[str, Any]] = []
        for lam in plan.config.lambdas:
            task_args = plan.config.build_gpu_task(
                lam=lam,
                backend=plan.overrides.backend,
                env_name=plan.env_name,
                offline_eval_root=plan.offline_eval_root,
            )
            results.extend(run_gpu_batch_task(task_args))
        return results

    tasks = plan.config.build_cpu_tasks(
        env_name=plan.env_name,
        offline_eval_root=plan.offline_eval_root,
    )
    if plan.platform == "gpu":
        results: list[ExperimentExecutionRow | dict[str, Any]] = list(
            run_deterministic_jax_tasks_multi_device(tasks)
        )
        return results
    with multiprocessing.Pool(processes=plan.overrides.cores) as pool:
        return pool.map(run_deterministic_jax_task, tasks)


def _flatten_result_row(result: ExperimentExecutionRow | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, ExperimentExecutionRow):
        return result.to_flat_row()

    row = {"lambda": result["lambda"], "seed": result["seed"]}
    row.update(result["results"])
    return row


def flatten_results(
    results: Iterable[ExperimentExecutionRow | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(_flatten_result_row(result))
    return rows


__all__ = [
    "RunnerPlan",
    "build_runner_plan",
    "execute_runner_plan",
    "flatten_results",
]
