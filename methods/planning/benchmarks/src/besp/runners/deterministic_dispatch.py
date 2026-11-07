"""Shared deterministic JAX dispatch for maintained runner paths."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, cast

import jax.numpy as jnp

from ..core.config import (
    canonical_algorithm_name,
    is_dd_waterfall_control_algorithm,
    is_dd_waterfall_pe_algorithm,
    is_geo_waterfall_control_algorithm,
    is_geo_waterfall_pe_algorithm,
    normalize_algorithm_config as normalize_algorithm_entries,
)
from ..core.types import (
    ControlTrace,
    CpuExperimentTask,
    ExperimentExecutionRow,
    MDP,
    PETrace,
    default_metric_mode_for_problem_type,
    resolve_evaluation_pipeline,
)
from ..core.offline_eval import plan_offline_eval_layout
from ..domains.control import (
    AISP_Control,
    DDWaterfallISP_Control,
    GeoWaterfallISP_Control,
    ISP_Control,
    MSVI_Control,
    Shadow_ISP_Control,
    VI_Control,
    WaterfallISP_Control,
    build_policy_matrices,
    pick_isp_params_control,
    solve_value,
    value_iteration_control,
)
from .offline_eval import (
    evaluate_offline_policy_bundle,
    load_value_slabs,
    persist_policy_capture_bundle,
)
from ..domains.policy_evaluation import (
    abesp_pe,
    besp_pe,
    dd_waterfall_besp_pe,
    geo_waterfall_besp_pe,
    osvi_pe,
    pick_besp_rho_half,
    shadow_besp_pe,
    vi_pe,
    waterfall_besp_pe,
)
from .deterministic_runtime import (
    LoadedDeterministicTarget,
    build_control_residual_fn,
    compute_value_error,
    control_target_value,
    evaluate_control_policy_value,
    load_deterministic_target_bundle,
    make_zero_value,
    reconstruct_policy_value_errors,
    seeded_shadow_init,
)
import numpy as np

logger = logging.getLogger(__name__)
_TASK_ATTR_ALIASES = {"lambda": "lambda_value"}

_WATERFALL_H_INIT_DEFAULT = 1e6
_WATERFALL_H_MIN_DEFAULT = 1.0
_WATERFALL_STALL_PATIENCE_DEFAULT = 3
_WATERFALL_DECAY_FACTOR_DEFAULT = 0.1
_WATERFALL_INNER_VI_TOL_DEFAULT = 1e-10
_WATERFALL_INNER_VI_MAX_ITERS_DEFAULT = 5000
_WATERFALL_DIVERGENCE_MULT = 1e4
_GEO_WATERFALL_ALPHA_DEFAULT = 0.5
_DD_WATERFALL_M_MAX_DEFAULT = 3
_AISP_TARGET_EPSILON_DEFAULT = 1e-12


@dataclass(frozen=True)
class DeterministicTaskSpec:
    env_name: str
    env_params: Mapping[str, Any]
    lam: float
    seed: int
    max_queries: int
    threshold: float
    algo_configs: Dict[str, Dict[str, Any]]
    error_mode: str
    problem_type: str
    model_norm: str
    value_norm: str
    value_normalized: bool
    emit_pe_traces: bool = False
    emit_control_traces: bool = False
    metric_mode: str = ""
    evaluation_pipeline: str = ""
    deterministic_target_manifest: str | None = None
    task_metadata: Mapping[str, Any] | None = None
    offline_eval_root: str | None = None

    def __post_init__(self) -> None:
        metric_mode = (
            self.metric_mode
            if self.metric_mode
            else default_metric_mode_for_problem_type(self.problem_type)
        )
        object.__setattr__(self, "metric_mode", metric_mode)
        object.__setattr__(
            self,
            "evaluation_pipeline",
            resolve_evaluation_pipeline(
                problem_type=self.problem_type,
                metric_mode=metric_mode,
                requested_pipeline=self.evaluation_pipeline,
            ),
        )


@dataclass(frozen=True)
class PreparedDeterministicTask:
    spec: DeterministicTaskSpec
    mdp: MDP
    p_hat: Any
    target_bundle: LoadedDeterministicTarget | None = None


def _task_attr_name(key: str) -> str:
    return _TASK_ATTR_ALIASES.get(key, key)


def _task_has_key(task_args: CpuExperimentTask | Mapping[str, Any], key: str) -> bool:
    if isinstance(task_args, Mapping):
        return key in task_args
    return hasattr(task_args, _task_attr_name(key))


def _task_value(task_args: CpuExperimentTask | Mapping[str, Any], key: str) -> Any:
    if isinstance(task_args, Mapping):
        return task_args[key]
    return getattr(task_args, _task_attr_name(key))


def normalize_algorithm_config(algorithms_arg: Any) -> Dict[str, Dict[str, Any]]:
    return {
        str(name): dict(params)
        for name, params in normalize_algorithm_entries(algorithms_arg).items()
    }


def _pe_algorithm_family(name: str) -> str:
    if is_dd_waterfall_pe_algorithm(name):
        return "dd_waterfall"
    if is_geo_waterfall_pe_algorithm(name):
        return "geo_waterfall"
    return canonical_algorithm_name(name).split("_")[0]


def _validate_geo_waterfall_config(
    params: Mapping[str, Any],
    *,
    context_name: str,
    valid_keys: set[str],
) -> Dict[str, Any]:
    cfg = dict(params)
    forbidden = sorted({"h_min", "stall_patience", "decay_factor"} & set(cfg))
    if forbidden:
        raise ValueError(
            f"{context_name} must not define staircase Waterfall parameters: {forbidden}."
        )
    unknown = sorted(set(cfg) - valid_keys)
    if unknown:
        raise ValueError(
            f"{context_name} accepts only parameters {sorted(valid_keys)}; received unsupported keys: {unknown}."
        )
    if "h_init" in cfg and float(cfg["h_init"]) <= 0.0:
        raise ValueError(f"{context_name} requires h_init > 0.")
    if "alpha" in cfg:
        alpha = float(cfg["alpha"])
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"{context_name} requires 0 < alpha < 1.")
    return cfg


def _validate_dd_waterfall_config(
    params: Mapping[str, Any],
    *,
    context_name: str,
    valid_keys: set[str],
) -> Dict[str, Any]:
    cfg = dict(params)
    forbidden = sorted({"h_min", "stall_patience", "alpha"} & set(cfg))
    if forbidden:
        raise ValueError(
            f"{context_name} must not define unsupported Waterfall parameters: {forbidden}."
        )
    unknown = sorted(set(cfg) - valid_keys)
    if unknown:
        raise ValueError(
            f"{context_name} accepts only parameters {sorted(valid_keys)}; received unsupported keys: {unknown}."
        )
    if "h_init" in cfg and float(cfg["h_init"]) <= 0.0:
        raise ValueError(f"{context_name} requires h_init > 0.")
    if "m_max" in cfg and int(cfg["m_max"]) < 1:
        raise ValueError(f"{context_name} requires m_max >= 1.")
    if "decay_factor" in cfg:
        decay_factor = float(cfg["decay_factor"])
        if not (0.0 < decay_factor < 1.0):
            raise ValueError(f"{context_name} requires 0 < decay_factor < 1.")
    return cfg


def _build_aisp_config(params: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(params)
    valid_keys = {"target_epsilon"}
    unknown = sorted(set(cfg) - valid_keys)
    if unknown:
        raise ValueError(
            f"aisp accepts only parameters {sorted(valid_keys)}; received unsupported keys: {unknown}."
        )
    if "target_epsilon" in cfg:
        target_epsilon = float(cfg["target_epsilon"])
        if not (0.0 < target_epsilon < 1.0):
            raise ValueError("aisp requires 0 < target_epsilon < 1.")
    return cfg


def parse_deterministic_task(task_args: CpuExperimentTask | Dict[str, Any]) -> DeterministicTaskSpec:
    """Normalize one maintained deterministic task into an internal immutable spec."""

    env_name = _task_value(task_args, "env_name")
    env_params = _task_value(task_args, "env_params")
    lam = _task_value(task_args, "lambda")
    seed = _task_value(task_args, "seed")
    max_queries = _task_value(task_args, "max_queries") if _task_has_key(task_args, "max_queries") else 5000
    threshold = _task_value(task_args, "threshold") if _task_has_key(task_args, "threshold") else 1e-3
    algorithms_arg = _task_value(task_args, "algorithms") if _task_has_key(task_args, "algorithms") else ["vi"]
    error_mode = (
        _task_value(task_args, "error_mode") if _task_has_key(task_args, "error_mode") else "support_uniform"
    )
    problem_type = (
        _task_value(task_args, "problem_type") if _task_has_key(task_args, "problem_type") else "control"
    )
    model_norm = _task_value(task_args, "model_norm") if _task_has_key(task_args, "model_norm") else "L1"
    value_norm = _task_value(task_args, "value_norm") if _task_has_key(task_args, "value_norm") else "L1"
    value_normalized = (
        _task_value(task_args, "value_normalized") if _task_has_key(task_args, "value_normalized") else True
    )
    emit_pe_traces = (
        _task_value(task_args, "emit_pe_traces") if _task_has_key(task_args, "emit_pe_traces") else False
    )
    emit_control_traces = (
        _task_value(task_args, "emit_control_traces")
        if _task_has_key(task_args, "emit_control_traces")
        else False
    )
    raw_metric_mode = (
        _task_value(task_args, "metric_mode")
        if _task_has_key(task_args, "metric_mode")
        else default_metric_mode_for_problem_type(str(problem_type))
    )
    metric_mode = (
        default_metric_mode_for_problem_type(str(problem_type))
        if raw_metric_mode is None or str(raw_metric_mode) == ""
        else str(raw_metric_mode)
    )
    deterministic_target_manifest = (
        _task_value(task_args, "deterministic_target_manifest")
        if _task_has_key(task_args, "deterministic_target_manifest")
        else None
    )
    task_metadata = _task_value(task_args, "task_metadata") if _task_has_key(task_args, "task_metadata") else None
    evaluation_pipeline = (
        _task_value(task_args, "evaluation_pipeline")
        if _task_has_key(task_args, "evaluation_pipeline")
        else ""
    )
    offline_eval_root = (
        _task_value(task_args, "offline_eval_root")
        if _task_has_key(task_args, "offline_eval_root")
        else None
    )
    return DeterministicTaskSpec(
        env_name=env_name,
        env_params=dict(env_params),
        lam=float(lam),
        seed=int(seed),
        max_queries=int(max_queries),
        threshold=float(threshold),
        algo_configs=normalize_algorithm_config(algorithms_arg),
        error_mode=str(error_mode),
        problem_type=str(problem_type),
        model_norm=str(model_norm),
        value_norm=str(value_norm),
        value_normalized=bool(value_normalized),
        emit_pe_traces=bool(emit_pe_traces),
        emit_control_traces=bool(emit_control_traces),
        metric_mode=str(metric_mode),
        evaluation_pipeline=str(evaluation_pipeline),
        deterministic_target_manifest=(
            None if deterministic_target_manifest in (None, "") else str(deterministic_target_manifest)
        ),
        task_metadata=None if task_metadata in (None, {}) else dict(task_metadata),
        offline_eval_root=None if offline_eval_root in (None, "") else str(offline_eval_root),
    )


def prepare_deterministic_task(
    spec: DeterministicTaskSpec,
    *,
    create_environment_fn: Callable[..., MDP],
    apply_model_error_fn: Callable[..., Any],
    shared_mdp: MDP | None = None,
) -> PreparedDeterministicTask:
    """Prepare one maintained deterministic task for execution."""

    env_params = dict(spec.env_params)
    env_params.setdefault("seed", spec.seed)
    mdp = shared_mdp if shared_mdp is not None else create_environment_fn(spec.env_name, **env_params)
    target_bundle = None
    if spec.deterministic_target_manifest:
        target_bundle = load_deterministic_target_bundle(
            spec.deterministic_target_manifest,
            mdp=mdp,
        )
    p_hat = apply_model_error_fn(
        mdp.P,
        lam=spec.lam,
        mode=spec.error_mode,
        seed=spec.seed,
        env_name=spec.env_name,
        env_params=env_params,
    )
    return PreparedDeterministicTask(spec=spec, mdp=mdp, p_hat=p_hat, target_bundle=target_bundle)


def _prepare_pe_target_inputs(
    mdp: MDP,
    p_hat: Any,
    *,
    target_bundle: LoadedDeterministicTarget | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if target_bundle is not None:
        pi_star = jnp.asarray(target_bundle.pi_star, dtype=jnp.int32)
        p_pi = jnp.asarray(target_bundle.p_pi, dtype=jnp.float64)
        r_pi = jnp.asarray(target_bundle.r_pi, dtype=jnp.float64)
        v_true = jnp.asarray(target_bundle.v_pi_star, dtype=jnp.float64)
    else:
        _, pi_star = value_iteration_control(mdp)
        p_pi, r_pi = build_policy_matrices(mdp, pi_star, mdp.P)
        v_true = solve_value(p_pi, r_pi, mdp.gamma)
        p_pi = jnp.asarray(p_pi, dtype=jnp.float64)
        r_pi = jnp.asarray(r_pi, dtype=jnp.float64)
        v_true = jnp.asarray(v_true, dtype=jnp.float64)

    p_hat_pi, _ = build_policy_matrices(mdp, pi_star, p_hat)
    return (
        pi_star,
        p_pi,
        jnp.asarray(p_hat_pi, dtype=jnp.float64),
        r_pi,
        v_true,
    )


def execute_prepared_deterministic_task(
    prepared: PreparedDeterministicTask,
    *,
    run_pe_algorithms_fn: Callable[..., Dict[str, int]] | None = None,
    run_control_algorithms_fn: Callable[..., Dict[str, int]] | None = None,
) -> ExperimentExecutionRow:
    """Execute one already-prepared maintained deterministic task."""

    spec = prepared.spec
    use_default_pe_runner = run_pe_algorithms_fn is None
    use_default_control_runner = run_control_algorithms_fn is None
    if run_pe_algorithms_fn is None:
        run_pe_algorithms_fn = run_pe_algorithms
    if run_control_algorithms_fn is None:
        run_control_algorithms_fn = run_control_algorithms

    if spec.problem_type == "pe":
        if spec.emit_pe_traces and use_default_pe_runner:
            results, traces = run_pe_algorithms_with_optional_traces(
                prepared.mdp,
                prepared.p_hat,
                spec.algo_configs,
                spec.max_queries,
                spec.threshold,
                spec.model_norm,
                spec.value_norm,
                spec.value_normalized,
                emit_traces=True,
                target_bundle=prepared.target_bundle,
            )
        else:
            if use_default_pe_runner:
                results = run_pe_algorithms_fn(
                    prepared.mdp,
                    prepared.p_hat,
                    spec.algo_configs,
                    spec.max_queries,
                    spec.threshold,
                    spec.model_norm,
                    spec.value_norm,
                    spec.value_normalized,
                    target_bundle=prepared.target_bundle,
                )
            else:
                results = _call_with_supported_kwargs(
                    run_pe_algorithms_fn,
                    prepared.mdp,
                    prepared.p_hat,
                    spec.algo_configs,
                    spec.max_queries,
                    spec.threshold,
                    spec.model_norm,
                    spec.value_norm,
                    spec.value_normalized,
                )
            traces = None
    else:
        if use_default_control_runner and (spec.emit_control_traces or spec.metric_mode == "policy_value_error"):
            results, traces = run_control_algorithms_with_optional_traces(
                prepared.mdp,
                prepared.p_hat,
                spec.algo_configs,
                spec.max_queries,
                spec.threshold,
                spec.seed,
                lambda_value=spec.lam,
                metric_mode=spec.metric_mode,
                value_norm=spec.value_norm,
                value_normalized=spec.value_normalized,
                emit_traces=True,
                evaluation_pipeline=spec.evaluation_pipeline,
                offline_eval_root=spec.offline_eval_root,
                target_bundle=prepared.target_bundle,
            )
        else:
            if use_default_control_runner:
                results = run_control_algorithms_fn(
                    prepared.mdp,
                    prepared.p_hat,
                    spec.algo_configs,
                    spec.max_queries,
                    spec.threshold,
                    spec.seed,
                    metric_mode=spec.metric_mode,
                    value_norm=spec.value_norm,
                    value_normalized=spec.value_normalized,
                    target_bundle=prepared.target_bundle,
                )
            else:
                results = _call_with_supported_kwargs(
                    run_control_algorithms_fn,
                    prepared.mdp,
                    prepared.p_hat,
                    spec.algo_configs,
                    spec.max_queries,
                    spec.threshold,
                    spec.seed,
                    metric_mode=spec.metric_mode,
                    value_norm=spec.value_norm,
                    value_normalized=spec.value_normalized,
                    target_bundle=prepared.target_bundle,
                )
            traces = None

    return ExperimentExecutionRow(
        lambda_value=spec.lam,
        seed=spec.seed,
        results=results,
        traces=traces,
        metadata=spec.task_metadata,
    )


def _call_with_supported_kwargs(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call an injected runner while preserving compatibility with narrow test doubles."""

    signature = inspect.signature(func)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return func(*args, **kwargs)

    accepted_kwargs = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return func(*args, **accepted_kwargs)


def _as_jax_array(value: Any) -> jnp.ndarray:
    return cast(jnp.ndarray, value)


def execute_batched_pe_tasks(
    prepared_tasks: List[PreparedDeterministicTask],
) -> List[ExperimentExecutionRow]:
    """Execute a maintained PE bundle with batched JAX kernels where safe."""

    if not prepared_tasks:
        return []

    prepared_inputs = _prepare_batched_pe_inputs(prepared_tasks)
    results_by_task: List[Dict[str, int]] = [dict() for _ in prepared_tasks]
    traces_by_task: List[Dict[str, PETrace]] = [dict() for _ in prepared_tasks]
    supported_groups = _group_supported_pe_algorithms(prepared_tasks)

    for group_key, task_indices in supported_groups.items():
        base_name = group_key[0]
        batch_indices = jnp.asarray(task_indices, dtype=jnp.int32)
        emit_pe_traces = group_key[6]
        if base_name == "vi" and emit_pe_traces:
            queries, errors = _batched_vi_traces(
                prepared_inputs["p_pi"][batch_indices],
                prepared_inputs["r_pi"][batch_indices],
                prepared_inputs["gamma"][batch_indices],
                prepared_inputs["v_true"][batch_indices],
                max_queries=group_key[2],
                value_norm=group_key[4],
                value_normalized=group_key[5],
            )
            hit_times = _batched_hit_times_from_errors(errors, threshold=group_key[3], max_queries=group_key[2])
        elif base_name == "osvi" and emit_pe_traces:
            queries, errors = _batched_osvi_traces(
                prepared_inputs["p_pi"][batch_indices],
                prepared_inputs["p_hat_pi"][batch_indices],
                prepared_inputs["r_pi"][batch_indices],
                prepared_inputs["gamma"][batch_indices],
                prepared_inputs["v_true"][batch_indices],
                max_queries=group_key[2],
                value_norm=group_key[4],
                value_normalized=group_key[5],
            )
            hit_times = _batched_hit_times_from_errors(errors, threshold=group_key[3], max_queries=group_key[2])
        elif base_name == "waterfall" and emit_pe_traces:
            (
                h_init_batch,
                h_min_batch,
                stall_patience_batch,
                decay_factor_batch,
            ) = _batched_waterfall_pe_params(prepared_tasks, task_indices, group_key[1])
            queries, errors = _batched_waterfall_pe_traces(
                prepared_inputs["p_pi"][batch_indices],
                prepared_inputs["p_hat_pi"][batch_indices],
                prepared_inputs["r_pi"][batch_indices],
                prepared_inputs["gamma"][batch_indices],
                prepared_inputs["v_true"][batch_indices],
                max_queries=group_key[2],
                value_norm=group_key[4],
                value_normalized=group_key[5],
                h_init=h_init_batch,
                h_min=h_min_batch,
                stall_patience=stall_patience_batch,
                decay_factor=decay_factor_batch,
            )
            hit_times = _batched_hit_times_from_errors(
                errors,
                threshold=group_key[3],
                max_queries=group_key[2],
            )
        elif base_name == "geo_waterfall" and emit_pe_traces:
            h_init_batch, alpha_batch = _batched_geo_waterfall_pe_params(
                prepared_tasks,
                task_indices,
                group_key[1],
            )
            queries, errors = _batched_geo_waterfall_pe_traces(
                prepared_inputs["p_pi"][batch_indices],
                prepared_inputs["p_hat_pi"][batch_indices],
                prepared_inputs["r_pi"][batch_indices],
                prepared_inputs["gamma"][batch_indices],
                prepared_inputs["v_true"][batch_indices],
                max_queries=group_key[2],
                value_norm=group_key[4],
                value_normalized=group_key[5],
                h_init=h_init_batch,
                alpha=alpha_batch,
            )
            hit_times = _batched_hit_times_from_errors(
                errors,
                threshold=group_key[3],
                max_queries=group_key[2],
            )
        elif base_name == "dd_waterfall" and emit_pe_traces:
            h_init_batch, m_max_batch, decay_factor_batch = _batched_dd_waterfall_pe_params(
                prepared_tasks,
                task_indices,
                group_key[1],
            )
            queries, errors = _batched_dd_waterfall_pe_traces(
                prepared_inputs["p_pi"][batch_indices],
                prepared_inputs["p_hat_pi"][batch_indices],
                prepared_inputs["r_pi"][batch_indices],
                prepared_inputs["gamma"][batch_indices],
                prepared_inputs["v_true"][batch_indices],
                max_queries=group_key[2],
                value_norm=group_key[4],
                value_normalized=group_key[5],
                h_init=h_init_batch,
                m_max=m_max_batch,
                decay_factor=decay_factor_batch,
            )
            hit_times = _batched_hit_times_from_errors(
                errors,
                threshold=group_key[3],
                max_queries=group_key[2],
            )
        elif base_name == "vi":
            hit_times = _batched_vi_hit_times(
                prepared_inputs["p_pi"][batch_indices],
                prepared_inputs["r_pi"][batch_indices],
                prepared_inputs["gamma"][batch_indices],
                prepared_inputs["v_true"][batch_indices],
                max_queries=group_key[2],
                threshold=group_key[3],
                value_norm=group_key[4],
                value_normalized=group_key[5],
            )
            queries = None
            errors = None
        elif base_name == "osvi":
            hit_times = _batched_osvi_hit_times(
                prepared_inputs["p_pi"][batch_indices],
                prepared_inputs["p_hat_pi"][batch_indices],
                prepared_inputs["r_pi"][batch_indices],
                prepared_inputs["gamma"][batch_indices],
                prepared_inputs["v_true"][batch_indices],
                max_queries=group_key[2],
                threshold=group_key[3],
                value_norm=group_key[4],
                value_normalized=group_key[5],
            )
            queries = None
            errors = None
        elif base_name == "waterfall":
            (
                h_init_batch,
                h_min_batch,
                stall_patience_batch,
                decay_factor_batch,
            ) = _batched_waterfall_pe_params(prepared_tasks, task_indices, group_key[1])
            hit_times = _batched_waterfall_pe_hit_times(
                prepared_inputs["p_pi"][batch_indices],
                prepared_inputs["p_hat_pi"][batch_indices],
                prepared_inputs["r_pi"][batch_indices],
                prepared_inputs["gamma"][batch_indices],
                prepared_inputs["v_true"][batch_indices],
                max_queries=group_key[2],
                threshold=group_key[3],
                value_norm=group_key[4],
                value_normalized=group_key[5],
                h_init=h_init_batch,
                h_min=h_min_batch,
                stall_patience=stall_patience_batch,
                decay_factor=decay_factor_batch,
            )
            queries = None
            errors = None
        elif base_name == "geo_waterfall":
            h_init_batch, alpha_batch = _batched_geo_waterfall_pe_params(
                prepared_tasks,
                task_indices,
                group_key[1],
            )
            hit_times = _batched_geo_waterfall_pe_hit_times(
                prepared_inputs["p_pi"][batch_indices],
                prepared_inputs["p_hat_pi"][batch_indices],
                prepared_inputs["r_pi"][batch_indices],
                prepared_inputs["gamma"][batch_indices],
                prepared_inputs["v_true"][batch_indices],
                max_queries=group_key[2],
                threshold=group_key[3],
                value_norm=group_key[4],
                value_normalized=group_key[5],
                h_init=h_init_batch,
                alpha=alpha_batch,
            )
            queries = None
            errors = None
        elif base_name == "dd_waterfall":
            h_init_batch, m_max_batch, decay_factor_batch = _batched_dd_waterfall_pe_params(
                prepared_tasks,
                task_indices,
                group_key[1],
            )
            hit_times = _batched_dd_waterfall_pe_hit_times(
                prepared_inputs["p_pi"][batch_indices],
                prepared_inputs["p_hat_pi"][batch_indices],
                prepared_inputs["r_pi"][batch_indices],
                prepared_inputs["gamma"][batch_indices],
                prepared_inputs["v_true"][batch_indices],
                max_queries=group_key[2],
                threshold=group_key[3],
                value_norm=group_key[4],
                value_normalized=group_key[5],
                h_init=h_init_batch,
                m_max=m_max_batch,
                decay_factor=decay_factor_batch,
            )
            queries = None
            errors = None
        else:
            continue

        for task_index, hit_time in zip(task_indices, hit_times):
            algo_name = group_key[1]
            results_by_task[task_index][algo_name] = hit_time
        if emit_pe_traces and queries is not None and errors is not None:
            for local_index, task_index in enumerate(task_indices):
                algo_name = group_key[1]
                traces_by_task[task_index][algo_name] = PETrace(
                    queries=[int(value) for value in queries[local_index].tolist()],
                    errors=[float(value) for value in errors[local_index].tolist()],
                    hit_time=int(hit_times[local_index]),
                    threshold=float(group_key[3]),
                    value_norm=str(group_key[4]),
                    value_normalized=bool(group_key[5]),
                )

    for task_index, prepared in enumerate(prepared_tasks):
        remaining_algorithms = {
            name: params
            for name, params in prepared.spec.algo_configs.items()
            if name not in results_by_task[task_index]
        }
        if remaining_algorithms:
            if prepared.spec.emit_pe_traces:
                extra_results, extra_traces = run_pe_algorithms_with_optional_traces(
                    prepared.mdp,
                    prepared.p_hat,
                    remaining_algorithms,
                    prepared.spec.max_queries,
                    prepared.spec.threshold,
                    prepared.spec.model_norm,
                    prepared.spec.value_norm,
                    prepared.spec.value_normalized,
                )
                results_by_task[task_index].update(extra_results)
                traces_by_task[task_index].update(extra_traces)
            else:
                results_by_task[task_index].update(
                    run_pe_algorithms(
                        prepared.mdp,
                        prepared.p_hat,
                        remaining_algorithms,
                        prepared.spec.max_queries,
                        prepared.spec.threshold,
                        prepared.spec.model_norm,
                        prepared.spec.value_norm,
                        prepared.spec.value_normalized,
                    )
                )

    return [
        ExperimentExecutionRow(
            lambda_value=prepared.spec.lam,
            seed=prepared.spec.seed,
            results=results_by_task[index],
            traces=traces_by_task[index] or None,
            metadata=prepared.spec.task_metadata,
        )
        for index, prepared in enumerate(prepared_tasks)
    ]


def execute_batched_control_tasks(
    prepared_tasks: List[PreparedDeterministicTask],
) -> List[ExperimentExecutionRow]:
    """Execute a maintained control bundle with batched JAX kernels where safe."""

    if not prepared_tasks:
        return []

    if any(prepared.spec.evaluation_pipeline == "train_then_offline" for prepared in prepared_tasks):
        return [execute_prepared_deterministic_task(prepared) for prepared in prepared_tasks]

    if not _can_stack_control_tasks(prepared_tasks):
        return [execute_prepared_deterministic_task(prepared) for prepared in prepared_tasks]

    prepared_inputs = _prepare_batched_control_inputs(prepared_tasks)
    results_by_task: List[Dict[str, int]] = [dict() for _ in prepared_tasks]
    traces_by_task: List[Dict[str, ControlTrace]] = [dict() for _ in prepared_tasks]
    supported_groups = _group_supported_control_algorithms(prepared_tasks)

    for group_key, task_indices in supported_groups.items():
        base_name = group_key[0]
        algo_name = group_key[1]
        batch_indices = jnp.asarray(task_indices, dtype=jnp.int32)
        metric_mode = group_key[6]
        emit_control_traces = group_key[7]
        osvi_fixed_steps = None if group_key[8] < 0 else int(group_key[8])
        try:
            if base_name == "vi" and (emit_control_traces or metric_mode == "policy_value_error"):
                queries, residuals, value_errors, policy_value_errors, policy_trace = (
                    _batched_vi_control_traces(
                        prepared_inputs["p"][batch_indices],
                        prepared_inputs["r"][batch_indices],
                        prepared_inputs["gamma"][batch_indices],
                        prepared_inputs["target_v"][batch_indices],
                        max_queries=group_key[2],
                        value_norm=group_key[4],
                        value_normalized=group_key[5],
                    )
                )
                if metric_mode == "policy_value_error":
                    hit_times = _batched_control_hit_times_from_policy_value_errors(
                        policy_value_errors,
                        threshold=group_key[3],
                        max_queries=group_key[2],
                    )
                else:
                    hit_times = _batched_control_hit_times_from_residuals(
                        residuals,
                        effect_threshold=prepared_inputs["effect_threshold"][batch_indices],
                        max_queries=group_key[2],
                    )
            elif base_name == "vi":
                hit_times = _batched_vi_control_hit_times(
                    prepared_inputs["p"][batch_indices],
                    prepared_inputs["r"][batch_indices],
                    prepared_inputs["gamma"][batch_indices],
                    prepared_inputs["effect_threshold"][batch_indices],
                    max_queries=group_key[2],
                )
                queries = None
                residuals = None
                value_errors = None
                policy_value_errors = None
                policy_trace = None
            elif base_name == "osvi" and (emit_control_traces or metric_mode == "policy_value_error"):
                queries, residuals, value_errors, policy_value_errors, policy_trace = (
                    _batched_osvi_control_traces(
                        prepared_inputs["p"][batch_indices],
                        prepared_inputs["p_hat"][batch_indices],
                        prepared_inputs["r"][batch_indices],
                        prepared_inputs["gamma"][batch_indices],
                        prepared_inputs["target_v"][batch_indices],
                        max_queries=group_key[2],
                        value_norm=group_key[4],
                        value_normalized=group_key[5],
                        surrogate_policy_iteration_steps=osvi_fixed_steps,
                    )
                )
                if metric_mode == "policy_value_error":
                    hit_times = _batched_control_hit_times_from_policy_value_errors(
                        policy_value_errors,
                        threshold=group_key[3],
                        max_queries=group_key[2],
                    )
                else:
                    hit_times = _batched_control_hit_times_from_residuals(
                        residuals,
                        effect_threshold=prepared_inputs["effect_threshold"][batch_indices],
                        max_queries=group_key[2],
                    )
            elif base_name == "osvi":
                hit_times = _batched_osvi_control_hit_times(
                    prepared_inputs["p"][batch_indices],
                    prepared_inputs["p_hat"][batch_indices],
                    prepared_inputs["r"][batch_indices],
                    prepared_inputs["gamma"][batch_indices],
                    prepared_inputs["effect_threshold"][batch_indices],
                    max_queries=group_key[2],
                    surrogate_policy_iteration_steps=osvi_fixed_steps,
                )
                queries = None
                residuals = None
                value_errors = None
                policy_value_errors = None
                policy_trace = None
            elif base_name == "waterfall" and (emit_control_traces or metric_mode == "policy_value_error"):
                (
                    h_init_batch,
                    h_min_batch,
                    stall_patience_batch,
                    decay_factor_batch,
                    inner_vi_tol_batch,
                    inner_vi_max_iters_batch,
                ) = _batched_waterfall_params(
                    prepared_tasks,
                    task_indices,
                    group_key[1],
                )
                queries, residuals, value_errors, policy_value_errors, policy_trace = (
                    _batched_waterfall_control_traces(
                        prepared_inputs["p"][batch_indices],
                        prepared_inputs["p_hat"][batch_indices],
                        prepared_inputs["r"][batch_indices],
                        prepared_inputs["gamma"][batch_indices],
                        prepared_inputs["target_v"][batch_indices],
                        h_init=h_init_batch,
                        h_min=h_min_batch,
                        stall_patience=stall_patience_batch,
                        decay_factor=decay_factor_batch,
                        inner_vi_tol=inner_vi_tol_batch,
                        inner_vi_max_iters=inner_vi_max_iters_batch,
                        max_queries=group_key[2],
                        value_norm=group_key[4],
                        value_normalized=group_key[5],
                    )
                )
                if metric_mode == "policy_value_error":
                    hit_times = _batched_control_hit_times_from_policy_value_errors(
                        policy_value_errors,
                        threshold=group_key[3],
                        max_queries=group_key[2],
                    )
                else:
                    hit_times = _batched_control_hit_times_from_residuals(
                        residuals,
                        effect_threshold=prepared_inputs["effect_threshold"][batch_indices],
                        max_queries=group_key[2],
                    )
            elif base_name == "geo_waterfall" and (emit_control_traces or metric_mode == "policy_value_error"):
                (
                    h_init_batch,
                    alpha_batch,
                    inner_vi_tol_batch,
                    inner_vi_max_iters_batch,
                ) = _batched_geo_waterfall_params(
                    prepared_tasks,
                    task_indices,
                    group_key[1],
                )
                queries, residuals, value_errors, policy_value_errors, policy_trace = (
                    _batched_geo_waterfall_control_traces(
                        prepared_inputs["p"][batch_indices],
                        prepared_inputs["p_hat"][batch_indices],
                        prepared_inputs["r"][batch_indices],
                        prepared_inputs["gamma"][batch_indices],
                        prepared_inputs["target_v"][batch_indices],
                        h_init=h_init_batch,
                        alpha=alpha_batch,
                        inner_vi_tol=inner_vi_tol_batch,
                        inner_vi_max_iters=inner_vi_max_iters_batch,
                        max_queries=group_key[2],
                        value_norm=group_key[4],
                        value_normalized=group_key[5],
                    )
                )
                if metric_mode == "policy_value_error":
                    hit_times = _batched_control_hit_times_from_policy_value_errors(
                        policy_value_errors,
                        threshold=group_key[3],
                        max_queries=group_key[2],
                    )
                else:
                    hit_times = _batched_control_hit_times_from_residuals(
                        residuals,
                        effect_threshold=prepared_inputs["effect_threshold"][batch_indices],
                        max_queries=group_key[2],
                    )
            elif base_name == "dd_waterfall" and (emit_control_traces or metric_mode == "policy_value_error"):
                (
                    h_init_batch,
                    m_max_batch,
                    decay_factor_batch,
                    inner_vi_tol_batch,
                    inner_vi_max_iters_batch,
                ) = _batched_dd_waterfall_params(
                    prepared_tasks,
                    task_indices,
                    group_key[1],
                )
                queries, residuals, value_errors, policy_value_errors, policy_trace = (
                    _batched_dd_waterfall_control_traces(
                        prepared_inputs["p"][batch_indices],
                        prepared_inputs["p_hat"][batch_indices],
                        prepared_inputs["r"][batch_indices],
                        prepared_inputs["gamma"][batch_indices],
                        prepared_inputs["target_v"][batch_indices],
                        h_init=h_init_batch,
                        m_max=m_max_batch,
                        decay_factor=decay_factor_batch,
                        inner_vi_tol=inner_vi_tol_batch,
                        inner_vi_max_iters=inner_vi_max_iters_batch,
                        max_queries=group_key[2],
                        value_norm=group_key[4],
                        value_normalized=group_key[5],
                    )
                )
                if metric_mode == "policy_value_error":
                    hit_times = _batched_control_hit_times_from_query_aligned_errors(
                        queries,
                        policy_value_errors,
                        threshold=group_key[3],
                        max_queries=group_key[2],
                    )
                else:
                    hit_times = _batched_control_hit_times_from_query_aligned_thresholds(
                        queries,
                        residuals,
                        effect_threshold=prepared_inputs["effect_threshold"][batch_indices],
                        max_queries=group_key[2],
                    )
            elif base_name == "isp" and (emit_control_traces or metric_mode == "policy_value_error"):
                h_batch, j_batch = _batched_isp_params(prepared_tasks, task_indices, group_key[1])
                queries, residuals, value_errors, policy_value_errors, policy_trace = (
                    _batched_isp_control_traces(
                        prepared_inputs["p"][batch_indices],
                        prepared_inputs["p_hat"][batch_indices],
                        prepared_inputs["r"][batch_indices],
                        prepared_inputs["gamma"][batch_indices],
                        prepared_inputs["target_v"][batch_indices],
                        h=h_batch,
                        j_steps=j_batch,
                        max_queries=group_key[2],
                        value_norm=group_key[4],
                        value_normalized=group_key[5],
                    )
                )
                if metric_mode == "policy_value_error":
                    hit_times = _batched_control_hit_times_from_query_aligned_errors(
                        queries,
                        policy_value_errors,
                        threshold=group_key[3],
                        max_queries=group_key[2],
                    )
                else:
                    hit_times = _batched_isp_control_hit_times(
                        prepared_inputs["p"][batch_indices],
                        prepared_inputs["p_hat"][batch_indices],
                        prepared_inputs["r"][batch_indices],
                        prepared_inputs["gamma"][batch_indices],
                        prepared_inputs["effect_threshold"][batch_indices],
                        h=h_batch,
                        j_steps=j_batch,
                        max_queries=group_key[2],
                    )
            elif base_name == "aisp" and (emit_control_traces or metric_mode == "policy_value_error"):
                target_epsilon_batch = _batched_aisp_target_epsilons(
                    prepared_tasks,
                    task_indices,
                    group_key[1],
                )
                queries, residuals, value_errors, policy_value_errors, policy_trace = (
                    _batched_aisp_control_traces(
                        prepared_inputs["p"][batch_indices],
                        prepared_inputs["p_hat"][batch_indices],
                        prepared_inputs["r"][batch_indices],
                        prepared_inputs["gamma"][batch_indices],
                        prepared_inputs["target_v"][batch_indices],
                        max_queries=group_key[2],
                        value_norm=group_key[4],
                        value_normalized=group_key[5],
                        target_epsilon=target_epsilon_batch,
                    )
                )
                if metric_mode == "policy_value_error":
                    hit_times = _batched_control_hit_times_from_query_aligned_errors(
                        queries,
                        policy_value_errors,
                        threshold=group_key[3],
                        max_queries=group_key[2],
                    )
                else:
                    hit_times = _batched_control_hit_times_from_query_aligned_thresholds(
                        queries,
                        residuals,
                        effect_threshold=prepared_inputs["effect_threshold"][batch_indices],
                        max_queries=group_key[2],
                    )
            elif base_name == "isp":
                h_batch, j_batch = _batched_isp_params(prepared_tasks, task_indices, group_key[1])
                hit_times = _batched_isp_control_hit_times(
                    prepared_inputs["p"][batch_indices],
                    prepared_inputs["p_hat"][batch_indices],
                    prepared_inputs["r"][batch_indices],
                    prepared_inputs["gamma"][batch_indices],
                    prepared_inputs["effect_threshold"][batch_indices],
                    h=h_batch,
                    j_steps=j_batch,
                    max_queries=group_key[2],
                )
                queries = None
                residuals = None
                value_errors = None
                policy_value_errors = None
                policy_trace = None
            elif base_name == "aisp":
                target_epsilon_batch = _batched_aisp_target_epsilons(
                    prepared_tasks,
                    task_indices,
                    group_key[1],
                )
                hit_times = _batched_aisp_control_hit_times(
                    prepared_inputs["p"][batch_indices],
                    prepared_inputs["p_hat"][batch_indices],
                    prepared_inputs["r"][batch_indices],
                    prepared_inputs["gamma"][batch_indices],
                    prepared_inputs["effect_threshold"][batch_indices],
                    max_queries=group_key[2],
                    target_epsilon=target_epsilon_batch,
                )
                queries = None
                residuals = None
                value_errors = None
                policy_value_errors = None
                policy_trace = None
            elif base_name == "shadow" and (emit_control_traces or metric_mode == "policy_value_error"):
                seed_batch, optimism_batch, dynamic_j_batch, fixed_j_batch = _batched_shadow_params(
                    prepared_tasks,
                    task_indices,
                    group_key[1],
                )
                queries, residuals, value_errors, policy_value_errors, policy_trace = (
                    _batched_shadow_control_traces(
                        prepared_inputs["p"][batch_indices],
                        prepared_inputs["p_hat"][batch_indices],
                        prepared_inputs["r"][batch_indices],
                        prepared_inputs["gamma"][batch_indices],
                        prepared_inputs["target_v"][batch_indices],
                        seeds=seed_batch,
                        optimism=optimism_batch,
                        dynamic_j=dynamic_j_batch,
                        fixed_j=fixed_j_batch,
                        max_queries=group_key[2],
                        value_norm=group_key[4],
                        value_normalized=group_key[5],
                    )
                )
                if metric_mode == "policy_value_error":
                    hit_times = _batched_control_hit_times_from_query_aligned_errors(
                        queries,
                        policy_value_errors,
                        threshold=group_key[3],
                        max_queries=group_key[2],
                    )
                else:
                    hit_times = _batched_control_hit_times_from_query_aligned_thresholds(
                        queries,
                        residuals,
                        effect_threshold=prepared_inputs["effect_threshold"][batch_indices],
                        max_queries=group_key[2],
                    )
            elif base_name == "shadow":
                seed_batch, optimism_batch, dynamic_j_batch, fixed_j_batch = _batched_shadow_params(
                    prepared_tasks,
                    task_indices,
                    group_key[1],
                )
                hit_times = _batched_shadow_control_hit_times(
                    prepared_inputs["p"][batch_indices],
                    prepared_inputs["p_hat"][batch_indices],
                    prepared_inputs["r"][batch_indices],
                    prepared_inputs["gamma"][batch_indices],
                    prepared_inputs["effect_threshold"][batch_indices],
                    seeds=seed_batch,
                    optimism=optimism_batch,
                    dynamic_j=dynamic_j_batch,
                    fixed_j=fixed_j_batch,
                    max_queries=group_key[2],
                )
                queries = None
                residuals = None
                value_errors = None
                policy_value_errors = None
                policy_trace = None
            elif base_name == "waterfall":
                (
                    h_init_batch,
                    h_min_batch,
                    stall_patience_batch,
                    decay_factor_batch,
                    inner_vi_tol_batch,
                    inner_vi_max_iters_batch,
                ) = _batched_waterfall_params(
                    prepared_tasks,
                    task_indices,
                    group_key[1],
                )
                hit_times = _batched_waterfall_control_hit_times(
                    prepared_inputs["p"][batch_indices],
                    prepared_inputs["p_hat"][batch_indices],
                    prepared_inputs["r"][batch_indices],
                    prepared_inputs["gamma"][batch_indices],
                    prepared_inputs["effect_threshold"][batch_indices],
                    h_init=h_init_batch,
                    h_min=h_min_batch,
                    stall_patience=stall_patience_batch,
                    decay_factor=decay_factor_batch,
                    inner_vi_tol=inner_vi_tol_batch,
                    inner_vi_max_iters=inner_vi_max_iters_batch,
                    max_queries=group_key[2],
                )
                queries = None
                residuals = None
                value_errors = None
                policy_value_errors = None
                policy_trace = None
            elif base_name == "geo_waterfall":
                (
                    h_init_batch,
                    alpha_batch,
                    inner_vi_tol_batch,
                    inner_vi_max_iters_batch,
                ) = _batched_geo_waterfall_params(
                    prepared_tasks,
                    task_indices,
                    group_key[1],
                )
                hit_times = _batched_geo_waterfall_control_hit_times(
                    prepared_inputs["p"][batch_indices],
                    prepared_inputs["p_hat"][batch_indices],
                    prepared_inputs["r"][batch_indices],
                    prepared_inputs["gamma"][batch_indices],
                    prepared_inputs["effect_threshold"][batch_indices],
                    h_init=h_init_batch,
                    alpha=alpha_batch,
                    inner_vi_tol=inner_vi_tol_batch,
                    inner_vi_max_iters=inner_vi_max_iters_batch,
                    max_queries=group_key[2],
                )
                queries = None
                residuals = None
                value_errors = None
                policy_value_errors = None
                policy_trace = None
            elif base_name == "dd_waterfall":
                (
                    h_init_batch,
                    m_max_batch,
                    decay_factor_batch,
                    inner_vi_tol_batch,
                    inner_vi_max_iters_batch,
                ) = _batched_dd_waterfall_params(
                    prepared_tasks,
                    task_indices,
                    group_key[1],
                )
                hit_times = _batched_dd_waterfall_control_hit_times(
                    prepared_inputs["p"][batch_indices],
                    prepared_inputs["p_hat"][batch_indices],
                    prepared_inputs["r"][batch_indices],
                    prepared_inputs["gamma"][batch_indices],
                    prepared_inputs["effect_threshold"][batch_indices],
                    h_init=h_init_batch,
                    m_max=m_max_batch,
                    decay_factor=decay_factor_batch,
                    inner_vi_tol=inner_vi_tol_batch,
                    inner_vi_max_iters=inner_vi_max_iters_batch,
                    max_queries=group_key[2],
                )
                queries = None
                residuals = None
                value_errors = None
                policy_value_errors = None
                policy_trace = None
            else:
                continue
        except ValueError:
            raise
        except Exception:
            continue

        for task_index, hit_time in zip(task_indices, hit_times):
            results_by_task[task_index][algo_name] = hit_time
        if (
            emit_control_traces
            and queries is not None
            and residuals is not None
            and value_errors is not None
            and policy_value_errors is not None
            and policy_trace is not None
        ):
            for local_index, task_index in enumerate(task_indices):
                prepared = prepared_tasks[task_index]
                target_bundle = prepared.target_bundle
                query_row = [int(value) for value in queries[local_index].tolist()]
                residual_row = [float(value) for value in residuals[local_index].tolist()]
                value_error_row = [float(value) for value in value_errors[local_index].tolist()]
                policy_value_error_row = [
                    float(value) for value in policy_value_errors[local_index].tolist()
                ]
                policy_row = [
                    [int(action) for action in policy]
                    for policy in policy_trace[local_index].tolist()
                ]
                finite_mask = [
                    math.isfinite(resid)
                    or math.isfinite(val_err)
                    or math.isfinite(policy_err)
                    for resid, val_err, policy_err in zip(
                        residual_row,
                        value_error_row,
                        policy_value_error_row,
                        strict=True,
                    )
                ]
                valid_length = (
                    max(index for index, valid in enumerate(finite_mask, start=1) if valid)
                    if any(finite_mask)
                    else 0
                )
                traces_by_task[task_index][algo_name] = ControlTrace(
                    queries=query_row[:valid_length],
                    residuals=residual_row[:valid_length],
                    value_errors=value_error_row[:valid_length],
                    policy_value_errors=policy_value_error_row[:valid_length],
                    policy_trace=policy_row[:valid_length],
                    hit_time=int(hit_times[local_index]),
                    threshold=float(group_key[3]),
                    metric_mode=str(metric_mode),
                    value_norm=str(group_key[4]),
                    value_normalized=bool(group_key[5]),
                    target_id=None if target_bundle is None else str(target_bundle.target_id),
                    target_manifest=None
                    if target_bundle is None
                    else str(target_bundle.manifest_path),
                )

    for task_index, prepared in enumerate(prepared_tasks):
        remaining_algorithms = {
            name: params
            for name, params in prepared.spec.algo_configs.items()
            if name not in results_by_task[task_index]
        }
        if remaining_algorithms:
            if prepared.spec.emit_control_traces or prepared.spec.metric_mode == "policy_value_error":
                extra_results, extra_traces = run_control_algorithms_with_optional_traces(
                    prepared.mdp,
                    prepared.p_hat,
                    remaining_algorithms,
                    prepared.spec.max_queries,
                    prepared.spec.threshold,
                    prepared.spec.seed,
                    metric_mode=prepared.spec.metric_mode,
                    value_norm=prepared.spec.value_norm,
                    value_normalized=prepared.spec.value_normalized,
                    emit_traces=prepared.spec.emit_control_traces,
                    target_bundle=prepared.target_bundle,
                )
                results_by_task[task_index].update(extra_results)
                traces_by_task[task_index].update(extra_traces)
            else:
                results_by_task[task_index].update(
                    run_control_algorithms(
                        prepared.mdp,
                        prepared.p_hat,
                        remaining_algorithms,
                        prepared.spec.max_queries,
                        prepared.spec.threshold,
                        prepared.spec.seed,
                    )
                )

    return [
        ExperimentExecutionRow(
            lambda_value=prepared.spec.lam,
            seed=prepared.spec.seed,
            results=results_by_task[index],
            traces=traces_by_task[index] or None,
            metadata=prepared.spec.task_metadata,
        )
        for index, prepared in enumerate(prepared_tasks)
    ]


def run_deterministic_task(
    task_args: CpuExperimentTask | Dict[str, Any],
    *,
    create_environment_fn: Callable[..., MDP],
    apply_model_error_fn: Callable[..., Any],
    run_pe_algorithms_fn: Callable[..., Dict[str, int]] | None = None,
    run_control_algorithms_fn: Callable[..., Dict[str, int]] | None = None,
) -> ExperimentExecutionRow:
    """Run one deterministic maintained task on the currently selected platform."""
    spec = parse_deterministic_task(task_args)
    prepared = prepare_deterministic_task(
        spec,
        create_environment_fn=create_environment_fn,
        apply_model_error_fn=apply_model_error_fn,
    )
    return execute_prepared_deterministic_task(
        prepared,
        run_pe_algorithms_fn=run_pe_algorithms_fn,
        run_control_algorithms_fn=run_control_algorithms_fn,
    )


def _prepare_batched_pe_inputs(
    prepared_tasks: List[PreparedDeterministicTask],
) -> Dict[str, jnp.ndarray]:
    shared_policy = None
    if all(prepared.mdp is prepared_tasks[0].mdp for prepared in prepared_tasks[1:]):
        first_target = prepared_tasks[0].target_bundle
        first_target_id = None if first_target is None else first_target.target_id
        if all(
            (
                prepared.target_bundle is None
                if first_target_id is None
                else prepared.target_bundle is not None and prepared.target_bundle.target_id == first_target_id
            )
            for prepared in prepared_tasks[1:]
        ):
            shared_policy = _prepare_pe_target_inputs(
                prepared_tasks[0].mdp,
                prepared_tasks[0].p_hat,
                target_bundle=first_target,
            )

    p_pi_list = []
    p_hat_pi_list = []
    r_pi_list = []
    v_true_list = []
    gamma_list = []

    for prepared in prepared_tasks:
        if shared_policy is None:
            pi_star, p_pi, _, r_pi, v_true = _prepare_pe_target_inputs(
                prepared.mdp,
                prepared.p_hat,
                target_bundle=prepared.target_bundle,
            )
        else:
            pi_star, p_pi, _, r_pi, v_true = shared_policy

        p_hat_pi, _ = build_policy_matrices(prepared.mdp, pi_star, prepared.p_hat)
        p_pi_list.append(jnp.asarray(p_pi, dtype=jnp.float64))
        p_hat_pi_list.append(jnp.asarray(p_hat_pi, dtype=jnp.float64))
        r_pi_list.append(jnp.asarray(r_pi, dtype=jnp.float64))
        v_true_list.append(jnp.asarray(v_true, dtype=jnp.float64))
        gamma_list.append(prepared.mdp.gamma)

    return {
        "p_pi": jnp.stack(p_pi_list, axis=0),
        "p_hat_pi": jnp.stack(p_hat_pi_list, axis=0),
        "r_pi": jnp.stack(r_pi_list, axis=0),
        "v_true": jnp.stack(v_true_list, axis=0),
        "gamma": jnp.asarray(gamma_list, dtype=jnp.float64),
    }


def _can_stack_control_tasks(prepared_tasks: List[PreparedDeterministicTask]) -> bool:
    if not prepared_tasks:
        return False
    p_shape = prepared_tasks[0].mdp.P.shape
    r_shape = prepared_tasks[0].mdp.R.shape
    p_hat_shape = jnp.asarray(prepared_tasks[0].p_hat).shape
    return all(
        prepared.mdp.P.shape == p_shape
        and prepared.mdp.R.shape == r_shape
        and jnp.asarray(prepared.p_hat).shape == p_hat_shape
        for prepared in prepared_tasks[1:]
    )


def _group_supported_pe_algorithms(
    prepared_tasks: List[PreparedDeterministicTask],
) -> Dict[tuple[str, str, int, float, str, bool, bool], List[int]]:
    groups: Dict[tuple[str, str, int, float, str, bool, bool], List[int]] = {}
    for index, prepared in enumerate(prepared_tasks):
        spec = prepared.spec
        for name in spec.algo_configs:
            base_name = _pe_algorithm_family(name)
            if base_name not in {"vi", "osvi", "waterfall", "geo_waterfall", "dd_waterfall"}:
                continue
            key = (
                base_name,
                name,
                spec.max_queries,
                spec.threshold,
                spec.value_norm,
                spec.value_normalized,
                spec.emit_pe_traces,
            )
            groups.setdefault(key, []).append(index)
    return groups


def _batched_waterfall_pe_params(
    prepared_tasks: List[PreparedDeterministicTask],
    task_indices: List[int],
    algo_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    h_init_values: list[float] = []
    h_min_values: list[float] = []
    stall_patience_values: list[int] = []
    decay_factor_values: list[float] = []

    for task_index in task_indices:
        prepared = prepared_tasks[task_index]
        cfg = _build_waterfall_config(prepared.spec.algo_configs[algo_name])
        h_init_values.append(float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)))
        h_min_values.append(float(cfg.get("h_min", _WATERFALL_H_MIN_DEFAULT)))
        stall_patience_values.append(
            max(int(cfg.get("stall_patience", _WATERFALL_STALL_PATIENCE_DEFAULT)), 1)
        )
        decay_factor_values.append(float(cfg.get("decay_factor", _WATERFALL_DECAY_FACTOR_DEFAULT)))

    return (
        jnp.asarray(h_init_values, dtype=jnp.float64),
        jnp.asarray(h_min_values, dtype=jnp.float64),
        jnp.asarray(stall_patience_values, dtype=jnp.int32),
        jnp.asarray(decay_factor_values, dtype=jnp.float64),
    )


def _build_geo_waterfall_pe_config(params: Dict[str, Any]) -> Dict[str, Any]:
    return _validate_geo_waterfall_config(
        params,
        context_name="geo_waterfall_besp",
        valid_keys={"h_init", "alpha"},
    )


def _build_geo_waterfall_control_config(params: Dict[str, Any]) -> Dict[str, Any]:
    return _validate_geo_waterfall_config(
        params,
        context_name="geo_waterfall",
        valid_keys={"h_init", "alpha", "inner_vi_tol", "inner_vi_max_iters"},
    )


def _build_dd_waterfall_pe_config(params: Dict[str, Any]) -> Dict[str, Any]:
    return _validate_dd_waterfall_config(
        params,
        context_name="dd_waterfall_besp",
        valid_keys={"h_init", "m_max", "decay_factor"},
    )


def _build_dd_waterfall_control_config(params: Dict[str, Any]) -> Dict[str, Any]:
    return _validate_dd_waterfall_config(
        params,
        context_name="dd_waterfall",
        valid_keys={"h_init", "m_max", "decay_factor", "inner_vi_tol", "inner_vi_max_iters"},
    )


def _batched_geo_waterfall_pe_params(
    prepared_tasks: List[PreparedDeterministicTask],
    task_indices: List[int],
    algo_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    h_init_values: list[float] = []
    alpha_values: list[float] = []

    for task_index in task_indices:
        prepared = prepared_tasks[task_index]
        cfg = _build_geo_waterfall_pe_config(prepared.spec.algo_configs[algo_name])
        h_init_values.append(float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)))
        alpha_values.append(float(cfg.get("alpha", _GEO_WATERFALL_ALPHA_DEFAULT)))

    return (
        jnp.asarray(h_init_values, dtype=jnp.float64),
        jnp.asarray(alpha_values, dtype=jnp.float64),
    )


def _batched_dd_waterfall_pe_params(
    prepared_tasks: List[PreparedDeterministicTask],
    task_indices: List[int],
    algo_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    h_init_values: list[float] = []
    m_max_values: list[int] = []
    decay_factor_values: list[float] = []

    for task_index in task_indices:
        prepared = prepared_tasks[task_index]
        cfg = _build_dd_waterfall_pe_config(prepared.spec.algo_configs[algo_name])
        h_init_values.append(float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)))
        m_max_values.append(max(int(cfg.get("m_max", _DD_WATERFALL_M_MAX_DEFAULT)), 1))
        decay_factor_values.append(float(cfg.get("decay_factor", _WATERFALL_DECAY_FACTOR_DEFAULT)))

    return (
        jnp.asarray(h_init_values, dtype=jnp.float64),
        jnp.asarray(m_max_values, dtype=jnp.int32),
        jnp.asarray(decay_factor_values, dtype=jnp.float64),
    )


def _prepare_batched_control_inputs(
    prepared_tasks: List[PreparedDeterministicTask],
) -> Dict[str, jnp.ndarray]:
    p_list = []
    p_hat_list = []
    r_list = []
    gamma_list = []
    threshold_list = []
    target_v_list = []

    for prepared in prepared_tasks:
        p_arr = jnp.asarray(prepared.mdp.P, dtype=jnp.float64)
        r_arr = jnp.asarray(prepared.mdp.R, dtype=jnp.float64)
        p_hat_arr = jnp.asarray(prepared.p_hat, dtype=jnp.float64)
        p_list.append(p_arr)
        p_hat_list.append(p_hat_arr)
        r_list.append(r_arr)
        gamma_list.append(prepared.mdp.gamma)
        threshold_list.append(prepared.spec.threshold)
        target_v_list.append(
            jnp.asarray(
                control_target_value(
                    prepared.mdp,
                    target_bundle=prepared.target_bundle,
                ),
                dtype=jnp.float64,
            )
        )

    p = jnp.stack(p_list, axis=0)
    r = jnp.stack(r_list, axis=0)
    effect_threshold = jnp.maximum(
        _batched_control_residual_norms(
            jnp.zeros((p.shape[0], p.shape[1]), dtype=jnp.float64),
            p,
            r,
            jnp.asarray(gamma_list, dtype=jnp.float64),
        )
        * jnp.asarray(threshold_list, dtype=jnp.float64),
        1e-10,
    )
    return {
        "p": p,
        "p_hat": jnp.stack(p_hat_list, axis=0),
        "r": r,
        "gamma": jnp.asarray(gamma_list, dtype=jnp.float64),
        "effect_threshold": effect_threshold,
        "target_v": jnp.stack(target_v_list, axis=0),
    }


def _group_supported_control_algorithms(
    prepared_tasks: List[PreparedDeterministicTask],
) -> Dict[tuple[str, str, int, float, str, bool, str, bool, int], List[int]]:
    groups: Dict[tuple[str, str, int, float, str, bool, str, bool, int], List[int]] = {}
    for index, prepared in enumerate(prepared_tasks):
        for name in prepared.spec.algo_configs:
            base_name = _control_algorithm_family(name)
            if base_name not in {
                "vi",
                "osvi",
                "isp",
                "aisp",
                "shadow",
                "waterfall",
                "geo_waterfall",
                "dd_waterfall",
            }:
                continue
            osvi_fixed_steps = -1
            if base_name == "osvi":
                raw_steps = prepared.spec.algo_configs[name].get("surrogate_policy_iteration_steps")
                osvi_fixed_steps = -1 if raw_steps is None else max(int(raw_steps), 1)
            key = (
                base_name,
                name,
                prepared.spec.max_queries,
                prepared.spec.threshold,
                prepared.spec.value_norm,
                prepared.spec.value_normalized,
                prepared.spec.metric_mode,
                prepared.spec.emit_control_traces,
                osvi_fixed_steps,
            )
            groups.setdefault(key, []).append(index)
    return groups


def _control_algorithm_family(name: str) -> str:
    canonical = canonical_algorithm_name(name)
    if is_dd_waterfall_control_algorithm(canonical):
        return "dd_waterfall"
    if is_geo_waterfall_control_algorithm(canonical):
        return "geo_waterfall"
    base_name = canonical.split("_")[0]
    if base_name == "shadow" or "dual" in canonical:
        return "shadow"
    return base_name


def _osvi_surrogate_policy_iteration_steps(params: Mapping[str, Any]) -> int | None:
    raw_value = params.get("surrogate_policy_iteration_steps")
    if raw_value is None:
        return None
    return max(int(raw_value), 1)


def _batched_isp_params(
    prepared_tasks: List[PreparedDeterministicTask],
    task_indices: List[int],
    algo_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    h_values: list[float] = []
    j_values: list[int] = []

    for task_index in task_indices:
        prepared = prepared_tasks[task_index]
        params = prepared.spec.algo_configs[algo_name]
        default_h, default_j = pick_isp_params_control(prepared.mdp, prepared.p_hat)
        h_values.append(float(params.get("h", default_h)))
        j_values.append(max(int(params.get("J", default_j)), 1))

    return (
        jnp.asarray(h_values, dtype=jnp.float64),
        jnp.asarray(j_values, dtype=jnp.int32),
    )


def _batched_aisp_target_epsilons(
    prepared_tasks: List[PreparedDeterministicTask],
    task_indices: List[int],
    algo_name: str,
) -> jnp.ndarray:
    target_epsilons: list[float] = []
    for task_index in task_indices:
        prepared = prepared_tasks[task_index]
        cfg = _build_aisp_config(prepared.spec.algo_configs[algo_name])
        target_epsilons.append(float(cfg.get("target_epsilon", _AISP_TARGET_EPSILON_DEFAULT)))
    return jnp.asarray(target_epsilons, dtype=jnp.float64)


def _batched_shadow_params(
    prepared_tasks: List[PreparedDeterministicTask],
    task_indices: List[int],
    algo_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    seeds: list[int] = []
    optimism_values: list[float] = []
    dynamic_j_values: list[bool] = []
    fixed_j_values: list[int] = []

    for task_index in task_indices:
        prepared = prepared_tasks[task_index]
        cfg = _build_shadow_config(algo_name, prepared.spec.algo_configs[algo_name])
        seeds.append(prepared.spec.seed)
        optimism_values.append(float(cfg.get("optimism", 1.0)))
        dynamic_j = bool(cfg.get("dynamic_J", False))
        fixed_j = cfg.get("fixed_J")
        if fixed_j is None and not dynamic_j:
            delta = max(1.0 - prepared.mdp.gamma, 1e-12)
            fixed_j = max(1, int(math.ceil(math.log2(8.0 / delta))))
        dynamic_j_values.append(dynamic_j)
        fixed_j_values.append(int(fixed_j) if fixed_j is not None else -1)

    return (
        jnp.asarray(seeds, dtype=jnp.int32),
        jnp.asarray(optimism_values, dtype=jnp.float64),
        jnp.asarray(dynamic_j_values, dtype=jnp.bool_),
        jnp.asarray(fixed_j_values, dtype=jnp.int32),
    )


def _batched_waterfall_params(
    prepared_tasks: List[PreparedDeterministicTask],
    task_indices: List[int],
    algo_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    h_init_values: list[float] = []
    h_min_values: list[float] = []
    stall_patience_values: list[int] = []
    decay_factor_values: list[float] = []
    inner_vi_tol_values: list[float] = []
    inner_vi_max_iters_values: list[int] = []

    for task_index in task_indices:
        prepared = prepared_tasks[task_index]
        cfg = _build_waterfall_config(prepared.spec.algo_configs[algo_name])
        h_init_values.append(float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)))
        h_min_values.append(float(cfg.get("h_min", _WATERFALL_H_MIN_DEFAULT)))
        stall_patience_values.append(
            max(int(cfg.get("stall_patience", _WATERFALL_STALL_PATIENCE_DEFAULT)), 1)
        )
        decay_factor_values.append(float(cfg.get("decay_factor", _WATERFALL_DECAY_FACTOR_DEFAULT)))
        inner_vi_tol_values.append(float(cfg.get("inner_vi_tol", _WATERFALL_INNER_VI_TOL_DEFAULT)))
        inner_vi_max_iters_values.append(
            max(int(cfg.get("inner_vi_max_iters", _WATERFALL_INNER_VI_MAX_ITERS_DEFAULT)), 1)
        )

    return (
        jnp.asarray(h_init_values, dtype=jnp.float64),
        jnp.asarray(h_min_values, dtype=jnp.float64),
        jnp.asarray(stall_patience_values, dtype=jnp.int32),
        jnp.asarray(decay_factor_values, dtype=jnp.float64),
        jnp.asarray(inner_vi_tol_values, dtype=jnp.float64),
        jnp.asarray(inner_vi_max_iters_values, dtype=jnp.int32),
    )


def _batched_geo_waterfall_params(
    prepared_tasks: List[PreparedDeterministicTask],
    task_indices: List[int],
    algo_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    h_init_values: list[float] = []
    alpha_values: list[float] = []
    inner_vi_tol_values: list[float] = []
    inner_vi_max_iters_values: list[int] = []

    for task_index in task_indices:
        prepared = prepared_tasks[task_index]
        cfg = _build_geo_waterfall_control_config(prepared.spec.algo_configs[algo_name])
        h_init_values.append(float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)))
        alpha_values.append(float(cfg.get("alpha", _GEO_WATERFALL_ALPHA_DEFAULT)))
        inner_vi_tol_values.append(float(cfg.get("inner_vi_tol", _WATERFALL_INNER_VI_TOL_DEFAULT)))
        inner_vi_max_iters_values.append(
            max(int(cfg.get("inner_vi_max_iters", _WATERFALL_INNER_VI_MAX_ITERS_DEFAULT)), 1)
        )

    return (
        jnp.asarray(h_init_values, dtype=jnp.float64),
        jnp.asarray(alpha_values, dtype=jnp.float64),
        jnp.asarray(inner_vi_tol_values, dtype=jnp.float64),
        jnp.asarray(inner_vi_max_iters_values, dtype=jnp.int32),
    )


def _batched_dd_waterfall_params(
    prepared_tasks: List[PreparedDeterministicTask],
    task_indices: List[int],
    algo_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    h_init_values: list[float] = []
    m_max_values: list[int] = []
    decay_factor_values: list[float] = []
    inner_vi_tol_values: list[float] = []
    inner_vi_max_iters_values: list[int] = []

    for task_index in task_indices:
        prepared = prepared_tasks[task_index]
        cfg = _build_dd_waterfall_control_config(prepared.spec.algo_configs[algo_name])
        h_init_values.append(float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)))
        m_max_values.append(max(int(cfg.get("m_max", _DD_WATERFALL_M_MAX_DEFAULT)), 1))
        decay_factor_values.append(float(cfg.get("decay_factor", _WATERFALL_DECAY_FACTOR_DEFAULT)))
        inner_vi_tol_values.append(float(cfg.get("inner_vi_tol", _WATERFALL_INNER_VI_TOL_DEFAULT)))
        inner_vi_max_iters_values.append(
            max(int(cfg.get("inner_vi_max_iters", _WATERFALL_INNER_VI_MAX_ITERS_DEFAULT)), 1)
        )

    return (
        jnp.asarray(h_init_values, dtype=jnp.float64),
        jnp.asarray(m_max_values, dtype=jnp.int32),
        jnp.asarray(decay_factor_values, dtype=jnp.float64),
        jnp.asarray(inner_vi_tol_values, dtype=jnp.float64),
        jnp.asarray(inner_vi_max_iters_values, dtype=jnp.int32),
    )


def _batched_value_norm(vec: jnp.ndarray, *, norm_name: str) -> jnp.ndarray:
    key = norm_name.lower()
    if key == "l2":
        return jnp.linalg.norm(vec, axis=1)
    if key in ("inf", "linf"):
        return jnp.max(jnp.abs(vec), axis=1)
    return jnp.sum(jnp.abs(vec), axis=1)


def _batched_compute_value_error(
    v: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    norm_name: str,
    normalized: bool,
    eps: float = 1e-12,
) -> jnp.ndarray:
    diff = v - v_true
    numer = _batched_value_norm(diff, norm_name=norm_name)
    if not normalized:
        return numer
    denom = _batched_value_norm(v_true, norm_name=norm_name) + eps
    return numer / denom


def _batched_vi_hit_times(
    p_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    threshold: float,
    value_norm: str,
    value_normalized: bool,
) -> List[int]:
    v = jnp.zeros_like(v_true)
    hit_times = jnp.full((p_pi.shape[0],), max_queries + 1, dtype=jnp.int32)
    errors = _batched_compute_value_error(
        v,
        v_true,
        norm_name=value_norm,
        normalized=value_normalized,
    )
    hit_times = jnp.where(errors <= threshold, 0, hit_times)

    for step in range(1, max_queries + 1):
        if not bool(jnp.any(hit_times == max_queries + 1)):
            break
        v = r_pi + gamma[:, None] * jnp.einsum("bij,bj->bi", p_pi, v)
        errors = _batched_compute_value_error(
            v,
            v_true,
            norm_name=value_norm,
            normalized=value_normalized,
        )
        new_hits = jnp.logical_and(hit_times == max_queries + 1, errors <= threshold)
        hit_times = jnp.where(new_hits, step, hit_times)

    return [int(value) for value in hit_times.tolist()]


def _batched_vi_traces(
    p_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    v = jnp.zeros_like(v_true)
    errors = [
        _batched_compute_value_error(
            v,
            v_true,
            norm_name=value_norm,
            normalized=value_normalized,
        )
    ]

    for _ in range(max_queries):
        v = r_pi + gamma[:, None] * jnp.einsum("bij,bj->bi", p_pi, v)
        errors.append(
            _batched_compute_value_error(
                v,
                v_true,
                norm_name=value_norm,
                normalized=value_normalized,
            )
        )

    error_trace = jnp.stack(errors, axis=1)
    query_trace = jnp.broadcast_to(
        jnp.arange(max_queries + 1, dtype=jnp.int32),
        error_trace.shape,
    )
    return query_trace, error_trace


def _batched_osvi_hit_times(
    p_pi: jnp.ndarray,
    p_hat_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    threshold: float,
    value_norm: str,
    value_normalized: bool,
) -> List[int]:
    batch_size, n_states, _ = p_pi.shape
    eye = jnp.broadcast_to(jnp.eye(n_states, dtype=jnp.float64), (batch_size, n_states, n_states))
    k_hat = eye - gamma[:, None, None] * p_hat_pi
    k_hat_inv = jnp.linalg.solve(k_hat, eye)

    v = jnp.zeros_like(v_true)
    hit_times = jnp.full((batch_size,), max_queries + 1, dtype=jnp.int32)
    errors = _batched_compute_value_error(
        v,
        v_true,
        norm_name=value_norm,
        normalized=value_normalized,
    )
    hit_times = jnp.where(errors <= threshold, 0, hit_times)

    for step in range(1, max_queries + 1):
        if not bool(jnp.any(hit_times == max_queries + 1)):
            break
        u_true = jnp.einsum("bij,bj->bi", p_pi, v)
        u_hat = jnp.einsum("bij,bj->bi", p_hat_pi, v)
        rhs = r_pi + gamma[:, None] * (u_true - u_hat)
        v = jnp.einsum("bij,bj->bi", k_hat_inv, rhs)
        errors = _batched_compute_value_error(
            v,
            v_true,
            norm_name=value_norm,
            normalized=value_normalized,
        )
        new_hits = jnp.logical_and(hit_times == max_queries + 1, errors <= threshold)
        hit_times = jnp.where(new_hits, step, hit_times)

    return [int(value) for value in hit_times.tolist()]


def _batched_osvi_traces(
    p_pi: jnp.ndarray,
    p_hat_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, n_states, _ = p_pi.shape
    eye = jnp.broadcast_to(jnp.eye(n_states, dtype=jnp.float64), (batch_size, n_states, n_states))
    k_hat = eye - gamma[:, None, None] * p_hat_pi
    k_hat_inv = jnp.linalg.solve(k_hat, eye)

    v = jnp.zeros_like(v_true)
    errors = [
        _batched_compute_value_error(
            v,
            v_true,
            norm_name=value_norm,
            normalized=value_normalized,
        )
    ]

    for _ in range(max_queries):
        u_true = jnp.einsum("bij,bj->bi", p_pi, v)
        u_hat = jnp.einsum("bij,bj->bi", p_hat_pi, v)
        rhs = r_pi + gamma[:, None] * (u_true - u_hat)
        v = jnp.einsum("bij,bj->bi", k_hat_inv, rhs)
        errors.append(
            _batched_compute_value_error(
                v,
                v_true,
                norm_name=value_norm,
                normalized=value_normalized,
            )
        )

    error_trace = jnp.stack(errors, axis=1)
    query_trace = jnp.broadcast_to(
        jnp.arange(max_queries + 1, dtype=jnp.int32),
        error_trace.shape,
    )
    return query_trace, error_trace


def _batched_hit_times_from_errors(
    errors: jnp.ndarray,
    *,
    threshold: float,
    max_queries: int,
) -> List[int]:
    hit_mask = errors <= threshold
    any_hit = jnp.any(hit_mask, axis=1)
    first_hit = jnp.argmax(hit_mask.astype(jnp.int32), axis=1)
    hit_times = jnp.where(any_hit, first_hit, max_queries + 1)
    return [int(value) for value in hit_times.tolist()]


def _forward_fill_error_trace(errors: jnp.ndarray) -> jnp.ndarray:
    filled = errors
    width = int(errors.shape[1])
    for index in range(1, width):
        prev = filled[:, index - 1]
        cur = filled[:, index]
        replace = jnp.logical_not(jnp.isfinite(cur))
        filled = filled.at[:, index].set(jnp.where(replace, prev, cur))
    return filled


def _batched_waterfall_pe_common(
    p_pi: jnp.ndarray,
    p_hat_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
    h_init: jnp.ndarray,
    h_min: jnp.ndarray,
    stall_patience: jnp.ndarray,
    decay_factor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, n_states, _ = p_pi.shape
    eye = jnp.broadcast_to(jnp.eye(n_states, dtype=jnp.float64), (batch_size, n_states, n_states))
    v = jnp.zeros_like(v_true)
    br = r_pi + gamma[:, None] * jnp.einsum("bij,bj->bi", p_pi, v) - v
    best_br_norm = jnp.max(jnp.abs(br), axis=1)
    total_queries = jnp.zeros((batch_size,), dtype=jnp.int32)
    h = jnp.asarray(h_init, dtype=jnp.float64)
    h_floor = jnp.asarray(h_min, dtype=jnp.float64)
    stall_patience_arr = jnp.asarray(stall_patience, dtype=jnp.int32)
    decay_factor_arr = jnp.asarray(decay_factor, dtype=jnp.float64)
    stall_count = jnp.zeros((batch_size,), dtype=jnp.int32)
    v_div_envelope = _WATERFALL_DIVERGENCE_MULT * jnp.maximum(
        jnp.max(jnp.abs(r_pi), axis=1) / jnp.maximum(1.0 - gamma, 1e-12),
        1.0,
    )

    errors = jnp.full((batch_size, max_queries + 1), jnp.nan, dtype=jnp.float64)
    errors = errors.at[:, 0].set(
        _batched_compute_value_error(
            v,
            v_true,
            norm_name=value_norm,
            normalized=value_normalized,
        )
    )

    while True:
        active = cast(jnp.ndarray, total_queries < max_queries)
        if not bool(jnp.any(active)):
            break

        mat = _as_jax_array((1.0 + h)[:, None, None] * eye - (h * gamma)[:, None, None] * p_hat_pi)
        w = _as_jax_array(jnp.linalg.solve(mat, br[..., None]))[..., 0]
        v_candidate = _as_jax_array(v + h[:, None] * w)

        diverged = jnp.max(jnp.abs(v_candidate), axis=1) > v_div_envelope
        new_br = r_pi + gamma[:, None] * jnp.einsum("bij,bj->bi", p_pi, v_candidate) - v_candidate
        new_br_norm = jnp.max(jnp.abs(new_br), axis=1)
        nonfinite = jnp.logical_or(
            jnp.logical_not(jnp.all(jnp.isfinite(v_candidate), axis=1)),
            jnp.logical_not(jnp.isfinite(new_br_norm)),
        )
        failed = jnp.logical_and(active, jnp.logical_or(diverged, nonfinite))
        charged_queries = _as_jax_array(jnp.where(active, total_queries + 1, total_queries))

        next_h_on_fail = jnp.maximum(h_floor, h * decay_factor_arr)
        should_stop_after_fail = jnp.logical_and(failed, next_h_on_fail <= h_floor)
        h = _as_jax_array(jnp.where(failed, next_h_on_fail, h))
        stall_count = _as_jax_array(jnp.where(failed, 0, stall_count))

        succeeded = jnp.logical_and(active, jnp.logical_not(failed))
        candidate_errors = _batched_compute_value_error(
            v_candidate,
            v_true,
            norm_name=value_norm,
            normalized=value_normalized,
        )
        retained_errors = _batched_compute_value_error(
            v,
            v_true,
            norm_name=value_norm,
            normalized=value_normalized,
        )
        current_errors = _as_jax_array(jnp.where(failed, retained_errors, candidate_errors))
        for local_index in range(batch_size):
            if bool(active[local_index]):
                errors = errors.at[local_index, int(charged_queries[local_index])].set(
                    current_errors[local_index]
                )

        improved = new_br_norm < best_br_norm
        best_br_norm = _as_jax_array(jnp.where(jnp.logical_and(succeeded, improved), new_br_norm, best_br_norm))
        tentative_stall_count = _as_jax_array(jnp.where(
            jnp.logical_and(succeeded, improved),
            0,
            jnp.where(succeeded, stall_count + 1, stall_count),
        ))
        should_decay_after_success = jnp.logical_and(
            succeeded,
            jnp.logical_and(jnp.logical_not(improved), tentative_stall_count >= stall_patience_arr),
        )
        h = _as_jax_array(jnp.where(should_decay_after_success, jnp.maximum(h_floor, h * decay_factor_arr), h))
        stall_count = _as_jax_array(jnp.where(
            jnp.logical_and(succeeded, improved),
            0,
            jnp.where(should_decay_after_success, 0, tentative_stall_count),
        ))
        total_queries = _as_jax_array(jnp.where(should_stop_after_fail, max_queries, charged_queries))
        v = _as_jax_array(jnp.where(succeeded[:, None], v_candidate, v))
        br = _as_jax_array(jnp.where(succeeded[:, None], new_br, br))

    errors = _forward_fill_error_trace(errors)
    queries = jnp.broadcast_to(
        jnp.arange(max_queries + 1, dtype=jnp.int32),
        errors.shape,
    )
    return queries, errors


def _batched_geo_waterfall_pe_common(
    p_pi: jnp.ndarray,
    p_hat_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
    h_init: jnp.ndarray,
    alpha: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, n_states, _ = p_pi.shape
    eye = jnp.broadcast_to(jnp.eye(n_states, dtype=jnp.float64), (batch_size, n_states, n_states))
    v = jnp.zeros_like(v_true)
    br = r_pi + gamma[:, None] * jnp.einsum("bij,bj->bi", p_pi, v) - v
    current_br_norm = jnp.max(jnp.abs(br), axis=1)
    best_br_norm = current_br_norm
    total_queries = jnp.zeros((batch_size,), dtype=jnp.int32)
    h = jnp.asarray(h_init, dtype=jnp.float64)
    alpha_arr = jnp.asarray(alpha, dtype=jnp.float64)
    v_div_envelope = _WATERFALL_DIVERGENCE_MULT * jnp.maximum(
        jnp.max(jnp.abs(r_pi), axis=1) / jnp.maximum(1.0 - gamma, 1e-12),
        1.0,
    )

    errors = jnp.full((batch_size, max_queries + 1), jnp.nan, dtype=jnp.float64)
    errors = errors.at[:, 0].set(
        _batched_compute_value_error(
            v,
            v_true,
            norm_name=value_norm,
            normalized=value_normalized,
        )
    )

    while True:
        active = cast(jnp.ndarray, total_queries < max_queries)
        if not bool(jnp.any(active)):
            break

        mat = _as_jax_array((1.0 + h)[:, None, None] * eye - (h * gamma)[:, None, None] * p_hat_pi)
        w = _as_jax_array(jnp.linalg.solve(mat, br[..., None]))[..., 0]
        v_candidate = _as_jax_array(v + h[:, None] * w)

        diverged = jnp.max(jnp.abs(v_candidate), axis=1) > v_div_envelope
        new_br = r_pi + gamma[:, None] * jnp.einsum("bij,bj->bi", p_pi, v_candidate) - v_candidate
        new_br_norm = jnp.max(jnp.abs(new_br), axis=1)
        nonfinite = jnp.logical_or(
            jnp.logical_not(jnp.all(jnp.isfinite(v_candidate), axis=1)),
            jnp.logical_not(jnp.isfinite(new_br_norm)),
        )
        valid = jnp.logical_and(active, jnp.logical_not(jnp.logical_or(diverged, nonfinite)))
        improved = new_br_norm < best_br_norm
        next_total_queries = _as_jax_array(jnp.where(active, total_queries + 1, total_queries))
        next_v = _as_jax_array(jnp.where(valid[:, None], v_candidate, v))
        next_br = _as_jax_array(jnp.where(valid[:, None], new_br, br))
        next_current_br_norm = _as_jax_array(jnp.where(valid, new_br_norm, current_br_norm))
        next_best_br_norm = _as_jax_array(jnp.where(jnp.logical_and(valid, improved), new_br_norm, best_br_norm))
        next_h = _as_jax_array(jnp.where(
            jnp.logical_and(valid, improved),
            h,
            jnp.where(active, jnp.maximum(1.0, alpha_arr * h), h),
        ))

        current_errors = _batched_compute_value_error(
            next_v,
            v_true,
            norm_name=value_norm,
            normalized=value_normalized,
        )
        for local_index in range(batch_size):
            if bool(active[local_index]):
                errors = errors.at[local_index, int(next_total_queries[local_index])].set(
                    current_errors[local_index]
                )

        total_queries = next_total_queries
        v = next_v
        br = next_br
        current_br_norm = next_current_br_norm
        best_br_norm = next_best_br_norm
        h = next_h

    queries = jnp.broadcast_to(
        jnp.arange(max_queries + 1, dtype=jnp.int32),
        errors.shape,
    )
    return queries, errors


def _batched_waterfall_pe_hit_times(
    p_pi: jnp.ndarray,
    p_hat_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    threshold: float,
    value_norm: str,
    value_normalized: bool,
    h_init: jnp.ndarray,
    h_min: jnp.ndarray,
    stall_patience: jnp.ndarray,
    decay_factor: jnp.ndarray,
) -> List[int]:
    _, errors = _batched_waterfall_pe_common(
        p_pi,
        p_hat_pi,
        r_pi,
        gamma,
        v_true,
        max_queries=max_queries,
        value_norm=value_norm,
        value_normalized=value_normalized,
        h_init=h_init,
        h_min=h_min,
        stall_patience=stall_patience,
        decay_factor=decay_factor,
    )
    return _batched_hit_times_from_errors(errors, threshold=threshold, max_queries=max_queries)


def _batched_waterfall_pe_traces(
    p_pi: jnp.ndarray,
    p_hat_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
    h_init: jnp.ndarray,
    h_min: jnp.ndarray,
    stall_patience: jnp.ndarray,
    decay_factor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _batched_waterfall_pe_common(
        p_pi,
        p_hat_pi,
        r_pi,
        gamma,
        v_true,
        max_queries=max_queries,
        value_norm=value_norm,
        value_normalized=value_normalized,
        h_init=h_init,
        h_min=h_min,
        stall_patience=stall_patience,
        decay_factor=decay_factor,
    )


def _batched_geo_waterfall_pe_hit_times(
    p_pi: jnp.ndarray,
    p_hat_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    threshold: float,
    value_norm: str,
    value_normalized: bool,
    h_init: jnp.ndarray,
    alpha: jnp.ndarray,
) -> List[int]:
    _, errors = _batched_geo_waterfall_pe_common(
        p_pi,
        p_hat_pi,
        r_pi,
        gamma,
        v_true,
        max_queries=max_queries,
        value_norm=value_norm,
        value_normalized=value_normalized,
        h_init=h_init,
        alpha=alpha,
    )
    return _batched_hit_times_from_errors(errors, threshold=threshold, max_queries=max_queries)


def _batched_geo_waterfall_pe_traces(
    p_pi: jnp.ndarray,
    p_hat_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
    h_init: jnp.ndarray,
    alpha: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _batched_geo_waterfall_pe_common(
        p_pi,
        p_hat_pi,
        r_pi,
        gamma,
        v_true,
        max_queries=max_queries,
        value_norm=value_norm,
        value_normalized=value_normalized,
        h_init=h_init,
        alpha=alpha,
    )


def _batched_dd_waterfall_pe_common(
    p_pi: jnp.ndarray,
    p_hat_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
    h_init: jnp.ndarray,
    m_max: jnp.ndarray,
    decay_factor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, n_states, _ = p_pi.shape
    eye = jnp.broadcast_to(jnp.eye(n_states, dtype=jnp.float64), (batch_size, n_states, n_states))
    v = jnp.zeros_like(v_true)
    br = r_pi + gamma[:, None] * jnp.einsum("bij,bj->bi", p_pi, v) - v
    current_br_norm = jnp.max(jnp.abs(br), axis=1)
    best_br_norm = current_br_norm
    total_queries = jnp.zeros((batch_size,), dtype=jnp.int32)
    h = jnp.asarray(h_init, dtype=jnp.float64)
    m_max_arr = jnp.asarray(m_max, dtype=jnp.int32)
    decay_factor_arr = jnp.asarray(decay_factor, dtype=jnp.float64)
    v_div_envelope = _WATERFALL_DIVERGENCE_MULT * jnp.maximum(
        jnp.max(jnp.abs(r_pi), axis=1) / jnp.maximum(1.0 - gamma, 1e-12),
        1.0,
    )
    max_depth = max(int(value) for value in m_max_arr.tolist())

    errors = jnp.full((batch_size, max_queries + 1), jnp.nan, dtype=jnp.float64)
    errors = errors.at[:, 0].set(
        _batched_compute_value_error(
            v,
            v_true,
            norm_name=value_norm,
            normalized=value_normalized,
        )
    )

    while True:
        active = cast(jnp.ndarray, total_queries < max_queries)
        if not bool(jnp.any(active)):
            break

        remaining_budget = jnp.asarray(max_queries, dtype=jnp.int32) - total_queries
        attempt_limit = _as_jax_array(jnp.minimum(m_max_arr, remaining_budget))
        anchor_v = v
        u = anchor_v
        u_br = br
        next_v = v
        next_br = br
        next_current_br_norm = current_br_norm
        next_best_br_norm = best_br_norm
        next_h = h
        depth_cost = jnp.zeros((batch_size,), dtype=jnp.int32)
        inv_1ph = 1.0 / (1.0 + h)
        blend = h * inv_1ph
        gamma_blend = gamma * blend
        mat = eye - gamma_blend[:, None, None] * p_hat_pi

        for depth in range(1, max_depth + 1):
            depth_active = _as_jax_array(jnp.logical_and(active, depth_cost == 0))
            depth_active = _as_jax_array(jnp.logical_and(depth_active, attempt_limit >= depth))
            if not bool(jnp.any(depth_active)):
                continue

            expected_u_hat = jnp.einsum("bij,bj->bi", p_hat_pi, u)
            r_tilde = _as_jax_array(
                inv_1ph[:, None] * anchor_v
                + blend[:, None] * (u_br + u)
                - gamma_blend[:, None] * expected_u_hat
            )
            u_candidate = _as_jax_array(jnp.linalg.solve(mat, r_tilde[..., None]))[..., 0]
            diverged = jnp.max(jnp.abs(u_candidate), axis=1) > v_div_envelope
            new_br = r_pi + gamma[:, None] * jnp.einsum("bij,bj->bi", p_pi, u_candidate) - u_candidate
            candidate_br_norm = jnp.max(jnp.abs(new_br), axis=1)
            valid = jnp.logical_and(
                depth_active,
                jnp.logical_and(
                    jnp.all(jnp.isfinite(u_candidate), axis=1),
                    jnp.logical_and(
                        jnp.isfinite(candidate_br_norm),
                        jnp.logical_not(diverged),
                    ),
                ),
            )
            invalid = jnp.logical_and(depth_active, jnp.logical_not(valid))
            progress = jnp.logical_and(valid, candidate_br_norm < best_br_norm)
            exhausted = jnp.logical_and(
                valid,
                jnp.logical_and(jnp.logical_not(progress), attempt_limit == depth),
            )
            terminal = jnp.logical_or(invalid, jnp.logical_or(progress, exhausted))

            terminal_v = _as_jax_array(jnp.where(valid[:, None], u_candidate, v))
            terminal_errors = _batched_compute_value_error(
                terminal_v,
                v_true,
                norm_name=value_norm,
                normalized=value_normalized,
            )
            terminal_slot = total_queries + depth
            for local_index in range(batch_size):
                if bool(terminal[local_index]):
                    errors = errors.at[local_index, int(terminal_slot[local_index])].set(
                        terminal_errors[local_index]
                    )

            accept_terminal = jnp.logical_and(terminal, valid)
            next_v = _as_jax_array(jnp.where(accept_terminal[:, None], u_candidate, next_v))
            next_br = _as_jax_array(jnp.where(accept_terminal[:, None], new_br, next_br))
            next_current_br_norm = _as_jax_array(jnp.where(accept_terminal, candidate_br_norm, next_current_br_norm))
            next_best_br_norm = _as_jax_array(jnp.where(progress, candidate_br_norm, next_best_br_norm))
            next_h = _as_jax_array(jnp.where(
                progress,
                h,
                jnp.where(terminal, jnp.maximum(1.0, decay_factor_arr * h), next_h),
            ))
            depth_cost = _as_jax_array(jnp.where(terminal, depth, depth_cost))

            continue_depth = _as_jax_array(jnp.logical_and(depth_active, jnp.logical_not(terminal)))
            u = _as_jax_array(jnp.where(continue_depth[:, None], u_candidate, u))
            u_br = _as_jax_array(jnp.where(continue_depth[:, None], new_br, u_br))

        total_queries = total_queries + depth_cost
        v = next_v
        br = next_br
        current_br_norm = next_current_br_norm
        best_br_norm = next_best_br_norm
        h = next_h

    errors = _forward_fill_error_trace(errors)
    queries = jnp.broadcast_to(
        jnp.arange(max_queries + 1, dtype=jnp.int32),
        errors.shape,
    )
    return queries, errors


def _batched_dd_waterfall_pe_hit_times(
    p_pi: jnp.ndarray,
    p_hat_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    threshold: float,
    value_norm: str,
    value_normalized: bool,
    h_init: jnp.ndarray,
    m_max: jnp.ndarray,
    decay_factor: jnp.ndarray,
) -> List[int]:
    _, errors = _batched_dd_waterfall_pe_common(
        p_pi,
        p_hat_pi,
        r_pi,
        gamma,
        v_true,
        max_queries=max_queries,
        value_norm=value_norm,
        value_normalized=value_normalized,
        h_init=h_init,
        m_max=m_max,
        decay_factor=decay_factor,
    )
    return _batched_hit_times_from_errors(errors, threshold=threshold, max_queries=max_queries)


def _batched_dd_waterfall_pe_traces(
    p_pi: jnp.ndarray,
    p_hat_pi: jnp.ndarray,
    r_pi: jnp.ndarray,
    gamma: jnp.ndarray,
    v_true: jnp.ndarray,
    *,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
    h_init: jnp.ndarray,
    m_max: jnp.ndarray,
    decay_factor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _batched_dd_waterfall_pe_common(
        p_pi,
        p_hat_pi,
        r_pi,
        gamma,
        v_true,
        max_queries=max_queries,
        value_norm=value_norm,
        value_normalized=value_normalized,
        h_init=h_init,
        m_max=m_max,
        decay_factor=decay_factor,
    )


def _build_pe_trace(
    queries: List[Any],
    errors: List[Any],
    *,
    threshold: float,
    value_norm: str,
    value_normalized: bool,
    max_queries: int,
) -> PETrace:
    dense_queries, dense_errors = _densify_pe_trace(queries, errors, max_queries=max_queries)
    return PETrace(
        queries=dense_queries,
        errors=dense_errors,
        hit_time=_get_hit_time(dense_queries, dense_errors, threshold, max_queries),
        threshold=threshold,
        value_norm=value_norm,
        value_normalized=value_normalized,
    )


def _densify_pe_trace(
    queries: List[Any],
    errors: List[Any],
    *,
    max_queries: int,
) -> tuple[List[int], List[float]]:
    dense_queries = list(range(max_queries + 1))
    dense_errors: list[float] = [math.nan] * (max_queries + 1)
    last_error = math.nan
    for raw_query, raw_error in zip(queries, errors):
        query = int(raw_query)
        error = float(raw_error)
        if 0 <= query <= max_queries:
            dense_errors[query] = error
            last_error = error
    for index in range(max_queries + 1):
        if not math.isfinite(dense_errors[index]):
            dense_errors[index] = last_error
        else:
            last_error = dense_errors[index]
    if not math.isfinite(dense_errors[0]):
        dense_errors[0] = float(errors[0]) if errors else math.nan
    return dense_queries, dense_errors


def run_pe_algorithms_with_optional_traces(
    mdp: MDP,
    p_hat: Any,
    algo_configs: Dict[str, Dict[str, Any]],
    max_queries: int,
    threshold: float,
    model_norm: str,
    value_norm: str,
    value_normalized: bool,
    *,
    emit_traces: bool = True,
    target_bundle: LoadedDeterministicTarget | None = None,
) -> tuple[Dict[str, int], Dict[str, PETrace]]:
    results: Dict[str, int] = {}
    traces: Dict[str, PETrace] = {}

    try:
        _, p_pi, p_hat_pi, r_pi, v_true = _prepare_pe_target_inputs(
            mdp,
            p_hat,
            target_bundle=target_bundle,
        )

        for name, params in algo_configs.items():
            base_name = _pe_algorithm_family(name)

            try:
                if base_name == "vi":
                    q, e = vi_pe(
                        p_pi,
                        r_pi,
                        mdp.gamma,
                        v_true,
                        max_queries,
                        value_norm=value_norm,
                        value_normalized=value_normalized,
                    )

                elif base_name == "osvi":
                    q, e = osvi_pe(
                        p_pi,
                        p_hat_pi,
                        r_pi,
                        mdp.gamma,
                        v_true,
                        max_queries,
                        value_norm=value_norm,
                        value_normalized=value_normalized,
                    )

                elif base_name == "besp":
                    params_besp = pick_besp_rho_half(
                        p_pi, p_hat_pi, mdp.gamma, norm_name=model_norm
                    )
                    m = params.get("m", params_besp.m)
                    h = params.get("h", params_besp.h)
                    q, e = besp_pe(
                        p_pi,
                        p_hat_pi,
                        r_pi,
                        mdp.gamma,
                        v_true,
                        m,
                        h,
                        max_queries,
                        value_norm=value_norm,
                        value_normalized=value_normalized,
                    )

                elif base_name == "shadow":
                    target_rho = params.get("target_rho", 0.95)
                    q, e = shadow_besp_pe(
                        p_pi,
                        p_hat_pi,
                        r_pi,
                        mdp.gamma,
                        v_true,
                        max_queries=max_queries,
                        value_norm=value_norm,
                        value_normalized=value_normalized,
                        target_rho=target_rho,
                    )

                elif base_name == "abesp":
                    target_eps = params.get("target_epsilon", 1e-6)
                    q, e = abesp_pe(
                        p_pi,
                        p_hat_pi,
                        r_pi,
                        mdp.gamma,
                        v_true,
                        target_epsilon=target_eps,
                        max_total_queries=max_queries,
                        value_norm=value_norm,
                        value_normalized=value_normalized,
                        model_norm=model_norm,
                    )
                elif base_name == "waterfall":
                    cfg = _build_waterfall_config(params)
                    q, e = waterfall_besp_pe(
                        p_pi,
                        p_hat_pi,
                        r_pi,
                        mdp.gamma,
                        v_true,
                        max_queries=max_queries,
                        value_norm=value_norm,
                        value_normalized=value_normalized,
                        h_init=float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)),
                        h_min=float(cfg.get("h_min", _WATERFALL_H_MIN_DEFAULT)),
                        stall_patience=int(
                            cfg.get("stall_patience", _WATERFALL_STALL_PATIENCE_DEFAULT)
                        ),
                        decay_factor=float(
                            cfg.get("decay_factor", _WATERFALL_DECAY_FACTOR_DEFAULT)
                        ),
                    )
                elif base_name == "geo_waterfall":
                    cfg = _build_geo_waterfall_pe_config(params)
                    q, e = geo_waterfall_besp_pe(
                        p_pi,
                        p_hat_pi,
                        r_pi,
                        mdp.gamma,
                        v_true,
                        max_queries=max_queries,
                        value_norm=value_norm,
                        value_normalized=value_normalized,
                        h_init=float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)),
                        alpha=float(cfg.get("alpha", _GEO_WATERFALL_ALPHA_DEFAULT)),
                    )
                elif base_name == "dd_waterfall":
                    cfg = _build_dd_waterfall_pe_config(params)
                    q, e = dd_waterfall_besp_pe(
                        p_pi,
                        p_hat_pi,
                        r_pi,
                        mdp.gamma,
                        v_true,
                        max_queries=max_queries,
                        value_norm=value_norm,
                        value_normalized=value_normalized,
                        h_init=float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)),
                        m_max=int(cfg.get("m_max", _DD_WATERFALL_M_MAX_DEFAULT)),
                        decay_factor=float(
                            cfg.get("decay_factor", _WATERFALL_DECAY_FACTOR_DEFAULT)
                        ),
                    )
                else:
                    logger.warning("Unknown PE algorithm %s", name)
                    results[name] = max_queries + 100
                    continue

                results[name] = _get_hit_time(q, e, threshold, max_queries)
                if emit_traces:
                    traces[name] = _build_pe_trace(
                        q,
                        e,
                        threshold=threshold,
                        value_norm=value_norm,
                        value_normalized=value_normalized,
                        max_queries=max_queries,
                    )
            except ValueError:
                raise
            except Exception:
                results[name] = max_queries + 100

    except ValueError:
        raise
    except Exception:
        for name in algo_configs:
            results[name] = max_queries + 100

    return results, traces


def _batched_control_residual_norms(
    v: jnp.ndarray,
    p: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
) -> jnp.ndarray:
    q = r + gamma[:, None, None] * jnp.einsum("bsak,bk->bsa", p, v)
    t_v = jnp.max(q, axis=2)
    return jnp.max(jnp.abs(t_v - v), axis=1)


def _batched_control_residual_sweeps(
    v: jnp.ndarray,
    p: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    true_term = jnp.einsum("bsak,bk->bsa", p, v)
    q = r + gamma[:, None, None] * true_term
    t_v = jnp.max(q, axis=2)
    residuals = jnp.max(jnp.abs(t_v - v), axis=1)
    return residuals, true_term


def _batched_vi_control_hit_times(
    p: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    effect_threshold: jnp.ndarray,
    *,
    max_queries: int,
) -> List[int]:
    v = jnp.zeros((p.shape[0], p.shape[1]), dtype=jnp.float64)
    hit_times = jnp.full((p.shape[0],), max_queries + 1, dtype=jnp.int32)

    for step in range(1, max_queries + 1):
        residuals = _batched_control_residual_norms(v, p, r, gamma)
        new_hits = jnp.logical_and(hit_times == max_queries + 1, residuals <= effect_threshold)
        hit_times = jnp.where(new_hits, step, hit_times)
        if not bool(jnp.any(hit_times == max_queries + 1)):
            break
        q = r + gamma[:, None, None] * jnp.einsum("bsak,bk->bsa", p, v)
        v = jnp.max(q, axis=2)

    return [int(value) for value in hit_times.tolist()]


def _batched_take_policy_rows_3d(tensor: jnp.ndarray, policy: jnp.ndarray) -> jnp.ndarray:
    return jnp.take_along_axis(
        tensor,
        jnp.broadcast_to(policy[:, :, None, None], (tensor.shape[0], tensor.shape[1], 1, tensor.shape[3])),
        axis=2,
    ).squeeze(axis=2)


def _batched_take_policy_rows_2d(tensor: jnp.ndarray, policy: jnp.ndarray) -> jnp.ndarray:
    return jnp.take_along_axis(tensor, policy[:, :, None], axis=2).squeeze(axis=2)


def _batched_evaluate_policy_values(
    p: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    policy: jnp.ndarray,
) -> jnp.ndarray:
    batch_size, n_states, _, _ = p.shape
    eye = jnp.broadcast_to(jnp.eye(n_states, dtype=jnp.float64), (batch_size, n_states, n_states))
    p_pi = _batched_take_policy_rows_3d(p, policy)
    r_pi = _batched_take_policy_rows_2d(r, policy)
    system = eye - gamma[:, None, None] * p_pi
    return jnp.linalg.solve(system, r_pi[..., None])[..., 0]


def _batched_value_errors(
    values: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    norm_name: str,
    normalized: bool,
) -> jnp.ndarray:
    diff = jnp.asarray(values, dtype=jnp.float64) - jnp.asarray(targets, dtype=jnp.float64)
    norm_key = norm_name.upper()
    if norm_key == "L1":
        numer = jnp.sum(jnp.abs(diff), axis=1)
        denom = jnp.sum(jnp.abs(targets), axis=1)
    elif norm_key == "L2":
        numer = jnp.linalg.norm(diff, axis=1)
        denom = jnp.linalg.norm(targets, axis=1)
    elif norm_key in {"LINF", "INF", "INFINITY"}:
        numer = jnp.max(jnp.abs(diff), axis=1)
        denom = jnp.max(jnp.abs(targets), axis=1)
    else:
        raise ValueError(f"Unsupported value norm {norm_name!r}")
    if not normalized:
        return numer
    return numer / (denom + 1e-12)


def _rowwise_policy_value_errors(
    mdp: MDP,
    policy_trace: jnp.ndarray,
    target_value: jnp.ndarray,
    *,
    value_norm: str,
    value_normalized: bool,
) -> list[float]:
    errors: list[float] = []
    for policy in np.asarray(policy_trace, dtype=np.int32):
        v_pi = evaluate_control_policy_value(mdp, policy)
        errors.append(
            compute_value_error(
                v_pi,
                target_value,
                norm_name=value_norm,
                normalized=value_normalized,
            )
        )
    return errors


def _pack_sparse_control_trace_rows(
    query_rows: list[list[int]],
    residual_rows: list[list[float]],
    value_error_rows: list[list[float]],
    policy_value_error_rows: list[list[float]],
    policy_rows: list[list[list[int]]],
    *,
    n_states: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size = len(query_rows)
    width = max((len(row) for row in query_rows), default=0)
    queries = np.zeros((batch_size, width), dtype=np.int32)
    residuals = np.full((batch_size, width), np.inf, dtype=np.float64)
    value_errors = np.full((batch_size, width), np.inf, dtype=np.float64)
    policy_value_errors = np.full((batch_size, width), np.inf, dtype=np.float64)
    policy_trace = np.zeros((batch_size, width, n_states), dtype=np.int32)

    for row_index in range(batch_size):
        length = len(query_rows[row_index])
        if length == 0:
            continue
        queries[row_index, :length] = np.asarray(query_rows[row_index], dtype=np.int32)
        residuals[row_index, :length] = np.asarray(residual_rows[row_index], dtype=np.float64)
        value_errors[row_index, :length] = np.asarray(
            value_error_rows[row_index], dtype=np.float64
        )
        policy_value_errors[row_index, :length] = np.asarray(
            policy_value_error_rows[row_index], dtype=np.float64
        )
        policy_trace[row_index, :length, :] = np.asarray(
            policy_rows[row_index], dtype=np.int32
        )

    return (
        jnp.asarray(queries, dtype=jnp.int32),
        jnp.asarray(residuals, dtype=jnp.float64),
        jnp.asarray(value_errors, dtype=jnp.float64),
        jnp.asarray(policy_value_errors, dtype=jnp.float64),
        jnp.asarray(policy_trace, dtype=jnp.int32),
    )


def _batched_control_hit_times_from_policy_value_errors(
    errors: jnp.ndarray,
    *,
    threshold: float,
    max_queries: int,
) -> List[int]:
    hit_times: list[int] = []
    for row in np.asarray(errors, dtype=np.float64):
        hit = max_queries + 1
        for index, err in enumerate(row, start=1):
            if math.isfinite(float(err)) and float(err) <= threshold:
                hit = index
                break
        hit_times.append(hit)
    return hit_times


def _batched_control_hit_times_from_query_aligned_errors(
    queries: jnp.ndarray,
    errors: jnp.ndarray,
    *,
    threshold: float,
    max_queries: int,
) -> List[int]:
    hit_times: list[int] = []
    query_rows = np.asarray(queries, dtype=np.int32)
    error_rows = np.asarray(errors, dtype=np.float64)
    for query_row, error_row in zip(query_rows, error_rows, strict=True):
        hit = max_queries + 1
        for query_value, err in zip(query_row, error_row, strict=True):
            if int(query_value) <= 0:
                continue
            if math.isfinite(float(err)) and float(err) <= threshold:
                hit = int(query_value)
                break
        hit_times.append(hit)
    return hit_times


def _batched_control_hit_times_from_query_aligned_thresholds(
    queries: jnp.ndarray,
    errors: jnp.ndarray,
    *,
    effect_threshold: jnp.ndarray,
    max_queries: int,
) -> List[int]:
    hit_times: list[int] = []
    query_rows = np.asarray(queries, dtype=np.int32)
    error_rows = np.asarray(errors, dtype=np.float64)
    threshold_rows = np.asarray(effect_threshold, dtype=np.float64)
    for query_row, error_row, threshold in zip(query_rows, error_rows, threshold_rows, strict=True):
        hit = max_queries + 1
        for query_value, err in zip(query_row, error_row, strict=True):
            if int(query_value) <= 0:
                continue
            if math.isfinite(float(err)) and float(err) <= float(threshold):
                hit = int(query_value)
                break
        hit_times.append(hit)
    return hit_times


def _batched_control_hit_times_from_residuals(
    residuals: jnp.ndarray,
    *,
    effect_threshold: jnp.ndarray,
    max_queries: int,
) -> List[int]:
    hit_times: list[int] = []
    residual_rows = np.asarray(residuals, dtype=np.float64)
    threshold_rows = np.asarray(effect_threshold, dtype=np.float64)
    for row, threshold in zip(residual_rows, threshold_rows, strict=True):
        hit = max_queries + 1
        for index, resid in enumerate(row, start=1):
            if math.isfinite(float(resid)) and float(resid) <= float(threshold):
                hit = index
                break
        hit_times.append(hit)
    return hit_times


def _batched_vi_control_traces(
    p: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    target_v: jnp.ndarray,
    *,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size, n_states, _, _ = p.shape
    v = jnp.zeros((batch_size, n_states), dtype=jnp.float64)
    residuals_trace: list[jnp.ndarray] = []
    value_errors_trace: list[jnp.ndarray] = []
    policy_value_errors_trace: list[jnp.ndarray] = []
    policy_trace: list[jnp.ndarray] = []

    for _ in range(max_queries):
        residuals = _batched_control_residual_norms(v, p, r, gamma)
        q = r + gamma[:, None, None] * jnp.einsum("bsak,bk->bsa", p, v)
        policy = jnp.argmax(q, axis=2).astype(jnp.int32)
        v_pi = _batched_evaluate_policy_values(p, r, gamma, policy)
        residuals_trace.append(residuals)
        value_errors_trace.append(
            _batched_value_errors(v, target_v, norm_name=value_norm, normalized=value_normalized)
        )
        policy_value_errors_trace.append(
            _batched_value_errors(v_pi, target_v, norm_name=value_norm, normalized=value_normalized)
        )
        policy_trace.append(policy)
        v = jnp.max(q, axis=2)

    queries = jnp.broadcast_to(
        jnp.arange(1, max_queries + 1, dtype=jnp.int32),
        (batch_size, max_queries),
    )
    return (
        queries,
        jnp.stack(residuals_trace, axis=1),
        jnp.stack(value_errors_trace, axis=1),
        jnp.stack(policy_value_errors_trace, axis=1),
        jnp.stack(policy_trace, axis=1),
    )


def _batched_osvi_control_traces(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    target_v: jnp.ndarray,
    *,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
    surrogate_policy_iteration_steps: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size, n_states, _, _ = p.shape
    v = jnp.zeros((batch_size, n_states), dtype=jnp.float64)
    residuals_trace: list[jnp.ndarray] = []
    value_errors_trace: list[jnp.ndarray] = []
    policy_value_errors_trace: list[jnp.ndarray] = []
    policy_trace: list[jnp.ndarray] = []
    current_policy: jnp.ndarray | None = None

    for _ in range(max_queries):
        residuals = _batched_control_residual_norms(v, p, r, gamma)
        expected_true = jnp.einsum("bsak,bk->bsa", p, v)
        expected_shadow = jnp.einsum("bsak,bk->bsa", p_hat, v)
        r_tilde = r + gamma[:, None, None] * (expected_true - expected_shadow)
        v_new, policy = _batched_solve_surrogate_control_with_policy(
            p_hat,
            r_tilde,
            gamma,
            init_policy=current_policy,
            surrogate_policy_iteration_steps=surrogate_policy_iteration_steps,
        )
        v_pi = _batched_evaluate_policy_values(p, r, gamma, policy)
        residuals_trace.append(residuals)
        value_errors_trace.append(
            _batched_value_errors(v, target_v, norm_name=value_norm, normalized=value_normalized)
        )
        policy_value_errors_trace.append(
            _batched_value_errors(v_pi, target_v, norm_name=value_norm, normalized=value_normalized)
        )
        policy_trace.append(policy)
        v = v_new
        current_policy = policy

    queries = jnp.broadcast_to(
        jnp.arange(1, max_queries + 1, dtype=jnp.int32),
        (batch_size, max_queries),
    )
    return (
        queries,
        jnp.stack(residuals_trace, axis=1),
        jnp.stack(value_errors_trace, axis=1),
        jnp.stack(policy_value_errors_trace, axis=1),
        jnp.stack(policy_trace, axis=1),
    )


def _batched_solve_surrogate_control(
    p_hat: jnp.ndarray,
    r_tilde: jnp.ndarray,
    gamma: jnp.ndarray,
    *,
    max_iter: int = 200,
    surrogate_policy_iteration_steps: int | None = None,
) -> jnp.ndarray:
    v, _ = _batched_solve_surrogate_control_with_policy(
        p_hat,
        r_tilde,
        gamma,
        max_iter=max_iter,
        surrogate_policy_iteration_steps=surrogate_policy_iteration_steps,
    )
    return v


def _batched_solve_surrogate_control_with_policy(
    p_hat: jnp.ndarray,
    r_tilde: jnp.ndarray,
    gamma: jnp.ndarray,
    *,
    init_policy: jnp.ndarray | None = None,
    max_iter: int = 200,
    surrogate_policy_iteration_steps: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, n_states, _, _ = p_hat.shape
    eye = jnp.broadcast_to(jnp.eye(n_states, dtype=jnp.float64), (batch_size, n_states, n_states))
    if init_policy is None:
        policy = jnp.argmax(r_tilde, axis=2).astype(jnp.int32)
    else:
        policy = jnp.asarray(init_policy, dtype=jnp.int32)
    v = jnp.zeros((batch_size, n_states), dtype=jnp.float64)
    fixed_steps = (
        None
        if surrogate_policy_iteration_steps is None
        else max(int(surrogate_policy_iteration_steps), 1)
    )
    loop_count = max_iter if fixed_steps is None else fixed_steps

    for _ in range(loop_count):
        p_pi = _batched_take_policy_rows_3d(p_hat, policy)
        r_pi = _batched_take_policy_rows_2d(r_tilde, policy)
        a = eye - gamma[:, None, None] * p_pi
        v = jnp.linalg.solve(a, r_pi[..., None])[..., 0]
        q = r_tilde + gamma[:, None, None] * jnp.einsum("bsak,bk->bsa", p_hat, v)
        new_policy = jnp.argmax(q, axis=2).astype(jnp.int32)
        if fixed_steps is None and bool(jnp.all(new_policy == policy)):
            break
        policy = new_policy
    return v, policy


def _batched_isp_control_step_with_policy(
    v_k: jnp.ndarray,
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    h: jnp.ndarray,
    j_steps: jnp.ndarray,
    *,
    initial_true_term: jnp.ndarray | None = None,
    init_policy: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    u = jnp.array(v_k, copy=True)
    policy = init_policy
    j_step_values = [int(value) for value in jnp.asarray(j_steps, dtype=jnp.int32).reshape(-1).tolist()]
    max_j = max(j_step_values, default=0)
    if max_j <= 0:
        n_states = p.shape[1]
        empty_policy = jnp.zeros((p.shape[0], n_states), dtype=jnp.int32)
        return u, empty_policy if policy is None else policy

    h_expanded = h[:, None, None]
    gamma_expanded = gamma[:, None, None]
    gamma_eff = (h * gamma) / (1.0 + h)

    for j_index in range(max_j):
        active = jnp.asarray(j_index < j_steps)
        if not bool(jnp.any(active)):
            break
        if j_index == 0 and initial_true_term is not None:
            true_term = initial_true_term
        else:
            true_term = jnp.einsum("bsak,bk->bsa", p, u)
        shadow_term = jnp.einsum("bsak,bk->bsa", p_hat, u)
        r_tilde = (
            (1.0 / (1.0 + h_expanded)) * v_k[:, :, None]
            + (h_expanded / (1.0 + h_expanded)) * r
            + ((h_expanded * gamma_expanded) / (1.0 + h_expanded)) * (true_term - shadow_term)
        )
        u_next, policy_next = _batched_solve_surrogate_control_with_policy(
            p_hat,
            r_tilde,
            gamma_eff,
            init_policy=policy,
        )
        u = jnp.where(active[:, None], u_next, u)
        if policy is None:
            policy = policy_next
        else:
            policy = jnp.where(active[:, None], policy_next, policy)

    if policy is None:
        policy = jnp.zeros((p.shape[0], p.shape[1]), dtype=jnp.int32)
    return u, policy


def _batched_isp_control_step(
    v_k: jnp.ndarray,
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    h: jnp.ndarray,
    j_steps: jnp.ndarray,
    *,
    initial_true_term: jnp.ndarray | None = None,
) -> jnp.ndarray:
    u, _ = _batched_isp_control_step_with_policy(
        v_k,
        p,
        p_hat,
        r,
        gamma,
        h,
        j_steps,
        initial_true_term=initial_true_term,
    )
    return u


def _batched_isp_control_hit_times(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    effect_threshold: jnp.ndarray,
    *,
    h: jnp.ndarray,
    j_steps: jnp.ndarray,
    max_queries: int,
) -> List[int]:
    v = jnp.zeros((p.shape[0], p.shape[1]), dtype=jnp.float64)
    total_queries = jnp.zeros((p.shape[0],), dtype=jnp.int32)
    hit_times = jnp.full((p.shape[0],), max_queries + 1, dtype=jnp.int32)
    additional_cost = jnp.maximum(j_steps - 1, 0)

    while True:
        active = _as_jax_array(jnp.logical_and(hit_times == max_queries + 1, total_queries < max_queries))
        if not bool(jnp.any(active)):
            break

        residuals, true_term = _batched_control_residual_sweeps(v, p, r, gamma)
        # The residual sweep at the current iterate is reused as the j=0 ISP sweep if we continue.
        total_queries = jnp.where(active, total_queries + 1, total_queries)
        new_hits = jnp.logical_and(active, residuals <= effect_threshold)
        hit_times = jnp.where(new_hits, total_queries, hit_times)

        continue_mask = jnp.logical_and(active, residuals > effect_threshold)
        can_step = jnp.logical_and(continue_mask, total_queries + additional_cost <= max_queries)
        exhausted = jnp.logical_and(continue_mask, jnp.logical_not(can_step))
        total_queries = jnp.where(exhausted, max_queries, total_queries)

        if not bool(jnp.any(can_step)):
            break

        v_next = _batched_isp_control_step(
            v,
            p,
            p_hat,
            r,
            gamma,
            h,
            j_steps,
            initial_true_term=true_term,
        )
        total_queries = jnp.where(can_step, total_queries + additional_cost, total_queries)
        v = jnp.where(can_step[:, None], v_next, v)

    return [int(value) for value in hit_times.tolist()]


def _batched_isp_control_traces(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    target_v: jnp.ndarray,
    *,
    h: jnp.ndarray,
    j_steps: jnp.ndarray,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size, n_states, _, _ = p.shape
    v = jnp.zeros((batch_size, n_states), dtype=jnp.float64)
    total_queries = jnp.zeros((batch_size,), dtype=jnp.int32)
    step_counts = jnp.zeros((batch_size,), dtype=jnp.int32)
    current_policy: jnp.ndarray | None = None

    queries = jnp.zeros((batch_size, max_queries), dtype=jnp.int32)
    residuals_trace = jnp.full((batch_size, max_queries), jnp.inf, dtype=jnp.float64)
    value_errors_trace = jnp.full((batch_size, max_queries), jnp.inf, dtype=jnp.float64)
    policy_value_errors_trace = jnp.full((batch_size, max_queries), jnp.inf, dtype=jnp.float64)
    policy_trace = jnp.zeros((batch_size, max_queries, n_states), dtype=jnp.int32)
    row_indices = jnp.arange(batch_size, dtype=jnp.int32)

    while True:
        active = total_queries + j_steps <= max_queries
        if not bool(jnp.any(active)):
            break

        residuals, true_term = _batched_control_residual_sweeps(v, p, r, gamma)
        value_errors = _batched_value_errors(
            v,
            target_v,
            norm_name=value_norm,
            normalized=value_normalized,
        )
        v_next, policy_next = _batched_isp_control_step_with_policy(
            v,
            p,
            p_hat,
            r,
            gamma,
            h,
            j_steps,
            initial_true_term=true_term,
            init_policy=current_policy,
        )
        v_pi = _batched_evaluate_policy_values(p, r, gamma, policy_next)
        policy_value_errors = _batched_value_errors(
            v_pi,
            target_v,
            norm_name=value_norm,
            normalized=value_normalized,
        )

        next_total_queries = total_queries + j_steps
        slot = step_counts
        current_query_rows = queries[row_indices, slot]
        current_residual_rows = residuals_trace[row_indices, slot]
        current_value_rows = value_errors_trace[row_indices, slot]
        current_policy_value_rows = policy_value_errors_trace[row_indices, slot]
        current_policy_rows = policy_trace[row_indices, slot, :]

        queries = queries.at[row_indices, slot].set(
            jnp.where(active, next_total_queries, current_query_rows)
        )
        residuals_trace = residuals_trace.at[row_indices, slot].set(
            jnp.where(active, residuals, current_residual_rows)
        )
        value_errors_trace = value_errors_trace.at[row_indices, slot].set(
            jnp.where(active, value_errors, current_value_rows)
        )
        policy_value_errors_trace = policy_value_errors_trace.at[row_indices, slot].set(
            jnp.where(active, policy_value_errors, current_policy_value_rows)
        )
        policy_trace = policy_trace.at[row_indices, slot, :].set(
            jnp.where(active[:, None], policy_next, current_policy_rows)
        )

        total_queries = jnp.where(active, next_total_queries, total_queries)
        step_counts = jnp.where(active, step_counts + 1, step_counts)
        v = jnp.where(active[:, None], v_next, v)
        if current_policy is None:
            current_policy = policy_next
        else:
            current_policy = jnp.where(active[:, None], policy_next, current_policy)

    return (
        queries,
        residuals_trace,
        value_errors_trace,
        policy_value_errors_trace,
        policy_trace,
    )


def _batched_aisp_control_traces(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    target_v: jnp.ndarray,
    *,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
    target_epsilon: Any = _AISP_TARGET_EPSILON_DEFAULT,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    query_rows: list[list[int]] = []
    residual_rows: list[list[float]] = []
    value_error_rows: list[list[float]] = []
    policy_value_error_rows: list[list[float]] = []
    policy_rows: list[list[list[int]]] = []

    target_eps_arr = jnp.broadcast_to(
        jnp.asarray(target_epsilon, dtype=jnp.float64),
        (p.shape[0],),
    )

    for row_index in range(p.shape[0]):
        mdp = MDP(P=p[row_index], R=r[row_index], gamma=float(gamma[row_index]))
        aisp = AISP_Control(mdp, p_hat[row_index])
        queries, residuals, value_trace, policy_trace = aisp.run_with_policy_trace(
            target_epsilon=float(target_eps_arr[row_index]),
            max_avps=max_queries,
        )
        target_value = jnp.asarray(target_v[row_index], dtype=jnp.float64)
        value_errors = [
            compute_value_error(
                value,
                target_value,
                norm_name=value_norm,
                normalized=value_normalized,
            )
            for value in value_trace
        ]
        policy_value_errors = _rowwise_policy_value_errors(
            mdp,
            policy_trace,
            target_value,
            value_norm=value_norm,
            value_normalized=value_normalized,
        )
        query_rows.append([int(value) for value in queries])
        residual_rows.append([float(value) for value in residuals])
        value_error_rows.append([float(value) for value in value_errors])
        policy_value_error_rows.append([float(value) for value in policy_value_errors])
        policy_rows.append(np.asarray(policy_trace, dtype=np.int32).tolist())

    return _pack_sparse_control_trace_rows(
        query_rows,
        residual_rows,
        value_error_rows,
        policy_value_error_rows,
        policy_rows,
        n_states=p.shape[1],
    )


def _batched_shadow_control_step(
    v_k: jnp.ndarray,
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    h: jnp.ndarray,
    m_cont: jnp.ndarray,
    optimism: jnp.ndarray,
    dynamic_j: jnp.ndarray,
    fixed_j: jnp.ndarray,
    *,
    init_policy: jnp.ndarray | None = None,
    alpha: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    q_true = r + gamma[:, None, None] * jnp.einsum("bsak,bk->bsa", p, v_k)
    q_shadow = r + gamma[:, None, None] * jnp.einsum("bsak,bk->bsa", p_hat, v_k)
    t_v_true = jnp.max(q_true, axis=2)
    t_v_shadow = jnp.max(q_shadow, axis=2)
    br_true = t_v_true - v_k
    br_shadow = t_v_shadow - v_k
    br_true_norm = jnp.max(jnp.abs(br_true), axis=1)

    diff = br_shadow - br_true
    numer = jnp.max(jnp.abs(diff), axis=1)
    denom = jnp.max(jnp.abs(v_k), axis=1) + 1e-6
    e_est = numer / denom
    delta = 1.0 - gamma
    e_used = e_est / optimism
    denom_h = 2.0 * e_used - delta
    h_target = jnp.where(denom_h < 1e-4, 1e6, 1.0 / denom_h)
    next_h = jnp.clip((1.0 - alpha) * h + alpha * h_target, 1e-9, 1e6)

    dynamic_target = jnp.log2(jnp.maximum(1.0, 12.0 * (e_used / jnp.maximum(delta, 1e-12)) - 6.0))
    updated_m_cont = (1.0 - alpha) * m_cont + alpha * dynamic_target
    next_m_cont = jnp.where(fixed_j >= 0, fixed_j.astype(jnp.float64), updated_m_cont)
    j_values = jnp.where(
        fixed_j >= 0,
        fixed_j,
        jnp.where(dynamic_j, jnp.ceil(next_m_cont).astype(jnp.int32), jnp.zeros_like(fixed_j)),
    )
    j_values = jnp.maximum(j_values, 1)

    diverged = jnp.logical_or(br_true_norm > 1e7, jnp.logical_not(jnp.all(jnp.isfinite(v_k), axis=1)))
    next_h = jnp.where(diverged, 1e-4, next_h)
    v_base = jnp.where(diverged[:, None], jnp.zeros_like(v_k), v_k)
    q_true_zero = r + gamma[:, None, None] * jnp.einsum("bsak,bk->bsa", p, jnp.zeros_like(v_k))
    q_true_for_step = jnp.where(diverged[:, None, None], q_true_zero, q_true)
    br_true_norm = jnp.where(
        diverged,
        jnp.max(jnp.abs(jnp.max(q_true_zero, axis=2)), axis=1),
        br_true_norm,
    )

    u = jnp.array(v_base, copy=True)
    policy = init_policy
    max_j = max([int(value) for value in jnp.asarray(j_values, dtype=jnp.int32).reshape(-1).tolist()], default=0)
    next_gamma = (next_h * gamma) / (1.0 + next_h)
    next_h_expanded = next_h[:, None, None]

    for j_index in range(max_j):
        active = jnp.asarray(j_index < j_values)
        if not bool(jnp.any(active)):
            break
        if j_index == 0:
            r_tilde = (
                (1.0 / (1.0 + next_h_expanded)) * v_base[:, :, None]
                + (next_h_expanded / (1.0 + next_h_expanded)) * r
                + (next_h_expanded / (1.0 + next_h_expanded)) * (q_true_for_step - q_shadow)
            )
        else:
            true_term = jnp.einsum("bsak,bk->bsa", p, u)
            shadow_term = jnp.einsum("bsak,bk->bsa", p_hat, u)
            gamma_expanded = gamma[:, None, None]
            r_tilde = (
                (1.0 / (1.0 + next_h_expanded)) * v_base[:, :, None]
                + (next_h_expanded / (1.0 + next_h_expanded)) * r
                + ((next_h_expanded * gamma_expanded) / (1.0 + next_h_expanded)) * (true_term - shadow_term)
            )
        u_candidate, policy_candidate = _batched_solve_surrogate_control_with_policy(
            p_hat,
            r_tilde,
            next_gamma,
            init_policy=policy,
        )
        u = jnp.where(active[:, None], u_candidate, u)
        if policy is None:
            policy = policy_candidate
        else:
            policy = jnp.where(active[:, None], policy_candidate, policy)

    if policy is None:
        policy = jnp.zeros((p.shape[0], p.shape[1]), dtype=jnp.int32)

    return u, j_values, br_true_norm, next_h, next_m_cont, policy


def _batched_shadow_control_hit_times(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    effect_threshold: jnp.ndarray,
    *,
    seeds: jnp.ndarray,
    optimism: jnp.ndarray,
    dynamic_j: jnp.ndarray,
    fixed_j: jnp.ndarray,
    max_queries: int,
) -> List[int]:
    n_states = p.shape[1]
    seed_values = [int(value) for value in jnp.asarray(seeds, dtype=jnp.int32).reshape(-1).tolist()]
    v = jnp.stack([seeded_shadow_init(n_states, seed) for seed in seed_values], axis=0)
    total_queries = jnp.zeros((p.shape[0],), dtype=jnp.int32)
    hit_times = jnp.full((p.shape[0],), max_queries + 1, dtype=jnp.int32)
    best_resid = jnp.full((p.shape[0],), jnp.inf, dtype=jnp.float64)
    divergent_streak = jnp.zeros((p.shape[0],), dtype=jnp.int32)
    h = jnp.ones((p.shape[0],), dtype=jnp.float64)
    m_cont = jnp.where(fixed_j >= 0, fixed_j.astype(jnp.float64), jnp.zeros((p.shape[0],), dtype=jnp.float64))
    current_policy: jnp.ndarray | None = None

    while True:
        active = jnp.logical_and(hit_times == max_queries + 1, total_queries < max_queries)
        if not bool(jnp.any(active)):
            break

        v_next, cost, resid, h, m_cont, current_policy = _batched_shadow_control_step(
            v,
            p,
            p_hat,
            r,
            gamma,
            h,
            m_cont,
            optimism,
            dynamic_j,
            fixed_j,
            init_policy=current_policy,
        )
        query_marks = total_queries
        new_hits = jnp.logical_and(active, resid <= effect_threshold)
        hit_times = jnp.where(new_hits, query_marks, hit_times)

        improved = resid < best_resid
        diverging = resid > jnp.maximum(1e3, best_resid * 100.0)
        divergent_streak = jnp.where(
            improved,
            0,
            jnp.where(diverging, divergent_streak + 1, 0),
        )
        best_resid = jnp.where(improved, resid, best_resid)

        total_queries = jnp.where(active, total_queries + cost, total_queries)
        halt = jnp.logical_and(active, divergent_streak >= 8)
        total_queries = jnp.where(halt, max_queries, total_queries)
        v = jnp.where(active[:, None], v_next, v)

    return [int(value) for value in hit_times.tolist()]


def _batched_shadow_control_traces(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    target_v: jnp.ndarray,
    *,
    seeds: jnp.ndarray,
    optimism: jnp.ndarray,
    dynamic_j: jnp.ndarray,
    fixed_j: jnp.ndarray,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    query_rows: list[list[int]] = []
    residual_rows: list[list[float]] = []
    value_error_rows: list[list[float]] = []
    policy_value_error_rows: list[list[float]] = []
    policy_rows: list[list[list[int]]] = []

    for row_index in range(p.shape[0]):
        fixed_j_value = int(fixed_j[row_index])
        mdp = MDP(P=p[row_index], R=r[row_index], gamma=float(gamma[row_index]))
        shadow = Shadow_ISP_Control(
            mdp,
            p_hat[row_index],
            optimism=float(optimism[row_index]),
            dynamic_J=bool(dynamic_j[row_index]),
            fixed_J=None if fixed_j_value < 0 else fixed_j_value,
        )
        v_init = seeded_shadow_init(mdp.P.shape[0], int(seeds[row_index]))
        queries, residuals, value_trace, policy_trace = shadow.run_with_policy_trace(
            max_avps=max_queries,
            V_init=v_init,
        )
        target_value = jnp.asarray(target_v[row_index], dtype=jnp.float64)
        value_errors = [
            compute_value_error(
                value,
                target_value,
                norm_name=value_norm,
                normalized=value_normalized,
            )
            for value in value_trace
        ]
        policy_value_errors = _rowwise_policy_value_errors(
            mdp,
            policy_trace,
            target_value,
            value_norm=value_norm,
            value_normalized=value_normalized,
        )
        query_rows.append([int(value) for value in queries])
        residual_rows.append([float(value) for value in residuals])
        value_error_rows.append([float(value) for value in value_errors])
        policy_value_error_rows.append([float(value) for value in policy_value_errors])
        policy_rows.append(np.asarray(policy_trace, dtype=np.int32).tolist())

    return _pack_sparse_control_trace_rows(
        query_rows,
        residual_rows,
        value_error_rows,
        policy_value_error_rows,
        policy_rows,
        n_states=p.shape[1],
    )


def _batched_waterfall_inner_vi(
    v: jnp.ndarray,
    p_hat: jnp.ndarray,
    r_tilde: jnp.ndarray,
    gamma_eff: jnp.ndarray,
    inner_vi_tol: jnp.ndarray,
    inner_vi_max_iters: jnp.ndarray,
) -> jnp.ndarray:
    v_new = jnp.array(v, copy=True)
    max_iters = max(
        [int(value) for value in jnp.asarray(inner_vi_max_iters, dtype=jnp.int32).reshape(-1).tolist()],
        default=0,
    )
    if max_iters <= 0:
        return v_new

    for iter_index in range(max_iters):
        active = jnp.asarray(iter_index < inner_vi_max_iters)
        if not bool(jnp.any(active)):
            break

        q_surr = r_tilde + gamma_eff[:, None, None] * jnp.einsum("bsak,bk->bsa", p_hat, v_new)
        u_new = jnp.max(q_surr, axis=2)
        delta = jnp.max(jnp.abs(u_new - v_new), axis=1)
        should_update = jnp.logical_and(active, delta >= inner_vi_tol)
        if not bool(jnp.any(should_update)):
            break
        v_new = jnp.where(should_update[:, None], u_new, v_new)

    return v_new


def _batched_surrogate_policies_from_values(
    p_hat: jnp.ndarray,
    r_tilde: jnp.ndarray,
    gamma_eff: jnp.ndarray,
    values: jnp.ndarray,
) -> jnp.ndarray:
    q_surr = r_tilde + gamma_eff[:, None, None] * jnp.einsum("bsak,bk->bsa", p_hat, values)
    return jnp.argmax(q_surr, axis=2).astype(jnp.int32)


def _batched_waterfall_control_hit_times(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    effect_threshold: jnp.ndarray,
    *,
    h_init: jnp.ndarray,
    h_min: jnp.ndarray,
    stall_patience: jnp.ndarray,
    decay_factor: jnp.ndarray,
    inner_vi_tol: jnp.ndarray,
    inner_vi_max_iters: jnp.ndarray,
    max_queries: int,
) -> List[int]:
    v = jnp.zeros((p.shape[0], p.shape[1]), dtype=jnp.float64)
    total_queries = jnp.zeros((p.shape[0],), dtype=jnp.int32)
    hit_times = jnp.full((p.shape[0],), max_queries + 1, dtype=jnp.int32)
    h = jnp.asarray(h_init, dtype=jnp.float64)
    h_floor = jnp.asarray(h_min, dtype=jnp.float64)
    stall_patience_arr = jnp.asarray(stall_patience, dtype=jnp.int32)
    decay_factor_arr = jnp.asarray(decay_factor, dtype=jnp.float64)
    current_resid = _batched_control_residual_norms(v, p, r, gamma)
    best_resid = current_resid
    stall_count = jnp.zeros((p.shape[0],), dtype=jnp.int32)
    last_expected_v_true = jnp.einsum("bsak,bk->bsa", p, v)
    v_div_envelope = _WATERFALL_DIVERGENCE_MULT * jnp.maximum(
        jnp.max(jnp.abs(r), axis=(1, 2)) / jnp.maximum(1.0 - gamma, 1e-12),
        1.0,
    )

    while True:
        active = jnp.logical_and(hit_times == max_queries + 1, total_queries < max_queries)
        if not bool(jnp.any(active)):
            break

        gamma_eff = (h * gamma) / (1.0 + h)
        expected_v_hat = jnp.einsum("bsak,bk->bsa", p_hat, v)
        h_expanded = jnp.expand_dims(jnp.expand_dims(h, axis=1), axis=2)
        gamma_expanded = jnp.expand_dims(jnp.expand_dims(gamma, axis=1), axis=2)
        r_tilde = (
            (1.0 / (1.0 + h_expanded)) * jnp.expand_dims(v, axis=2)
            + (h_expanded / (1.0 + h_expanded)) * r
            + ((h_expanded * gamma_expanded) / (1.0 + h_expanded))
            * (last_expected_v_true - expected_v_hat)
        )
        v_candidate = _batched_waterfall_inner_vi(
            v,
            p_hat,
            r_tilde,
            gamma_eff,
            jnp.asarray(inner_vi_tol, dtype=jnp.float64),
            jnp.asarray(inner_vi_max_iters, dtype=jnp.int32),
        )

        diverged = jnp.max(jnp.abs(v_candidate), axis=1) > v_div_envelope
        expected_v_new_true = jnp.einsum("bsak,bk->bsa", p, v_candidate)
        q_new = r + gamma[:, None, None] * expected_v_new_true
        t_v_new = jnp.max(q_new, axis=2)
        resid = jnp.max(jnp.abs(t_v_new - v_candidate), axis=1)
        nonfinite = jnp.logical_or(
            jnp.logical_not(jnp.all(jnp.isfinite(v_candidate), axis=1)),
            jnp.logical_not(jnp.isfinite(resid)),
        )
        failed = jnp.logical_and(active, jnp.logical_or(diverged, nonfinite))
        charged_queries = _as_jax_array(jnp.where(active, total_queries + 1, total_queries))
        next_h_on_fail = jnp.maximum(h_floor, h * decay_factor_arr)
        should_stop_after_fail = jnp.logical_and(failed, next_h_on_fail <= h_floor)
        h = jnp.where(failed, next_h_on_fail, h)
        stall_count = jnp.where(failed, 0, stall_count)

        succeeded = jnp.logical_and(active, jnp.logical_not(failed))
        next_current_resid = jnp.where(succeeded, resid, current_resid)
        new_hits = jnp.logical_and(active, next_current_resid <= effect_threshold)
        hit_times = jnp.where(
            jnp.logical_and(hit_times == max_queries + 1, new_hits),
            charged_queries,
            hit_times,
        )

        improved = resid < best_resid
        best_resid = jnp.where(jnp.logical_and(succeeded, improved), resid, best_resid)
        tentative_stall_count = jnp.where(
            jnp.logical_and(succeeded, improved),
            0,
            jnp.where(succeeded, stall_count + 1, stall_count),
        )
        should_decay_after_success = jnp.logical_and(
            succeeded,
            jnp.logical_and(jnp.logical_not(improved), tentative_stall_count >= stall_patience_arr),
        )
        h = jnp.where(should_decay_after_success, jnp.maximum(h_floor, h * decay_factor_arr), h)
        stall_count = jnp.where(
            jnp.logical_and(succeeded, improved),
            0,
            jnp.where(should_decay_after_success, 0, tentative_stall_count),
        )
        total_queries = jnp.where(should_stop_after_fail, max_queries, charged_queries)
        v = jnp.where(succeeded[:, None], v_candidate, v)
        last_expected_v_true = jnp.where(
            succeeded[:, None, None],
            expected_v_new_true,
            last_expected_v_true,
        )
        current_resid = next_current_resid

    return [int(value) for value in hit_times.tolist()]


def _batched_waterfall_control_traces(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    target_v: jnp.ndarray,
    *,
    h_init: jnp.ndarray,
    h_min: jnp.ndarray,
    stall_patience: jnp.ndarray,
    decay_factor: jnp.ndarray,
    inner_vi_tol: jnp.ndarray,
    inner_vi_max_iters: jnp.ndarray,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size, n_states, _, _ = p.shape
    v = jnp.zeros((batch_size, n_states), dtype=jnp.float64)
    total_queries = jnp.zeros((batch_size,), dtype=jnp.int32)
    alive = jnp.ones((batch_size,), dtype=jnp.bool_)
    h = jnp.asarray(h_init, dtype=jnp.float64)
    h_floor = jnp.asarray(h_min, dtype=jnp.float64)
    stall_patience_arr = jnp.asarray(stall_patience, dtype=jnp.int32)
    decay_factor_arr = jnp.asarray(decay_factor, dtype=jnp.float64)
    current_resid = _batched_control_residual_norms(v, p, r, gamma)
    best_resid = current_resid
    stall_count = jnp.zeros((batch_size,), dtype=jnp.int32)
    last_expected_v_true = jnp.einsum("bsak,bk->bsa", p, v)
    v_div_envelope = _WATERFALL_DIVERGENCE_MULT * jnp.maximum(
        jnp.max(jnp.abs(r), axis=(1, 2)) / jnp.maximum(1.0 - gamma, 1e-12),
        1.0,
    )
    queries = jnp.broadcast_to(
        jnp.arange(1, max_queries + 1, dtype=jnp.int32),
        (batch_size, max_queries),
    )
    residuals_trace = jnp.full((batch_size, max_queries), jnp.inf, dtype=jnp.float64)
    value_errors_trace = jnp.full((batch_size, max_queries), jnp.inf, dtype=jnp.float64)
    policy_value_errors_trace = jnp.full((batch_size, max_queries), jnp.inf, dtype=jnp.float64)
    policy_trace = jnp.zeros((batch_size, max_queries, n_states), dtype=jnp.int32)
    row_indices = jnp.arange(batch_size, dtype=jnp.int32)
    last_policy = jnp.zeros((batch_size, n_states), dtype=jnp.int32)

    while True:
        active = jnp.logical_and(alive, total_queries < max_queries)
        if not bool(jnp.any(active)):
            break

        gamma_eff = (h * gamma) / (1.0 + h)
        expected_v_hat = jnp.einsum("bsak,bk->bsa", p_hat, v)
        h_expanded = jnp.expand_dims(jnp.expand_dims(h, axis=1), axis=2)
        gamma_expanded = jnp.expand_dims(jnp.expand_dims(gamma, axis=1), axis=2)
        r_tilde = (
            (1.0 / (1.0 + h_expanded)) * jnp.expand_dims(v, axis=2)
            + (h_expanded / (1.0 + h_expanded)) * r
            + ((h_expanded * gamma_expanded) / (1.0 + h_expanded))
            * (last_expected_v_true - expected_v_hat)
        )
        v_candidate = _batched_waterfall_inner_vi(
            v,
            p_hat,
            r_tilde,
            gamma_eff,
            jnp.asarray(inner_vi_tol, dtype=jnp.float64),
            jnp.asarray(inner_vi_max_iters, dtype=jnp.int32),
        )

        diverged = jnp.max(jnp.abs(v_candidate), axis=1) > v_div_envelope
        expected_v_new_true = jnp.einsum("bsak,bk->bsa", p, v_candidate)
        q_new = r + gamma[:, None, None] * expected_v_new_true
        t_v_new = jnp.max(q_new, axis=2)
        resid = jnp.max(jnp.abs(t_v_new - v_candidate), axis=1)
        nonfinite = jnp.logical_or(
            jnp.logical_not(jnp.all(jnp.isfinite(v_candidate), axis=1)),
            jnp.logical_not(jnp.isfinite(resid)),
        )
        failed = jnp.logical_and(active, jnp.logical_or(diverged, nonfinite))
        charged_queries = jnp.where(active, total_queries + 1, total_queries)
        next_h_on_fail = jnp.maximum(h_floor, h * decay_factor_arr)
        should_stop_after_fail = jnp.logical_and(failed, next_h_on_fail <= h_floor)
        h = jnp.where(failed, next_h_on_fail, h)
        stall_count = jnp.where(failed, 0, stall_count)

        succeeded = jnp.logical_and(active, jnp.logical_not(failed))
        candidate_q_surr = r_tilde + gamma_eff[:, None, None] * jnp.einsum("bsak,bk->bsa", p_hat, v_candidate)
        candidate_policy = jnp.argmax(candidate_q_surr, axis=2).astype(jnp.int32)
        candidate_policy_available = jnp.logical_and(
            jnp.all(jnp.isfinite(v_candidate), axis=1),
            jnp.all(jnp.isfinite(candidate_q_surr), axis=(1, 2)),
        )
        step_policy = jnp.where(candidate_policy_available[:, None], candidate_policy, last_policy)
        v_pi = _batched_evaluate_policy_values(p, r, gamma, step_policy)
        current_value_errors = _batched_value_errors(
            v_candidate,
            target_v,
            norm_name=value_norm,
            normalized=value_normalized,
        )
        current_policy_value_errors = _batched_value_errors(
            v_pi,
            target_v,
            norm_name=value_norm,
            normalized=value_normalized,
        )
        retained_value_errors = _batched_value_errors(
            v,
            target_v,
            norm_name=value_norm,
            normalized=value_normalized,
        )

        slot = total_queries
        current_residual_rows = residuals_trace[row_indices, slot]
        current_value_rows = value_errors_trace[row_indices, slot]
        current_policy_value_rows = policy_value_errors_trace[row_indices, slot]
        current_policy_rows = policy_trace[row_indices, slot, :]

        stored_residuals = jnp.where(succeeded, resid, current_resid)
        stored_value_errors = jnp.where(succeeded, current_value_errors, retained_value_errors)
        stored_policy_value_errors = current_policy_value_errors
        stored_policy = step_policy

        residuals_trace = residuals_trace.at[row_indices, slot].set(
            jnp.where(active, stored_residuals, current_residual_rows)
        )
        value_errors_trace = value_errors_trace.at[row_indices, slot].set(
            jnp.where(active, stored_value_errors, current_value_rows)
        )
        policy_value_errors_trace = policy_value_errors_trace.at[row_indices, slot].set(
            jnp.where(active, stored_policy_value_errors, current_policy_value_rows)
        )
        policy_trace = policy_trace.at[row_indices, slot, :].set(
            jnp.where(active[:, None], stored_policy, current_policy_rows)
        )

        improved = resid < best_resid
        best_resid = jnp.where(jnp.logical_and(succeeded, improved), resid, best_resid)
        tentative_stall_count = jnp.where(
            jnp.logical_and(succeeded, improved),
            0,
            jnp.where(succeeded, stall_count + 1, stall_count),
        )
        should_decay_after_success = jnp.logical_and(
            succeeded,
            jnp.logical_and(
                jnp.logical_not(improved),
                tentative_stall_count >= stall_patience_arr,
            ),
        )
        h = jnp.where(
            should_decay_after_success,
            jnp.maximum(h_floor, h * decay_factor_arr),
            h,
        )
        stall_count = jnp.where(
            jnp.logical_and(succeeded, improved),
            0,
            jnp.where(should_decay_after_success, 0, tentative_stall_count),
        )
        total_queries = charged_queries
        alive = jnp.where(should_stop_after_fail, False, alive)
        v = jnp.where(succeeded[:, None], v_candidate, v)
        last_expected_v_true = jnp.where(
            succeeded[:, None, None],
            expected_v_new_true,
            last_expected_v_true,
        )
        current_resid = jnp.where(succeeded, resid, current_resid)
        last_policy = jnp.where(active[:, None], step_policy, last_policy)

    return (
        queries,
        residuals_trace,
        value_errors_trace,
        policy_value_errors_trace,
        policy_trace,
    )


def _batched_geo_waterfall_control_hit_times(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    effect_threshold: jnp.ndarray,
    *,
    h_init: jnp.ndarray,
    alpha: jnp.ndarray,
    inner_vi_tol: jnp.ndarray,
    inner_vi_max_iters: jnp.ndarray,
    max_queries: int,
) -> List[int]:
    v = jnp.zeros((p.shape[0], p.shape[1]), dtype=jnp.float64)
    total_queries = jnp.zeros((p.shape[0],), dtype=jnp.int32)
    hit_times = jnp.full((p.shape[0],), max_queries + 1, dtype=jnp.int32)
    h = jnp.asarray(h_init, dtype=jnp.float64)
    alpha_arr = jnp.asarray(alpha, dtype=jnp.float64)
    current_resid = _batched_control_residual_norms(v, p, r, gamma)
    best_resid = current_resid
    last_expected_v_true = jnp.einsum("bsak,bk->bsa", p, v)
    v_div_envelope = _WATERFALL_DIVERGENCE_MULT * jnp.maximum(
        jnp.max(jnp.abs(r), axis=(1, 2)) / jnp.maximum(1.0 - gamma, 1e-12),
        1.0,
    )

    while True:
        active = jnp.logical_and(hit_times == max_queries + 1, total_queries < max_queries)
        if not bool(jnp.any(active)):
            break

        inv_1ph = 1.0 / (1.0 + h)
        blend = h * inv_1ph
        gamma_blend = gamma * blend
        expected_v_hat = jnp.einsum("bsak,bk->bsa", p_hat, v)
        r_tilde = (
            inv_1ph[:, None, None] * jnp.expand_dims(v, axis=2)
            + blend[:, None, None] * r
            + gamma_blend[:, None, None] * (last_expected_v_true - expected_v_hat)
        )
        v_candidate = _batched_waterfall_inner_vi(
            v,
            p_hat,
            r_tilde,
            gamma_blend,
            jnp.asarray(inner_vi_tol, dtype=jnp.float64),
            jnp.asarray(inner_vi_max_iters, dtype=jnp.int32),
        )

        diverged = jnp.max(jnp.abs(v_candidate), axis=1) > v_div_envelope
        expected_v_new_true = jnp.einsum("bsak,bk->bsa", p, v_candidate)
        q_new = r + gamma[:, None, None] * expected_v_new_true
        t_v_new = jnp.max(q_new, axis=2)
        resid = jnp.max(jnp.abs(t_v_new - v_candidate), axis=1)
        nonfinite = jnp.logical_or(
            jnp.logical_not(jnp.all(jnp.isfinite(v_candidate), axis=1)),
            jnp.logical_not(jnp.isfinite(resid)),
        )
        valid = jnp.logical_and(active, jnp.logical_not(jnp.logical_or(diverged, nonfinite)))
        improved = resid < best_resid
        next_total_queries = jnp.where(active, total_queries + 1, total_queries)
        next_v = jnp.where(valid[:, None], v_candidate, v)
        next_expected_v_true = jnp.where(
            valid[:, None, None],
            expected_v_new_true,
            last_expected_v_true,
        )
        next_current_resid = jnp.where(valid, resid, current_resid)
        next_best_resid = jnp.where(jnp.logical_and(valid, improved), resid, best_resid)
        next_h = jnp.where(
            jnp.logical_and(valid, improved),
            h,
            jnp.where(active, jnp.maximum(1.0, alpha_arr * h), h),
        )
        new_hits = jnp.logical_and(active, next_current_resid <= effect_threshold)
        hit_times = jnp.where(
            jnp.logical_and(hit_times == max_queries + 1, new_hits),
            next_total_queries,
            hit_times,
        )

        total_queries = next_total_queries
        v = next_v
        last_expected_v_true = next_expected_v_true
        current_resid = next_current_resid
        best_resid = next_best_resid
        h = next_h

    return [int(value) for value in hit_times.tolist()]


def _batched_geo_waterfall_control_traces(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    target_v: jnp.ndarray,
    *,
    h_init: jnp.ndarray,
    alpha: jnp.ndarray,
    inner_vi_tol: jnp.ndarray,
    inner_vi_max_iters: jnp.ndarray,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size, n_states, _, _ = p.shape
    v = jnp.zeros((batch_size, n_states), dtype=jnp.float64)
    total_queries = jnp.zeros((batch_size,), dtype=jnp.int32)
    h = jnp.asarray(h_init, dtype=jnp.float64)
    alpha_arr = jnp.asarray(alpha, dtype=jnp.float64)
    current_resid = _batched_control_residual_norms(v, p, r, gamma)
    best_resid = current_resid
    last_expected_v_true = jnp.einsum("bsak,bk->bsa", p, v)
    last_policy = jnp.zeros((batch_size, n_states), dtype=jnp.int32)
    v_div_envelope = _WATERFALL_DIVERGENCE_MULT * jnp.maximum(
        jnp.max(jnp.abs(r), axis=(1, 2)) / jnp.maximum(1.0 - gamma, 1e-12),
        1.0,
    )
    queries = jnp.broadcast_to(
        jnp.arange(1, max_queries + 1, dtype=jnp.int32),
        (batch_size, max_queries),
    )
    residuals_trace = jnp.full((batch_size, max_queries), jnp.inf, dtype=jnp.float64)
    value_errors_trace = jnp.full((batch_size, max_queries), jnp.inf, dtype=jnp.float64)
    policy_value_errors_trace = jnp.full((batch_size, max_queries), jnp.inf, dtype=jnp.float64)
    policy_trace = jnp.zeros((batch_size, max_queries, n_states), dtype=jnp.int32)
    row_indices = jnp.arange(batch_size, dtype=jnp.int32)

    while True:
        active = total_queries < max_queries
        if not bool(jnp.any(active)):
            break

        inv_1ph = 1.0 / (1.0 + h)
        blend = h * inv_1ph
        gamma_blend = gamma * blend
        expected_v_hat = jnp.einsum("bsak,bk->bsa", p_hat, v)
        r_tilde = (
            inv_1ph[:, None, None] * jnp.expand_dims(v, axis=2)
            + blend[:, None, None] * r
            + gamma_blend[:, None, None] * (last_expected_v_true - expected_v_hat)
        )
        v_candidate = _batched_waterfall_inner_vi(
            v,
            p_hat,
            r_tilde,
            gamma_blend,
            jnp.asarray(inner_vi_tol, dtype=jnp.float64),
            jnp.asarray(inner_vi_max_iters, dtype=jnp.int32),
        )

        diverged = jnp.max(jnp.abs(v_candidate), axis=1) > v_div_envelope
        expected_v_new_true = jnp.einsum("bsak,bk->bsa", p, v_candidate)
        q_new = r + gamma[:, None, None] * expected_v_new_true
        t_v_new = jnp.max(q_new, axis=2)
        resid = jnp.max(jnp.abs(t_v_new - v_candidate), axis=1)
        nonfinite = jnp.logical_or(
            jnp.logical_not(jnp.all(jnp.isfinite(v_candidate), axis=1)),
            jnp.logical_not(jnp.isfinite(resid)),
        )
        valid = jnp.logical_and(active, jnp.logical_not(jnp.logical_or(diverged, nonfinite)))
        improved = resid < best_resid
        next_total_queries = jnp.where(active, total_queries + 1, total_queries)
        next_v = jnp.where(valid[:, None], v_candidate, v)
        next_expected_v_true = jnp.where(
            valid[:, None, None],
            expected_v_new_true,
            last_expected_v_true,
        )
        next_current_resid = jnp.where(valid, resid, current_resid)
        next_best_resid = jnp.where(jnp.logical_and(valid, improved), resid, best_resid)
        next_h = jnp.where(
            jnp.logical_and(valid, improved),
            h,
            jnp.where(active, jnp.maximum(1.0, alpha_arr * h), h),
        )

        candidate_q_surr = r_tilde + gamma_blend[:, None, None] * jnp.einsum(
            "bsak,bk->bsa", p_hat, v_candidate
        )
        candidate_policy = jnp.argmax(candidate_q_surr, axis=2).astype(jnp.int32)
        candidate_policy_available = jnp.logical_and(
            jnp.all(jnp.isfinite(v_candidate), axis=1),
            jnp.all(jnp.isfinite(candidate_q_surr), axis=(1, 2)),
        )
        step_policy = jnp.where(candidate_policy_available[:, None], candidate_policy, last_policy)
        v_pi = _batched_evaluate_policy_values(p, r, gamma, step_policy)
        current_value_errors = _batched_value_errors(
            next_v,
            target_v,
            norm_name=value_norm,
            normalized=value_normalized,
        )
        current_policy_value_errors = _batched_value_errors(
            v_pi,
            target_v,
            norm_name=value_norm,
            normalized=value_normalized,
        )

        slot = total_queries
        current_residual_rows = residuals_trace[row_indices, slot]
        current_value_rows = value_errors_trace[row_indices, slot]
        current_policy_value_rows = policy_value_errors_trace[row_indices, slot]
        current_policy_rows = policy_trace[row_indices, slot, :]

        residuals_trace = residuals_trace.at[row_indices, slot].set(
            jnp.where(active, next_current_resid, current_residual_rows)
        )
        value_errors_trace = value_errors_trace.at[row_indices, slot].set(
            jnp.where(active, current_value_errors, current_value_rows)
        )
        policy_value_errors_trace = policy_value_errors_trace.at[row_indices, slot].set(
            jnp.where(active, current_policy_value_errors, current_policy_value_rows)
        )
        policy_trace = policy_trace.at[row_indices, slot, :].set(
            jnp.where(active[:, None], step_policy, current_policy_rows)
        )

        total_queries = next_total_queries
        v = next_v
        last_expected_v_true = next_expected_v_true
        current_resid = next_current_resid
        best_resid = next_best_resid
        h = next_h
        last_policy = jnp.where(active[:, None], step_policy, last_policy)

    return (
        queries,
        residuals_trace,
        value_errors_trace,
        policy_value_errors_trace,
        policy_trace,
    )


def _batched_dd_waterfall_control_hit_times(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    effect_threshold: jnp.ndarray,
    *,
    h_init: jnp.ndarray,
    m_max: jnp.ndarray,
    decay_factor: jnp.ndarray,
    inner_vi_tol: jnp.ndarray,
    inner_vi_max_iters: jnp.ndarray,
    max_queries: int,
) -> List[int]:
    v = jnp.zeros((p.shape[0], p.shape[1]), dtype=jnp.float64)
    total_queries = jnp.zeros((p.shape[0],), dtype=jnp.int32)
    hit_times = jnp.full((p.shape[0],), max_queries + 1, dtype=jnp.int32)
    h = jnp.asarray(h_init, dtype=jnp.float64)
    m_max_arr = jnp.asarray(m_max, dtype=jnp.int32)
    decay_factor_arr = jnp.asarray(decay_factor, dtype=jnp.float64)
    current_resid = _batched_control_residual_norms(v, p, r, gamma)
    best_resid = current_resid
    last_expected_v_true = jnp.einsum("bsak,bk->bsa", p, v)
    v_div_envelope = _WATERFALL_DIVERGENCE_MULT * jnp.maximum(
        jnp.max(jnp.abs(r), axis=(1, 2)) / jnp.maximum(1.0 - gamma, 1e-12),
        1.0,
    )
    max_depth = max(int(value) for value in m_max_arr.tolist())

    while True:
        active = jnp.logical_and(hit_times == max_queries + 1, total_queries < max_queries)
        if not bool(jnp.any(active)):
            break

        remaining_budget = jnp.asarray(max_queries, dtype=jnp.int32) - total_queries
        attempt_limit = jnp.minimum(m_max_arr, remaining_budget)
        anchor_v = v
        u = anchor_v
        u_expected_true = last_expected_v_true
        next_v = v
        next_expected_true = last_expected_v_true
        next_current_resid = current_resid
        next_best_resid = best_resid
        next_h = h
        depth_cost = jnp.zeros((p.shape[0],), dtype=jnp.int32)
        inv_1ph = 1.0 / (1.0 + h)
        blend = h * inv_1ph
        gamma_blend = gamma * blend

        for depth in range(1, max_depth + 1):
            depth_active = jnp.logical_and(active, depth_cost == 0)
            depth_active = jnp.logical_and(depth_active, attempt_limit >= depth)
            if not bool(jnp.any(depth_active)):
                continue

            expected_u_hat = jnp.einsum("bsak,bk->bsa", p_hat, u)
            r_tilde = (
                inv_1ph[:, None, None] * jnp.expand_dims(anchor_v, axis=2)
                + blend[:, None, None] * r
                + gamma_blend[:, None, None] * (u_expected_true - expected_u_hat)
            )
            v_candidate = _batched_waterfall_inner_vi(
                u,
                p_hat,
                r_tilde,
                gamma_blend,
                jnp.asarray(inner_vi_tol, dtype=jnp.float64),
                jnp.asarray(inner_vi_max_iters, dtype=jnp.int32),
            )
            diverged = jnp.max(jnp.abs(v_candidate), axis=1) > v_div_envelope
            expected_v_new_true = jnp.einsum("bsak,bk->bsa", p, v_candidate)
            q_new = r + gamma[:, None, None] * expected_v_new_true
            t_v_new = jnp.max(q_new, axis=2)
            resid = jnp.max(jnp.abs(t_v_new - v_candidate), axis=1)
            valid = jnp.logical_and(
                depth_active,
                jnp.logical_and(
                    jnp.all(jnp.isfinite(v_candidate), axis=1),
                    jnp.logical_and(
                        jnp.isfinite(resid),
                        jnp.logical_not(diverged),
                    ),
                ),
            )
            invalid = jnp.logical_and(depth_active, jnp.logical_not(valid))
            progress = jnp.logical_and(valid, resid < best_resid)
            exhausted = jnp.logical_and(
                valid,
                jnp.logical_and(jnp.logical_not(progress), attempt_limit == depth),
            )
            terminal = jnp.logical_or(invalid, jnp.logical_or(progress, exhausted))

            accept_terminal = jnp.logical_and(terminal, valid)
            next_v = jnp.where(accept_terminal[:, None], v_candidate, next_v)
            next_expected_true = jnp.where(
                accept_terminal[:, None, None],
                expected_v_new_true,
                next_expected_true,
            )
            next_current_resid = jnp.where(accept_terminal, resid, next_current_resid)
            next_best_resid = jnp.where(progress, resid, next_best_resid)
            next_h = jnp.where(
                progress,
                h,
                jnp.where(terminal, jnp.maximum(1.0, decay_factor_arr * h), next_h),
            )
            depth_cost = jnp.where(terminal, depth, depth_cost)

            continue_depth = jnp.logical_and(depth_active, jnp.logical_not(terminal))
            u = jnp.where(continue_depth[:, None], v_candidate, u)
            u_expected_true = jnp.where(
                continue_depth[:, None, None],
                expected_v_new_true,
                u_expected_true,
            )

        next_total_queries = total_queries + depth_cost
        new_hits = jnp.logical_and(active, next_current_resid <= effect_threshold)
        hit_times = jnp.where(
            jnp.logical_and(hit_times == max_queries + 1, new_hits),
            next_total_queries,
            hit_times,
        )

        total_queries = next_total_queries
        v = next_v
        last_expected_v_true = next_expected_true
        current_resid = next_current_resid
        best_resid = next_best_resid
        h = next_h

    return [int(value) for value in hit_times.tolist()]


def _batched_dd_waterfall_control_traces(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    target_v: jnp.ndarray,
    *,
    h_init: jnp.ndarray,
    m_max: jnp.ndarray,
    decay_factor: jnp.ndarray,
    inner_vi_tol: jnp.ndarray,
    inner_vi_max_iters: jnp.ndarray,
    max_queries: int,
    value_norm: str,
    value_normalized: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size, n_states, _, _ = p.shape
    v = jnp.zeros((batch_size, n_states), dtype=jnp.float64)
    total_queries = jnp.zeros((batch_size,), dtype=jnp.int32)
    step_counts = jnp.zeros((batch_size,), dtype=jnp.int32)
    h = jnp.asarray(h_init, dtype=jnp.float64)
    m_max_arr = jnp.asarray(m_max, dtype=jnp.int32)
    decay_factor_arr = jnp.asarray(decay_factor, dtype=jnp.float64)
    current_resid = _batched_control_residual_norms(v, p, r, gamma)
    best_resid = current_resid
    last_expected_v_true = jnp.einsum("bsak,bk->bsa", p, v)
    last_policy = jnp.zeros((batch_size, n_states), dtype=jnp.int32)
    v_div_envelope = _WATERFALL_DIVERGENCE_MULT * jnp.maximum(
        jnp.max(jnp.abs(r), axis=(1, 2)) / jnp.maximum(1.0 - gamma, 1e-12),
        1.0,
    )
    max_depth = max(int(value) for value in m_max_arr.tolist())
    queries = jnp.zeros((batch_size, max_queries), dtype=jnp.int32)
    residuals_trace = jnp.full((batch_size, max_queries), jnp.inf, dtype=jnp.float64)
    value_errors_trace = jnp.full((batch_size, max_queries), jnp.inf, dtype=jnp.float64)
    policy_value_errors_trace = jnp.full((batch_size, max_queries), jnp.inf, dtype=jnp.float64)
    policy_trace = jnp.zeros((batch_size, max_queries, n_states), dtype=jnp.int32)
    row_indices = jnp.arange(batch_size, dtype=jnp.int32)

    while True:
        active = total_queries < max_queries
        if not bool(jnp.any(active)):
            break

        remaining_budget = jnp.asarray(max_queries, dtype=jnp.int32) - total_queries
        attempt_limit = jnp.minimum(m_max_arr, remaining_budget)
        anchor_v = v
        u = anchor_v
        u_expected_true = last_expected_v_true
        next_v = v
        next_expected_true = last_expected_v_true
        next_current_resid = current_resid
        next_best_resid = best_resid
        next_h = h
        next_policy = last_policy
        depth_cost = jnp.zeros((batch_size,), dtype=jnp.int32)
        inv_1ph = 1.0 / (1.0 + h)
        blend = h * inv_1ph
        gamma_blend = gamma * blend

        for depth in range(1, max_depth + 1):
            depth_active = jnp.logical_and(active, depth_cost == 0)
            depth_active = jnp.logical_and(depth_active, attempt_limit >= depth)
            if not bool(jnp.any(depth_active)):
                continue

            expected_u_hat = jnp.einsum("bsak,bk->bsa", p_hat, u)
            r_tilde = (
                inv_1ph[:, None, None] * jnp.expand_dims(anchor_v, axis=2)
                + blend[:, None, None] * r
                + gamma_blend[:, None, None] * (u_expected_true - expected_u_hat)
            )
            v_candidate = _batched_waterfall_inner_vi(
                u,
                p_hat,
                r_tilde,
                gamma_blend,
                jnp.asarray(inner_vi_tol, dtype=jnp.float64),
                jnp.asarray(inner_vi_max_iters, dtype=jnp.int32),
            )
            diverged = jnp.max(jnp.abs(v_candidate), axis=1) > v_div_envelope
            expected_v_new_true = jnp.einsum("bsak,bk->bsa", p, v_candidate)
            q_new = r + gamma[:, None, None] * expected_v_new_true
            t_v_new = jnp.max(q_new, axis=2)
            resid = jnp.max(jnp.abs(t_v_new - v_candidate), axis=1)
            valid = jnp.logical_and(
                depth_active,
                jnp.logical_and(
                    jnp.all(jnp.isfinite(v_candidate), axis=1),
                    jnp.logical_and(
                        jnp.isfinite(resid),
                        jnp.logical_not(diverged),
                    ),
                ),
            )
            invalid = jnp.logical_and(depth_active, jnp.logical_not(valid))
            progress = jnp.logical_and(valid, resid < best_resid)
            exhausted = jnp.logical_and(
                valid,
                jnp.logical_and(jnp.logical_not(progress), attempt_limit == depth),
            )
            terminal = jnp.logical_or(invalid, jnp.logical_or(progress, exhausted))

            candidate_q_surr = r_tilde + gamma_blend[:, None, None] * jnp.einsum(
                "bsak,bk->bsa", p_hat, v_candidate
            )
            candidate_policy = jnp.argmax(candidate_q_surr, axis=2).astype(jnp.int32)
            candidate_policy_available = jnp.logical_and(
                jnp.all(jnp.isfinite(v_candidate), axis=1),
                jnp.all(jnp.isfinite(candidate_q_surr), axis=(1, 2)),
            )
            step_policy = jnp.where(candidate_policy_available[:, None], candidate_policy, last_policy)

            accept_terminal = jnp.logical_and(terminal, valid)
            next_v = jnp.where(accept_terminal[:, None], v_candidate, next_v)
            next_expected_true = jnp.where(
                accept_terminal[:, None, None],
                expected_v_new_true,
                next_expected_true,
            )
            next_current_resid = jnp.where(accept_terminal, resid, next_current_resid)
            next_best_resid = jnp.where(progress, resid, next_best_resid)
            next_h = jnp.where(
                progress,
                h,
                jnp.where(terminal, jnp.maximum(1.0, decay_factor_arr * h), next_h),
            )
            next_policy = jnp.where(terminal[:, None], step_policy, next_policy)
            depth_cost = jnp.where(terminal, depth, depth_cost)

            continue_depth = jnp.logical_and(depth_active, jnp.logical_not(terminal))
            u = jnp.where(continue_depth[:, None], v_candidate, u)
            u_expected_true = jnp.where(
                continue_depth[:, None, None],
                expected_v_new_true,
                u_expected_true,
            )

        next_total_queries = total_queries + depth_cost
        slot = step_counts
        current_query_rows = queries[row_indices, slot]
        current_residual_rows = residuals_trace[row_indices, slot]
        current_value_rows = value_errors_trace[row_indices, slot]
        current_policy_value_rows = policy_value_errors_trace[row_indices, slot]
        current_policy_rows = policy_trace[row_indices, slot, :]
        v_pi = _batched_evaluate_policy_values(p, r, gamma, next_policy)
        value_errors = _batched_value_errors(
            next_v,
            target_v,
            norm_name=value_norm,
            normalized=value_normalized,
        )
        policy_value_errors = _batched_value_errors(
            v_pi,
            target_v,
            norm_name=value_norm,
            normalized=value_normalized,
        )

        queries = queries.at[row_indices, slot].set(
            jnp.where(active, next_total_queries, current_query_rows)
        )
        residuals_trace = residuals_trace.at[row_indices, slot].set(
            jnp.where(active, next_current_resid, current_residual_rows)
        )
        value_errors_trace = value_errors_trace.at[row_indices, slot].set(
            jnp.where(active, value_errors, current_value_rows)
        )
        policy_value_errors_trace = policy_value_errors_trace.at[row_indices, slot].set(
            jnp.where(active, policy_value_errors, current_policy_value_rows)
        )
        policy_trace = policy_trace.at[row_indices, slot, :].set(
            jnp.where(active[:, None], next_policy, current_policy_rows)
        )

        total_queries = next_total_queries
        step_counts = jnp.where(active, step_counts + 1, step_counts)
        v = next_v
        last_expected_v_true = next_expected_true
        current_resid = next_current_resid
        best_resid = next_best_resid
        h = next_h
        last_policy = next_policy

    return (
        queries,
        residuals_trace,
        value_errors_trace,
        policy_value_errors_trace,
        policy_trace,
    )


def _batched_aisp_control_hit_times(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    effect_threshold: jnp.ndarray,
    *,
    max_queries: int,
    target_epsilon: Any = _AISP_TARGET_EPSILON_DEFAULT,
) -> List[int]:
    v0 = jnp.zeros((p.shape[0], p.shape[1]), dtype=jnp.float64)
    b0 = _batched_control_residual_norms(v0, p, r, gamma)
    total_queries = jnp.ones((p.shape[0],), dtype=jnp.int32)
    hit_times = jnp.where(
        b0 <= effect_threshold,
        total_queries,
        jnp.full((p.shape[0],), max_queries + 1, dtype=jnp.int32),
    )

    delta = 1.0 - gamma
    safe_delta = jnp.maximum(delta, 1e-12)
    target_eps_arr = jnp.broadcast_to(
        jnp.asarray(target_epsilon, dtype=jnp.float64),
        (p.shape[0],),
    )
    k_prime = jnp.maximum(
        1.0,
        jnp.log(1.0 / target_eps_arr) + 2.0 * jnp.log((1.0 + gamma) / safe_delta),
    )
    j_steps = jnp.maximum(
        jnp.ceil(jnp.log2(8.0 / safe_delta)).astype(jnp.int32),
        jnp.ones((p.shape[0],), dtype=jnp.int32),
    )
    r_hat = jnp.ones((p.shape[0],), dtype=jnp.float64)

    while True:
        active = jnp.logical_and(hit_times == max_queries + 1, total_queries < max_queries)
        if not bool(jnp.any(active)):
            break

        denom = delta * (2.0 * r_hat - 1.0)
        h_i = jnp.where(denom > 1e-12, 1.0 / denom, 1.0 / safe_delta)
        n_i = jnp.ceil(4.0 * r_hat * k_prime).astype(jnp.int32)

        v = jnp.array(v0, copy=True)
        max_n = max([int(value) for value in n_i.tolist()], default=0)
        for step_index in range(max_n):
            can_step = jnp.logical_and(active, step_index < n_i)
            can_step = jnp.logical_and(can_step, total_queries < max_queries)
            if not bool(jnp.any(can_step)):
                break

            v_next = _batched_isp_control_step(v, p, p_hat, r, gamma, h_i, j_steps)
            total_queries = jnp.where(can_step, total_queries + j_steps, total_queries)
            v = jnp.where(can_step[:, None], v_next, v)

        eval_active = jnp.logical_and(active, total_queries < max_queries)
        if not bool(jnp.any(eval_active)):
            continue

        current_resid = _batched_control_residual_norms(v, p, r, gamma)
        total_queries = jnp.where(eval_active, total_queries + 1, total_queries)
        new_hits = jnp.logical_and(eval_active, current_resid <= effect_threshold)
        hit_times = jnp.where(new_hits, total_queries, hit_times)
        r_hat = jnp.where(jnp.logical_and(eval_active, jnp.logical_not(new_hits)), r_hat * 2.0, r_hat)

    return [int(value) for value in hit_times.tolist()]


def _batched_osvi_control_hit_times(
    p: jnp.ndarray,
    p_hat: jnp.ndarray,
    r: jnp.ndarray,
    gamma: jnp.ndarray,
    effect_threshold: jnp.ndarray,
    *,
    max_queries: int,
    surrogate_policy_iteration_steps: int | None = None,
) -> List[int]:
    v = jnp.zeros((p.shape[0], p.shape[1]), dtype=jnp.float64)
    hit_times = jnp.full((p.shape[0],), max_queries + 1, dtype=jnp.int32)
    current_policy: jnp.ndarray | None = None

    for step in range(1, max_queries + 1):
        residuals = _batched_control_residual_norms(v, p, r, gamma)
        new_hits = jnp.logical_and(hit_times == max_queries + 1, residuals <= effect_threshold)
        hit_times = jnp.where(new_hits, step, hit_times)
        if not bool(jnp.any(hit_times == max_queries + 1)):
            break

        expected_true = jnp.einsum("bsak,bk->bsa", p, v)
        expected_shadow = jnp.einsum("bsak,bk->bsa", p_hat, v)
        r_tilde = r + gamma[:, None, None] * (expected_true - expected_shadow)
        v, current_policy = _batched_solve_surrogate_control_with_policy(
            p_hat,
            r_tilde,
            gamma,
            init_policy=current_policy,
            surrogate_policy_iteration_steps=surrogate_policy_iteration_steps,
        )

    return [int(value) for value in hit_times.tolist()]


def run_pe_algorithms(
    mdp: MDP,
    p_hat: Any,
    algo_configs: Dict[str, Dict[str, Any]],
    max_queries: int,
    threshold: float,
    model_norm: str,
    value_norm: str,
    value_normalized: bool,
    target_bundle: LoadedDeterministicTarget | None = None,
) -> Dict[str, int]:
    """Run maintained policy-evaluation algorithms."""
    results, _ = run_pe_algorithms_with_optional_traces(
        mdp,
        p_hat,
        algo_configs,
        max_queries,
        threshold,
        model_norm,
        value_norm,
        value_normalized,
        emit_traces=False,
        target_bundle=target_bundle,
    )
    return results


def run_control_algorithms(
    mdp: MDP,
    p_hat: Any,
    algo_configs: Dict[str, Dict[str, Any]],
    max_queries: int,
    threshold: float,
    seed: int,
    *,
    metric_mode: str = "policy_value_error",
    value_norm: str = "L1",
    value_normalized: bool = True,
    target_bundle: LoadedDeterministicTarget | None = None,
    vi_control_cls: type = VI_Control,
    msvi_control_cls: type = MSVI_Control,
    isp_control_cls: type = ISP_Control,
    aisp_control_cls: type = AISP_Control,
    shadow_isp_control_cls: type = Shadow_ISP_Control,
    waterfall_isp_control_cls: type = WaterfallISP_Control,
    pick_isp_params_control_fn: Callable[..., tuple[Any, Any]] = pick_isp_params_control,
    seeded_shadow_init_fn: Callable[[int, int], Any] = seeded_shadow_init,
) -> Dict[str, int]:
    """Run maintained deterministic control algorithms."""
    results, _ = run_control_algorithms_with_optional_traces(
        mdp,
        p_hat,
        algo_configs,
        max_queries,
        threshold,
        seed,
        metric_mode=metric_mode,
        value_norm=value_norm,
        value_normalized=value_normalized,
        emit_traces=False,
        target_bundle=target_bundle,
        vi_control_cls=vi_control_cls,
        msvi_control_cls=msvi_control_cls,
        isp_control_cls=isp_control_cls,
        aisp_control_cls=aisp_control_cls,
        shadow_isp_control_cls=shadow_isp_control_cls,
        waterfall_isp_control_cls=waterfall_isp_control_cls,
        pick_isp_params_control_fn=pick_isp_params_control_fn,
        seeded_shadow_init_fn=seeded_shadow_init_fn,
    )
    return results


def _build_control_trace(
    *,
    queries: List[int],
    residuals: List[float],
    value_errors: List[float],
    policy_value_errors: List[float],
    policy_trace: Any,
    threshold: float,
    metric_mode: str,
    value_norm: str,
    value_normalized: bool,
    max_queries: int,
    target_bundle: LoadedDeterministicTarget | None,
) -> ControlTrace:
    metric_errors = policy_value_errors if metric_mode == "policy_value_error" else residuals
    hit_time = _get_hit_time(queries, metric_errors, threshold, max_queries)
    return ControlTrace(
        queries=list(queries),
        residuals=[float(value) for value in residuals],
        value_errors=[float(value) for value in value_errors],
        policy_value_errors=[float(value) for value in policy_value_errors],
        policy_trace=[
            [int(action) for action in policy]
            for policy in jnp.asarray(policy_trace, dtype=jnp.int32).tolist()
        ],
        hit_time=hit_time,
        threshold=float(threshold),
        metric_mode=str(metric_mode),
        value_norm=str(value_norm),
        value_normalized=bool(value_normalized),
        target_id=None if target_bundle is None else str(target_bundle.target_id),
        target_manifest=None if target_bundle is None else str(target_bundle.manifest_path),
    )


def _offline_eval_case_root(
    offline_eval_root: str,
    *,
    algorithm_name: str,
    lambda_value: float,
    seed: int,
) -> Path:
    lambda_token = str(lambda_value).replace("-", "m").replace(".", "p")
    return (
        Path(offline_eval_root)
        / canonical_algorithm_name(algorithm_name)
        / f"lambda_{lambda_token}_seed_{int(seed)}"
    )


def _policy_value_errors_from_trace(
    *,
    mdp: MDP,
    queries: List[int],
    policy_trace: Any,
    target_value: jnp.ndarray,
    target_bundle: LoadedDeterministicTarget | None,
    norm_name: str,
    normalized: bool,
    evaluation_pipeline: str,
    offline_eval_root: str | None,
    algorithm_name: str,
    lambda_value: float,
    seed: int,
) -> list[float]:
    policies = np.asarray(policy_trace, dtype=np.int32)
    if policies.ndim == 1:
        policies = policies[None, :]
    if policies.shape[0] == 0:
        return []
    if evaluation_pipeline != "train_then_offline" or not offline_eval_root:
        return reconstruct_policy_value_errors(
            mdp,
            policies,
            target_bundle=target_bundle,
            norm_name=norm_name,
            normalized=normalized,
        )

    sample_axis = tuple(int(query) for query in queries)
    plan = plan_offline_eval_layout(
        training_steps=max(sample_axis),
        env_batch_size=1,
        n_states=mdp.P.shape[0],
        sample_axis=sample_axis,
    )
    bundle_root = _offline_eval_case_root(
        offline_eval_root,
        algorithm_name=algorithm_name,
        lambda_value=lambda_value,
        seed=seed,
    )
    manifest = persist_policy_capture_bundle(
        bundle_root=bundle_root,
        algorithm=algorithm_name,
        problem_type="control",
        lambda_value=lambda_value,
        seed=seed,
        n_seeds=1,
        plan=plan,
        sample_axis=sample_axis,
        policy_history=policies[:, None, :].astype(np.uint8, copy=False),
    )
    evaluate_offline_policy_bundle(
        bundle_root=bundle_root,
        manifest=manifest,
        mdp_P=jnp.asarray(mdp.P, dtype=jnp.float64),
        mdp_R=jnp.asarray(mdp.R, dtype=jnp.float64),
        gamma=float(mdp.gamma),
        mode="reload",
    )
    value_history = load_value_slabs(bundle_root=bundle_root, manifest=manifest)[:, 0, :]
    return [
        compute_value_error(
            value_row,
            target_value,
            norm_name=norm_name,
            normalized=normalized,
        )
        for value_row in value_history
    ]


def run_control_algorithms_with_optional_traces(
    mdp: MDP,
    p_hat: Any,
    algo_configs: Dict[str, Dict[str, Any]],
    max_queries: int,
    threshold: float,
    seed: int,
    *,
    lambda_value: float = 0.0,
    metric_mode: str = "policy_value_error",
    value_norm: str = "L1",
    value_normalized: bool = True,
    emit_traces: bool = True,
    evaluation_pipeline: str = "online",
    offline_eval_root: str | None = None,
    target_bundle: LoadedDeterministicTarget | None = None,
    vi_control_cls: type = VI_Control,
    msvi_control_cls: type = MSVI_Control,
    isp_control_cls: type = ISP_Control,
    aisp_control_cls: type = AISP_Control,
    shadow_isp_control_cls: type = Shadow_ISP_Control,
    waterfall_isp_control_cls: type = WaterfallISP_Control,
    pick_isp_params_control_fn: Callable[..., tuple[Any, Any]] = pick_isp_params_control,
    seeded_shadow_init_fn: Callable[[int, int], Any] = seeded_shadow_init,
) -> tuple[Dict[str, int], Dict[str, ControlTrace]]:
    results: Dict[str, int] = {}
    traces: Dict[str, ControlTrace] = {}
    compute_resid = build_control_residual_fn(mdp)
    b0 = compute_resid(make_zero_value(mdp))
    effect_threshold = max(b0 * threshold, 1e-10)
    target_value = jnp.asarray(control_target_value(mdp, target_bundle=target_bundle), dtype=jnp.float64)

    for name, params in algo_configs.items():
        base_name = _control_algorithm_family(name)
        try:
            if base_name == "vi":
                vi = vi_control_cls(mdp)
                value_trace, policy_trace = vi.run(n_iter=max_queries)
                residuals = [compute_resid(v) for v in value_trace]
                value_errors = [
                    compute_value_error(
                        v,
                        target_value,
                        norm_name=value_norm,
                        normalized=value_normalized,
                    )
                    for v in value_trace
                ]
                queries = list(range(1, len(residuals) + 1))
                policy_value_errors = _policy_value_errors_from_trace(
                    mdp=mdp,
                    queries=queries,
                    policy_trace=policy_trace,
                    target_value=target_value,
                    target_bundle=target_bundle,
                    norm_name=value_norm,
                    normalized=value_normalized,
                    evaluation_pipeline=evaluation_pipeline,
                    offline_eval_root=offline_eval_root,
                    algorithm_name=name,
                    lambda_value=lambda_value,
                    seed=seed,
                )
            elif base_name == "osvi":
                msvi = msvi_control_cls(
                    mdp,
                    p_hat,
                    surrogate_policy_iteration_steps=_osvi_surrogate_policy_iteration_steps(params),
                )
                value_trace, policy_trace = msvi.run(n_iter=max_queries)
                residuals = [compute_resid(v) for v in value_trace]
                value_errors = [
                    compute_value_error(
                        v,
                        target_value,
                        norm_name=value_norm,
                        normalized=value_normalized,
                    )
                    for v in value_trace
                ]
                queries = list(range(1, len(residuals) + 1))
                policy_value_errors = _policy_value_errors_from_trace(
                    mdp=mdp,
                    queries=queries,
                    policy_trace=policy_trace,
                    target_value=target_value,
                    target_bundle=target_bundle,
                    norm_name=value_norm,
                    normalized=value_normalized,
                    evaluation_pipeline=evaluation_pipeline,
                    offline_eval_root=offline_eval_root,
                    algorithm_name=name,
                    lambda_value=lambda_value,
                    seed=seed,
                )
            elif base_name == "isp" and (emit_traces or metric_mode == "policy_value_error"):
                isp = isp_control_cls(mdp, p_hat)
                default_h, default_j = pick_isp_params_control_fn(mdp, p_hat)
                h = params.get("h", default_h)
                j_val = params.get("J", default_j)
                queries, residuals, value_trace, policy_trace = isp.run_with_policy_trace(
                    max_avps=max_queries,
                    h=h,
                    J=j_val,
                )
                value_errors = [
                    compute_value_error(
                        v,
                        target_value,
                        norm_name=value_norm,
                        normalized=value_normalized,
                    )
                    for v in value_trace
                ]
                if len(policy_trace) > 0:
                    policy_value_errors = _policy_value_errors_from_trace(
                        mdp=mdp,
                        queries=queries,
                        policy_trace=policy_trace,
                        target_value=target_value,
                        target_bundle=target_bundle,
                        norm_name=value_norm,
                        normalized=value_normalized,
                        evaluation_pipeline=evaluation_pipeline,
                        offline_eval_root=offline_eval_root,
                        algorithm_name=name,
                        lambda_value=lambda_value,
                        seed=seed,
                    )
                else:
                    policy_trace = jnp.zeros((0, mdp.P.shape[0]), dtype=jnp.int32)
                    policy_value_errors = []
            elif base_name == "aisp" and (emit_traces or metric_mode == "policy_value_error"):
                cfg = _build_aisp_config(params)
                aisp = aisp_control_cls(mdp, p_hat)
                queries, residuals, value_trace, policy_trace = aisp.run_with_policy_trace(
                    target_epsilon=float(
                        cfg.get("target_epsilon", _AISP_TARGET_EPSILON_DEFAULT)
                    ),
                    max_avps=max_queries,
                )
                value_errors = [
                    compute_value_error(
                        v,
                        target_value,
                        norm_name=value_norm,
                        normalized=value_normalized,
                    )
                    for v in value_trace
                ]
                if len(policy_trace) > 0:
                    policy_value_errors = _policy_value_errors_from_trace(
                        mdp=mdp,
                        queries=queries,
                        policy_trace=policy_trace,
                        target_value=target_value,
                        target_bundle=target_bundle,
                        norm_name=value_norm,
                        normalized=value_normalized,
                        evaluation_pipeline=evaluation_pipeline,
                        offline_eval_root=offline_eval_root,
                        algorithm_name=name,
                        lambda_value=lambda_value,
                        seed=seed,
                    )
                else:
                    policy_trace = jnp.zeros((0, mdp.P.shape[0]), dtype=jnp.int32)
                    policy_value_errors = []
            elif base_name == "shadow" and (emit_traces or metric_mode == "policy_value_error"):
                cfg = _build_shadow_config(name, params)
                shadow = shadow_isp_control_cls(mdp, p_hat, **cfg)
                v_init = seeded_shadow_init_fn(mdp.P.shape[0], seed)
                queries, residuals, value_trace, policy_trace = shadow.run_with_policy_trace(
                    max_avps=max_queries,
                    V_init=v_init,
                )
                value_errors = [
                    compute_value_error(
                        v,
                        target_value,
                        norm_name=value_norm,
                        normalized=value_normalized,
                    )
                    for v in value_trace
                ]
                if len(policy_trace) > 0:
                    policy_value_errors = _policy_value_errors_from_trace(
                        mdp=mdp,
                        queries=queries,
                        policy_trace=policy_trace,
                        target_value=target_value,
                        target_bundle=target_bundle,
                        norm_name=value_norm,
                        normalized=value_normalized,
                        evaluation_pipeline=evaluation_pipeline,
                        offline_eval_root=offline_eval_root,
                        algorithm_name=name,
                        lambda_value=lambda_value,
                        seed=seed,
                    )
                else:
                    policy_trace = jnp.zeros((0, mdp.P.shape[0]), dtype=jnp.int32)
                    policy_value_errors = []
            elif base_name == "waterfall" and (emit_traces or metric_mode == "policy_value_error"):
                waterfall = waterfall_isp_control_cls(
                    mdp.P,
                    p_hat,
                    mdp.R,
                    mdp.gamma,
                    **_build_waterfall_config(params),
                )
                queries, residuals, value_trace, policy_trace = waterfall.run_with_policy_trace(
                    max_queries=max_queries
                )
                value_errors = [
                    compute_value_error(
                        v,
                        target_value,
                        norm_name=value_norm,
                        normalized=value_normalized,
                    )
                    for v in value_trace
                ]
                if len(policy_trace) > 0:
                    policy_value_errors = _policy_value_errors_from_trace(
                        mdp=mdp,
                        queries=queries,
                        policy_trace=policy_trace,
                        target_value=target_value,
                        target_bundle=target_bundle,
                        norm_name=value_norm,
                        normalized=value_normalized,
                        evaluation_pipeline=evaluation_pipeline,
                        offline_eval_root=offline_eval_root,
                        algorithm_name=name,
                        lambda_value=lambda_value,
                        seed=seed,
                    )
                else:
                    policy_trace = jnp.zeros((0, mdp.P.shape[0]), dtype=jnp.int32)
                    policy_value_errors = []
            elif base_name == "geo_waterfall" and (emit_traces or metric_mode == "policy_value_error"):
                cfg = _build_geo_waterfall_control_config(params)
                waterfall = GeoWaterfallISP_Control(
                    mdp.P,
                    p_hat,
                    mdp.R,
                    mdp.gamma,
                    h_init=float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)),
                    alpha=float(cfg.get("alpha", _GEO_WATERFALL_ALPHA_DEFAULT)),
                    inner_vi_tol=float(cfg.get("inner_vi_tol", _WATERFALL_INNER_VI_TOL_DEFAULT)),
                    inner_vi_max_iters=int(
                        cfg.get("inner_vi_max_iters", _WATERFALL_INNER_VI_MAX_ITERS_DEFAULT)
                    ),
                )
                queries, residuals, value_trace, policy_trace = waterfall.run_with_policy_trace(
                    max_queries=max_queries
                )
                value_errors = [
                    compute_value_error(
                        v,
                        target_value,
                        norm_name=value_norm,
                        normalized=value_normalized,
                    )
                    for v in value_trace
                ]
                if len(policy_trace) > 0:
                    policy_value_errors = _policy_value_errors_from_trace(
                        mdp=mdp,
                        queries=queries,
                        policy_trace=policy_trace,
                        target_value=target_value,
                        target_bundle=target_bundle,
                        norm_name=value_norm,
                        normalized=value_normalized,
                        evaluation_pipeline=evaluation_pipeline,
                        offline_eval_root=offline_eval_root,
                        algorithm_name=name,
                        lambda_value=lambda_value,
                        seed=seed,
                    )
                else:
                    policy_trace = jnp.zeros((0, mdp.P.shape[0]), dtype=jnp.int32)
                    policy_value_errors = []
            elif base_name == "dd_waterfall" and (emit_traces or metric_mode == "policy_value_error"):
                cfg = _build_dd_waterfall_control_config(params)
                waterfall = DDWaterfallISP_Control(
                    mdp.P,
                    p_hat,
                    mdp.R,
                    mdp.gamma,
                    h_init=float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)),
                    m_max=int(cfg.get("m_max", _DD_WATERFALL_M_MAX_DEFAULT)),
                    decay_factor=float(cfg.get("decay_factor", _WATERFALL_DECAY_FACTOR_DEFAULT)),
                    inner_vi_tol=float(cfg.get("inner_vi_tol", _WATERFALL_INNER_VI_TOL_DEFAULT)),
                    inner_vi_max_iters=int(
                        cfg.get("inner_vi_max_iters", _WATERFALL_INNER_VI_MAX_ITERS_DEFAULT)
                    ),
                )
                queries, residuals, value_trace, policy_trace = waterfall.run_with_policy_trace(
                    max_queries=max_queries
                )
                value_errors = [
                    compute_value_error(
                        v,
                        target_value,
                        norm_name=value_norm,
                        normalized=value_normalized,
                    )
                    for v in value_trace
                ]
                if len(policy_trace) > 0:
                    policy_value_errors = _policy_value_errors_from_trace(
                        mdp=mdp,
                        queries=queries,
                        policy_trace=policy_trace,
                        target_value=target_value,
                        target_bundle=target_bundle,
                        norm_name=value_norm,
                        normalized=value_normalized,
                        evaluation_pipeline=evaluation_pipeline,
                        offline_eval_root=offline_eval_root,
                        algorithm_name=name,
                        lambda_value=lambda_value,
                        seed=seed,
                    )
                else:
                    policy_trace = jnp.zeros((0, mdp.P.shape[0]), dtype=jnp.int32)
                    policy_value_errors = []
            else:
                hit = _run_single_control_algorithm(
                    name,
                    base_name,
                    params,
                    mdp,
                    p_hat,
                    max_queries,
                    effect_threshold,
                    compute_resid,
                    seed,
                    vi_control_cls=vi_control_cls,
                    msvi_control_cls=msvi_control_cls,
                    isp_control_cls=isp_control_cls,
                    aisp_control_cls=aisp_control_cls,
                    shadow_isp_control_cls=shadow_isp_control_cls,
                    waterfall_isp_control_cls=waterfall_isp_control_cls,
                    pick_isp_params_control_fn=pick_isp_params_control_fn,
                    seeded_shadow_init_fn=seeded_shadow_init_fn,
                )
                results[name] = hit
                continue

            metric_errors = policy_value_errors if metric_mode == "policy_value_error" else residuals
            metric_threshold = threshold if metric_mode == "policy_value_error" else effect_threshold
            results[name] = _get_hit_time(queries, metric_errors, metric_threshold, max_queries)
            if emit_traces:
                traces[name] = _build_control_trace(
                    queries=queries,
                    residuals=residuals,
                    value_errors=value_errors,
                    policy_value_errors=policy_value_errors,
                    policy_trace=policy_trace,
                    threshold=threshold if metric_mode == "policy_value_error" else effect_threshold,
                    metric_mode=metric_mode,
                    value_norm=value_norm,
                    value_normalized=value_normalized,
                    max_queries=max_queries,
                    target_bundle=target_bundle,
                )
        except ValueError:
            raise
        except Exception:
            results[name] = max_queries + 100

    return results, traces


def _run_single_control_algorithm(
    name: str,
    base_name: str,
    params: Dict[str, Any],
    mdp: MDP,
    p_hat: Any,
    max_queries: int,
    effect_threshold: float,
    compute_resid: Callable[[Any], float],
    seed: int,
    *,
    vi_control_cls: type,
    msvi_control_cls: type,
    isp_control_cls: type,
    aisp_control_cls: type,
    shadow_isp_control_cls: type,
    waterfall_isp_control_cls: type,
    pick_isp_params_control_fn: Callable[..., tuple[Any, Any]],
    seeded_shadow_init_fn: Callable[[int, int], Any],
) -> int:
    if base_name == "vi":
        vi = vi_control_cls(mdp)
        trace, _ = vi.run(n_iter=max_queries)
        res = [compute_resid(v) for v in trace]
        return next((i + 1 for i, resid in enumerate(res) if resid <= effect_threshold), max_queries + 1)

    if base_name == "osvi":
        msvi = msvi_control_cls(
            mdp,
            p_hat,
            surrogate_policy_iteration_steps=_osvi_surrogate_policy_iteration_steps(params),
        )
        trace, _ = msvi.run(n_iter=max_queries)
        res = [compute_resid(v) for v in trace]
        return next((i + 1 for i, resid in enumerate(res) if resid <= effect_threshold), max_queries + 1)

    if base_name == "isp":
        isp = isp_control_cls(mdp, p_hat)
        default_h, default_j = pick_isp_params_control_fn(mdp, p_hat)
        h = params.get("h", default_h)
        j_val = params.get("J", default_j)
        q_tr, r_tr = isp.run(
            max_avps=max_queries,
            h=h,
            J=j_val,
            target_residual=effect_threshold,
        )
        return _find_hit_in_trace(q_tr, r_tr, effect_threshold, max_queries)

    if base_name == "aisp":
        cfg = _build_aisp_config(params)
        aisp = aisp_control_cls(mdp, p_hat)
        q_tr, r_tr = aisp.run(
            target_epsilon=float(cfg.get("target_epsilon", _AISP_TARGET_EPSILON_DEFAULT)),
            max_avps=max_queries,
        )
        return _find_hit_in_trace(q_tr, r_tr, effect_threshold, max_queries)

    if base_name == "shadow" or "dual" in name:
        cfg = _build_shadow_config(name, params)
        shadow = shadow_isp_control_cls(mdp, p_hat, **cfg)
        v_init = seeded_shadow_init_fn(mdp.P.shape[0], seed)
        q_tr, r_tr = shadow.run(
            max_avps=max_queries,
            target_residual=effect_threshold,
            V_init=v_init,
        )
        return _find_hit_in_trace(q_tr, r_tr, effect_threshold, max_queries)

    if base_name == "waterfall":
        waterfall = waterfall_isp_control_cls(
            mdp.P,
            p_hat,
            mdp.R,
            mdp.gamma,
            **_build_waterfall_config(params),
        )
        while waterfall.total_queries < max_queries:
            _, resid = waterfall.step()
            if resid <= effect_threshold:
                return waterfall.total_queries
            if getattr(waterfall, "stop_due_to_failed_floor", False):
                break
        return max_queries + 1

    if base_name == "geo_waterfall":
        cfg = _build_geo_waterfall_control_config(params)
        waterfall = GeoWaterfallISP_Control(
            mdp.P,
            p_hat,
            mdp.R,
            mdp.gamma,
            h_init=float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)),
            alpha=float(cfg.get("alpha", _GEO_WATERFALL_ALPHA_DEFAULT)),
            inner_vi_tol=float(cfg.get("inner_vi_tol", _WATERFALL_INNER_VI_TOL_DEFAULT)),
            inner_vi_max_iters=int(cfg.get("inner_vi_max_iters", _WATERFALL_INNER_VI_MAX_ITERS_DEFAULT)),
        )
        while waterfall.total_queries < max_queries:
            _, resid = waterfall.step()
            if resid <= effect_threshold:
                return waterfall.total_queries
        return max_queries + 1

    if base_name == "dd_waterfall":
        cfg = _build_dd_waterfall_control_config(params)
        waterfall = DDWaterfallISP_Control(
            mdp.P,
            p_hat,
            mdp.R,
            mdp.gamma,
            h_init=float(cfg.get("h_init", _WATERFALL_H_INIT_DEFAULT)),
            m_max=int(cfg.get("m_max", _DD_WATERFALL_M_MAX_DEFAULT)),
            decay_factor=float(cfg.get("decay_factor", _WATERFALL_DECAY_FACTOR_DEFAULT)),
            inner_vi_tol=float(cfg.get("inner_vi_tol", _WATERFALL_INNER_VI_TOL_DEFAULT)),
            inner_vi_max_iters=int(cfg.get("inner_vi_max_iters", _WATERFALL_INNER_VI_MAX_ITERS_DEFAULT)),
        )
        while waterfall.total_queries < max_queries:
            remaining = max_queries - waterfall.total_queries
            _, resid = waterfall.step(max_depth=remaining)
            if resid <= effect_threshold:
                return waterfall.total_queries
        return max_queries + 1

    logger.warning("Unknown control algorithm %s", name)
    return max_queries + 100


def _get_hit_time(q_list: List[Any], err_list: List[Any], thresh: float, max_queries: int) -> int:
    if not err_list:
        return max_queries + 100
    for i, err in enumerate(err_list):
        if err <= thresh:
            return q_list[i]
    return max_queries + 1


def _find_hit_in_trace(
    q_tr: List[Any], r_tr: List[Any], threshold: float, max_queries: int
) -> int:
    for i, resid in enumerate(r_tr):
        if resid <= threshold:
            return q_tr[i]
    return max_queries + 1


def _build_shadow_config(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"optimism": 1.0, "dynamic_J": False}
    if "dual" in name:
        cfg["dynamic_J"] = True
    cfg.update(params)
    valid_keys = {"optimism", "dynamic_J", "fixed_J"}
    return {key: value for key, value in cfg.items() if key in valid_keys}


def _build_waterfall_config(params: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if "h_init" in params:
        cfg["h_init"] = float(params["h_init"])
    if "h_min" in params:
        cfg["h_min"] = float(params["h_min"])
    if "stall_patience" in params:
        cfg["stall_patience"] = int(params["stall_patience"])
    if "decay_factor" in params:
        cfg["decay_factor"] = float(params["decay_factor"])
    if "inner_vi_tol" in params:
        cfg["inner_vi_tol"] = float(params["inner_vi_tol"])
    if "inner_vi_max_iters" in params:
        cfg["inner_vi_max_iters"] = int(params["inner_vi_max_iters"])
    return cfg


__all__ = [
    "DeterministicTaskSpec",
    "PreparedDeterministicTask",
    "execute_batched_control_tasks",
    "execute_batched_pe_tasks",
    "execute_prepared_deterministic_task",
    "normalize_algorithm_config",
    "parse_deterministic_task",
    "prepare_deterministic_task",
    "run_control_algorithms",
    "run_deterministic_task",
    "run_pe_algorithms",
]
