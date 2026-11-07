
"""JAX hyper-batch orchestration for maintained stochastic and deterministic runs."""

import hashlib
import inspect
import json
from dataclasses import replace
from functools import partial
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence, Callable, Mapping

import numpy as np

from besp.algorithms.jax.base import initial_model_estimate_pure
from besp.core.config import (
    canonical_algorithm_name,
    is_alt_isp_dyna_algorithm,
    is_gen_alt_isp_dyna_algorithm,
    is_arch_isp_dyna_algorithm,
    is_nr_rm_isp_dyna_algorithm,
    is_nr1_rm_isp_dyna_algorithm,
    is_nr1_staircase_waterfall_rm_isp_dyna_algorithm,
    is_nr1_geo_waterfall_rm_isp_dyna_algorithm,
    is_nr1_waterfall_rm_isp_dyna_algorithm,
    is_arch_waterfall_isp_dyna_algorithm,
    is_dd_geo_waterfall_isp_dyna_algorithm,
    is_elastic_dd_geo_waterfall_isp_dyna_algorithm,
    is_isp_dyna_algorithm,
    is_scheduled_waterfall_isp_dyna_algorithm,
    normalize_lr_type_name,
)
from besp.domains.jax_rl.error_modes import jax_rl_stochastic_model_error_code
from besp.core.artifact_io import save_json_payload
from besp.core.offline_eval import plan_offline_eval_layout
from besp.core.types import (
    CpuExperimentTask,
    ExperimentExecutionRow,
    HyperBatchTrace,
    normalize_sampling_contract,
)
from besp.runners.factory import create_environment_jax
from besp.runners.offline_eval import (
    clear_stream_checkpoint,
    evaluate_policy_capture_slab_start_state,
    initialize_policy_capture_bundle,
    load_offline_eval_manifest,
    load_start_state_history,
    load_stream_checkpoint,
    materialize_start_state_trace_from_history,
    materialize_start_state_trace_from_values,
    save_stream_checkpoint,
    write_offline_eval_phase_manifest,
    write_offline_eval_manifest,
    write_start_state_history_slab,
)
from besp.runners.hyper_batching import (
    assemble_algorithm_results,
    group_configs_by_algorithm,
    normalize_hyper_batch_configs,
    resolve_jax_algorithm_class,
    validate_packed_stochastic_group,
)
from .gpu_worker import run_deterministic_jax_tasks, run_deterministic_jax_tasks_multi_device

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

# Handle shard_map location (stable vs experimental)
try:
    from jax import shard_map  # JAX 0.4.36+
except ImportError:
    from jax.experimental.shard_map import shard_map  # JAX 0.4.35 and earlier

_SHARDMAP_SIGNATURE = inspect.signature(shard_map)
_SHARDMAP_SUPPORTS_CHECK_VMA = "check_vma" in _SHARDMAP_SIGNATURE.parameters
_SHARDMAP_SUPPORTS_CHECK_REP = "check_rep" in _SHARDMAP_SIGNATURE.parameters
# Patch jax.shard_map so @partial(jax.shard_map, ...) works
jax.shard_map = shard_map

logger = logging.getLogger(__name__)


def _default_replay_tape_root(*, offline_root_path: Path | None) -> Path:
    if offline_root_path is not None:
        return offline_root_path.parent / "replay_tapes"
    return Path("outputs") / "autoresearch" / "jax_rl" / "replay_tapes"


def _replay_tape_bundle_root(
    *,
    replay_root: Path,
    env_name: str,
    env_params: dict[str, Any],
    seed_values: Sequence[int],
    tape_steps: int,
) -> Path:
    from besp.domains.jax_rl import replay_tape_contract_env_params

    contract_env_params = replay_tape_contract_env_params(env_params, contract_scope="transition_law")
    params_json = json.dumps(contract_env_params, sort_keys=True, separators=(",", ":"))
    params_hash = hashlib.sha1(params_json.encode("utf-8")).hexdigest()[:12]
    params_token = "no_env_params" if not contract_env_params else f"env_{params_hash}"
    seed_token = "seeds_" + "_".join(str(int(value)) for value in seed_values)
    return (
        replay_root
        / canonical_algorithm_name(env_name)
        / "transition_law"
        / "uniform_sa"
        / params_token
        / seed_token
        / f"tape_{int(tape_steps)}"
    )


def _replay_row_index_for_envs(
    *,
    env_seed_values: jnp.ndarray,
    unique_seed_values: Sequence[int],
) -> jnp.ndarray:
    lookup = {int(seed): index for index, seed in enumerate(unique_seed_values)}
    return jnp.asarray(
        [lookup[int(seed)] for seed in np.asarray(env_seed_values, dtype=np.int32).tolist()],
        dtype=jnp.int32,
    )


def _pad_edge_1d(arr: jnp.ndarray, padding: int) -> jnp.ndarray:
    if padding <= 0:
        return arr
    tail = jnp.repeat(arr[-1:], padding, axis=0)
    return jnp.concatenate([arr, tail], axis=0)


def _log_runtime_debug() -> None:
    """Emit environment diagnostics only when debug logging is enabled."""
    import os

    logger.debug("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    logger.debug("LD_LIBRARY_PATH=%s", os.environ.get("LD_LIBRARY_PATH"))
    try:
        logger.debug("JAX Devices=%s", jax.devices())
        logger.debug("JAX Backend=%s", jax.default_backend())
    except Exception as exc:
        logger.debug("Error listing JAX devices: %s", exc)


def _maybe_emit_progress(
    *,
    progress_callback: Callable[[str, dict[str, Any]], None] | None,
    progress_flush_every_evals: int | None,
    algo_name: str,
    cfgs,
    data: jnp.ndarray,
    n_seeds: int,
    x_axis: list[int],
    eval_count: int,
    steps_completed: int,
    total_samples: int,
    started_at: float,
    device_count: int,
    backend_mode: str,
    diagnostics: dict[str, jnp.ndarray] | None = None,
) -> None:
    if progress_callback is None:
        return
    flush_every = max(int(progress_flush_every_evals or 0), 1)
    if eval_count < flush_every and steps_completed < total_samples:
        return
    if eval_count % flush_every != 0 and steps_completed < total_samples:
        return

    elapsed_seconds = time.perf_counter() - started_at
    projected_remaining_seconds: float | None = None
    if steps_completed > 0:
        projected_remaining_seconds = elapsed_seconds * max(total_samples - steps_completed, 0) / steps_completed
    payload = {
        "eval_checkpoints_completed": eval_count,
        "steps_completed": steps_completed,
        "elapsed_seconds": elapsed_seconds,
        "projected_remaining_seconds": projected_remaining_seconds,
        "device_count": device_count,
        "backend_mode": backend_mode,
        "batch_shape": {"configs": len(cfgs), "seeds": n_seeds},
        "x_axis": list(x_axis),
        "results": assemble_algorithm_results(
            cfgs=cfgs,
            data=jax.device_get(data),
            n_seeds=n_seeds,
            x_axis=x_axis,
            diagnostics=jax.device_get(diagnostics) if diagnostics is not None else None,
        ),
    }
    progress_callback(algo_name, payload)


def _capture_policy_slab_single_device(
    *,
    state,
    mdp_data,
    params,
    step_fn,
    policy_fn,
    capture_start: int,
    capture_stop: int,
) -> tuple[Any, jax.Array]:
    capture_count = int(capture_stop) - int(capture_start)
    if capture_count <= 0:
        raise ValueError("capture_stop must be greater than capture_start.")

    def _scan_step(carry_state, k_value):
        next_state, _ = step_fn(carry_state, mdp_data, params, k_value)
        next_policy = jnp.asarray(policy_fn(next_state, params), dtype=jnp.uint8)
        return next_state, next_policy

    if capture_start == 0:
        initial_policy = jnp.asarray(policy_fn(state, params), dtype=jnp.uint8)
        if capture_count == 1:
            return state, initial_policy[None, ...]
        step_indices = jnp.arange(0, capture_count - 1, dtype=jnp.int32)
        final_state, scanned_policies = jax.lax.scan(_scan_step, state, step_indices)
        policies = jnp.concatenate([initial_policy[None, ...], scanned_policies], axis=0)
        return final_state, policies

    step_indices = jnp.arange(capture_start - 1, capture_stop - 1, dtype=jnp.int32)
    final_state, policies = jax.lax.scan(_scan_step, state, step_indices)
    return final_state, policies


def _capture_start_state_slab_single_device(
    *,
    state,
    mdp_data,
    params,
    step_fn,
    policy_fn,
    capture_start: int,
    capture_stop: int,
    mdp_P: jnp.ndarray,
    mdp_R: jnp.ndarray,
    gamma: float,
    solve_chunk_size: int,
) -> tuple[Any, jax.Array]:
    next_state, slab_policies = _capture_policy_slab_single_device(
        state=state,
        mdp_data=mdp_data,
        params=params,
        step_fn=step_fn,
        policy_fn=policy_fn,
        capture_start=capture_start,
        capture_stop=capture_stop,
    )
    start_state_values = evaluate_policy_capture_slab_start_state(
        policy_slab=slab_policies,
        mdp_P=mdp_P,
        mdp_R=mdp_R,
        gamma=gamma,
        chunk_size=solve_chunk_size,
        start_state_idx=0,
    )
    return next_state, start_state_values


def _materialize_hyper_batch_trace_from_offline_values(
    *,
    value_history: jnp.ndarray | Any,
    sample_axis: Sequence[int],
    eval_freq: int,
    n_seeds: int,
    cfgs,
) -> dict[str, HyperBatchTrace]:
    payload = materialize_start_state_trace_from_values(
        value_history=value_history,
        sample_axis=sample_axis,
        eval_freq=eval_freq,
        start_state_idx=0,
    )
    return assemble_algorithm_results(
        cfgs=cfgs,
        data=jnp.asarray(payload["seeds"], dtype=jnp.float64),
        n_seeds=n_seeds,
        x_axis=payload["x_axis"].tolist(),
        diagnostics=None,
    )


def _materialize_hyper_batch_trace_from_start_state_history(
    *,
    start_state_history: jnp.ndarray | Any,
    sample_axis: Sequence[int],
    eval_freq: int,
    n_seeds: int,
    cfgs,
) -> dict[str, HyperBatchTrace]:
    payload = materialize_start_state_trace_from_history(
        start_state_history=start_state_history,
        sample_axis=sample_axis,
        eval_freq=eval_freq,
    )
    return assemble_algorithm_results(
        cfgs=cfgs,
        data=jnp.asarray(payload["seeds"], dtype=jnp.float64),
        n_seeds=n_seeds,
        x_axis=payload["x_axis"].tolist(),
        diagnostics=None,
    )


def _expand_env_seed_values(cfgs, *, n_seeds: int, seed_values: Sequence[int] | None) -> jnp.ndarray:
    seeds = list(range(n_seeds)) if seed_values is None else [int(value) for value in seed_values]
    if len(seeds) != n_seeds:
        raise ValueError(f"Expected {n_seeds} seed values, received {len(seeds)}.")
    expanded = [seed for _cfg in cfgs for seed in seeds]
    return jnp.asarray(expanded, dtype=jnp.int32)


def _validate_isp_dyna_hyper_batch_contract(
    configs,
    *,
    samples: int,
    eval_freq: int,
) -> None:
    for cfg in configs:
        if is_isp_dyna_algorithm(cfg.algorithm):
            span = int(cfg.J) * int(cfg.batch_size or 0)
            if span <= 0 or samples % span != 0:
                raise ValueError(
                    "ISP-Dyna requires the physical sample budget to be divisible by J * batch_size; "
                    f"received samples={samples}, J={cfg.J}, batch_size={cfg.batch_size}."
                )
            if eval_freq % span != 0:
                raise ValueError(
                    "ISP-Dyna hyper-batch traces must evaluate only on completed manuscript phases; "
                    "eval_freq must be divisible by J * batch_size. "
                    f"received eval_freq={eval_freq}, J={cfg.J}, batch_size={cfg.batch_size}."
                )
            continue
        if is_arch_isp_dyna_algorithm(cfg.algorithm):
            span = int(cfg.J)
            if span <= 0 or samples % span != 0:
                raise ValueError(
                    "Arch-ISP-Dyna requires the physical sample budget to be divisible by J; "
                    f"received samples={samples}, J={cfg.J}."
                )
            if eval_freq % span != 0:
                raise ValueError(
                    "Arch-ISP-Dyna hyper-batch traces must evaluate only on completed archive phases; "
                    "eval_freq must be divisible by J. "
                    f"received eval_freq={eval_freq}, J={cfg.J}."
                )
            continue
        if is_scheduled_waterfall_isp_dyna_algorithm(cfg.algorithm):
            span = int(cfg.J)
            if span <= 0 or samples % span != 0:
                raise ValueError(
                    "Scheduled-Waterfall-ISP-Dyna requires the physical sample budget to be divisible by J; "
                    f"received samples={samples}, J={cfg.J}."
                )
            if eval_freq % span != 0:
                raise ValueError(
                    "Scheduled-Waterfall-ISP-Dyna hyper-batch traces must evaluate only on completed cascade phases; "
                    "eval_freq must be divisible by J. "
                    f"received eval_freq={eval_freq}, J={cfg.J}."
                )
            continue
        if is_arch_waterfall_isp_dyna_algorithm(cfg.algorithm):
            span = int(cfg.J)
            if span <= 0 or samples % span != 0:
                raise ValueError(
                    "Arch-Waterfall-ISP-Dyna requires the physical sample budget to be divisible by J; "
                    f"received samples={samples}, J={cfg.J}."
                )
            if eval_freq % span != 0:
                raise ValueError(
                    "Arch-Waterfall-ISP-Dyna hyper-batch traces must evaluate only on completed cascade phases; "
                    "eval_freq must be divisible by J. "
                    f"received eval_freq={eval_freq}, J={cfg.J}."
                )
            continue
        if is_dd_geo_waterfall_isp_dyna_algorithm(cfg.algorithm) or is_elastic_dd_geo_waterfall_isp_dyna_algorithm(cfg.algorithm):
            batch_size = int(cfg.batch_size or 0)
            if batch_size <= 0 or samples % batch_size != 0:
                raise ValueError(
                    f"{cfg.algorithm} requires the physical sample budget to be divisible by batch_size; "
                    f"received samples={samples}, batch_size={cfg.batch_size}."
                )
            if eval_freq % batch_size != 0:
                raise ValueError(
                    f"{cfg.algorithm} evaluates on batch-aligned sample grids only; "
                    "eval_freq must be divisible by batch_size. "
                    f"received eval_freq={eval_freq}, batch_size={cfg.batch_size}."
                )


def _waterfall_trace_diagnostics_from_state(state) -> dict[str, jnp.ndarray]:
    return {
        "status_trace": jnp.asarray(state.waterfall_last_status, dtype=jnp.int32),
        "depth_trace": jnp.asarray(state.waterfall_inner_depth, dtype=jnp.int32),
        "h_trace": jnp.asarray(state.waterfall_h, dtype=jnp.float64),
        "g_candidate_trace": jnp.asarray(state.waterfall_g_candidate, dtype=jnp.float64),
        "g_best_trace": jnp.asarray(state.waterfall_g_best, dtype=jnp.float64),
        "reward_abs_max_trace": jnp.asarray(state.waterfall_reward_abs_max, dtype=jnp.float64),
        "d_base_trace": jnp.asarray(
            state.waterfall_d_base
            if state.waterfall_d_base is not None
            else jnp.zeros_like(state.waterfall_h),
            dtype=jnp.float64,
        ),
        "d_refine_trace": jnp.asarray(
            state.waterfall_d_refine
            if state.waterfall_d_refine is not None
            else jnp.zeros_like(state.waterfall_h),
            dtype=jnp.float64,
        ),
        "normalized_d_base_trace": jnp.asarray(state.waterfall_g_best, dtype=jnp.float64),
        "normalized_d_refine_trace": jnp.asarray(state.waterfall_g_candidate, dtype=jnp.float64),
        "committed_v0_trace": jnp.asarray(state.V[:, 0], dtype=jnp.float64),
    }


def _alt_isp_trace_diagnostics_from_state(state) -> dict[str, jnp.ndarray]:
    return {
        "effective_h_trace": jnp.asarray(state.waterfall_h, dtype=jnp.float64),
        "empirical_error_trace": jnp.asarray(state.waterfall_g_candidate, dtype=jnp.float64),
        "visited_fraction_trace": jnp.asarray(state.R_E_factor, dtype=jnp.float64),
        "inner_depth_trace": jnp.asarray(state.waterfall_inner_depth, dtype=jnp.int32),
        "reward_abs_max_trace": jnp.asarray(state.waterfall_reward_abs_max, dtype=jnp.float64),
        "committed_v0_trace": jnp.asarray(state.V[:, 0], dtype=jnp.float64),
    }


def _rm_isp_trace_diagnostics_from_state(state) -> dict[str, jnp.ndarray]:
    defect_r = state.defect_r if state.defect_r is not None else state.R_hat
    working_v = state.working_V if state.working_V is not None else state.V
    return {
        "inner_depth_trace": jnp.asarray(state.waterfall_inner_depth, dtype=jnp.int32),
        "defect_abs_max_trace": jnp.max(jnp.abs(defect_r), axis=(1, 2)),
        "working_v0_trace": jnp.asarray(working_v[:, 0], dtype=jnp.float64),
        "committed_v0_trace": jnp.asarray(state.V[:, 0], dtype=jnp.float64),
    }


def _nr_rm_isp_trace_diagnostics_from_state(state) -> dict[str, jnp.ndarray]:
    defect_r = state.defect_r if state.defect_r is not None else state.r_k
    defect_init = (
        state.defect_init
        if state.defect_init is not None
        else jnp.zeros_like(defect_r)
    )
    working_v = state.working_V if state.working_V is not None else state.V
    reset_indicator = jnp.zeros((state.V.shape[0],), dtype=jnp.float64)
    return {
        "defect_abs_max_trace": jnp.max(jnp.abs(defect_r), axis=(1, 2)),
        "defect_baseline_gap_abs_max_trace": jnp.max(jnp.abs(defect_r - defect_init), axis=(1, 2)),
        "working_v0_trace": jnp.asarray(working_v[:, 0], dtype=jnp.float64),
        "committed_v0_trace": jnp.asarray(state.V[:, 0], dtype=jnp.float64),
        "reset_indicator_trace": reset_indicator,
    }


def run_hyper_batched_deterministic_tasks(
    task_args_list: Sequence[CpuExperimentTask | Dict[str, Any]],
) -> List[ExperimentExecutionRow]:
    """Run maintained deterministic tasks through the active hyper-batch JAX surface."""

    tasks = list(task_args_list)
    if not tasks:
        return []

    n_devices = len(jax.local_devices())
    if n_devices > 1:
        print(f"HyperBatch Deterministic Backend: JAX (Devices: {n_devices}, Mode=MULTI_DEVICE)")
        return run_deterministic_jax_tasks_multi_device(tasks)

    print(f"HyperBatch Deterministic Backend: JAX (Devices: {n_devices}, Mode=SINGLE_DEVICE)")
    return run_deterministic_jax_tasks(tasks)

def run_hyper_batched_task(
    env_name: str,
    configs: List[Dict[str, Any]],
    n_seeds: int,
    algorithms: List[str],
    samples: int = 5000,
    eval_freq: int = 50,
    env_params: Optional[Dict] = None,
    backend: str = "jax",
    use_fused_loop: bool = False,
    use_shard_map: bool = True,
    seed_values: Sequence[int] | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    progress_flush_every_evals: int | None = None,
    evaluation_pipeline: str = "online",
    offline_eval_root: str | Path | None = None,
    sampling_contract: Mapping[str, Any] | None = None,
) -> Dict[str, Dict[str, HyperBatchTrace]]:
    """Run a synchronized JAX hyper-batch experiment."""
    if env_params is None:
        env_params = {}
    if backend != "jax":
        raise ValueError("run_hyper_batched_task supports only backend='jax' in the active codebase.")
    resolved_evaluation_pipeline = str(evaluation_pipeline).strip().lower().replace("-", "_")
    if resolved_evaluation_pipeline not in {"online", "train_then_offline"}:
        raise ValueError("evaluation_pipeline must be one of: online, train_then_offline.")
    offline_root_path = Path(offline_eval_root) if offline_eval_root is not None else None

    env_params = dict(env_params)
    effective_gamma = env_params.get("gamma", configs[0].get("gamma", 0.99))
    env_kwargs = {key: value for key, value in env_params.items() if key != "gamma"}
    mdp = create_environment_jax(env_name, gamma=effective_gamma, **env_kwargs)
    _log_runtime_debug()
    mdp_P_jax = jnp.asarray(mdp.P, dtype=jnp.float64)
    mdp_R_jax = jnp.asarray(mdp.R, dtype=jnp.float64)
    n_s, n_a, _ = mdp.P.shape

    from besp.domains.jax_rl import (
        AgentState,
        AlgorithmParams,
        MDPData,
        ensure_uniform_sa_replay_tape,
        evaluate_batched_pure,
        initialize_env_rng_keys_and_states_from_seed_values,
    )

    resolved_seed_values_global = (
        tuple(range(int(n_seeds)))
        if seed_values is None
        else tuple(int(value) for value in seed_values)
    )
    resolved_sampling_contract = normalize_sampling_contract(
        requested=sampling_contract,
        seed_values=resolved_seed_values_global,
        consumed_steps=int(samples),
    )
    replay_bundle = None
    replay_states_jax = None
    replay_actions_jax = None
    replay_next_states_jax = None
    replay_manifest_path = resolved_sampling_contract.replay_manifest_path
    replay_checksum = resolved_sampling_contract.tape_checksum
    if resolved_sampling_contract.mode == "replay_tape":
        if replay_manifest_path:
            replay_bundle_root = (
                Path(replay_manifest_path).expanduser().resolve().parent
                if Path(replay_manifest_path).name == "manifest.json"
                else Path(replay_manifest_path).expanduser().resolve()
            )
        else:
            replay_bundle_root = _replay_tape_bundle_root(
                replay_root=_default_replay_tape_root(offline_root_path=offline_root_path),
                env_name=env_name,
                env_params=env_params,
                seed_values=resolved_seed_values_global,
                tape_steps=resolved_sampling_contract.tape_steps,
            )
        replay_bundle = ensure_uniform_sa_replay_tape(
            bundle_root=replay_bundle_root,
            mdp_P=np.asarray(mdp.P, dtype=np.float64),
            mdp_R=np.asarray(mdp.R, dtype=np.float64),
            env_name=env_name,
            env_params=env_params,
            gamma=float(effective_gamma),
            seed_values=resolved_seed_values_global,
            tape_steps=resolved_sampling_contract.tape_steps,
        )
        replay_states_jax = jnp.asarray(replay_bundle.states, dtype=jnp.int32)
        replay_actions_jax = jnp.asarray(replay_bundle.actions, dtype=jnp.int32)
        replay_next_states_jax = jnp.asarray(replay_bundle.next_states, dtype=jnp.int32)
        replay_manifest_path = str((replay_bundle_root / "manifest.json").resolve())
        replay_checksum = replay_bundle.manifest.tape_checksum
    sampling_contract_payload = {
        **resolved_sampling_contract.as_mapping(),
        "seed_values": list(resolved_seed_values_global),
        "replay_manifest_path": replay_manifest_path,
        "tape_checksum": replay_checksum,
    }

    try:
        n_devices = jax.local_device_count()
    except Exception as e:
        logger.warning(f"JAX device count failed: {e}. Defaulting to 1.")
        n_devices = 1

    use_multi_gpu = n_devices > 1
    if use_fused_loop and progress_callback is not None:
        logger.warning("Progress flushing requires non-fused hyper-batch loops; disabling use_fused_loop.")
        use_fused_loop = False
    if use_multi_gpu:
        if use_shard_map:
            logger.info("HyperBatch Backend: JAX (Devices: %s, Mode=SHARD_MAP)", n_devices)
        else:
            logger.info("HyperBatch Backend: JAX (Devices: %s, Mode=PMAP)", n_devices)
    else:
        logger.info("HyperBatch Backend: JAX (Devices: %s, Mode=SINGLE_DEVICE)", n_devices)

    
    normalized_configs = normalize_hyper_batch_configs(configs)
    _validate_isp_dyna_hyper_batch_contract(
        normalized_configs,
        samples=samples,
        eval_freq=eval_freq,
    )
    algo_configs = group_configs_by_algorithm(normalized_configs, algorithms)
    
    # 3. Build and Run Each Algorithm
    all_results = {}
    
    for algo_name in algorithms:
        cfgs = algo_configs.get(algo_name, [])
        if not cfgs:
            logger.warning(f"No configs for {algo_name}, skipping.")
            continue
        algo_started_at = time.perf_counter()
        is_alt_isp_algo = is_alt_isp_dyna_algorithm(algo_name)
        is_gen_alt_isp_algo = is_gen_alt_isp_dyna_algorithm(algo_name)
        is_alt_isp_family_algo = is_alt_isp_algo or is_gen_alt_isp_algo
        is_rm_isp_algo = False
        is_nr_rm_isp_algo = is_nr_rm_isp_dyna_algorithm(algo_name)
        is_nr1_rm_isp_algo = is_nr1_rm_isp_dyna_algorithm(algo_name)
        is_nr1_staircase_waterfall_algo = is_nr1_staircase_waterfall_rm_isp_dyna_algorithm(algo_name)
        is_nr1_geo_waterfall_algo = is_nr1_geo_waterfall_rm_isp_dyna_algorithm(algo_name)
        is_nr1_waterfall_algo = is_nr1_waterfall_rm_isp_dyna_algorithm(algo_name)
        is_nr_family_algo = is_nr_rm_isp_algo or is_nr1_rm_isp_algo or is_nr1_waterfall_algo
        if is_nr1_waterfall_algo and use_multi_gpu:
            raise NotImplementedError(
                "Waterfall-NR1 variants currently support only the single-device path; "
                "multi-device (pmap/shard_map) wiring is not yet implemented."
            )
        is_dd_geo_algo = is_dd_geo_waterfall_isp_dyna_algorithm(algo_name) or is_elastic_dd_geo_waterfall_isp_dyna_algorithm(algo_name)
        is_arch_waterfall_algo = is_arch_waterfall_isp_dyna_algorithm(algo_name)
        is_scheduled_waterfall_algo = is_scheduled_waterfall_isp_dyna_algorithm(algo_name)
        is_schedule_waterfall_family = is_scheduled_waterfall_algo or is_arch_waterfall_algo
        diagnostics_from_state = (
            _waterfall_trace_diagnostics_from_state
            if is_dd_geo_algo
            else (
                _alt_isp_trace_diagnostics_from_state
                if is_alt_isp_family_algo
                else (
                    _nr_rm_isp_trace_diagnostics_from_state
                    if is_nr_family_algo
                    else _rm_isp_trace_diagnostics_from_state
                )
            )
        )
        if is_dd_geo_algo and use_fused_loop:
            logger.warning(
                "DD-Geo-Waterfall-ISP-Dyna progress and diagnostics currently use the non-fused outer loop."
            )
        if is_dd_geo_algo:
            batch_sizes_for_algo = {int(cfg.batch_size or 0) for cfg in cfgs}
            if len(batch_sizes_for_algo) != 1:
                raise ValueError(
                    "DD-Geo-Waterfall-ISP-Dyna currently requires one shared batch_size per hyper-batch algorithm group."
                )
        compatibility_key = validate_packed_stochastic_group(cfgs)
        model_error_mode_scalar = int(jax_rl_stochastic_model_error_code(cfgs[0].error_mode))
        logical_batch_size = int(cfgs[0].batch_size or 1) if is_dd_geo_algo else 1
        logical_sample_stride = logical_batch_size
        logical_samples = samples // logical_batch_size if is_dd_geo_algo else samples
        logical_eval_freq = eval_freq // logical_batch_size if is_dd_geo_algo else eval_freq
        if is_nr1_rm_isp_algo or is_nr1_waterfall_algo:
            logical_sample_stride = max(int(cfgs[0].J or 1), 1)
            logical_samples = samples
            logical_eval_freq = eval_freq
        physical_total_samples = logical_samples * logical_sample_stride
        full_sample_axis = tuple(step * logical_sample_stride for step in range(logical_samples + 1))
        sampling_contract_payload = {
            **sampling_contract_payload,
            "logical_steps": int(logical_samples),
            "sample_stride": int(logical_sample_stride),
            "logical_eval_freq": int(logical_eval_freq),
            "consumed_steps": int(physical_total_samples),
        }
        algo_use_fused_loop = use_fused_loop and not is_dd_geo_algo
        
        nr_mode_scalar = 0
        if is_nr_family_algo:
            nr_mode_scalar = {
                "limit": 0,
                "fixed_h": 1,
                "scheduled_h": 2,
                "waterfall_staircase": 1,
                "waterfall_geo": 1,
            }[compatibility_key.h_profile_kind]
        if (
            resolved_sampling_contract.mode == "replay_tape"
            and int(sampling_contract_payload["consumed_steps"]) < physical_total_samples
        ):
            raise ValueError(
                "sampling_contract.consumed_steps is shorter than the required physical replay "
                f"prefix for {algo_name}: required={physical_total_samples}, "
                f"received={sampling_contract_payload['consumed_steps']}."
            )

        num_envs_total = len(cfgs) * n_seeds
        env_seed_arr = _expand_env_seed_values(cfgs, n_seeds=n_seeds, seed_values=seed_values)
        replay_row_index_arr = (
            _replay_row_index_for_envs(
                env_seed_values=env_seed_arr,
                unique_seed_values=resolved_seed_values_global,
            )
            if resolved_sampling_contract.mode == "replay_tape"
            else None
        )
        replay_row_base = (
            replay_row_index_arr
            if replay_row_index_arr is not None
            else jnp.zeros_like(env_seed_arr, dtype=jnp.int32)
        )
        
        # Build parameter vectors for THIS algorithm's configs
        lambdas, gammas, lrs, delays, h_vals, h_floor_vals, h_terminal_vals, nr_mode_vals, j_vals, planning_steps, batch_sizes, lr_schedules, decay_factors, epsilons = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        waterfall_h_init_vals: list[float] = []
        waterfall_h_min_vals: list[float] = []
        waterfall_v_max_vals: list[float] = []
        waterfall_decay_factor_shared = 0.5
        waterfall_stall_patience_shared = 0
        waterfall_alpha_shared = 1.0
        waterfall_divergence_mult_shared = 2.5
        waterfall_schedule_kind_shared = 0
        for cfg in cfgs:
            for _ in range(n_seeds):
                lambdas.append(cfg.lambda_value)
                gammas.append(cfg.gamma)
                lrs.append(cfg.start_lr)
                delays.append(cfg.delay)
                if is_nr1_waterfall_algo:
                    h_vals.append(float(cfg.h_init if cfg.h_init is not None else 1.0))
                else:
                    h_vals.append(
                        cfg.h_target
                        if is_alt_isp_family_algo and cfg.h_target is not None
                        else (
                            cfg.h
                            if cfg.h is not None
                            else (
                                cfg.h_max
                                if cfg.h_max is not None
                                else (cfg.h_init if cfg.h_init is not None else 1.0 / max(1e-9, 1.0 - cfg.gamma))
                            )
                        )
                    )
                h_floor_vals.append(cfg.h_min if cfg.h_min is not None else 1.0)
                if is_nr_family_algo and cfg.h_floor is not None:
                    h_floor_vals[-1] = cfg.h_floor
                if is_nr1_waterfall_algo:
                    waterfall_h_init_vals.append(float(cfg.h_init if cfg.h_init is not None else 1.0))
                    waterfall_h_min_vals.append(float(cfg.h_min if cfg.h_min is not None else 1.0))
                    waterfall_v_max_vals.append(float(cfg.v_max))
                else:
                    waterfall_h_init_vals.append(1.0)
                    waterfall_h_min_vals.append(1.0)
                    waterfall_v_max_vals.append(0.0)
                h_terminal_vals.append(
                    cfg.h_terminal
                    if cfg.h_terminal is not None
                    else (
                        cfg.h
                        if cfg.h is not None
                        else (
                            cfg.h_init
                            if cfg.h_init is not None
                            else 1.0
                        )
                    )
                )
                nr_mode_vals.append(nr_mode_scalar)
                j_vals.append(cfg.J_max if is_dd_geo_algo else cfg.J)
                planning_steps.append(cfg.planning_steps)
                batch_sizes.append(cfg.batch_size or 1)
                lr_schedules.append(0 if normalize_lr_type_name(cfg.lr_type) == "constant_delay" else 1)
                decay_factors.append(
                    cfg.decay_factor if not is_schedule_waterfall_family else cfg.schedule_decay_rate
                )
                epsilons.append(cfg.epsilon)
        
        # Parameter arrays (explicit float64 for JAX 0.4.35 dtype consistency)
        lam_arr = jnp.asarray(lambdas, dtype=jnp.float64)
        gamma_arr = jnp.asarray(gammas, dtype=jnp.float64)
        lr_arr = jnp.asarray(lrs, dtype=jnp.float64)
        delay_arr = jnp.asarray(delays, dtype=jnp.float64)
        h_arr = jnp.asarray(h_vals, dtype=jnp.float64)
        h_floor_arr = jnp.asarray(h_floor_vals, dtype=jnp.float64)
        h_terminal_arr = jnp.asarray(h_terminal_vals, dtype=jnp.float64)
        nr_mode_arr = jnp.asarray(nr_mode_vals, dtype=jnp.int32)
        j_arr = jnp.asarray(j_vals, dtype=jnp.int32)
        ps_arr = jnp.asarray(planning_steps, dtype=jnp.int32)
        batch_arr = jnp.asarray(batch_sizes, dtype=jnp.int32)
        sched_arr = jnp.asarray(lr_schedules, dtype=jnp.int32)
        decay_arr = jnp.asarray(decay_factors, dtype=jnp.float64)
        epsilon_arr = jnp.asarray(epsilons, dtype=jnp.float64)
        waterfall_h_init_arr = jnp.asarray(waterfall_h_init_vals, dtype=jnp.float64)
        waterfall_h_min_arr = jnp.asarray(waterfall_h_min_vals, dtype=jnp.float64)
        waterfall_v_max_arr = jnp.asarray(waterfall_v_max_vals, dtype=jnp.float64)
        if is_nr1_waterfall_algo:
            waterfall_decay_factor_shared = float(cfgs[0].decay_factor or 0.5)
            waterfall_stall_patience_shared = int(cfgs[0].stall_patience or 0)
            waterfall_alpha_shared = float(cfgs[0].alpha or 1.0)
            waterfall_divergence_mult_shared = float(cfgs[0].divergence_mult)
            waterfall_schedule_kind_shared = 0 if is_nr1_staircase_waterfall_algo else 1
        max_j_scalar = int(jnp.max(j_arr))
        max_batch_scalar = int(jnp.max(batch_arr))
        decay_scalar = float(decay_arr[0])
        nr_schedule_shape_scalar = 0
        nr_schedule_finish_frac_scalar = 1.0
        if is_nr_family_algo:
            nr_schedule_shape_scalar = 0 if str(cfgs[0].h_schedule_shape) == "exponential" else 1
            nr_schedule_finish_frac_scalar = float(cfgs[0].h_schedule_finish_frac)
        if is_dd_geo_algo and not bool(jnp.all(decay_arr == decay_arr[0])):
            raise ValueError(
                "DD-Geo-Waterfall-ISP-Dyna currently requires one shared decay_factor per hyper-batch algorithm group."
            )
        
        # Helper to construct AlgorithmParams
        def build_params(
            cls_name,
            n_s,
            n_a,
            _gam,
            _lam,
            _h,
            _h_floor,
            _h_terminal,
            _nr_mode,
            _j,
            _lr,
            _del,
            _ps,
            _batch,
            _decay,
            _epsilon,
        ):
            # Creates a JAX AlgorithmParams object (possibly sharded)
            # Logic mirrored from JaxBatchedRLAgent classes
            
            # Common Calculations
            disc = (_h * _gam) / (1 + _h)
            inv_1ph = 1.0 / (1 + _h)
            h_inv_1ph = _h / (1 + _h)
            sched_int = 0 # Assume constant_delay for now (as per wrapper defaults)
            
            # Shape handling for params:
            # If PMAP, these are (D, B_sub). If JIT, (B,).
            # AlgorithmParams expects leaves to be arrays or scalars.
            # Using arrays is safer.
            
            return AlgorithmParams(
                gamma=_gam,
                lambda_val=_lam,
                h=_h,
                J_init=_j,
                surrogate_tol=1e-6,
                surrogate_iters=25,  # PI typically converges in 1-5 iters
                max_J=max_j_scalar,
                n_s=n_s,
                n_a=n_a,
                disc=disc,
                inv_1ph=inv_1ph,
                h_inv_1ph=h_inv_1ph,
                lr_start=_lr,
                lr_delay=_del,
                lr_schedule=sched_int,
                planning_steps=_ps,
                inner_batch_size=max_batch_scalar,
                batch_size=_batch,
                base_h=_h_floor,
                decay_factor=decay_scalar,
                h_init=_h,
                epsilon=_epsilon,
                h_terminal=_h_terminal,
                h_schedule_horizon=int(logical_samples),
                h_schedule_shape=nr_schedule_shape_scalar,
                h_schedule_finish_frac=nr_schedule_finish_frac_scalar,
                nr_mode=nr_mode_scalar,
                model_error_mode=model_error_mode_scalar,
            )
        
        # -------------------------------------------------------------------
        # EXECUTION BRANCH: SHARD_MAP vs PMAP vs SINGLE-DEVICE
        # -------------------------------------------------------------------

        if use_multi_gpu and resolved_evaluation_pipeline == "train_then_offline":
            if use_shard_map:
                logger.warning(
                    "train_then_offline multi-device currently uses the PMAP stream backend; "
                    "falling back from shard_map for this run."
                )
            cls, step_fn, pol_fn = resolve_jax_algorithm_class(algo_name)

            remainder = num_envs_total % n_devices
            padding = 0 if remainder == 0 else (n_devices - remainder)
            padded_size = num_envs_total + padding
            batch_per_device = padded_size // n_devices

            def pad_arr(arr: jnp.ndarray) -> jnp.ndarray:
                return _pad_edge_1d(arr, padding)

            lam_s = pad_arr(lam_arr).reshape(n_devices, batch_per_device)
            gam_s = pad_arr(gamma_arr).reshape(n_devices, batch_per_device)
            lr_s = pad_arr(lr_arr).reshape(n_devices, batch_per_device)
            del_s = pad_arr(delay_arr).reshape(n_devices, batch_per_device)
            h_s = pad_arr(h_arr).reshape(n_devices, batch_per_device)
            h_floor_s = pad_arr(h_floor_arr).reshape(n_devices, batch_per_device)
            h_terminal_s = pad_arr(h_terminal_arr).reshape(n_devices, batch_per_device)
            nr_mode_s = pad_arr(nr_mode_arr).reshape(n_devices, batch_per_device)
            j_s = pad_arr(j_arr).reshape(n_devices, batch_per_device)
            ps_s = pad_arr(ps_arr).reshape(n_devices, batch_per_device)
            batch_s = pad_arr(batch_arr).reshape(n_devices, batch_per_device)
            sched_s = pad_arr(sched_arr).reshape(n_devices, batch_per_device)
            decay_s = pad_arr(decay_arr).reshape(n_devices, batch_per_device)
            epsilon_s = pad_arr(epsilon_arr).reshape(n_devices, batch_per_device)
            seed_s = pad_arr(env_seed_arr).reshape(n_devices, batch_per_device)
            replay_row_s = pad_arr(replay_row_base).reshape(n_devices, batch_per_device)

            is_q_learning = "Q" in algo_name or "q" in algo_name

            @jax.pmap
            def pmap_setup(my_J, my_h, my_lam, my_seed, my_replay_row):
                rng, states = initialize_env_rng_keys_and_states_from_seed_values(my_seed, n_s=n_s)
                float_dtype = mdp_P_jax.dtype
                # R_hat is zero-initialized for all algorithms (no oracle reward prior).
                R_hat = jnp.zeros((batch_per_device, n_s, n_a), dtype=float_dtype)
                z_sa = jnp.zeros((batch_per_device, n_s, n_a), dtype=float_dtype)
                V = jnp.zeros((batch_per_device, n_s), dtype=float_dtype)
                P_hat = initial_model_estimate_pure(
                    num_envs=batch_per_device,
                    n_s=n_s,
                    n_a=n_a,
                    lambda_val=my_lam,
                    model_error_mode=model_error_mode_scalar,
                    dtype=float_dtype,
                )
                N_sas = jnp.zeros((batch_per_device, n_s, n_a, n_s), dtype=float_dtype)
                return AgentState(
                    V=V,
                    P_hat=P_hat,
                    N_sas=N_sas,
                    R_hat=R_hat,
                    N_sa=z_sa,
                    R_sum=z_sa,
                    states=states,
                    J_per_env=my_J,
                    prev_score=jnp.full((batch_per_device,), 1e9, dtype=float_dtype),
                    rng_key=rng,
                    r_k=z_sa,
                    Q=z_sa,
                    bar_r=z_sa,
                    h_s=jnp.zeros((batch_per_device, n_s), dtype=float_dtype),
                    R_E_factor=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    V_checkpoint=V,
                    steps_since_check=jnp.zeros((batch_per_device,), dtype=jnp.int32),
                    working_V=V,
                    policy_cache=jnp.zeros((batch_per_device, n_s), dtype=jnp.int32),
                    waterfall_h=my_h,
                    waterfall_g_best=jnp.full((batch_per_device,), jnp.inf, dtype=float_dtype),
                    waterfall_g_candidate=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    waterfall_reward_abs_max=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    waterfall_inner_depth=jnp.zeros((batch_per_device,), dtype=jnp.int32),
                    waterfall_last_status=jnp.zeros((batch_per_device,), dtype=jnp.int32),
                    waterfall_d_base=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    waterfall_d_refine=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    defect_r=jnp.zeros_like(R_hat),
                    defect_avg=jnp.zeros_like(R_hat),
                    defect_init=jnp.zeros_like(R_hat),
                    defect_clip_fraction=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    defect_gamma_bound=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    replay_states=replay_states_jax,
                    replay_actions=replay_actions_jax,
                    replay_next_states=replay_next_states_jax,
                    replay_row_index=my_replay_row,
                    replay_sample_index=jnp.asarray(0, dtype=jnp.int32),
                )

            @partial(
                jax.pmap,
                in_axes=(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                ),
                static_broadcasted_argnums=(15, 16),
            )
            def pmap_capture_eval_slab(
                state,
                gam,
                lam,
                h,
                h_floor,
                h_terminal,
                nr_mode,
                j,
                lr,
                delay,
                ps,
                batch,
                sched,
                decay,
                epsilon,
                capture_start,
                capture_stop,
            ):
                if is_q_learning:
                    gamma_b, lambda_b, h_b = gam, lam, h
                    lr_b = lr
                else:
                    gamma_b = gam[:, None, None]
                    lambda_b = lam[:, None]
                    h_b = h[:, None, None]
                    lr_b = lr[:, None, None]

                params = AlgorithmParams(
                    gamma=gamma_b,
                    lambda_val=lambda_b,
                    h=h_b,
                    J_init=j,
                    surrogate_tol=1e-6,
                    surrogate_iters=25,
                    max_J=max_j_scalar,
                    n_s=n_s,
                    n_a=n_a,
                    disc=(h_b * gamma_b) / (1 + h_b),
                    inv_1ph=1.0 / (1 + h_b),
                    h_inv_1ph=h_b / (1 + h_b),
                    lr_start=lr_b,
                    lr_delay=delay,
                    lr_schedule=sched,
                    planning_steps=ps,
                    inner_batch_size=max_batch_scalar,
                    batch_size=batch,
                    base_h=h_floor,
                    sampling_distribution=0,
                    decay_factor=decay_scalar,
                    h_init=h,
                    epsilon=epsilon,
                    h_terminal=h_terminal,
                    h_schedule_horizon=int(logical_samples),
                    h_schedule_shape=nr_schedule_shape_scalar,
                    h_schedule_finish_frac=nr_schedule_finish_frac_scalar,
                    nr_mode=nr_mode_scalar,
                    model_error_mode=model_error_mode_scalar,
                )
                mdp_local = MDPData(P=mdp_P_jax, R=mdp_R_jax, gamma=params.gamma)

                def _scan_step(carry_state, k_value):
                    next_state, _ = step_fn(carry_state, mdp_local, params, k_value)
                    next_policy = jnp.asarray(pol_fn(next_state, params), dtype=jnp.int32)
                    return next_state, next_policy

                capture_count = int(capture_stop) - int(capture_start)
                if capture_start == 0:
                    initial_policy = jnp.asarray(pol_fn(state, params), dtype=jnp.int32)
                    if capture_count == 1:
                        final_state = state
                        slab_policies = initial_policy[None, ...]
                    else:
                        step_indices = jnp.arange(0, capture_count - 1, dtype=jnp.int32)
                        final_state, scanned_policies = jax.lax.scan(_scan_step, state, step_indices)
                        slab_policies = jnp.concatenate([initial_policy[None, ...], scanned_policies], axis=0)
                else:
                    step_indices = jnp.arange(capture_start - 1, capture_stop - 1, dtype=jnp.int32)
                    final_state, slab_policies = jax.lax.scan(_scan_step, state, step_indices)

                start_state_values = evaluate_policy_capture_slab_start_state(
                    policy_slab=slab_policies,
                    mdp_P=mdp_P_jax,
                    mdp_R=mdp_R_jax,
                    gamma=gam,
                    chunk_size=plan.solve_chunk_size,
                    start_state_idx=0,
                )
                return final_state, start_state_values

            sample_axis = full_sample_axis
            plan = plan_offline_eval_layout(
                training_steps=logical_samples,
                env_batch_size=num_envs_total,
                n_states=n_s,
                sample_axis=sample_axis,
            )
            resolved_seed_values = (
                tuple(range(n_seeds))
                if seed_values is None
                else tuple(int(value) for value in seed_values)
            )
            config_labels = tuple(str(cfg.label) for cfg in cfgs)
            bundle_root = (
                (offline_root_path or (Path("outputs") / "runtime_tmp" / "offline_eval"))
                / canonical_algorithm_name(algo_name)
            )
            manifest_path = bundle_root / "manifest.json"
            checkpoint_payload = None
            if manifest_path.exists():
                manifest = load_offline_eval_manifest(bundle_root=bundle_root)
                if (
                    manifest.plan.sample_axis != sample_axis
                    or manifest.plan.env_batch_size != num_envs_total
                    or tuple(manifest.config_labels) != config_labels
                    or tuple(manifest.resolved_seed_values) != resolved_seed_values
                ):
                    raise ValueError(
                        "Existing offline bundle is incompatible with the requested packed run."
                    )
                checkpoint_payload = load_stream_checkpoint(bundle_root=bundle_root)
                state_sharded = (
                    checkpoint_payload["state"]
                    if checkpoint_payload is not None
                    else pmap_setup(j_s, h_s, seed_s, replay_row_s)
                )
            else:
                manifest = initialize_policy_capture_bundle(
                    bundle_root=bundle_root,
                    algorithm=algo_name,
                    problem_type="control",
                    lambda_value=float(cfgs[0].lambda_value) if cfgs else None,
                    seed=None,
                    n_seeds=n_seeds,
                    plan=plan,
                    sample_axis=sample_axis,
                    value_storage_mode="start_state_only",
                    config_labels=config_labels,
                    seed_values=resolved_seed_values,
                    sampling_contract=sampling_contract_payload,
                )
                state_sharded = pmap_setup(j_s, h_s, lam_s, seed_s, replay_row_s)

            completed_slabs = set(int(value) for value in manifest.completed_slabs)
            write_offline_eval_phase_manifest(
                bundle_root=bundle_root,
                payload={
                    "phase": "training",
                    "current_sample": int(sample_axis[0]),
                    "current_slab": 0,
                    "completed_slabs": sorted(completed_slabs),
                    "config_count": len(cfgs),
                    "seeds": list(resolved_seed_values),
                    "device_count": n_devices,
                    "backend_mode": "PMAP_STREAM",
                    "sampling_contract": sampling_contract_payload,
                },
            )
            stream_started_at = time.perf_counter()
            for slab in plan.slabs:
                if slab.slab_index in completed_slabs:
                    continue
                state_sharded, slab_values = pmap_capture_eval_slab(
                    state_sharded,
                    gam_s,
                    lam_s,
                    h_s,
                    h_floor_s,
                    h_terminal_s,
                    nr_mode_s,
                    j_s,
                    lr_s,
                    del_s,
                    ps_s,
                    batch_s,
                    sched_s,
                    decay_s,
                    epsilon_s,
                    slab.capture_start,
                    slab.capture_stop,
                )
                host_slab = np.asarray(jax.device_get(slab_values), dtype=np.float64)
                host_slab = np.transpose(host_slab, (1, 0, 2)).reshape(slab.capture_count, -1)[:, :num_envs_total]
                write_start_state_history_slab(
                    bundle_root=bundle_root,
                    manifest=manifest,
                    slab_index=slab.slab_index,
                    start_state_slab=host_slab,
                )
                completed_slabs.add(int(slab.slab_index))
                manifest = save_stream_checkpoint(
                    bundle_root=bundle_root,
                    manifest=manifest,
                    state=state_sharded,
                    current_sample=int(sample_axis[slab.capture_stop - 1]),
                    completed_slabs=sorted(completed_slabs),
                )

            manifest = replace(manifest, completed_slabs=tuple(sorted(completed_slabs)))
            write_offline_eval_manifest(bundle_root=bundle_root, manifest=manifest)
            clear_stream_checkpoint(bundle_root=bundle_root)
            save_json_payload(
                bundle_root / manifest.summary_path,
                {
                    "phase": "evaluated",
                    "mode": "stream_first",
                    "value_storage_mode": manifest.value_storage_mode,
                    "total_seconds": time.perf_counter() - stream_started_at,
                    "completed_slabs": sorted(completed_slabs),
                    "sampling_contract": sampling_contract_payload,
                },
            )
            write_offline_eval_phase_manifest(
                bundle_root=bundle_root,
                payload={
                    "phase": "completed",
                    "current_sample": int(sample_axis[-1]),
                    "current_slab": len(plan.slabs) - 1,
                    "completed_slabs": sorted(completed_slabs),
                    "config_count": len(cfgs),
                    "seeds": list(resolved_seed_values),
                    "device_count": n_devices,
                    "backend_mode": "PMAP_STREAM",
                    "sampling_contract": sampling_contract_payload,
                },
            )
            start_state_history = load_start_state_history(bundle_root=bundle_root, manifest=manifest)
            all_results[algo_name] = _materialize_hyper_batch_trace_from_start_state_history(
                start_state_history=start_state_history,
                sample_axis=sample_axis,
                eval_freq=logical_eval_freq * logical_sample_stride,
                n_seeds=n_seeds,
                cfgs=cfgs,
            )

        elif use_multi_gpu and use_shard_map:
            # =================================================================
            # SHARD_MAP IMPLEMENTATION (Modern, JIT-compatible, fori_loop-friendly)
            # =================================================================
            cls, step_fn, pol_fn = resolve_jax_algorithm_class(algo_name)
            
            # Pad to be divisible by n_devices
            remainder = num_envs_total % n_devices
            padding = 0
            if remainder != 0:
                padding = n_devices - remainder
            
            padded_size = num_envs_total + padding
            batch_per_device = padded_size // n_devices
            
            # Pad arrays (keeping flat shape for shard_map)
            def pad_arr(arr: jnp.ndarray) -> jnp.ndarray:
                return _pad_edge_1d(arr, padding)
            
            # For shard_map, arrays stay FLAT (padded_size,) - no reshaping to (D, B_sub)
            lam_flat = pad_arr(lam_arr)
            gam_flat = pad_arr(gamma_arr)
            lr_flat = pad_arr(lr_arr)
            del_flat = pad_arr(delay_arr)
            h_flat = pad_arr(h_arr)
            h_floor_flat = pad_arr(h_floor_arr)
            h_terminal_flat = pad_arr(h_terminal_arr)
            nr_mode_flat = pad_arr(nr_mode_arr)
            j_flat = pad_arr(j_arr)
            ps_flat = pad_arr(ps_arr)
            batch_flat = pad_arr(batch_arr)
            sched_flat = pad_arr(sched_arr)
            decay_flat = pad_arr(decay_arr)
            epsilon_flat = pad_arr(epsilon_arr)
            seed_flat = pad_arr(env_seed_arr)
            replay_row_flat = pad_arr(replay_row_base)
            
            # Create device mesh for shard_map
            devices = list(jax.devices()[:n_devices])
            mesh = Mesh(devices, axis_names=('d',))
            
            # MDP Data (explicit float64 for JAX 0.4.35 dtype consistency)
            mdp_P_jax = jnp.asarray(mdp.P, dtype=jnp.float64)
            mdp_R_jax = jnp.asarray(mdp.R, dtype=jnp.float64)
            n_s, n_a, _ = mdp.P.shape
            
            # Helper to identify algorithm type
            is_q_learning = "Q" in algo_name or "q" in algo_name
            
            # Build AlgorithmParams with correct shapes INSIDE shard_map
            def build_local_params(
                _gam,
                _lam,
                _h,
                _h_floor,
                _h_terminal,
                _nr_mode,
                _j,
                _lr,
                _del,
                _ps,
                _batch,
                _sched,
                _decay,
                _epsilon,
            ):
                """Build params inside shard_map with correct per-shard shapes."""
                # Inside shard_map, shapes are already (B_sub,)
                if is_q_learning:
                    gamma_b = _gam
                    lr_b = _lr
                    lambda_b = _lam
                    h_b = _h
                else:
                    gamma_b = _gam[:, None, None]
                    lambda_b = _lam[:, None]
                    h_b = _h[:, None, None]
                    lr_b = _lr[:, None, None]

                disc = (h_b * gamma_b) / (1 + h_b)
                inv_1ph = 1.0 / (1 + h_b)
                h_inv_1ph = h_b / (1 + h_b)
                
                return AlgorithmParams(
                    gamma=gamma_b, lambda_val=lambda_b, h=h_b, J_init=_j,
                    surrogate_tol=1e-6, surrogate_iters=25, max_J=max_j_scalar,
                    n_s=n_s, n_a=n_a, disc=disc, inv_1ph=inv_1ph, h_inv_1ph=h_inv_1ph,
                    lr_start=lr_b, lr_delay=_del, lr_schedule=_sched, planning_steps=_ps,
                    inner_batch_size=max_batch_scalar,
                    batch_size=_batch,
                    base_h=_h_floor,
                    sampling_distribution=0,
                    decay_factor=decay_scalar,
                    h_init=_h,
                    epsilon=_epsilon,
                    h_terminal=_h_terminal,
                    h_schedule_horizon=int(logical_samples),
                    h_schedule_shape=nr_schedule_shape_scalar,
                    h_schedule_finish_frac=nr_schedule_finish_frac_scalar,
                    nr_mode=nr_mode_scalar,
                    model_error_mode=model_error_mode_scalar,
                )
            
            # -----------------------------------------------------------------
            # SHARD_MAP'd SETUP FUNCTION
            # -----------------------------------------------------------------

            @partial(jax.shard_map, mesh=mesh,
                     in_specs=(P('d'), P('d'), P('d'), P('d'), P('d')), 
                     out_specs=P('d'))
            def shmap_setup(j_local, h_local, lam_local, seed_local, replay_row_local):
                """Initialize agent state per shard (Dense Backend)."""
                rng, states = initialize_env_rng_keys_and_states_from_seed_values(seed_local, n_s=n_s)
                
                # Common Fields (float64 for JAX 0.4.35 dtype consistency)
                # R_hat is zero-initialized for all algorithms (no oracle reward prior).
                R_hat = jnp.zeros((batch_per_device, n_s, n_a), dtype=jnp.float64)
                z_sa = jnp.zeros((batch_per_device, n_s, n_a), dtype=jnp.float64)
                V = jnp.zeros((batch_per_device, n_s), dtype=jnp.float64)
                
                # Dense Only Initialization (float64)
                P_hat = initial_model_estimate_pure(
                    num_envs=batch_per_device,
                    n_s=n_s,
                    n_a=n_a,
                    lambda_val=lam_local,
                    model_error_mode=model_error_mode_scalar,
                    dtype=jnp.float64,
                )
                N_sas = jnp.zeros((batch_per_device, n_s, n_a, n_s), dtype=jnp.float64)

                return AgentState(
                    V=V, 
                    P_hat=P_hat, 
                    N_sas=N_sas,
                    R_hat=R_hat,
                    N_sa=z_sa, R_sum=z_sa,
                    states=states,
                    J_per_env=j_local,
                    prev_score=jnp.full((batch_per_device,), 1e9, dtype=jnp.float64),
                    rng_key=rng,
                    r_k=z_sa, Q=z_sa, bar_r=z_sa,
                    h_s=jnp.zeros((batch_per_device, n_s), dtype=jnp.float64),
                    R_E_factor=jnp.zeros((batch_per_device,), dtype=jnp.float64),
                    V_checkpoint=V, 
                    steps_since_check=jnp.zeros((batch_per_device,), dtype=jnp.int32),
                    working_V=V,
                    policy_cache=jnp.zeros((batch_per_device, n_s), dtype=jnp.int32),
                    waterfall_h=h_local,
                    waterfall_g_best=jnp.full((batch_per_device,), jnp.inf, dtype=jnp.float64),
                    waterfall_g_candidate=jnp.zeros((batch_per_device,), dtype=jnp.float64),
                    waterfall_reward_abs_max=jnp.zeros((batch_per_device,), dtype=jnp.float64),
                    waterfall_inner_depth=jnp.zeros((batch_per_device,), dtype=jnp.int32),
                    waterfall_last_status=jnp.zeros((batch_per_device,), dtype=jnp.int32),
                    waterfall_d_base=jnp.zeros((batch_per_device,), dtype=jnp.float64),
                    waterfall_d_refine=jnp.zeros((batch_per_device,), dtype=jnp.float64),
                    defect_r=jnp.zeros_like(R_hat),
                    defect_avg=jnp.zeros_like(R_hat),
                    defect_init=jnp.zeros_like(R_hat),
                    defect_clip_fraction=jnp.zeros((batch_per_device,), dtype=jnp.float64),
                    defect_gamma_bound=jnp.zeros((batch_per_device,), dtype=jnp.float64),
                    replay_states=replay_states_jax,
                    replay_actions=replay_actions_jax,
                    replay_next_states=replay_next_states_jax,
                    replay_row_index=replay_row_local,
                    replay_sample_index=jnp.asarray(0, dtype=jnp.int32),
                )
            
            # Initialize state using shard_map (seeds as flat array)
            state_sharded = shmap_setup(j_flat, h_flat, lam_flat, seed_flat, replay_row_flat)
            
            # -----------------------------------------------------------------
            # SHARD_MAP'd TRAINING BLOCK
            # -----------------------------------------------------------------
            # Build shard_map kwargs conditionally (check_vma only in JAX 0.4.36+)
            _shmap_kwargs: dict[str, Any] = dict(
                mesh=mesh,
                in_specs=(
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P(),
                    P(),
                ),
                out_specs=P('d'),
            )
            if _SHARDMAP_SUPPORTS_CHECK_REP:
                _shmap_kwargs['check_rep'] = False
            if _SHARDMAP_SUPPORTS_CHECK_VMA:
                _shmap_kwargs['check_vma'] = False  # Disable VMA check - outputs are safe
            
            @partial(jax.shard_map, **_shmap_kwargs)
            def shmap_run_block(state, gam, lam, h, h_floor, h_terminal, nr_mode, j, lr, delay, ps, batch, sched, decay, epsilon, start_k, n_steps):
                """Run training block per shard."""
                params = build_local_params(
                    gam,
                    lam,
                    h,
                    h_floor,
                    h_terminal,
                    nr_mode,
                    j,
                    lr,
                    delay,
                    ps,
                    batch,
                    sched,
                    decay,
                    epsilon,
                )
                mdp_local = MDPData(P=mdp_P_jax, R=mdp_R_jax, gamma=params.gamma)
                
                def body_fn(i, carry):
                    bst, _ = carry
                    curr_k = start_k + i
                    nst, val = step_fn(bst, mdp_local, params, curr_k)
                    return (nst, val)
                
                # Use a per-batch initial value to match the output shape from step_fn
                # The mean value is computed per batch, so we need shape (B_sub,) or scalar per shard
                init_val = jnp.mean(state.V[:, 0])  # Scalar per shard - naturally varying
                final_state, _ = jax.lax.fori_loop(0, n_steps, body_fn, (state, init_val))
                return final_state
            
            # -----------------------------------------------------------------
            # SHARD_MAP'd EVALUATION
            # -----------------------------------------------------------------
            _eval_shmap_kwargs: dict[str, Any] = dict(
                mesh=mesh,
                in_specs=(
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                    P('d'),
                ),
                out_specs=P('d'),
            )
            if _SHARDMAP_SUPPORTS_CHECK_REP:
                _eval_shmap_kwargs["check_rep"] = False
            if _SHARDMAP_SUPPORTS_CHECK_VMA:
                _eval_shmap_kwargs["check_vma"] = False

            @partial(jax.shard_map, **_eval_shmap_kwargs)
            def shmap_eval_full(state, gam, lam, h, h_floor, h_terminal, nr_mode, j, lr, delay, ps, batch, sched, decay, epsilon):
                """Evaluate policy per shard."""
                params = build_local_params(
                    gam,
                    lam,
                    h,
                    h_floor,
                    h_terminal,
                    nr_mode,
                    j,
                    lr,
                    delay,
                    ps,
                    batch,
                    sched,
                    decay,
                    epsilon,
                )
                pol = pol_fn(state, params)
                gamma_flat_local = gam  # (B_sub,)
                return evaluate_batched_pure(mdp_P_jax, mdp_R_jax, pol, gamma_flat_local)
            
            # -----------------------------------------------------------------
            # GPU-RESIDENT TRACES OPTIMIZATION
            # -----------------------------------------------------------------
            n_evals = (logical_samples // logical_eval_freq) + 1
            
            # Pre-allocate GPU array for traces: (n_evals, padded_size)
            traces_gpu = jnp.zeros((n_evals, padded_size), dtype=mdp_P_jax.dtype)
            x_axis = [0]
            diagnostics_gpu = (
                {
                    name: jnp.zeros((n_evals, padded_size), dtype=mdp_P_jax.dtype)
                    for name in diagnostics_from_state(state_sharded)
                }
                if (
                    is_dd_geo_algo
                    or is_alt_isp_family_algo
                    or is_rm_isp_algo
                    or is_nr_family_algo
                )
                else None
            )
            
            # Initial eval
            vals_flat = shmap_eval_full(
                state_sharded, gam_flat, lam_flat, h_flat, h_floor_flat, h_terminal_flat, nr_mode_flat, j_flat, 
                lr_flat, del_flat, ps_flat, batch_flat, sched_flat, decay_flat, epsilon_flat
            )
            traces_gpu = traces_gpu.at[0].set(vals_flat)
            if diagnostics_gpu is not None:
                initial_diagnostics = diagnostics_from_state(state_sharded)
                for name, values in initial_diagnostics.items():
                    diagnostics_gpu[name] = diagnostics_gpu[name].at[0].set(values)
            
            # JIT-compile block runners for fast Python dispatch
            # Keep n_steps dynamic to avoid recompiling partial final blocks.
            # Static bounds can permit unrolling of small fixed blocks.
            shmap_run_block_jit = jax.jit(shmap_run_block)
            shmap_eval_full_jit = jax.jit(shmap_eval_full)

            if algo_use_fused_loop:
                # FUSED OUTER LOOP using lax.fori_loop
                # The fused outer loop uses shard_map.
                n_iters = logical_samples // logical_eval_freq
                x_axis_gpu = jnp.arange(
                    0,
                    (n_iters + 1) * logical_eval_freq * logical_sample_stride,
                    logical_eval_freq * logical_sample_stride,
                    dtype=jnp.int32,
                )
                
                # JIT-compile the entire fused loop for maximum performance
                @jax.jit
                def run_fused_training(init_state, init_traces):
                    def fused_body_fn(i, carry):
                        state, traces = carry
                        start_step = i * logical_eval_freq
                        
                        # Run training block
                        state = shmap_run_block(
                            state, gam_flat, lam_flat, h_flat, h_floor_flat, h_terminal_flat, nr_mode_flat, j_flat,
                            lr_flat, del_flat, ps_flat, batch_flat, sched_flat, decay_flat, epsilon_flat,
                            start_step, logical_eval_freq
                        )
                        
                        # Evaluate and store
                        vals = shmap_eval_full(
                            state, gam_flat, lam_flat, h_flat, h_floor_flat, h_terminal_flat, nr_mode_flat, j_flat,
                            lr_flat, del_flat, ps_flat, batch_flat, sched_flat, decay_flat, epsilon_flat
                        )
                        traces = traces.at[i + 1].set(vals)
                        
                        return (state, traces)
                    
                    return jax.lax.fori_loop(0, n_iters, fused_body_fn, (init_state, init_traces))
                
                state_sharded, traces_gpu = run_fused_training(state_sharded, traces_gpu)
                x_axis = jax.device_get(x_axis_gpu).tolist()
                eval_idx = n_iters + 1
                diagnostics_payload = None
                if diagnostics_gpu is not None:
                    diagnostics_payload = {
                        name: values[:eval_idx, :num_envs_total]
                        for name, values in diagnostics_gpu.items()
                    }
                    diagnostics_payload["sample_axis"] = jnp.asarray(
                        full_sample_axis if is_nr1_rm_isp_algo else x_axis,
                        dtype=jnp.int32,
                    )
                _maybe_emit_progress(
                    progress_callback=progress_callback,
                    progress_flush_every_evals=progress_flush_every_evals,
                    algo_name=algo_name,
                    cfgs=cfgs,
                    data=traces_gpu[:eval_idx, :num_envs_total],
                    n_seeds=n_seeds,
                    x_axis=x_axis,
                    eval_count=eval_idx,
                    steps_completed=physical_total_samples,
                    total_samples=physical_total_samples,
                    started_at=algo_started_at,
                    device_count=n_devices,
                    backend_mode="SHARD_MAP",
                    diagnostics=diagnostics_payload,
                )
                
            else:
                # PYTHON OUTER LOOP
                # Using JIT-compiled block runners for speed
                steps_done = 0
                eval_idx = 1
                first_block_time = 0.0
                subsequent_time = 0.0
                is_first_block = True
                
                while steps_done < logical_samples:
                    next_eval = min(logical_samples, steps_done + logical_eval_freq)
                    steps_to_run = next_eval - steps_done
                    
                    block_start = time.perf_counter()
                    
                    # Use JIT version
                    state_sharded = shmap_run_block_jit(
                        state_sharded, gam_flat, lam_flat, h_flat, h_floor_flat, h_terminal_flat, nr_mode_flat, j_flat,
                        lr_flat, del_flat, ps_flat, batch_flat, sched_flat, decay_flat, epsilon_flat, steps_done, steps_to_run
                    )
                    
                    # Use JIT version
                    vals_flat = shmap_eval_full_jit(
                        state_sharded, gam_flat, lam_flat, h_flat, h_floor_flat, h_terminal_flat, nr_mode_flat, j_flat,
                        lr_flat, del_flat, ps_flat, batch_flat, sched_flat, decay_flat, epsilon_flat
                    )
                    traces_gpu = traces_gpu.at[eval_idx].set(vals_flat)
                    if diagnostics_gpu is not None:
                        current_diagnostics = diagnostics_from_state(state_sharded)
                        for name, values in current_diagnostics.items():
                            diagnostics_gpu[name] = diagnostics_gpu[name].at[eval_idx].set(values)
                    
                    # Force sync for accurate timing (minimal overhead - just waits for GPU)
                    vals_flat.block_until_ready()
                    
                    block_end = time.perf_counter()
                    block_elapsed = block_end - block_start
                    
                    if is_first_block:
                        first_block_time = block_elapsed
                        is_first_block = False
                    else:
                        subsequent_time += block_elapsed
                    
                    steps_done = next_eval
                    x_axis.append(steps_done * logical_sample_stride)
                    eval_idx += 1
                    diagnostics_payload = None
                    if diagnostics_gpu is not None:
                        diagnostics_payload = {
                            name: values[:eval_idx, :num_envs_total]
                            for name, values in diagnostics_gpu.items()
                        }
                        diagnostics_payload["sample_axis"] = jnp.asarray(
                            full_sample_axis if is_nr1_rm_isp_algo else x_axis,
                            dtype=jnp.int32,
                        )
                    _maybe_emit_progress(
                        progress_callback=progress_callback,
                        progress_flush_every_evals=progress_flush_every_evals,
                        algo_name=algo_name,
                        cfgs=cfgs,
                        data=traces_gpu[:eval_idx, :num_envs_total],
                        n_seeds=n_seeds,
                        x_axis=x_axis,
                        eval_count=eval_idx,
                        steps_completed=steps_done * logical_sample_stride,
                        total_samples=physical_total_samples,
                        started_at=algo_started_at,
                        device_count=n_devices,
                        backend_mode="SHARD_MAP",
                        diagnostics=diagnostics_payload,
                    )
                
                # Log timing summary for this algorithm
                n_subsequent = eval_idx - 2  # Number of blocks after first
                avg_subsequent = subsequent_time / n_subsequent if n_subsequent > 0 else 0.0
                print(f"    {algo_name} timing: first_block={first_block_time:.2f}s (incl. JIT), "
                      f"subsequent={subsequent_time:.2f}s ({n_subsequent} blocks, avg={avg_subsequent:.3f}s/block)", flush=True)
            
            # SINGLE GPU→CPU transfer at the end
            data = traces_gpu[:eval_idx, :num_envs_total]  # Unpad

            all_results[algo_name] = assemble_algorithm_results(
                cfgs=cfgs,
                data=data,
                n_seeds=n_seeds,
                x_axis=x_axis,
                diagnostics=(
                    {
                        name: values[:eval_idx, :num_envs_total]
                        for name, values in diagnostics_gpu.items()
                    }
                    | {
                        "sample_axis": jnp.asarray(
                            full_sample_axis if is_nr1_rm_isp_algo else x_axis,
                            dtype=jnp.int32,
                        )
                    }
                    if diagnostics_gpu is not None
                    else None
                ),
            )

        elif use_multi_gpu and not use_shard_map:
            # =================================================================
            # LEGACY PMAP IMPLEMENTATION (Deprecated, kept for comparison)
            # =================================================================
            logger.warning(
                "Using deprecated PMAP implementation. Consider use_shard_map=True "
                "for better performance with fori_loop fusion."
            )
            cls, step_fn, pol_fn = resolve_jax_algorithm_class(algo_name)
            
            # Pad to be divisible by n_devices
            remainder = num_envs_total % n_devices
            padding = 0
            if remainder != 0:
                padding = n_devices - remainder
            
            padded_size = num_envs_total + padding
            batch_per_device = padded_size // n_devices
            
            # Pad arrays
            def pad_arr(arr: jnp.ndarray) -> jnp.ndarray:
                return _pad_edge_1d(arr, padding)
            
            lam_p = pad_arr(lam_arr)
            gam_p = pad_arr(gamma_arr)
            lr_p = pad_arr(lr_arr)
            del_p = pad_arr(delay_arr)
            h_p = pad_arr(h_arr)
            h_floor_p = pad_arr(h_floor_arr)
            h_terminal_p = pad_arr(h_terminal_arr)
            nr_mode_p = pad_arr(nr_mode_arr)
            j_p = pad_arr(j_arr)
            ps_p = pad_arr(ps_arr)
            decay_p = pad_arr(decay_arr)
            
            # Reshape for PMAP: (D, B_sub, ...)
            def shard(arr):
                return arr.reshape(n_devices, batch_per_device, *arr.shape[1:])
                
            lam_s = shard(lam_p)
            gam_s = shard(gam_p)
            lr_s = shard(lr_p)
            del_s = shard(del_p)
            h_s = shard(h_p)
            h_floor_s = shard(h_floor_p)
            h_terminal_s = shard(h_terminal_p)
            nr_mode_s = shard(nr_mode_p)
            j_s = shard(j_p)
            ps_s = shard(ps_p)
            batch_s = shard(pad_arr(batch_arr))
            sched_s = shard(pad_arr(sched_arr))
            decay_s = shard(decay_p)
            epsilon_s = shard(pad_arr(epsilon_arr))
            seed_s = shard(pad_arr(env_seed_arr))
            replay_row_s = shard(pad_arr(replay_row_base))
            
            is_q_learning = "Q" in algo_name or "q" in algo_name

            @jax.pmap
            def pmap_setup(my_J, my_h, my_lam, my_seed, my_replay_row):
                """Initialize agent state per device (Dense Backend)."""
                rng, states = initialize_env_rng_keys_and_states_from_seed_values(my_seed, n_s=n_s)
                float_dtype = mdp_P_jax.dtype
                
                # R_hat is zero-initialized for all algorithms (no oracle reward prior).
                R_hat = jnp.zeros((batch_per_device, n_s, n_a), dtype=float_dtype)
                z_sa = jnp.zeros((batch_per_device, n_s, n_a), dtype=float_dtype)
                V = jnp.zeros((batch_per_device, n_s), dtype=float_dtype)

                # Dense Only Initialization
                P_hat = initial_model_estimate_pure(
                    num_envs=batch_per_device,
                    n_s=n_s,
                    n_a=n_a,
                    lambda_val=my_lam,
                    model_error_mode=model_error_mode_scalar,
                    dtype=float_dtype,
                )
                N_sas = jnp.zeros((batch_per_device, n_s, n_a, n_s), dtype=float_dtype)

                return AgentState(
                    V=V, 
                    P_hat=P_hat, 
                    N_sas=N_sas,
                    R_hat=R_hat,
                    N_sa=z_sa, R_sum=z_sa,
                    states=states, J_per_env=my_J,
                    prev_score=jnp.full((batch_per_device,), 1e9, dtype=float_dtype),
                    rng_key=rng, r_k=z_sa, Q=z_sa, bar_r=z_sa,
                    h_s=jnp.zeros((batch_per_device, n_s), dtype=float_dtype),
                    R_E_factor=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    V_checkpoint=V,
                    steps_since_check=jnp.zeros((batch_per_device,), dtype=jnp.int32),
                    working_V=V,
                    policy_cache=jnp.zeros((batch_per_device, n_s), dtype=jnp.int32),
                    waterfall_h=my_h,
                    waterfall_g_best=jnp.full((batch_per_device,), jnp.inf, dtype=float_dtype),
                    waterfall_g_candidate=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    waterfall_reward_abs_max=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    waterfall_inner_depth=jnp.zeros((batch_per_device,), dtype=jnp.int32),
                    waterfall_last_status=jnp.zeros((batch_per_device,), dtype=jnp.int32),
                    waterfall_d_base=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    waterfall_d_refine=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    defect_r=jnp.zeros_like(R_hat),
                    defect_avg=jnp.zeros_like(R_hat),
                    defect_init=jnp.zeros_like(R_hat),
                    defect_clip_fraction=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    defect_gamma_bound=jnp.zeros((batch_per_device,), dtype=float_dtype),
                    replay_states=replay_states_jax,
                    replay_actions=replay_actions_jax,
                    replay_next_states=replay_next_states_jax,
                    replay_row_index=my_replay_row,
                    replay_sample_index=jnp.asarray(0, dtype=jnp.int32),
                )
            
            state_sharded = pmap_setup(j_s, h_s, lam_s, seed_s, replay_row_s)
            
            def run_block_pure(state, gam, lam, h, h_floor, h_terminal, nr_mode, j, lr, delay, ps, batch, sched, decay, epsilon, start_k, n_steps):
                if is_q_learning:
                    gamma_b, lr_b, lambda_b, h_b = gam, lr, lam, h
                else:
                    gamma_b = gam[:, None, None]
                    lambda_b = lam[:, None]
                    h_b = h[:, None, None]
                    lr_b = lr[:, None, None]

                disc = (h_b * gamma_b) / (1 + h_b)
                inv_1ph = 1.0 / (1 + h_b)
                h_inv_1ph = h_b / (1 + h_b)
                
                params = AlgorithmParams(
                    gamma=gamma_b, lambda_val=lambda_b, h=h_b, J_init=j,
                    surrogate_tol=1e-6, surrogate_iters=25, max_J=max_j_scalar,
                    n_s=n_s, n_a=n_a, disc=disc, inv_1ph=inv_1ph, h_inv_1ph=h_inv_1ph,
                    lr_start=lr_b, lr_delay=delay, lr_schedule=sched, planning_steps=ps,
                    inner_batch_size=max_batch_scalar,
                    batch_size=batch,
                    base_h=h_floor,
                    sampling_distribution=0,
                    decay_factor=decay_scalar,
                    h_init=h,
                    epsilon=epsilon,
                    h_terminal=h_terminal,
                    h_schedule_horizon=int(logical_samples),
                    h_schedule_shape=nr_schedule_shape_scalar,
                    h_schedule_finish_frac=nr_schedule_finish_frac_scalar,
                    nr_mode=nr_mode_scalar,
                    model_error_mode=model_error_mode_scalar,
                )
                
                mdp_local = MDPData(P=mdp_P_jax, R=mdp_R_jax, gamma=params.gamma)
                
                def body_fn(i, carry):
                    bst, _ = carry
                    curr_k = start_k + i
                    nst, val = step_fn(bst, mdp_local, params, curr_k)
                    return (nst, val)
                
                init_metric = jnp.zeros((), dtype=state.V.dtype)
                final_state, _ = jax.lax.fori_loop(0, n_steps, body_fn, (state, init_metric))
                return final_state

            pmap_run_block = jax.pmap(
                run_block_pure,
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
            )

            @jax.pmap
            def pmap_eval_full(state, gam, lam, h, h_floor, h_terminal, nr_mode, j, lr, delay, ps, batch, sched, decay, epsilon):
                if is_q_learning:
                    gamma_b, lambda_b, h_b = gam, lam, h
                else:
                    gamma_b = gam[:, None, None]
                    lambda_b = lam[:, None]
                    h_b = h[:, None, None]
                
                disc = (h_b * gamma_b) / (1 + h_b)
                inv_1ph = 1.0 / (1 + h_b)
                h_inv_1ph = h_b / (1 + h_b)
                
                params = AlgorithmParams(
                    gamma=gamma_b, lambda_val=lambda_b, h=h_b, J_init=j,
                    surrogate_tol=1e-6, surrogate_iters=25, max_J=max_j_scalar,
                    n_s=n_s, n_a=n_a, disc=disc, inv_1ph=inv_1ph, h_inv_1ph=h_inv_1ph,
                    lr_start=lr, lr_delay=delay, lr_schedule=sched, planning_steps=ps,
                    inner_batch_size=max_batch_scalar,
                    batch_size=batch,
                    base_h=h_floor,
                    sampling_distribution=0,
                    decay_factor=decay_scalar,
                    h_init=h,
                    epsilon=epsilon,
                    h_terminal=h_terminal,
                    h_schedule_horizon=int(logical_samples),
                    h_schedule_shape=nr_schedule_shape_scalar,
                    h_schedule_finish_frac=nr_schedule_finish_frac_scalar,
                    nr_mode=nr_mode_scalar,
                    model_error_mode=model_error_mode_scalar,
                )
                
                pol = pol_fn(state, params)
                gamma_flat = gam
                return evaluate_batched_pure(mdp_P_jax, mdp_R_jax, pol, gamma_flat)

            n_evals = (logical_samples // logical_eval_freq) + 1
            traces_gpu = jnp.zeros((n_evals, n_devices, batch_per_device), dtype=mdp_P_jax.dtype)
            x_axis = [0]
            diagnostics_gpu = (
                {
                    name: jnp.zeros((n_evals, n_devices, batch_per_device), dtype=mdp_P_jax.dtype)
                    for name in diagnostics_from_state(state_sharded)
                }
                    if (
                        is_dd_geo_algo
                        or is_alt_isp_family_algo
                        or is_rm_isp_algo
                        or is_nr_family_algo
                    )
                else None
            )
            
            vals_s = pmap_eval_full(
                state_sharded,
                gam_s,
                lam_s,
                h_s,
                h_floor_s,
                h_terminal_s,
                nr_mode_s,
                j_s,
                lr_s,
                del_s,
                ps_s,
                batch_s,
                sched_s,
                decay_s,
                epsilon_s,
            )
            traces_gpu = traces_gpu.at[0].set(vals_s)
            if diagnostics_gpu is not None:
                initial_diagnostics = diagnostics_from_state(state_sharded)
                for name, values in initial_diagnostics.items():
                    diagnostics_gpu[name] = diagnostics_gpu[name].at[0].set(values)
            
            if algo_use_fused_loop:
                logger.warning(
                    "use_fused_loop=True with PMAP may be slower due to jit-of-pmap inefficiency. "
                    "Consider using use_shard_map=True for better performance."
                )
                n_iters = logical_samples // logical_eval_freq
                x_axis_gpu = jnp.arange(
                    0,
                    (n_iters + 1) * logical_eval_freq * logical_sample_stride,
                    logical_eval_freq * logical_sample_stride,
                    dtype=jnp.int32,
                )
                
                gam_s_jax = jnp.asarray(gam_s)
                lam_s_jax = jnp.asarray(lam_s)
                h_s_jax = jnp.asarray(h_s)
                h_floor_s_jax = jnp.asarray(h_floor_s)
                h_terminal_s_jax = jnp.asarray(h_terminal_s)
                nr_mode_s_jax = jnp.asarray(nr_mode_s)
                j_s_jax = jnp.asarray(j_s)
                lr_s_jax = jnp.asarray(lr_s)
                del_s_jax = jnp.asarray(del_s)
                ps_s_jax = jnp.asarray(ps_s)
                batch_s_jax = jnp.asarray(batch_s)
                sched_s_jax = jnp.asarray(sched_s)
                decay_s_jax = jnp.asarray(decay_s)
                epsilon_s_jax = jnp.asarray(epsilon_s)
                
                def fused_body_fn(i, carry):
                    state, traces = carry
                    start_step = i * logical_eval_freq
                    
                    state = pmap_run_block(
                        state, gam_s_jax, lam_s_jax, h_s_jax, h_floor_s_jax, h_terminal_s_jax, nr_mode_s_jax, j_s_jax, 
                        lr_s_jax, del_s_jax, ps_s_jax, batch_s_jax, sched_s_jax, decay_s_jax, epsilon_s_jax, start_step, logical_eval_freq
                    )
                    
                    vals = pmap_eval_full(
                        state, gam_s_jax, lam_s_jax, h_s_jax, h_floor_s_jax, h_terminal_s_jax, nr_mode_s_jax, j_s_jax,
                        lr_s_jax, del_s_jax, ps_s_jax, batch_s_jax, sched_s_jax, decay_s_jax, epsilon_s_jax
                    )
                    traces = traces.at[i + 1].set(vals)
                    
                    return (state, traces)
                
                init_carry = (state_sharded, traces_gpu)
                state_sharded, traces_gpu = jax.lax.fori_loop(0, n_iters, fused_body_fn, init_carry)
                
                x_axis = jax.device_get(x_axis_gpu).tolist()
                eval_idx = n_iters + 1
                _maybe_emit_progress(
                    progress_callback=progress_callback,
                    progress_flush_every_evals=progress_flush_every_evals,
                    algo_name=algo_name,
                    cfgs=cfgs,
                    data=traces_gpu[:eval_idx].reshape(eval_idx, -1)[:, :num_envs_total],
                    n_seeds=n_seeds,
                    x_axis=x_axis,
                    eval_count=eval_idx,
                    steps_completed=physical_total_samples,
                    total_samples=physical_total_samples,
                    started_at=algo_started_at,
                    device_count=n_devices,
                    backend_mode="PMAP",
                )
                
            else:
                steps_done = 0
                eval_idx = 1
                while steps_done < logical_samples:
                    next_eval = min(logical_samples, steps_done + logical_eval_freq)
                    steps_to_run = next_eval - steps_done
                    
                    state_sharded = pmap_run_block(
                        state_sharded, gam_s, lam_s, h_s, h_floor_s, h_terminal_s, nr_mode_s, j_s, lr_s, del_s, ps_s, batch_s, sched_s, decay_s, epsilon_s, steps_done, steps_to_run
                    )
                    
                    vals_s = pmap_eval_full(
                        state_sharded,
                        gam_s,
                        lam_s,
                        h_s,
                        h_floor_s,
                        h_terminal_s,
                        nr_mode_s,
                        j_s,
                        lr_s,
                        del_s,
                        ps_s,
                        batch_s,
                        sched_s,
                        decay_s,
                        epsilon_s,
                    )
                    traces_gpu = traces_gpu.at[eval_idx].set(vals_s)
                    if diagnostics_gpu is not None:
                        current_diagnostics = diagnostics_from_state(state_sharded)
                        for name, values in current_diagnostics.items():
                            diagnostics_gpu[name] = diagnostics_gpu[name].at[eval_idx].set(values)
                    
                    steps_done = next_eval
                    x_axis.append(steps_done * logical_sample_stride)
                    eval_idx += 1
                    diagnostics_payload = None
                    if diagnostics_gpu is not None:
                        diagnostics_payload = {
                            name: values[:eval_idx].reshape(eval_idx, -1)[:, :num_envs_total]
                            for name, values in diagnostics_gpu.items()
                        }
                        diagnostics_payload["sample_axis"] = jnp.asarray(
                            full_sample_axis if is_nr1_rm_isp_algo else x_axis,
                            dtype=jnp.int32,
                        )
                    _maybe_emit_progress(
                        progress_callback=progress_callback,
                        progress_flush_every_evals=progress_flush_every_evals,
                        algo_name=algo_name,
                        cfgs=cfgs,
                        data=traces_gpu[:eval_idx].reshape(eval_idx, -1)[:, :num_envs_total],
                        n_seeds=n_seeds,
                        x_axis=x_axis,
                        eval_count=eval_idx,
                        steps_completed=steps_done * logical_sample_stride,
                        total_samples=physical_total_samples,
                        started_at=algo_started_at,
                        device_count=n_devices,
                        backend_mode="PMAP",
                        diagnostics=diagnostics_payload,
                    )
            
            data = traces_gpu[:eval_idx].reshape(eval_idx, -1)[:, :num_envs_total]
            
            all_results[algo_name] = assemble_algorithm_results(
                cfgs=cfgs,
                data=data,
                n_seeds=n_seeds,
                x_axis=x_axis,
                diagnostics=(
                    {
                        name: values[:eval_idx].reshape(eval_idx, -1)[:, :num_envs_total]
                        for name, values in diagnostics_gpu.items()
                    }
                    | {
                        "sample_axis": jnp.asarray(
                            full_sample_axis if is_nr1_rm_isp_algo else x_axis,
                            dtype=jnp.int32,
                        )
                    }
                    if diagnostics_gpu is not None
                    else None
                ),
            )

        else:
            # SINGLE DEVICE JAX (Legacy/Fallback)
            cls, step_fn, policy_fn = resolve_jax_algorithm_class(algo_name)
            
            # Resolve algorithm-specific arguments through the class wrapper.
            
            # Prepare args dict
            algo_kwargs = {}
            is_q = "QLearning" in cls.__name__
            is_dyna = "Dyna" in cls.__name__ and "ISP" not in cls.__name__ and "OS" not in cls.__name__
            is_os = "OSDyna" in cls.__name__
            is_dd_geo = "DDGeoWaterfall" in cls.__name__
            is_gen_alt_isp = "GenAltISP" in cls.__name__
            is_alt_isp = "AltISP" in cls.__name__ and not is_gen_alt_isp
            is_arch_waterfall = "ArchWaterfallISP" in cls.__name__
            is_scheduled_waterfall = "ScheduledWaterfallISP" in cls.__name__
            is_arch_isp = "ArchISP" in cls.__name__ and not is_arch_waterfall
            is_nr_rm_isp = is_nr_family_algo
            is_rm_isp = "RMISP" in cls.__name__ and not is_nr_family_algo
            is_isp = (
                "ISP" in cls.__name__
                and not is_dd_geo
                and not is_alt_isp
                and not is_scheduled_waterfall
                and not is_arch_isp
                and not is_arch_waterfall
                and not is_rm_isp
                and not is_nr_rm_isp
            )
            schedule_name = normalize_lr_type_name(cfgs[0].lr_type)
            
            if is_q:
                algo_kwargs["lr_params"] = (lr_arr, delay_arr)
                algo_kwargs["lr_schedule"] = schedule_name
            elif is_os:
                algo_kwargs["lambda_val"] = lam_arr
                algo_kwargs["lr_params"] = (lr_arr, delay_arr)
                algo_kwargs["lr_schedule"] = schedule_name
            elif is_dyna:
                algo_kwargs["lambda_val"] = lam_arr
                algo_kwargs["planning_steps"] = (
                    int(ps_arr[0]) if bool(jnp.all(ps_arr == ps_arr[0])) else ps_arr
                )
                algo_kwargs["lr_params"] = (lr_arr, delay_arr)
                algo_kwargs["lr_schedule"] = schedule_name
            elif is_arch_isp:
                algo_kwargs["lambda_val"] = lam_arr
                algo_kwargs["h"] = h_arr
                algo_kwargs["J"] = j_arr
                algo_kwargs["lr_params"] = (lr_arr, delay_arr)
                algo_kwargs["lr_schedule"] = schedule_name
                algo_kwargs["sampling_distribution"] = cfgs[0].sampling_distribution
            elif is_rm_isp:
                algo_kwargs["lambda_val"] = lam_arr
                algo_kwargs["h"] = h_arr
                algo_kwargs["J"] = j_arr
                algo_kwargs["lr_params"] = (lr_arr, delay_arr)
                algo_kwargs["lr_schedule"] = schedule_name
                algo_kwargs["sampling_distribution"] = cfgs[0].sampling_distribution
            elif is_nr_family_algo:
                algo_kwargs["lambda_val"] = lam_arr
                algo_kwargs["lr_params"] = (lr_arr, delay_arr)
                algo_kwargs["lr_schedule"] = schedule_name
                algo_kwargs["sampling_distribution"] = cfgs[0].sampling_distribution
                if is_nr1_staircase_waterfall_algo:
                    algo_kwargs["J"] = j_arr
                    algo_kwargs["h_init"] = h_arr
                    algo_kwargs["h_min"] = h_floor_arr
                    algo_kwargs["stall_patience"] = waterfall_stall_patience_shared
                    algo_kwargs["decay_factor"] = waterfall_decay_factor_shared
                    algo_kwargs["divergence_mult"] = waterfall_divergence_mult_shared
                    algo_kwargs["v_max"] = waterfall_v_max_arr
                elif is_nr1_geo_waterfall_algo:
                    algo_kwargs["J"] = j_arr
                    algo_kwargs["h_init"] = h_arr
                    algo_kwargs["h_min"] = h_floor_arr
                    algo_kwargs["alpha"] = waterfall_alpha_shared
                elif bool(jnp.any(nr_mode_arr == 2)):
                    algo_kwargs["J"] = j_arr
                    algo_kwargs["h_init"] = h_arr
                    algo_kwargs["h_terminal"] = h_terminal_arr
                    algo_kwargs["h_floor"] = h_floor_arr
                    algo_kwargs["h_schedule_horizon"] = logical_samples
                    algo_kwargs["h_schedule_shape"] = cfgs[0].h_schedule_shape
                    algo_kwargs["h_schedule_finish_frac"] = cfgs[0].h_schedule_finish_frac
                elif bool(jnp.any(nr_mode_arr == 1)):
                    algo_kwargs["J"] = j_arr
                    algo_kwargs["h"] = h_arr
            elif is_alt_isp:
                algo_kwargs["lambda_val"] = lam_arr
                algo_kwargs["h_target"] = h_arr
                algo_kwargs["J"] = j_arr
                algo_kwargs["epsilon"] = epsilon_arr
            elif is_gen_alt_isp:
                algo_kwargs["lambda_val"] = lam_arr
                algo_kwargs["h_target"] = h_arr
                algo_kwargs["J"] = j_arr
            elif is_isp:
                algo_kwargs["lambda_val"] = lam_arr
                algo_kwargs["h"] = h_arr
                algo_kwargs["J"] = j_arr
                algo_kwargs["batch_size"] = batch_arr
                algo_kwargs["lr_params"] = (lr_arr, delay_arr)
                algo_kwargs["lr_schedule"] = schedule_name
                algo_kwargs["sampling_distribution"] = cfgs[0].sampling_distribution
            elif is_scheduled_waterfall:
                algo_kwargs["lambda_val"] = lam_arr
                algo_kwargs["h_max"] = h_arr
                algo_kwargs["h_min"] = h_floor_arr
                algo_kwargs["J"] = j_arr
                algo_kwargs["lr_params"] = (lr_arr, delay_arr)
                algo_kwargs["lr_schedule"] = schedule_name
                algo_kwargs["schedule_decay_rate"] = float(decay_arr[0])
                algo_kwargs["sampling_distribution"] = cfgs[0].sampling_distribution
            elif is_arch_waterfall:
                algo_kwargs["lambda_val"] = lam_arr
                algo_kwargs["h_max"] = h_arr
                algo_kwargs["h_min"] = h_floor_arr
                algo_kwargs["J"] = j_arr
                algo_kwargs["lr_params"] = (lr_arr, delay_arr)
                algo_kwargs["lr_schedule"] = schedule_name
                algo_kwargs["schedule_decay_rate"] = float(decay_arr[0])
                algo_kwargs["sampling_distribution"] = cfgs[0].sampling_distribution
            elif is_dd_geo:
                algo_kwargs["lambda_val"] = lam_arr
                algo_kwargs["h_init"] = h_arr
                algo_kwargs["J_max"] = j_arr
                algo_kwargs["batch_size"] = batch_arr
                algo_kwargs["lr_params"] = (lr_arr, delay_arr)
                algo_kwargs["lr_schedule"] = schedule_name
                algo_kwargs["decay_factor"] = float(decay_arr[0])
                algo_kwargs["sampling_distribution"] = cfgs[0].sampling_distribution
            
            agent = cls(
                mdp_P_jax, mdp_R_jax, gamma_arr,
                num_envs=num_envs_total, seed=env_seed_arr, **algo_kwargs
            )
            if not is_q:
                agent.params = agent.params._replace(model_error_mode=model_error_mode_scalar)
                agent.state = agent.state._replace(
                    P_hat=initial_model_estimate_pure(
                        num_envs=num_envs_total,
                        n_s=n_s,
                        n_a=n_a,
                        lambda_val=lam_arr,
                        model_error_mode=model_error_mode_scalar,
                        dtype=mdp_P_jax.dtype,
                    )
                )
            if resolved_sampling_contract.mode == "replay_tape":
                agent.state = agent.state._replace(
                    replay_states=replay_states_jax,
                    replay_actions=replay_actions_jax,
                    replay_next_states=replay_next_states_jax,
                    replay_row_index=replay_row_index_arr,
                    replay_sample_index=jnp.asarray(0, dtype=jnp.int32),
                )
            gamma_jax = jnp.asarray(gamma_arr, dtype=jnp.float64)

            if resolved_evaluation_pipeline == "train_then_offline":
                sample_axis = full_sample_axis
                plan = plan_offline_eval_layout(
                    training_steps=logical_samples,
                    env_batch_size=num_envs_total,
                    n_states=n_s,
                    sample_axis=sample_axis,
                )
                resolved_seed_values = (
                    resolved_seed_values_global
                )
                config_labels = tuple(str(cfg.label) for cfg in cfgs)
                bundle_root = (
                    (offline_root_path or (Path("outputs") / "runtime_tmp" / "offline_eval"))
                    / canonical_algorithm_name(algo_name)
                )
                manifest_path = bundle_root / "manifest.json"
                checkpoint_payload = None
                if manifest_path.exists():
                    manifest = load_offline_eval_manifest(bundle_root=bundle_root)
                    if (
                        manifest.plan.sample_axis != sample_axis
                        or manifest.plan.env_batch_size != num_envs_total
                        or tuple(manifest.config_labels) != config_labels
                        or tuple(manifest.resolved_seed_values) != resolved_seed_values
                    ):
                        raise ValueError(
                            "Existing offline bundle is incompatible with the requested packed run."
                        )
                    checkpoint_payload = load_stream_checkpoint(bundle_root=bundle_root)
                    current_state = (
                        jax.device_put(checkpoint_payload["state"])
                        if checkpoint_payload is not None
                        else agent.state
                    )
                else:
                    manifest = initialize_policy_capture_bundle(
                        bundle_root=bundle_root,
                        algorithm=algo_name,
                        problem_type="control",
                        lambda_value=float(cfgs[0].lambda_value) if cfgs else None,
                        seed=None,
                        n_seeds=n_seeds,
                        plan=plan,
                        sample_axis=sample_axis,
                        value_storage_mode="start_state_only",
                        config_labels=config_labels,
                        seed_values=resolved_seed_values,
                        sampling_contract=sampling_contract_payload,
                    )
                    current_state = agent.state
                completed_slabs = set(int(value) for value in manifest.completed_slabs)
                write_offline_eval_phase_manifest(
                    bundle_root=bundle_root,
                    payload={
                        "phase": "training",
                    "current_sample": int(sample_axis[0]),
                        "current_slab": 0,
                        "completed_slabs": sorted(completed_slabs),
                        "config_count": len(cfgs),
                        "seeds": list(resolved_seed_values),
                        "device_count": 1,
                        "backend_mode": "SINGLE_DEVICE_STREAM",
                        "sampling_contract": sampling_contract_payload,
                    },
                )
                stream_started_at = time.perf_counter()
                for slab in plan.slabs:
                    if slab.slab_index in completed_slabs:
                        continue
                    write_offline_eval_phase_manifest(
                        bundle_root=bundle_root,
                        payload={
                            "phase": "training",
                            "current_sample": int(sample_axis[max(0, slab.capture_start)]),
                            "current_slab": int(slab.slab_index),
                            "completed_slabs": sorted(completed_slabs),
                            "config_count": len(cfgs),
                            "seeds": list(resolved_seed_values),
                            "device_count": 1,
                            "backend_mode": "SINGLE_DEVICE_STREAM",
                            "sampling_contract": sampling_contract_payload,
                        },
                    )
                    current_state, start_state_slab = _capture_start_state_slab_single_device(
                        state=current_state,
                        mdp_data=agent.mdp_data,
                        params=agent.params,
                        step_fn=step_fn,
                        policy_fn=policy_fn,
                        capture_start=slab.capture_start,
                        capture_stop=slab.capture_stop,
                        mdp_P=mdp_P_jax,
                        mdp_R=mdp_R_jax,
                        gamma=float(gamma_arr[0]),
                        solve_chunk_size=manifest.plan.solve_chunk_size,
                    )
                    write_offline_eval_phase_manifest(
                        bundle_root=bundle_root,
                        payload={
                            "phase": "offline_eval",
                            "current_sample": int(sample_axis[slab.capture_stop - 1]),
                            "current_slab": int(slab.slab_index),
                            "completed_slabs": sorted(completed_slabs),
                            "config_count": len(cfgs),
                            "seeds": list(resolved_seed_values),
                            "device_count": 1,
                            "backend_mode": "SINGLE_DEVICE_STREAM",
                            "sampling_contract": sampling_contract_payload,
                        },
                    )
                    write_start_state_history_slab(
                        bundle_root=bundle_root,
                        manifest=manifest,
                        slab_index=slab.slab_index,
                        start_state_slab=np.asarray(jax.device_get(start_state_slab), dtype=np.float64),
                    )
                    completed_slabs.add(int(slab.slab_index))
                    manifest = save_stream_checkpoint(
                        bundle_root=bundle_root,
                        manifest=manifest,
                        state=current_state,
                        current_sample=int(sample_axis[slab.capture_stop - 1]),
                        completed_slabs=sorted(completed_slabs),
                    )
                agent.state = current_state
                manifest = replace(
                    manifest,
                    completed_slabs=tuple(sorted(completed_slabs)),
                )
                write_offline_eval_manifest(bundle_root=bundle_root, manifest=manifest)
                clear_stream_checkpoint(bundle_root=bundle_root)
                save_json_payload(
                    bundle_root / manifest.summary_path,
                    {
                        "phase": "evaluated",
                        "mode": "stream_first",
                        "value_storage_mode": manifest.value_storage_mode,
                        "total_seconds": time.perf_counter() - stream_started_at,
                        "completed_slabs": sorted(completed_slabs),
                        "sampling_contract": sampling_contract_payload,
                    },
                )
                write_offline_eval_phase_manifest(
                    bundle_root=bundle_root,
                    payload={
                        "phase": "summarizing",
                        "current_sample": int(sample_axis[-1]),
                        "current_slab": len(plan.slabs) - 1,
                        "completed_slabs": sorted(completed_slabs),
                        "config_count": len(cfgs),
                        "seeds": list(resolved_seed_values),
                        "device_count": 1,
                        "backend_mode": "SINGLE_DEVICE_STREAM",
                        "sampling_contract": sampling_contract_payload,
                    },
                )
                start_state_history = load_start_state_history(
                    bundle_root=bundle_root,
                    manifest=manifest,
                )
                all_results[algo_name] = _materialize_hyper_batch_trace_from_start_state_history(
                    start_state_history=start_state_history,
                    sample_axis=sample_axis,
                    eval_freq=logical_eval_freq * logical_sample_stride,
                    n_seeds=n_seeds,
                    cfgs=cfgs,
                )
                write_offline_eval_phase_manifest(
                    bundle_root=bundle_root,
                    payload={
                        "phase": "completed",
                        "current_sample": int(sample_axis[-1]),
                        "current_slab": len(plan.slabs) - 1,
                        "completed_slabs": sorted(completed_slabs),
                        "config_count": len(cfgs),
                        "seeds": list(resolved_seed_values),
                        "device_count": 1,
                        "backend_mode": "SINGLE_DEVICE_STREAM",
                        "sampling_contract": sampling_contract_payload,
                    },
                )
            else:
                # GPU-RESIDENT TRACES OPTIMIZATION
                # Keep all eval results on GPU until the end to avoid GPU→CPU sync overhead
                n_evals = (logical_samples // logical_eval_freq) + 1  # +1 for initial eval
                traces_gpu = jnp.zeros((n_evals, num_envs_total), dtype=mdp_P_jax.dtype)
                x_axis = [0]
                diagnostics_gpu = (
                    {
                        name: jnp.zeros((n_evals, num_envs_total), dtype=mdp_P_jax.dtype)
                        for name in diagnostics_from_state(agent.state)
                    }
                    if (
                        is_dd_geo_algo
                        or is_alt_isp_family_algo
                        or is_rm_isp_algo
                        or is_nr_family_algo
                    )
                    else None
                )
                
                # Eval 0
                pol = agent.get_policy()
                v_jax = evaluate_batched_pure(mdp_P_jax, mdp_R_jax, pol, gamma_jax)
                traces_gpu = traces_gpu.at[0].set(v_jax)
                if diagnostics_gpu is not None:
                    initial_diagnostics = diagnostics_from_state(agent.state)
                    for name, values in initial_diagnostics.items():
                        diagnostics_gpu[name] = diagnostics_gpu[name].at[0].set(values)
                
                steps_done = 0
                eval_idx = 1
                while steps_done < logical_samples:
                    next_eval = min(logical_samples, steps_done + logical_eval_freq)
                    steps_to_run = next_eval - steps_done
                    
                    agent.run_steps(steps_done, steps_to_run)
                    
                    pol = agent.get_policy()
                    v_jax = evaluate_batched_pure(mdp_P_jax, mdp_R_jax, pol, gamma_jax)
                    traces_gpu = traces_gpu.at[eval_idx].set(v_jax)
                    if diagnostics_gpu is not None:
                        current_diagnostics = diagnostics_from_state(agent.state)
                        for name, values in current_diagnostics.items():
                            diagnostics_gpu[name] = diagnostics_gpu[name].at[eval_idx].set(values)
                    
                    steps_done = next_eval
                    x_axis.append(steps_done * logical_sample_stride)
                    eval_idx += 1
                    diagnostics_payload = None
                    if diagnostics_gpu is not None:
                        diagnostics_payload = {
                            name: values[:eval_idx]
                            for name, values in diagnostics_gpu.items()
                        }
                        diagnostics_payload["sample_axis"] = jnp.asarray(
                            full_sample_axis if is_nr1_rm_isp_algo else x_axis,
                            dtype=jnp.int32,
                        )
                    _maybe_emit_progress(
                        progress_callback=progress_callback,
                        progress_flush_every_evals=progress_flush_every_evals,
                        algo_name=algo_name,
                        cfgs=cfgs,
                        data=traces_gpu[:eval_idx],
                        n_seeds=n_seeds,
                        x_axis=x_axis,
                        eval_count=eval_idx,
                        steps_completed=steps_done * logical_sample_stride,
                        total_samples=physical_total_samples,
                        started_at=algo_started_at,
                        device_count=n_devices,
                        backend_mode="SINGLE_DEVICE",
                        diagnostics=diagnostics_payload,
                    )
                
                # SINGLE GPU→CPU transfer at the end
                data = traces_gpu[:eval_idx]  # (T, num_envs)
                all_results[algo_name] = assemble_algorithm_results(
                    cfgs=cfgs,
                    data=data,
                    n_seeds=n_seeds,
                    x_axis=x_axis,
                    diagnostics=(
                        {name: values[:eval_idx] for name, values in diagnostics_gpu.items()}
                        | {
                            "sample_axis": jnp.asarray(
                                full_sample_axis if is_nr1_rm_isp_algo else x_axis,
                                dtype=jnp.int32,
                            )
                        }
                        if diagnostics_gpu is not None
                        else None
                    ),
                )
            del agent

    return all_results
