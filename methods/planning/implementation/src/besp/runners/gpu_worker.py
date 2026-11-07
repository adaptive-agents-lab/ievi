"""Single-device JAX worker helpers for deterministic and stochastic experiments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Mapping

import jax
import jax.numpy as jnp

from ..core.config import (
    _validate_alt_isp_dyna_params,
    _validate_gen_alt_isp_dyna_params,
    _validate_arch_isp_dyna_params,
    _validate_nr_rm_isp_dyna_params,
    _validate_nr1_rm_isp_dyna_params,
    _validate_arch_waterfall_isp_dyna_params,
    _validate_dd_geo_waterfall_isp_dyna_params,
    _validate_elastic_dd_geo_waterfall_isp_dyna_params,
    _validate_isp_dyna_params,
    _validate_scheduled_waterfall_isp_dyna_params,
    canonical_algorithm_name,
    is_alt_isp_dyna_algorithm,
    is_gen_alt_isp_dyna_algorithm,
    is_arch_isp_dyna_algorithm,
    is_nr_rm_isp_dyna_algorithm,
    is_nr1_rm_isp_dyna_algorithm,
    is_arch_waterfall_isp_dyna_algorithm,
    is_dd_geo_waterfall_isp_dyna_algorithm,
    is_elastic_dd_geo_waterfall_isp_dyna_algorithm,
    is_isp_dyna_algorithm,
    is_scheduled_waterfall_isp_dyna_algorithm,
    normalize_lr_type_name,
)
from ..core.types import CpuExperimentTask, ExperimentExecutionRow, GpuBatchTask
from .deterministic_dispatch import (
    execute_batched_control_tasks,
    DeterministicTaskSpec,
    PreparedDeterministicTask,
    execute_batched_pe_tasks,
    execute_prepared_deterministic_task,
    normalize_algorithm_config as _shared_normalize_algorithm_config,
    parse_deterministic_task,
    prepare_deterministic_task,
    run_deterministic_task,
)
from .deterministic_runtime import apply_model_error_for_task as _apply_model_error_for_task
from .factory import create_environment_jax as create_environment

_SEED_SENSITIVE_ENVIRONMENTS = frozenset(
    {"garnet", "stochastic_grid", "four_rooms", "dense_reward_grid"}
)
_TASK_ATTR_ALIASES = {"lambda": "lambda_value"}
_BATCHABLE_MODEL_ERROR_MODES = frozenset(
    {
        "support_uniform",
        "smooth",
        "self_loop",
        "identity",
        "temperature",
        "temp",
        "softmax",
        "dirichlet",
        "dirichlet_noise",
    }
)


def _resolve_jax_class(name: str):
    from ..domains.jax_rl import (
        JaxBatchedAltISPDyna,
        JaxBatchedGenAltISPDyna,
        JaxBatchedArchISPDyna,
        JaxBatchedNRRMISPDyna,
        JaxBatchedNR1RMISPDyna,
        JaxBatchedArchWaterfallISPDyna,
        JaxBatchedDDGeoWaterfallISPDyna,
        JaxBatchedDyna,
        JaxBatchedElasticDDGeoWaterfallISPDyna,
        JaxBatchedISPDyna,
        JaxBatchedOSDyna,
        JaxBatchedQLearning,
        JaxBatchedScheduledWaterfallISPDyna,
    )

    lowered = canonical_algorithm_name(name)
    if lowered in {"qlearning", "q_learning"}:
        return JaxBatchedQLearning
    if (
        "msdyna" in lowered
        or "osvi" in lowered
        or "os_dyna" in lowered
        or ("os" in lowered and "dyna" in lowered)
    ):
        return JaxBatchedOSDyna
    if lowered == "alt_isp_dyna":
        return JaxBatchedAltISPDyna
    if lowered == "gen_alt_isp_dyna":
        return JaxBatchedGenAltISPDyna
    if lowered == "arch_isp_dyna":
        return JaxBatchedArchISPDyna
    if lowered == "nr_rm_isp_dyna":
        return JaxBatchedNRRMISPDyna
    if lowered == "nr1_rm_isp_dyna":
        return JaxBatchedNR1RMISPDyna
    if lowered == "dd_geo_waterfall_isp_dyna":
        return JaxBatchedDDGeoWaterfallISPDyna
    if lowered == "elastic_dd_geo_waterfall_isp_dyna":
        return JaxBatchedElasticDDGeoWaterfallISPDyna
    if lowered == "arch_waterfall_isp_dyna":
        return JaxBatchedArchWaterfallISPDyna
    if lowered == "scheduled_waterfall_isp_dyna":
        return JaxBatchedScheduledWaterfallISPDyna
    if "isp" in lowered and "dyna" in lowered:
        return JaxBatchedISPDyna
    if "dyna" in lowered:
        return JaxBatchedDyna
    return None


def _task_attr_name(key: str) -> str:
    return _TASK_ATTR_ALIASES.get(key, key)


def _task_has_key(task_args: GpuBatchTask | CpuExperimentTask | Mapping[str, Any], key: str) -> bool:
    if isinstance(task_args, Mapping):
        return key in task_args
    return hasattr(task_args, _task_attr_name(key))


def _task_value(task_args: GpuBatchTask | CpuExperimentTask | Mapping[str, Any], key: str) -> Any:
    if isinstance(task_args, Mapping):
        return task_args[key]
    return getattr(task_args, _task_attr_name(key))


def _run_gpu_batch_task_train_then_offline(
    *,
    task_args: GpuBatchTask | Dict[str, Any],
    env_name: str,
    env_params: Mapping[str, Any],
    lam: float,
    base_seed: int,
    n_seeds: int,
    max_queries: int,
    algo_configs: Dict[str, Dict[str, Any]],
    mdp,
) -> List[ExperimentExecutionRow]:
    from ..domains.control import value_iteration_control
    from ..domains.jax_rl import (
        JAX_RL_DEFAULT_RETURN_TOLERANCE_REL,
        JAX_RL_DEFAULT_SUSTAINED_WINDOW,
        jax_rl_band_radius,
        jax_rl_seedwise_hit_metrics,
    )
    from .hyper_batch import run_hyper_batched_task

    seed_values = [base_seed + index for index in range(n_seeds)]
    configs = []
    for name, params in algo_configs.items():
        config = {"label": name, "algorithm": name}
        config.update(params)
        configs.append(config)

    traces = run_hyper_batched_task(
        env_name=env_name,
        configs=configs,
        n_seeds=n_seeds,
        algorithms=list(algo_configs.keys()),
        samples=max_queries,
        eval_freq=1,
        env_params=dict(env_params),
        backend="jax",
        use_fused_loop=False,
        use_shard_map=False,
        seed_values=seed_values,
        evaluation_pipeline="train_then_offline",
        offline_eval_root=(
            _task_value(task_args, "offline_eval_root")
            if _task_has_key(task_args, "offline_eval_root")
            else None
        ),
    )

    v_star, _ = value_iteration_control(mdp)
    target_value = float(jnp.asarray(v_star, dtype=jnp.float64)[0])
    band_radius = float(
        jax_rl_band_radius(target_value, JAX_RL_DEFAULT_RETURN_TOLERANCE_REL)
    )

    first_hit_by_algorithm: dict[str, tuple[int, ...]] = {}
    for name in algo_configs:
        trace = traces[name][name]
        metrics = jax_rl_seedwise_hit_metrics(
            x_axis=[int(value) for value in jnp.asarray(trace.x_axis).tolist()],
            mean_trace=[float(value) for value in jnp.asarray(trace.mean).tolist()],
            seed_payload=jnp.asarray(trace.seeds),
            target_value=target_value,
            band_radius=band_radius,
            sustained_window=JAX_RL_DEFAULT_SUSTAINED_WINDOW,
            max_samples=max_queries,
        )
        first_hit_by_algorithm[name] = metrics.first_hit_seedwise_samples

    return [
        ExperimentExecutionRow(
            lambda_value=lam,
            seed=base_seed + index,
            results={
                name: int(first_hit_by_algorithm[name][index]) for name in algo_configs
            },
        )
        for index in range(n_seeds)
    ]


def run_deterministic_jax_task(
    task_args: CpuExperimentTask | Dict[str, Any],
) -> ExperimentExecutionRow:
    """Run one maintained deterministic JAX task on the currently selected device."""
    return run_deterministic_task(
        task_args,
        create_environment_fn=create_environment,
        apply_model_error_fn=_apply_model_error_for_task,
    )


def _can_share_deterministic_environment(specs: List[DeterministicTaskSpec]) -> bool:
    if not specs:
        return False
    first = specs[0]
    if first.env_name in _SEED_SENSITIVE_ENVIRONMENTS:
        return False
    return all(
        spec.env_name == first.env_name and spec.env_params == first.env_params for spec in specs[1:]
    )


def _can_batch_deterministic_model_error(specs: List[DeterministicTaskSpec]) -> bool:
    if not specs:
        return False
    first_mode = specs[0].error_mode.lower().replace("-", "_")
    if first_mode not in _BATCHABLE_MODEL_ERROR_MODES:
        return False
    return all(spec.error_mode.lower().replace("-", "_") == first_mode for spec in specs[1:])


def _build_batched_p_hat(shared_mdp, specs: List[DeterministicTaskSpec]) -> jax.Array:
    lam_batch = jnp.asarray([spec.lam for spec in specs], dtype=jnp.float64)
    seed_batch = jnp.asarray([spec.seed for spec in specs], dtype=jnp.int32)
    first = specs[0]

    def apply_one(lam: jax.Array, seed: jax.Array) -> jax.Array:
        return _apply_model_error_for_task(
            shared_mdp.P,
            lam=lam,
            mode=first.error_mode,
            seed=seed,
            env_name=first.env_name,
            env_params=first.env_params,
        )

    return jax.vmap(apply_one)(lam_batch, seed_batch)


def run_deterministic_jax_tasks(
    task_args_list: Iterable[CpuExperimentTask | Dict[str, Any]],
) -> List[ExperimentExecutionRow]:
    """Run a deterministic maintained task bundle on one selected JAX device."""

    specs = [parse_deterministic_task(task_args) for task_args in task_args_list]
    if not specs:
        return []

    shared_mdp = None
    if _can_share_deterministic_environment(specs):
        shared_env_params = dict(specs[0].env_params)
        shared_mdp = create_environment(specs[0].env_name, **shared_env_params)

    prepared_tasks = []
    batched_p_hat = None
    if shared_mdp is not None and _can_batch_deterministic_model_error(specs):
        batched_p_hat = _build_batched_p_hat(shared_mdp, specs)

    for index, spec in enumerate(specs):
        if shared_mdp is not None and batched_p_hat is not None:
            prepared = PreparedDeterministicTask(spec=spec, mdp=shared_mdp, p_hat=batched_p_hat[index])
        else:
            prepared = prepare_deterministic_task(
                spec,
                create_environment_fn=create_environment,
                apply_model_error_fn=_apply_model_error_for_task,
                shared_mdp=shared_mdp,
            )
        prepared_tasks.append(prepared)

    if all(prepared.spec.problem_type == "pe" for prepared in prepared_tasks):
        return execute_batched_pe_tasks(prepared_tasks)
    if all(prepared.spec.problem_type == "control" for prepared in prepared_tasks):
        return execute_batched_control_tasks(prepared_tasks)

    return [execute_prepared_deterministic_task(prepared) for prepared in prepared_tasks]


def _partition_deterministic_tasks_for_devices(
    task_args_list: List[CpuExperimentTask | Dict[str, Any]],
    num_devices: int,
) -> List[List[tuple[int, CpuExperimentTask | Dict[str, Any]]]]:
    partitions: List[List[tuple[int, CpuExperimentTask | Dict[str, Any]]]] = [
        [] for _ in range(max(1, num_devices))
    ]
    for index, task in enumerate(task_args_list):
        partitions[index % max(1, num_devices)].append((index, task))
    return [partition for partition in partitions if partition]


def _run_deterministic_jax_task_chunk_on_device(
    indexed_tasks: List[tuple[int, CpuExperimentTask | Dict[str, Any]]],
    device: Any,
) -> List[tuple[int, ExperimentExecutionRow]]:
    with jax.default_device(device):
        rows = run_deterministic_jax_tasks(task for _, task in indexed_tasks)
    return [(index, row) for (index, _), row in zip(indexed_tasks, rows)]


def run_deterministic_jax_tasks_multi_device(
    task_args_list: Iterable[CpuExperimentTask | Dict[str, Any]],
) -> List[ExperimentExecutionRow]:
    """Run maintained deterministic tasks across visible JAX devices when available."""

    tasks = list(task_args_list)
    if not tasks:
        return []

    devices = list(jax.local_devices())
    if len(devices) <= 1:
        return run_deterministic_jax_tasks(tasks)

    partitions = _partition_deterministic_tasks_for_devices(tasks, len(devices))
    print(
        f"Deterministic JAX Multi-Device Runner: {len(devices)} devices, {len(tasks)} tasks, {len(partitions)} shards"
    )

    indexed_rows: List[tuple[int, ExperimentExecutionRow]] = []
    with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
        futures = [
            executor.submit(_run_deterministic_jax_task_chunk_on_device, partition, devices[partition_index])
            for partition_index, partition in enumerate(partitions)
        ]
        for future in futures:
            indexed_rows.extend(future.result())

    indexed_rows.sort(key=lambda item: item[0])
    return [row for _, row in indexed_rows]


def run_gpu_batch_task(task_args: GpuBatchTask | Dict[str, Any]) -> List[ExperimentExecutionRow]:
    """Run a batch of JAX stochastic baseline experiments on one device."""

    backend = _task_value(task_args, "backend") if _task_has_key(task_args, "backend") else "jax"
    if backend != "jax":
        raise ValueError("Active GPU stochastic execution is JAX-only.")

    env_name = _task_value(task_args, "env_name")
    env_params = _task_value(task_args, "env_params")
    lam = _task_value(task_args, "lambda")
    base_seed = _task_value(task_args, "seed")
    n_seeds = _task_value(task_args, "n_seeds")
    max_queries = _task_value(task_args, "max_queries") if _task_has_key(task_args, "max_queries") else 5000
    threshold = _task_value(task_args, "threshold") if _task_has_key(task_args, "threshold") else 1e-3
    algorithms_arg = (
        _task_value(task_args, "algorithms") if _task_has_key(task_args, "algorithms") else ["qlearning"]
    )
    evaluation_pipeline = (
        _task_value(task_args, "evaluation_pipeline")
        if _task_has_key(task_args, "evaluation_pipeline")
        else "online"
    )

    local_env_params = env_params.copy()
    local_env_params.setdefault("seed", base_seed)
    mdp = create_environment(env_name, **local_env_params)

    algo_configs = _normalize_algorithm_config(algorithms_arg)
    for name, params in algo_configs.items():
        if is_isp_dyna_algorithm(name):
            _validate_isp_dyna_params(dict(params), total_samples=max_queries, context_name="isp_dyna")
        if is_alt_isp_dyna_algorithm(name):
            _validate_alt_isp_dyna_params(
                dict(params),
                total_samples=max_queries,
                context_name="alt_isp_dyna",
            )
        if is_gen_alt_isp_dyna_algorithm(name):
            _validate_gen_alt_isp_dyna_params(
                dict(params),
                total_samples=max_queries,
                context_name="gen_alt_isp_dyna",
            )
        if is_arch_isp_dyna_algorithm(name):
            _validate_arch_isp_dyna_params(
                dict(params),
                total_samples=max_queries,
                context_name="arch_isp_dyna",
            )
        if is_nr_rm_isp_dyna_algorithm(name):
            _validate_nr_rm_isp_dyna_params(
                dict(params),
                total_samples=max_queries,
                context_name="nr_rm_isp_dyna",
            )
        if is_nr1_rm_isp_dyna_algorithm(name):
            _validate_nr1_rm_isp_dyna_params(
                dict(params),
                total_samples=max_queries,
                context_name="nr1_rm_isp_dyna",
            )
        if is_scheduled_waterfall_isp_dyna_algorithm(name):
            _validate_scheduled_waterfall_isp_dyna_params(
                dict(params),
                total_samples=max_queries,
                context_name="scheduled_waterfall_isp_dyna",
            )
        if is_arch_waterfall_isp_dyna_algorithm(name):
            _validate_arch_waterfall_isp_dyna_params(
                dict(params),
                total_samples=max_queries,
                context_name="arch_waterfall_isp_dyna",
            )
        if is_dd_geo_waterfall_isp_dyna_algorithm(name):
            _validate_dd_geo_waterfall_isp_dyna_params(
                dict(params),
                total_samples=max_queries,
                context_name="dd_geo_waterfall_isp_dyna",
            )
        if is_elastic_dd_geo_waterfall_isp_dyna_algorithm(name):
            _validate_elastic_dd_geo_waterfall_isp_dyna_params(
                dict(params),
                total_samples=max_queries,
                context_name="elastic_dd_geo_waterfall_isp_dyna",
            )
    isp_epoch_spans = {
        name: int(params["J"]) * int(params["batch_size"])
        for name, params in algo_configs.items()
        if is_isp_dyna_algorithm(name)
    }
    arch_epoch_spans = {
        name: int(params["J"])
        for name, params in algo_configs.items()
        if is_arch_isp_dyna_algorithm(name)
    }
    rm_epoch_spans: dict[str, int] = {}
    nr1_epoch_spans = {
        name: max(int(params.get("J", 1) or 1), 1)
        for name, params in algo_configs.items()
        if is_nr1_rm_isp_dyna_algorithm(name)
    }
    t1_rm_epoch_spans: dict[str, int] = {}
    scheduled_epoch_spans = {
        name: int(params["J"])
        for name, params in algo_configs.items()
        if is_scheduled_waterfall_isp_dyna_algorithm(name)
    }
    arch_waterfall_epoch_spans = {
        name: int(params["J"])
        for name, params in algo_configs.items()
        if is_arch_waterfall_isp_dyna_algorithm(name)
    }
    dd_geo_batch_sizes = {
        name: int(params["batch_size"])
        for name, params in algo_configs.items()
        if is_dd_geo_waterfall_isp_dyna_algorithm(name) or is_elastic_dd_geo_waterfall_isp_dyna_algorithm(name)
    }
    if evaluation_pipeline == "train_then_offline":
        return _run_gpu_batch_task_train_then_offline(
            task_args=task_args,
            env_name=env_name,
            env_params=local_env_params,
            lam=lam,
            base_seed=base_seed,
            n_seeds=n_seeds,
            max_queries=max_queries,
            algo_configs=algo_configs,
            mdp=mdp,
        )
    agents = {
        name: agent
        for name, params in algo_configs.items()
        if (agent := _create_jax_agent(name, params, mdp, n_seeds, lam)) is not None
    }
    if not agents:
        return []

    hit_times = {name: [max_queries + 100 for _ in range(n_seeds)] for name in agents}
    converged_mask = {name: [False for _ in range(n_seeds)] for name in agents}

    first_agent = next(iter(agents.values()))
    max_r = float(jnp.max(jnp.abs(first_agent.mdp_R)))
    effect_threshold = max(max_r * threshold, 1e-10)

    for k in range(max_queries):
        for name, agent in agents.items():
            if name in dd_geo_batch_sizes:
                batch_size = dd_geo_batch_sizes[name]
                if (k + 1) % batch_size != 0:
                    continue
                agent.step(k // batch_size)
            else:
                agent.step(k)
            if all(converged_mask[name]):
                continue
            if name in isp_epoch_spans and (k + 1) % isp_epoch_spans[name] != 0:
                continue
            if name in arch_epoch_spans and (k + 1) % arch_epoch_spans[name] != 0:
                continue
            if name in rm_epoch_spans and (k + 1) % rm_epoch_spans[name] != 0:
                continue
            if name in t1_rm_epoch_spans and (k + 1) % t1_rm_epoch_spans[name] != 0:
                continue
            if name in scheduled_epoch_spans and (k + 1) % scheduled_epoch_spans[name] != 0:
                continue
            if name in arch_waterfall_epoch_spans and (k + 1) % arch_waterfall_epoch_spans[name] != 0:
                continue

            resid = _compute_batched_residual(agent, float(mdp.gamma))
            for index, residual in enumerate(resid):
                if converged_mask[name][index]:
                    continue
                if float(residual) <= effect_threshold:
                    hit_times[name][index] = (
                        (k + 1) * nr1_epoch_spans[name]
                        if name in nr1_epoch_spans
                        else k + 1
                    )
                    converged_mask[name][index] = True

        if all(all(mask) for mask in converged_mask.values()):
            break

    return [
        ExperimentExecutionRow(
            lambda_value=lam,
            seed=base_seed + i,
            results={name: int(hit_times[name][i]) for name in agents},
        )
        for i in range(n_seeds)
    ]


def _normalize_algorithm_config(algorithms_arg) -> Dict[str, Dict[str, Any]]:
    return _shared_normalize_algorithm_config(algorithms_arg)


def _create_jax_agent(name: str, params: Dict[str, Any], mdp, n_seeds: int, lam: float):
    cls = _resolve_jax_class(name)
    if cls is None:
        return None

    kwargs: Dict[str, Any] = {"seed": 0}
    lowered = canonical_algorithm_name(name)
    lr_schedule = normalize_lr_type_name(params.get("lr_type", "constant_delay"))

    if lowered in {"qlearning", "q_learning"}:
        kwargs["lr_params"] = (params.get("start_lr", 0.5), params.get("delay", 0))
        kwargs["lr_schedule"] = lr_schedule
    elif (
        "msdyna" in lowered
        or "osvi" in lowered
        or "os_dyna" in lowered
        or ("os" in lowered and "dyna" in lowered)
    ):
        kwargs["lambda_val"] = lam
        kwargs["lr_params"] = (params.get("start_lr", 0.5), params.get("delay", 0))
        kwargs["lr_schedule"] = lr_schedule
    elif is_alt_isp_dyna_algorithm(name):
        kwargs["h_target"] = params["h_target"]
        kwargs["J"] = params["J"]
        kwargs["epsilon"] = params.get("epsilon", 0.1)
        kwargs["lambda_val"] = lam
    elif is_gen_alt_isp_dyna_algorithm(name):
        kwargs["h_target"] = params["h_target"]
        kwargs["J"] = params["J"]
        kwargs["lambda_val"] = lam
    elif is_arch_isp_dyna_algorithm(name):
        kwargs["h"] = params["h"]
        kwargs["J"] = params["J"]
        kwargs["lambda_val"] = lam
        kwargs["lr_params"] = (params["start_lr"], params["delay"])
        kwargs["lr_schedule"] = lr_schedule
        kwargs["sampling_distribution"] = params.get("sampling_distribution", "uniform_sa")
    elif is_nr_rm_isp_dyna_algorithm(name) or is_nr1_rm_isp_dyna_algorithm(name):
        kwargs["lambda_val"] = lam
        kwargs["lr_params"] = (params["start_lr"], params["delay"])
        kwargs["lr_schedule"] = lr_schedule
        kwargs["sampling_distribution"] = params.get("sampling_distribution", "uniform_sa")
        if "h" in params:
            kwargs["J"] = params["J"]
            kwargs["h"] = params["h"]
        elif any(key in params for key in ("h_init", "h_terminal", "h_floor")):
            kwargs["J"] = params["J"]
            kwargs["h_init"] = params["h_init"]
            kwargs["h_terminal"] = params["h_terminal"]
            kwargs["h_floor"] = params.get("h_floor", 1.0)
    elif is_dd_geo_waterfall_isp_dyna_algorithm(name) or is_elastic_dd_geo_waterfall_isp_dyna_algorithm(name):
        kwargs["h_init"] = params["h_init"]
        kwargs["J_max"] = params["J_max"]
        kwargs["batch_size"] = params["batch_size"]
        kwargs["lambda_val"] = lam
        kwargs["lr_params"] = (params["start_lr"], params["delay"])
        kwargs["lr_schedule"] = lr_schedule
        kwargs["decay_factor"] = params.get("decay_factor", 0.5)
        kwargs["sampling_distribution"] = params.get("sampling_distribution", "uniform_sa")
    elif is_scheduled_waterfall_isp_dyna_algorithm(name):
        kwargs["h_max"] = params["h_max"]
        kwargs["h_min"] = params["h_min"]
        kwargs["J"] = params["J"]
        kwargs["lambda_val"] = lam
        kwargs["lr_params"] = (params["start_lr"], params["delay"])
        kwargs["lr_schedule"] = lr_schedule
        kwargs["schedule_decay_rate"] = params.get("schedule_decay_rate", 5.0)
        kwargs["sampling_distribution"] = params.get("sampling_distribution", "uniform_sa")
    elif is_arch_waterfall_isp_dyna_algorithm(name):
        kwargs["h_max"] = params["h_max"]
        kwargs["h_min"] = params["h_min"]
        kwargs["J"] = params["J"]
        kwargs["lambda_val"] = lam
        kwargs["lr_params"] = (params["start_lr"], params["delay"])
        kwargs["lr_schedule"] = lr_schedule
        kwargs["schedule_decay_rate"] = params.get("schedule_decay_rate", 5.0)
        kwargs["sampling_distribution"] = params.get("sampling_distribution", "uniform_sa")
    elif is_isp_dyna_algorithm(name):
        kwargs["h"] = params["h"]
        kwargs["J"] = params["J"]
        kwargs["batch_size"] = params["batch_size"]
        kwargs["lambda_val"] = lam
        kwargs["lr_params"] = (params["start_lr"], params["delay"])
        kwargs["lr_schedule"] = lr_schedule
        kwargs["sampling_distribution"] = params.get("sampling_distribution", "uniform_sa")
    else:
        kwargs["lambda_val"] = lam
        kwargs["planning_steps"] = params.get("planning_steps", 10)
        kwargs["lr_params"] = (params.get("start_lr", 0.5), params.get("delay", 0))
        kwargs["lr_schedule"] = lr_schedule

    return cls(mdp.P, mdp.R, mdp.gamma, n_seeds, **kwargs)


def _compute_batched_residual(agent, gamma: float) -> jax.Array:
    v = jnp.asarray(agent.state.V, dtype=jnp.float64)
    p = jnp.asarray(agent.mdp_P, dtype=jnp.float64)
    r = jnp.asarray(agent.mdp_R, dtype=jnp.float64)

    term = jnp.einsum("sak,bk->bsa", p, v)
    q = r[None, ...] + gamma * term
    t_v = jnp.max(q, axis=2)
    return jnp.max(jnp.abs(t_v - v), axis=1)
