"""CPU-host JAX worker for deterministic PE/control experiments."""

import os
from typing import Any, Dict

if os.environ.get("JAX_PLATFORMS") == "cpu":
    import logging

    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

from .factory import create_environment_jax as create_environment
from ..core.types import CpuExperimentTask, ExperimentExecutionRow, MDP
from ..domains.control import (
    AISP_Control,
    ISP_Control,
    MSVI_Control,
    Shadow_ISP_Control,
    VI_Control,
    WaterfallISP_Control,
    pick_isp_params_control,
)
from .deterministic_dispatch import (
    _build_shadow_config,
    _build_waterfall_config,
    _find_hit_in_trace,
    _get_hit_time,
    normalize_algorithm_config as _shared_normalize_algorithm_config,
    run_control_algorithms as _shared_run_control_algorithms,
    run_deterministic_task,
    run_pe_algorithms as _shared_run_pe_algorithms,
)
from .deterministic_runtime import (
    apply_model_error_for_task as _apply_model_error_for_task,
    seeded_shadow_init,
)


def worker_task(task_args: CpuExperimentTask | Dict[str, Any]) -> ExperimentExecutionRow:
    """Run one deterministic maintained task on a CPU JAX host."""
    return run_deterministic_task(
        task_args,
        create_environment_fn=create_environment,
        apply_model_error_fn=_apply_model_error_for_task,
        run_pe_algorithms_fn=_run_pe_algorithms,
        run_control_algorithms_fn=_run_control_algorithms,
    )


def _normalize_algorithm_config(algorithms_arg) -> Dict[str, Dict]:
    """Convert algorithm argument to normalized dict format."""
    return _shared_normalize_algorithm_config(algorithms_arg)


def _run_pe_algorithms(
    mdp: MDP,
    P_hat: Any,
    algo_configs: Dict[str, Dict],
    max_queries: int,
    threshold: float,
    model_norm: str,
    value_norm: str,
    value_normalized: bool,
) -> Dict[str, int]:
    """Run Policy Evaluation algorithms."""
    return _shared_run_pe_algorithms(
        mdp,
        P_hat,
        algo_configs,
        max_queries,
        threshold,
        model_norm,
        value_norm,
        value_normalized,
    )


def _run_control_algorithms(
    mdp: MDP,
    P_hat: Any,
    algo_configs: Dict[str, Dict],
    max_queries: int,
    threshold: float,
    seed: int,
    *,
    metric_mode: str = "policy_value_error",
    value_norm: str = "L1",
    value_normalized: bool = True,
    target_bundle: Any | None = None,
) -> Dict[str, int]:
    """Run Control algorithms."""
    return _shared_run_control_algorithms(
        mdp,
        P_hat,
        algo_configs,
        max_queries,
        threshold,
        seed,
        metric_mode=metric_mode,
        value_norm=value_norm,
        value_normalized=value_normalized,
        target_bundle=target_bundle,
        vi_control_cls=VI_Control,
        msvi_control_cls=MSVI_Control,
        isp_control_cls=ISP_Control,
        aisp_control_cls=AISP_Control,
        shadow_isp_control_cls=Shadow_ISP_Control,
        waterfall_isp_control_cls=WaterfallISP_Control,
        pick_isp_params_control_fn=pick_isp_params_control,
        seeded_shadow_init_fn=seeded_shadow_init,
    )


__all__ = [
    "worker_task",
    "_build_shadow_config",
    "_build_waterfall_config",
    "_find_hit_in_trace",
    "_get_hit_time",
    "_normalize_algorithm_config",
    "_run_control_algorithms",
    "_run_pe_algorithms",
]
