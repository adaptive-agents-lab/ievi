"""BESP JAX-native package surface."""
# ruff: noqa: E402

from __future__ import annotations

import os
from typing import Any, Callable

os.environ.setdefault("JAX_ENABLE_X64", "True")

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass


def _missing_optional_callable(feature: str, dependency: str, exc: Exception) -> Callable[..., Any]:
    def _raiser(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            f"{feature} requires the optional dependency '{dependency}'."
        ) from exc

    return _raiser


def _lazy_export(module_name: str, attr_name: str) -> Any:
    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[attr_name] = value
    return value

from .core.types import MDP
from .domains.control import (
    AISP_Control,
    ISP_Control,
    MSVI_Control,
    Shadow_ISP_Control,
    VI_Control,
    build_policy_matrices,
    pick_isp_params_control,
    run_control_experiment,
    solve_value,
    value_iteration_control,
)
from .domains.policy_evaluation import (
    BESPParams,
    Shadow_BESP,
    abesp_pe,
    besp_pe,
    besp_precompute,
    model_error_stats,
    osvi_pe,
    pick_besp_rho_half,
    shadow_besp_pe,
    vi_pe,
)
from .environments import (
    generate_garnet,
    generate_river_swim,
    generate_chain,
    generate_modified_cliffwalk,
    generate_maze,
    generate_stochastic_grid,
    generate_four_rooms,
    generate_dense_reward_grid,
    generate_taxi,
    generate_mini_breakout,
)
from .runners import (
    create_environment,
    worker_task,
    run_gpu_batch_task,
)

try:
    from .runners.hyper_batch import run_hyper_batched_task
except ImportError as exc:
    run_hyper_batched_task = _missing_optional_callable(
        "run_hyper_batched_task",
        "jax",
        exc,
    )

_LAZY_EXPORTS = {
    "max_row_sum_norm": ("besp.utils", "max_row_sum_norm"),
    "model_error_norm": ("besp.utils", "model_error_norm"),
    "compute_value_error": ("besp.utils", "compute_value_error"),
    "solve_inverse": ("besp.utils", "solve_inverse"),
    "solve_iterative": ("besp.utils", "solve_iterative"),
    "normalized_l1_error": ("besp.utils", "normalized_l1_error"),
    "get_bellman_residual_norm": ("besp.utils", "get_bellman_residual_norm"),
    "smooth_model": ("besp.models", "smooth_model"),
    "apply_model_error": ("besp.models", "apply_model_error"),
    "mix_transitions": ("besp.models", "mix_transitions"),
    "right_constant_curve": ("besp.analysis", "right_constant_curve"),
    "aggregate_curves": ("besp.analysis", "aggregate_curves"),
    "aggregate_thresholds": ("besp.analysis", "aggregate_thresholds"),
    "queries_to_threshold": ("besp.analysis", "queries_to_threshold"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return _lazy_export(module_name, attr_name)

__version__ = "2.0.0"

__all__ = [
    "MDP",
    "max_row_sum_norm",
    "model_error_norm",
    "compute_value_error",
    "solve_inverse",
    "solve_iterative",
    "normalized_l1_error",
    "get_bellman_residual_norm",
    "smooth_model",
    "apply_model_error",
    "mix_transitions",
    "generate_garnet",
    "generate_river_swim",
    "generate_chain",
    "generate_modified_cliffwalk",
    "generate_maze",
    "generate_stochastic_grid",
    "generate_four_rooms",
    "generate_dense_reward_grid",
    "generate_taxi",
    "generate_mini_breakout",
    "vi_pe",
    "osvi_pe",
    "besp_pe",
    "besp_precompute",
    "Shadow_BESP",
    "shadow_besp_pe",
    "abesp_pe",
    "value_iteration_control",
    "build_policy_matrices",
    "solve_value",
    "VI_Control",
    "MSVI_Control",
    "ISP_Control",
    "AISP_Control",
    "Shadow_ISP_Control",
    "run_control_experiment",
    "BESPParams",
    "model_error_stats",
    "pick_besp_rho_half",
    "pick_isp_params_control",
    "create_environment",
    "worker_task",
    "run_gpu_batch_task",
    "right_constant_curve",
    "aggregate_curves",
    "aggregate_thresholds",
    "queries_to_threshold",
    "run_hyper_batched_task",
]
