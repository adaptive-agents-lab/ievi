"""Typed configuration adapters for active BESP entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .env_variants import load_env_variants_manifest
from .types import (
    CpuExperimentTask,
    GpuBatchTask,
    default_evaluation_pipeline_for_problem_type,
    default_metric_mode_for_problem_type,
    resolve_evaluation_pipeline,
)


GPU_ONLY_ALGORITHMS = frozenset(
    {
        "qlearning",
        "isp_dyna",
        "alt_isp_dyna",
        "gen_alt_isp_dyna",
        "arch_isp_dyna",
        "nr_rm_isp_dyna",
        "nr1_rm_isp_dyna",
        "scheduled_waterfall_isp_dyna",
        "arch_waterfall_isp_dyna",
        "dd_geo_waterfall_isp_dyna",
        "elastic_dd_geo_waterfall_isp_dyna",
        "nr1_staircase_waterfall_rm_isp_dyna",
        "nr1_geo_waterfall_rm_isp_dyna",
        "msdyna_control",
        "dyna_control",
        "os_dyna",
        "osvi_control",
    }
)

_CANONICAL_ENV_PARAM_KEYS = frozenset(
    {
        "gamma",
        "n_states",
        "n_actions",
        "branching",
        "reward_states",
        "size",
        "slip_prob",
        "hole_positions",
    }
)

_MAINTAINED_EXPERIMENT_KEYS = frozenset(
    {
        "name",
        "env",
        "env_params",
        "n_seeds",
        "seed",
        "lambdas",
        "max_queries",
        "threshold",
        "error_mode",
        "problem_type",
        "model_norm",
        "value_norm",
        "value_normalized",
        "emit_pe_traces",
        "emit_control_traces",
        "metric_mode",
        "evaluation_pipeline",
        "deterministic_target_manifest",
        "env_variants_manifest",
        "algorithms",
    }
)

_ISP_DYNA_REQUIRED_PARAMS = frozenset({"h", "J", "batch_size", "start_lr", "delay"})
_ISP_DYNA_FORBIDDEN_PARAMS = frozenset({"lambda", "planning_steps", "use_online_model"})
_ALT_ISP_DYNA_REQUIRED_PARAMS = frozenset({"h_target", "J", "epsilon"})
_ALT_ISP_DYNA_FORBIDDEN_PARAMS = frozenset(
    {
        "lambda",
        "interaction_mode",
        "sampling_distribution",
        "start_lr",
        "delay",
        "lr_type",
        "batch_size",
        "planning_steps",
        "use_online_model",
        "h",
        "h_max",
        "h_min",
        "h_init",
        "J_max",
    }
)
_GEN_ALT_ISP_DYNA_REQUIRED_PARAMS = frozenset({"h_target", "J"})
_GEN_ALT_ISP_DYNA_FORBIDDEN_PARAMS = frozenset(
    {
        "lambda",
        "epsilon",
        "interaction_mode",
        "sampling_distribution",
        "start_lr",
        "delay",
        "lr_type",
        "batch_size",
        "planning_steps",
        "use_online_model",
        "h",
        "h_max",
        "h_min",
        "h_init",
        "J_max",
    }
)
_ARCH_ISP_DYNA_REQUIRED_PARAMS = frozenset({"h", "J", "start_lr", "delay"})
_ARCH_ISP_DYNA_FORBIDDEN_PARAMS = frozenset(
    {"lambda", "batch_size", "planning_steps", "use_online_model"}
)
_NR_RM_ISP_DYNA_REQUIRED_PARAMS = frozenset({"start_lr", "delay"})
_NR_RM_ISP_DYNA_FORBIDDEN_PARAMS = frozenset(
    {
        "batch_size",
        "planning_steps",
        "use_online_model",
        "h_target",
        "h_max",
        "h_min",
        "J_max",
        "decay_factor",
        "schedule_decay_rate",
        "epsilon",
    }
)
_SCHEDULED_WATERFALL_ISP_DYNA_REQUIRED_PARAMS = frozenset(
    {"h_max", "h_min", "J", "start_lr", "delay"}
)
_SCHEDULED_WATERFALL_ISP_DYNA_FORBIDDEN_PARAMS = frozenset(
    {
        "lambda",
        "h",
        "h_init",
        "J_max",
        "batch_size",
        "planning_steps",
        "use_online_model",
        "decay_factor",
    }
)
_ARCH_WATERFALL_ISP_DYNA_REQUIRED_PARAMS = _SCHEDULED_WATERFALL_ISP_DYNA_REQUIRED_PARAMS
_ARCH_WATERFALL_ISP_DYNA_FORBIDDEN_PARAMS = _SCHEDULED_WATERFALL_ISP_DYNA_FORBIDDEN_PARAMS
_DD_GEO_WATERFALL_ISP_DYNA_REQUIRED_PARAMS = frozenset(
    {"h_init", "J_max", "batch_size", "start_lr", "delay"}
)
_DD_GEO_WATERFALL_ISP_DYNA_FORBIDDEN_PARAMS = frozenset(
    {
        "lambda",
        "h",
        "J",
        "planning_steps",
        "use_online_model",
        "h_min",
        "stall_patience",
        "alpha",
    }
)
_ELASTIC_DD_GEO_WATERFALL_ISP_DYNA_REQUIRED_PARAMS = _DD_GEO_WATERFALL_ISP_DYNA_REQUIRED_PARAMS
_ELASTIC_DD_GEO_WATERFALL_ISP_DYNA_FORBIDDEN_PARAMS = _DD_GEO_WATERFALL_ISP_DYNA_FORBIDDEN_PARAMS
_NR1_STAIRCASE_WATERFALL_RM_ISP_DYNA_REQUIRED_PARAMS = frozenset(
    {"J", "h_init", "h_min", "stall_patience", "decay_factor", "start_lr", "delay"}
)
_NR1_STAIRCASE_WATERFALL_RM_ISP_DYNA_FORBIDDEN_PARAMS = frozenset(
    {
        "h",
        "h_target",
        "h_max",
        "h_terminal",
        "h_floor",
        "h_schedule_horizon",
        "h_schedule_shape",
        "h_schedule_finish_frac",
        "J_max",
        "alpha",
        "batch_size",
        "planning_steps",
        "use_online_model",
        "schedule_decay_rate",
        "epsilon",
    }
)
_NR1_GEO_WATERFALL_RM_ISP_DYNA_REQUIRED_PARAMS = frozenset(
    {"J", "h_init", "h_min", "alpha", "start_lr", "delay"}
)
_NR1_GEO_WATERFALL_RM_ISP_DYNA_FORBIDDEN_PARAMS = frozenset(
    {
        "h",
        "h_target",
        "h_max",
        "h_terminal",
        "h_floor",
        "h_schedule_horizon",
        "h_schedule_shape",
        "h_schedule_finish_frac",
        "J_max",
        "stall_patience",
        "decay_factor",
        "batch_size",
        "planning_steps",
        "use_online_model",
        "schedule_decay_rate",
        "epsilon",
    }
)
_GEO_WATERFALL_STAIRCASE_PARAMS = frozenset({"h_min", "stall_patience", "decay_factor"})
_GEO_WATERFALL_PE_ALLOWED_PARAMS = frozenset({"h_init", "alpha"})
_GEO_WATERFALL_CONTROL_ALLOWED_PARAMS = frozenset(
    {"h_init", "alpha", "inner_vi_tol", "inner_vi_max_iters"}
)
_DD_WATERFALL_FORBIDDEN_PARAMS = frozenset({"h_min", "stall_patience", "alpha"})
_DD_WATERFALL_PE_ALLOWED_PARAMS = frozenset({"h_init", "m_max", "decay_factor"})
_DD_WATERFALL_CONTROL_ALLOWED_PARAMS = frozenset(
    {"h_init", "m_max", "decay_factor", "inner_vi_tol", "inner_vi_max_iters"}
)
NR_H_SCHEDULE_SHAPES = frozenset({"exponential", "linear"})


def canonical_algorithm_name(name: str) -> str:
    lowered = str(name).strip().lower().replace("-", "_")
    return lowered


def is_isp_dyna_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "isp_dyna"


def is_alt_isp_dyna_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "alt_isp_dyna"


def is_gen_alt_isp_dyna_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "gen_alt_isp_dyna"


def is_arch_isp_dyna_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "arch_isp_dyna"


def is_nr_rm_isp_dyna_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "nr_rm_isp_dyna"


def is_nr1_rm_isp_dyna_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "nr1_rm_isp_dyna"


def is_scheduled_waterfall_isp_dyna_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "scheduled_waterfall_isp_dyna"


def is_arch_waterfall_isp_dyna_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "arch_waterfall_isp_dyna"


def is_dd_geo_waterfall_isp_dyna_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "dd_geo_waterfall_isp_dyna"


def is_elastic_dd_geo_waterfall_isp_dyna_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "elastic_dd_geo_waterfall_isp_dyna"


def is_nr1_staircase_waterfall_rm_isp_dyna_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "nr1_staircase_waterfall_rm_isp_dyna"


def is_nr1_geo_waterfall_rm_isp_dyna_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "nr1_geo_waterfall_rm_isp_dyna"


def is_nr1_waterfall_rm_isp_dyna_algorithm(name: str) -> bool:
    return is_nr1_staircase_waterfall_rm_isp_dyna_algorithm(name) or is_nr1_geo_waterfall_rm_isp_dyna_algorithm(name)


def is_geo_waterfall_pe_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "geo_waterfall_besp"


def is_geo_waterfall_control_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "geo_waterfall"


def is_dd_waterfall_pe_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "dd_waterfall_besp"


def is_dd_waterfall_control_algorithm(name: str) -> bool:
    return canonical_algorithm_name(name) == "dd_waterfall"


def normalize_lr_type_name(value: Any) -> str:
    lowered = str(value or "constant_delay").strip().lower().replace("-", "_")
    if lowered in {"constant_delay", "constantanddelay"}:
        return "constant_delay"
    if lowered in {"rescaled_linear", "rescaledlinear"}:
        return "rescaled_linear"
    raise ValueError(
        "Supported lr_type values are 'constant_delay'/'ConstantAndDelay' "
        "and 'rescaled_linear'/'RescaledLinear'."
    )


def normalize_nr_h_schedule_shape(value: Any) -> str:
    normalized = str(value if value is not None else "exponential").strip().lower()
    if normalized not in NR_H_SCHEDULE_SHAPES:
        supported = ", ".join(sorted(NR_H_SCHEDULE_SHAPES))
        raise ValueError(f"h_schedule_shape must be one of: {supported}.")
    return normalized


def validate_nr_h_schedule_finish_frac(value: Any) -> float:
    finish_frac = float(value if value is not None else 1.0)
    if not (0.0 < finish_frac <= 1.0):
        raise ValueError("h_schedule_finish_frac must satisfy 0 < h_schedule_finish_frac <= 1.")
    return finish_frac


def _isp_dyna_epoch_span(params: Mapping[str, Any]) -> int:
    return int(params["J"]) * int(params["batch_size"])


def _validate_isp_dyna_params(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
) -> None:
    missing = sorted(key for key in _ISP_DYNA_REQUIRED_PARAMS if key not in params)
    if missing:
        raise ValueError(f"{context_name} requires explicit parameters: {missing}")

    forbidden = sorted(key for key in _ISP_DYNA_FORBIDDEN_PARAMS if key in params)
    if forbidden:
        raise ValueError(f"{context_name} must not define legacy Dyna parameters: {forbidden}")

    if str(params.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
        raise ValueError(f"{context_name} currently supports only sampling_distribution='uniform_sa'.")

    normalize_lr_type_name(params.get("lr_type", "constant_delay"))

    span = _isp_dyna_epoch_span(params)
    if span <= 0:
        raise ValueError(f"{context_name} requires positive J and batch_size values.")
    if total_samples % span != 0:
        raise ValueError(
            f"{context_name} requires the physical sample budget to be divisible by J * batch_size; "
            f"received total_samples={total_samples}, J={params['J']}, batch_size={params['batch_size']}."
        )


def _validate_alt_isp_dyna_params(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
) -> None:
    del total_samples
    missing = sorted(key for key in _ALT_ISP_DYNA_REQUIRED_PARAMS if key not in params)
    if missing:
        raise ValueError(f"{context_name} requires explicit parameters: {missing}")

    forbidden = sorted(key for key in _ALT_ISP_DYNA_FORBIDDEN_PARAMS if key in params)
    if forbidden:
        raise ValueError(
            f"{context_name} is trajectory-only and must not define unsupported parameters: {forbidden}"
        )

    h_target = float(params["h_target"])
    if h_target <= 0.0:
        raise ValueError(f"{context_name} requires h_target > 0.")

    j_depth = int(params["J"])
    if j_depth < 1:
        raise ValueError(f"{context_name} requires J >= 1.")

    epsilon = float(params["epsilon"])
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError(f"{context_name} requires 0 <= epsilon <= 1.")


def _validate_gen_alt_isp_dyna_params(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
) -> None:
    del total_samples
    missing = sorted(key for key in _GEN_ALT_ISP_DYNA_REQUIRED_PARAMS if key not in params)
    if missing:
        raise ValueError(f"{context_name} requires explicit parameters: {missing}")

    forbidden = sorted(key for key in _GEN_ALT_ISP_DYNA_FORBIDDEN_PARAMS if key in params)
    if forbidden:
        raise ValueError(
            f"{context_name} is generative-only and must not define unsupported parameters: {forbidden}"
        )

    h_target = float(params["h_target"])
    if h_target <= 0.0:
        raise ValueError(f"{context_name} requires h_target > 0.")

    j_depth = int(params["J"])
    if j_depth < 1:
        raise ValueError(f"{context_name} requires J >= 1.")


def _validate_arch_isp_dyna_params(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
) -> None:
    missing = sorted(key for key in _ARCH_ISP_DYNA_REQUIRED_PARAMS if key not in params)
    if missing:
        raise ValueError(f"{context_name} requires explicit parameters: {missing}")

    forbidden = sorted(key for key in _ARCH_ISP_DYNA_FORBIDDEN_PARAMS if key in params)
    if forbidden:
        raise ValueError(f"{context_name} must not define legacy or ambiguous parameters: {forbidden}")

    if str(params.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
        raise ValueError(f"{context_name} currently supports only sampling_distribution='uniform_sa'.")

    normalize_lr_type_name(params.get("lr_type", "constant_delay"))

    j_depth = int(params["J"])
    if j_depth < 1:
        raise ValueError(f"{context_name} requires J >= 1.")
    if total_samples % j_depth != 0:
        raise ValueError(
            f"{context_name} requires the physical sample budget to be divisible by J; "
            f"received total_samples={total_samples}, J={params['J']}."
        )


def _validate_nr_rm_isp_dyna_params_common(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
    treat_total_samples_as_outer_steps: bool,
) -> None:
    missing = sorted(key for key in _NR_RM_ISP_DYNA_REQUIRED_PARAMS if key not in params)
    if missing:
        raise ValueError(f"{context_name} requires explicit parameters: {missing}")

    forbidden = sorted(key for key in _NR_RM_ISP_DYNA_FORBIDDEN_PARAMS if key in params)
    if forbidden:
        raise ValueError(f"{context_name} must not define legacy or ambiguous parameters: {forbidden}")

    if str(params.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
        raise ValueError(f"{context_name} currently supports only sampling_distribution='uniform_sa'.")

    normalize_lr_type_name(params.get("lr_type", "constant_delay"))

    if "defect_init_mode" in params:
        raise ValueError(
            f"{context_name} no longer supports the 'defect_init_mode' field; the "
            "defect is always zero-initialized (no oracle reward prior). "
            "Remove the field from your config."
        )

    has_fixed_h = "h" in params
    has_scheduled_h = any(key in params for key in ("h_init", "h_terminal", "h_floor"))
    if has_fixed_h and has_scheduled_h:
        raise ValueError(
            f"{context_name} must define exactly one NR mode: limit, fixed_h, or scheduled_h."
        )
    if not has_fixed_h and not has_scheduled_h:
        if "h_schedule_shape" in params or "h_schedule_finish_frac" in params:
            raise ValueError(f"{context_name} limit mode must not define schedule-shape fields.")
        return
    if has_fixed_h:
        missing_fixed = sorted(key for key in ("h", "J") if key not in params)
        if missing_fixed:
            raise ValueError(f"{context_name} fixed_h mode requires explicit parameters: {missing_fixed}")
        if "h_schedule_shape" in params or "h_schedule_finish_frac" in params:
            raise ValueError(f"{context_name} fixed_h mode must not define schedule-shape fields.")
        if int(params["J"]) < 1:
            raise ValueError(f"{context_name} fixed_h mode requires J >= 1.")
        if float(params["h"]) <= 0.0:
            raise ValueError(f"{context_name} fixed_h mode requires h > 0.")
        if not treat_total_samples_as_outer_steps and total_samples % int(params["J"]) != 0:
            raise ValueError(
                f"{context_name} fixed_h mode requires the physical sample budget to be divisible by J; "
                f"received total_samples={total_samples}, J={params['J']}."
            )
        return
    missing_scheduled = sorted(key for key in ("J", "h_init", "h_terminal") if key not in params)
    if missing_scheduled:
        raise ValueError(
            f"{context_name} scheduled_h mode requires explicit parameters: {missing_scheduled}"
        )
    j_depth = int(params["J"])
    if j_depth < 1:
        raise ValueError(f"{context_name} scheduled_h mode requires J >= 1.")
    h_floor = float(params.get("h_floor", 1.0))
    h_init = float(params["h_init"])
    h_terminal = float(params["h_terminal"])
    if h_floor <= 0.0:
        raise ValueError(f"{context_name} scheduled_h mode requires h_floor > 0.")
    if h_init <= 0.0:
        raise ValueError(f"{context_name} scheduled_h mode requires h_init > 0.")
    if h_terminal <= 0.0:
        raise ValueError(f"{context_name} scheduled_h mode requires h_terminal > 0.")
    if h_init < h_terminal:
        raise ValueError(f"{context_name} scheduled_h mode requires h_init >= h_terminal.")
    if h_terminal < h_floor:
        raise ValueError(f"{context_name} scheduled_h mode requires h_terminal >= h_floor.")
    normalize_nr_h_schedule_shape(params.get("h_schedule_shape", "exponential"))
    validate_nr_h_schedule_finish_frac(params.get("h_schedule_finish_frac", 1.0))
    if not treat_total_samples_as_outer_steps and total_samples % j_depth != 0:
        raise ValueError(
            f"{context_name} scheduled_h mode requires the physical sample budget to be divisible by J; "
            f"received total_samples={total_samples}, J={params['J']}."
        )


def _validate_nr_rm_isp_dyna_params(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
) -> None:
    _validate_nr_rm_isp_dyna_params_common(
        params,
        total_samples=total_samples,
        context_name=context_name,
        treat_total_samples_as_outer_steps=False,
    )


def _validate_nr1_rm_isp_dyna_params(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
) -> None:
    _validate_nr_rm_isp_dyna_params_common(
        params,
        total_samples=total_samples,
        context_name=context_name,
        treat_total_samples_as_outer_steps=True,
    )


def _validate_scheduled_waterfall_isp_dyna_params(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
) -> None:
    missing = sorted(key for key in _SCHEDULED_WATERFALL_ISP_DYNA_REQUIRED_PARAMS if key not in params)
    if missing:
        raise ValueError(f"{context_name} requires explicit parameters: {missing}")

    forbidden = sorted(
        key for key in _SCHEDULED_WATERFALL_ISP_DYNA_FORBIDDEN_PARAMS if key in params
    )
    if forbidden:
        raise ValueError(f"{context_name} must not define legacy or ambiguous parameters: {forbidden}")

    if str(params.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
        raise ValueError(f"{context_name} currently supports only sampling_distribution='uniform_sa'.")

    normalize_lr_type_name(params.get("lr_type", "constant_delay"))

    h_max = float(params["h_max"])
    h_min = float(params["h_min"])
    if h_max <= 0.0:
        raise ValueError(f"{context_name} requires h_max > 0.")
    if h_min <= 0.0:
        raise ValueError(f"{context_name} requires h_min > 0.")
    if h_min > h_max:
        raise ValueError(f"{context_name} requires h_min <= h_max.")

    j_depth = int(params["J"])
    if j_depth < 2:
        raise ValueError(f"{context_name} requires J >= 2.")

    schedule_decay_rate = float(params.get("schedule_decay_rate", 5.0))
    if schedule_decay_rate <= 0.0:
        raise ValueError(f"{context_name} requires schedule_decay_rate > 0.")

    if total_samples % j_depth != 0:
        raise ValueError(
            f"{context_name} requires the physical sample budget to be divisible by J; "
            f"received total_samples={total_samples}, J={params['J']}."
        )


def _validate_arch_waterfall_isp_dyna_params(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
) -> None:
    missing = sorted(key for key in _ARCH_WATERFALL_ISP_DYNA_REQUIRED_PARAMS if key not in params)
    if missing:
        raise ValueError(f"{context_name} requires explicit parameters: {missing}")

    forbidden = sorted(
        key for key in _ARCH_WATERFALL_ISP_DYNA_FORBIDDEN_PARAMS if key in params
    )
    if forbidden:
        raise ValueError(f"{context_name} must not define legacy or ambiguous parameters: {forbidden}")

    if str(params.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
        raise ValueError(f"{context_name} currently supports only sampling_distribution='uniform_sa'.")

    normalize_lr_type_name(params.get("lr_type", "constant_delay"))

    h_max = float(params["h_max"])
    h_min = float(params["h_min"])
    if h_max <= 0.0:
        raise ValueError(f"{context_name} requires h_max > 0.")
    if h_min <= 0.0:
        raise ValueError(f"{context_name} requires h_min > 0.")
    if h_min > h_max:
        raise ValueError(f"{context_name} requires h_min <= h_max.")

    j_depth = int(params["J"])
    if j_depth < 1:
        raise ValueError(f"{context_name} requires J >= 1.")

    schedule_decay_rate = float(params.get("schedule_decay_rate", 5.0))
    if schedule_decay_rate <= 0.0:
        raise ValueError(f"{context_name} requires schedule_decay_rate > 0.")

    if total_samples % j_depth != 0:
        raise ValueError(
            f"{context_name} requires the physical sample budget to be divisible by J; "
            f"received total_samples={total_samples}, J={params['J']}."
        )


def _validate_dd_geo_waterfall_isp_dyna_params(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
) -> None:
    missing = sorted(key for key in _DD_GEO_WATERFALL_ISP_DYNA_REQUIRED_PARAMS if key not in params)
    if missing:
        raise ValueError(f"{context_name} requires explicit parameters: {missing}")

    forbidden = sorted(key for key in _DD_GEO_WATERFALL_ISP_DYNA_FORBIDDEN_PARAMS if key in params)
    if forbidden:
        raise ValueError(f"{context_name} must not define legacy or ambiguous parameters: {forbidden}")

    if str(params.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
        raise ValueError(f"{context_name} currently supports only sampling_distribution='uniform_sa'.")

    normalize_lr_type_name(params.get("lr_type", "constant_delay"))

    h_init = float(params["h_init"])
    if h_init <= 0.0:
        raise ValueError(f"{context_name} requires h_init > 0.")

    j_max = int(params["J_max"])
    if j_max < 1:
        raise ValueError(f"{context_name} requires J_max >= 1.")

    batch_size = int(params["batch_size"])
    if batch_size < 1:
        raise ValueError(f"{context_name} requires batch_size >= 1.")

    decay_factor = float(params.get("decay_factor", 0.5))
    if not (0.0 < decay_factor < 1.0):
        raise ValueError(f"{context_name} requires 0 < decay_factor < 1.")

    if total_samples % batch_size != 0:
        raise ValueError(
            f"{context_name} requires the physical sample budget to be divisible by batch_size; "
            f"received total_samples={total_samples}, batch_size={params['batch_size']}."
        )


def _validate_elastic_dd_geo_waterfall_isp_dyna_params(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
) -> None:
    missing = sorted(key for key in _ELASTIC_DD_GEO_WATERFALL_ISP_DYNA_REQUIRED_PARAMS if key not in params)
    if missing:
        raise ValueError(f"{context_name} requires explicit parameters: {missing}")

    forbidden = sorted(key for key in _ELASTIC_DD_GEO_WATERFALL_ISP_DYNA_FORBIDDEN_PARAMS if key in params)
    if forbidden:
        raise ValueError(f"{context_name} must not define legacy or ambiguous parameters: {forbidden}")

    if str(params.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
        raise ValueError(f"{context_name} currently supports only sampling_distribution='uniform_sa'.")

    normalize_lr_type_name(params.get("lr_type", "constant_delay"))

    h_init = float(params["h_init"])
    if h_init <= 0.0:
        raise ValueError(f"{context_name} requires h_init > 0.")

    j_max = int(params["J_max"])
    if j_max < 2:
        raise ValueError(f"{context_name} requires J_max >= 2.")

    batch_size = int(params["batch_size"])
    if batch_size < 1:
        raise ValueError(f"{context_name} requires batch_size >= 1.")

    decay_factor = float(params.get("decay_factor", 0.5))
    if not (0.0 < decay_factor < 1.0):
        raise ValueError(f"{context_name} requires 0 < decay_factor < 1.")

    if total_samples % batch_size != 0:
        raise ValueError(
            f"{context_name} requires the physical sample budget to be divisible by batch_size; "
            f"received total_samples={total_samples}, batch_size={params['batch_size']}."
        )


def _validate_nr1_waterfall_rm_isp_dyna_params_common(
    params: Mapping[str, Any],
    *,
    required: frozenset[str],
    forbidden: frozenset[str],
    context_name: str,
) -> None:
    missing = sorted(key for key in required if key not in params)
    if missing:
        raise ValueError(f"{context_name} requires explicit parameters: {missing}")
    banned = sorted(key for key in forbidden if key in params)
    if banned:
        raise ValueError(f"{context_name} must not define legacy or ambiguous parameters: {banned}")

    if str(params.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
        raise ValueError(f"{context_name} currently supports only sampling_distribution='uniform_sa'.")

    normalize_lr_type_name(params.get("lr_type", "constant_delay"))

    if "defect_init_mode" in params:
        raise ValueError(
            f"{context_name} no longer supports the 'defect_init_mode' field; the "
            "defect is always zero-initialized (no oracle reward prior). "
            "Remove the field from your config."
        )

    j_depth = int(params["J"])
    if j_depth < 1:
        raise ValueError(f"{context_name} requires J >= 1.")
    h_init = float(params["h_init"])
    h_min = float(params["h_min"])
    if h_init <= 0.0:
        raise ValueError(f"{context_name} requires h_init > 0.")
    if h_min <= 0.0:
        raise ValueError(f"{context_name} requires h_min > 0.")
    if h_min > h_init:
        raise ValueError(f"{context_name} requires h_min <= h_init.")


def _validate_nr1_staircase_waterfall_rm_isp_dyna_params(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
) -> None:
    del total_samples
    _validate_nr1_waterfall_rm_isp_dyna_params_common(
        params,
        required=_NR1_STAIRCASE_WATERFALL_RM_ISP_DYNA_REQUIRED_PARAMS,
        forbidden=_NR1_STAIRCASE_WATERFALL_RM_ISP_DYNA_FORBIDDEN_PARAMS,
        context_name=context_name,
    )
    stall_patience = int(params["stall_patience"])
    if stall_patience < 1:
        raise ValueError(f"{context_name} requires stall_patience >= 1.")
    decay_factor = float(params["decay_factor"])
    if not (0.0 < decay_factor < 1.0):
        raise ValueError(f"{context_name} requires 0 < decay_factor < 1.")
    divergence_mult = float(params.get("divergence_mult", 2.5))
    if divergence_mult <= 0.0:
        raise ValueError(f"{context_name} requires divergence_mult > 0.")


def _validate_nr1_geo_waterfall_rm_isp_dyna_params(
    params: Mapping[str, Any],
    *,
    total_samples: int,
    context_name: str,
) -> None:
    del total_samples
    _validate_nr1_waterfall_rm_isp_dyna_params_common(
        params,
        required=_NR1_GEO_WATERFALL_RM_ISP_DYNA_REQUIRED_PARAMS,
        forbidden=_NR1_GEO_WATERFALL_RM_ISP_DYNA_FORBIDDEN_PARAMS,
        context_name=context_name,
    )
    alpha = float(params["alpha"])
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"{context_name} requires 0 < alpha <= 1.")


def _validate_algorithm_runtime_contracts(
    algorithms: Mapping[str, Mapping[str, Any]],
    *,
    total_samples: int,
) -> None:
    for name, raw_params in algorithms.items():
        params = dict(raw_params or {})
        if is_isp_dyna_algorithm(name):
            _validate_isp_dyna_params(params, total_samples=total_samples, context_name="isp_dyna")
        if is_alt_isp_dyna_algorithm(name):
            _validate_alt_isp_dyna_params(
                params,
                total_samples=total_samples,
                context_name="alt_isp_dyna",
            )
        if is_gen_alt_isp_dyna_algorithm(name):
            _validate_gen_alt_isp_dyna_params(
                params,
                total_samples=total_samples,
                context_name="gen_alt_isp_dyna",
            )
        if is_arch_isp_dyna_algorithm(name):
            _validate_arch_isp_dyna_params(
                params,
                total_samples=total_samples,
                context_name="arch_isp_dyna",
            )
        if is_nr_rm_isp_dyna_algorithm(name):
            _validate_nr_rm_isp_dyna_params(
                params,
                total_samples=total_samples,
                context_name="nr_rm_isp_dyna",
            )
        if is_nr1_rm_isp_dyna_algorithm(name):
            _validate_nr1_rm_isp_dyna_params(
                params,
                total_samples=total_samples,
                context_name="nr1_rm_isp_dyna",
            )
        if is_scheduled_waterfall_isp_dyna_algorithm(name):
            _validate_scheduled_waterfall_isp_dyna_params(
                params,
                total_samples=total_samples,
                context_name="scheduled_waterfall_isp_dyna",
            )
        if is_arch_waterfall_isp_dyna_algorithm(name):
            _validate_arch_waterfall_isp_dyna_params(
                params,
                total_samples=total_samples,
                context_name="arch_waterfall_isp_dyna",
            )
        if is_dd_geo_waterfall_isp_dyna_algorithm(name):
            _validate_dd_geo_waterfall_isp_dyna_params(
                params,
                total_samples=total_samples,
                context_name="dd_geo_waterfall_isp_dyna",
            )
        if is_elastic_dd_geo_waterfall_isp_dyna_algorithm(name):
            _validate_elastic_dd_geo_waterfall_isp_dyna_params(
                params,
                total_samples=total_samples,
                context_name="elastic_dd_geo_waterfall_isp_dyna",
            )
        if is_nr1_staircase_waterfall_rm_isp_dyna_algorithm(name):
            _validate_nr1_staircase_waterfall_rm_isp_dyna_params(
                params,
                total_samples=total_samples,
                context_name="nr1_staircase_waterfall_rm_isp_dyna",
            )
        if is_nr1_geo_waterfall_rm_isp_dyna_algorithm(name):
            _validate_nr1_geo_waterfall_rm_isp_dyna_params(
                params,
                total_samples=total_samples,
                context_name="nr1_geo_waterfall_rm_isp_dyna",
            )
        if is_geo_waterfall_pe_algorithm(name):
            _validate_geo_waterfall_params(
                params,
                context_name="geo_waterfall_besp",
                allowed_keys=_GEO_WATERFALL_PE_ALLOWED_PARAMS,
            )
        if is_geo_waterfall_control_algorithm(name):
            _validate_geo_waterfall_params(
                params,
                context_name="geo_waterfall",
                allowed_keys=_GEO_WATERFALL_CONTROL_ALLOWED_PARAMS,
            )
        if is_dd_waterfall_pe_algorithm(name):
            _validate_dd_waterfall_params(
                params,
                context_name="dd_waterfall_besp",
                allowed_keys=_DD_WATERFALL_PE_ALLOWED_PARAMS,
            )
        if is_dd_waterfall_control_algorithm(name):
            _validate_dd_waterfall_params(
                params,
                context_name="dd_waterfall",
                allowed_keys=_DD_WATERFALL_CONTROL_ALLOWED_PARAMS,
            )


def _validate_geo_waterfall_params(
    params: Mapping[str, Any],
    *,
    context_name: str,
    allowed_keys: frozenset[str],
) -> None:
    forbidden = sorted(key for key in _GEO_WATERFALL_STAIRCASE_PARAMS if key in params)
    if forbidden:
        raise ValueError(
            f"{context_name} must not define staircase Waterfall parameters: {forbidden}."
        )

    unknown = sorted(key for key in params if key not in allowed_keys)
    if unknown:
        raise ValueError(
            f"{context_name} accepts only parameters {sorted(allowed_keys)}; received unsupported keys: {unknown}."
        )

    if "h_init" in params and float(params["h_init"]) <= 0.0:
        raise ValueError(f"{context_name} requires h_init > 0.")

    if "alpha" in params:
        alpha = float(params["alpha"])
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"{context_name} requires 0 < alpha < 1.")


def _validate_dd_waterfall_params(
    params: Mapping[str, Any],
    *,
    context_name: str,
    allowed_keys: frozenset[str],
) -> None:
    forbidden = sorted(key for key in _DD_WATERFALL_FORBIDDEN_PARAMS if key in params)
    if forbidden:
        raise ValueError(
            f"{context_name} must not define unsupported Waterfall parameters: {forbidden}."
        )

    unknown = sorted(key for key in params if key not in allowed_keys)
    if unknown:
        raise ValueError(
            f"{context_name} accepts only parameters {sorted(allowed_keys)}; received unsupported keys: {unknown}."
        )

    if "h_init" in params and float(params["h_init"]) <= 0.0:
        raise ValueError(f"{context_name} requires h_init > 0.")

    if "m_max" in params and int(params["m_max"]) < 1:
        raise ValueError(f"{context_name} requires m_max >= 1.")

    if "decay_factor" in params:
        decay_factor = float(params["decay_factor"])
        if not (0.0 < decay_factor < 1.0):
            raise ValueError(f"{context_name} requires 0 < decay_factor < 1.")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(val) for key, val in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _normalize_env_params(env_params: Any, *, strict_maintained: bool) -> Mapping[str, Any]:
    if env_params is None:
        env_params = {}
    if not isinstance(env_params, Mapping):
        raise ValueError("Experiment configs require env_params to be a mapping.")

    normalized: dict[str, Any] = {}
    for raw_key, value in env_params.items():
        key = str(raw_key)
        if key == "n_s":
            raise ValueError(
                "Experiment configs must use canonical env_params keys; "
                "use 'n_states' instead of 'n_s'."
            )
        if key == "n_a":
            raise ValueError(
                "Experiment configs must use canonical env_params keys; "
                "use 'n_actions' instead of 'n_a'."
            )

        canonical = key
        if strict_maintained and canonical not in _CANONICAL_ENV_PARAM_KEYS:
            raise ValueError(
                f"Unsupported maintained env_params key '{canonical}'. "
                f"Allowed keys: {sorted(_CANONICAL_ENV_PARAM_KEYS)}"
            )

        frozen_value = _deep_freeze(value)
        if canonical in normalized and normalized[canonical] != frozen_value:
            raise ValueError(f"Conflicting values supplied for env_params key '{canonical}'.")
        normalized[canonical] = frozen_value

    return MappingProxyType(normalized)


def _validate_maintained_algorithms(algorithms_arg: Any) -> None:
    if isinstance(algorithms_arg, Mapping):
        for name, params in algorithms_arg.items():
            if not isinstance(name, str):
                raise ValueError(
                    "Maintained experiment configs require algorithm names to be strings."
                )
            if params is not None and not isinstance(params, Mapping):
                raise ValueError(
                    f"Maintained experiment configs require algorithm parameters for '{name}' to be a mapping."
                )
        return

    if isinstance(algorithms_arg, list):
        for index, entry in enumerate(algorithms_arg):
            if isinstance(entry, str):
                continue
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"Maintained experiment configs require algorithms[{index}] to be a string or mapping."
                )
            if "name" not in entry or not isinstance(entry["name"], str):
                raise ValueError(
                    f"Maintained experiment configs require algorithms[{index}] to include a string 'name'."
                )
        return

    raise ValueError("Maintained experiment configs require algorithms to be a list or mapping.")


def _validate_maintained_experiment_mapping(mapping: Mapping[str, Any]) -> None:
    unknown_keys = sorted(
        str(key) for key in mapping if str(key) not in _MAINTAINED_EXPERIMENT_KEYS
    )
    if unknown_keys:
        raise ValueError(
            "Maintained experiment configs use a canonical top-level schema; "
            f"unsupported keys: {unknown_keys}"
        )

    if "name" not in mapping or "env" not in mapping:
        raise ValueError("Maintained experiment configs require explicit 'name' and 'env' keys.")

    if "env_params" not in mapping:
        raise ValueError("Maintained experiment configs require an explicit 'env_params' mapping.")
    _normalize_env_params(mapping.get("env_params"), strict_maintained=True)

    if "n_seeds" not in mapping:
        raise ValueError("Maintained experiment configs require an explicit 'n_seeds' key.")

    if "lambdas" not in mapping:
        raise ValueError("Maintained experiment configs require an explicit 'lambdas' key.")

    if "max_queries" not in mapping:
        raise ValueError("Maintained experiment configs require an explicit 'max_queries' key.")

    if "threshold" not in mapping:
        raise ValueError("Maintained experiment configs require an explicit 'threshold' key.")

    if "algorithms" not in mapping:
        raise ValueError("Maintained experiment configs require an explicit 'algorithms' key.")
    _validate_maintained_algorithms(mapping["algorithms"])


def parse_lambda_values(raw_lambdas: Any) -> tuple[float, ...]:
    if isinstance(raw_lambdas, str):
        start, end, step = map(float, raw_lambdas.split(":"))
        values: list[float] = []
        current = start
        while current <= end + 1e-9:
            values.append(float(round(current, 12)))
            current += step
        return tuple(values)
    if raw_lambdas is None:
        return (0.0, 0.5, 1.0)
    return tuple(float(value) for value in raw_lambdas)


def normalize_algorithm_config(algorithms_arg: Any) -> Mapping[str, Mapping[str, Any]]:
    if isinstance(algorithms_arg, Mapping):
        normalized = {
            str(name): _deep_freeze(dict(params or {}))
            for name, params in algorithms_arg.items()
            if isinstance(name, str)
        }
        return MappingProxyType(normalized)

    normalized: dict[str, Mapping[str, Any]] = {}
    for entry in algorithms_arg or ("vi", "osvi", "besp"):
        if isinstance(entry, str):
            normalized[entry] = MappingProxyType({})
            continue
        if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
            params = {
                str(key): _deep_freeze(value) for key, value in entry.items() if str(key) != "name"
            }
            normalized[str(entry["name"])] = MappingProxyType(params)
    return MappingProxyType(normalized)


def extract_algorithm_names(algorithms_arg: Any) -> set[str]:
    if isinstance(algorithms_arg, Mapping):
        return {str(name) for name in algorithms_arg if isinstance(name, str)}
    names: set[str] = set()
    for entry in algorithms_arg or ():
        if isinstance(entry, str):
            names.add(entry)
        elif isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
            names.add(str(entry["name"]))
    return names


@dataclass(frozen=True)
class RunnerOverrides:
    platform: str = "auto"
    backend: str = "jax"
    cores: int = 1
    env: str | None = None
    output_dir: Path | None = None

    @classmethod
    def from_cli(
        cls,
        *,
        platform: str | None,
        backend: str,
        cores: int,
        env: str | None,
        output_dir: str | None,
    ) -> "RunnerOverrides":
        resolved_platform = "auto" if platform is None else str(platform)
        if resolved_platform not in {"auto", "cpu", "gpu"}:
            raise ValueError("Runner platform must be one of: auto, cpu, gpu.")
        return cls(
            platform=resolved_platform,
            backend=backend,
            cores=max(int(cores), 1),
            env=env,
            output_dir=Path(output_dir) if output_dir else None,
        )

    def resolved_platform(self, *, uses_gpu_algorithms: bool) -> str:
        if self.platform != "auto":
            return self.platform
        return "gpu" if uses_gpu_algorithms else "cpu"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    env: str
    env_params: Mapping[str, Any]
    n_seeds: int
    seed: int
    lambdas: tuple[float, ...]
    max_queries: int
    threshold: float
    error_mode: str
    problem_type: str
    model_norm: str
    value_norm: str
    value_normalized: bool
    emit_pe_traces: bool
    emit_control_traces: bool
    metric_mode: str
    evaluation_pipeline: str
    deterministic_target_manifest: str | None
    env_variants_manifest: str | None
    algorithms: Mapping[str, Mapping[str, Any]]
    raw_mapping: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        strict_maintained: bool = False,
    ) -> "ExperimentConfig":
        raw = dict(mapping)
        if strict_maintained:
            _validate_maintained_experiment_mapping(raw)
        env_params = _normalize_env_params(
            raw.get("env_params", {}), strict_maintained=strict_maintained
        )
        algorithms = normalize_algorithm_config(raw.get("algorithms", ("vi", "osvi", "besp")))
        max_queries = max(int(raw.get("max_queries", 5000)), 1)
        _validate_algorithm_runtime_contracts(algorithms, total_samples=max_queries)
        emit_pe_traces = bool(raw.get("emit_pe_traces", False))
        emit_control_traces = bool(raw.get("emit_control_traces", False))
        problem_type = str(raw.get("problem_type", "control"))
        metric_mode = str(
            raw.get("metric_mode", default_metric_mode_for_problem_type(problem_type))
        )
        raw.setdefault("metric_mode", metric_mode)
        evaluation_pipeline = str(
            raw.get(
                "evaluation_pipeline",
                default_evaluation_pipeline_for_problem_type(problem_type),
            )
        )
        raw.setdefault("evaluation_pipeline", evaluation_pipeline)
        deterministic_target_manifest = raw.get("deterministic_target_manifest")
        if deterministic_target_manifest is not None:
            deterministic_target_manifest = str(deterministic_target_manifest)
        env_variants_manifest = raw.get("env_variants_manifest")
        if env_variants_manifest is not None:
            env_variants_manifest = str(env_variants_manifest)
        if emit_pe_traces and problem_type != "pe":
            raise ValueError("emit_pe_traces is supported only when problem_type='pe'.")
        if emit_control_traces and problem_type != "control":
            raise ValueError("emit_control_traces is supported only when problem_type='control'.")
        if metric_mode not in {"residual_hit_time", "policy_value_error"}:
            raise ValueError(
                "metric_mode must be one of: residual_hit_time, policy_value_error."
            )
        if metric_mode != "residual_hit_time" and problem_type != "control":
            raise ValueError(
                "metric_mode is supported only when problem_type='control' unless left at the default residual_hit_time."
            )
        resolve_evaluation_pipeline(
            problem_type=problem_type,
            metric_mode=metric_mode,
            requested_pipeline=evaluation_pipeline,
        )
        return cls(
            name=str(raw.get("name", "experiment")),
            env=str(raw.get("env", "cliffwalk")),
            env_params=env_params,
            n_seeds=max(int(raw.get("n_seeds", 1)), 1),
            seed=int(raw.get("seed", 42)),
            lambdas=parse_lambda_values(raw.get("lambdas", (0.0, 0.5, 1.0))),
            max_queries=max_queries,
            threshold=float(raw.get("threshold", 1e-3)),
            error_mode=str(raw.get("error_mode", "support_uniform")),
            problem_type=problem_type,
            model_norm=str(raw.get("model_norm", "L1")),
            value_norm=str(raw.get("value_norm", "L1")),
            value_normalized=bool(raw.get("value_normalized", True)),
            emit_pe_traces=emit_pe_traces,
            emit_control_traces=emit_control_traces,
            metric_mode=metric_mode,
            evaluation_pipeline=evaluation_pipeline,
            deterministic_target_manifest=deterministic_target_manifest,
            env_variants_manifest=env_variants_manifest,
            algorithms=algorithms,
            raw_mapping=_deep_freeze(raw),
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        strict_maintained: bool = False,
    ) -> "ExperimentConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(f"Experiment config at {path} did not decode to a mapping.")
        return cls.from_mapping(data, strict_maintained=strict_maintained)

    @classmethod
    def validate_maintained_mapping(cls, mapping: Mapping[str, Any]) -> None:
        cls.from_mapping(mapping, strict_maintained=True)

    @classmethod
    def validate_maintained_yaml(cls, path: str | Path) -> None:
        cls.from_yaml(path, strict_maintained=True)

    def to_mapping(self) -> dict[str, Any]:
        mapping = _deep_thaw(self.raw_mapping)
        mapping["env_params"] = _deep_thaw(self.env_params)
        return mapping

    @property
    def algorithm_names(self) -> tuple[str, ...]:
        return tuple(self.algorithms.keys())

    @property
    def uses_gpu_algorithms(self) -> bool:
        return bool({canonical_algorithm_name(name) for name in self.algorithm_names} & GPU_ONLY_ALGORITHMS)

    def resolved_env(self, overrides: RunnerOverrides) -> str:
        return overrides.env or self.env

    def resolved_evaluation_pipeline(self) -> str:
        return resolve_evaluation_pipeline(
            problem_type=self.problem_type,
            metric_mode=self.metric_mode,
            requested_pipeline=self.evaluation_pipeline,
        )

    def build_gpu_task(
        self,
        *,
        lam: float,
        backend: str,
        env_name: str,
        offline_eval_root: Path | None = None,
    ) -> GpuBatchTask:
        return GpuBatchTask(
            env_name=env_name,
            env_params=_deep_thaw(self.env_params),
            lambda_value=lam,
            seed=self.seed + int(lam * 1000),
            n_seeds=self.n_seeds,
            max_queries=self.max_queries,
            threshold=self.threshold,
            algorithms=_deep_thaw(self.algorithms),
            problem_type=self.problem_type,
            backend=backend,
            evaluation_pipeline=self.resolved_evaluation_pipeline(),
            offline_eval_root=str(offline_eval_root) if offline_eval_root is not None else None,
        )

    def build_cpu_tasks(
        self,
        *,
        env_name: str,
        offline_eval_root: Path | None = None,
    ) -> list[CpuExperimentTask]:
        tasks: list[CpuExperimentTask] = []
        if self.env_variants_manifest:
            variants = load_env_variants_manifest(
                self.env_variants_manifest,
                default_env_name=env_name,
                base_env_params=_deep_thaw(self.env_params),
                default_target_manifest=self.deterministic_target_manifest,
            )
            for lam in self.lambdas:
                for seed_index in range(self.n_seeds):
                    base_seed = self.seed + int(lam * 1000)
                    for variant_index, variant in enumerate(variants):
                        task_metadata = dict(variant.metadata)
                        task_metadata.setdefault("variant_id", variant.variant_id)
                        task_metadata.setdefault("env_name", variant.env_name)
                        task_metadata.setdefault("env_params", dict(variant.env_params))
                        if variant.deterministic_target_manifest is not None:
                            task_metadata.setdefault(
                                "deterministic_target_manifest",
                                variant.deterministic_target_manifest,
                            )
                        try:
                            variant_seed_offset = int(task_metadata.get("source_seed", variant_index))
                        except (TypeError, ValueError):
                            variant_seed_offset = variant_index
                        seed = base_seed + variant_seed_offset * self.n_seeds + seed_index
                        tasks.append(
                            CpuExperimentTask(
                                env_name=variant.env_name,
                                env_params=dict(variant.env_params),
                                lambda_value=lam,
                                seed=seed,
                                max_queries=self.max_queries,
                                threshold=self.threshold,
                                algorithms=_deep_thaw(self.algorithms),
                                error_mode=self.error_mode,
                                problem_type=self.problem_type,
                                model_norm=self.model_norm,
                                value_norm=self.value_norm,
                                value_normalized=self.value_normalized,
                                emit_pe_traces=self.emit_pe_traces,
                                emit_control_traces=self.emit_control_traces,
                                metric_mode=self.metric_mode,
                                evaluation_pipeline=self.resolved_evaluation_pipeline(),
                                deterministic_target_manifest=variant.deterministic_target_manifest,
                                task_metadata=task_metadata,
                                offline_eval_root=(
                                    str(offline_eval_root) if offline_eval_root is not None else None
                                ),
                            )
                        )
            return tasks

        for lam in self.lambdas:
            for seed_index in range(self.n_seeds):
                seed = self.seed + int(lam * 1000) + seed_index
                tasks.append(
                    CpuExperimentTask(
                        env_name=env_name,
                        env_params=_deep_thaw(self.env_params),
                        lambda_value=lam,
                        seed=seed,
                        max_queries=self.max_queries,
                        threshold=self.threshold,
                        algorithms=_deep_thaw(self.algorithms),
                        error_mode=self.error_mode,
                        problem_type=self.problem_type,
                        model_norm=self.model_norm,
                        value_norm=self.value_norm,
                        value_normalized=self.value_normalized,
                        emit_pe_traces=self.emit_pe_traces,
                        emit_control_traces=self.emit_control_traces,
                        metric_mode=self.metric_mode,
                        evaluation_pipeline=self.resolved_evaluation_pipeline(),
                        deterministic_target_manifest=self.deterministic_target_manifest,
                        offline_eval_root=(
                            str(offline_eval_root) if offline_eval_root is not None else None
                        ),
                    )
                )
        return tasks


@dataclass(frozen=True)
class ProviderRoutingConfig:
    provider_order: tuple[str, ...]
    auto_provider_order: tuple[str, ...]
    pending_sla_seconds: int
    providers: Mapping[str, Mapping[str, Any]]
    workflows: Mapping[str, Mapping[str, Any]]
    raw_mapping: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ProviderRoutingConfig":
        raw = dict(mapping)
        provider_order = tuple(str(value) for value in raw.get("provider_order", ()))
        auto_provider_order = tuple(
            str(value)
            for value in raw.get("auto_provider_order", raw.get("provider_order", ()))
        )
        return cls(
            provider_order=provider_order,
            auto_provider_order=auto_provider_order,
            pending_sla_seconds=int(raw.get("pending_sla_seconds", 180)),
            providers=_deep_freeze(dict(raw.get("providers", {}))),
            workflows=_deep_freeze(dict(raw.get("workflows", {}))),
            raw_mapping=_deep_freeze(raw),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "ProviderRoutingConfig":
        import json

        raw_text = Path(path).read_text(encoding="utf-8")
        try:
            import yaml

            decoded = yaml.safe_load(raw_text)
        except Exception:
            decoded = json.loads(raw_text)

        if not isinstance(decoded, Mapping):
            raise ValueError(f"Provider routing config at {path} did not decode to a mapping.")
        return cls.from_mapping(decoded)

    def to_mapping(self) -> dict[str, Any]:
        return _deep_thaw(self.raw_mapping)
