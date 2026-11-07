"""Config normalization for maintained hyper-batch runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ...core.config import (
    is_alt_isp_dyna_algorithm,
    is_gen_alt_isp_dyna_algorithm,
    is_arch_isp_dyna_algorithm,
    is_nr_rm_isp_dyna_algorithm,
    is_nr1_rm_isp_dyna_algorithm,
    is_nr1_staircase_waterfall_rm_isp_dyna_algorithm,
    is_nr1_geo_waterfall_rm_isp_dyna_algorithm,
    is_arch_waterfall_isp_dyna_algorithm,
    is_dd_geo_waterfall_isp_dyna_algorithm,
    is_elastic_dd_geo_waterfall_isp_dyna_algorithm,
    is_isp_dyna_algorithm,
    is_scheduled_waterfall_isp_dyna_algorithm,
    normalize_lr_type_name,
    normalize_nr_h_schedule_shape,
    validate_nr_h_schedule_finish_frac,
)
from ...domains.jax_rl.error_modes import canonical_jax_rl_stochastic_error_mode

@dataclass(frozen=True)
class HyperBatchConfig:
    label: str
    algorithm: str
    gamma: float
    start_lr: float = 0.5
    delay: float = 0.0
    lambda_value: float = 0.0
    h: float | None = None
    h_target: float | None = None
    h_max: float | None = None
    h_min: float | None = None
    J: int = 5
    h_init: float | None = None
    h_terminal: float | None = None
    h_floor: float | None = None
    h_schedule_shape: str = "exponential"
    h_schedule_finish_frac: float = 1.0
    J_max: int = 3
    planning_steps: int = -1
    batch_size: int | None = None
    lr_type: str = "constant_delay"
    sampling_distribution: str = "uniform_sa"
    decay_factor: float = 0.5
    schedule_decay_rate: float = 5.0
    epsilon: float = 0.1
    error_mode: str = "support_uniform"
    # Waterfall-NR1 adaptive-h fields.
    stall_patience: int | None = None
    alpha: float | None = None
    divergence_mult: float = 2.5
    v_max: float = 0.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "HyperBatchConfig":
        label = str(mapping["label"])
        if mapping.get("algorithm") is None:
            raise ValueError(
                "Maintained hyper-batch configs require an explicit 'algorithm' field."
            )
        algorithm = str(mapping["algorithm"])
        if is_alt_isp_dyna_algorithm(algorithm):
            missing = sorted(key for key in ("h_target", "J", "epsilon") if key not in mapping)
            if missing:
                raise ValueError(f"Alt-ISP-Dyna requires explicit parameters: {missing}")
            forbidden = sorted(
                key
                for key in (
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
                )
                if key in mapping
            )
            if forbidden:
                raise ValueError(
                    "Alt-ISP-Dyna is trajectory-only and must not define unsupported parameters: "
                    f"{forbidden}"
                )
            if float(mapping["h_target"]) <= 0.0:
                raise ValueError("Alt-ISP-Dyna requires h_target > 0.")
            if int(mapping["J"]) < 1:
                raise ValueError("Alt-ISP-Dyna requires J >= 1.")
            if not (0.0 <= float(mapping["epsilon"]) <= 1.0):
                raise ValueError("Alt-ISP-Dyna requires 0 <= epsilon <= 1.")
        if is_gen_alt_isp_dyna_algorithm(algorithm):
            missing = sorted(key for key in ("h_target", "J") if key not in mapping)
            if missing:
                raise ValueError(f"Gen-Alt-ISP-Dyna requires explicit parameters: {missing}")
            forbidden = sorted(
                key
                for key in (
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
                )
                if key in mapping
            )
            if forbidden:
                raise ValueError(
                    "Gen-Alt-ISP-Dyna is generative-only and must not define unsupported parameters: "
                    f"{forbidden}"
                )
            if float(mapping["h_target"]) <= 0.0:
                raise ValueError("Gen-Alt-ISP-Dyna requires h_target > 0.")
            if int(mapping["J"]) < 1:
                raise ValueError("Gen-Alt-ISP-Dyna requires J >= 1.")
        if is_isp_dyna_algorithm(algorithm):
            missing = sorted(
                key for key in ("h", "J", "batch_size", "start_lr", "delay") if key not in mapping
            )
            if missing:
                raise ValueError(f"ISP-Dyna requires explicit parameters: {missing}")
            forbidden = sorted(
                key for key in ("planning_steps", "use_online_model") if key in mapping
            )
            if forbidden:
                raise ValueError(f"ISP-Dyna must not define legacy Dyna parameters: {forbidden}")
            if str(mapping.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
                raise ValueError("ISP-Dyna currently supports only sampling_distribution='uniform_sa'.")
        if is_arch_isp_dyna_algorithm(algorithm):
            missing = sorted(
                key for key in ("h", "J", "start_lr", "delay") if key not in mapping
            )
            if missing:
                raise ValueError(f"Arch-ISP-Dyna requires explicit parameters: {missing}")
            forbidden = sorted(
                key
                for key in ("batch_size", "planning_steps", "use_online_model")
                if key in mapping
            )
            if forbidden:
                raise ValueError(
                    f"Arch-ISP-Dyna must not define legacy or ambiguous parameters: {forbidden}"
                )
            if str(mapping.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
                raise ValueError(
                    "Arch-ISP-Dyna currently supports only sampling_distribution='uniform_sa'."
                )
            if int(mapping["J"]) < 1:
                raise ValueError("Arch-ISP-Dyna requires J >= 1.")
        if is_nr_rm_isp_dyna_algorithm(algorithm) or is_nr1_rm_isp_dyna_algorithm(algorithm):
            missing = sorted(key for key in ("start_lr", "delay") if key not in mapping)
            if missing:
                raise ValueError(f"NR-RM-ISP-Dyna requires explicit parameters: {missing}")
            forbidden = sorted(
                key
                for key in (
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
                )
                if key in mapping
            )
            if forbidden:
                raise ValueError(
                    f"NR-RM-ISP-Dyna must not define legacy or ambiguous parameters: {forbidden}"
                )
            if str(mapping.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
                raise ValueError(
                    "NR-RM-ISP-Dyna currently supports only sampling_distribution='uniform_sa'."
                )
            if "defect_init_mode" in mapping:
                raise ValueError(
                    "NR-RM-ISP-Dyna no longer supports the 'defect_init_mode' field; the "
                    "defect is always zero-initialized (no oracle reward prior). "
                    "Remove the field from your config."
                )
            has_fixed_h = "h" in mapping
            has_scheduled_h = any(key in mapping for key in ("h_init", "h_terminal", "h_floor"))
            if has_fixed_h and has_scheduled_h:
                raise ValueError(
                    "NR-RM-ISP-Dyna must define exactly one mode: limit, fixed_h, or scheduled_h."
                )
            if has_fixed_h:
                missing_fixed = sorted(key for key in ("h", "J") if key not in mapping)
                if missing_fixed:
                    raise ValueError(
                        f"NR-RM-ISP-Dyna fixed_h mode requires explicit parameters: {missing_fixed}"
                    )
                if int(mapping["J"]) < 1:
                    raise ValueError("NR-RM-ISP-Dyna fixed_h mode requires J >= 1.")
                if float(mapping["h"]) <= 0.0:
                    raise ValueError("NR-RM-ISP-Dyna fixed_h mode requires h > 0.")
                if "h_schedule_shape" in mapping or "h_schedule_finish_frac" in mapping:
                    raise ValueError(
                        "NR-RM-ISP-Dyna fixed_h mode must not define schedule-shape fields."
                    )
            elif has_scheduled_h:
                missing_scheduled = sorted(
                    key for key in ("J", "h_init", "h_terminal") if key not in mapping
                )
                if missing_scheduled:
                    raise ValueError(
                        "NR-RM-ISP-Dyna scheduled_h mode requires explicit parameters: "
                        f"{missing_scheduled}"
                    )
                if int(mapping["J"]) < 1:
                    raise ValueError("NR-RM-ISP-Dyna scheduled_h mode requires J >= 1.")
                h_floor = float(mapping.get("h_floor", 1.0))
                h_init = float(mapping["h_init"])
                h_terminal = float(mapping["h_terminal"])
                if h_floor <= 0.0:
                    raise ValueError("NR-RM-ISP-Dyna scheduled_h mode requires h_floor > 0.")
                if h_init <= 0.0:
                    raise ValueError("NR-RM-ISP-Dyna scheduled_h mode requires h_init > 0.")
                if h_terminal <= 0.0:
                    raise ValueError("NR-RM-ISP-Dyna scheduled_h mode requires h_terminal > 0.")
                if h_init < h_terminal:
                    raise ValueError(
                        "NR-RM-ISP-Dyna scheduled_h mode requires h_init >= h_terminal."
                    )
                if h_terminal < h_floor:
                    raise ValueError(
                        "NR-RM-ISP-Dyna scheduled_h mode requires h_terminal >= h_floor."
                    )
                normalize_nr_h_schedule_shape(mapping.get("h_schedule_shape", "exponential"))
                validate_nr_h_schedule_finish_frac(mapping.get("h_schedule_finish_frac", 1.0))
            elif "J" in mapping:
                raise ValueError("NR-RM-ISP-Dyna fixed_h mode requires explicit parameters: ['h']")
            elif "h_schedule_shape" in mapping or "h_schedule_finish_frac" in mapping:
                raise ValueError(
                    "NR-RM-ISP-Dyna limit mode must not define schedule-shape fields."
                )
        if is_scheduled_waterfall_isp_dyna_algorithm(algorithm):
            missing = sorted(
                key for key in ("h_max", "h_min", "J", "start_lr", "delay") if key not in mapping
            )
            if missing:
                raise ValueError(
                    f"Scheduled-Waterfall-ISP-Dyna requires explicit parameters: {missing}"
                )
            forbidden = sorted(
                key
                for key in (
                    "h",
                    "h_init",
                    "J_max",
                    "batch_size",
                    "planning_steps",
                    "use_online_model",
                    "decay_factor",
                )
                if key in mapping
            )
            if forbidden:
                raise ValueError(
                    "Scheduled-Waterfall-ISP-Dyna must not define legacy or ambiguous parameters: "
                    f"{forbidden}"
                )
            if str(mapping.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
                raise ValueError(
                    "Scheduled-Waterfall-ISP-Dyna currently supports only sampling_distribution='uniform_sa'."
                )
            if float(mapping["h_max"]) <= 0.0:
                raise ValueError("Scheduled-Waterfall-ISP-Dyna requires h_max > 0.")
            if float(mapping["h_min"]) <= 0.0:
                raise ValueError("Scheduled-Waterfall-ISP-Dyna requires h_min > 0.")
            if float(mapping["h_min"]) > float(mapping["h_max"]):
                raise ValueError("Scheduled-Waterfall-ISP-Dyna requires h_min <= h_max.")
            if int(mapping["J"]) < 2:
                raise ValueError("Scheduled-Waterfall-ISP-Dyna requires J >= 2.")
            if float(mapping.get("schedule_decay_rate", 5.0)) <= 0.0:
                raise ValueError("Scheduled-Waterfall-ISP-Dyna requires schedule_decay_rate > 0.")
        if is_arch_waterfall_isp_dyna_algorithm(algorithm):
            missing = sorted(
                key for key in ("h_max", "h_min", "J", "start_lr", "delay") if key not in mapping
            )
            if missing:
                raise ValueError(
                    f"Arch-Waterfall-ISP-Dyna requires explicit parameters: {missing}"
                )
            forbidden = sorted(
                key
                for key in (
                    "h",
                    "h_init",
                    "J_max",
                    "batch_size",
                    "planning_steps",
                    "use_online_model",
                    "decay_factor",
                )
                if key in mapping
            )
            if forbidden:
                raise ValueError(
                    "Arch-Waterfall-ISP-Dyna must not define legacy or ambiguous parameters: "
                    f"{forbidden}"
                )
            if str(mapping.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
                raise ValueError(
                    "Arch-Waterfall-ISP-Dyna currently supports only sampling_distribution='uniform_sa'."
                )
            if float(mapping["h_max"]) <= 0.0:
                raise ValueError("Arch-Waterfall-ISP-Dyna requires h_max > 0.")
            if float(mapping["h_min"]) <= 0.0:
                raise ValueError("Arch-Waterfall-ISP-Dyna requires h_min > 0.")
            if float(mapping["h_min"]) > float(mapping["h_max"]):
                raise ValueError("Arch-Waterfall-ISP-Dyna requires h_min <= h_max.")
            if int(mapping["J"]) < 1:
                raise ValueError("Arch-Waterfall-ISP-Dyna requires J >= 1.")
            if float(mapping.get("schedule_decay_rate", 5.0)) <= 0.0:
                raise ValueError("Arch-Waterfall-ISP-Dyna requires schedule_decay_rate > 0.")
        if is_dd_geo_waterfall_isp_dyna_algorithm(algorithm):
            missing = sorted(
                key
                for key in ("h_init", "J_max", "batch_size", "start_lr", "delay")
                if key not in mapping
            )
            if missing:
                raise ValueError(f"DD-Geo-Waterfall-ISP-Dyna requires explicit parameters: {missing}")
            forbidden = sorted(
                key
                for key in (
                    "h",
                    "J",
                    "planning_steps",
                    "use_online_model",
                    "h_min",
                    "stall_patience",
                    "alpha",
                )
                if key in mapping
            )
            if forbidden:
                raise ValueError(
                    f"DD-Geo-Waterfall-ISP-Dyna must not define legacy or ambiguous parameters: {forbidden}"
                )
            if str(mapping.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
                raise ValueError(
                    "DD-Geo-Waterfall-ISP-Dyna currently supports only sampling_distribution='uniform_sa'."
                )
            if float(mapping["h_init"]) <= 0.0:
                raise ValueError("DD-Geo-Waterfall-ISP-Dyna requires h_init > 0.")
            if int(mapping["J_max"]) < 1:
                raise ValueError("DD-Geo-Waterfall-ISP-Dyna requires J_max >= 1.")
            if int(mapping["batch_size"]) < 1:
                raise ValueError("DD-Geo-Waterfall-ISP-Dyna requires batch_size >= 1.")
            decay_factor = float(mapping.get("decay_factor", 0.5))
            if not (0.0 < decay_factor < 1.0):
                raise ValueError("DD-Geo-Waterfall-ISP-Dyna requires 0 < decay_factor < 1.")
        if is_elastic_dd_geo_waterfall_isp_dyna_algorithm(algorithm):
            missing = sorted(
                key
                for key in ("h_init", "J_max", "batch_size", "start_lr", "delay")
                if key not in mapping
            )
            if missing:
                raise ValueError(
                    f"Elastic-DD-Geo-Waterfall-ISP-Dyna requires explicit parameters: {missing}"
                )
            forbidden = sorted(
                key
                for key in (
                    "h",
                    "J",
                    "planning_steps",
                    "use_online_model",
                    "h_min",
                    "stall_patience",
                    "alpha",
                )
                if key in mapping
            )
            if forbidden:
                raise ValueError(
                    "Elastic-DD-Geo-Waterfall-ISP-Dyna must not define legacy or ambiguous parameters: "
                    f"{forbidden}"
                )
            if str(mapping.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
                raise ValueError(
                    "Elastic-DD-Geo-Waterfall-ISP-Dyna currently supports only sampling_distribution='uniform_sa'."
                )
            if float(mapping["h_init"]) <= 0.0:
                raise ValueError("Elastic-DD-Geo-Waterfall-ISP-Dyna requires h_init > 0.")
            if int(mapping["J_max"]) < 2:
                raise ValueError("Elastic-DD-Geo-Waterfall-ISP-Dyna requires J_max >= 2.")
            if int(mapping["batch_size"]) < 1:
                raise ValueError("Elastic-DD-Geo-Waterfall-ISP-Dyna requires batch_size >= 1.")
            decay_factor = float(mapping.get("decay_factor", 0.5))
            if not (0.0 < decay_factor < 1.0):
                raise ValueError("Elastic-DD-Geo-Waterfall-ISP-Dyna requires 0 < decay_factor < 1.")
        if is_nr1_staircase_waterfall_rm_isp_dyna_algorithm(algorithm):
            missing = sorted(
                key
                for key in ("J", "h_init", "h_min", "stall_patience", "decay_factor", "start_lr", "delay")
                if key not in mapping
            )
            if missing:
                raise ValueError(
                    f"NR1-Staircase-Waterfall-RM-ISP-Dyna requires explicit parameters: {missing}"
                )
            forbidden = sorted(
                key
                for key in (
                    "h",
                    "h_target",
                    "h_max",
                    "h_terminal",
                    "h_floor",
                    "h_schedule_shape",
                    "h_schedule_finish_frac",
                    "J_max",
                    "alpha",
                    "batch_size",
                    "planning_steps",
                    "use_online_model",
                    "schedule_decay_rate",
                    "epsilon",
                )
                if key in mapping
            )
            if forbidden:
                raise ValueError(
                    "NR1-Staircase-Waterfall-RM-ISP-Dyna must not define legacy or ambiguous parameters: "
                    f"{forbidden}"
                )
            if str(mapping.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
                raise ValueError(
                    "NR1-Staircase-Waterfall-RM-ISP-Dyna currently supports only sampling_distribution='uniform_sa'."
                )
            if int(mapping["J"]) < 1:
                raise ValueError("NR1-Staircase-Waterfall-RM-ISP-Dyna requires J >= 1.")
            if float(mapping["h_init"]) <= 0.0:
                raise ValueError("NR1-Staircase-Waterfall-RM-ISP-Dyna requires h_init > 0.")
            if float(mapping["h_min"]) <= 0.0:
                raise ValueError("NR1-Staircase-Waterfall-RM-ISP-Dyna requires h_min > 0.")
            if float(mapping["h_min"]) > float(mapping["h_init"]):
                raise ValueError("NR1-Staircase-Waterfall-RM-ISP-Dyna requires h_min <= h_init.")
            if int(mapping["stall_patience"]) < 1:
                raise ValueError("NR1-Staircase-Waterfall-RM-ISP-Dyna requires stall_patience >= 1.")
            decay_factor = float(mapping["decay_factor"])
            if not (0.0 < decay_factor < 1.0):
                raise ValueError("NR1-Staircase-Waterfall-RM-ISP-Dyna requires 0 < decay_factor < 1.")
            if float(mapping.get("divergence_mult", 2.5)) <= 0.0:
                raise ValueError("NR1-Staircase-Waterfall-RM-ISP-Dyna requires divergence_mult > 0.")
        if is_nr1_geo_waterfall_rm_isp_dyna_algorithm(algorithm):
            missing = sorted(
                key
                for key in ("J", "h_init", "h_min", "alpha", "start_lr", "delay")
                if key not in mapping
            )
            if missing:
                raise ValueError(
                    f"NR1-Geo-Waterfall-RM-ISP-Dyna requires explicit parameters: {missing}"
                )
            forbidden = sorted(
                key
                for key in (
                    "h",
                    "h_target",
                    "h_max",
                    "h_terminal",
                    "h_floor",
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
                )
                if key in mapping
            )
            if forbidden:
                raise ValueError(
                    "NR1-Geo-Waterfall-RM-ISP-Dyna must not define legacy or ambiguous parameters: "
                    f"{forbidden}"
                )
            if str(mapping.get("sampling_distribution", "uniform_sa")).lower() != "uniform_sa":
                raise ValueError(
                    "NR1-Geo-Waterfall-RM-ISP-Dyna currently supports only sampling_distribution='uniform_sa'."
                )
            if int(mapping["J"]) < 1:
                raise ValueError("NR1-Geo-Waterfall-RM-ISP-Dyna requires J >= 1.")
            if float(mapping["h_init"]) <= 0.0:
                raise ValueError("NR1-Geo-Waterfall-RM-ISP-Dyna requires h_init > 0.")
            if float(mapping["h_min"]) <= 0.0:
                raise ValueError("NR1-Geo-Waterfall-RM-ISP-Dyna requires h_min > 0.")
            if float(mapping["h_min"]) > float(mapping["h_init"]):
                raise ValueError("NR1-Geo-Waterfall-RM-ISP-Dyna requires h_min <= h_init.")
            alpha = float(mapping["alpha"])
            if not (0.0 < alpha <= 1.0):
                raise ValueError("NR1-Geo-Waterfall-RM-ISP-Dyna requires 0 < alpha <= 1.")
        return cls(
            label=label,
            algorithm=algorithm,
            gamma=float(mapping.get("gamma", 0.99)),
            start_lr=float(mapping.get("start_lr", 0.5)),
            delay=float(mapping.get("delay", 0.0)),
            lambda_value=float(mapping.get("lambda", 0.0)),
            h=float(mapping["h"]) if mapping.get("h") is not None else None,
            h_target=float(mapping["h_target"]) if mapping.get("h_target") is not None else None,
            h_max=float(mapping["h_max"]) if mapping.get("h_max") is not None else None,
            h_min=float(mapping["h_min"]) if mapping.get("h_min") is not None else None,
            J=int(mapping.get("J", 5)),
            h_init=float(mapping["h_init"]) if mapping.get("h_init") is not None else None,
            h_terminal=(
                float(mapping["h_terminal"])
                if mapping.get("h_terminal") is not None
                else None
            ),
            h_floor=float(mapping["h_floor"]) if mapping.get("h_floor") is not None else None,
            h_schedule_shape=normalize_nr_h_schedule_shape(mapping.get("h_schedule_shape", "exponential")),
            h_schedule_finish_frac=validate_nr_h_schedule_finish_frac(
                mapping.get("h_schedule_finish_frac", 1.0)
            ),
            J_max=int(mapping.get("J_max", 3)),
            planning_steps=int(mapping.get("planning_steps", -1)),
            batch_size=int(mapping["batch_size"]) if mapping.get("batch_size") is not None else None,
            lr_type=normalize_lr_type_name(mapping.get("lr_type", "constant_delay")),
            sampling_distribution=str(mapping.get("sampling_distribution", "uniform_sa")),
            decay_factor=float(mapping.get("decay_factor", 0.5)),
            schedule_decay_rate=float(mapping.get("schedule_decay_rate", 5.0)),
            epsilon=float(mapping.get("epsilon", 0.1)),
            error_mode=canonical_jax_rl_stochastic_error_mode(mapping.get("error_mode")),
            stall_patience=int(mapping["stall_patience"]) if mapping.get("stall_patience") is not None else None,
            alpha=float(mapping["alpha"]) if mapping.get("alpha") is not None else None,
            divergence_mult=float(mapping.get("divergence_mult", 2.5)),
            v_max=float(mapping.get("v_max", 0.0)),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "label": self.label,
            "algorithm": self.algorithm,
            "gamma": self.gamma,
            "start_lr": self.start_lr,
            "delay": self.delay,
            "lambda": self.lambda_value,
            "lr_type": self.lr_type,
            "sampling_distribution": self.sampling_distribution,
            "error_mode": self.error_mode,
        }
        if is_alt_isp_dyna_algorithm(self.algorithm):
            payload["J"] = self.J
            payload["h_target"] = self.h_target
            payload["epsilon"] = self.epsilon
            payload.pop("start_lr", None)
            payload.pop("delay", None)
            payload.pop("lr_type", None)
            payload.pop("sampling_distribution", None)
            return payload
        if is_gen_alt_isp_dyna_algorithm(self.algorithm):
            payload["J"] = self.J
            payload["h_target"] = self.h_target
            payload.pop("start_lr", None)
            payload.pop("delay", None)
            payload.pop("lr_type", None)
            payload.pop("sampling_distribution", None)
            return payload
        if is_isp_dyna_algorithm(self.algorithm):
            payload["J"] = self.J
            payload["batch_size"] = self.batch_size
            if self.h is not None:
                payload["h"] = self.h
            return payload
        if is_arch_isp_dyna_algorithm(self.algorithm):
            payload["J"] = self.J
            if self.h is not None:
                payload["h"] = self.h
            return payload
        if is_nr_rm_isp_dyna_algorithm(self.algorithm) or is_nr1_rm_isp_dyna_algorithm(self.algorithm):
            if self.h_terminal is not None or self.h_init is not None:
                payload["J"] = self.J
                if self.h_init is not None:
                    payload["h_init"] = self.h_init
                if self.h_terminal is not None:
                    payload["h_terminal"] = self.h_terminal
                if self.h_floor is not None:
                    payload["h_floor"] = self.h_floor
                payload["h_schedule_shape"] = self.h_schedule_shape
                payload["h_schedule_finish_frac"] = self.h_schedule_finish_frac
            elif self.h is not None:
                payload["J"] = self.J
                payload["h"] = self.h
            return payload
        if is_scheduled_waterfall_isp_dyna_algorithm(self.algorithm):
            payload["J"] = self.J
            if self.h_max is not None:
                payload["h_max"] = self.h_max
            if self.h_min is not None:
                payload["h_min"] = self.h_min
            payload["schedule_decay_rate"] = self.schedule_decay_rate
            return payload
        if is_arch_waterfall_isp_dyna_algorithm(self.algorithm):
            payload["J"] = self.J
            if self.h_max is not None:
                payload["h_max"] = self.h_max
            if self.h_min is not None:
                payload["h_min"] = self.h_min
            payload["schedule_decay_rate"] = self.schedule_decay_rate
            return payload
        if is_dd_geo_waterfall_isp_dyna_algorithm(self.algorithm):
            payload["J_max"] = self.J_max
            payload["batch_size"] = self.batch_size
            payload["decay_factor"] = self.decay_factor
            if self.h_init is not None:
                payload["h_init"] = self.h_init
            return payload
        if is_elastic_dd_geo_waterfall_isp_dyna_algorithm(self.algorithm):
            payload["J_max"] = self.J_max
            payload["batch_size"] = self.batch_size
            payload["decay_factor"] = self.decay_factor
            if self.h_init is not None:
                payload["h_init"] = self.h_init
            return payload
        if is_nr1_staircase_waterfall_rm_isp_dyna_algorithm(self.algorithm):
            payload["J"] = self.J
            if self.h_init is not None:
                payload["h_init"] = self.h_init
            if self.h_min is not None:
                payload["h_min"] = self.h_min
            if self.stall_patience is not None:
                payload["stall_patience"] = self.stall_patience
            payload["decay_factor"] = self.decay_factor
            payload["divergence_mult"] = self.divergence_mult
            payload["v_max"] = self.v_max
            return payload
        if is_nr1_geo_waterfall_rm_isp_dyna_algorithm(self.algorithm):
            payload["J"] = self.J
            if self.h_init is not None:
                payload["h_init"] = self.h_init
            if self.h_min is not None:
                payload["h_min"] = self.h_min
            if self.alpha is not None:
                payload["alpha"] = self.alpha
            return payload
        payload["planning_steps"] = self.planning_steps
        if self.h is not None:
            payload["h"] = self.h
        return payload

def normalize_hyper_batch_configs(configs: Iterable[Mapping[str, Any]]) -> list[HyperBatchConfig]:
    return [HyperBatchConfig.from_mapping(cfg) for cfg in configs]
