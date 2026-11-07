"""Compatibility and packing helpers for maintained stochastic hyper-batch runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ...core.offline_eval import (
    OFFLINE_EVAL_FLUSH_COUNT,
    OFFLINE_EVAL_MEMORY_BUDGET_BYTES,
    compute_safe_exact_solve_chunk_size,
    plan_offline_eval_layout,
)
from ...core.config import (
    canonical_algorithm_name,
    is_nr_rm_isp_dyna_algorithm,
    is_nr1_rm_isp_dyna_algorithm,
    is_nr1_staircase_waterfall_rm_isp_dyna_algorithm,
    is_nr1_geo_waterfall_rm_isp_dyna_algorithm,
    normalize_lr_type_name,
)
from .specs import HyperBatchConfig


@dataclass(frozen=True)
class StochasticBatchCompatibilityKey:
    algorithm: str
    gamma: float
    lr_type: str
    sampling_distribution: str
    error_mode: str
    h_profile_kind: str
    shared_scalars: tuple[tuple[str, float | int | str], ...]


def _nr_h_profile_kind(cfg: HyperBatchConfig) -> str:
    if cfg.h_init is not None or cfg.h_terminal is not None:
        return "scheduled_h"
    if cfg.h is not None:
        return "fixed_h"
    return "limit"


def stochastic_batch_compatibility_key(cfg: HyperBatchConfig) -> StochasticBatchCompatibilityKey:
    algorithm = canonical_algorithm_name(cfg.algorithm)
    h_profile_kind = "static"
    shared_scalars: list[tuple[str, float | int | str]] = []
    if is_nr1_staircase_waterfall_rm_isp_dyna_algorithm(algorithm):
        h_profile_kind = "waterfall_staircase"
        shared_scalars.extend(
            (
                ("J", int(cfg.J)),
                ("h_init", float(cfg.h_init if cfg.h_init is not None else 1.0)),
                ("h_min", float(cfg.h_min if cfg.h_min is not None else 1.0)),
                ("decay_factor", float(cfg.decay_factor if cfg.decay_factor is not None else 0.5)),
                ("stall_patience", int(cfg.stall_patience if cfg.stall_patience is not None else 0)),
            )
        )
    elif is_nr1_geo_waterfall_rm_isp_dyna_algorithm(algorithm):
        h_profile_kind = "waterfall_geo"
        shared_scalars.extend(
            (
                ("J", int(cfg.J)),
                ("h_init", float(cfg.h_init if cfg.h_init is not None else 1.0)),
                ("h_min", float(cfg.h_min if cfg.h_min is not None else 1.0)),
                ("alpha", float(cfg.alpha if cfg.alpha is not None else 1.0)),
            )
        )
    elif is_nr_rm_isp_dyna_algorithm(algorithm) or is_nr1_rm_isp_dyna_algorithm(algorithm):
        h_profile_kind = _nr_h_profile_kind(cfg)
        if h_profile_kind == "fixed_h":
            shared_scalars.extend((("J", int(cfg.J)), ("h", float(cfg.h if cfg.h is not None else 1.0))))
        elif h_profile_kind == "scheduled_h":
            shared_scalars.extend(
                (
                    ("J", int(cfg.J)),
                    ("h_init", float(cfg.h_init if cfg.h_init is not None else 1.0)),
                    ("h_terminal", float(cfg.h_terminal if cfg.h_terminal is not None else 1.0)),
                    ("h_floor", float(cfg.h_floor if cfg.h_floor is not None else 1.0)),
                    ("h_schedule_shape", str(cfg.h_schedule_shape)),
                    ("h_schedule_finish_frac", float(cfg.h_schedule_finish_frac)),
                )
            )
    return StochasticBatchCompatibilityKey(
        algorithm=algorithm,
        gamma=float(cfg.gamma),
        lr_type=normalize_lr_type_name(cfg.lr_type),
        sampling_distribution=str(cfg.sampling_distribution).lower(),
        error_mode=str(cfg.error_mode),
        h_profile_kind=h_profile_kind,
        shared_scalars=tuple(shared_scalars),
    )


def validate_packed_stochastic_group(
    cfgs: Sequence[HyperBatchConfig],
) -> StochasticBatchCompatibilityKey:
    if not cfgs:
        raise ValueError("Packed stochastic groups must contain at least one config.")
    keys = {stochastic_batch_compatibility_key(cfg) for cfg in cfgs}
    if len(keys) != 1:
        raise ValueError(
            "Packed stochastic configs must share one compatibility key; "
            f"received {sorted(keys, key=lambda item: (item.algorithm, item.lr_type, item.h_profile_kind))}."
        )
    return next(iter(keys))


def compute_packable_config_count(
    *,
    width_candidates: Iterable[int],
    n_seeds: int,
    training_steps: int,
    n_states: int,
    device_count: int = 1,
    flush_count: int = OFFLINE_EVAL_FLUSH_COUNT,
    memory_budget_bytes: int = OFFLINE_EVAL_MEMORY_BUDGET_BYTES,
) -> int:
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive.")
    if device_count <= 0:
        raise ValueError("device_count must be positive.")
    bytes_per_policy_row = ((2 * n_states * n_states) + (4 * n_states)) * 8
    best = 0
    for candidate in sorted({int(value) for value in width_candidates if int(value) > 0}):
        env_batch_size = candidate * n_seeds
        layout = plan_offline_eval_layout(
            training_steps=training_steps,
            env_batch_size=env_batch_size,
            n_states=n_states,
            flush_count=flush_count,
            memory_budget_bytes=memory_budget_bytes,
        )
        largest_capture_count = max(slab.capture_count for slab in layout.slabs)
        per_device_rows = (largest_capture_count * env_batch_size) / device_count
        policy_bytes = int(per_device_rows * n_states)
        start_state_bytes = int(per_device_rows * 8)
        residual_budget = int(memory_budget_bytes) - policy_bytes - start_state_bytes
        if residual_budget <= 0:
            continue
        solve_chunk_size = compute_safe_exact_solve_chunk_size(
            n_states=n_states,
            memory_budget_bytes=residual_budget,
        )
        solve_workspace_bytes = bytes_per_policy_row * solve_chunk_size
        if policy_bytes + start_state_bytes + solve_workspace_bytes <= memory_budget_bytes:
            best = candidate
    return best


__all__ = [
    "StochasticBatchCompatibilityKey",
    "compute_packable_config_count",
    "stochastic_batch_compatibility_key",
    "validate_packed_stochastic_group",
]
