"""Shared planning utilities for offline policy capture and evaluation."""

from __future__ import annotations

from typing import Sequence

from .types import OfflineEvalPlan, OfflineEvalSlabSpec


OFFLINE_EVAL_FLUSH_COUNT = 4
OFFLINE_EVAL_POLICY_DTYPE = "uint8"
OFFLINE_EVAL_VALUE_DTYPE = "float64"
OFFLINE_EVAL_MEMORY_BUDGET_BYTES = 600 * 1024 * 1024
_CHUNK_ALIGNMENT = 256


def _split_evenly(total: int, parts: int) -> tuple[int, ...]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    base = total // parts
    remainder = total % parts
    return tuple(base + (1 if index < remainder else 0) for index in range(parts))


def _aligned_positive(value: int, *, alignment: int) -> int:
    if value <= 0:
        return 1
    if value < alignment:
        return value
    return max(alignment, (value // alignment) * alignment)


def compute_safe_exact_solve_chunk_size(
    *,
    n_states: int,
    memory_budget_bytes: int = OFFLINE_EVAL_MEMORY_BUDGET_BYTES,
) -> int:
    if n_states <= 0:
        raise ValueError("n_states must be positive")
    if memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")
    bytes_per_policy = ((2 * n_states * n_states) + (4 * n_states)) * 8
    raw_chunk = max(1, memory_budget_bytes // max(1, bytes_per_policy))
    return _aligned_positive(int(raw_chunk), alignment=_CHUNK_ALIGNMENT)


def build_sample_axis(training_steps: int) -> tuple[int, ...]:
    if training_steps < 0:
        raise ValueError("training_steps must be non-negative")
    return tuple(range(training_steps + 1))


def plan_offline_eval_layout(
    *,
    training_steps: int,
    env_batch_size: int,
    n_states: int,
    flush_count: int = OFFLINE_EVAL_FLUSH_COUNT,
    policy_dtype: str = OFFLINE_EVAL_POLICY_DTYPE,
    value_dtype: str = OFFLINE_EVAL_VALUE_DTYPE,
    memory_budget_bytes: int = OFFLINE_EVAL_MEMORY_BUDGET_BYTES,
    sample_axis: Sequence[int] | None = None,
) -> OfflineEvalPlan:
    if env_batch_size <= 0:
        raise ValueError("env_batch_size must be positive")
    if flush_count <= 0:
        raise ValueError("flush_count must be positive")
    resolved_axis = tuple(int(value) for value in (sample_axis or build_sample_axis(training_steps)))
    if not resolved_axis:
        raise ValueError("sample_axis must not be empty")
    capture_counts = _split_evenly(len(resolved_axis), flush_count)
    slabs: list[OfflineEvalSlabSpec] = []
    capture_cursor = 0
    row_cursor = 0
    for slab_index, capture_count in enumerate(capture_counts):
        capture_start = capture_cursor
        capture_stop = capture_start + capture_count
        row_start = row_cursor
        row_stop = row_start + capture_count * env_batch_size
        slabs.append(
            OfflineEvalSlabSpec(
                slab_index=slab_index,
                capture_start=capture_start,
                capture_stop=capture_stop,
                row_start=row_start,
                row_stop=row_stop,
            )
        )
        capture_cursor = capture_stop
        row_cursor = row_stop
    return OfflineEvalPlan(
        training_steps=training_steps,
        env_batch_size=env_batch_size,
        n_states=n_states,
        sample_axis=resolved_axis,
        slabs=tuple(slabs),
        flush_count=flush_count,
        policy_dtype=policy_dtype,
        value_dtype=value_dtype,
        solve_chunk_size=compute_safe_exact_solve_chunk_size(
            n_states=n_states,
            memory_budget_bytes=memory_budget_bytes,
        ),
        memory_budget_bytes=memory_budget_bytes,
    )


__all__ = [
    "OFFLINE_EVAL_FLUSH_COUNT",
    "OFFLINE_EVAL_MEMORY_BUDGET_BYTES",
    "OFFLINE_EVAL_POLICY_DTYPE",
    "OFFLINE_EVAL_VALUE_DTYPE",
    "build_sample_axis",
    "compute_safe_exact_solve_chunk_size",
    "plan_offline_eval_layout",
]
