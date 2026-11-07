"""Result assembly helpers for maintained hyper-batch runs."""

from __future__ import annotations

from typing import Iterable

import jax.numpy as jnp

from ...core.types import HyperBatchTrace
from .specs import HyperBatchConfig


def build_trace(
    mean: jnp.ndarray,
    std: jnp.ndarray,
    seeds: jnp.ndarray,
    x_axis: jnp.ndarray,
    diagnostics: dict[str, jnp.ndarray] | None = None,
) -> HyperBatchTrace:
    return HyperBatchTrace(mean=mean, std=std, seeds=seeds, x_axis=x_axis, diagnostics=diagnostics)


def assemble_algorithm_results(
    *,
    cfgs: Iterable[HyperBatchConfig],
    data: jnp.ndarray,
    n_seeds: int,
    x_axis: list[int],
    diagnostics: dict[str, jnp.ndarray] | None = None,
) -> dict[str, HyperBatchTrace]:
    payload: dict[str, HyperBatchTrace] = {}
    data_jax = jnp.asarray(data)
    x_axis_jax = jnp.asarray(x_axis, dtype=jnp.int32)
    for cfg_idx, cfg in enumerate(cfgs):
        start = cfg_idx * n_seeds
        end = start + n_seeds
        subset = data_jax[:, start:end]
        trace_diagnostics: dict[str, jnp.ndarray] | None = None
        if diagnostics:
            trace_diagnostics = {}
            for name, values in diagnostics.items():
                values_jax = jnp.asarray(values)
                if values_jax.ndim == 1:
                    trace_diagnostics[name] = values_jax
                else:
                    trace_diagnostics[name] = values_jax[:, start:end]
        payload[cfg.label] = build_trace(
            mean=jnp.mean(subset, axis=1),
            std=jnp.std(subset, axis=1),
            seeds=subset,
            x_axis=x_axis_jax,
            diagnostics=trace_diagnostics,
        )
    return payload
