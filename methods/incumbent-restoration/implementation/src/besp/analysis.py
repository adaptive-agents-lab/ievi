"""Compatibility aggregation helpers over JAX/Python-native arrays."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp


def right_constant_curve(queries: list[int], errors: list[float], q_grid: Any) -> jax.Array:
    q_grid_arr = jnp.asarray(q_grid, dtype=jnp.float64)
    mlen = min(len(queries), len(errors))
    if mlen == 0:
        return jnp.full(q_grid_arr.shape, jnp.nan, dtype=jnp.float64)

    queries = queries[:mlen]
    errors = errors[:mlen]
    errs: list[float] = []
    idx = 0
    for q in q_grid_arr.tolist():
        while idx + 1 < len(queries) and queries[idx + 1] <= q:
            idx += 1
        errs.append(float(errors[idx]))
    return jnp.asarray(errs, dtype=jnp.float64)


def aggregate_curves(
    instances: list[dict[str, tuple[list[int], list[float]]]],
    q_grid: Any,
) -> dict[str, jax.Array]:
    algs = set().union(*(set(inst.keys()) for inst in instances))
    agg: dict[str, list[jax.Array]] = {alg: [] for alg in algs}
    for inst in instances:
        for alg, (q, e) in inst.items():
            agg[alg].append(right_constant_curve(q, e, q_grid))
    return {alg: jnp.nanmean(jnp.stack(curves, axis=0), axis=0) for alg, curves in agg.items()}


def aggregate_thresholds(
    instances: list[dict[str, tuple[list[int], list[float]]]],
    eps: float,
) -> dict[str, list[float]]:
    res: dict[str, list[float]] = {}
    for inst in instances:
        for alg, (q, e) in inst.items():
            hit = next((q[i] for i, err in enumerate(e) if err <= eps), math.inf)
            res.setdefault(alg, []).append(hit)
    return res


def queries_to_threshold(curves: dict[str, Any], q_grid: Any, eps: float) -> dict[str, float]:
    q_grid_arr = jnp.asarray(q_grid, dtype=jnp.float64)
    res: dict[str, float] = {}
    for alg, err_curve in curves.items():
        hits = jnp.where(jnp.asarray(err_curve, dtype=jnp.float64) <= eps)[0]
        if hits.size == 0:
            res[alg] = math.inf
        else:
            res[alg] = float(q_grid_arr[int(hits[0])])
    return res
