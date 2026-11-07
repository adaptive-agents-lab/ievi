"""Start-state relative error helpers for EMP-MAIN-RL-CW."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


VSTAR_START = 819.1545470513453
DEFAULT_SAMPLE = 40000
REPLAY_SEEDS = tuple(range(20, 30))


def relative_error(value: float, v_star: float = VSTAR_START) -> float:
    return abs(float(value) - float(v_star)) / abs(float(v_star))


def _index_at_sample(x_axis: Sequence[int], sample: int) -> int:
    wanted = int(sample)
    for index, query in enumerate(x_axis):
        if int(query) == wanted:
            return index
    raise KeyError(f"sample {wanted} missing from x_axis")


def endpoint_summary(
    *,
    x_axis: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float] | None = None,
    seeds: Sequence[Sequence[float]] | None = None,
    sample: int = DEFAULT_SAMPLE,
    v_star: float = VSTAR_START,
    seed_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    index = _index_at_sample(x_axis, sample)
    mean_value = float(mean[index])
    seed_values = [float(value) for value in seeds[index]] if seeds else [mean_value]
    n_seeds = len(seed_values)
    if std is not None:
        std_value = float(std[index])
    elif n_seeds > 1:
        center = sum(seed_values) / n_seeds
        std_value = math.sqrt(sum((value - center) ** 2 for value in seed_values) / n_seeds)
    else:
        std_value = 0.0
    stderr = std_value / math.sqrt(n_seeds) if n_seeds else 0.0
    resolved_ids = [int(value) for value in (seed_ids or range(n_seeds))]
    return {
        "sample": int(sample),
        "v_star": float(v_star),
        "mean": mean_value,
        "std": std_value,
        "stderr": stderr,
        "n_seeds": n_seeds,
        "rel_error": relative_error(mean_value, v_star),
        "rel_error_se": stderr / abs(float(v_star)),
        "seed_ids": resolved_ids,
        "seed_values": seed_values,
        "seed_rel_errors": [relative_error(value, v_star) for value in seed_values],
    }


def compare_cell(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    mean_rel_atol: float = 0.01,
    seed_value_atol: float = 1.0,
) -> dict[str, Any]:
    actual_rel = float(actual["rel_error"])
    expected_rel = float(expected["rel_error"])
    mean_ok = abs(actual_rel - expected_rel) <= mean_rel_atol
    seed_pairs: list[dict[str, Any]] = []
    seeds_ok = True
    actual_by_seed = {
        int(seed): float(value)
        for seed, value in zip(actual.get("seed_ids", ()), actual.get("seed_values", ()), strict=False)
    }
    expected_by_seed = {
        int(seed): float(value)
        for seed, value in zip(expected.get("seed_ids", ()), expected.get("seed_values", ()), strict=False)
    }
    for seed in sorted(set(actual_by_seed) & set(expected_by_seed)):
        gap = abs(actual_by_seed[seed] - expected_by_seed[seed])
        seed_ok = gap <= seed_value_atol
        seeds_ok = seeds_ok and seed_ok
        seed_pairs.append(
            {
                "seed": seed,
                "actual": actual_by_seed[seed],
                "expected": expected_by_seed[seed],
                "abs_gap": gap,
                "ok": seed_ok,
            }
        )
    return {
        "id": actual.get("id", expected.get("id")),
        "mean_ok": mean_ok,
        "seeds_ok": seeds_ok,
        "actual_rel_error": actual_rel,
        "expected_rel_error": expected_rel,
        "rel_error_gap": abs(actual_rel - expected_rel),
        "seed_pairs": seed_pairs,
    }
