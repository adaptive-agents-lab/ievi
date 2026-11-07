"""Maintained metric helpers for stochastic-control JAX-RL experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


JAX_RL_DEFAULT_RETURN_TOLERANCE_REL = 1e-2
JAX_RL_DEFAULT_SUSTAINED_WINDOW = 20
JAX_RL_DEFAULT_EXCURSION_WINDOW = 20

JAX_RL_PRIMARY_METRIC_NAME = "sustained_hit_mean_samples"
JAX_RL_SECONDARY_METRIC_NAME = "first_hit_mean_samples"
JAX_RL_TERTIARY_METRIC_NAME = "final_target_gap"

JAX_RL_FIRST_HIT_MEAN_METRIC_NAME = "first_hit_mean_samples"
JAX_RL_FIRST_HIT_MEDIAN_METRIC_NAME = "first_hit_median_samples"
JAX_RL_FIRST_HIT_RATE_METRIC_NAME = "first_hit_rate_by_budget"
JAX_RL_FIRST_HIT_MEAN_CURVE_METRIC_NAME = "first_hit_mean_curve_samples"
JAX_RL_FIRST_HIT_SUCCESS_RATE_METRIC_NAME = "first_hit_success_rate"
JAX_RL_FIRST_HIT_MEAN_CONDITIONAL_METRIC_NAME = "first_hit_mean_conditional_samples"
JAX_RL_FIRST_HIT_MEDIAN_CONDITIONAL_METRIC_NAME = "first_hit_median_conditional_samples"
JAX_RL_FIRST_HIT_IQM_METRIC_NAME = "first_hit_iqm_samples"

JAX_RL_SUSTAINED_HIT_MEAN_METRIC_NAME = "sustained_hit_mean_samples"
JAX_RL_SUSTAINED_HIT_MEDIAN_METRIC_NAME = "sustained_hit_median_samples"
JAX_RL_SUSTAINED_HIT_RATE_METRIC_NAME = "sustained_hit_rate_by_budget"
JAX_RL_SUSTAINED_HIT_MEAN_CURVE_METRIC_NAME = "sustained_hit_mean_curve_samples"
JAX_RL_SUSTAINED_HIT_SUCCESS_RATE_METRIC_NAME = "sustained_hit_success_rate"
JAX_RL_SUSTAINED_HIT_MEAN_CONDITIONAL_METRIC_NAME = "sustained_hit_mean_conditional_samples"
JAX_RL_SUSTAINED_HIT_MEDIAN_CONDITIONAL_METRIC_NAME = "sustained_hit_median_conditional_samples"
JAX_RL_SUSTAINED_HIT_IQM_METRIC_NAME = "sustained_hit_iqm_samples"

JAX_RL_RS_HIT_MEAN_METRIC_NAME = "rs_hit_mean_samples"
JAX_RL_RS_HIT_MEDIAN_METRIC_NAME = "rs_hit_median_samples"
JAX_RL_RS_HIT_RATE_METRIC_NAME = "rs_hit_rate_by_budget"
JAX_RL_RS_HIT_MEAN_CURVE_METRIC_NAME = "rs_hit_mean_curve_samples"
JAX_RL_RS_HIT_SUCCESS_RATE_METRIC_NAME = "rs_hit_success_rate"
JAX_RL_RS_HIT_MEAN_CONDITIONAL_METRIC_NAME = "rs_hit_mean_conditional_samples"
JAX_RL_RS_HIT_MEDIAN_CONDITIONAL_METRIC_NAME = "rs_hit_median_conditional_samples"
JAX_RL_RS_HIT_IQM_METRIC_NAME = "rs_hit_iqm_samples"

JAX_RL_DEFAULT_BOOTSTRAP_RESAMPLES = 2000
JAX_RL_DEFAULT_BOOTSTRAP_CI = 0.95
JAX_RL_DEFAULT_IQM_TRIM_FRACTION = 0.25


@dataclass(frozen=True)
class JaxRlSeedwiseHitMetrics:
    """Aggregated hit statistics derived from per-seed return traces.

    The per-family ``*_success_rate`` field equals ``*_rate_by_budget`` by construction
    (sentinels equal ``max_samples + 1`` and are therefore strictly greater than any
    budget ``<= max_samples``). It is exposed as a semantic alias so success-rate-first
    reports can pull it by a more descriptive name without a second source of truth.
    """

    first_hit_seedwise_samples: tuple[int, ...]
    sustained_hit_seedwise_samples: tuple[int, ...]
    rs_hit_seedwise_samples: tuple[int, ...]
    first_hit_mean_samples: float
    first_hit_median_samples: float
    first_hit_rate_by_budget: float
    sustained_hit_mean_samples: float
    sustained_hit_median_samples: float
    sustained_hit_rate_by_budget: float
    rs_hit_mean_samples: float
    rs_hit_median_samples: float
    rs_hit_rate_by_budget: float
    first_hit_success_rate: float
    first_hit_mean_conditional_samples: float
    first_hit_median_conditional_samples: float
    first_hit_iqm_samples: float
    sustained_hit_success_rate: float
    sustained_hit_mean_conditional_samples: float
    sustained_hit_median_conditional_samples: float
    sustained_hit_iqm_samples: float
    rs_hit_success_rate: float
    rs_hit_mean_conditional_samples: float
    rs_hit_median_conditional_samples: float
    rs_hit_iqm_samples: float
    first_hit_mean_curve_samples: int
    sustained_hit_mean_curve_samples: int
    rs_hit_mean_curve_samples: int

    def as_mapping(self) -> dict[str, Any]:
        return {
            "first_hit_seedwise_samples": list(self.first_hit_seedwise_samples),
            "sustained_hit_seedwise_samples": list(self.sustained_hit_seedwise_samples),
            "rs_hit_seedwise_samples": list(self.rs_hit_seedwise_samples),
            JAX_RL_FIRST_HIT_MEAN_METRIC_NAME: self.first_hit_mean_samples,
            JAX_RL_FIRST_HIT_MEDIAN_METRIC_NAME: self.first_hit_median_samples,
            JAX_RL_FIRST_HIT_RATE_METRIC_NAME: self.first_hit_rate_by_budget,
            JAX_RL_SUSTAINED_HIT_MEAN_METRIC_NAME: self.sustained_hit_mean_samples,
            JAX_RL_SUSTAINED_HIT_MEDIAN_METRIC_NAME: self.sustained_hit_median_samples,
            JAX_RL_SUSTAINED_HIT_RATE_METRIC_NAME: self.sustained_hit_rate_by_budget,
            JAX_RL_RS_HIT_MEAN_METRIC_NAME: self.rs_hit_mean_samples,
            JAX_RL_RS_HIT_MEDIAN_METRIC_NAME: self.rs_hit_median_samples,
            JAX_RL_RS_HIT_RATE_METRIC_NAME: self.rs_hit_rate_by_budget,
            JAX_RL_FIRST_HIT_SUCCESS_RATE_METRIC_NAME: self.first_hit_success_rate,
            JAX_RL_FIRST_HIT_MEAN_CONDITIONAL_METRIC_NAME: self.first_hit_mean_conditional_samples,
            JAX_RL_FIRST_HIT_MEDIAN_CONDITIONAL_METRIC_NAME: self.first_hit_median_conditional_samples,
            JAX_RL_FIRST_HIT_IQM_METRIC_NAME: self.first_hit_iqm_samples,
            JAX_RL_SUSTAINED_HIT_SUCCESS_RATE_METRIC_NAME: self.sustained_hit_success_rate,
            JAX_RL_SUSTAINED_HIT_MEAN_CONDITIONAL_METRIC_NAME: self.sustained_hit_mean_conditional_samples,
            JAX_RL_SUSTAINED_HIT_MEDIAN_CONDITIONAL_METRIC_NAME: self.sustained_hit_median_conditional_samples,
            JAX_RL_SUSTAINED_HIT_IQM_METRIC_NAME: self.sustained_hit_iqm_samples,
            JAX_RL_RS_HIT_SUCCESS_RATE_METRIC_NAME: self.rs_hit_success_rate,
            JAX_RL_RS_HIT_MEAN_CONDITIONAL_METRIC_NAME: self.rs_hit_mean_conditional_samples,
            JAX_RL_RS_HIT_MEDIAN_CONDITIONAL_METRIC_NAME: self.rs_hit_median_conditional_samples,
            JAX_RL_RS_HIT_IQM_METRIC_NAME: self.rs_hit_iqm_samples,
            JAX_RL_FIRST_HIT_MEAN_CURVE_METRIC_NAME: self.first_hit_mean_curve_samples,
            JAX_RL_SUSTAINED_HIT_MEAN_CURVE_METRIC_NAME: self.sustained_hit_mean_curve_samples,
            JAX_RL_RS_HIT_MEAN_CURVE_METRIC_NAME: self.rs_hit_mean_curve_samples,
        }


def jax_rl_band_radius(target_value: float, tolerance_rel: float) -> float:
    return float(tolerance_rel) * max(1.0, abs(float(target_value)))


def jax_rl_payload_is_empty(payload: Any) -> bool:
    """Return True when a sequence-like trace payload is empty.

    This helper is safe for Python containers and JAX/NumPy arrays, and avoids
    ambiguous truth-value checks like ``if not payload`` on array-backed data.
    """

    if payload is None:
        return True
    shape = getattr(payload, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            return False
        return int(shape[0]) == 0
    try:
        return len(payload) == 0
    except TypeError:
        return False


def _in_band_mask_1d(
    trace: Sequence[float], target_value: float, band_radius: float
) -> list[bool]:
    """Return a per-tick boolean mask of whether a scalar trace sits inside the sleeve.

    Reused by the sustained-hit and RS-hit scanners so that any future sleeve-based
    metric can be built on top of the same primitive.
    """

    radius = float(band_radius)
    target = float(target_value)
    return [abs(float(value) - target) <= radius for value in trace]


def _last_long_false_run_end(mask: Sequence[bool], window: int) -> int:
    """Return the rightmost end-index of a False-run whose length exceeds ``window``.

    A *run* is a maximal contiguous subsequence of ``False`` values in ``mask``;
    its end-index is the last position inside that run. We scan left-to-right and
    keep the most recent qualifying end. Returns ``-1`` when no False-run is long
    enough. Used by RS-hit / robust-settling-time style metrics.
    """

    long_window = max(int(window), 0)
    run_length = 0
    last_end = -1
    for idx, flag in enumerate(mask):
        if not flag:
            run_length += 1
        else:
            if run_length > long_window:
                last_end = idx - 1
            run_length = 0
    if run_length > long_window:
        last_end = len(mask) - 1 if mask else -1
    return last_end


def _success_mask(seedwise: np.ndarray, max_samples: int) -> np.ndarray:
    """Boolean array of per-seed convergence: True iff the seed settled within the budget.

    Seeds that never settle carry the sentinel ``T_s = max_samples + 1``, so a simple
    ``<= max_samples`` comparison cleanly separates converged seeds from censored ones.
    Reused by ``_summarize_seedwise_samples`` and by any downstream consumer that needs
    to filter on convergence.
    """

    return np.asarray(seedwise, dtype=np.int64) <= int(max_samples)


def _interquartile_mean(
    values: np.ndarray,
    *,
    trim_fraction: float = JAX_RL_DEFAULT_IQM_TRIM_FRACTION,
) -> float:
    """25%-trimmed mean (a la rliable's IQM).

    Sorts ``values`` ascending, drops ``floor(N * trim_fraction)`` from each tail,
    and averages the middle. Returns ``NaN`` when fewer than 2 values survive the
    trim. ``trim_fraction == 0`` reduces to the plain mean. Generic enough to be
    reused by any robust-point-estimate report; not specific to hit-time metrics.
    """

    if not 0.0 <= float(trim_fraction) <= 0.5:
        raise ValueError("trim_fraction must lie in [0.0, 0.5].")
    array = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = int(array.size)
    if n == 0:
        return float("nan")
    trim = int(np.floor(n * float(trim_fraction)))
    if 2 * trim >= n:
        return float("nan")
    return float(np.mean(array[trim : n - trim]))


def _summarize_seedwise_samples(
    seedwise: np.ndarray,
    max_samples: int,
    *,
    iqm_trim_fraction: float = JAX_RL_DEFAULT_IQM_TRIM_FRACTION,
) -> dict[str, float]:
    """Compute all per-config summary statistics from a seedwise-hit array.

    Returns a dict with keys ``mean``, ``median``, ``rate_by_budget``, ``success_rate``,
    ``mean_conditional``, ``median_conditional``, ``iqm``. ``rate_by_budget`` and
    ``success_rate`` are identical by construction (sentinels equal ``max_samples + 1``,
    which is strictly greater than ``max_samples``); both are emitted to keep callers
    free of this arithmetic detail.

    Fields ``mean``, ``median`` and ``rate_by_budget`` use NumPy summary
    operations on integer seedwise inputs in the stated evaluation order.

    For comparisons at a shared physical-sample budget, entries above that
    budget are replaced with the corresponding censoring sentinel before
    summary statistics are computed. The stored raw results are unchanged.
    """

    array = np.asarray(seedwise, dtype=np.int64)
    if array.size == 0:
        raise ValueError("Cannot summarise an empty seedwise-hit array.")
    seed_count = int(array.size)
    budget = int(max_samples)
    mask = _success_mask(array, budget)
    converged_count = int(np.sum(mask))
    summary: dict[str, float] = {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "rate_by_budget": float(np.sum(array <= budget) / seed_count),
    }
    summary["success_rate"] = summary["rate_by_budget"]
    if converged_count == 0:
        summary["mean_conditional"] = float("nan")
        summary["median_conditional"] = float("nan")
    else:
        converged = array[mask]
        summary["mean_conditional"] = float(np.mean(converged))
        summary["median_conditional"] = float(np.median(converged))
    summary["iqm"] = _interquartile_mean(array, trim_fraction=iqm_trim_fraction)
    return summary


def _percentile_bootstrap_ci(
    values: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    *,
    n_resamples: int = JAX_RL_DEFAULT_BOOTSTRAP_RESAMPLES,
    ci: float = JAX_RL_DEFAULT_BOOTSTRAP_CI,
    seed: int = 0,
    drop_sentinel_at: int | None = None,
) -> tuple[float, float, float]:
    """Percentile-bootstrap confidence interval for an arbitrary statistic.

    Returns ``(low, point, high)`` where ``point = statistic_fn(values)`` on the
    (possibly sentinel-filtered) values and ``(low, high)`` are the lower/upper
    ``(1 - ci) / 2`` percentiles over ``n_resamples`` bootstrap resamples.

    - ``drop_sentinel_at``: if provided, drop entries ``> drop_sentinel_at`` before
      resampling; the convention for jax-rl seedwise-hit arrays is to pass the budget
      so censored seeds are excluded (matches conditional statistics).
    - Reproducibility: uses ``np.random.default_rng(seed)`` so two calls with the same
      seed return bit-identical tuples.
    - Degenerate cases: empty input returns ``(nan, nan, nan)``; size-one input
      returns ``(v, v, v)``.
    """

    if not 0.0 < float(ci) < 1.0:
        raise ValueError("ci must lie in (0, 1).")
    array = np.asarray(values, dtype=np.float64).ravel()
    if drop_sentinel_at is not None:
        array = array[array <= float(drop_sentinel_at)]
    if array.size == 0:
        nan = float("nan")
        return (nan, nan, nan)
    if array.size == 1:
        value = float(statistic_fn(array))
        return (value, value, value)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(n_resamples), array.size))
    resamples = array[indices]
    stats = np.array(
        [float(statistic_fn(resamples[r, :])) for r in range(int(n_resamples))],
        dtype=np.float64,
    )
    alpha = (1.0 - float(ci)) / 2.0
    low = float(np.quantile(stats, alpha))
    high = float(np.quantile(stats, 1.0 - alpha))
    point = float(statistic_fn(array))
    return (low, point, high)


def jax_rl_first_hit_samples(
    *,
    x_axis: Sequence[int],
    trace: Sequence[float],
    target_value: float,
    band_radius: float,
    max_samples: int,
) -> int:
    for sample_count, value in zip(x_axis, trace):
        if abs(float(value) - float(target_value)) <= float(band_radius):
            return int(sample_count)
    return int(max_samples) + 1


def jax_rl_sustained_hit_samples(
    *,
    x_axis: Sequence[int],
    trace: Sequence[float],
    target_value: float,
    band_radius: float,
    sustained_window: int,
    max_samples: int,
) -> int:
    window = max(int(sustained_window), 1)
    if len(trace) < window:
        return int(max_samples) + 1
    in_band = _in_band_mask_1d(trace, target_value, band_radius)
    for idx in range(0, len(in_band) - window + 1):
        if all(in_band[idx : idx + window]):
            return int(x_axis[idx])
    return int(max_samples) + 1


def jax_rl_rs_hit_samples(
    *,
    x_axis: Sequence[int],
    trace: Sequence[float],
    target_value: float,
    band_radius: float,
    excursion_window: int,
    max_samples: int,
) -> int:
    """Reverse-sustained-hit: earliest tick from which every out-of-sleeve excursion
    has length ≤ ``excursion_window`` AND the terminal tick is inside the sleeve.

    Returns ``max_samples + 1`` when the trace ends out-of-band or when the last
    qualifying excursion is still longer than the window (i.e. non-converged).
    """

    in_band = _in_band_mask_1d(trace, target_value, band_radius)
    if not in_band or not in_band[-1]:
        return int(max_samples) + 1
    last_long_end = _last_long_false_run_end(in_band, excursion_window)
    if last_long_end < 0:
        return jax_rl_first_hit_samples(
            x_axis=x_axis,
            trace=trace,
            target_value=target_value,
            band_radius=band_radius,
            max_samples=max_samples,
        )
    for idx in range(last_long_end + 1, len(in_band)):
        if in_band[idx]:
            return int(x_axis[idx])
    return int(max_samples) + 1


def jax_rl_hit_rate_by_budget_curve(
    *,
    x_axis: Sequence[int],
    hit_samples: Sequence[int],
) -> tuple[float, ...]:
    if not hit_samples:
        raise ValueError("Cannot compute hit-rate curves without seedwise hit samples.")
    seed_count = len(hit_samples)
    return tuple(
        sum(int(hit) <= int(sample_budget) for hit in hit_samples) / seed_count for sample_budget in x_axis
    )


def jax_rl_seedwise_hit_metrics(
    *,
    x_axis: Sequence[int],
    mean_trace: Sequence[float],
    seed_payload: Sequence[Any],
    target_value: float,
    band_radius: float,
    sustained_window: int,
    max_samples: int,
    excursion_window: int | None = None,
) -> JaxRlSeedwiseHitMetrics:
    effective_excursion_window = int(
        sustained_window if excursion_window is None else excursion_window
    )
    seed_matrix = _coerce_seed_matrix(seed_payload, expected_points=len(x_axis))
    if seed_matrix is not None:
        return _jax_rl_seedwise_hit_metrics_from_matrix(
            x_axis=x_axis,
            mean_trace=mean_trace,
            seed_matrix=seed_matrix,
            target_value=target_value,
            band_radius=band_radius,
            sustained_window=sustained_window,
            max_samples=max_samples,
            excursion_window=effective_excursion_window,
        )
    seed_traces = _coerce_seed_traces(seed_payload, expected_points=len(x_axis))
    if jax_rl_payload_is_empty(seed_traces):
        raise ValueError("Cannot compute seedwise hit metrics without per-seed traces.")

    first_hits = tuple(
        jax_rl_first_hit_samples(
            x_axis=x_axis,
            trace=seed_trace,
            target_value=target_value,
            band_radius=band_radius,
            max_samples=max_samples,
        )
        for seed_trace in seed_traces
    )
    sustained_hits = tuple(
        jax_rl_sustained_hit_samples(
            x_axis=x_axis,
            trace=seed_trace,
            target_value=target_value,
            band_radius=band_radius,
            sustained_window=sustained_window,
            max_samples=max_samples,
        )
        for seed_trace in seed_traces
    )
    rs_hits = tuple(
        jax_rl_rs_hit_samples(
            x_axis=x_axis,
            trace=seed_trace,
            target_value=target_value,
            band_radius=band_radius,
            excursion_window=effective_excursion_window,
            max_samples=max_samples,
        )
        for seed_trace in seed_traces
    )
    budget = int(max_samples)
    first_stats = _summarize_seedwise_samples(np.asarray(first_hits, dtype=np.int64), budget)
    sustained_stats = _summarize_seedwise_samples(np.asarray(sustained_hits, dtype=np.int64), budget)
    rs_stats = _summarize_seedwise_samples(np.asarray(rs_hits, dtype=np.int64), budget)
    return JaxRlSeedwiseHitMetrics(
        first_hit_seedwise_samples=first_hits,
        sustained_hit_seedwise_samples=sustained_hits,
        rs_hit_seedwise_samples=rs_hits,
        first_hit_mean_samples=first_stats["mean"],
        first_hit_median_samples=first_stats["median"],
        first_hit_rate_by_budget=first_stats["rate_by_budget"],
        sustained_hit_mean_samples=sustained_stats["mean"],
        sustained_hit_median_samples=sustained_stats["median"],
        sustained_hit_rate_by_budget=sustained_stats["rate_by_budget"],
        rs_hit_mean_samples=rs_stats["mean"],
        rs_hit_median_samples=rs_stats["median"],
        rs_hit_rate_by_budget=rs_stats["rate_by_budget"],
        first_hit_success_rate=first_stats["success_rate"],
        first_hit_mean_conditional_samples=first_stats["mean_conditional"],
        first_hit_median_conditional_samples=first_stats["median_conditional"],
        first_hit_iqm_samples=first_stats["iqm"],
        sustained_hit_success_rate=sustained_stats["success_rate"],
        sustained_hit_mean_conditional_samples=sustained_stats["mean_conditional"],
        sustained_hit_median_conditional_samples=sustained_stats["median_conditional"],
        sustained_hit_iqm_samples=sustained_stats["iqm"],
        rs_hit_success_rate=rs_stats["success_rate"],
        rs_hit_mean_conditional_samples=rs_stats["mean_conditional"],
        rs_hit_median_conditional_samples=rs_stats["median_conditional"],
        rs_hit_iqm_samples=rs_stats["iqm"],
        first_hit_mean_curve_samples=jax_rl_first_hit_samples(
            x_axis=x_axis,
            trace=mean_trace,
            target_value=target_value,
            band_radius=band_radius,
            max_samples=max_samples,
        ),
        sustained_hit_mean_curve_samples=jax_rl_sustained_hit_samples(
            x_axis=x_axis,
            trace=mean_trace,
            target_value=target_value,
            band_radius=band_radius,
            sustained_window=sustained_window,
            max_samples=max_samples,
        ),
        rs_hit_mean_curve_samples=jax_rl_rs_hit_samples(
            x_axis=x_axis,
            trace=mean_trace,
            target_value=target_value,
            band_radius=band_radius,
            excursion_window=effective_excursion_window,
            max_samples=max_samples,
        ),
    )


def _coerce_seed_matrix(seed_payload: Sequence[Any], *, expected_points: int) -> np.ndarray | None:
    shape = getattr(seed_payload, "shape", None)
    if shape is None:
        return None
    matrix = np.asarray(seed_payload, dtype=np.float64)
    if matrix.ndim != 2:
        return None
    if matrix.shape[0] == expected_points:
        return matrix
    if matrix.shape[1] == expected_points:
        return matrix.T
    raise ValueError(
        f"Seed matrix shape {matrix.shape} does not match x-axis length {expected_points}."
    )


def _first_hit_samples_from_mask(
    *,
    x_axis: np.ndarray,
    in_band: np.ndarray,
    max_samples: int,
) -> np.ndarray:
    if in_band.ndim != 2:
        raise ValueError("in_band must have shape (time, seeds).")
    any_hit = np.any(in_band, axis=0)
    first_indices = np.argmax(in_band, axis=0)
    return np.where(any_hit, x_axis[first_indices], int(max_samples) + 1).astype(np.int64)


def _sustained_hit_samples_from_mask(
    *,
    x_axis: np.ndarray,
    in_band: np.ndarray,
    sustained_window: int,
    max_samples: int,
) -> np.ndarray:
    if in_band.ndim != 2:
        raise ValueError("in_band must have shape (time, seeds).")
    window = max(int(sustained_window), 1)
    if in_band.shape[0] < window:
        return np.full((in_band.shape[1],), int(max_samples) + 1, dtype=np.int64)
    if window == 1:
        sustained_mask = in_band
    else:
        cumulative = np.cumsum(in_band.astype(np.int32), axis=0)
        sustained_sums = cumulative[window - 1 :].copy()
        if window < in_band.shape[0]:
            sustained_sums[1:] -= cumulative[:-window]
        sustained_mask = sustained_sums == window
    any_hit = np.any(sustained_mask, axis=0)
    first_indices = np.argmax(sustained_mask, axis=0)
    sustained_x_axis = x_axis[: sustained_mask.shape[0]]
    return np.where(any_hit, sustained_x_axis[first_indices], int(max_samples) + 1).astype(np.int64)


def _last_long_false_run_end_per_seed(mask: np.ndarray, window: int) -> np.ndarray:
    """Per-seed rightmost False-run end whose length exceeds ``window`` (``-1`` if none).

    Operates on a ``(time, seeds)`` boolean matrix. With seed counts typically in
    the low tens and trace lengths ≤ a few hundred points, a small Python loop
    over columns is both clear and fast enough; switch to a vectorised variant
    only if this becomes a hot spot.
    """

    if mask.ndim != 2:
        raise ValueError("mask must have shape (time, seeds).")
    long_window = max(int(window), 0)
    time_points, seed_count = mask.shape
    out = np.full((seed_count,), -1, dtype=np.int64)
    if time_points == 0:
        return out
    bool_mask = np.asarray(mask, dtype=bool)
    for seed_idx in range(seed_count):
        run_length = 0
        last_end = -1
        column = bool_mask[:, seed_idx]
        for tick_idx in range(time_points):
            if not column[tick_idx]:
                run_length += 1
            else:
                if run_length > long_window:
                    last_end = tick_idx - 1
                run_length = 0
        if run_length > long_window:
            last_end = time_points - 1
        out[seed_idx] = last_end
    return out


def _rs_hit_samples_from_mask(
    *,
    x_axis: np.ndarray,
    in_band: np.ndarray,
    excursion_window: int,
    max_samples: int,
) -> np.ndarray:
    """Vectorised (per-column) RS-hit scanner over a ``(time, seeds)`` boolean mask."""

    if in_band.ndim != 2:
        raise ValueError("in_band must have shape (time, seeds).")
    sentinel = int(max_samples) + 1
    time_points, seed_count = in_band.shape
    out = np.full((seed_count,), sentinel, dtype=np.int64)
    if time_points == 0:
        return out
    terminal_in_band = np.asarray(in_band[-1, :], dtype=bool)
    long_ends = _last_long_false_run_end_per_seed(in_band, excursion_window)
    for seed_idx in range(seed_count):
        if not terminal_in_band[seed_idx]:
            continue
        end_idx = int(long_ends[seed_idx])
        column = in_band[:, seed_idx]
        if end_idx < 0:
            if np.any(column):
                out[seed_idx] = int(x_axis[int(np.argmax(column))])
            continue
        tail = column[end_idx + 1 :]
        if np.any(tail):
            out[seed_idx] = int(x_axis[end_idx + 1 + int(np.argmax(tail))])
    return out


def _jax_rl_seedwise_hit_metrics_from_matrix(
    *,
    x_axis: Sequence[int],
    mean_trace: Sequence[float],
    seed_matrix: np.ndarray,
    target_value: float,
    band_radius: float,
    sustained_window: int,
    max_samples: int,
    excursion_window: int | None = None,
) -> JaxRlSeedwiseHitMetrics:
    effective_excursion_window = int(
        sustained_window if excursion_window is None else excursion_window
    )
    x_axis_arr = np.asarray(x_axis, dtype=np.int64)
    if seed_matrix.shape[0] != len(x_axis_arr):
        raise ValueError(
            f"Seed matrix first axis {seed_matrix.shape[0]} does not match x-axis length {len(x_axis_arr)}."
        )
    in_band = np.abs(seed_matrix - float(target_value)) <= float(band_radius)
    first_hits_arr = _first_hit_samples_from_mask(
        x_axis=x_axis_arr,
        in_band=in_band,
        max_samples=max_samples,
    )
    sustained_hits_arr = _sustained_hit_samples_from_mask(
        x_axis=x_axis_arr,
        in_band=in_band,
        sustained_window=sustained_window,
        max_samples=max_samples,
    )
    rs_hits_arr = _rs_hit_samples_from_mask(
        x_axis=x_axis_arr,
        in_band=in_band,
        excursion_window=effective_excursion_window,
        max_samples=max_samples,
    )
    mean_trace_arr = np.asarray(mean_trace, dtype=np.float64)
    mean_in_band = np.abs(mean_trace_arr - float(target_value)) <= float(band_radius)
    mean_first_hit = _first_hit_samples_from_mask(
        x_axis=x_axis_arr,
        in_band=mean_in_band[:, None],
        max_samples=max_samples,
    )[0]
    mean_sustained_hit = _sustained_hit_samples_from_mask(
        x_axis=x_axis_arr,
        in_band=mean_in_band[:, None],
        sustained_window=sustained_window,
        max_samples=max_samples,
    )[0]
    mean_rs_hit = _rs_hit_samples_from_mask(
        x_axis=x_axis_arr,
        in_band=mean_in_band[:, None],
        excursion_window=effective_excursion_window,
        max_samples=max_samples,
    )[0]
    budget = int(max_samples)
    first_stats = _summarize_seedwise_samples(first_hits_arr, budget)
    sustained_stats = _summarize_seedwise_samples(sustained_hits_arr, budget)
    rs_stats = _summarize_seedwise_samples(rs_hits_arr, budget)
    return JaxRlSeedwiseHitMetrics(
        first_hit_seedwise_samples=tuple(int(value) for value in first_hits_arr.tolist()),
        sustained_hit_seedwise_samples=tuple(int(value) for value in sustained_hits_arr.tolist()),
        rs_hit_seedwise_samples=tuple(int(value) for value in rs_hits_arr.tolist()),
        first_hit_mean_samples=first_stats["mean"],
        first_hit_median_samples=first_stats["median"],
        first_hit_rate_by_budget=first_stats["rate_by_budget"],
        sustained_hit_mean_samples=sustained_stats["mean"],
        sustained_hit_median_samples=sustained_stats["median"],
        sustained_hit_rate_by_budget=sustained_stats["rate_by_budget"],
        rs_hit_mean_samples=rs_stats["mean"],
        rs_hit_median_samples=rs_stats["median"],
        rs_hit_rate_by_budget=rs_stats["rate_by_budget"],
        first_hit_success_rate=first_stats["success_rate"],
        first_hit_mean_conditional_samples=first_stats["mean_conditional"],
        first_hit_median_conditional_samples=first_stats["median_conditional"],
        first_hit_iqm_samples=first_stats["iqm"],
        sustained_hit_success_rate=sustained_stats["success_rate"],
        sustained_hit_mean_conditional_samples=sustained_stats["mean_conditional"],
        sustained_hit_median_conditional_samples=sustained_stats["median_conditional"],
        sustained_hit_iqm_samples=sustained_stats["iqm"],
        rs_hit_success_rate=rs_stats["success_rate"],
        rs_hit_mean_conditional_samples=rs_stats["mean_conditional"],
        rs_hit_median_conditional_samples=rs_stats["median_conditional"],
        rs_hit_iqm_samples=rs_stats["iqm"],
        first_hit_mean_curve_samples=int(mean_first_hit),
        sustained_hit_mean_curve_samples=int(mean_sustained_hit),
        rs_hit_mean_curve_samples=int(mean_rs_hit),
    )


def _coerce_seed_traces(seed_payload: Sequence[Any], *, expected_points: int) -> list[list[float]]:
    if jax_rl_payload_is_empty(seed_payload):
        return []

    first = seed_payload[0]
    if isinstance(first, dict):
        traces: list[list[float]] = []
        for entry in seed_payload:
            if not isinstance(entry, dict) or "trace" not in entry:
                raise TypeError("Expected each seed entry to be a mapping with a 'trace' field.")
            trace = [float(value) for value in entry["trace"]]
            if len(trace) != expected_points:
                raise ValueError(
                    f"Seed trace length {len(trace)} does not match x-axis length {expected_points}."
                )
            traces.append(trace)
        return traces

    matrix = [[float(value) for value in row] for row in seed_payload]
    row_length = len(matrix[0]) if matrix else 0
    if any(len(row) != row_length for row in matrix):
        raise ValueError("Per-seed trace payload must be rectangular.")
    if len(matrix) == expected_points:
        return [
            [matrix[time_idx][seed_idx] for time_idx in range(len(matrix))]
            for seed_idx in range(row_length)
        ]
    if row_length == expected_points:
        return matrix
    raise ValueError(
        "Could not infer seed-trace orientation from payload "
        f"(shape={len(matrix)}x{row_length}, expected_points={expected_points})."
    )


__all__ = [
    "JAX_RL_DEFAULT_BOOTSTRAP_CI",
    "JAX_RL_DEFAULT_BOOTSTRAP_RESAMPLES",
    "JAX_RL_DEFAULT_EXCURSION_WINDOW",
    "JAX_RL_DEFAULT_IQM_TRIM_FRACTION",
    "JAX_RL_DEFAULT_RETURN_TOLERANCE_REL",
    "JAX_RL_DEFAULT_SUSTAINED_WINDOW",
    "JAX_RL_FIRST_HIT_IQM_METRIC_NAME",
    "JAX_RL_FIRST_HIT_MEAN_CONDITIONAL_METRIC_NAME",
    "JAX_RL_FIRST_HIT_MEAN_CURVE_METRIC_NAME",
    "JAX_RL_FIRST_HIT_MEAN_METRIC_NAME",
    "JAX_RL_FIRST_HIT_MEDIAN_CONDITIONAL_METRIC_NAME",
    "JAX_RL_FIRST_HIT_MEDIAN_METRIC_NAME",
    "JAX_RL_FIRST_HIT_RATE_METRIC_NAME",
    "JAX_RL_FIRST_HIT_SUCCESS_RATE_METRIC_NAME",
    "JAX_RL_PRIMARY_METRIC_NAME",
    "JAX_RL_RS_HIT_IQM_METRIC_NAME",
    "JAX_RL_RS_HIT_MEAN_CONDITIONAL_METRIC_NAME",
    "JAX_RL_RS_HIT_MEAN_CURVE_METRIC_NAME",
    "JAX_RL_RS_HIT_MEAN_METRIC_NAME",
    "JAX_RL_RS_HIT_MEDIAN_CONDITIONAL_METRIC_NAME",
    "JAX_RL_RS_HIT_MEDIAN_METRIC_NAME",
    "JAX_RL_RS_HIT_RATE_METRIC_NAME",
    "JAX_RL_RS_HIT_SUCCESS_RATE_METRIC_NAME",
    "JAX_RL_SECONDARY_METRIC_NAME",
    "JAX_RL_SUSTAINED_HIT_IQM_METRIC_NAME",
    "JAX_RL_SUSTAINED_HIT_MEAN_CONDITIONAL_METRIC_NAME",
    "JAX_RL_SUSTAINED_HIT_MEAN_CURVE_METRIC_NAME",
    "JAX_RL_SUSTAINED_HIT_MEAN_METRIC_NAME",
    "JAX_RL_SUSTAINED_HIT_MEDIAN_CONDITIONAL_METRIC_NAME",
    "JAX_RL_SUSTAINED_HIT_MEDIAN_METRIC_NAME",
    "JAX_RL_SUSTAINED_HIT_RATE_METRIC_NAME",
    "JAX_RL_SUSTAINED_HIT_SUCCESS_RATE_METRIC_NAME",
    "JAX_RL_TERTIARY_METRIC_NAME",
    "JaxRlSeedwiseHitMetrics",
    "_interquartile_mean",
    "_percentile_bootstrap_ci",
    "_success_mask",
    "_summarize_seedwise_samples",
    "jax_rl_band_radius",
    "jax_rl_first_hit_samples",
    "jax_rl_hit_rate_by_budget_curve",
    "jax_rl_payload_is_empty",
    "jax_rl_rs_hit_samples",
    "jax_rl_seedwise_hit_metrics",
    "jax_rl_sustained_hit_samples",
]
