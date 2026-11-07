#!/usr/bin/env python3
"""Matched Policy Evaluation horizon-scaling experiment.

Every reported algorithm is executed from scratch for every condition:
  * VI
  * OS-VI
  * IE-VI with Equation-16 parameters
  * IE-VI with Theorem-27 constructive joint tuning

The experiment uses a fixed true MRP induced by one policy in Modified
Cliffwalk, paired nested empirical surrogates, and direct instrumentation of:
  N_P, N_Phat, factorizations, linear solves, stable-hit query count, and
  algorithm-only wall-clock time including required factorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

# Keep BLAS deterministic/single-threaded before importing NumPy/SciPy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve
from threadpoolctl import threadpool_info, threadpool_limits

H_GRID = np.asarray([10, 20, 40, 80, 160, 320, 640], dtype=np.int64)
SCHEDULE_ORDER = ("fixed80", "linear16")
ALGORITHM_ORDER = ("VI", "OS-VI", "IE-VI Eq16", "IE-VI Joint")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--replications", type=int, default=30)
    p.add_argument("--base-seed", type=int, required=True)
    p.add_argument("--target", type=float, default=1e-3)
    p.add_argument("--stable-tail", type=int, default=50)
    p.add_argument("--max-true-applications", type=int, default=250_000)
    p.add_argument("--timing-repetitions", type=int, default=5)
    p.add_argument("--timing-batch-target-seconds", type=float, default=0.008)
    p.add_argument("--skip-timing", action="store_true")
    return p.parse_args()


def build_modified_cliffwalk() -> tuple[np.ndarray, np.ndarray]:
    """Exact 6x6 Modified Cliffwalk transition and reward semantics."""
    n = 6
    S, A = n * n, 4
    goal = (0, 5)
    hazards = {(r, c) for r in (0, 2, 4) for c in range(1, 5)}
    reward_by_row = {0: -32.0, 2: -16.0, 4: -8.0}
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    P = np.zeros((S, A, S), dtype=np.float64)
    R = np.full((S, A), -1.0, dtype=np.float64)

    def idx(r: int, c: int) -> int:
        return n * r + c

    for r in range(n):
        for c in range(n):
            s = idx(r, c)
            if (r, c) == goal:
                R[s, :] = 20.0
            elif (r, c) in hazards:
                R[s, :] = reward_by_row[r]
            terminal = (r, c) == goal or (r, c) in hazards
            for a in range(A):
                if terminal:
                    P[s, a, s] = 1.0
                    continue
                for actual, (dr, dc) in enumerate(actions):
                    prob = 0.9 if actual == a else 0.1 / 3.0
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= n or nc < 0 or nc >= n:
                        nr, nc = r, c
                    P[s, a, idx(nr, nc)] += prob
    return P, R


def solve_control_policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    init_policy: np.ndarray | None = None,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    S = P.shape[0]
    rows = np.arange(S)
    policy = (
        np.argmax(R, axis=1).astype(np.int64)
        if init_policy is None
        else np.asarray(init_policy, dtype=np.int64).copy()
    )
    for _ in range(max_iter):
        value = np.linalg.solve(
            np.eye(S) - gamma * P[rows, policy, :],
            R[rows, policy],
        )
        q = R + gamma * np.einsum("sak,k->sa", P, value)
        new_policy = np.argmax(q, axis=1).astype(np.int64)
        if np.array_equal(new_policy, policy):
            return value, policy
        policy = new_policy
    raise RuntimeError("Policy iteration did not converge")


def n_for(schedule: str, H: int) -> int:
    if schedule == "fixed80":
        return 80
    if schedule == "linear16":
        return 16 * H
    raise KeyError(schedule)


def relative_linf(value: np.ndarray, target: np.ndarray) -> float:
    denom = max(float(np.max(np.abs(target))), 1e-300)
    return float(np.max(np.abs(value - target)) / denom)


def model_error(P: np.ndarray, P_hat: np.ndarray, gamma: float) -> float:
    return float(gamma * np.max(np.sum(np.abs(P - P_hat), axis=1)))


def mu_envelope(h: float, m: int, gamma: float, E_inf: float) -> float:
    d = 1.0 - gamma
    rho = h * E_inf / (1.0 + h * d)
    return float(
        1.0 / (1.0 + h * d)
        + rho**m
        * h
        * (1.0 + gamma + h * d)
        / ((1.0 + h) * (1.0 + h * d))
    )


def eq16_parameters(gamma: float, E_inf: float) -> tuple[float, int, float, float]:
    d = 1.0 - gamma
    R_E = E_inf / d
    Rbar = max(1.0, R_E)
    h = 1.0 / (d * (2.0 * Rbar - 1.0))
    m = int(math.ceil(math.log2(8.0 / d)))
    mu = mu_envelope(h, m, gamma, E_inf)
    target_mu = 1.0 - 1.0 / (4.0 * Rbar)
    if mu > target_mu + 5e-12:
        raise AssertionError(("Eq16 envelope failure", mu, target_mu, h, m, R_E))
    return h, m, mu, Rbar


def theorem27_parameters(gamma: float, E_inf: float) -> tuple[float, int, float, float]:
    """Constructive joint tuning from Theorem 27."""
    d = 1.0 - gamma
    H = 1.0 / d
    R_E = E_inf / d
    Lambda = max(1.0, R_E)
    if E_inf <= 0.0:
        h, m, mu, _ = eq16_parameters(gamma, 0.0)
        return h, m, mu, Lambda
    m = 1
    lhs_base = 4.0 * Lambda / R_E
    while True:
        if m * math.log(lhs_base) >= math.log(
            8.0 * math.e * (m + 1.0) * min(H, Lambda)
        ):
            break
        m += 1
        if m > 10_000:
            raise RuntimeError("Theorem-27 tuner failed")
    u = (m + 1.0) / (4.0 * Lambda * m)
    h = u / (d * (1.0 - u))
    mu = mu_envelope(h, m, gamma, E_inf)
    target_mu = 1.0 - 1.0 / (4.0 * Lambda)
    if mu > target_mu + 5e-12:
        raise AssertionError(("Theorem27 envelope failure", mu, target_mu, h, m, R_E))
    return h, m, mu, Lambda


def matrix_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def spectral_radius(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(a))))


def osvi_error_matrix(P: np.ndarray, P_hat: np.ndarray, gamma: float) -> np.ndarray:
    A = np.eye(P.shape[0]) - gamma * P_hat
    return np.linalg.solve(A, gamma * (P - P_hat))


def ievi_error_matrix(
    P: np.ndarray,
    P_hat: np.ndarray,
    gamma: float,
    h: float,
    m: int,
) -> np.ndarray:
    S = P.shape[0]
    A = np.eye(S) - (h * gamma / (1.0 + h)) * P_hat
    Ainv = np.linalg.inv(A)
    C = Ainv / (1.0 + h)
    D = Ainv @ (h * gamma * (P - P_hat) / (1.0 + h))
    M = np.linalg.matrix_power(D, m)
    Dpow = np.eye(S)
    for _ in range(m):
        M = M + Dpow @ C
        Dpow = Dpow @ D
    return M


@dataclass
class CountResult:
    hit: bool
    stable_hit_true_applications: int
    stable_hit_outer_iterations: int
    verification_true_applications: int
    verification_outer_iterations: int
    error_at_hit: float
    error_after_stable_tail: float


def _stable_update(
    err: float,
    index: int,
    target: float,
    first: int | None,
    tail: int,
) -> tuple[int | None, int]:
    if err <= target:
        if first is None:
            first = index
        tail += 1
    else:
        first = None
        tail = 0
    return first, tail


def count_vi(
    P: np.ndarray,
    r: np.ndarray,
    gamma: float,
    target_value: np.ndarray,
    target: float,
    stable_tail: int,
    cap: int,
) -> CountResult:
    v = np.zeros_like(r)
    first: int | None = None
    tail = 0
    err_at_hit = math.inf
    for q in range(1, cap + 1):
        v = r + gamma * (P @ v)
        err = relative_linf(v, target_value)
        old_first = first
        first, tail = _stable_update(err, q, target, first, tail)
        if old_first is None and first is not None:
            err_at_hit = err
        if tail >= stable_tail:
            assert first is not None
            return CountResult(True, first, first, q, q, err_at_hit, err)
    return CountResult(False, cap, cap, cap, cap, math.inf, relative_linf(v, target_value))


def count_osvi(
    P: np.ndarray,
    P_hat: np.ndarray,
    r: np.ndarray,
    gamma: float,
    target_value: np.ndarray,
    target: float,
    stable_tail: int,
    cap: int,
) -> CountResult:
    S = P.shape[0]
    lu, piv = lu_factor(np.eye(S) - gamma * P_hat, check_finite=False)
    v = np.zeros_like(r)
    first: int | None = None
    tail = 0
    err_at_hit = math.inf
    for q in range(1, cap + 1):
        p_v = P @ v
        ph_v = P_hat @ v
        rhs = r + gamma * (p_v - ph_v)
        v = lu_solve((lu, piv), rhs, check_finite=False)
        if not np.all(np.isfinite(v)) or float(np.max(np.abs(v))) > 1e150:
            return CountResult(False, q, q, q, q, math.inf, math.inf)
        err = relative_linf(v, target_value)
        old_first = first
        first, tail = _stable_update(err, q, target, first, tail)
        if old_first is None and first is not None:
            err_at_hit = err
        if tail >= stable_tail:
            assert first is not None
            return CountResult(True, first, first, q, q, err_at_hit, err)
    return CountResult(False, cap, cap, cap, cap, math.inf, relative_linf(v, target_value))


def count_ievi(
    P: np.ndarray,
    P_hat: np.ndarray,
    r: np.ndarray,
    gamma: float,
    h: float,
    m: int,
    target_value: np.ndarray,
    target: float,
    stable_tail: int,
    cap: int,
) -> CountResult:
    S = P.shape[0]
    gamma_h = h * gamma / (1.0 + h)
    lu, piv = lu_factor(np.eye(S) - gamma_h * P_hat, check_finite=False)
    v = np.zeros_like(r)
    first_outer: int | None = None
    tail = 0
    err_at_hit = math.inf
    outer = 0
    q = 0
    while q + m <= cap:
        outer += 1
        u = v.copy()
        for _ in range(m):
            p_u = P @ u
            ph_u = P_hat @ u
            delta = r + gamma * (p_u - ph_u)
            rhs = v / (1.0 + h) + (h / (1.0 + h)) * delta
            u = lu_solve((lu, piv), rhs, check_finite=False)
            q += 1
        v = u
        if not np.all(np.isfinite(v)) or float(np.max(np.abs(v))) > 1e150:
            return CountResult(False, q, outer, q, outer, math.inf, math.inf)
        err = relative_linf(v, target_value)
        old_first = first_outer
        first_outer, tail = _stable_update(err, outer, target, first_outer, tail)
        if old_first is None and first_outer is not None:
            err_at_hit = err
        if tail >= stable_tail:
            assert first_outer is not None
            return CountResult(
                True,
                first_outer * m,
                first_outer,
                q,
                outer,
                err_at_hit,
                err,
            )
    return CountResult(False, q, outer, q, outer, math.inf, relative_linf(v, target_value))


def execute_vi(P: np.ndarray, r: np.ndarray, gamma: float, q: int) -> np.ndarray:
    v = np.zeros_like(r)
    for _ in range(q):
        v = r + gamma * (P @ v)
    return v


def execute_osvi(
    P: np.ndarray,
    P_hat: np.ndarray,
    r: np.ndarray,
    gamma: float,
    q: int,
) -> np.ndarray:
    S = P.shape[0]
    lu, piv = lu_factor(np.eye(S) - gamma * P_hat, check_finite=False)
    v = np.zeros_like(r)
    for _ in range(q):
        p_v = P @ v
        ph_v = P_hat @ v
        v = lu_solve((lu, piv), r + gamma * (p_v - ph_v), check_finite=False)
    return v


def execute_ievi(
    P: np.ndarray,
    P_hat: np.ndarray,
    r: np.ndarray,
    gamma: float,
    h: float,
    m: int,
    q: int,
) -> np.ndarray:
    if q % m != 0:
        raise ValueError((q, m))
    S = P.shape[0]
    lu, piv = lu_factor(
        np.eye(S) - (h * gamma / (1.0 + h)) * P_hat,
        check_finite=False,
    )
    v = np.zeros_like(r)
    for _ in range(q // m):
        u = v.copy()
        for _ in range(m):
            p_u = P @ u
            ph_u = P_hat @ u
            delta = r + gamma * (p_u - ph_u)
            rhs = v / (1.0 + h) + (h / (1.0 + h)) * delta
            u = lu_solve((lu, piv), rhs, check_finite=False)
        v = u
    return v


def benchmark_callable(
    fn: Callable[[], np.ndarray],
    repetitions: int,
    target_batch_seconds: float,
) -> tuple[list[float], int]:
    # Warm-up and estimate one-call duration.
    fn()
    t0 = time.perf_counter_ns()
    fn()
    one = max((time.perf_counter_ns() - t0) / 1e9, 1e-9)
    batch = int(math.ceil(target_batch_seconds / one))
    batch = max(1, min(batch, 500))
    samples: list[float] = []
    for _ in range(repetitions):
        t0 = time.perf_counter_ns()
        checksum = 0.0
        for _ in range(batch):
            checksum += float(fn()[0])
        elapsed = (time.perf_counter_ns() - t0) / 1e9 / batch
        if not math.isfinite(checksum):
            raise RuntimeError("Non-finite timing checksum")
        samples.append(elapsed)
    return samples, batch


def loglog_slope(x: Iterable[float], y: Iterable[float]) -> float:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya) & (xa > 0) & (ya > 0)
    if np.unique(xa[mask]).size < 2:
        return float("nan")
    return float(np.polyfit(np.log(xa[mask]), np.log(ya[mask]), 1)[0])


def bootstrap_slope_details(matrix: np.ndarray, x: np.ndarray, n_boot: int, seed: int) -> dict:
    """Paired-seed bootstrap; unsupported fits remain explicitly unavailable.

    An interval requires every requested resample to have two distinct valid
    horizons. We retain the point estimate but withhold the interval otherwise.
    """
    unavailable = dict(slope=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                       bootstrap_valid_resamples=0, bootstrap_requested_resamples=n_boot,
                       status="insufficient_distinct_horizons")
    if matrix.shape[0] == 0 or x.size < 2:
        return unavailable
    point = loglog_slope(x, np.median(matrix, axis=0))
    if not np.isfinite(point):
        return unavailable
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    n = matrix.shape[0]
    for b in range(n_boot):
        sample = matrix[rng.integers(0, n, n)]
        vals[b] = loglog_slope(x, np.median(sample, axis=0))
    valid = np.isfinite(vals)
    info = dict(unavailable, slope=point, bootstrap_valid_resamples=int(valid.sum()),
                status="insufficient_bootstrap_support")
    if n_boot > 0 and valid.all():
        lo, hi = np.quantile(vals, [0.025, 0.975])
        info.update(ci_low=float(lo), ci_high=float(hi), status="available")
    return info


def bootstrap_slope(matrix: np.ndarray, x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    info = bootstrap_slope_details(matrix, x, n_boot, seed)
    return info["slope"], info["ci_low"], info["ci_high"]


def mad(values: pd.Series) -> float:
    a = values.to_numpy(dtype=np.float64)
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def capture_environment(root: Path, args: argparse.Namespace) -> None:
    env = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "threadpool_info": threadpool_info(),
        "arguments": vars(args) | {"output_root": str(args.output_root)},
    }
    (root / "environment" / "environment.json").write_text(
        json.dumps(env, indent=2, default=str) + "\n"
    )
    try:
        cpu = subprocess.run(
            ["bash", "-lc", "lscpu"], capture_output=True, text=True, check=False
        ).stdout
        (root / "environment" / "lscpu.txt").write_text(cpu)
    except Exception as exc:
        (root / "environment" / "lscpu.txt").write_text(f"Unavailable: {exc}\n")


def make_models(
    P: np.ndarray,
    replications: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    S = P.shape[0]
    nmax = max(n_for(schedule, int(H)) for schedule in SCHEDULE_ORDER for H in H_GRID)
    models = np.empty(
        (replications, len(SCHEDULE_ORDER), len(H_GRID), S, S), dtype=np.float64
    )
    seeds = np.empty(replications, dtype=np.int64)
    for rep in range(replications):
        seed = base_seed + rep
        seeds[rep] = seed
        rng = np.random.default_rng(seed)
        samples = np.stack(
            [rng.choice(S, size=nmax, replace=True, p=P[s]) for s in range(S)],
            axis=0,
        )
        for si, schedule in enumerate(SCHEDULE_ORDER):
            for hi, H in enumerate(H_GRID):
                n = n_for(schedule, int(H))
                ph = np.zeros_like(P)
                for s in range(S):
                    ph[s] = np.bincount(samples[s, :n], minlength=S) / float(n)
                if not np.allclose(ph.sum(axis=1), 1.0, atol=1e-14):
                    raise AssertionError("Non-stochastic empirical row")
                # Every nonzero probability must be an integer multiple of 1/n.
                if not np.allclose(ph * n, np.rint(ph * n), atol=1e-10):
                    raise AssertionError("Empirical row violates exact sample count")
                models[rep, si, hi] = ph
    return models, seeds


def aggregate_results(root: Path, records: pd.DataFrame, timings: pd.DataFrame, *, timing_requested: bool = True) -> dict:
    derived = root / "data" / "derived"
    for directory in (derived, root / "data" / "raw"):
        directory.mkdir(parents=True, exist_ok=True)
    valid_times = np.isfinite(records.wall_time_median_seconds) & (records.wall_time_median_seconds > 0)
    if timing_requested and not valid_times.any():
        raise ValueError("Timing was requested but no finite positive timing measurements are available")
    if not timing_requested and (valid_times.any() or len(timings)):
        raise ValueError("Timing is marked not requested but timing observations were supplied")
    timing_status = "not_requested" if not timing_requested else ("available" if valid_times.all() else "partial")

    if timings.empty:
        timings = timings.reindex(columns=["replication", "seed", "schedule", "H", "algorithm", "timing_repetition", "seconds", "batch_repeats"])
    records.to_csv(root / "data" / "raw" / "records.csv", index=False)
    timings.to_csv(root / "data" / "raw" / "timing_repetitions.csv", index=False)

    summary = (
        records.groupby(["schedule", "H", "algorithm"], as_index=False)
        .agg(
            replications=("replication", "nunique"),
            hit_rate=("hit", "mean"),
            median_E_infinity=("E_infinity", "median"),
            q25_E_infinity=("E_infinity", lambda x: x.quantile(0.25)),
            q75_E_infinity=("E_infinity", lambda x: x.quantile(0.75)),
            median_R_E=("R_E", "median"),
            median_N_P=("N_P", "median"),
            q25_N_P=("N_P", lambda x: x.quantile(0.25)),
            q75_N_P=("N_P", lambda x: x.quantile(0.75)),
            median_N_Phat=("N_Phat", "median"),
            median_N_factorizations=("N_factorizations", "median"),
            median_N_linear_solves=("N_linear_solves", "median"),
            median_wall_time_seconds=("wall_time_median_seconds", "median"),
            q25_wall_time_seconds=("wall_time_median_seconds", lambda x: x.quantile(0.25)),
            q75_wall_time_seconds=("wall_time_median_seconds", lambda x: x.quantile(0.75)),
            median_m=("m", "median"),
            median_h=("h", "median"),
            median_actual_spectral_radius=("actual_spectral_radius", "median"),
            median_actual_inf_norm=("actual_inf_norm", "median"),
        )
        .sort_values(["schedule", "H", "algorithm"])
    )
    summary.to_csv(derived / "summary_by_H.csv", index=False)

    # Bootstrap slopes across paired replications.
    slope_rows: list[dict] = []
    for schedule in SCHEDULE_ORDER:
        e = records[
            (records.schedule == schedule) & (records.algorithm == "IE-VI Joint")
        ].pivot(index="replication", columns="H", values="E_infinity")
        e = e.reindex(columns=H_GRID)
        for grid_name, mask in (
            ("all_H", np.ones(len(H_GRID), dtype=bool)),
            ("H_ge_80", H_GRID >= 80),
        ):
            fit = bootstrap_slope_details(
                e.to_numpy()[:, mask], H_GRID[mask].astype(float), 4000, 4400
            )
            slope_rows.append(
                {
                    "schedule": schedule,
                    "quantity": "E_infinity",
                    "grid": grid_name,
                    **fit,
                }
            )
        for algorithm in ALGORITHM_ORDER:
            q = records[
                (records.schedule == schedule) & (records.algorithm == algorithm)
            ].pivot(index="replication", columns="H", values="N_P")
            q = q.reindex(columns=H_GRID)
            for grid_name, mask in (
                ("all_H", np.ones(len(H_GRID), dtype=bool)),
                ("H_ge_80", H_GRID >= 80),
            ):
                fit = bootstrap_slope_details(
                    q.to_numpy()[:, mask], H_GRID[mask].astype(float), 4000, 4500
                )
                slope_rows.append(
                    {
                        "schedule": schedule,
                        "quantity": f"N_P::{algorithm}",
                        "grid": grid_name,
                        **fit,
                    }
                )
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(derived / "slope_summary.csv", index=False)

    hmax = int(H_GRID.max())
    endpoint = summary[summary.H == hmax].copy()
    vi = endpoint[endpoint.algorithm == "VI"][["schedule", "median_N_P", "median_wall_time_seconds"]].rename(
        columns={"median_N_P": "VI_median_N_P", "median_wall_time_seconds": "VI_median_wall_time_seconds"}
    )
    endpoint = endpoint.merge(vi, on="schedule", how="left")
    endpoint["VI_over_algorithm_N_P"] = endpoint.VI_median_N_P / endpoint.median_N_P
    endpoint["VI_over_algorithm_wall_time"] = endpoint.VI_median_wall_time_seconds / endpoint.median_wall_time_seconds
    endpoint.to_csv(derived / f"endpoint_H{hmax}.csv", index=False)

    # Primitive-count table at the actual maximum selected horizon.
    primitive_cols = [
        "schedule", "algorithm", "median_N_P", "median_N_Phat",
        "median_N_factorizations", "median_N_linear_solves",
        "median_wall_time_seconds", "VI_over_algorithm_N_P",
        "VI_over_algorithm_wall_time", "hit_rate", "median_m",
    ]
    endpoint[primitive_cols].to_csv(derived / f"endpoint_cost_table_H{hmax}.csv", index=False)

    # Per-run timing dispersion.
    if len(timings) > 0:
        timing_summary = (
            timings.groupby(["replication", "schedule", "H", "algorithm"], as_index=False)
            .agg(
                timing_repetitions=("timing_repetition", "count"),
                median_seconds=("seconds", "median"),
                mad_seconds=("seconds", mad),
                min_seconds=("seconds", "min"),
                max_seconds=("seconds", "max"),
                batch_repeats=("batch_repeats", "max"),
            )
        )
    else:
        timing_summary = pd.DataFrame(
            columns=[
                "replication", "schedule", "H", "algorithm",
                "timing_repetitions", "median_seconds", "mad_seconds",
                "min_seconds", "max_seconds", "batch_repeats",
            ]
        )
    timing_summary.to_csv(derived / "timing_summary_per_run.csv", index=False)

    result = {
        "records": int(len(records)),
        "timing_records": int(len(timings)),
        "replications": int(records.replication.nunique()),
        "H_grid": [int(x) for x in H_GRID],
        "schedules": list(SCHEDULE_ORDER),
        "algorithms": list(ALGORITHM_ORDER),
        "all_hit": bool(records.hit.all()),
        "endpoint_H": hmax,
        "endpoint": endpoint.to_dict(orient="records"),
        "endpoint_file": f"endpoint_H{hmax}.csv",
        "timing_status": timing_status,
        "slopes": slopes.to_dict(orient="records"),
    }
    if hmax == 640:
        result["endpoint_H640"] = result["endpoint"]  # Full-grid consumer compatibility.
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    result = json_safe(result)
    (derived / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    for d in (
        root / "data" / "raw",
        root / "data" / "derived",
        root / "logs",
        root / "environment",
        root / "verification",
    ):
        d.mkdir(parents=True, exist_ok=True)

    capture_environment(root, args)
    log_path = root / "logs" / "run.log"

    with log_path.open("w", buffering=1) as log, threadpool_limits(limits=1):
        def note(msg: str) -> None:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            line = f"[{stamp}] {msg}"
            print(line, flush=True)
            print(line, file=log, flush=True)

        note("Starting matched policy-evaluation horizon study")
        note(json.dumps(vars(args) | {"output_root": str(root)}, default=str))

        P_full, R_full = build_modified_cliffwalk()
        _, policy = solve_control_policy_iteration(P_full, R_full, 1.0 - 1.0 / 640.0)
        rows = np.arange(P_full.shape[0])
        P = P_full[rows, policy, :]
        r = R_full[rows, policy]
        policy_hash = matrix_sha256(policy.astype(np.int64))
        note(f"Fixed policy SHA256={policy_hash}")

        targets = np.stack(
            [np.linalg.solve(np.eye(P.shape[0]) - (1.0 - 1.0 / H) * P, r) for H in H_GRID],
            axis=0,
        )
        target_residuals = []
        for H, target_value in zip(H_GRID, targets):
            gamma = 1.0 - 1.0 / float(H)
            residual = np.max(np.abs(target_value - (r + gamma * (P @ target_value))))
            target_residuals.append(float(residual))
            if residual > 1e-10:
                raise AssertionError((H, residual))

        models, seeds = make_models(P, args.replications, args.base_seed)
        np.savez_compressed(
            root / "data" / "raw" / "experiment_inputs.npz",
            P_policy=P,
            r_policy=r,
            fixed_policy=policy,
            targets=targets,
            H_grid=H_GRID,
            schedules=np.asarray(SCHEDULE_ORDER),
            surrogate_models=models,
            seeds=seeds,
        )
        note(f"Generated and archived {models.shape[0] * models.shape[1] * models.shape[2]} surrogate models")

        records: list[dict] = []
        timing_rows: list[dict] = []
        start = time.perf_counter()

        for rep in range(args.replications):
            for hi, H_np in enumerate(H_GRID):
                H = int(H_np)
                gamma = 1.0 - 1.0 / H
                target_value = targets[hi]
                for si, schedule in enumerate(SCHEDULE_ORDER):
                    P_hat = models[rep, si, hi]
                    n = n_for(schedule, H)
                    E_inf = model_error(P, P_hat, gamma)
                    R_E = E_inf / (1.0 - gamma)
                    model_hash = matrix_sha256(P_hat)

                    # Parameters and exact PE error matrices.
                    h_eq, m_eq, mu_eq, rbar_eq = eq16_parameters(gamma, E_inf)
                    h_joint, m_joint, mu_joint, lambda_joint = theorem27_parameters(gamma, E_inf)
                    G_os = osvi_error_matrix(P, P_hat, gamma)
                    M_eq = ievi_error_matrix(P, P_hat, gamma, h_eq, m_eq)
                    M_joint = ievi_error_matrix(P, P_hat, gamma, h_joint, m_joint)

                    specs = [
                        (
                            "VI",
                            math.nan,
                            1,
                            math.nan,
                            gamma,
                            float(np.linalg.norm(gamma * P, ord=np.inf)),
                            lambda: count_vi(P, r, gamma, target_value, args.target, args.stable_tail, args.max_true_applications),
                        ),
                        (
                            "OS-VI",
                            math.inf,
                            1,
                            R_E,
                            spectral_radius(G_os),
                            float(np.linalg.norm(G_os, ord=np.inf)),
                            lambda: count_osvi(P, P_hat, r, gamma, target_value, args.target, args.stable_tail, args.max_true_applications),
                        ),
                        (
                            "IE-VI Eq16",
                            h_eq,
                            m_eq,
                            mu_eq,
                            spectral_radius(M_eq),
                            float(np.linalg.norm(M_eq, ord=np.inf)),
                            lambda: count_ievi(P, P_hat, r, gamma, h_eq, m_eq, target_value, args.target, args.stable_tail, args.max_true_applications),
                        ),
                        (
                            "IE-VI Joint",
                            h_joint,
                            m_joint,
                            mu_joint,
                            spectral_radius(M_joint),
                            float(np.linalg.norm(M_joint, ord=np.inf)),
                            lambda: count_ievi(P, P_hat, r, gamma, h_joint, m_joint, target_value, args.target, args.stable_tail, args.max_true_applications),
                        ),
                    ]

                    for algorithm, h, m, mu, actual_sr, actual_inf, counter in specs:
                        result = counter()
                        if not result.hit:
                            note(f"WARNING no hit: rep={rep} H={H} schedule={schedule} alg={algorithm}")
                        q = result.stable_hit_true_applications

                        if algorithm == "VI":
                            N_P = q
                            N_Phat = 0
                            N_fact = 0
                            N_lin = 0
                            timed_fn = lambda P=P, r=r, gamma=gamma, q=q: execute_vi(P, r, gamma, q)
                        elif algorithm == "OS-VI":
                            N_P = q
                            N_Phat = q
                            N_fact = 1
                            N_lin = q
                            timed_fn = lambda P=P, P_hat=P_hat, r=r, gamma=gamma, q=q: execute_osvi(P, P_hat, r, gamma, q)
                        else:
                            N_P = q
                            N_Phat = q
                            N_fact = 1
                            N_lin = q
                            timed_fn = lambda P=P, P_hat=P_hat, r=r, gamma=gamma, h=h, m=m, q=q: execute_ievi(P, P_hat, r, gamma, h, m, q)

                        timing_samples: list[float] = []
                        batch_repeats = 0
                        if result.hit and not args.skip_timing:
                            timing_samples, batch_repeats = benchmark_callable(
                                timed_fn,
                                args.timing_repetitions,
                                args.timing_batch_target_seconds,
                            )
                            timed_value = timed_fn()
                            timed_error = relative_linf(timed_value, target_value)
                            if timed_error > args.target * (1.0 + 1e-10):
                                raise AssertionError(("Timed rerun missed target", rep, H, schedule, algorithm, timed_error, q))
                            for tr, seconds in enumerate(timing_samples):
                                timing_rows.append(
                                    {
                                        "replication": rep,
                                        "seed": int(seeds[rep]),
                                        "schedule": schedule,
                                        "H": H,
                                        "algorithm": algorithm,
                                        "timing_repetition": tr,
                                        "seconds": seconds,
                                        "batch_repeats": batch_repeats,
                                    }
                                )

                        med_time = float(np.median(timing_samples)) if timing_samples else math.nan
                        mad_time = (
                            float(np.median(np.abs(np.asarray(timing_samples) - med_time)))
                            if timing_samples
                            else math.nan
                        )

                        records.append(
                            {
                                "replication": rep,
                                "seed": int(seeds[rep]),
                                "schedule": schedule,
                                "H": H,
                                "gamma": gamma,
                                "n_per_policy_row": n,
                                "offline_transition_samples": n * P.shape[0],
                                "algorithm": algorithm,
                                "target_relative_linf": args.target,
                                "stable_tail_outer_updates": args.stable_tail,
                                "E_infinity": E_inf,
                                "R_E": R_E,
                                "h": h,
                                "m": m,
                                "mu_envelope": mu,
                                "actual_spectral_radius": actual_sr,
                                "actual_inf_norm": actual_inf,
                                "hit": result.hit,
                                "stable_hit_true_applications": result.stable_hit_true_applications,
                                "stable_hit_outer_iterations": result.stable_hit_outer_iterations,
                                "verification_true_applications": result.verification_true_applications,
                                "verification_outer_iterations": result.verification_outer_iterations,
                                "error_at_hit": result.error_at_hit,
                                "error_after_stable_tail": result.error_after_stable_tail,
                                "N_P": N_P,
                                "N_Phat": N_Phat,
                                "N_factorizations": N_fact,
                                "N_linear_solves": N_lin,
                                "wall_time_median_seconds": med_time,
                                "wall_time_mad_seconds": mad_time,
                                "timing_batch_repeats": batch_repeats,
                                "surrogate_model_sha256": model_hash,
                                "fixed_policy_sha256": policy_hash,
                            }
                        )
            note(f"Completed replication {rep + 1}/{args.replications}; elapsed={time.perf_counter() - start:.1f}s")

        records_df = pd.DataFrame(records)
        timings_df = pd.DataFrame(timing_rows)
        result_summary = aggregate_results(root, records_df, timings_df, timing_requested=not args.skip_timing)

        # Verification report.
        verification = {
            "records_expected": args.replications * len(H_GRID) * len(SCHEDULE_ORDER) * len(ALGORITHM_ORDER),
            "records_observed": int(len(records_df)),
            "all_algorithms_executed_per_condition": bool(
                records_df.groupby(["replication", "schedule", "H"]).algorithm.nunique().eq(len(ALGORITHM_ORDER)).all()
            ),
            "all_hits": bool(records_df.hit.all()),
            "all_error_at_hit_within_target": bool(
                (records_df.loc[records_df.hit, "error_at_hit"] <= args.target * (1 + 1e-10)).all()
            ),
            "all_stable_tail_errors_within_target": bool(
                (records_df.loc[records_df.hit, "error_after_stable_tail"] <= args.target * (1 + 1e-10)).all()
            ),
            "all_true_query_counts_positive": bool((records_df.N_P > 0).all()),
            "cost_identities": {
                "VI": bool(
                    (
                        (records_df[records_df.algorithm == "VI"].N_Phat == 0)
                        & (records_df[records_df.algorithm == "VI"].N_factorizations == 0)
                        & (records_df[records_df.algorithm == "VI"].N_linear_solves == 0)
                    ).all()
                ),
                "OS_VI": bool(
                    (
                        (records_df[records_df.algorithm == "OS-VI"].N_P == records_df[records_df.algorithm == "OS-VI"].N_Phat)
                        & (records_df[records_df.algorithm == "OS-VI"].N_P == records_df[records_df.algorithm == "OS-VI"].N_linear_solves)
                        & (records_df[records_df.algorithm == "OS-VI"].N_factorizations == 1)
                    ).all()
                ),
                "IE_VI": bool(
                    (
                        (records_df[records_df.algorithm.str.startswith("IE-VI")].N_P == records_df[records_df.algorithm.str.startswith("IE-VI")].N_Phat)
                        & (records_df[records_df.algorithm.str.startswith("IE-VI")].N_P == records_df[records_df.algorithm.str.startswith("IE-VI")].N_linear_solves)
                        & (records_df[records_df.algorithm.str.startswith("IE-VI")].N_factorizations == 1)
                    ).all()
                ),
            },
            "target_bellman_residuals": target_residuals,
            "row_stochastic_true": bool(np.allclose(P.sum(axis=1), 1.0, atol=1e-14)),
            "row_stochastic_surrogates": bool(np.allclose(models.sum(axis=-1), 1.0, atol=1e-14)),
            "result_summary": result_summary,
        }
        verification["passed"] = bool(
            verification["records_expected"] == verification["records_observed"]
            and verification["all_algorithms_executed_per_condition"]
            and verification["all_hits"]
            and verification["all_error_at_hit_within_target"]
            and verification["all_stable_tail_errors_within_target"]
            and verification["all_true_query_counts_positive"]
            and all(verification["cost_identities"].values())
            and verification["row_stochastic_true"]
            and verification["row_stochastic_surrogates"]
        )
        (root / "verification" / "verification_report.json").write_text(
            json.dumps(verification, indent=2) + "\n"
        )
        if not verification["passed"]:
            raise RuntimeError("Verification failed; inspect verification_report.json")
        note("All verification checks passed")
        note(f"Total elapsed={time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
