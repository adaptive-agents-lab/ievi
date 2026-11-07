"""Shared utilities for the IE-VI common-solver control study.

The comparison uses one common surrogate value-iteration
solver for OS-VI and practical IE-VI. The solver terminates by a certified
absolute L-infinity value-error target:

    ||T_hat U - U||_inf <= (1-beta) * delta_solve.

Every true/surrogate operator application is counted at the call site.
Offline policy evaluation is diagnostic and excluded from algorithm cost.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Policy = NDArray[np.int32]

LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.625, 0.65, 0.675,
           0.7, 0.725, 0.75, 0.775, 0.8, 0.85, 0.9, 0.95, 1.0]
ANCHOR_LAMBDAS = [0.0, 0.7, 1.0]
GAMMA = 0.99
POLICY_THRESHOLD = 1e-3
TRUE_BUDGET = 1000
DIVERGENCE_MULT = 1e4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_modified_cliffwalk(gamma: float = GAMMA) -> tuple[Array, Array]:
    del gamma
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


def support_uniform_surrogate(P: Array, lam: float) -> Array:
    out = np.zeros_like(P)
    for s in range(P.shape[0]):
        for a in range(P.shape[1]):
            support = np.flatnonzero(P[s, a] > 0)
            uniform = np.zeros(P.shape[2], dtype=np.float64)
            uniform[support] = 1.0 / len(support)
            out[s, a] = (1.0 - lam) * P[s, a] + lam * uniform
    return out


def evaluate_policy(P: Array, R: Array, gamma: float, policy: Policy) -> Array:
    rows = np.arange(P.shape[0])
    return np.linalg.solve(
        np.eye(P.shape[0], dtype=np.float64) - gamma * P[rows, policy, :],
        R[rows, policy],
    )


def relative_l1(value: Array, target: Array) -> float:
    return float(np.sum(np.abs(value - target)) / (np.sum(np.abs(target)) + 1e-12))


def true_bellman_residual(P: Array, R: Array, gamma: float, value: Array,
                          pv: Array | None = None) -> tuple[float, Array]:
    if pv is None:
        pv = np.einsum("sak,k->sa", P, value)
    tv = np.max(R + gamma * pv, axis=1)
    return float(np.max(np.abs(tv - value))), pv


class PolicyErrorCache:
    def __init__(self, P: Array, R: Array, gamma: float, target: Array):
        self.P, self.R, self.gamma, self.target = P, R, gamma, target
        self._cache: dict[bytes, float] = {}

    def error(self, policy: Policy) -> float:
        key = np.asarray(policy, dtype=np.int32).tobytes()
        if key not in self._cache:
            v_pi = evaluate_policy(self.P, self.R, self.gamma, np.asarray(policy, dtype=np.int32))
            self._cache[key] = relative_l1(v_pi, self.target)
        return self._cache[key]

    @property
    def unique_policies(self) -> int:
        return len(self._cache)


@dataclass
class Counters:
    n_p: int = 0
    n_phat: int = 0
    direct_solves: int = 0
    factorizations: int = 0
    pi_rounds: int = 0
    surrogate_vi_sweeps: int = 0
    attempts: int = 0
    accepted: int = 0
    rejected: int = 0
    inner_failures: int = 0

    def copy_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class InnerSolveResult:
    value: Array
    policy: Policy
    sweeps: int
    new_phat_apps: int
    residual: float
    value_error_bound: float
    converged: bool


def certified_surrogate_vi_control(
    P_hat: Array,
    rewards: Array,
    beta: float,
    init_value: Array,
    delta_solve: float,
    *,
    initial_phat_value: Array | None = None,
    max_sweeps: int = 5000,
) -> InnerSolveResult:
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must be in [0,1), got {beta}")
    if delta_solve <= 0:
        raise ValueError("delta_solve must be positive")
    tol_resid = (1.0 - beta) * delta_solve
    u = np.asarray(init_value, dtype=np.float64).copy()
    new_apps = 0
    last_q: Array | None = None
    last_resid = math.inf
    for sweep in range(1, max_sweeps + 1):
        if sweep == 1 and initial_phat_value is not None:
            phu = np.asarray(initial_phat_value, dtype=np.float64)
        else:
            phu = np.einsum("sak,k->sa", P_hat, u)
            new_apps += 1
        q = rewards + beta * phu
        u_new = np.max(q, axis=1)
        resid = float(np.max(np.abs(u_new - u)))
        last_q = q
        last_resid = resid
        if resid <= tol_resid:
            policy = np.argmax(q, axis=1).astype(np.int32)
            return InnerSolveResult(
                value=u,
                policy=policy,
                sweeps=sweep,
                new_phat_apps=new_apps,
                residual=resid,
                value_error_bound=resid / max(1.0 - beta, 1e-300),
                converged=True,
            )
        u = u_new
    if last_q is None:
        last_q = rewards + beta * np.einsum("sak,k->sa", P_hat, u)
        new_apps += 1
        last_resid = float(np.max(np.abs(np.max(last_q, axis=1) - u)))
    return InnerSolveResult(
        value=u,
        policy=np.argmax(last_q, axis=1).astype(np.int32),
        sweeps=max_sweeps,
        new_phat_apps=new_apps,
        residual=last_resid,
        value_error_bound=last_resid / max(1.0 - beta, 1e-300),
        converged=False,
    )


def stable_hit_index(errors: Iterable[float], threshold: float = POLICY_THRESHOLD) -> int | None:
    vals = list(float(x) for x in errors)
    if not vals:
        return None
    last_bad = None
    for i, e in enumerate(vals):
        if not (math.isfinite(e) and e < threshold):
            last_bad = i
    idx = 0 if last_bad is None else last_bad + 1
    return idx if idx < len(vals) else None


def first_hit_index(errors: Iterable[float], threshold: float = POLICY_THRESHOLD) -> int | None:
    for i, e in enumerate(errors):
        if math.isfinite(float(e)) and float(e) < threshold:
            return i
    return None


def summarize_timing(values: list[float]) -> dict[str, float | int | list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    return {
        "n": int(arr.size),
        "median_seconds": med,
        "mad_seconds": mad,
        "q1_seconds": float(q1),
        "q3_seconds": float(q3),
        "iqr_seconds": float(q3 - q1),
        "raw_seconds": [float(x) for x in arr],
    }


def environment_record() -> dict[str, Any]:
    rec: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "thread_env": {k: os.environ.get(k) for k in [
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "JAX_PLATFORMS", "XLA_FLAGS"
        ]},
    }
    try:
        import scipy
        rec["scipy"] = scipy.__version__
    except Exception:
        rec["scipy"] = None
    try:
        import jax
        rec["jax"] = jax.__version__
        rec["jax_devices"] = [str(x) for x in jax.devices()]
    except Exception:
        rec["jax"] = None
    return rec


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
