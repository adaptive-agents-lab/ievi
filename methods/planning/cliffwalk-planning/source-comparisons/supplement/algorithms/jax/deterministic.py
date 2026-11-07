"""JAX-backed deterministic PE and control implementations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import math
from typing import Any, List, Tuple

import jax
import jax.numpy as jnp

from ...core.types import MDP


DEFAULT_TARGET_MAX_CONTROL_ITERS = 500000


def _as_jax(array: Any, *, dtype: Any | None = None) -> jnp.ndarray:
    return jnp.asarray(array, dtype=dtype)


def _value_norm(vec: jnp.ndarray, *, norm_name: str) -> float:
    key = norm_name.lower()
    if key == "l2":
        return float(jnp.linalg.norm(vec))
    if key in ("inf", "linf"):
        return float(jnp.max(jnp.abs(vec)))
    return float(jnp.sum(jnp.abs(vec)))


def _compute_value_error(
    v: Any,
    v_true: Any,
    *,
    norm_name: str = "L1",
    normalized: bool = True,
    eps: float = 1e-12,
) -> float:
    v_arr = _as_jax(v, dtype=jnp.float64)
    v_true_arr = _as_jax(v_true, dtype=jnp.float64)
    diff = v_arr - v_true_arr
    num = _value_norm(diff, norm_name=norm_name)
    if not normalized:
        return num
    denom = _value_norm(v_true_arr, norm_name=norm_name) + eps
    return num / denom


def _is_finite_scalar(value: Any) -> bool:
    return math.isfinite(float(value))


def _model_error_norm(p: Any, p_hat: Any, *, norm_name: str = "Linf") -> float:
    p_arr = _as_jax(p, dtype=jnp.float64)
    p_hat_arr = _as_jax(p_hat, dtype=jnp.float64)
    diff = p_hat_arr - p_arr
    key = norm_name.lower()

    if key in ("l2", "spectral"):
        svals = jnp.linalg.svd(diff.reshape(-1, diff.shape[-1]), compute_uv=False)
        return float(jnp.max(svals))
    if key in ("l1", "col_sum", "column_sum"):
        mat = diff.reshape(-1, diff.shape[-1])
        return float(jnp.max(jnp.sum(jnp.abs(mat), axis=0)))
    if key in ("entry_inf", "entryinf", "maxabs"):
        return float(jnp.max(jnp.abs(diff)))
    if key in ("fro", "frob", "frobenius"):
        return float(jnp.linalg.norm(diff))

    mat = diff.reshape(-1, diff.shape[-1])
    return float(jnp.max(jnp.sum(jnp.abs(mat), axis=1)))


def _get_bellman_residual_norm(
    v: Any,
    p_pi: Any,
    r_pi: Any,
    gamma: float,
    *,
    norm_name: str = "Linf",
) -> float:
    v_arr = _as_jax(v, dtype=jnp.float64)
    p_pi_arr = _as_jax(p_pi, dtype=jnp.float64)
    r_pi_arr = _as_jax(r_pi, dtype=jnp.float64)
    br = r_pi_arr + gamma * (p_pi_arr @ v_arr) - v_arr
    return _compute_value_error(br, jnp.zeros_like(br), norm_name=norm_name, normalized=False)


import numpy as np


def _get_bellman_residual_norm_high_precision(
    v: Any,
    p_pi: Any,
    r_pi: Any,
    gamma: float,
) -> float | None:
    try:
        v_arr = np.asarray(v, dtype=np.longdouble)
        p_pi_arr = np.asarray(p_pi, dtype=np.longdouble)
        r_pi_arr = np.asarray(r_pi, dtype=np.longdouble)
        gamma_arr = np.longdouble(str(float(gamma)))
        br = r_pi_arr + gamma_arr * (p_pi_arr @ v_arr) - v_arr
        return float(np.max(np.abs(br)))
    except Exception:
        return None


def _solve_linear(mat: Any, rhs: Any) -> jnp.ndarray:
    mat_arr = _as_jax(mat, dtype=jnp.float64)
    rhs_arr = _as_jax(rhs, dtype=jnp.float64)
    try:
        return jnp.linalg.solve(mat_arr, rhs_arr)
    except Exception:
        return jnp.linalg.lstsq(mat_arr, rhs_arr, rcond=None)[0]


def _decimal_precision_digits(residual_tolerance: float) -> int:
    return max(50, int(math.ceil(-math.log10(max(residual_tolerance, 1e-30)))) + 24)


def _to_decimal(value: Any) -> Decimal:
    return Decimal(str(float(value)))


def _solve_linear_decimal(mat: Any, rhs: Any, *, precision_digits: int) -> list[Decimal]:
    mat_arr = np.asarray(mat, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    n = int(rhs_arr.shape[0])
    with localcontext() as ctx:
        ctx.prec = precision_digits
        augmented = [
            [_to_decimal(mat_arr[i, j]) for j in range(n)] + [_to_decimal(rhs_arr[i])]
            for i in range(n)
        ]
        for pivot_idx in range(n):
            pivot_row = max(range(pivot_idx, n), key=lambda row: abs(augmented[row][pivot_idx]))
            pivot_value = augmented[pivot_row][pivot_idx]
            if pivot_value == 0:
                raise ValueError("Singular matrix in decimal policy-value fallback.")
            if pivot_row != pivot_idx:
                augmented[pivot_idx], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_idx]
            pivot_value = augmented[pivot_idx][pivot_idx]
            for col in range(pivot_idx, n + 1):
                augmented[pivot_idx][col] /= pivot_value
            for row in range(n):
                if row == pivot_idx:
                    continue
                factor = augmented[row][pivot_idx]
                if factor == 0:
                    continue
                for col in range(pivot_idx, n + 1):
                    augmented[row][col] -= factor * augmented[pivot_idx][col]
        return [augmented[row][n] for row in range(n)]


def _get_bellman_residual_norm_decimal(
    v_decimal: list[Decimal],
    p_pi: Any,
    r_pi: Any,
    gamma: float,
    *,
    precision_digits: int,
) -> float:
    p_pi_arr = np.asarray(p_pi, dtype=np.float64)
    r_pi_arr = np.asarray(r_pi, dtype=np.float64)
    with localcontext() as ctx:
        ctx.prec = precision_digits
        gamma_decimal = _to_decimal(gamma)
        residual_max = Decimal("0")
        for state_idx in range(len(v_decimal)):
            bellman_value = _to_decimal(r_pi_arr[state_idx])
            transition_sum = Decimal("0")
            for next_state_idx, prob in enumerate(p_pi_arr[state_idx]):
                transition_sum += _to_decimal(prob) * v_decimal[next_state_idx]
            bellman_value += gamma_decimal * transition_sum
            residual_max = max(residual_max, abs(bellman_value - v_decimal[state_idx]))
        return float(residual_max)


def _control_q(p: jnp.ndarray, r: jnp.ndarray, gamma: float, v: jnp.ndarray) -> jnp.ndarray:
    return r + gamma * jnp.einsum("sak,k->sa", p, v)


def bellman_residual_tolerance(gamma: float, epsilon_target: float = 1e-8) -> float:
    gamma_value = float(gamma)
    epsilon_value = float(epsilon_target)
    if not 0.0 <= gamma_value < 1.0:
        raise ValueError(f"gamma must be in [0, 1), got {gamma_value}")
    if epsilon_value <= 0.0:
        raise ValueError(f"epsilon_target must be positive, got {epsilon_value}")
    return (1.0 - gamma_value) * epsilon_value


def value_iteration_control(
    mdp: MDP, tol: float = 1e-10, max_iters: int = 1000
) -> Tuple[jax.Array, jax.Array]:
    p = _as_jax(mdp.P, dtype=jnp.float64)
    r = _as_jax(mdp.R, dtype=jnp.float64)
    v = jnp.zeros(p.shape[0], dtype=jnp.float64)
    q = _control_q(p, r, mdp.gamma, v)

    for _ in range(max_iters):
        q = _control_q(p, r, mdp.gamma, v)
        v_new = jnp.max(q, axis=1)
        if float(jnp.max(jnp.abs(v_new - v))) < tol:
            v = v_new
            break
        v = v_new

    return v, jnp.argmax(q, axis=1).astype(jnp.int32)


def certify_optimal_control_target(
    mdp: MDP,
    *,
    epsilon_target: float = 1e-8,
    max_iters: int = DEFAULT_TARGET_MAX_CONTROL_ITERS,
) -> dict[str, Any]:
    p = _as_jax(mdp.P, dtype=jnp.float64)
    r = _as_jax(mdp.R, dtype=jnp.float64)
    residual_tolerance = bellman_residual_tolerance(mdp.gamma, epsilon_target)
    v = jnp.zeros(p.shape[0], dtype=jnp.float64)
    policy = jnp.zeros(p.shape[0], dtype=jnp.int32)
    residual = math.inf
    greedy_policy_stable = False

    for iteration in range(1, max_iters + 1):
        q = _control_q(p, r, mdp.gamma, v)
        v = jnp.max(q, axis=1)
        policy = jnp.argmax(q, axis=1).astype(jnp.int32)
        q_followup = _control_q(p, r, mdp.gamma, v)
        t_v = jnp.max(q_followup, axis=1)
        residual = float(jnp.max(jnp.abs(t_v - v)))
        followup_policy = jnp.argmax(q_followup, axis=1).astype(jnp.int32)
        greedy_policy_stable = bool(jnp.array_equal(policy, followup_policy))
        if residual <= residual_tolerance and greedy_policy_stable:
            return {
                "value": v,
                "policy": policy,
                "certificate": {
                    "certified": True,
                    "stop_reason": "certified",
                    "iterations": iteration,
                    "residual_norm": residual,
                    "residual_tolerance": residual_tolerance,
                    "value_error_bound": residual / max(1.0 - float(mdp.gamma), 1e-15),
                    "greedy_policy_stable": True,
                    "epsilon_target": float(epsilon_target),
                    "max_iters": int(max_iters),
                },
            }

    return {
        "value": v,
        "policy": policy,
        "certificate": {
            "certified": False,
            "stop_reason": "iteration_cap",
            "iterations": int(max_iters),
            "residual_norm": residual,
            "residual_tolerance": residual_tolerance,
            "value_error_bound": residual / max(1.0 - float(mdp.gamma), 1e-15),
            "greedy_policy_stable": greedy_policy_stable,
            "epsilon_target": float(epsilon_target),
            "max_iters": int(max_iters),
        },
    }


def build_policy_matrices(
    mdp: MDP, pi: Any, p_model: Any | None = None
) -> Tuple[jax.Array, jax.Array]:
    p_base = _as_jax(p_model if p_model is not None else mdp.P, dtype=jnp.float64)
    r = _as_jax(mdp.R, dtype=jnp.float64)
    pi_arr = _as_jax(pi, dtype=jnp.int32)
    rows = jnp.arange(p_base.shape[0], dtype=jnp.int32)
    p_pi = p_base[rows, pi_arr, :]
    r_pi = r[rows, pi_arr]
    return p_pi, r_pi


def solve_value(p_pi: Any, r_pi: Any, gamma: float) -> jax.Array:
    p_arr = _as_jax(p_pi, dtype=jnp.float64)
    r_arr = _as_jax(r_pi, dtype=jnp.float64)
    identity = jnp.eye(p_arr.shape[0], dtype=jnp.float64)
    return _solve_linear(identity - gamma * p_arr, r_arr)


def certify_policy_value_target(
    mdp: MDP,
    pi: Any,
    *,
    epsilon_target: float = 1e-8,
    max_refinements: int = 3,
) -> dict[str, Any]:
    p_pi, r_pi = build_policy_matrices(mdp, pi, mdp.P)
    p_pi_arr = _as_jax(p_pi, dtype=jnp.float64)
    r_pi_arr = _as_jax(r_pi, dtype=jnp.float64)
    identity = jnp.eye(p_pi_arr.shape[0], dtype=jnp.float64)
    system_matrix = identity - mdp.gamma * p_pi_arr
    residual_tolerance = bellman_residual_tolerance(mdp.gamma, epsilon_target)
    v = solve_value(p_pi_arr, r_pi_arr, mdp.gamma)
    residual = _get_bellman_residual_norm(v, p_pi_arr, r_pi_arr, mdp.gamma)
    float64_residual = residual
    refinements = 0
    decimal_precision_digits = None
    decimal_value_strings = None

    while residual > residual_tolerance and refinements < max_refinements:
        residual_vec = r_pi_arr + mdp.gamma * (p_pi_arr @ v) - v
        correction = _solve_linear(system_matrix, residual_vec)
        v = v + correction
        refinements += 1
        residual = _get_bellman_residual_norm(v, p_pi_arr, r_pi_arr, mdp.gamma)
        float64_residual = residual

    residual_audit_backend = "jax_float64"
    if residual > residual_tolerance:
        audited_residual = _get_bellman_residual_norm_high_precision(v, p_pi_arr, r_pi_arr, mdp.gamma)
        if audited_residual is not None:
            residual = audited_residual
            residual_audit_backend = "numpy_longdouble"
    if residual > residual_tolerance:
        decimal_precision_digits = _decimal_precision_digits(residual_tolerance)
        decimal_solution = _solve_linear_decimal(system_matrix, r_pi_arr, precision_digits=decimal_precision_digits)
        decimal_residual = _get_bellman_residual_norm_decimal(
            decimal_solution,
            p_pi_arr,
            r_pi_arr,
            mdp.gamma,
            precision_digits=decimal_precision_digits,
        )
        if decimal_residual <= residual_tolerance:
            v = jnp.asarray(np.asarray([float(item) for item in decimal_solution], dtype=np.float64))
            residual = decimal_residual
            residual_audit_backend = "decimal"
            decimal_value_strings = [str(item) for item in decimal_solution]

    certified = residual <= residual_tolerance
    if certified and refinements == 0:
        stop_reason = "direct_solve"
    elif certified and residual_audit_backend == "decimal":
        stop_reason = "decimal_linear_solve"
    elif certified and residual_audit_backend == "numpy_longdouble":
        stop_reason = "longdouble_residual_audit"
    elif certified:
        stop_reason = "iterative_refinement"
    else:
        stop_reason = "residual_above_tolerance"

    return {
        "value": v,
        "p_pi": p_pi_arr,
        "r_pi": r_pi_arr,
        "certificate": {
            "certified": certified,
            "stop_reason": stop_reason,
            "iterations": refinements,
            "residual_norm": residual,
            "residual_norm_float64": float64_residual,
            "residual_tolerance": residual_tolerance,
            "value_error_bound": residual / max(1.0 - float(mdp.gamma), 1e-15),
            "greedy_policy_stable": None,
            "residual_audit_backend": residual_audit_backend,
            "decimal_precision_digits": decimal_precision_digits,
            "epsilon_target": float(epsilon_target),
            "max_refinements": int(max_refinements),
        },
        "value_decimal_strings": decimal_value_strings,
    }


def prepare_certified_deterministic_target(
    mdp: MDP,
    *,
    epsilon_target: float = 1e-8,
    max_control_iters: int = DEFAULT_TARGET_MAX_CONTROL_ITERS,
    max_value_refinements: int = 3,
) -> dict[str, Any]:
    control_target = certify_optimal_control_target(
        mdp,
        epsilon_target=epsilon_target,
        max_iters=max_control_iters,
    )
    if not control_target["certificate"]["certified"]:
        raise ValueError(
            "Failed to certify optimal-control target: "
            f"{control_target['certificate']['stop_reason']} "
            f"(residual={control_target['certificate']['residual_norm']:.3e}, "
            f"tolerance={control_target['certificate']['residual_tolerance']:.3e})."
        )

    policy_value_target = certify_policy_value_target(
        mdp,
        control_target["policy"],
        epsilon_target=epsilon_target,
        max_refinements=max_value_refinements,
    )
    if not policy_value_target["certificate"]["certified"]:
        raise ValueError(
            "Failed to certify fixed-policy target: "
            f"{policy_value_target['certificate']['stop_reason']} "
            f"(residual={policy_value_target['certificate']['residual_norm']:.3e}, "
            f"tolerance={policy_value_target['certificate']['residual_tolerance']:.3e})."
        )

    return {
        "pi_star": control_target["policy"],
        "v_star_control": control_target["value"],
        "v_pi_star": policy_value_target["value"],
        "v_pi_star_decimal": policy_value_target.get("value_decimal_strings"),
        "p_pi": policy_value_target["p_pi"],
        "r_pi": policy_value_target["r_pi"],
        "control_certificate": control_target["certificate"],
        "policy_value_certificate": policy_value_target["certificate"],
        "epsilon_target": float(epsilon_target),
    }


def vi_pe(
    p_pi: Any,
    r_pi: Any,
    gamma: float,
    v_true: Any,
    max_queries: int,
    value_norm: str = "L1",
    value_normalized: bool = True,
) -> Tuple[List[int], List[float]]:
    p_pi_arr = _as_jax(p_pi, dtype=jnp.float64)
    r_pi_arr = _as_jax(r_pi, dtype=jnp.float64)
    v_true_arr = _as_jax(v_true, dtype=jnp.float64)

    v = jnp.zeros_like(v_true_arr)
    queries = [0]
    errors = [_compute_value_error(v, v_true_arr, norm_name=value_norm, normalized=value_normalized)]
    for _ in range(max_queries):
        v = r_pi_arr + gamma * (p_pi_arr @ v)
        queries.append(queries[-1] + 1)
        errors.append(_compute_value_error(v, v_true_arr, norm_name=value_norm, normalized=value_normalized))
    return queries, errors


def osvi_pe(
    p_pi: Any,
    p_hat_pi: Any,
    r_pi: Any,
    gamma: float,
    v_true: Any,
    max_queries: int,
    value_norm: str = "L1",
    value_normalized: bool = True,
    solver: str = "direct",
    solver_kwargs: dict[str, float] | None = None,
) -> Tuple[List[int], List[float]]:
    p_pi_arr = _as_jax(p_pi, dtype=jnp.float64)
    p_hat_pi_arr = _as_jax(p_hat_pi, dtype=jnp.float64)
    r_pi_arr = _as_jax(r_pi, dtype=jnp.float64)
    v_true_arr = _as_jax(v_true, dtype=jnp.float64)
    solver_kwargs = solver_kwargs or {}

    n_s = p_pi_arr.shape[0]
    k_hat_inv = None
    if solver == "direct":
        try:
            k_hat = jnp.eye(n_s, dtype=jnp.float64) - gamma * p_hat_pi_arr
            k_hat_inv = _solve_linear(k_hat, jnp.eye(n_s, dtype=jnp.float64))
        except Exception:
            return [0], [math.nan]

    v = jnp.zeros_like(v_true_arr)
    queries = [0]
    errors = [_compute_value_error(v, v_true_arr, norm_name=value_norm, normalized=value_normalized)]
    for _ in range(max_queries):
        u_true = p_pi_arr @ v
        u_hat = p_hat_pi_arr @ v
        rhs = r_pi_arr + gamma * (u_true - u_hat)

        if solver == "direct":
            v = k_hat_inv @ rhs
        elif solver == "iterative":
            tol = float(solver_kwargs.get("tol", 1e-10))
            max_iter = int(solver_kwargs.get("max_iter", 100000))
            x = jnp.zeros_like(rhs)
            for _ in range(max_iter):
                x_new = rhs + gamma * (p_hat_pi_arr @ x)
                if float(jnp.max(jnp.abs(x_new - x))) <= tol:
                    x = x_new
                    break
                x = x_new
            v = x
        else:
            raise ValueError(f"Unknown solver: {solver}")

        queries.append(queries[-1] + 1)
        err = _compute_value_error(v, v_true_arr, norm_name=value_norm, normalized=value_normalized)
        if not _is_finite_scalar(err):
            break
        errors.append(err)
    return queries, errors


def besp_precompute(p_pi: Any, p_hat_pi: Any, r_pi: Any, gamma: float, m: int, h: float) -> dict[str, Any]:
    """Precompute the truncated BESP inverse for manuscript-aligned inner depth ``m``."""
    inner_iterations = _normalize_besp_inner_iterations(m)
    series_degree = inner_iterations - 1
    p_pi_arr = _as_jax(p_pi, dtype=jnp.float64)
    p_hat_pi_arr = _as_jax(p_hat_pi, dtype=jnp.float64)
    r_pi_arr = _as_jax(r_pi, dtype=jnp.float64)
    n_s = p_pi_arr.shape[0]

    a = gamma * p_pi_arr - jnp.eye(n_s, dtype=jnp.float64)
    a_hat = gamma * p_hat_pi_arr - jnp.eye(n_s, dtype=jnp.float64)
    delta = a_hat - a
    k_hat = jnp.eye(n_s, dtype=jnp.float64) - h * a_hat
    k_hat_inv = _solve_linear(k_hat, jnp.eye(n_s, dtype=jnp.float64))
    y = -h * (k_hat_inv @ delta)
    s_m = jnp.eye(n_s, dtype=jnp.float64)
    power = jnp.eye(n_s, dtype=jnp.float64)
    for _ in range(series_degree):
        power = power @ y
        s_m = s_m + power
    return {"A": a, "K_tilde_inv": s_m @ k_hat_inv, "r_pi": r_pi_arr}


def besp_pe(
    p_pi: Any,
    p_hat_pi: Any,
    r_pi: Any,
    gamma: float,
    v_true: Any,
    m: int,
    h: float,
    max_queries: int,
    value_norm: str = "L1",
    value_normalized: bool = True,
) -> Tuple[List[int], List[float]]:
    """Run BESP with manuscript-aligned inner depth ``m``."""
    inner_iterations = _normalize_besp_inner_iterations(m)
    v_true_arr = _as_jax(v_true, dtype=jnp.float64)
    try:
        pre = besp_precompute(p_pi, p_hat_pi, r_pi, gamma, inner_iterations, h)
    except Exception:
        return [0], [math.nan]

    a = pre["A"]
    k_tilde_inv = pre["K_tilde_inv"]
    r_pi_arr = pre["r_pi"]
    v = jnp.zeros_like(v_true_arr)
    queries = [0]
    errors = [_compute_value_error(v, v_true_arr, norm_name=value_norm, normalized=value_normalized)]
    while True:
        cost = inner_iterations
        if queries[-1] + cost > max_queries:
            break
        br = r_pi_arr + a @ v
        v = v + h * (k_tilde_inv @ br)
        queries.append(queries[-1] + cost)
        err = _compute_value_error(v, v_true_arr, norm_name=value_norm, normalized=value_normalized)
        if not _is_finite_scalar(err):
            break
        errors.append(err)
    return queries, errors


class _AdaptiveStepManager:
    def __init__(
        self,
        *,
        delta: float,
        h_init: float = 1.0,
        h_smoothing: float = 0.2,
        optimism: float = 1.0,
        dynamic_m: bool = True,
        fixed_m: int | None = None,
        min_h: float = 1e-9,
        max_h: float = 1e6,
    ):
        self.delta = float(delta)
        self.h = float(h_init)
        self.alpha = float(h_smoothing)
        self.optimism = float(optimism)
        self.dynamic_m = bool(dynamic_m)
        self.fixed_m = fixed_m
        self.min_h = float(min_h)
        self.max_h = float(max_h)
        self.history_h = [float(h_init)]
        self.history_m: list[int] = []
        self.history_E_est: list[float] = []
        self.m_continuous = float(fixed_m) if fixed_m is not None else 0.0

    def _norm(self, vec: Any) -> float:
        arr = _as_jax(vec, dtype=jnp.float64)
        return float(jnp.max(jnp.abs(arr)))

    def estimate_error(self, r_true: Any, r_shadow: Any, v_k: Any) -> float:
        diff = _as_jax(r_shadow, dtype=jnp.float64) - _as_jax(r_true, dtype=jnp.float64)
        numer = self._norm(diff)
        denom = self._norm(v_k) + 1e-6
        e_est = numer / denom
        self.history_E_est.append(e_est)
        return e_est

    def update(self, e_est: float) -> Tuple[float, int]:
        e_used = e_est / self.optimism
        denom = 2.0 * e_used - self.delta
        if denom < 1e-4:
            h_target = self.max_h
        else:
            h_target = 1.0 / denom
        self.h = (1.0 - self.alpha) * self.h + self.alpha * h_target
        self.h = max(self.min_h, min(self.h, self.max_h))
        self.history_h.append(self.h)

        if self.fixed_m is not None:
            m_val = int(self.fixed_m)
        elif self.dynamic_m:
            denom_delta = max(self.delta, 1e-12)
            re_loc = e_used / denom_delta
            arg = max(1.0, 12.0 * re_loc - 6.0)
            val_target = math.log2(arg)
            self.m_continuous = (1.0 - self.alpha) * self.m_continuous + self.alpha * val_target
            m_val = max(0, int(math.ceil(self.m_continuous)))
        else:
            m_val = 0

        self.history_m.append(m_val)
        return self.h, m_val


class Shadow_BESP:
    def __init__(
        self,
        p_pi: Any,
        p_hat_pi: Any,
        r_pi: Any,
        gamma: float,
        h_init: float = 1.0,
        h_smoothing: float = 0.2,
        eps: float = 1e-12,
    ):
        self.p_pi = _as_jax(p_pi, dtype=jnp.float64)
        self.p_hat_pi = _as_jax(p_hat_pi, dtype=jnp.float64)
        self.r_pi = _as_jax(r_pi, dtype=jnp.float64)
        self.gamma = gamma
        self.delta = 1.0 - gamma
        self.n_s = self.p_pi.shape[0]
        self.adapter = _AdaptiveStepManager(delta=self.delta, h_init=h_init, h_smoothing=h_smoothing)
        self.v = jnp.zeros(self.n_s, dtype=jnp.float64)
        self.total_queries = 0
        self.identity = jnp.eye(self.n_s, dtype=jnp.float64)
        self.eps = eps

    @property
    def history_h(self) -> list[float]:
        return self.adapter.history_h

    @property
    def history_m(self) -> list[int]:
        return self.adapter.history_m

    @property
    def history_E_est(self) -> list[float]:
        return self.adapter.history_E_est

    @property
    def h(self) -> float:
        return self.adapter.h

    @h.setter
    def h(self, value: float) -> None:
        self.adapter.h = value

    @property
    def V(self) -> jax.Array:
        return self.v

    @V.setter
    def V(self, value: Any) -> None:
        self.v = _as_jax(value, dtype=jnp.float64)

    def _solve_k_hat(self, rhs: Any, h: float) -> jnp.ndarray:
        mat = self.identity * (1.0 + h) - (h * self.gamma) * self.p_hat_pi
        return _solve_linear(mat, rhs)

    def step(self) -> Tuple[int, float]:
        p_v = self.p_pi @ self.v
        r_true = self.r_pi + self.gamma * p_v - self.v
        step_query_cost = 1

        p_hat_v = self.p_hat_pi @ self.v
        r_shadow = self.r_pi + self.gamma * p_hat_v - self.v

        e_est = self.adapter.estimate_error(r_true, r_shadow, self.v)
        h, m_k = self.adapter.update(e_est)
        step_query_cost += m_k

        try:
            x0 = self._solve_k_hat(r_true, h)
            current_term = x0
            d_update = jnp.array(x0, copy=True)
            non_stochastic_p_hat = self.gamma * self.p_hat_pi
            non_stochastic_p = self.gamma * self.p_pi

            for _ in range(m_k):
                delta_v = non_stochastic_p_hat @ current_term - non_stochastic_p @ current_term
                current_term = self._solve_k_hat(-h * delta_v, h)
                d_update = d_update + current_term

            self.v = self.v + h * d_update
        except Exception:
            return 0, 0.0

        self.total_queries += step_query_cost
        return step_query_cost, 0.0


def shadow_besp_pe(
    p_pi: Any,
    p_hat_pi: Any,
    r_pi: Any,
    gamma: float,
    v_true: Any,
    max_queries: int,
    value_norm: str = "L1",
    value_normalized: bool = True,
    target_rho: float = 0.95,
    smoothing: float = 0.1,
    gain: float = 0.2,
    h_init: float = 1.0,
) -> Tuple[List[int], List[float]]:
    del target_rho, smoothing, gain

    v_true_arr = _as_jax(v_true, dtype=jnp.float64)
    solver = Shadow_BESP(p_pi, p_hat_pi, r_pi, gamma, h_init=h_init, h_smoothing=0.2)

    queries = [0]
    errors = [_compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)]
    while solver.total_queries < max_queries:
        solver.step()
        queries.append(solver.total_queries)
        err = _compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)
        errors.append(err)
        if not _is_finite_scalar(err):
            break
    return queries, errors


def _discount_gap_depth_theory(delta: float, eps: float = 1e-12) -> int:
    safe_delta = max(float(delta), eps)
    return max(1, int(math.ceil(math.log2(8.0 / safe_delta))))


def _discount_gap_depth_pe_impl(delta: float, eps: float = 1e-12) -> int:
    """Compatibility helper returning the manuscript-aligned BESP inner depth."""
    return _discount_gap_depth_theory(delta, eps=eps)


def _normalize_besp_inner_iterations(m: int) -> int:
    inner_iterations = int(m)
    if inner_iterations < 1:
        raise ValueError("BESP m must be >= 1 and matches the manuscript inner-loop count.")
    return inner_iterations


def model_error_stats(p_pi: Any, p_hat_pi: Any, gamma: float, norm_name: str = "Linf", eps: float = 1e-12) -> Tuple[float, float, float]:
    delta = 1.0 - gamma
    scaled_p = gamma * _as_jax(p_pi, dtype=jnp.float64)
    scaled_p_hat = gamma * _as_jax(p_hat_pi, dtype=jnp.float64)
    e_star = _model_error_norm(scaled_p, scaled_p_hat, norm_name=norm_name)
    re = e_star / (delta + eps)
    return delta, e_star, re


def pick_besp_rho_half(p_pi: Any, p_hat_pi: Any, gamma: float, norm_name: str = "L1") -> tuple[int, float, float, float, float, float, str]:
    delta, e_star, re = model_error_stats(p_pi, p_hat_pi, gamma, norm_name=norm_name)
    denom = 2.0 * e_star - delta
    if denom > 0:
        h = 1.0 / denom
        rho = 0.5
        source = "rho=0.5"
    else:
        h = 1.0 / max(delta, 1e-12)
        rho = (h * e_star) / (1.0 + h * delta) if delta > 0 else 0.0
        source = "fallback_h=1/delta"
    return _discount_gap_depth_pe_impl(delta), h, delta, e_star, re, rho, source


def pick_isp_params_control(mdp: MDP, p_hat: Any) -> Tuple[float, int]:
    gamma = mdp.gamma
    delta = 1.0 - gamma
    p_true = _as_jax(mdp.P, dtype=jnp.float64)
    p_hat_arr = _as_jax(p_hat, dtype=jnp.float64)
    diff = gamma * (p_hat_arr - p_true)
    row_sums = jnp.sum(jnp.abs(diff), axis=2)
    e_star = float(jnp.max(row_sums))
    denom = 2.0 * e_star - delta
    if denom > 1e-12:
        h = 1.0 / denom
    else:
        h = 1.0 / max(delta, 1e-12)
    return h, _discount_gap_depth_theory(delta)


def abesp_pe(
    p_pi: Any,
    p_hat_pi: Any,
    r_pi: Any,
    gamma: float,
    v_true: Any,
    target_epsilon: float,
    max_total_queries: int,
    value_norm: str = "L1",
    value_normalized: bool = True,
    model_norm: str = "L1",
) -> Tuple[List[int], List[float]]:
    p_pi_arr = _as_jax(p_pi, dtype=jnp.float64)
    p_hat_pi_arr = _as_jax(p_hat_pi, dtype=jnp.float64)
    r_pi_arr = _as_jax(r_pi, dtype=jnp.float64)
    v_true_arr = _as_jax(v_true, dtype=jnp.float64)

    delta = 1.0 - gamma
    v0 = jnp.zeros_like(v_true_arr)
    b0 = _get_bellman_residual_norm(v0, p_pi_arr, r_pi_arr, gamma, norm_name=model_norm)
    t_thresh = target_epsilon * (delta / (1.0 + gamma)) * b0
    safe_eps = max(target_epsilon, 1e-16)
    k_prime = max(1.0, math.log(1.0 / safe_eps) + 2.0 * math.log((1.0 + gamma) / delta))
    m_i = _discount_gap_depth_pe_impl(delta)

    queries = [0]
    errors = [_compute_value_error(v0, v_true_arr, norm_name=value_norm, normalized=value_normalized)]
    r_hat = 1.0
    total_queries = 0

    while total_queries < max_total_queries:
        if total_queries > 0:
            queries.append(total_queries)
            errors.append(errors[0])

        denom = delta * (2.0 * r_hat - 1.0)
        h_i = 1.0 / delta if denom <= 1e-12 else 1.0 / denom
        n_i = int(math.ceil(4.0 * r_hat * k_prime))

        try:
            pre = besp_precompute(p_pi_arr, p_hat_pi_arr, r_pi_arr, gamma, m_i, h_i)
        except Exception:
            r_hat *= 2.0
            continue

        a_op = pre["A"]
        k_tilde_inv = pre["K_tilde_inv"]
        v = jnp.array(v0, copy=True)
        cost_per_iter = m_i

        for _ in range(n_i):
            if total_queries + cost_per_iter > max_total_queries:
                break
            br = r_pi_arr + a_op @ v
            v = v + h_i * (k_tilde_inv @ br)
            total_queries += cost_per_iter
            queries.append(total_queries)
            errors.append(_compute_value_error(v, v_true_arr, norm_name=value_norm, normalized=value_normalized))

        if total_queries >= max_total_queries:
            break

        b_i = _get_bellman_residual_norm(v, p_pi_arr, r_pi_arr, gamma, norm_name=model_norm)
        total_queries += 1
        if b_i <= t_thresh:
            break
        r_hat *= 2.0

    return queries, errors


class VI_Control:
    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self.n_s = mdp.P.shape[0]
        self.n_a = mdp.P.shape[1]
        self.p = _as_jax(mdp.P, dtype=jnp.float64)
        self.r = _as_jax(mdp.R, dtype=jnp.float64)

    def run(self, n_iter: int) -> Tuple[jax.Array, jax.Array]:
        if n_iter <= 0:
            return (
                jnp.zeros((0, self.n_s), dtype=jnp.float64),
                jnp.zeros((0, self.n_s), dtype=jnp.int32),
            )
        v = jnp.zeros(self.n_s, dtype=jnp.float64)
        v_trace: list[jax.Array] = []
        policy_trace: list[jax.Array] = []

        for _ in range(n_iter):
            v_trace.append(v)
            q = _control_q(self.p, self.r, self.mdp.gamma, v)
            policy = jnp.argmax(q, axis=1).astype(jnp.int32)
            policy_trace.append(policy)
            v = jnp.max(q, axis=1)

        return jnp.stack(v_trace, axis=0), jnp.stack(policy_trace, axis=0)

    def compute_bellman_residual_norm(self, v: Any) -> float:
        v_arr = _as_jax(v, dtype=jnp.float64)
        q = _control_q(self.p, self.r, self.mdp.gamma, v_arr)
        t_v = jnp.max(q, axis=1)
        return float(jnp.max(jnp.abs(t_v - v_arr)))


class MSVI_Control:
    def __init__(
        self,
        mdp: MDP,
        p_hat: Any,
        solver_caching: str | bool = "auto",
        surrogate_policy_iteration_steps: int | None = None,
    ):
        self.mdp = mdp
        self.p_hat = _as_jax(p_hat, dtype=jnp.float64)
        self.n_s = mdp.P.shape[0]
        self.n_a = mdp.P.shape[1]
        self.p = _as_jax(mdp.P, dtype=jnp.float64)
        self.r = _as_jax(mdp.R, dtype=jnp.float64)
        self.identity = jnp.eye(self.n_s, dtype=jnp.float64)
        self.rows = jnp.arange(self.n_s, dtype=jnp.int32)
        self.use_caching = self.n_s >= 200 if solver_caching == "auto" else bool(solver_caching)
        self.cache_policy = None
        self.cache_gamma = None
        self.cache_inv = None
        self.surrogate_policy_iteration_steps = (
            None
            if surrogate_policy_iteration_steps is None
            else max(int(surrogate_policy_iteration_steps), 1)
        )

    def run(self, n_iter: int) -> Tuple[jax.Array, jax.Array]:
        if n_iter <= 0:
            return (
                jnp.zeros((0, self.n_s), dtype=jnp.float64),
                jnp.zeros((0, self.n_s), dtype=jnp.int32),
            )
        v = jnp.zeros(self.n_s, dtype=jnp.float64)
        v_trace: list[jax.Array] = []
        policy_trace: list[jax.Array] = []

        current_policy = None
        for _ in range(n_iter):
            v_trace.append(v)
            correction = self.mdp.gamma * (_control_q(self.p, jnp.zeros_like(self.r), 1.0, v) - _control_q(self.p_hat, jnp.zeros_like(self.r), 1.0, v))
            r_k = self.r + correction
            v_new, policy = self._solve_surrogate_control(
                self.p_hat,
                r_k,
                self.mdp.gamma,
                init_policy=current_policy,
                surrogate_policy_iteration_steps=self.surrogate_policy_iteration_steps,
            )
            current_policy = policy
            policy_trace.append(policy)
            v = v_new

        return jnp.stack(v_trace, axis=0), jnp.stack(policy_trace, axis=0)

    def _solve_surrogate_control(
        self,
        p: Any,
        r: Any,
        gamma: float,
        tol: float = 1e-8,
        max_iter: int = 200,
        init_policy: Any | None = None,
        surrogate_policy_iteration_steps: int | None = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        del tol
        p_arr = _as_jax(p, dtype=jnp.float64)
        r_arr = _as_jax(r, dtype=jnp.float64)
        policy = _as_jax(init_policy, dtype=jnp.int32) if init_policy is not None else jnp.argmax(r_arr, axis=1).astype(jnp.int32)
        v = jnp.zeros(self.n_s, dtype=jnp.float64)
        fixed_steps = (
            None
            if surrogate_policy_iteration_steps is None
            else max(int(surrogate_policy_iteration_steps), 1)
        )
        loop_count = max_iter if fixed_steps is None else fixed_steps

        for _ in range(loop_count):
            hit = False
            if self.use_caching and self.cache_policy is not None:
                hit = bool(jnp.array_equal(policy, self.cache_policy)) and self.cache_gamma == gamma

            r_pi = r_arr[self.rows, policy]
            if hit and self.cache_inv is not None:
                v = self.cache_inv @ r_pi
            else:
                p_pi = p_arr[self.rows, policy, :]
                a = self.identity - gamma * p_pi
                try:
                    inv = _solve_linear(a, self.identity)
                    v = inv @ r_pi
                    if self.use_caching:
                        self.cache_inv = inv
                        self.cache_policy = jnp.array(policy, copy=True)
                        self.cache_gamma = gamma
                except Exception:
                    v = _solve_linear(a, r_pi)
                    if self.use_caching:
                        self.cache_inv = None

            q = r_arr + gamma * jnp.einsum("sak,k->sa", p_arr, v)
            new_policy = jnp.argmax(q, axis=1).astype(jnp.int32)
            if fixed_steps is None and bool(jnp.array_equal(new_policy, policy)):
                return v, new_policy
            policy = new_policy
        return v, policy


class ISP_Control:
    def __init__(self, mdp: MDP, p_hat: Any, solver_caching: str | bool = "auto"):
        self.mdp = mdp
        self.p_hat = _as_jax(p_hat, dtype=jnp.float64)
        self.n_s = mdp.P.shape[0]
        self.n_a = mdp.P.shape[1]
        self.p = _as_jax(mdp.P, dtype=jnp.float64)
        self.r = _as_jax(mdp.R, dtype=jnp.float64)
        self.identity = jnp.eye(self.n_s, dtype=jnp.float64)
        self.rows = jnp.arange(self.n_s, dtype=jnp.int32)
        self.use_caching = self.n_s >= 200 if solver_caching == "auto" else bool(solver_caching)
        self.cache_policy = None
        self.cache_h = None
        self.cache_inv = None

    @property
    def P_dev(self) -> jax.Array:
        return self.p

    @property
    def R_dev(self) -> jax.Array:
        return self.r

    @property
    def P_hat_dev(self) -> jax.Array:
        return self.p_hat

    @property
    def P_hat_flat(self) -> jax.Array:
        return self.p_hat.reshape(self.n_s * self.n_a, self.n_s)

    def compute_true_sweep(self, v: Any) -> Tuple[jnp.ndarray, float]:
        v_arr = _as_jax(v, dtype=jnp.float64)
        true_term = jnp.einsum("sak,k->sa", self.p, v_arr)
        q_true = self.r + self.mdp.gamma * true_term
        t_v = jnp.max(q_true, axis=1)
        resid = float(jnp.max(jnp.abs(t_v - v_arr)))
        return true_term, resid

    def compute_shifted_reward(
        self,
        v_k: Any,
        u_j: Any,
        h: float,
        *,
        true_term: Any | None = None,
    ) -> jnp.ndarray:
        v_k_arr = _as_jax(v_k, dtype=jnp.float64)
        u_j_arr = _as_jax(u_j, dtype=jnp.float64)
        true_term_arr = (
            _as_jax(true_term, dtype=jnp.float64)
            if true_term is not None
            else jnp.einsum("sak,k->sa", self.p, u_j_arr)
        )
        shadow_term = jnp.einsum("sak,k->sa", self.p_hat, u_j_arr)
        return (
            (1.0 / (1.0 + h)) * v_k_arr[:, None]
            + (h / (1.0 + h)) * self.r
            + ((h * self.mdp.gamma) / (1.0 + h)) * (true_term_arr - shadow_term)
        )

    def solve_surrogate_fixed_point(
        self,
        r_tilde: Any,
        h: float,
        tol: float = 1e-8,
        max_iter: int = 200,
        init_policy: Any | None = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        del tol
        r_tilde_arr = _as_jax(r_tilde, dtype=jnp.float64)
        gamma_tilde = (h * self.mdp.gamma) / (1.0 + h)
        policy = _as_jax(init_policy, dtype=jnp.int32) if init_policy is not None else jnp.argmax(r_tilde_arr, axis=1).astype(jnp.int32)
        v = jnp.zeros(self.n_s, dtype=jnp.float64)

        for _ in range(max_iter):
            hit = False
            if self.use_caching and self.cache_policy is not None and self.cache_h is not None:
                hit = bool(jnp.array_equal(policy, self.cache_policy)) and abs(self.cache_h - h) < 1e-12

            r_pi = r_tilde_arr[self.rows, policy]
            if hit and self.cache_inv is not None:
                v = self.cache_inv @ r_pi
            else:
                p_pi = self.p_hat[self.rows, policy, :]
                a = self.identity - gamma_tilde * p_pi
                try:
                    inv = _solve_linear(a, self.identity)
                    v = inv @ r_pi
                    if self.use_caching:
                        self.cache_inv = inv
                        self.cache_policy = jnp.array(policy, copy=True)
                        self.cache_h = h
                except Exception:
                    v = _solve_linear(a, r_pi)
                    if self.use_caching:
                        self.cache_inv = None

            q = r_tilde_arr + gamma_tilde * jnp.einsum("sak,k->sa", self.p_hat, v)
            new_policy = jnp.argmax(q, axis=1).astype(jnp.int32)
            if bool(jnp.array_equal(new_policy, policy)):
                return v, new_policy
            policy = new_policy
        return v, policy

    def run_isp_step(
        self,
        v_k: Any,
        h: float,
        J: int,
        init_policy: Any | None = None,
        *,
        initial_true_term: Any | None = None,
    ) -> Tuple[jnp.ndarray, int, jnp.ndarray | None]:
        u = jnp.array(_as_jax(v_k, dtype=jnp.float64), copy=True)
        total_queries = 0
        current_policy = _as_jax(init_policy, dtype=jnp.int32) if init_policy is not None else None

        for j_index in range(J):
            if j_index == 0 and initial_true_term is not None:
                r_tilde = self.compute_shifted_reward(v_k, u, h, true_term=initial_true_term)
            else:
                r_tilde = self.compute_shifted_reward(v_k, u, h)
                total_queries += 1
            u, current_policy = self.solve_surrogate_fixed_point(r_tilde, h, init_policy=current_policy)

        return u, total_queries, current_policy

    def run(
        self,
        max_avps: int,
        h: float,
        J: int,
        v_init: Any | None = None,
        target_residual: float | None = None,
        V_init: Any | None = None,
    ) -> Tuple[List[int], List[float]]:
        if V_init is not None and v_init is None:
            v_init = V_init
        v = _as_jax(v_init, dtype=jnp.float64) if v_init is not None else jnp.zeros(self.n_s, dtype=jnp.float64)
        total_queries = 0
        queries_trace: list[int] = []
        residuals: list[float] = []
        current_policy = None
        additional_step_cost = max(0, int(J) - 1)

        while total_queries < max_avps:
            true_term, resid = self.compute_true_sweep(v)
            # This sweep certifies BR*(v) and can be reused as the j=0 ISP sweep when continuing.
            total_queries += 1
            queries_trace.append(total_queries)
            residuals.append(resid)
            if target_residual is not None and resid <= target_residual:
                break
            if total_queries >= max_avps:
                break
            if total_queries + additional_step_cost > max_avps:
                break
            v, cost_step, current_policy = self.run_isp_step(
                v,
                h,
                J,
                init_policy=current_policy,
                initial_true_term=true_term,
            )
            total_queries += cost_step
        return queries_trace, residuals

    def run_with_policy_trace(
        self,
        max_avps: int,
        h: float,
        J: int,
        v_init: Any | None = None,
        V_init: Any | None = None,
    ) -> Tuple[List[int], List[float], jax.Array, jax.Array]:
        if V_init is not None and v_init is None:
            v_init = V_init
        v = (
            _as_jax(v_init, dtype=jnp.float64)
            if v_init is not None
            else jnp.zeros(self.n_s, dtype=jnp.float64)
        )
        total_queries = 0
        queries_trace: list[int] = []
        residuals: list[float] = []
        value_trace: list[jax.Array] = []
        policy_trace: list[jax.Array] = []
        current_policy = None
        additional_step_cost = max(0, int(J) - 1)

        while total_queries < max_avps:
            true_term, resid = self.compute_true_sweep(v)
            total_queries += 1
            if total_queries + additional_step_cost > max_avps:
                break

            value_trace.append(jnp.array(v, copy=True))
            v, cost_step, current_policy = self.run_isp_step(
                v,
                h,
                J,
                init_policy=current_policy,
                initial_true_term=true_term,
            )
            total_queries += cost_step
            queries_trace.append(total_queries)
            residuals.append(resid)
            policy_trace.append(
                jnp.array(
                    current_policy
                    if current_policy is not None
                    else jnp.zeros(self.n_s, dtype=jnp.int32),
                    copy=True,
                )
            )

        empty_values = jnp.zeros((0, self.n_s), dtype=jnp.float64)
        empty_policies = jnp.zeros((0, self.n_s), dtype=jnp.int32)
        return (
            queries_trace,
            residuals,
            jnp.stack(value_trace, axis=0) if value_trace else empty_values,
            jnp.stack(policy_trace, axis=0) if policy_trace else empty_policies,
        )


class AISP_Control:
    def __init__(self, mdp: MDP, p_hat: Any, solver_caching: str | bool = "auto"):
        self.isp = ISP_Control(mdp, p_hat, solver_caching=solver_caching)
        self.mdp = mdp
        self.n_s = mdp.P.shape[0]
        self.p = _as_jax(mdp.P, dtype=jnp.float64)
        self.r = _as_jax(mdp.R, dtype=jnp.float64)

    def compute_bellman_residual_norm(self, v: Any) -> float:
        v_arr = _as_jax(v, dtype=jnp.float64)
        q = _control_q(self.p, self.r, self.mdp.gamma, v_arr)
        t_v = jnp.max(q, axis=1)
        return float(jnp.max(jnp.abs(t_v - v_arr)))

    def run(self, target_epsilon: float, max_avps: int, V_init: Any | None = None, v_init: Any | None = None) -> Tuple[List[int], List[float]]:
        if V_init is not None and v_init is None:
            v_init = V_init
        gamma = self.mdp.gamma
        delta = 1.0 - gamma
        v0 = _as_jax(v_init, dtype=jnp.float64) if v_init is not None else jnp.zeros(self.n_s, dtype=jnp.float64)
        b0 = self.compute_bellman_residual_norm(v0)
        cumulative_queries = 1
        queries_trace = [cumulative_queries]
        residuals_trace = [b0]
        t_tol = target_epsilon * (delta / (1.0 + gamma)) * b0
        if b0 <= t_tol:
            return queries_trace, residuals_trace
        try:
            k_prime = max(1.0, math.log(1.0 / target_epsilon) + 2.0 * math.log((1.0 + gamma) / delta))
        except ValueError:
            k_prime = 100.0
        J = int(math.ceil(math.log2(8.0 / delta)))
        r_hat = 1.0
        active = True
        while active and cumulative_queries < max_avps:
            denom = delta * (2.0 * r_hat - 1.0)
            h_i = 1.0 / denom if denom > 1e-12 else 1.0 / delta
            n_i = int(math.ceil(4.0 * r_hat * k_prime))
            v = jnp.array(v0, copy=True)
            current_policy = None

            for _ in range(n_i):
                if cumulative_queries >= max_avps:
                    active = False
                    break
                v, cost, current_policy = self.isp.run_isp_step(v, h_i, J, init_policy=current_policy)
                cumulative_queries += cost
            if active:
                current_resid = self.compute_bellman_residual_norm(v)
                cumulative_queries += 1
                queries_trace.append(cumulative_queries)
                residuals_trace.append(current_resid)
                if current_resid <= t_tol:
                    active = False
                else:
                    r_hat *= 2.0
        return queries_trace, residuals_trace

    def run_with_policy_trace(
        self,
        target_epsilon: float,
        max_avps: int,
        V_init: Any | None = None,
        v_init: Any | None = None,
    ) -> Tuple[List[int], List[float], jax.Array, jax.Array]:
        if V_init is not None and v_init is None:
            v_init = V_init
        gamma = self.mdp.gamma
        delta = 1.0 - gamma
        v0 = (
            _as_jax(v_init, dtype=jnp.float64)
            if v_init is not None
            else jnp.zeros(self.n_s, dtype=jnp.float64)
        )
        q0 = _control_q(self.p, self.r, gamma, v0)
        t_v0 = jnp.max(q0, axis=1)
        b0 = float(jnp.max(jnp.abs(t_v0 - v0)))
        t_tol = target_epsilon * (delta / (1.0 + gamma)) * b0
        if b0 <= t_tol or max_avps <= 1:
            empty_values = jnp.zeros((0, self.n_s), dtype=jnp.float64)
            empty_policies = jnp.zeros((0, self.n_s), dtype=jnp.int32)
            return [], [], empty_values, empty_policies

        try:
            k_prime = max(
                1.0,
                math.log(1.0 / target_epsilon)
                + 2.0 * math.log((1.0 + gamma) / max(delta, 1e-12)),
            )
        except ValueError:
            k_prime = 100.0
        J = int(math.ceil(math.log2(8.0 / max(delta, 1e-12))))
        r_hat = 1.0
        total_queries = 1
        queries_trace: list[int] = []
        residuals_trace: list[float] = []
        value_trace: list[jax.Array] = []
        policy_trace: list[jax.Array] = []
        active = True

        while active and total_queries < max_avps:
            denom = delta * (2.0 * r_hat - 1.0)
            h_i = 1.0 / denom if denom > 1e-12 else 1.0 / max(delta, 1e-12)
            n_i = int(math.ceil(4.0 * r_hat * k_prime))
            v = jnp.array(v0, copy=True)
            current_policy = None
            completed_epoch = True

            for _ in range(n_i):
                if total_queries >= max_avps:
                    active = False
                    completed_epoch = False
                    break
                v_next, cost, current_policy = self.isp.run_isp_step(
                    v,
                    h_i,
                    J,
                    init_policy=current_policy,
                )
                if total_queries + cost + 1 > max_avps:
                    active = False
                    completed_epoch = False
                    break

                v = v_next
                total_queries += cost
                current_resid = self.compute_bellman_residual_norm(v)
                total_queries += 1
                queries_trace.append(total_queries)
                residuals_trace.append(current_resid)
                value_trace.append(jnp.array(v, copy=True))
                policy_trace.append(
                    jnp.array(
                        current_policy
                        if current_policy is not None
                        else jnp.zeros(self.n_s, dtype=jnp.int32),
                        copy=True,
                    )
                )
                if current_resid <= t_tol:
                    active = False
                    break

            if active and completed_epoch:
                r_hat *= 2.0

        empty_values = jnp.zeros((0, self.n_s), dtype=jnp.float64)
        empty_policies = jnp.zeros((0, self.n_s), dtype=jnp.int32)
        return (
            queries_trace,
            residuals_trace,
            jnp.stack(value_trace, axis=0) if value_trace else empty_values,
            jnp.stack(policy_trace, axis=0) if policy_trace else empty_policies,
        )


class Shadow_ISP_Control:
    def __init__(
        self,
        mdp: MDP,
        p_hat: Any,
        h_init: float = 1.0,
        alpha: float = 0.5,
        eps: float = 1e-12,
        optimism: float = 1.0,
        dynamic_J: bool = True,
        fixed_J: int | None = None,
        solver_caching: str | bool = "auto",
    ):
        self.mdp = mdp
        self.isp = ISP_Control(mdp, p_hat, solver_caching=solver_caching)
        self.n_s = mdp.P.shape[0]
        self.n_a = mdp.P.shape[1]
        self.gamma = mdp.gamma
        self.delta = 1.0 - mdp.gamma
        self.p = _as_jax(mdp.P, dtype=jnp.float64)
        self.r = _as_jax(mdp.R, dtype=jnp.float64)

        if fixed_J is not None:
            dynamic_J = False
        if not dynamic_J and fixed_J is None:
            fixed_J = int(math.ceil(math.log2(8.0 / max(self.delta, 1e-12))))
            fixed_J = max(1, fixed_J)

        self.adapter = _AdaptiveStepManager(
            delta=self.delta,
            h_init=h_init,
            h_smoothing=alpha,
            optimism=optimism,
            dynamic_m=dynamic_J,
            fixed_m=fixed_J,
        )
        self.eps = eps

    @property
    def history_h(self) -> list[float]:
        return self.adapter.history_h

    @property
    def history_E(self) -> list[float]:
        return self.adapter.history_E_est

    @property
    def h(self) -> float:
        return self.adapter.h

    @h.setter
    def h(self, value: float) -> None:
        self.adapter.h = value

    @property
    def dynamic_J(self) -> bool:
        return self.adapter.dynamic_m

    @property
    def J(self) -> int:
        if self.adapter.fixed_m is not None:
            return self.adapter.fixed_m
        return self.adapter.history_m[-1] if self.adapter.history_m else int(math.ceil(math.log2(8.0 / self.delta)))

    def step(self, v_k: Any, init_policy: Any | None = None) -> Tuple[jnp.ndarray, int, float, jnp.ndarray | None]:
        v_k_arr = _as_jax(v_k, dtype=jnp.float64)
        q_true = _control_q(self.p, self.r, self.gamma, v_k_arr)
        q_shadow = _control_q(self.isp.p_hat, self.r, self.gamma, v_k_arr)
        t_v_true = jnp.max(q_true, axis=1)
        t_v_shadow = jnp.max(q_shadow, axis=1)
        br_true = t_v_true - v_k_arr
        br_shadow = t_v_shadow - v_k_arr
        br_true_norm = float(jnp.max(jnp.abs(br_true)))

        e_est = self.adapter.estimate_error(br_true, br_shadow, v_k_arr)
        h, J = self.adapter.update(e_est)
        if J < 1:
            J = 1

        if br_true_norm > 1e7 or not bool(jnp.all(jnp.isfinite(v_k_arr))):
            self.adapter.h = 1e-4
            h = 1e-4
            v_k_arr = jnp.zeros_like(v_k_arr)
            q_true = _control_q(self.p, self.r, self.gamma, v_k_arr)
            t_v_true = jnp.max(q_true, axis=1)
            br_true_norm = float(jnp.max(jnp.abs(t_v_true - v_k_arr)))

        u = jnp.array(v_k_arr, copy=True)
        current_policy = _as_jax(init_policy, dtype=jnp.int32) if init_policy is not None else None

        for j in range(J):
            if j == 0:
                r_tilde = (
                    (1.0 / (1.0 + h)) * v_k_arr[:, None]
                    + (h / (1.0 + h)) * self.r
                    + (h / (1.0 + h)) * (q_true - q_shadow)
                )
                u, current_policy = self.isp.solve_surrogate_fixed_point(r_tilde, h, init_policy=current_policy)
            else:
                r_tilde = self.isp.compute_shifted_reward(v_k_arr, u, h)
                u, current_policy = self.isp.solve_surrogate_fixed_point(r_tilde, h, init_policy=current_policy)
        return u, max(1, J), br_true_norm, current_policy

    def run(
        self,
        max_avps: int,
        target_residual: float | None = None,
        V_init: Any | None = None,
        v_init: Any | None = None,
    ) -> Tuple[List[int], List[float]]:
        if V_init is not None and v_init is None:
            v_init = V_init
        v = _as_jax(v_init, dtype=jnp.float64) if v_init is not None else jnp.zeros(self.n_s, dtype=jnp.float64)
        total_queries = 0
        queries_trace = [0]
        residuals: list[float] = []
        current_policy = None
        best_resid = float("inf")
        divergent_streak = 0

        while total_queries < max_avps:
            v, cost, resid, current_policy = self.step(v, init_policy=current_policy)
            queries_trace.append(total_queries + 1)
            residuals.append(resid)
            total_queries += cost
            if target_residual is not None and resid <= target_residual:
                break
            if resid < best_resid:
                best_resid = resid
                divergent_streak = 0
            elif resid > max(1e3, best_resid * 100.0):
                divergent_streak += 1
                if divergent_streak >= 8:
                    break
            else:
                divergent_streak = 0
        return queries_trace, residuals

    def run_with_policy_trace(
        self,
        max_avps: int,
        target_residual: float | None = None,
        V_init: Any | None = None,
        v_init: Any | None = None,
    ) -> Tuple[List[int], List[float], jax.Array, jax.Array]:
        if V_init is not None and v_init is None:
            v_init = V_init
        v = (
            _as_jax(v_init, dtype=jnp.float64)
            if v_init is not None
            else jnp.zeros(self.n_s, dtype=jnp.float64)
        )
        total_queries = 0
        queries_trace: list[int] = []
        residuals: list[float] = []
        value_trace: list[jax.Array] = []
        policy_trace: list[jax.Array] = []
        current_policy = None
        best_resid = float("inf")
        divergent_streak = 0

        while total_queries < max_avps:
            v_next, cost, resid, current_policy = self.step(v, init_policy=current_policy)
            if total_queries + cost > max_avps:
                break

            total_queries += cost
            v = v_next
            queries_trace.append(total_queries)
            residuals.append(resid)
            value_trace.append(jnp.array(v, copy=True))
            policy_trace.append(
                jnp.array(
                    current_policy
                    if current_policy is not None
                    else jnp.zeros(self.n_s, dtype=jnp.int32),
                    copy=True,
                )
            )
            if target_residual is not None and resid <= target_residual:
                break
            if resid < best_resid:
                best_resid = resid
                divergent_streak = 0
            elif resid > max(1e3, best_resid * 100.0):
                divergent_streak += 1
                if divergent_streak >= 8:
                    break
            else:
                divergent_streak = 0

        empty_values = jnp.zeros((0, self.n_s), dtype=jnp.float64)
        empty_policies = jnp.zeros((0, self.n_s), dtype=jnp.int32)
        return (
            queries_trace,
            residuals,
            jnp.stack(value_trace, axis=0) if value_trace else empty_values,
            jnp.stack(policy_trace, axis=0) if policy_trace else empty_policies,
        )


H_INIT = 1e6
H_MIN = 1.0
STALL_PATIENCE = 3
DECAY_FACTOR = 0.1
DIVERGENCE_MULT = 1e4
GEO_ALPHA = 0.5
DD_M_MAX = 3


@dataclass(frozen=True)
class _GeoWaterfallTransition:
    next_v: jnp.ndarray
    next_h: float
    next_best_br_norm: float
    next_residual_norm: float
    accepted: bool


def _geo_waterfall_transition(
    *,
    current_v: Any,
    candidate_v: Any,
    current_residual_norm: float,
    h: float,
    alpha: float,
    best_br_norm: float,
    candidate_valid: bool,
    candidate_br_norm: float,
) -> _GeoWaterfallTransition:
    current_v_arr = _as_jax(current_v, dtype=jnp.float64)
    candidate_v_arr = _as_jax(candidate_v, dtype=jnp.float64)
    if not candidate_valid:
        return _GeoWaterfallTransition(
            next_v=current_v_arr,
            next_h=max(1.0, alpha * h),
            next_best_br_norm=best_br_norm,
            next_residual_norm=current_residual_norm,
            accepted=False,
        )
    if candidate_br_norm < best_br_norm:
        return _GeoWaterfallTransition(
            next_v=candidate_v_arr,
            next_h=h,
            next_best_br_norm=candidate_br_norm,
            next_residual_norm=candidate_br_norm,
            accepted=True,
        )
    return _GeoWaterfallTransition(
        next_v=candidate_v_arr,
        next_h=max(1.0, alpha * h),
        next_best_br_norm=best_br_norm,
        next_residual_norm=candidate_br_norm,
        accepted=True,
    )


@dataclass(frozen=True)
class _DDWaterfallAttemptTransition:
    next_v: jnp.ndarray
    next_h: float
    next_best_br_norm: float
    next_residual_norm: float
    accepted: bool
    status: str


def _dd_waterfall_attempt_transition(
    *,
    current_v: Any,
    candidate_v: Any,
    current_residual_norm: float,
    h: float,
    decay_factor: float,
    best_br_norm: float,
    candidate_valid: bool,
    candidate_br_norm: float,
    exhausted: bool,
) -> _DDWaterfallAttemptTransition:
    current_v_arr = _as_jax(current_v, dtype=jnp.float64)
    candidate_v_arr = _as_jax(candidate_v, dtype=jnp.float64)
    if not candidate_valid:
        return _DDWaterfallAttemptTransition(
            next_v=current_v_arr,
            next_h=max(1.0, decay_factor * h),
            next_best_br_norm=best_br_norm,
            next_residual_norm=current_residual_norm,
            accepted=False,
            status="invalid",
        )
    if candidate_br_norm < best_br_norm:
        return _DDWaterfallAttemptTransition(
            next_v=candidate_v_arr,
            next_h=h,
            next_best_br_norm=candidate_br_norm,
            next_residual_norm=candidate_br_norm,
            accepted=True,
            status="progress",
        )
    if exhausted:
        return _DDWaterfallAttemptTransition(
            next_v=candidate_v_arr,
            next_h=max(1.0, decay_factor * h),
            next_best_br_norm=best_br_norm,
            next_residual_norm=candidate_br_norm,
            accepted=True,
            status="exhausted",
        )
    raise ValueError("DD-Waterfall transition requires an invalid, progress, or exhausted terminal outcome.")


def _surrogate_control_value_iteration(
    v_init: Any,
    p_hat: Any,
    r_tilde: Any,
    gamma_eff: float,
    *,
    tol: float,
    max_iters: int,
) -> jnp.ndarray:
    v_new = _as_jax(v_init, dtype=jnp.float64)
    p_hat_arr = _as_jax(p_hat, dtype=jnp.float64)
    r_tilde_arr = _as_jax(r_tilde, dtype=jnp.float64)
    for _ in range(max(int(max_iters), 0)):
        q_surr = r_tilde_arr + gamma_eff * jnp.einsum("sak,k->sa", p_hat_arr, v_new)
        u_new = jnp.max(q_surr, axis=1)
        if float(jnp.max(jnp.abs(u_new - v_new))) < tol:
            return u_new
        v_new = u_new
    return v_new


def _surrogate_control_policy_from_value(
    v: Any,
    p_hat: Any,
    r_tilde: Any,
    gamma_eff: float,
) -> jnp.ndarray:
    v_arr = _as_jax(v, dtype=jnp.float64)
    p_hat_arr = _as_jax(p_hat, dtype=jnp.float64)
    r_tilde_arr = _as_jax(r_tilde, dtype=jnp.float64)
    q_surr = r_tilde_arr + gamma_eff * jnp.einsum("sak,k->sa", p_hat_arr, v_arr)
    return jnp.argmax(q_surr, axis=1).astype(jnp.int32)


class WaterfallBESP_PE:
    def __init__(
        self,
        p_pi: Any,
        p_hat_pi: Any,
        r_pi: Any,
        gamma: float,
        h_init: float = H_INIT,
        h_min: float = H_MIN,
        stall_patience: int = STALL_PATIENCE,
        decay_factor: float = DECAY_FACTOR,
    ):
        self.p_pi = _as_jax(p_pi, dtype=jnp.float64)
        self.p_hat_pi = _as_jax(p_hat_pi, dtype=jnp.float64)
        self.r_pi = _as_jax(r_pi, dtype=jnp.float64)
        self.gamma = gamma
        self.n_s = self.p_pi.shape[0]
        self.h = h_init
        self.h_min = h_min
        self.stall_patience = stall_patience
        self.decay_factor = decay_factor
        self.v = jnp.zeros(self.n_s, dtype=jnp.float64)
        self.total_queries = 0
        self.identity = jnp.eye(self.n_s, dtype=jnp.float64)
        br = self.r_pi + self.gamma * (self.p_pi @ self.v) - self.v
        self.br = br
        self.best_br_norm = float(jnp.max(jnp.abs(br)))
        self.current_br_norm = self.best_br_norm
        self.stall_count = 0
        self.stop_due_to_failed_floor = False
        self.history_h = [self.h]
        self.history_br = [self.best_br_norm]

    @property
    def V(self) -> jax.Array:
        return self.v

    def _solve_khat(self, rhs: Any, h: float) -> jnp.ndarray:
        mat = (1.0 + h) * self.identity - h * self.gamma * self.p_hat_pi
        return _solve_linear(mat, rhs)

    def _v_max_theory(self) -> float:
        return float(jnp.max(jnp.abs(self.r_pi))) / (1.0 - self.gamma)

    def _record_rejected_attempt(self) -> Tuple[int, float]:
        self.h = max(self.h_min, self.h * self.decay_factor)
        self.stall_count = 0
        self.total_queries += 1
        self.stop_due_to_failed_floor = self.h <= self.h_min
        self.history_h.append(self.h)
        self.history_br.append(self.current_br_norm)
        return 1, self.current_br_norm

    def step(self) -> Tuple[int, float]:
        try:
            w = self._solve_khat(self.br, self.h)
            v_new = self.v + self.h * w
        except Exception:
            return self._record_rejected_attempt()

        v_div_envelope = DIVERGENCE_MULT * max(self._v_max_theory(), 1.0)
        if float(jnp.max(jnp.abs(v_new))) > v_div_envelope:
            return self._record_rejected_attempt()

        new_br = self.r_pi + self.gamma * (self.p_pi @ v_new) - v_new
        new_br_norm = float(jnp.max(jnp.abs(new_br)))
        if not _is_finite_scalar(new_br_norm):
            return self._record_rejected_attempt()

        self.v = v_new
        self.br = new_br
        self.total_queries += 1
        self.stop_due_to_failed_floor = False
        self.current_br_norm = new_br_norm
        if new_br_norm < self.best_br_norm:
            self.best_br_norm = new_br_norm
            self.stall_count = 0
        else:
            self.stall_count += 1
            if self.stall_count >= self.stall_patience:
                self.h = max(self.h_min, self.h * self.decay_factor)
                self.stall_count = 0

        self.history_h.append(self.h)
        self.history_br.append(new_br_norm)
        return 1, new_br_norm


class WaterfallISP_Control:
    def __init__(
        self,
        p: Any,
        p_hat: Any,
        r: Any,
        gamma: float,
        h_init: float = H_INIT,
        h_min: float = H_MIN,
        stall_patience: int = STALL_PATIENCE,
        decay_factor: float = DECAY_FACTOR,
        inner_vi_tol: float = 1e-10,
        inner_vi_max_iters: int = 5000,
    ):
        self.p = _as_jax(p, dtype=jnp.float64)
        self.p_hat = _as_jax(p_hat, dtype=jnp.float64)
        self.r = _as_jax(r, dtype=jnp.float64)
        self.gamma = gamma
        self.n_s = self.p.shape[0]
        self.n_a = self.p.shape[1]
        self.h = h_init
        self.h_min = h_min
        self.stall_patience = stall_patience
        self.decay_factor = decay_factor
        self.inner_vi_tol = inner_vi_tol
        self.inner_vi_max_iters = inner_vi_max_iters
        self.v = jnp.zeros(self.n_s, dtype=jnp.float64)
        self.total_queries = 0
        q = _control_q(self.p, self.r, self.gamma, self.v)
        t_v = jnp.max(q, axis=1)
        self.best_br_norm = float(jnp.max(jnp.abs(t_v - self.v)))
        self.current_br_norm = self.best_br_norm
        self.stall_count = 0
        self.stop_due_to_failed_floor = False
        self._last_expected_v_true = jnp.einsum("sak,k->sa", self.p, self.v)
        self.last_policy: jnp.ndarray | None = None
        self.history_h = [self.h]
        self.history_br = [self.best_br_norm]

    @property
    def V(self) -> jax.Array:
        return self.v

    def _v_max_theory(self) -> float:
        return float(jnp.max(jnp.abs(self.r))) / (1.0 - self.gamma)

    def _record_rejected_attempt(self) -> Tuple[int, float]:
        self.h = max(self.h_min, self.h * self.decay_factor)
        self.stall_count = 0
        self.total_queries += 1
        self.stop_due_to_failed_floor = self.h <= self.h_min
        self.history_h.append(self.h)
        self.history_br.append(self.current_br_norm)
        return 1, self.current_br_norm

    def step_with_policy(self) -> Tuple[int, float, jnp.ndarray | None]:
        h = self.h
        gamma_eff = (h * self.gamma) / (1.0 + h)
        policy_candidate: jnp.ndarray | None = None

        try:
            expected_v_true = self._last_expected_v_true
            expected_v_hat = jnp.einsum("sak,k->sa", self.p_hat, self.v)
            r_tilde = (
                (1.0 / (1.0 + h)) * self.v[:, None]
                + (h / (1.0 + h)) * self.r
                + ((h * self.gamma) / (1.0 + h)) * (expected_v_true - expected_v_hat)
            )
            v_new = jnp.array(self.v, copy=True)
            for _ in range(self.inner_vi_max_iters):
                q_surr = r_tilde + gamma_eff * jnp.einsum("sak,k->sa", self.p_hat, v_new)
                u_new = jnp.max(q_surr, axis=1)
                if float(jnp.max(jnp.abs(u_new - v_new))) < self.inner_vi_tol:
                    break
                v_new = u_new
            if bool(jnp.all(jnp.isfinite(v_new))):
                q_surr = r_tilde + gamma_eff * jnp.einsum("sak,k->sa", self.p_hat, v_new)
                if bool(jnp.all(jnp.isfinite(q_surr))):
                    policy_candidate = jnp.argmax(q_surr, axis=1).astype(jnp.int32)
        except Exception:
            cost, resid = self._record_rejected_attempt()
            return cost, resid, self.last_policy

        v_div_envelope = DIVERGENCE_MULT * max(self._v_max_theory(), 1.0)
        if float(jnp.max(jnp.abs(v_new))) > v_div_envelope:
            cost, resid = self._record_rejected_attempt()
            if policy_candidate is not None:
                self.last_policy = policy_candidate
            return cost, resid, self.last_policy

        expected_v_new_true = jnp.einsum("sak,k->sa", self.p, v_new)
        q_new = self.r + self.gamma * expected_v_new_true
        t_v_new = jnp.max(q_new, axis=1)
        new_br_norm = float(jnp.max(jnp.abs(t_v_new - v_new)))
        if not _is_finite_scalar(new_br_norm):
            cost, resid = self._record_rejected_attempt()
            if policy_candidate is not None:
                self.last_policy = policy_candidate
            return cost, resid, self.last_policy

        self.v = v_new
        self.total_queries += 1
        self.stop_due_to_failed_floor = False
        self._last_expected_v_true = expected_v_new_true
        self.current_br_norm = new_br_norm
        if new_br_norm < self.best_br_norm:
            self.best_br_norm = new_br_norm
            self.stall_count = 0
        else:
            self.stall_count += 1
            if self.stall_count >= self.stall_patience:
                self.h = max(self.h_min, self.h * self.decay_factor)
                self.stall_count = 0

        self.history_h.append(self.h)
        self.history_br.append(new_br_norm)
        if policy_candidate is not None:
            self.last_policy = policy_candidate
        return 1, new_br_norm, self.last_policy

    def step(self) -> Tuple[int, float]:
        cost, resid, _ = self.step_with_policy()
        return cost, resid

    def run_with_policy_trace(
        self,
        max_queries: int,
    ) -> Tuple[List[int], List[float], jnp.ndarray, jnp.ndarray]:
        queries: list[int] = []
        residuals: list[float] = []
        value_trace: list[jnp.ndarray] = []
        policy_trace: list[jnp.ndarray] = []
        zero_policy = jnp.zeros(self.n_s, dtype=jnp.int32)

        while self.total_queries < max_queries:
            _, resid, policy = self.step_with_policy()
            queries.append(int(self.total_queries))
            residuals.append(float(resid))
            value_trace.append(jnp.asarray(self.v, dtype=jnp.float64))
            policy_trace.append(
                jnp.asarray(zero_policy if policy is None else policy, dtype=jnp.int32)
            )
            if self.stop_due_to_failed_floor:
                break

        value_arr = (
            jnp.stack(value_trace, axis=0)
            if value_trace
            else jnp.zeros((0, self.n_s), dtype=jnp.float64)
        )
        policy_arr = (
            jnp.stack(policy_trace, axis=0)
            if policy_trace
            else jnp.zeros((0, self.n_s), dtype=jnp.int32)
        )
        return queries, residuals, value_arr, policy_arr


class GeoWaterfallBESP_PE:
    def __init__(
        self,
        p_pi: Any,
        p_hat_pi: Any,
        r_pi: Any,
        gamma: float,
        h_init: float = H_INIT,
        alpha: float = GEO_ALPHA,
    ):
        self.p_pi = _as_jax(p_pi, dtype=jnp.float64)
        self.p_hat_pi = _as_jax(p_hat_pi, dtype=jnp.float64)
        self.r_pi = _as_jax(r_pi, dtype=jnp.float64)
        self.gamma = gamma
        self.n_s = self.p_pi.shape[0]
        self.h = h_init
        self.alpha = alpha
        self.v = jnp.zeros(self.n_s, dtype=jnp.float64)
        self.total_queries = 0
        self.identity = jnp.eye(self.n_s, dtype=jnp.float64)
        br = self.r_pi + self.gamma * (self.p_pi @ self.v) - self.v
        self.br = br
        self.current_br_norm = float(jnp.max(jnp.abs(br)))
        self.best_br_norm = self.current_br_norm
        self.history_h = [self.h]
        self.history_br = [self.current_br_norm]

    @property
    def V(self) -> jax.Array:
        return self.v

    def _solve_khat(self, rhs: Any, h: float) -> jnp.ndarray:
        mat = (1.0 + h) * self.identity - h * self.gamma * self.p_hat_pi
        return _solve_linear(mat, rhs)

    def _v_max_theory(self) -> float:
        return float(jnp.max(jnp.abs(self.r_pi))) / (1.0 - self.gamma)

    def step(self) -> Tuple[int, float]:
        v_candidate = self.v
        new_br = self.br
        candidate_valid = False
        candidate_br_norm = math.nan
        v_div_envelope = DIVERGENCE_MULT * max(self._v_max_theory(), 1.0)

        try:
            w = self._solve_khat(self.br, self.h)
            v_candidate = self.v + self.h * w
            if (
                bool(jnp.all(jnp.isfinite(v_candidate)))
                and float(jnp.max(jnp.abs(v_candidate))) <= v_div_envelope
            ):
                new_br = self.r_pi + self.gamma * (self.p_pi @ v_candidate) - v_candidate
                candidate_br_norm = float(jnp.max(jnp.abs(new_br)))
                candidate_valid = _is_finite_scalar(candidate_br_norm)
        except Exception:
            candidate_valid = False

        transition = _geo_waterfall_transition(
            current_v=self.v,
            candidate_v=v_candidate,
            current_residual_norm=self.current_br_norm,
            h=self.h,
            alpha=self.alpha,
            best_br_norm=self.best_br_norm,
            candidate_valid=candidate_valid,
            candidate_br_norm=candidate_br_norm,
        )

        self.total_queries += 1
        self.h = transition.next_h
        self.best_br_norm = transition.next_best_br_norm
        self.current_br_norm = transition.next_residual_norm
        if transition.accepted:
            self.v = v_candidate
            self.br = new_br

        self.history_h.append(self.h)
        self.history_br.append(self.current_br_norm)
        return 1, self.current_br_norm


class GeoWaterfallISP_Control:
    def __init__(
        self,
        p: Any,
        p_hat: Any,
        r: Any,
        gamma: float,
        h_init: float = H_INIT,
        alpha: float = GEO_ALPHA,
        inner_vi_tol: float = 1e-10,
        inner_vi_max_iters: int = 5000,
    ):
        self.p = _as_jax(p, dtype=jnp.float64)
        self.p_hat = _as_jax(p_hat, dtype=jnp.float64)
        self.r = _as_jax(r, dtype=jnp.float64)
        self.gamma = gamma
        self.n_s = self.p.shape[0]
        self.n_a = self.p.shape[1]
        self.h = h_init
        self.alpha = alpha
        self.inner_vi_tol = inner_vi_tol
        self.inner_vi_max_iters = inner_vi_max_iters
        self.v = jnp.zeros(self.n_s, dtype=jnp.float64)
        self.total_queries = 0
        self._last_expected_v_true = jnp.einsum("sak,k->sa", self.p, self.v)
        q = _control_q(self.p, self.r, self.gamma, self.v)
        t_v = jnp.max(q, axis=1)
        self.current_br_norm = float(jnp.max(jnp.abs(t_v - self.v)))
        self.best_br_norm = self.current_br_norm
        self.last_policy: jnp.ndarray | None = None
        self.history_h = [self.h]
        self.history_br = [self.current_br_norm]

    @property
    def V(self) -> jax.Array:
        return self.v

    def _v_max_theory(self) -> float:
        return float(jnp.max(jnp.abs(self.r))) / (1.0 - self.gamma)

    def step_with_policy(self) -> Tuple[int, float, jnp.ndarray | None]:
        h = self.h
        inv_1ph = 1.0 / (1.0 + h)
        blend = h * inv_1ph
        gamma_blend = self.gamma * blend
        v_candidate = self.v
        expected_v_new_true = self._last_expected_v_true
        policy_candidate: jnp.ndarray | None = None
        candidate_valid = False
        candidate_br_norm = math.nan
        v_div_envelope = DIVERGENCE_MULT * max(self._v_max_theory(), 1.0)

        try:
            expected_v_true = self._last_expected_v_true
            expected_v_hat = jnp.einsum("sak,k->sa", self.p_hat, self.v)
            r_tilde = inv_1ph * self.v[:, None] + blend * self.r + gamma_blend * (
                expected_v_true - expected_v_hat
            )
            v_candidate = _surrogate_control_value_iteration(
                self.v,
                self.p_hat,
                r_tilde,
                gamma_blend,
                tol=self.inner_vi_tol,
                max_iters=self.inner_vi_max_iters,
            )
            if bool(jnp.all(jnp.isfinite(v_candidate))):
                q_surr = r_tilde + gamma_blend * jnp.einsum("sak,k->sa", self.p_hat, v_candidate)
                if bool(jnp.all(jnp.isfinite(q_surr))):
                    policy_candidate = jnp.argmax(q_surr, axis=1).astype(jnp.int32)
            if (
                bool(jnp.all(jnp.isfinite(v_candidate)))
                and float(jnp.max(jnp.abs(v_candidate))) <= v_div_envelope
            ):
                expected_v_new_true = jnp.einsum("sak,k->sa", self.p, v_candidate)
                q_new = self.r + self.gamma * expected_v_new_true
                t_v_new = jnp.max(q_new, axis=1)
                candidate_br_norm = float(jnp.max(jnp.abs(t_v_new - v_candidate)))
                candidate_valid = _is_finite_scalar(candidate_br_norm)
        except Exception:
            candidate_valid = False

        transition = _geo_waterfall_transition(
            current_v=self.v,
            candidate_v=v_candidate,
            current_residual_norm=self.current_br_norm,
            h=self.h,
            alpha=self.alpha,
            best_br_norm=self.best_br_norm,
            candidate_valid=candidate_valid,
            candidate_br_norm=candidate_br_norm,
        )

        self.total_queries += 1
        self.h = transition.next_h
        self.best_br_norm = transition.next_best_br_norm
        self.current_br_norm = transition.next_residual_norm
        if transition.accepted:
            self.v = v_candidate
            self._last_expected_v_true = expected_v_new_true

        self.history_h.append(self.h)
        self.history_br.append(self.current_br_norm)
        if policy_candidate is not None:
            self.last_policy = policy_candidate
        return 1, self.current_br_norm, self.last_policy

    def step(self) -> Tuple[int, float]:
        cost, resid, _ = self.step_with_policy()
        return cost, resid

    def run_with_policy_trace(
        self,
        max_queries: int,
    ) -> Tuple[List[int], List[float], jnp.ndarray, jnp.ndarray]:
        queries: list[int] = []
        residuals: list[float] = []
        value_trace: list[jnp.ndarray] = []
        policy_trace: list[jnp.ndarray] = []
        zero_policy = jnp.zeros(self.n_s, dtype=jnp.int32)

        while self.total_queries < max_queries:
            _, resid, policy = self.step_with_policy()
            queries.append(int(self.total_queries))
            residuals.append(float(resid))
            value_trace.append(jnp.asarray(self.v, dtype=jnp.float64))
            policy_trace.append(
                jnp.asarray(zero_policy if policy is None else policy, dtype=jnp.int32)
            )

        value_arr = (
            jnp.stack(value_trace, axis=0)
            if value_trace
            else jnp.zeros((0, self.n_s), dtype=jnp.float64)
        )
        policy_arr = (
            jnp.stack(policy_trace, axis=0)
            if policy_trace
            else jnp.zeros((0, self.n_s), dtype=jnp.int32)
        )
        return queries, residuals, value_arr, policy_arr


class DDWaterfallBESP_PE:
    def __init__(
        self,
        p_pi: Any,
        p_hat_pi: Any,
        r_pi: Any,
        gamma: float,
        h_init: float = H_INIT,
        m_max: int = DD_M_MAX,
        decay_factor: float = DECAY_FACTOR,
    ):
        self.p_pi = _as_jax(p_pi, dtype=jnp.float64)
        self.p_hat_pi = _as_jax(p_hat_pi, dtype=jnp.float64)
        self.r_pi = _as_jax(r_pi, dtype=jnp.float64)
        self.gamma = gamma
        self.n_s = self.p_pi.shape[0]
        self.h = h_init
        self.m_max = max(int(m_max), 1)
        self.decay_factor = decay_factor
        self.v = jnp.zeros(self.n_s, dtype=jnp.float64)
        self.total_queries = 0
        self.identity = jnp.eye(self.n_s, dtype=jnp.float64)
        br = self.r_pi + self.gamma * (self.p_pi @ self.v) - self.v
        self.br = br
        self.current_br_norm = float(jnp.max(jnp.abs(br)))
        self.best_br_norm = self.current_br_norm
        self.history_h = [self.h]
        self.history_br = [self.current_br_norm]

    @property
    def V(self) -> jax.Array:
        return self.v

    def _v_max_theory(self) -> float:
        return float(jnp.max(jnp.abs(self.r_pi))) / (1.0 - self.gamma)

    def step(self, *, max_depth: int | None = None) -> Tuple[int, float]:
        depth_limit = self.m_max if max_depth is None else max(1, min(int(max_depth), self.m_max))
        h = self.h
        inv_1ph = 1.0 / (1.0 + h)
        blend = h * inv_1ph
        gamma_blend = self.gamma * blend
        mat = self.identity - gamma_blend * self.p_hat_pi
        v_div_envelope = DIVERGENCE_MULT * max(self._v_max_theory(), 1.0)

        u = self.v
        u_br = self.br
        terminal = None
        next_br = self.br
        depth_used = 0

        for depth in range(1, depth_limit + 1):
            depth_used = depth
            candidate_valid = False
            candidate_br_norm = math.nan
            exhausted = depth == depth_limit
            u_candidate = u
            try:
                expected_u_hat = self.p_hat_pi @ u
                r_tilde = inv_1ph * self.v + blend * (u_br + u) - gamma_blend * expected_u_hat
                u_candidate = _solve_linear(mat, r_tilde)
                if (
                    bool(jnp.all(jnp.isfinite(u_candidate)))
                    and float(jnp.max(jnp.abs(u_candidate))) <= v_div_envelope
                ):
                    next_br = self.r_pi + self.gamma * (self.p_pi @ u_candidate) - u_candidate
                    candidate_br_norm = float(jnp.max(jnp.abs(next_br)))
                    candidate_valid = _is_finite_scalar(candidate_br_norm)
            except Exception:
                candidate_valid = False

            invalid = not candidate_valid
            progress = candidate_valid and candidate_br_norm < self.best_br_norm
            if invalid or progress or exhausted:
                terminal = _dd_waterfall_attempt_transition(
                    current_v=self.v,
                    candidate_v=u_candidate,
                    current_residual_norm=self.current_br_norm,
                    h=self.h,
                    decay_factor=self.decay_factor,
                    best_br_norm=self.best_br_norm,
                    candidate_valid=candidate_valid,
                    candidate_br_norm=candidate_br_norm,
                    exhausted=exhausted,
                )
                break

            u = u_candidate
            u_br = next_br

        if terminal is None:
            raise RuntimeError("DD-Waterfall PE step reached a non-terminal state.")

        self.total_queries += depth_used
        self.h = terminal.next_h
        self.best_br_norm = terminal.next_best_br_norm
        self.current_br_norm = terminal.next_residual_norm
        if terminal.accepted:
            self.v = terminal.next_v
            self.br = next_br

        self.history_h.append(self.h)
        self.history_br.append(self.current_br_norm)
        return depth_used, self.current_br_norm


class DDWaterfallISP_Control:
    def __init__(
        self,
        p: Any,
        p_hat: Any,
        r: Any,
        gamma: float,
        h_init: float = H_INIT,
        m_max: int = DD_M_MAX,
        decay_factor: float = DECAY_FACTOR,
        inner_vi_tol: float = 1e-10,
        inner_vi_max_iters: int = 5000,
    ):
        self.p = _as_jax(p, dtype=jnp.float64)
        self.p_hat = _as_jax(p_hat, dtype=jnp.float64)
        self.r = _as_jax(r, dtype=jnp.float64)
        self.gamma = gamma
        self.n_s = self.p.shape[0]
        self.n_a = self.p.shape[1]
        self.h = h_init
        self.m_max = max(int(m_max), 1)
        self.decay_factor = decay_factor
        self.inner_vi_tol = inner_vi_tol
        self.inner_vi_max_iters = inner_vi_max_iters
        self.v = jnp.zeros(self.n_s, dtype=jnp.float64)
        self.total_queries = 0
        self._last_expected_v_true = jnp.einsum("sak,k->sa", self.p, self.v)
        q = _control_q(self.p, self.r, self.gamma, self.v)
        t_v = jnp.max(q, axis=1)
        self.current_br_norm = float(jnp.max(jnp.abs(t_v - self.v)))
        self.best_br_norm = self.current_br_norm
        self.last_policy: jnp.ndarray | None = None
        self.history_h = [self.h]
        self.history_br = [self.current_br_norm]

    @property
    def V(self) -> jax.Array:
        return self.v

    def _v_max_theory(self) -> float:
        return float(jnp.max(jnp.abs(self.r))) / (1.0 - self.gamma)

    def step_with_policy(
        self, *, max_depth: int | None = None
    ) -> Tuple[int, float, jnp.ndarray | None]:
        depth_limit = self.m_max if max_depth is None else max(1, min(int(max_depth), self.m_max))
        h = self.h
        inv_1ph = 1.0 / (1.0 + h)
        blend = h * inv_1ph
        gamma_blend = self.gamma * blend
        v_div_envelope = DIVERGENCE_MULT * max(self._v_max_theory(), 1.0)

        anchor_v = self.v
        u = anchor_v
        u_expected_true = self._last_expected_v_true
        terminal = None
        next_expected_true = self._last_expected_v_true
        depth_used = 0
        terminal_policy: jnp.ndarray | None = self.last_policy

        for depth in range(1, depth_limit + 1):
            depth_used = depth
            candidate_valid = False
            candidate_br_norm = math.nan
            exhausted = depth == depth_limit
            v_candidate = u
            policy_candidate: jnp.ndarray | None = None
            try:
                expected_u_hat = jnp.einsum("sak,k->sa", self.p_hat, u)
                r_tilde = (
                    inv_1ph * anchor_v[:, None]
                    + blend * self.r
                    + gamma_blend * (u_expected_true - expected_u_hat)
                )
                v_candidate = _surrogate_control_value_iteration(
                    u,
                    self.p_hat,
                    r_tilde,
                    gamma_blend,
                    tol=self.inner_vi_tol,
                    max_iters=self.inner_vi_max_iters,
                )
                if bool(jnp.all(jnp.isfinite(v_candidate))):
                    q_surr = r_tilde + gamma_blend * jnp.einsum("sak,k->sa", self.p_hat, v_candidate)
                    if bool(jnp.all(jnp.isfinite(q_surr))):
                        policy_candidate = jnp.argmax(q_surr, axis=1).astype(jnp.int32)
                if (
                    bool(jnp.all(jnp.isfinite(v_candidate)))
                    and float(jnp.max(jnp.abs(v_candidate))) <= v_div_envelope
                ):
                    next_expected_true = jnp.einsum("sak,k->sa", self.p, v_candidate)
                    q_new = self.r + self.gamma * next_expected_true
                    t_v_new = jnp.max(q_new, axis=1)
                    candidate_br_norm = float(jnp.max(jnp.abs(t_v_new - v_candidate)))
                    candidate_valid = _is_finite_scalar(candidate_br_norm)
            except Exception:
                candidate_valid = False

            invalid = not candidate_valid
            progress = candidate_valid and candidate_br_norm < self.best_br_norm
            if invalid or progress or exhausted:
                terminal = _dd_waterfall_attempt_transition(
                    current_v=self.v,
                    candidate_v=v_candidate,
                    current_residual_norm=self.current_br_norm,
                    h=self.h,
                    decay_factor=self.decay_factor,
                    best_br_norm=self.best_br_norm,
                    candidate_valid=candidate_valid,
                    candidate_br_norm=candidate_br_norm,
                    exhausted=exhausted,
                )
                if policy_candidate is not None:
                    terminal_policy = policy_candidate
                break

            u = v_candidate
            u_expected_true = next_expected_true

        if terminal is None:
            raise RuntimeError("DD-Waterfall control step reached a non-terminal state.")

        self.total_queries += depth_used
        self.h = terminal.next_h
        self.best_br_norm = terminal.next_best_br_norm
        self.current_br_norm = terminal.next_residual_norm
        if terminal.accepted:
            self.v = terminal.next_v
            self._last_expected_v_true = next_expected_true

        self.history_h.append(self.h)
        self.history_br.append(self.current_br_norm)
        if terminal_policy is not None:
            self.last_policy = terminal_policy
        return depth_used, self.current_br_norm, self.last_policy

    def step(self, *, max_depth: int | None = None) -> Tuple[int, float]:
        cost, resid, _ = self.step_with_policy(max_depth=max_depth)
        return cost, resid

    def run_with_policy_trace(
        self,
        max_queries: int,
    ) -> Tuple[List[int], List[float], jnp.ndarray, jnp.ndarray]:
        queries: list[int] = []
        residuals: list[float] = []
        value_trace: list[jnp.ndarray] = []
        policy_trace: list[jnp.ndarray] = []
        zero_policy = jnp.zeros(self.n_s, dtype=jnp.int32)

        while self.total_queries < max_queries:
            remaining = max_queries - self.total_queries
            _, resid, policy = self.step_with_policy(max_depth=remaining)
            queries.append(int(self.total_queries))
            residuals.append(float(resid))
            value_trace.append(jnp.asarray(self.v, dtype=jnp.float64))
            policy_trace.append(
                jnp.asarray(zero_policy if policy is None else policy, dtype=jnp.int32)
            )

        value_arr = (
            jnp.stack(value_trace, axis=0)
            if value_trace
            else jnp.zeros((0, self.n_s), dtype=jnp.float64)
        )
        policy_arr = (
            jnp.stack(policy_trace, axis=0)
            if policy_trace
            else jnp.zeros((0, self.n_s), dtype=jnp.int32)
        )
        return queries, residuals, value_arr, policy_arr


def waterfall_besp_pe(
    p_pi: Any,
    p_hat_pi: Any,
    r_pi: Any,
    gamma: float,
    v_true: Any,
    max_queries: int,
    value_norm: str = "L1",
    value_normalized: bool = True,
    h_init: float = H_INIT,
    h_min: float = H_MIN,
    stall_patience: int = STALL_PATIENCE,
    decay_factor: float = DECAY_FACTOR,
) -> Tuple[List[int], List[float]]:
    v_true_arr = _as_jax(v_true, dtype=jnp.float64)
    solver = WaterfallBESP_PE(
        p_pi,
        p_hat_pi,
        r_pi,
        gamma,
        h_init=h_init,
        h_min=h_min,
        stall_patience=stall_patience,
        decay_factor=decay_factor,
    )
    queries = [0]
    errors = [_compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)]
    while solver.total_queries < max_queries:
        solver.step()
        queries.append(solver.total_queries)
        err = _compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)
        errors.append(err)
        if not _is_finite_scalar(err):
            break
        if getattr(solver, "stop_due_to_failed_floor", False):
            break
    return queries, errors


def waterfall_isp_control(
    p: Any,
    p_hat: Any,
    r: Any,
    gamma: float,
    v_true: Any,
    max_queries: int,
    value_norm: str = "L1",
    value_normalized: bool = True,
    h_init: float = H_INIT,
    h_min: float = H_MIN,
    stall_patience: int = STALL_PATIENCE,
    decay_factor: float = DECAY_FACTOR,
) -> Tuple[List[int], List[float]]:
    v_true_arr = _as_jax(v_true, dtype=jnp.float64)
    solver = WaterfallISP_Control(
        p,
        p_hat,
        r,
        gamma,
        h_init=h_init,
        h_min=h_min,
        stall_patience=stall_patience,
        decay_factor=decay_factor,
    )
    queries = [0]
    errors = [_compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)]
    while solver.total_queries < max_queries:
        solver.step()
        queries.append(solver.total_queries)
        err = _compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)
        errors.append(err)
        if not _is_finite_scalar(err):
            break
        if getattr(solver, "stop_due_to_failed_floor", False):
            break
    return queries, errors


def geo_waterfall_besp_pe(
    p_pi: Any,
    p_hat_pi: Any,
    r_pi: Any,
    gamma: float,
    v_true: Any,
    max_queries: int,
    value_norm: str = "L1",
    value_normalized: bool = True,
    h_init: float = H_INIT,
    alpha: float = GEO_ALPHA,
) -> Tuple[List[int], List[float]]:
    v_true_arr = _as_jax(v_true, dtype=jnp.float64)
    solver = GeoWaterfallBESP_PE(
        p_pi,
        p_hat_pi,
        r_pi,
        gamma,
        h_init=h_init,
        alpha=alpha,
    )
    queries = [0]
    errors = [_compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)]
    while solver.total_queries < max_queries:
        solver.step()
        queries.append(solver.total_queries)
        err = _compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)
        errors.append(err)
        if not _is_finite_scalar(err):
            break
    return queries, errors


def geo_waterfall_isp_control(
    p: Any,
    p_hat: Any,
    r: Any,
    gamma: float,
    v_true: Any,
    max_queries: int,
    value_norm: str = "L1",
    value_normalized: bool = True,
    h_init: float = H_INIT,
    alpha: float = GEO_ALPHA,
    inner_vi_tol: float = 1e-10,
    inner_vi_max_iters: int = 5000,
) -> Tuple[List[int], List[float]]:
    v_true_arr = _as_jax(v_true, dtype=jnp.float64)
    solver = GeoWaterfallISP_Control(
        p,
        p_hat,
        r,
        gamma,
        h_init=h_init,
        alpha=alpha,
        inner_vi_tol=inner_vi_tol,
        inner_vi_max_iters=inner_vi_max_iters,
    )
    queries = [0]
    errors = [_compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)]
    while solver.total_queries < max_queries:
        solver.step()
        queries.append(solver.total_queries)
        err = _compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)
        errors.append(err)
        if not _is_finite_scalar(err):
            break
    return queries, errors


def dd_waterfall_besp_pe(
    p_pi: Any,
    p_hat_pi: Any,
    r_pi: Any,
    gamma: float,
    v_true: Any,
    max_queries: int,
    value_norm: str = "L1",
    value_normalized: bool = True,
    h_init: float = H_INIT,
    m_max: int = DD_M_MAX,
    decay_factor: float = DECAY_FACTOR,
) -> Tuple[List[int], List[float]]:
    v_true_arr = _as_jax(v_true, dtype=jnp.float64)
    solver = DDWaterfallBESP_PE(
        p_pi,
        p_hat_pi,
        r_pi,
        gamma,
        h_init=h_init,
        m_max=m_max,
        decay_factor=decay_factor,
    )
    queries = [0]
    errors = [_compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)]
    while solver.total_queries < max_queries:
        remaining = max_queries - solver.total_queries
        solver.step(max_depth=remaining)
        queries.append(solver.total_queries)
        err = _compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)
        errors.append(err)
        if not _is_finite_scalar(err):
            break
    return queries, errors


def dd_waterfall_isp_control(
    p: Any,
    p_hat: Any,
    r: Any,
    gamma: float,
    v_true: Any,
    max_queries: int,
    value_norm: str = "L1",
    value_normalized: bool = True,
    h_init: float = H_INIT,
    m_max: int = DD_M_MAX,
    decay_factor: float = DECAY_FACTOR,
    inner_vi_tol: float = 1e-10,
    inner_vi_max_iters: int = 5000,
) -> Tuple[List[int], List[float]]:
    v_true_arr = _as_jax(v_true, dtype=jnp.float64)
    solver = DDWaterfallISP_Control(
        p,
        p_hat,
        r,
        gamma,
        h_init=h_init,
        m_max=m_max,
        decay_factor=decay_factor,
        inner_vi_tol=inner_vi_tol,
        inner_vi_max_iters=inner_vi_max_iters,
    )
    queries = [0]
    errors = [_compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)]
    while solver.total_queries < max_queries:
        remaining = max_queries - solver.total_queries
        solver.step(max_depth=remaining)
        queries.append(solver.total_queries)
        err = _compute_value_error(solver.v, v_true_arr, norm_name=value_norm, normalized=value_normalized)
        errors.append(err)
        if not _is_finite_scalar(err):
            break
    return queries, errors


__all__ = [
    "AISP_Control",
    "DECAY_FACTOR",
    "DD_M_MAX",
    "DDWaterfallBESP_PE",
    "DDWaterfallISP_Control",
    "DIVERGENCE_MULT",
    "GEO_ALPHA",
    "GeoWaterfallBESP_PE",
    "GeoWaterfallISP_Control",
    "H_INIT",
    "H_MIN",
    "ISP_Control",
    "MSVI_Control",
    "STALL_PATIENCE",
    "Shadow_BESP",
    "Shadow_ISP_Control",
    "VI_Control",
    "WaterfallBESP_PE",
    "WaterfallISP_Control",
    "_discount_gap_depth_pe_impl",
    "_discount_gap_depth_theory",
    "abesp_pe",
    "besp_pe",
    "besp_precompute",
    "build_policy_matrices",
    "dd_waterfall_besp_pe",
    "dd_waterfall_isp_control",
    "model_error_stats",
    "osvi_pe",
    "pick_besp_rho_half",
    "pick_isp_params_control",
    "shadow_besp_pe",
    "solve_value",
    "value_iteration_control",
    "vi_pe",
    "geo_waterfall_besp_pe",
    "geo_waterfall_isp_control",
    "waterfall_besp_pe",
    "waterfall_isp_control",
]
