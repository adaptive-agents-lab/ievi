"""Value-iteration-only planning algorithms and operation accounting.

No GMRES, direct factorization, or matrix inversion is used anywhere in this
experiment. Ordinary VI solves the true problem. Every IE-VI surrogate
subproblem is solved by ordinary fixed-point value iteration with the coarse
matrix-free simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Any

import numpy as np

from micro_simulator import MicroSimulator


@dataclass
class RunResult:
    method: str
    grid_side: int
    state_count: int
    repetition: int
    runtime_seconds: float
    converged: bool
    final_relative_true_residual: float
    sup_error_to_reference: float
    true_operator_calls: int
    true_microsteps: int
    surrogate_operator_calls: int
    surrogate_microsteps: int
    outer_iterations: int
    inner_vi_iterations: int
    surrogate_microsteps_per_call: int | None = None
    h: float | None = None
    m: int | None = None
    gamma_h: float | None = None
    inner_vi_rtol: float | None = None
    inner_solver_converged: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def true_residual(
    value: np.ndarray,
    reward: np.ndarray,
    gamma: float,
    true_operator: MicroSimulator,
) -> tuple[float, np.ndarray]:
    transition_value = true_operator.apply(value)
    residual = reward + gamma * transition_value - value
    return float(np.max(np.abs(residual))), transition_value


def run_value_iteration(
    *,
    reward: np.ndarray,
    gamma: float,
    true_operator: MicroSimulator,
    target_relative_residual: float,
    reference_value: np.ndarray,
    grid_side: int,
    repetition: int,
    maximum_iterations: int = 100_000,
) -> tuple[RunResult, np.ndarray]:
    """Ordinary policy-evaluation VI using the high-fidelity simulator."""
    true_operator.reset_counters()
    value = np.zeros_like(reward)
    initial_residual = float(np.max(np.abs(reward)))
    threshold = target_relative_residual * initial_residual
    iterations = 0

    start = perf_counter()
    while True:
        residual, transition_value = true_residual(value, reward, gamma, true_operator)
        if residual <= threshold:
            converged = True
            break
        if iterations >= maximum_iterations:
            converged = False
            break
        value = reward + gamma * transition_value
        iterations += 1
    runtime = perf_counter() - start

    result = RunResult(
        method="VI",
        grid_side=grid_side,
        state_count=grid_side * grid_side,
        repetition=repetition,
        runtime_seconds=runtime,
        converged=converged,
        final_relative_true_residual=residual / initial_residual,
        sup_error_to_reference=float(np.max(np.abs(value - reference_value))),
        true_operator_calls=true_operator.calls,
        true_microsteps=true_operator.microstep_count,
        surrogate_operator_calls=0,
        surrogate_microsteps=0,
        outer_iterations=iterations,
        inner_vi_iterations=0,
        inner_solver_converged=None,
    )
    return result, value


def solve_surrogate_by_value_iteration(
    *,
    right_hand_side: np.ndarray,
    initial_guess: np.ndarray,
    gamma_h: float,
    surrogate_operator: MicroSimulator,
    relative_bellman_residual: float,
    maximum_iterations: int = 200_000,
) -> tuple[np.ndarray, int, bool]:
    """Solve U = b + gamma_h P_hat U by ordinary value iteration.

    The stopping criterion is the surrogate Bellman residual in the infinity
    norm, relative to ||b||_infinity. The latest IE-VI iterate is used as a warm
    start. Each loop performs exactly one surrogate-operator application.
    """
    if not 0.0 <= gamma_h < 1.0:
        raise ValueError("gamma_h must lie in [0,1)")
    if relative_bellman_residual <= 0.0:
        raise ValueError("relative_bellman_residual must be positive")

    value = np.asarray(initial_guess, dtype=np.float64).copy()
    scale = max(float(np.max(np.abs(right_hand_side))), np.finfo(np.float64).tiny)
    threshold = relative_bellman_residual * scale

    for iteration in range(1, maximum_iterations + 1):
        next_value = right_hand_side + gamma_h * surrogate_operator.apply(value)
        bellman_residual = float(np.max(np.abs(next_value - value)))
        value = next_value
        if bellman_residual <= threshold:
            return value, iteration, True
    return value, maximum_iterations, False


def run_ievi(
    *,
    reward: np.ndarray,
    gamma: float,
    true_operator: MicroSimulator,
    surrogate_operator: MicroSimulator,
    h: float,
    m: int,
    inner_vi_rtol: float,
    target_relative_residual: float,
    reference_value: np.ndarray,
    grid_side: int,
    repetition: int,
    maximum_outer_iterations: int = 5_000,
) -> tuple[RunResult, np.ndarray]:
    """IE-VI with ordinary VI for every shifted-reward surrogate solve."""
    if h <= 0.0:
        raise ValueError("h must be positive")
    if m < 1:
        raise ValueError("m must be positive")

    true_operator.reset_counters()
    surrogate_operator.reset_counters()
    value = np.zeros_like(reward)
    true_value_at_current = np.zeros_like(reward)  # P 0 = 0 exactly.
    initial_residual = float(np.max(np.abs(reward)))
    threshold = target_relative_residual * initial_residual
    outer_iterations = 0
    total_inner_vi_iterations = 0
    every_inner_solve_converged = True
    alpha = h / (1.0 + h)
    gamma_h = alpha * gamma

    start = perf_counter()
    while outer_iterations < maximum_outer_iterations:
        outer_value = value.copy()
        inner_value = value.copy()
        true_value_at_inner = true_value_at_current

        for correction in range(m):
            # This application is required to form (P - P_hat) U.
            surrogate_value_at_inner = surrogate_operator.apply(inner_value)
            corrected_reward = reward + gamma * (
                true_value_at_inner - surrogate_value_at_inner
            )
            right_hand_side = (outer_value + h * corrected_reward) / (1.0 + h)
            inner_value, inner_iterations, solved = solve_surrogate_by_value_iteration(
                right_hand_side=right_hand_side,
                initial_guess=inner_value,
                gamma_h=gamma_h,
                surrogate_operator=surrogate_operator,
                relative_bellman_residual=inner_vi_rtol,
            )
            total_inner_vi_iterations += inner_iterations
            every_inner_solve_converged = every_inner_solve_converged and solved
            if correction < m - 1:
                true_value_at_inner = true_operator.apply(inner_value)

        value = inner_value
        outer_iterations += 1
        residual, true_value_at_current = true_residual(
            value, reward, gamma, true_operator
        )
        if residual <= threshold:
            converged = every_inner_solve_converged
            break
    else:
        residual, true_value_at_current = true_residual(
            value, reward, gamma, true_operator
        )
        converged = False

    runtime = perf_counter() - start
    method = f"IEVI_VI_Ks{surrogate_operator.microsteps_per_call}"
    result = RunResult(
        method=method,
        grid_side=grid_side,
        state_count=grid_side * grid_side,
        repetition=repetition,
        runtime_seconds=runtime,
        converged=converged,
        final_relative_true_residual=residual / initial_residual,
        sup_error_to_reference=float(np.max(np.abs(value - reference_value))),
        true_operator_calls=true_operator.calls,
        true_microsteps=true_operator.microstep_count,
        surrogate_operator_calls=surrogate_operator.calls,
        surrogate_microsteps=surrogate_operator.microstep_count,
        outer_iterations=outer_iterations,
        inner_vi_iterations=total_inner_vi_iterations,
        surrogate_microsteps_per_call=surrogate_operator.microsteps_per_call,
        h=h,
        m=m,
        gamma_h=gamma_h,
        inner_vi_rtol=inner_vi_rtol,
        inner_solver_converged=every_inner_solve_converged,
    )
    return result, value


def compute_reference_by_value_iteration(
    *,
    reward: np.ndarray,
    gamma: float,
    true_operator: MicroSimulator,
    relative_tolerance: float = 1e-10,
    maximum_iterations: int = 100_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """High-accuracy ordinary-VI reference, excluded from benchmark timings."""
    true_operator.reset_counters()
    value = np.zeros_like(reward)
    initial_residual = float(np.max(np.abs(reward)))
    threshold = relative_tolerance * initial_residual
    iterations = 0
    start = perf_counter()
    while iterations < maximum_iterations:
        residual, transition_value = true_residual(value, reward, gamma, true_operator)
        if residual <= threshold:
            break
        value = reward + gamma * transition_value
        iterations += 1
    runtime = perf_counter() - start
    converged = residual <= threshold
    metadata = {
        "method": "ordinary value iteration",
        "runtime_seconds": runtime,
        "iterations": iterations,
        "true_operator_calls": true_operator.calls,
        "true_microsteps": true_operator.microstep_count,
        "final_absolute_true_residual": residual,
        "final_relative_true_residual": residual / initial_residual,
        "relative_tolerance": relative_tolerance,
        "converged": converged,
    }
    if not converged:
        raise RuntimeError(f"Reference VI failed: {metadata}")
    return value, metadata
