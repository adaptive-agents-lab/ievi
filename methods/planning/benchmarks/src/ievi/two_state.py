"""Two-state closed form for the schematic inset (NumPy).

Port of ``slow_mode_panel_data`` / ``first_below`` from
``plot_strict_swap_inset.py``. Left-panel inset diamonds stay schematic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SLOW_MODE_THRESHOLD = 1e-3
PAPER_GAMMA = 0.99
PAPER_A = 0.6
PAPER_B = 0.0
PAPER_M = 3


def strict_swap_transition() -> np.ndarray:
    return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)


def asymmetric_self_loop_transition(a: float, b: float) -> np.ndarray:
    return np.asarray([[a, 1.0 - a], [1.0 - b, b]], dtype=float)


def modal_basis() -> np.ndarray:
    return np.asarray([[1.0, 1.0], [1.0, -1.0]])


def modal(value: np.ndarray, value_star: np.ndarray, initial_modal: np.ndarray) -> np.ndarray:
    return np.linalg.solve(modal_basis(), value - value_star) / initial_modal


def solve_problem(*, gamma: float, a: float, b: float) -> dict[str, np.ndarray]:
    p = strict_swap_transition()
    p_hat = asymmetric_self_loop_transition(a, b)
    reward = np.asarray([1.0, 0.0], dtype=float)
    value_star = np.linalg.solve(np.eye(2) - gamma * p, reward)
    initial_error = -value_star
    initial_modal = np.linalg.solve(modal_basis(), initial_error)
    return {
        "p": p,
        "p_hat": p_hat,
        "reward": reward,
        "value_star": value_star,
        "initial_error": initial_error,
        "initial_modal": initial_modal,
    }


def ievi_outer_trace(
    *,
    gamma: float,
    a: float,
    b: float,
    h: float,
    m: int,
    outer_steps: int,
    use_p_hat_eq_p: bool = False,
) -> np.ndarray:
    problem = solve_problem(gamma=gamma, a=a, b=b)
    p = problem["p"]
    p_hat = p if use_p_hat_eq_p else problem["p_hat"]
    r = problem["reward"]
    v_star = problem["value_star"]
    init_modal = problem["initial_modal"]
    gamma_h = h * gamma / (1.0 + h)
    solve_matrix = np.eye(2) - gamma_h * p_hat

    v = np.zeros(2)
    modes: list[np.ndarray] = []
    for _ in range(outer_steps + 1):
        modes.append(modal(v, v_star, init_modal))
        anchor = v.copy()
        inner = anchor.copy()
        for _j in range(m):
            defect = r + gamma * (p - p_hat) @ inner
            shifted = anchor / (1.0 + h) + h * defect / (1.0 + h)
            inner = np.linalg.solve(solve_matrix, shifted)
        v = inner
    return np.asarray(modes)


def value_function_error_norm_from_modes(
    modes: np.ndarray,
    *,
    gamma: float,
    a: float,
    b: float,
) -> np.ndarray:
    problem = solve_problem(gamma=gamma, a=a, b=b)
    init_modal = problem["initial_modal"]
    basis = modal_basis()
    unnormalized_modal = modes * init_modal[None, :]
    standard_errors = unnormalized_modal @ basis.T
    return np.max(np.abs(standard_errors), axis=1)


def first_below(q_arr: np.ndarray, y_arr: np.ndarray, thresh: float) -> int:
    """First query with ``y <= thresh``. Returns ``-1`` if none."""

    mask = y_arr <= thresh
    if not mask.any():
        return -1
    return int(q_arr[int(np.argmax(mask))])


def slow_mode_panel_data(
    *,
    gamma: float,
    a: float,
    b: float,
    h: float,
    m: int,
    outer_steps: int = 14,
) -> dict[str, Any]:
    exact_modes = ievi_outer_trace(
        gamma=gamma,
        a=a,
        b=b,
        h=h,
        m=1,
        outer_steps=outer_steps,
        use_p_hat_eq_p=True,
    )
    surrogate_modes = ievi_outer_trace(
        gamma=gamma,
        a=a,
        b=b,
        h=h,
        m=m,
        outer_steps=outer_steps,
    )
    max_query = outer_steps * m
    vi_queries = np.arange(max_query + 1)
    exact_queries = np.arange(outer_steps + 1)
    surrogate_queries = np.arange(outer_steps + 1) * m
    vi_threshold_query = math.ceil(math.log(SLOW_MODE_THRESHOLD) / math.log(gamma))

    exact_norms = value_function_error_norm_from_modes(
        exact_modes, gamma=gamma, a=a, b=b
    )
    surrogate_norms = value_function_error_norm_from_modes(
        surrogate_modes, gamma=gamma, a=a, b=b
    )
    exact_relative = exact_norms / exact_norms[0]
    surrogate_relative = surrogate_norms / surrogate_norms[0]
    vi_relative = gamma ** vi_queries

    return {
        "vi_queries": vi_queries,
        "vi_relative_error": vi_relative,
        "exact_queries": exact_queries,
        "exact_relative_error": exact_relative,
        "surrogate_queries": surrogate_queries,
        "surrogate_relative_error": surrogate_relative,
        "threshold": SLOW_MODE_THRESHOLD,
        "vi_threshold_query": vi_threshold_query,
        "max_query": max_query,
    }


def two_state_threshold_crossings(
    *,
    gamma: float = PAPER_GAMMA,
    a: float = PAPER_A,
    b: float = PAPER_B,
    m: int = PAPER_M,
    h: float | None = None,
    outer_steps: int = 14,
) -> dict[str, int]:
    step = 1.0 / (1.0 - gamma) if h is None else h
    data = slow_mode_panel_data(
        gamma=gamma, a=a, b=b, h=step, m=m, outer_steps=outer_steps
    )
    threshold = float(data["threshold"])
    return {
        "exact_ie_vi": first_below(
            np.asarray(data["exact_queries"], dtype=float),
            np.asarray(data["exact_relative_error"], dtype=float),
            threshold,
        ),
        "surrogate_ie_vi": first_below(
            np.asarray(data["surrogate_queries"], dtype=float),
            np.asarray(data["surrogate_relative_error"], dtype=float),
            threshold,
        ),
        "vi": int(data["vi_threshold_query"]),
    }
