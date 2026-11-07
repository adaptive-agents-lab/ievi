"""Canonical deterministic control experiment helpers."""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp

from ...algorithms.jax.deterministic import MSVI_Control, VI_Control
from ...core.types import Array, MDP
from ...environments.jax_native import create_environment_jax_native


def _build_control_experiment_mdp(env_name: str, seed: int) -> MDP:
    if env_name == "garnet":
        return create_environment_jax_native(
            "garnet",
            gamma=0.99,
            n_states=10,
            n_actions=3,
            branching=5,
            reward_states=2,
            seed=seed,
        )
    if env_name == "maze33":
        return create_environment_jax_native("maze", gamma=0.9)
    if env_name == "cliffwalk":
        return create_environment_jax_native("cliffwalk", gamma=0.9)
    raise ValueError(f"Unknown env: {env_name}")


def _apply_experiment_model_error(p: Array, lam: float, mode: str) -> jax.Array:
    p_arr = jnp.asarray(p, dtype=jnp.float64)
    mode_key = mode.lower()
    if mode_key == "support_uniform":
        support = p_arr > 0
        support_size = jnp.sum(support, axis=2, keepdims=True)
        safe_support_size = jnp.where(support_size > 0, support_size, 1.0)
        uniform = jnp.where(support, 1.0 / safe_support_size, 0.0)
        mixed = (1.0 - lam) * p_arr + lam * uniform
        return jnp.where(support_size > 0, mixed, p_arr)
    if mode_key == "self_loop":
        identity = jnp.eye(p_arr.shape[0], dtype=p_arr.dtype)
        return (1.0 - lam) * p_arr + lam * identity[:, None, :]
    raise ValueError(f"Unsupported control experiment mode: {mode}")


def run_control_experiment(
    env_name: str,
    smoothing_mode: str,
    alphas: List[float],
    n_iter: int = 50,
    seed: int = 0,
) -> Dict:
    mdp = _build_control_experiment_mdp(env_name, seed)

    results = {
        "env": env_name,
        "mode": smoothing_mode,
        "alphas": alphas,
        "vi": {},
        "msvi": {},
    }
    vi = VI_Control(mdp)
    v_star_trace, _ = vi.run(n_iter=500)
    v_star = v_star_trace[-1]
    v_vi_trace, _ = vi.run(n_iter=n_iter)
    results["vi"]["V_trace"] = v_vi_trace
    results["V_star"] = v_star

    error_mode = "support_uniform" if smoothing_mode == "LocallySmoothed" else "self_loop"
    for alpha in alphas:
        p_hat = _apply_experiment_model_error(mdp.P, lam=alpha, mode=error_mode)
        msvi = MSVI_Control(mdp, p_hat)
        v_msvi_trace, _ = msvi.run(n_iter=n_iter)
        results["msvi"][alpha] = v_msvi_trace
    return results
