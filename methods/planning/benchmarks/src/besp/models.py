"""Model-error helpers over the shared JAX-backed runtime.

The maintained model-perturbation behavior lives in
``besp.model_error_runtime``.

Exact NumPy ``dirichlet`` parity is checked outside the package for
regression checks only.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random

from .model_error_runtime import apply_model_error_for_task

TransitionMatrix = Any | jax.Array


def _as_jax(array: Any, *, dtype: Any | None = None) -> jax.Array:
    return jnp.asarray(array, dtype=dtype)


def _seed_from_rng(rng: Any | None) -> int:
    if rng is None:
        return 0

    integers = getattr(rng, "integers", None)
    if callable(integers):
        try:
            return int(integers(0, 2**31 - 1))
        except TypeError:
            return int(integers(0, 2**31 - 1, dtype="int64"))

    return abs(hash((type(rng).__name__, id(rng)))) % (2**31 - 1)


def _return_like(reference: Any, value: Any) -> jax.Array:
    return _as_jax(value, dtype=getattr(reference, "dtype", None))


def robust_dirichlet(alpha: Any, rng: Any | None = None) -> jax.Array:
    """Sample from a Dirichlet distribution using the maintained JAX path."""
    alpha_arr = _as_jax(alpha, dtype=jnp.float64)
    seed = _seed_from_rng(rng)
    key = random.PRNGKey(seed)
    support = alpha_arr > 0
    min_alpha = jnp.min(jnp.where(support, alpha_arr, jnp.inf))

    def sample_logspace() -> jax.Array:
        u = random.uniform(
            key,
            alpha_arr.shape,
            minval=jnp.finfo(jnp.float64).tiny,
            maxval=1.0,
            dtype=jnp.float64,
        )
        log_gammas = jnp.where(support, jnp.log(u) / (alpha_arr + 1e-20), -jnp.inf)
        probs = jax.nn.softmax(log_gammas, axis=-1)
        return jnp.where(support, probs, 0.0)

    def sample_gamma() -> jax.Array:
        gammas = random.gamma(key, alpha_arr, dtype=jnp.float64)
        gammas = jnp.where(support, gammas, 0.0)
        total = jnp.sum(gammas, axis=-1, keepdims=True)
        return jnp.where(total > 0, gammas / total, 0.0)

    sampled = jax.lax.cond(min_alpha < 0.01, sample_logspace, sample_gamma)
    return _return_like(alpha, sampled)


def smooth_model(P: TransitionMatrix, lam: float) -> TransitionMatrix:
    """Apply support-preserving smoothing through the shared JAX runtime."""
    return _return_like(
        P,
        apply_model_error_for_task(P, lam, "support_uniform", seed=0),
    )


def apply_model_error(
    P: TransitionMatrix,
    lam: float,
    mode: str = "support_uniform",
    rng: Any | None = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    state_mask: Any | None = None,
    blur_scale: float = 1.0,
    min_rank: int = 1,
) -> TransitionMatrix:
    """Apply one maintained model-error mode through the shared JAX runtime."""
    return _return_like(
        P,
        apply_model_error_for_task(
            P,
            lam,
            mode,
            seed=_seed_from_rng(rng),
            grid_shape=grid_shape,
            state_mask=state_mask,
            blur_scale=blur_scale,
            min_rank=min_rank,
        ),
    )


def mix_transitions(base: Any, stay_idx: int | None = None) -> jax.Array:
    """Clamp one state to an absorbing self-loop in the returned transition tensor."""
    p = _as_jax(base)
    if stay_idx is None:
        return _return_like(base, p)

    updated = p.at[stay_idx].set(0.0)
    updated = updated.at[stay_idx, stay_idx].set(1.0)
    return _return_like(base, updated)


__all__ = [
    "TransitionMatrix",
    "apply_model_error",
    "mix_transitions",
    "robust_dirichlet",
    "smooth_model",
]
