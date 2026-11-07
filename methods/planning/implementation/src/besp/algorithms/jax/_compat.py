"""Private compatibility helpers kept off the public JAX RL package surface."""

from __future__ import annotations

from typing import Union

import jax
import jax.numpy as jnp


@jax.jit
def _compute_policy_from_q_compat(Q: jnp.ndarray) -> jnp.ndarray:
    """Argmax policy from Q."""
    return jnp.argmax(Q, axis=2)


@jax.jit
def _compute_policy_os_dyna_compat(
    r_k: jnp.ndarray,
    P_hat: jnp.ndarray,
    V: jnp.ndarray,
    gamma: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Argmax policy for OS-Dyna using its surrogate Q values."""
    gamma_3d = gamma.reshape(-1, 1, 1) if hasattr(gamma, "reshape") else gamma
    term_fut = jnp.einsum("bzas,bs->bza", P_hat, V)
    Q = r_k + gamma_3d * term_fut
    return jnp.argmax(Q, axis=2)


__all__ = ["_compute_policy_from_q_compat", "_compute_policy_os_dyna_compat"]
