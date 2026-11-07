from __future__ import annotations

from typing import Any, Iterable

import jax.numpy as jnp

from ..core.types import MDP
from .jax_native import (
    _default_branching as _native_default_branching,
    _default_reward_states as _native_default_reward_states,
    generate_garnet_jax,
)


def _default_branching(n_s: int) -> int:
    return _native_default_branching(n_s)


def _default_reward_states(n_s: int) -> int:
    return _native_default_reward_states(n_s)


def _probs_from_sorted_cuts(cuts: Iterable[float]) -> list[float]:
    probs: list[float] = []
    previous = 0.0
    for cut in cuts:
        current = float(cut)
        probs.append(current - previous)
        previous = current
    probs.append(1.0 - previous)
    return probs


def _generate_garnet_from_rng(
    *,
    n_s: int,
    n_a: int,
    branching: int,
    reward_states: int,
    gamma: float,
    rng: Any,
) -> MDP:
    p = jnp.zeros((n_s, n_a, n_s), dtype=jnp.float64)
    for state in range(n_s):
        for action in range(n_a):
            next_states = rng.choice(n_s, size=branching, replace=False)
            cuts = sorted(float(value) for value in rng.random(branching - 1))
            probs = _probs_from_sorted_cuts(cuts)
            p = p.at[state, action, jnp.asarray(next_states, dtype=jnp.int32)].set(
                jnp.asarray(probs, dtype=jnp.float64)
            )

    r = jnp.zeros((n_s, n_a), dtype=jnp.float64)
    reward_idx = rng.choice(n_s, size=reward_states, replace=False)
    reward_values = rng.random((reward_states, n_a))
    r = r.at[jnp.asarray(reward_idx, dtype=jnp.int32), :].set(
        jnp.asarray(reward_values, dtype=jnp.float64)
    )
    return MDP(P=p, R=r, gamma=gamma)


def generate_garnet(
    n_s: int = 50,
    n_a: int = 4,
    branching: int | None = None,
    reward_states: int | None = None,
    gamma: float = 0.99,
    rng: Any | None = None,
    seed: int | None = None,
    **_: Any,
) -> MDP:
    branching = branching if branching is not None else _default_branching(n_s)
    reward_states = reward_states if reward_states is not None else _default_reward_states(n_s)

    if rng is not None:
        return _generate_garnet_from_rng(
            n_s=n_s,
            n_a=n_a,
            branching=branching,
            reward_states=reward_states,
            gamma=gamma,
            rng=rng,
        )

    return generate_garnet_jax(
        n_s=n_s,
        n_a=n_a,
        branching=branching,
        reward_states=reward_states,
        gamma=gamma,
        seed=seed,
    )


__all__ = ["_default_branching", "_default_reward_states", "generate_garnet"]
