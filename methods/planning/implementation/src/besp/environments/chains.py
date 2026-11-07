"""Canonical chain-style environment helpers backed by the JAX-native builders."""

from __future__ import annotations

from ..core.types import MDP
from .jax_native import generate_chain_jax, generate_river_swim_jax


def generate_river_swim(
    n_s: int = 6,
    small_reward: float = 0.005,
    big_reward: float = 1.0,
    gamma: float = 0.99,
) -> MDP:
    return generate_river_swim_jax(
        n_s=n_s,
        small_reward=small_reward,
        big_reward=big_reward,
        gamma=gamma,
    )


def generate_chain(
    n_s: int = 5,
    p_forward: float = 0.8,
    small_reward: float = 0.01,
    big_reward: float = 1.0,
    gamma: float = 0.99,
) -> MDP:
    return generate_chain_jax(
        n_s=n_s,
        p_forward=p_forward,
        small_reward=small_reward,
        big_reward=big_reward,
        gamma=gamma,
    )


__all__ = ["generate_chain", "generate_river_swim"]
