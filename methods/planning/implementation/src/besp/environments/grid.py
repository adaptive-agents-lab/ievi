"""Canonical grid-style environment helpers backed by the JAX-native builders."""

from __future__ import annotations

from typing import Any

from ..core.types import MDP
from .jax_native import (
    generate_dense_reward_grid_jax,
    generate_four_rooms_jax,
    generate_frozen_lake_jax,
    generate_maze_jax,
    generate_modified_cliffwalk_jax,
    generate_slippery_cliffwalk_jax,
    generate_stochastic_grid_jax,
)


def generate_modified_cliffwalk(gamma: float = 0.9) -> MDP:
    return generate_modified_cliffwalk_jax(gamma=gamma)


def generate_maze(gamma: float = 0.9) -> MDP:
    return generate_maze_jax(gamma=gamma)


def generate_stochastic_grid(
    n: int = 10,
    slip: float = 0.1,
    wall_prob: float = 0.1,
    goal_reward: float = 1.0,
    trap_reward: float = -1.0,
    gamma: float = 0.99,
    rng: Any | None = None,
) -> MDP:
    return generate_stochastic_grid_jax(
        n=n,
        slip=slip,
        wall_prob=wall_prob,
        goal_reward=goal_reward,
        trap_reward=trap_reward,
        gamma=gamma,
        rng=rng,
    )


def generate_four_rooms(
    size: int = 15,
    slip: float = 0.1,
    gamma: float = 0.99,
    goal_reward: float = 1.0,
    trap_reward: float = -1.0,
    rng: Any | None = None,
) -> MDP:
    return generate_four_rooms_jax(
        size=size,
        slip=slip,
        gamma=gamma,
        goal_reward=goal_reward,
        trap_reward=trap_reward,
        rng=rng,
    )


def generate_dense_reward_grid(
    n: int = 10,
    slip: float = 0.1,
    gamma: float = 0.99,
    rng: Any | None = None,
) -> MDP:
    return generate_dense_reward_grid_jax(
        n=n,
        slip=slip,
        gamma=gamma,
        rng=rng,
    )


def generate_frozen_lake(
    size: int = 8,
    slip_prob: float = 1 / 3,
    gamma: float = 0.99,
    hole_positions: list[tuple[int, int]] | None = None,
) -> MDP:
    return generate_frozen_lake_jax(
        size=size,
        slip_prob=slip_prob,
        gamma=gamma,
        hole_positions=hole_positions,
    )


def generate_slippery_cliffwalk(
    slip_prob: float = 0.1,
    gamma: float = 0.99,
) -> MDP:
    return generate_slippery_cliffwalk_jax(
        slip_prob=slip_prob,
        gamma=gamma,
    )


__all__ = [
    "generate_dense_reward_grid",
    "generate_four_rooms",
    "generate_frozen_lake",
    "generate_maze",
    "generate_modified_cliffwalk",
    "generate_slippery_cliffwalk",
    "generate_stochastic_grid",
]
