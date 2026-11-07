"""
Environment factory for creating MDP instances.
"""

import jax.numpy as jnp

from ..core.types import MDP
from ..environments import (
    generate_modified_cliffwalk,
    generate_garnet,
    generate_maze,
    generate_chain,
    generate_river_swim,
    generate_four_rooms,
    generate_stochastic_grid,
    generate_dense_reward_grid,
    generate_taxi,
    generate_mini_breakout,
    generate_frozen_lake,
    generate_slippery_cliffwalk,
)
from ..environments.jax_native import create_environment_jax_native


def create_environment(env_name: str, **kwargs) -> MDP:
    """
    Factory to create an MDP instance based on name and parameters.
    
    Args:
        env_name: Name of the environment ('cliffwalk', 'garnet', 'maze', etc.)
        **kwargs: Environment-specific parameters including:
            - gamma: Discount factor (default: 0.99)
            - n_states: Number of states (for scalable environments)
            - n_actions: Number of actions (for scalable environments)
            - slip_prob: Slip probability (for stochastic environments)
            - size: Grid size (for grid environments)
            - rng: Random number generator for stochastic environments
            
    Returns:
        MDP instance with P (transition), R (reward), and gamma
        
    Raises:
        ValueError: If env_name is not recognized
    """
    gamma = kwargs.get("gamma", 0.99)
    rng = kwargs.get("rng")

    if env_name == "cliffwalk":
        return generate_modified_cliffwalk(gamma=gamma)
    
    elif env_name == "garnet":
        return generate_garnet(
            n_s=kwargs.get("n_states", 30), 
            n_a=kwargs.get("n_actions", 4), 
            branching=kwargs.get("branching"),
            reward_states=kwargs.get("reward_states"),
            gamma=gamma,
            rng=rng
        )
    
    elif env_name == "maze":
        return generate_maze(gamma=gamma)
    
    elif env_name == "chain":
        return generate_chain(n_s=kwargs.get("n_states", 20), gamma=gamma)
    
    elif env_name == "river_swim":
        return generate_river_swim(n_s=kwargs.get("n_states", 6), gamma=gamma)
    
    elif env_name == "four_rooms":
        return generate_four_rooms(gamma=gamma)
    
    elif env_name == "stochastic_grid":
        return generate_stochastic_grid(
            n=kwargs.get("size", 5),
            gamma=gamma,
            rng=rng
        )
    
    elif env_name == "dense_reward_grid":
        return generate_dense_reward_grid(
            n=kwargs.get("size", 5),
            gamma=gamma
        )
    
    elif env_name == "taxi":
        return generate_taxi(gamma=gamma)
    
    elif env_name == "mini_breakout":
        return generate_mini_breakout(gamma=gamma)
    
    elif env_name == "frozen_lake":
        return generate_frozen_lake(
            size=kwargs.get("size", 8),
            slip_prob=kwargs.get("slip_prob", 1/3),
            gamma=gamma,
            hole_positions=kwargs.get("hole_positions"),
        )
    
    elif env_name == "slippery_cliffwalk":
        return generate_slippery_cliffwalk(
            slip_prob=kwargs.get("slip_prob", 0.1),
            gamma=gamma,
        )
    
    else:
        raise ValueError(f"Unknown environment: {env_name}. "
                        f"Available: cliffwalk, garnet, maze, chain, river_swim, "
                        f"four_rooms, stochastic_grid, dense_reward_grid, taxi, "
                        f"mini_breakout, frozen_lake, slippery_cliffwalk")


def create_environment_jax(env_name: str, **kwargs) -> MDP:
    """Create an MDP directly on the active JAX backend for maintained environments."""

    try:
        return create_environment_jax_native(env_name, **kwargs)
    except ValueError:
        mdp = create_environment(env_name, **kwargs)
        return MDP(
            P=jnp.asarray(mdp.P, dtype=jnp.float64),
            R=jnp.asarray(mdp.R, dtype=jnp.float64),
            gamma=mdp.gamma,
        )
