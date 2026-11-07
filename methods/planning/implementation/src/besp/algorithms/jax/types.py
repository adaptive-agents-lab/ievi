"""
JAX RL Types and Data Structures (Dense Backend).
"""
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


ScalarOrArray = float | int | jax.Array | Any


class MDPData(NamedTuple):
    """Immutable MDP data for JAX operations."""
    P: jnp.ndarray  # (S, A, S) transition probabilities
    R: jnp.ndarray  # (S, A) rewards
    gamma: ScalarOrArray

class AgentState(NamedTuple):
    """Mutable agent state for JAX operations (Dense Backend)."""
    V: jnp.ndarray           # (B, S) value estimates
    
    # Dense Mode Fields
    P_hat: jnp.ndarray       # (B, S, A, S) model estimates
    N_sas: jnp.ndarray       # (B, S, A, S) state-action-next_state counts
    
    # Common Fields
    R_hat: jnp.ndarray       # (B, S, A) reward estimates
    N_sa: jnp.ndarray        # (B, S, A) state-action counts
    R_sum: jnp.ndarray       # (B, S, A) cumulative rewards
    states: jnp.ndarray      # (B,) current states
    J_per_env: jnp.ndarray   # (B,) auxiliary per-environment integer state
    prev_score: jnp.ndarray  # (B,) auxiliary score history
    rng_key: jnp.ndarray     # (B, 2) per-environment PRNG keys
    r_k: jnp.ndarray         # (B, S, A) surrogate rewards (OS-Dyna)
    Q: jnp.ndarray           # (B, S, A) Q-values (Q-Learning)
    bar_r: jnp.ndarray       # (B, S, A) ISP-Dyna reward tracker
    
    # Optional variant scratch space.
    h_s: jnp.ndarray         # (B, S) auxiliary per-state values
    
    # Optional checkpointing workspace.
    R_E_factor: jnp.ndarray   # (B,) auxiliary scalar factors
    V_checkpoint: jnp.ndarray # (B, S) checkpointed value
    steps_since_check: jnp.ndarray # (B,) step counters
    working_V: jnp.ndarray | None = None
    policy_cache: jnp.ndarray | None = None
    waterfall_h: jnp.ndarray | None = None
    waterfall_g_best: jnp.ndarray | None = None
    waterfall_g_candidate: jnp.ndarray | None = None
    waterfall_reward_abs_max: jnp.ndarray | None = None
    waterfall_inner_depth: jnp.ndarray | None = None
    waterfall_last_status: jnp.ndarray | None = None
    waterfall_d_base: jnp.ndarray | None = None
    waterfall_d_refine: jnp.ndarray | None = None
    defect_r: jnp.ndarray | None = None
    defect_avg: jnp.ndarray | None = None
    defect_init: jnp.ndarray | None = None
    defect_clip_fraction: jnp.ndarray | None = None
    defect_gamma_bound: jnp.ndarray | None = None
    replay_states: jnp.ndarray | None = None
    replay_actions: jnp.ndarray | None = None
    replay_next_states: jnp.ndarray | None = None
    replay_row_index: jnp.ndarray | None = None
    replay_sample_index: jnp.ndarray | int = 0

class AlgorithmParams(NamedTuple):
    """Immutable algorithm parameters for JIT compilation."""
    gamma: ScalarOrArray
    lambda_val: ScalarOrArray
    h: ScalarOrArray
    J_init: int
    surrogate_tol: float
    surrogate_iters: int
    max_J: int
    n_s: int
    n_a: int
    disc: ScalarOrArray
    inv_1ph: ScalarOrArray
    h_inv_1ph: ScalarOrArray
    # Parameters for Learning Rate
    lr_start: ScalarOrArray = 0.0
    lr_delay: ScalarOrArray = 0.0
    lr_schedule: int = 0
    # Parameter for Dyna
    planning_steps: ScalarOrArray = -1
    use_vi_solve: bool = True
    inner_batch_size: int = 10
    alpha_h: float = 0.1
    batch_size: ScalarOrArray = 50
    base_h: ScalarOrArray = 1.0
    sampling_distribution: ScalarOrArray = 0
    decay_factor: float = 0.5
    h_init: ScalarOrArray = 1.0
    epsilon: ScalarOrArray = 0.1
    h_terminal: ScalarOrArray = 1.0
    h_schedule_horizon: int = 0
    h_schedule_shape: int = 0
    h_schedule_finish_frac: float = 1.0
    nr_mode: int = 0
    model_error_mode: int = 0
    # Waterfall-NR1 (adaptive h) fields.
    waterfall_h_init: ScalarOrArray = 1.0
    waterfall_h_min: ScalarOrArray = 1.0
    waterfall_decay_factor: float = 0.5
    waterfall_stall_patience: int = 0
    waterfall_alpha: float = 1.0
    waterfall_divergence_mult: float = 2.5
    waterfall_v_max: ScalarOrArray = 0.0
    waterfall_schedule_kind: int = 0
