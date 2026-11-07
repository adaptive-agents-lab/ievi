"""
JAX-accelerated Batched RL Algorithms - Base Utilities.

This module contains shared utilities used by all JAX algorithm implementations:
- Constants (PI_DEFAULT_ITERS)
- State initialization (create_initial_state)
- Policy Iteration kernel 
- Surrogate VI solver
- Sampling functions
- Policy computation helpers
- Learning rate computation

"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from typing import Any, Sequence, Tuple, Union

from .error_modes import jax_rl_stochastic_model_error_code
from .types import AgentState, MDPData

# =============================================================================
# Constants
# =============================================================================

# Default policy-iteration limit for surrogate solves.
# PI converges in 1-5 iterations typically, so 5 is a reasonable limit
PI_DEFAULT_ITERS = 5


def initial_model_estimate_pure(
    *,
    num_envs: int,
    n_s: int,
    n_a: int,
    lambda_val: Any,
    model_error_mode: Any = 0,
    dtype: Any = jnp.float64,
) -> jnp.ndarray:
    """Build the maintained stochastic surrogate model initialization."""
    lam_arr = jnp.asarray(lambda_val, dtype=dtype).reshape(-1)
    if lam_arr.size == 1:
        lam_arr = jnp.broadcast_to(lam_arr, (num_envs,))
    elif lam_arr.shape != (num_envs,):
        lam_arr = jnp.broadcast_to(lam_arr, (num_envs,))
    lam_arr = lam_arr.reshape(num_envs, 1, 1, 1)

    mode_arr = jnp.asarray(model_error_mode, dtype=jnp.int32).reshape(-1)
    if mode_arr.size == 1:
        mode_arr = jnp.broadcast_to(mode_arr, (num_envs,))
    elif mode_arr.shape != (num_envs,):
        mode_arr = jnp.broadcast_to(mode_arr, (num_envs,))
    mode_arr = mode_arr.reshape(num_envs, 1, 1, 1)

    uniform_rows = jnp.ones((num_envs, n_s, n_a, n_s), dtype=dtype) / float(n_s)
    identity_rows = jnp.broadcast_to(
        jnp.eye(n_s, dtype=dtype)[None, :, None, :],
        (num_envs, n_s, n_a, n_s),
    )
    prior_rows = jnp.where(mode_arr == 1, identity_rows, uniform_rows)
    return (1.0 - lam_arr) * uniform_rows + lam_arr * prior_rows


# =============================================================================
# State Initialization
# =============================================================================

def create_initial_state(
    mdp_P: Any,
    mdp_R: Any,
    num_envs: int,
    lambda_val: float,
    error_mode: str = "support_uniform",
    J_init: int = 5,
    seed: int | Sequence[int] | jax.Array = 0,
    replay_states: jax.Array | None = None,
    replay_actions: jax.Array | None = None,
    replay_next_states: jax.Array | None = None,
    replay_row_index: jax.Array | None = None,
) -> Tuple[MDPData, AgentState]:
    """
    Create initial MDP data and agent state.
    
    Args:
        mdp_P: (S, A, S) transition matrix
        mdp_R: (S, A) reward matrix
        num_envs: Number of parallel environments
        lambda_val: Smoothing parameter
        J_init: Initial value for auxiliary per-environment depth state
        seed: Random seed or explicit per-environment seed values
        
    Returns:
        (mdp_data, agent_state) tuple
    """
    n_s, n_a, _ = mdp_P.shape
    
    # Move to JAX arrays
    P = jnp.asarray(mdp_P, dtype=jnp.float64)
    R = jnp.asarray(mdp_R, dtype=jnp.float64)
    
    model_error_mode = jax_rl_stochastic_model_error_code(error_mode)
    P_hat = initial_model_estimate_pure(
        num_envs=num_envs,
        n_s=n_s,
        n_a=n_a,
        lambda_val=lambda_val,
        model_error_mode=model_error_mode,
        dtype=jnp.float64,
    ).copy()
    
    # R_hat is zero-initialized. Algorithms that learn the reward model
    # maintain it empirically (e.g., Standard-Dyna computes R_sum / N_sa);
    # algorithms that do not need a reward model simply never read this field.
    # No algorithm is permitted to seed R_hat with the oracle reward.
    R_hat = jnp.zeros((num_envs, n_s, n_a), dtype=jnp.float64)
    
    # Initialize counts to zeros
    N_sas = jnp.zeros((num_envs, n_s, n_a, n_s), dtype=jnp.float64)
    N_sa = jnp.zeros((num_envs, n_s, n_a), dtype=jnp.float64)
    R_sum = jnp.zeros((num_envs, n_s, n_a), dtype=jnp.float64)
    
    # Value estimates
    V = jnp.zeros((num_envs, n_s), dtype=jnp.float64)
    
    rng_key, states = initialize_env_rng_keys_and_states(seed, num_envs=num_envs, n_s=n_s)
    
    # Optional auxiliary state used by higher-level variants.
    J_per_env = jnp.full((num_envs,), J_init, dtype=jnp.int32)
    prev_score = jnp.full((num_envs,), 1e9, dtype=jnp.float64)
    
    # OS-Dyna / Q-Learning fields
    r_k = jnp.zeros((num_envs, n_s, n_a), dtype=jnp.float64)
    Q = jnp.zeros((num_envs, n_s, n_a), dtype=jnp.float64)
    bar_r = jnp.zeros((num_envs, n_s, n_a), dtype=jnp.float64)
    
    # Optional per-state auxiliary workspace.
    h_s = jnp.zeros((num_envs, n_s), dtype=jnp.float64)
    
    # Optional checkpointing workspace.
    R_E_factor = jnp.ones((num_envs,), dtype=jnp.float64)
    V_checkpoint = jnp.zeros((num_envs, n_s), dtype=jnp.float64)
    steps_since_check = jnp.zeros((num_envs,), dtype=jnp.int32)
    policy_cache = jnp.zeros((num_envs, n_s), dtype=jnp.int32)
    working_V = V
    waterfall_h = jnp.ones((num_envs,), dtype=jnp.float64)
    waterfall_g_best = jnp.full((num_envs,), jnp.inf, dtype=jnp.float64)
    waterfall_g_candidate = jnp.zeros((num_envs,), dtype=jnp.float64)
    waterfall_reward_abs_max = jnp.zeros((num_envs,), dtype=jnp.float64)
    waterfall_inner_depth = jnp.zeros((num_envs,), dtype=jnp.int32)
    waterfall_last_status = jnp.zeros((num_envs,), dtype=jnp.int32)
    waterfall_d_base = jnp.zeros((num_envs,), dtype=jnp.float64)
    waterfall_d_refine = jnp.zeros((num_envs,), dtype=jnp.float64)
    defect_avg = jnp.zeros((num_envs, n_s, n_a), dtype=jnp.float64)
    defect_init = jnp.zeros((num_envs, n_s, n_a), dtype=jnp.float64)
    defect_clip_fraction = jnp.zeros((num_envs,), dtype=jnp.float64)
    defect_gamma_bound = jnp.zeros((num_envs,), dtype=jnp.float64)
    
    mdp_data = MDPData(P=P, R=R, gamma=0.99)  # gamma will be overridden
    agent_state = AgentState(
        V=V,
        P_hat=P_hat,
        R_hat=R_hat,
        N_sas=N_sas,
        # Sparse fields (Dummies)
        N_sa=N_sa,
        R_sum=R_sum,
        states=states,
        J_per_env=J_per_env,
        prev_score=prev_score,
        rng_key=rng_key,
        r_k=r_k,
        Q=Q,
        bar_r=bar_r,
        h_s=h_s,
        R_E_factor=R_E_factor,
        V_checkpoint=V_checkpoint,
        steps_since_check=steps_since_check,
        working_V=working_V,
        policy_cache=policy_cache,
        waterfall_h=waterfall_h,
        waterfall_g_best=waterfall_g_best,
        waterfall_g_candidate=waterfall_g_candidate,
        waterfall_reward_abs_max=waterfall_reward_abs_max,
        waterfall_inner_depth=waterfall_inner_depth,
        waterfall_last_status=waterfall_last_status,
        waterfall_d_base=waterfall_d_base,
        waterfall_d_refine=waterfall_d_refine,
        defect_r=jnp.zeros_like(R_hat),
        defect_avg=defect_avg,
        defect_init=defect_init,
        defect_clip_fraction=defect_clip_fraction,
        defect_gamma_bound=defect_gamma_bound,
        replay_states=replay_states,
        replay_actions=replay_actions,
        replay_next_states=replay_next_states,
        replay_row_index=replay_row_index,
        replay_sample_index=0,
    )
    
    return mdp_data, agent_state


def batched_rng_keys_from_seed_values(seed_values: jnp.ndarray) -> jnp.ndarray:
    """Return one PRNG key per environment seed value."""
    seed_values = jnp.asarray(seed_values, dtype=jnp.int32).reshape(-1)
    return jax.vmap(random.PRNGKey)(seed_values)


def initialize_env_rng_keys_and_states_from_seed_values(
    seed_values: jnp.ndarray,
    *,
    n_s: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build per-environment RNG keys and initial states from explicit seeds."""
    seed_keys = batched_rng_keys_from_seed_values(seed_values)
    split_keys = jax.vmap(lambda key: random.split(key, 2))(seed_keys)
    rng_keys = split_keys[:, 0, :]
    state_keys = split_keys[:, 1, :]
    states = jax.vmap(lambda key: random.randint(key, (), 0, n_s, dtype=jnp.int32))(state_keys)
    return rng_keys, states


def initialize_env_rng_keys_and_states(
    seed: int | Sequence[int] | jax.Array,
    *,
    num_envs: int,
    n_s: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build per-environment RNG keys and initial states.

    Scalar seeds preserve the old initial-state draw while upgrading the runtime
    state to one PRNG stream per environment. Explicit seed arrays route each
    environment through its own stable `(config, seed)` replay stream.
    """
    seed_values = jnp.asarray(seed, dtype=jnp.int32)
    if seed_values.ndim == 0:
        base_key = random.PRNGKey(seed_values.item())
        base_rng_key, state_key = random.split(base_key)
        states = random.randint(state_key, (num_envs,), 0, n_s, dtype=jnp.int32)
        env_ids = jnp.arange(num_envs, dtype=jnp.int32)
        rng_keys = jax.vmap(lambda env_id: random.fold_in(base_rng_key, env_id))(env_ids)
        return rng_keys, states
    flat_seed_values = seed_values.reshape(-1)
    if flat_seed_values.shape[0] != num_envs:
        raise ValueError(f"Expected {num_envs} per-environment seeds, received {flat_seed_values.shape[0]}.")
    return initialize_env_rng_keys_and_states_from_seed_values(flat_seed_values, n_s=n_s)


def ensure_batched_rng_keys(rng_key: jnp.ndarray, *, num_envs: int) -> jnp.ndarray:
    """Normalize RNG storage to one key per environment.

    This accepts legacy scalar keys in tests and older call sites, but the
    maintained runtime should pass `(B, 2)` key arrays.
    """
    key_array = jnp.asarray(rng_key)
    if key_array.ndim == 1:
        env_ids = jnp.arange(num_envs, dtype=jnp.int32)
        return jax.vmap(lambda env_id: random.fold_in(key_array, env_id))(env_ids)
    if key_array.ndim == 2:
        if key_array.shape[0] != num_envs:
            raise ValueError(f"Expected rng_key leading dimension {num_envs}, got {key_array.shape[0]}.")
        return key_array
    raise ValueError(f"Unsupported rng_key rank {key_array.ndim}; expected 1 or 2.")


def split_batched_rng_keys(
    rng_key: jnp.ndarray,
    *,
    num_envs: int,
    num_splits: int,
) -> tuple[jnp.ndarray, ...]:
    """Split one key per environment into `num_splits + 1` key arrays."""
    env_keys = ensure_batched_rng_keys(rng_key, num_envs=num_envs)
    split_keys = jax.vmap(lambda key: random.split(key, num_splits + 1))(env_keys)
    return tuple(split_keys[:, index, :] for index in range(num_splits + 1))


def batched_random_int(
    keys: jnp.ndarray,
    *,
    low: int,
    high: int,
    dtype: jnp.dtype = jnp.int32,
) -> jnp.ndarray:
    """Sample one integer per environment key."""
    return jax.vmap(lambda key: random.randint(key, (), low, high, dtype=dtype))(keys)


def batched_random_uniform(keys: jnp.ndarray, *, dtype: jnp.dtype = jnp.float64) -> jnp.ndarray:
    """Sample one uniform scalar per environment key."""
    return jax.vmap(lambda key: random.uniform(key, (), dtype=dtype))(keys)


def is_scalar_like(value: Any) -> bool:
    return jnp.asarray(value).ndim == 0


def as_column_parameter(value: Any, num_envs: int, *, dtype: Any) -> jnp.ndarray:
    if is_scalar_like(value):
        return jnp.full((num_envs, 1), value, dtype=dtype)
    return jnp.asarray(value, dtype=dtype).reshape(-1, 1)


def as_vector_parameter(value: Any, num_envs: int, *, dtype: Any) -> jnp.ndarray:
    if is_scalar_like(value):
        return jnp.full((num_envs,), value, dtype=dtype)
    return jnp.asarray(value, dtype=dtype).reshape(-1)


# =============================================================================
# Policy Iteration Kernel (Sole Backend)
# =============================================================================


def _pi_cond(carry):
    """Continue if policies have not stabilized and max_iters not reached."""
    pi, pi_prev, U, P_hat, shifted_R, disc, i, max_iters = carry
    # Check if policy changed: any(pi != pi_prev)
    # AND i < max_iters
    # Initial iter (i=0): pi_prev is -1, so effectively always runs at least once
    policy_changed = jnp.any(pi != pi_prev)
    iter_ok = i < max_iters
    # Ensure scalar boolean output by explicitly reducing to scalar
    result = policy_changed & iter_ok
    # Force scalar: if result is somehow batched, take first element
    return jnp.asarray(result).reshape(()).astype(bool)

def _pi_body(carry):
    """
    Policy Iteration Step:
    1. Evaluate current policy pi exactly: (I - disc * P_pi) U = R_pi
    2. Improve policy: pi_new = argmax(R + disc * P * U)
    """
    pi, _, _, P_hat, shifted_R, disc, i, max_iters = carry
    
    B, S, A, _ = P_hat.shape
    
    # --- Step A: Gather System Matrices for Policy pi ---
    # Indices for advanced indexing
    b_idx = jnp.arange(B)[:, None]  # (B, 1)
    s_idx = jnp.arange(S)[None, :]  # (1, S)
    
    # P_pi: (B, S, S) - Transition matrix induced by pi
    # P_hat is (B, S, A, S) -> Select action pi[b,s] for each state
    P_pi = P_hat[b_idx, s_idx, pi, :]
    
    # R_pi: (B, S) - Reward vector induced by pi
    # shifted_R is (B, S, A) -> Select action pi[b,s]
    R_pi = shifted_R[b_idx, s_idx, pi]
    
    # --- Step B: Exact Policy Evaluation (Linear Solve) ---
    # Equation: (I - disc * P_pi) U = R_pi
    
    # Expand indices for broadcasting
    # disc is (B, 1, 1).
    # identity is (S, S) -> (B, S, S)
    identity = jnp.eye(S)
    I_batch = jnp.broadcast_to(identity, (B, S, S))
    
    # Construct A matrix: (B, S, S)
    # With JAX x64 enabled globally, we're already in float64 precision
    disc_bc = disc  # (B, 1, 1)
    A_mat = I_batch - disc_bc * P_pi
    R_vec = R_pi[..., None]  # (B, S, 1)
    
    # Linear Solve: A * x = b => x = A^-1 b
    # Result U is (B, S, 1)
    U_solve = jnp.linalg.solve(A_mat, R_vec)
    
    # Squeeze to (B, S)
    U_eval = U_solve.squeeze(-1)
    
    # --- Step C: Policy Improvement ---
    # Q = shifted_R + disc * (P_hat @ U_eval)
    # P_hat: (B, S, A, S), U_eval: (B, S) -> (B, S, A)
    term_fut = jnp.einsum('bzas,bs->bza', P_hat, U_eval)
    Q = shifted_R + disc * term_fut
    
    pi_new = jnp.argmax(Q, axis=2) # (B, S)
    
    return (pi_new, pi, U_eval, P_hat, shifted_R, disc, i + 1, max_iters)


def _solve_policy_iteration(P_hat, shifted_R, U_init, disc, max_iters):
    """Driver for Policy Iteration backend.
    
    Uses lax.fori_loop with fixed iterations instead of lax.while_loop
    to avoid JAX tracing issues with nested loops.
    
    Note: All inputs are cast to float64 for numerical stability and 
    dtype consistency (required by JAX 0.4.35's stricter lax.fori_loop).
    """
    # Cast all inputs to float64 for consistency (lax.fori_loop requires matching dtypes)
    P_hat = P_hat.astype(jnp.float64)
    shifted_R = shifted_R.astype(jnp.float64)
    U_init = U_init.astype(jnp.float64)
    disc = disc.astype(jnp.float64)
    
    # Fixed-iteration PI body
    def pi_step(i, carry):
        U, P_hat, shifted_R, disc = carry
        
        B, S, A, _ = P_hat.shape
        
        # Greedy policy from current U
        term_fut = jnp.einsum('bzas,bs->bza', P_hat, U)
        Q = shifted_R + disc * term_fut
        pi = jnp.argmax(Q, axis=2)  # (B, S)
        
        # Policy Evaluation: Solve (I - disc * P_pi) U = R_pi
        b_idx = jnp.arange(B)[:, None]
        s_idx = jnp.arange(S)[None, :]
        P_pi = P_hat[b_idx, s_idx, pi, :]  # (B, S, S)
        R_pi = shifted_R[b_idx, s_idx, pi]  # (B, S)
        
        identity = jnp.eye(S)
        I_batch = jnp.broadcast_to(identity, (B, S, S))
        A_mat = I_batch - disc * P_pi
        R_vec = R_pi[..., None]  # (B, S, 1)
        
        U_new = jnp.linalg.solve(A_mat, R_vec).squeeze(-1)  # (B, S)
        
        return (U_new, P_hat, shifted_R, disc)
    
    # Run fixed number of PI iterations
    init_carry = (U_init, P_hat, shifted_R, disc)
    final_carry = lax.fori_loop(0, max_iters, pi_step, init_carry)
    
    U_final = final_carry[0]
    return U_final


@jax.jit
def surrogate_vi_solve(
    P_hat: jnp.ndarray,
    shifted_R: jnp.ndarray,
    U_init: jnp.ndarray,
    disc: jnp.ndarray,
    tol: float = 1e-6,
    max_iters: int = PI_DEFAULT_ITERS,
) -> jnp.ndarray:
    """
    Policy Iteration solver for Surrogate MDPs.
    
    Uses exact linear solve per step: (I - gP)^-1 R.
    Propagates rewards infinitely in one step via direct solve.
    
    Note: Complexity is O(S^3) per iteration. Best for S < 500.
    PI typically converges in 1-5 iterations (max_iters=25 is safe cap).
    
    Args:
        P_hat: (B, S, A, S) model
        shifted_R: (B, S, A) reward
        U_init: (B, S) initial value
        disc: (B, 1, 1) effective discount
        tol: Unused (kept for API compatibility)
        max_iters: Safety cap (default: 25)
        
    Returns:
        U: (B, S) converged value function
    """
    return _solve_policy_iteration(P_hat, shifted_R, U_init, disc, max_iters)


# =============================================================================
# Sample Step (Generative Model Access)
# =============================================================================

@jax.jit
def sample_step_pure(
    mdp_P: jnp.ndarray,
    mdp_R: jnp.ndarray,
    states: jnp.ndarray,
    rng_key: jnp.ndarray,
    n_s: int,
    n_a: int,
    replay_states: jnp.ndarray | None = None,
    replay_actions: jnp.ndarray | None = None,
    replay_next_states: jnp.ndarray | None = None,
    replay_row_index: jnp.ndarray | None = None,
    replay_sample_index: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure JAX sampling step."""
    if (
        replay_states is not None
        and replay_actions is not None
        and replay_next_states is not None
        and replay_row_index is not None
    ):
        row_index = jnp.asarray(replay_row_index, dtype=jnp.int32)
        sample_index = jnp.asarray(replay_sample_index, dtype=jnp.int32).reshape(())
        sampled_states = jnp.asarray(replay_states, dtype=jnp.int32)[row_index, sample_index]
        sampled_actions = jnp.asarray(replay_actions, dtype=jnp.int32)[row_index, sample_index]
        sampled_next_states = jnp.asarray(replay_next_states, dtype=jnp.int32)[row_index, sample_index]
        rewards = mdp_R[sampled_states, sampled_actions]
        return (
            sampled_states,
            sampled_actions,
            rewards,
            sampled_next_states,
            sampled_next_states,
            rng_key,
        )
    num_envs = states.shape[0]
    rng_key, key_action, key_state, key_trans = split_batched_rng_keys(
        rng_key,
        num_envs=num_envs,
        num_splits=3,
    )

    actions = batched_random_int(key_action, low=0, high=n_a, dtype=jnp.int32)
    new_states = batched_random_int(key_state, low=0, high=n_s, dtype=jnp.int32)
    
    # Get transition probs
    # mdp_P is (S, A, S), gathered by (B,) states and (B,) actions -> (B, S)
    P_probs = mdp_P[states, actions]
    
    # Sample next state via CDF inversion
    cdf = jnp.cumsum(P_probs, axis=1)
    r_rand = batched_random_uniform(key_trans, dtype=jnp.float64)[:, None]
    mask = (cdf > r_rand).astype(jnp.int32)
    next_states = jnp.argmax(mask, axis=1).astype(jnp.int32)
    
    rewards = mdp_R[states, actions]
    
    return states, actions, rewards, next_states, new_states, rng_key


def sample_generative_transition_pure(
    mdp_P: jnp.ndarray,
    mdp_R: jnp.ndarray,
    rng_key: jnp.ndarray,
    num_envs: int,
    n_s: int,
    n_a: int,
    sampling_distribution: int = 0,
    replay_states: jnp.ndarray | None = None,
    replay_actions: jnp.ndarray | None = None,
    replay_next_states: jnp.ndarray | None = None,
    replay_row_index: jnp.ndarray | None = None,
    replay_sample_index: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generative-model sampling hook for model-based RL baselines.

    V1 supports only `sampling_distribution == 0`, corresponding to uniform
    sampling over state-action pairs.
    """
    if (
        replay_states is not None
        and replay_actions is not None
        and replay_next_states is not None
        and replay_row_index is not None
    ):
        row_index = jnp.asarray(replay_row_index, dtype=jnp.int32)
        sample_index = jnp.asarray(replay_sample_index, dtype=jnp.int32).reshape(())
        states = jnp.asarray(replay_states, dtype=jnp.int32)[row_index, sample_index]
        actions = jnp.asarray(replay_actions, dtype=jnp.int32)[row_index, sample_index]
        next_states = jnp.asarray(replay_next_states, dtype=jnp.int32)[row_index, sample_index]
        rewards = mdp_R[states, actions]
        return states, actions, rewards, next_states, rng_key
    del sampling_distribution
    rng_key, key_state, key_action, key_trans = split_batched_rng_keys(
        rng_key,
        num_envs=num_envs,
        num_splits=3,
    )
    states = batched_random_int(key_state, low=0, high=n_s, dtype=jnp.int32)
    actions = batched_random_int(key_action, low=0, high=n_a, dtype=jnp.int32)
    probs = mdp_P[states, actions]
    cdf = jnp.cumsum(probs, axis=1)
    draws = batched_random_uniform(key_trans, dtype=jnp.float64)[:, None]
    next_states = jnp.argmax((cdf > draws).astype(jnp.int32), axis=1).astype(jnp.int32)
    rewards = mdp_R[states, actions]
    return states, actions, rewards, next_states, rng_key


@jax.jit
def epsilon_greedy_actions_pure(
    greedy_actions: jnp.ndarray,
    epsilon: jnp.ndarray,
    n_a: int,
    rng_key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample epsilon-greedy actions for a batched trajectory policy."""
    num_envs = greedy_actions.shape[0]
    rng_key, key_mix, key_rand = split_batched_rng_keys(
        rng_key,
        num_envs=num_envs,
        num_splits=2,
    )
    epsilon_vec = jnp.asarray(epsilon, dtype=jnp.float64).reshape(-1)
    random_actions = batched_random_int(key_rand, low=0, high=n_a, dtype=jnp.int32)
    explore_mask = batched_random_uniform(key_mix, dtype=jnp.float64) < epsilon_vec
    actions = jnp.where(explore_mask, random_actions, greedy_actions).astype(jnp.int32)
    return actions, rng_key


@jax.jit
def sample_trajectory_transition_pure(
    mdp_P: jnp.ndarray,
    mdp_R: jnp.ndarray,
    states: jnp.ndarray,
    greedy_actions: jnp.ndarray,
    epsilon: jnp.ndarray,
    rng_key: jnp.ndarray,
    n_a: int,
    replay_states: jnp.ndarray | None = None,
    replay_actions: jnp.ndarray | None = None,
    replay_next_states: jnp.ndarray | None = None,
    replay_row_index: jnp.ndarray | None = None,
    replay_sample_index: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample one physical transition from the current trajectory state."""
    if (
        replay_states is not None
        and replay_actions is not None
        and replay_next_states is not None
        and replay_row_index is not None
    ):
        row_index = jnp.asarray(replay_row_index, dtype=jnp.int32)
        sample_index = jnp.asarray(replay_sample_index, dtype=jnp.int32).reshape(())
        sampled_states = jnp.asarray(replay_states, dtype=jnp.int32)[row_index, sample_index]
        sampled_actions = jnp.asarray(replay_actions, dtype=jnp.int32)[row_index, sample_index]
        sampled_next_states = jnp.asarray(replay_next_states, dtype=jnp.int32)[row_index, sample_index]
        rewards = mdp_R[sampled_states, sampled_actions]
        return (
            sampled_states,
            sampled_actions,
            rewards,
            sampled_next_states,
            sampled_next_states,
            rng_key,
        )
    actions, rng_key = epsilon_greedy_actions_pure(greedy_actions, epsilon, n_a, rng_key)
    rng_key, key_trans = split_batched_rng_keys(
        rng_key,
        num_envs=states.shape[0],
        num_splits=1,
    )
    probs = mdp_P[states, actions]
    cdf = jnp.cumsum(probs, axis=1)
    draws = batched_random_uniform(key_trans, dtype=jnp.float64)[:, None]
    next_states = jnp.argmax((cdf > draws).astype(jnp.int32), axis=1).astype(jnp.int32)
    rewards = mdp_R[states, actions]
    return states, actions, rewards, next_states, next_states, rng_key


def advance_replay_sample_index(
    replay_sample_index: jnp.ndarray | int,
    replay_states: jnp.ndarray | None,
    *,
    delta: int | jnp.ndarray = 1,
) -> jnp.ndarray | int:
    if replay_states is None:
        return replay_sample_index
    return jnp.asarray(replay_sample_index, dtype=jnp.int32) + jnp.asarray(delta, dtype=jnp.int32)


# =============================================================================
# MSBE Computation
# =============================================================================

@jax.jit
def _compute_msbe_pure(
    V: jnp.ndarray, 
    P_emp: jnp.ndarray, 
    R_emp: jnp.ndarray, 
    N_sa: jnp.ndarray,
    gamma: float
) -> jnp.ndarray:
    """Compute mean squared Bellman error per environment."""
    total_counts = jnp.sum(N_sa, axis=(1, 2))
    
    term_fut = jnp.einsum('bzas,bs->bza', P_emp, V)
    
    # Ensure gamma broadcasts correctly with (B, S, A) tensors
    gamma_3d = jnp.asarray(gamma, dtype=jnp.float64).reshape(-1, 1, 1)
    
    Q_sa = R_emp + gamma_3d * term_fut
    diff = Q_sa - V[:, :, None]
    
    weighted_sq = jnp.sum((diff ** 2) * N_sa, axis=(1, 2))
    return weighted_sq / (total_counts + 1e-6)


# =============================================================================
# Policy Computation Helpers
# =============================================================================

@jax.jit
def _compute_policy_from_q(Q: jnp.ndarray) -> jnp.ndarray:
    """Argmax policy from Q."""
    return jnp.argmax(Q, axis=2)

@jax.jit
def _compute_policy_os_dyna(
    r_k: jnp.ndarray,
    P_hat: jnp.ndarray,
    V: jnp.ndarray,
    gamma: Union[float, jnp.ndarray]
) -> jnp.ndarray:
    """Argmax policy for OS-Dyna (using surrogate Q)."""
    # Q = r_k + gamma * P_hat @ V
    gamma_3d = jnp.asarray(gamma, dtype=jnp.float64).reshape(-1, 1, 1)
    term_fut = jnp.einsum('bzas,bs->bza', P_hat, V)
    Q = r_k + gamma_3d * term_fut
    return jnp.argmax(Q, axis=2)


@jax.jit
def _compute_policy_isp_dyna(
    bar_r: jnp.ndarray,
    P_hat: jnp.ndarray,
    V: jnp.ndarray,
    gamma: Union[float, jnp.ndarray],
    h: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Argmax policy for ISP-Dyna using the manuscript final extraction rule."""
    gamma_vec = jnp.asarray(gamma, dtype=jnp.float64).reshape(-1)
    h_vec = jnp.asarray(h, dtype=jnp.float64).reshape(-1)
    disc = ((h_vec * gamma_vec) / (1.0 + h_vec)).reshape(-1, 1, 1)
    term_fut = jnp.einsum("bzas,bs->bza", P_hat, V)
    q_values = bar_r + disc * term_fut
    return jnp.argmax(q_values, axis=2)


# =============================================================================
# Learning Rate Computation
# =============================================================================

@jax.jit
def compute_lr_pure(k: int, start_lr: jnp.ndarray, delay: jnp.ndarray, schedule_type: int) -> jnp.ndarray:
    """
    Compute learning rate.
    
    schedule_type: 0 = constant_delay, 1 = rescaled_linear
    """
    start_lr = start_lr.reshape(-1)
    delay = delay.reshape(-1)
    
    # Constant Delay (schedule=0)
    # lr = start_lr for k < delay, start_lr / (k - delay + 1) for k >= delay
    mask_warmup = (k < delay)
    denom_cd = jnp.where(mask_warmup, jnp.ones_like(delay), k - delay + 1.0)
    lr_cd = start_lr / denom_cd
    
    # Rescaled Linear (schedule=1)
    # lr = start_lr / (1 + (1 - u) * (k + 1))  where u = delay
    u = delay
    lr_rl = start_lr / (1.0 + (1.0 - u) * (k + 1.0))
    
    return jnp.where(schedule_type == 0, lr_cd, lr_rl)


# =============================================================================
# Exact Policy Evaluation (Pure JAX)
# =============================================================================

@jax.jit
def evaluate_policy_vectors_batched_pure(
    mdp_P: jnp.ndarray, # (S, A, S) or (B, S, A, S)
    mdp_R: jnp.ndarray, # (S, A) or (B, S, A)
    policies: jnp.ndarray, # (B, S) indices
    gamma: jnp.ndarray, # (B,)
 ) -> jnp.ndarray:
    """
    Pure JAX exact policy evaluation: V^pi = (I - gamma P^pi)^-1 R^pi.
    
    Args:
        mdp_P: Transition tensor. Can be shared (S,A,S) or batched (B,S,A,S).
        mdp_R: Reward tensor. Can be shared (S,A) or batched (B,S,A).
        policies: Integer indices of actions (B, S).
        gamma: Discount factors (B,).

    Returns:
        Full policy values: (B, S).
    """
    B = policies.shape[0]
    S = policies.shape[1]
    
    # Broadcast P/R if needed
    if mdp_P.ndim == 3: # (S, A, S) -> (B, S, A, S)
        mdp_P = jnp.broadcast_to(mdp_P, (B, *mdp_P.shape))
    if mdp_R.ndim == 2: # (S, A) -> (B, S, A)
        mdp_R = jnp.broadcast_to(mdp_R, (B, *mdp_R.shape))
        
    # Gather P^pi and R^pi
    # P: (B, S, A, S) -> gather A using policies (B, S)
    # R: (B, S, A) -> gather A using policies (B, S)
    
    # Advanced indexing
    # We want P[b, s, pi[b,s], :]
    b_idx = jnp.arange(B)[:, None] # (B, 1)
    s_idx = jnp.arange(S)[None, :] # (1, S)
    
    # (B, S, S)
    P_pi = mdp_P[b_idx, s_idx, policies, :] 
    
    # (B, S)
    R_pi = mdp_R[b_idx, s_idx, policies]
    
    # Solve (I - gamma * P_pi) V = R_pi
    identity = jnp.eye(S) # (S, S)
    I_batch = jnp.broadcast_to(identity, (B, S, S))
    
    # Gamma broadcast: (B,) -> (B, 1, 1)
    gamma_batch = gamma[:, None, None]
    
    A_mat = I_batch - gamma_batch * P_pi

    # Linear Solve (batched)
    # JAX solve requires (..., M, 1) for vector b
    V_pi = jnp.linalg.solve(A_mat, R_pi[..., None]).squeeze(-1) # (B, S)
    return V_pi


@jax.jit
def evaluate_policy_start_values_batched_pure(
    mdp_P: jnp.ndarray, # (S, A, S) or (B, S, A, S)
    mdp_R: jnp.ndarray, # (S, A) or (B, S, A)
    policies: jnp.ndarray, # (B, S) indices
    gamma: jnp.ndarray, # (B,)
    start_state_idx: int = 0,
) -> jnp.ndarray:
    """Pure JAX exact start-state policy evaluation without returning full vectors."""
    B = policies.shape[0]
    S = policies.shape[1]

    if mdp_P.ndim == 3:
        mdp_P = jnp.broadcast_to(mdp_P, (B, *mdp_P.shape))
    if mdp_R.ndim == 2:
        mdp_R = jnp.broadcast_to(mdp_R, (B, *mdp_R.shape))

    b_idx = jnp.arange(B)[:, None]
    s_idx = jnp.arange(S)[None, :]
    P_pi = mdp_P[b_idx, s_idx, policies, :]
    R_pi = mdp_R[b_idx, s_idx, policies]

    identity = jnp.eye(S)
    I_batch = jnp.broadcast_to(identity, (B, S, S))
    gamma_batch = gamma[:, None, None]
    A_mat = I_batch - gamma_batch * P_pi

    V_start = jnp.linalg.solve(A_mat, R_pi[..., None]).squeeze(-1)[:, start_state_idx]
    return V_start


@jax.jit
def evaluate_batched_pure(
    mdp_P: jnp.ndarray, # (S, A, S) or (B, S, A, S)
    mdp_R: jnp.ndarray, # (S, A) or (B, S, A)
    policies: jnp.ndarray, # (B, S) indices
    gamma: jnp.ndarray, # (B,)
    start_state_idx: int = 0
) -> jnp.ndarray:
    """Pure JAX exact start-state policy evaluation wrapper."""
    return evaluate_policy_start_values_batched_pure(
        mdp_P,
        mdp_R,
        policies,
        gamma,
        start_state_idx=start_state_idx,
    )


# =============================================================================
# Benchmark utilities
# =============================================================================

def benchmark_vi(
    n_s: int = 48,
    n_a: int = 4,
    num_envs: int = 100,
    num_iters: int = 1000,
    gamma: float = 0.99,
) -> dict:
    """
    Benchmark JAX VI solve.
    
    Returns timing information.
    """
    import time
    
    # Create random MDP
    key = random.PRNGKey(0)
    key, k1, k2, k3 = random.split(key, 4)
    
    P_hat = random.uniform(k1, (num_envs, n_s, n_a, n_s))
    P_hat = P_hat / P_hat.sum(axis=-1, keepdims=True)  # Normalize
    
    shifted_R = random.uniform(k2, (num_envs, n_s, n_a))
    U_init = random.uniform(k3, (num_envs, n_s))
    
    disc = (1.0 * gamma) / 2.0  # h=1
    
    # Warmup JIT
    _ = surrogate_vi_solve(P_hat, shifted_R, U_init, disc, 1e-6, num_iters)
    jax.block_until_ready(_)
    
    # Benchmark
    start = time.perf_counter()
    result = surrogate_vi_solve(P_hat, shifted_R, U_init, disc, 1e-6, num_iters)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - start
    
    return {
        "elapsed_ms": elapsed * 1000,
        "iters": num_iters,
        "envs": num_envs,
        "states": n_s,
        "actions": n_a,
        "ms_per_iter": elapsed * 1000 / num_iters,
    }
