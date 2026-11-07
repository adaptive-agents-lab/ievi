"""
JAX-accelerated Dyna Algorithm Implementations.

This module contains the pure JAX implementations of Dyna algorithms:
- JaxBatchedDyna: Standard Dyna with full surrogate-solve and sample-based planning modes
- JaxBatchedOSDyna: OS-Dyna (Operator Splitting Dyna)

"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from functools import partial
from jax import random
from typing import Any, Tuple, Union

from .error_modes import jax_rl_stochastic_model_error_code
from .types import AgentState, MDPData, AlgorithmParams
from .base import (
    PI_DEFAULT_ITERS,
    advance_replay_sample_index,
    as_column_parameter,
    as_vector_parameter,
    batched_random_int,
    batched_random_uniform,
    _compute_policy_isp_dyna,
    create_initial_state,
    ensure_batched_rng_keys,
    initial_model_estimate_pure,
    sample_generative_transition_pure,
    sample_trajectory_transition_pure,
    sample_step_pure,
    split_batched_rng_keys,
    compute_lr_pure,
    surrogate_vi_solve,
    _compute_policy_from_q,
    _compute_policy_os_dyna,
)


# =============================================================================
# OS-Dyna Step Function
# =============================================================================

@jax.jit
def os_dyna_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """Pure JAX step for OS-Dyna."""
    
    # 1. Sample
    (s, a, r, ns, new_states, rng_key) = sample_step_pure(
        mdp.P,
        mdp.R,
        state.states,
        state.rng_key,
        params.n_s,
        params.n_a,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )
    
    # 2. Compute LR
    lr = compute_lr_pure(k, params.lr_start, params.lr_delay, params.lr_schedule)
    lr = lr.reshape(-1) # (B,)
    
    # 3. Update Counts
    num_envs = state.V.shape[0]
    b_idx = jnp.arange(num_envs)
    
    N_sas = state.N_sas.at[b_idx, s, a, ns].add(1)
    # N_sa is not required for the OS-Dyna update itself, but keeping it in the
    # shared state simplifies the wrapper APIs and statistics hooks.
    
    # 4. Update P_hat (Incremental)
    row_c = N_sas[b_idx, s, a] # (B, S)
    _, _, row_p = _compute_model_error_noisy_row_pure(
        row_counts=row_c,
        row_states=s,
        n_s=params.n_s,
        lambda_val=params.lambda_val,
        model_error_mode=params.model_error_mode,
    )
    P_hat = state.P_hat.at[b_idx, s, a].set(row_p)
    
    # 5. Update r_k
    # target = r + gamma * V[ns] - gamma * sum(P_hat[s,a] * V)
    V_ns = state.V[b_idx, ns]
    P_hat_sa = P_hat[b_idx, s, a] # (B, S)
    phat_v = jnp.sum(P_hat_sa * state.V, axis=1) # (B,)
    
    # gamma must be (B,)
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
        
    target_term = r.reshape(-1) + gamma_vec * V_ns - gamma_vec * phat_v
    
    old_rk = state.r_k[b_idx, s, a]
    diff_rk = target_term - old_rk
    
    r_k = state.r_k.at[b_idx, s, a].add(lr * diff_rk)
    
    # 6. Solve the surrogate control problem with fixed-step policy iteration
    # Q_surr = r_k + gamma * P_hat @ V
    # We use surrogate_vi_solve with shifted_R = r_k and disc = gamma
    
    # Gamma for the surrogate solver needs to broadcast as (B, 1, 1).
    disc_vi = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1, 1, 1)
    
    U_next = surrogate_vi_solve(
        P_hat, r_k, state.V, disc_vi, 
        tol=params.surrogate_tol, max_iters=params.surrogate_iters,
    )
    
    new_state = AgentState(
        V=U_next,
        P_hat=P_hat,
        R_hat=state.R_hat,
        N_sas=N_sas,
        N_sa=state.N_sa,
        R_sum=state.R_sum,
        states=new_states,
        J_per_env=state.J_per_env,
        prev_score=state.prev_score,
        rng_key=rng_key,
        r_k=r_k,
        Q=state.Q,
        bar_r=state.bar_r,
        h_s=state.h_s,
        R_E_factor=state.R_E_factor,
        V_checkpoint=state.V_checkpoint,
        steps_since_check=state.steps_since_check,
        working_V=state.working_V,
        policy_cache=state.policy_cache,
        waterfall_h=(
            state.waterfall_h
            if state.waterfall_h is not None
            else jnp.ones((state.V.shape[0],), dtype=jnp.float64)
        ),
        waterfall_g_best=state.waterfall_g_best,
        waterfall_g_candidate=state.waterfall_g_candidate,
        waterfall_reward_abs_max=state.waterfall_reward_abs_max,
        waterfall_inner_depth=state.waterfall_inner_depth,
        waterfall_last_status=state.waterfall_last_status,
        waterfall_d_base=state.waterfall_d_base,
        waterfall_d_refine=state.waterfall_d_refine,
        defect_r=state.defect_r,
        defect_avg=state.defect_avg,
        defect_init=state.defect_init,
        defect_clip_fraction=state.defect_clip_fraction,
        defect_gamma_bound=state.defect_gamma_bound,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=advance_replay_sample_index(
            state.replay_sample_index,
            state.replay_states,
        ),
    )
    
    return new_state, jnp.mean(U_next[:, 0])


@jax.jit
def run_n_steps_os_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    """Run N steps of OS-Dyna in a single JIT-compiled loop."""
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = os_dyna_step_pure(bst, mdp, params, curr_k)
        return (nst, val)
        
    init_metric = jnp.zeros((), dtype=state.V.dtype)
    final_carry = lax.fori_loop(0, n_steps, body_fn, (state, init_metric))
    return final_carry


# =============================================================================
# OS-Dyna Class
# =============================================================================

class JaxBatchedOSDyna:
    """
    JAX implementation of Batched OS-Dyna (Operator Splitting Dyna).
    
    Uses pure JAX functions for JIT compilation and GPU acceleration.
    """
    
    def __init__(self, mdp_P: Any, mdp_R: Any, gamma: float, num_envs: int,
                 lambda_val: Union[float, Any],
                 lr_params: Tuple[float, int] = (0.02, 30000),
                 lr_schedule: str = "constant_delay",
                 vi_tol: float = 1e-6,
                 vi_max_iters: int = PI_DEFAULT_ITERS,
                 seed: int = 0,
                 trajectory_mode: bool = False):
                 
        self.num_envs = num_envs
        self.gamma = gamma
        self.lambda_val = lambda_val
        self.vi_tol = vi_tol
        self.vi_max_iters = vi_max_iters
        
        # Parse LR params
        start_lr_val, delay_val = lr_params
        
        # Handle vectorization
        self.lr_start = as_column_parameter(start_lr_val, num_envs, dtype=jnp.float64)
        self.lr_delay = as_column_parameter(delay_val, num_envs, dtype=jnp.float64)
            
        self.lr_schedule_int = 0 if lr_schedule == "constant_delay" else 1
        
        # Lambda vectorization
        self.lambda_arr = as_column_parameter(lambda_val, num_envs, dtype=jnp.float64)
             
        # JAX MDP
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        
        # Gamma vectorization
        self.gamma_arr = as_column_parameter(gamma, num_envs, dtype=jnp.float64)

        # Init State
        self.mdp_data = MDPData(P=self.mdp_P, R=self.mdp_R, gamma=gamma)
        self.mdp_data, self.state = create_initial_state(
             self.mdp_P, self.mdp_R, num_envs, 0.0, seed=seed
        )
        
        # Params
        self.params = AlgorithmParams(
            gamma=self.gamma_arr,
            lambda_val=self.lambda_arr,
            h=1.0, # unused
            J_init=0, # unused
            surrogate_tol=vi_tol,
            surrogate_iters=vi_max_iters,
            max_J=0, # unused
            n_s=self.n_s,
            n_a=self.n_a,
            disc=self.gamma_arr.reshape(-1, 1, 1), # Used as disc for VI
            inv_1ph=0.0, # unused
            h_inv_1ph=0.0, # unused
            lr_start=self.lr_start,
            lr_delay=self.lr_delay,
            lr_schedule=self.lr_schedule_int
        )
        
    def step(self, k: int) -> float:
        self.state, mean_val = os_dyna_step_pure(
             self.state, self.mdp_data, self.params, k
        )
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        """Return the current checkpoint policy π_k induced by the OS surrogate solve."""
        return _compute_policy_os_dyna(self.state.r_k, self.state.P_hat, self.state.V, self.params.gamma)
        
    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_os_dyna_pure(
             self.state, self.mdp_data, self.params, start_k, n_steps
        )
        return float(mean_val)


# =============================================================================
# Dyna Step Function
# =============================================================================

@jax.jit
def dyna_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """Pure JAX step for Dyna (full surrogate solve or sample-based planning)."""
    
    # 1. Sample
    (s, a, r, ns, new_states, rng_key) = sample_step_pure(
        mdp.P,
        mdp.R,
        state.states,
        state.rng_key,
        params.n_s,
        params.n_a,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )
    
    # 2. Per-step LR for sample-based planning mode.
    lr = compute_lr_pure(k, params.lr_start, params.lr_delay, params.lr_schedule)
    lr = lr.reshape(-1)

    # 3. Model Update (Counts, R_sum, N_sa, P_hat)
    b_idx = jnp.arange(state.V.shape[0])
    N_sas = state.N_sas.at[b_idx, s, a, ns].add(1)
    N_sa = state.N_sa.at[b_idx, s, a].add(1)
    R_sum = state.R_sum.at[b_idx, s, a].add(r)
    
    # Incremental P_hat
    row_c = N_sas[b_idx, s, a]
    _, _, row_p = _compute_model_error_noisy_row_pure(
        row_counts=row_c,
        row_states=s,
        n_s=params.n_s,
        lambda_val=params.lambda_val,
        model_error_mode=params.model_error_mode,
    )
    P_hat = state.P_hat.at[b_idx, s, a].set(row_p)
    
    # 4. Planning
    def _plan_vi(_Q):
        # Full surrogate-solve mode.
        # Align with the original OSVI Dyna baseline: the real sample updates only
        # the learned model, and the new control iterate comes solely from solving
        # that updated model.
        N_sa_safe = jnp.where(N_sa == 0, jnp.ones_like(N_sa), N_sa)
        R_hat = R_sum / N_sa_safe
        
        # Initialize V from the current control estimate.
        V_init = jnp.max(_Q, axis=2)
        disc_vi = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1, 1, 1)
        
        # surrogate_vi_solve is the maintained fixed-step policy-iteration backend.
        # Here shifted_R = R_hat and disc = gamma.
        
        U_converged = surrogate_vi_solve(
            P_hat, R_hat, V_init, disc_vi, 
            tol=params.surrogate_tol, max_iters=params.surrogate_iters,
        )
        # Recompute Q from converged V
        term_fut = jnp.einsum('bzas,bs->bza', P_hat, U_converged)
        Q_new = R_hat + disc_vi * term_fut
        return Q_new

    def _plan_sample(_Q):
        # Sample-Based Mode
        # Loop planning_steps times
        # Note: planning_steps is vector (B,). params.max_J holds max steps.
        
        # To avoid passing full state in loop, we define body closing over consts
        def _sample_step(i, current_Q):
            # Mask
            active = (i < params.planning_steps) # (B,)
            
            # Sample (s, a) uniformly
            rng_k = jax.vmap(lambda key: random.fold_in(key, k + i * 1000))(
                ensure_batched_rng_keys(rng_key, num_envs=state.V.shape[0])
            )
            _, key_s, key_a, key_trans = split_batched_rng_keys(
                rng_k,
                num_envs=state.V.shape[0],
                num_splits=3,
            )
            
            # Concrete dimensions from outer scope state/mdp (safest)
            # state is available in closure from dyna_step_pure
            B_ = int(state.V.shape[0]) 
            n_s_conc = int(mdp.P.shape[0])
            n_a_conc = int(mdp.P.shape[1])
            
            s_sim = batched_random_int(key_s, low=0, high=n_s_conc, dtype=jnp.int32)
            a_sim = batched_random_int(key_a, low=0, high=n_a_conc, dtype=jnp.int32)
            
            b_idx_Loop = jnp.arange(B_)
            
            # Visited check
            n_sa_sim = N_sa[b_idx_Loop, s_sim, a_sim]
            visited = (n_sa_sim > 0)
            mask = active & visited
            
            # R_hat
            r_hat = R_sum[b_idx_Loop, s_sim, a_sim] / jnp.where(n_sa_sim==0, 1.0, n_sa_sim)
            
            # Next state
            P_probs = P_hat[b_idx_Loop, s_sim, a_sim]
            cdf = jnp.cumsum(P_probs, axis=1)
            r_rand = batched_random_uniform(key_trans, dtype=jnp.float64)[:, None]
            mask_s = (cdf > r_rand).astype(jnp.int32)
            ns_sim = jnp.argmax(mask_s, axis=1)
            
            # Update
            next_v = jnp.max(current_Q[b_idx_Loop, ns_sim], axis=1)
            target_sim = r_hat + gamma_vec * next_v
            diff_sim = target_sim - current_Q[b_idx_Loop, s_sim, a_sim]
            
            delta = lr * diff_sim * mask.astype(jnp.float64)
            
            new_Q = current_Q.at[b_idx_Loop, s_sim, a_sim].add(delta)
            return new_Q

        final_Q_samp = lax.fori_loop(0, params.max_J, _sample_step, _Q)
        return final_Q_samp

    # Dynamic dispatch using lax.cond to avoid TracerBoolConversionError.
    # In full-solve mode we warm-start from the prior model-control estimate rather
    # than from a fresh direct TD/Q update, matching the original OSVI Dyna code.
    # The classical direct Q update remains reachable for sample-based planning.
    Q_prior_f64 = state.Q.astype(jnp.float64)
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    next_vals = jnp.max(state.Q[b_idx, ns], axis=1).reshape(-1)
    target = r.reshape(-1) + gamma_vec * next_vals
    diff = target - state.Q[b_idx, s, a]
    Q_real_f64 = state.Q.at[b_idx, s, a].add(lr * diff).astype(jnp.float64)
    Q_final = lax.cond(
        params.use_vi_solve,
        lambda _: _plan_vi(Q_prior_f64),
        lambda _: _plan_sample(Q_real_f64),
        operand=None,
    )
        
    V_final = jnp.max(Q_final, axis=2)
    
    new_state = AgentState(
        V=V_final,
        P_hat=P_hat,
        R_hat=state.R_hat,
        N_sas=N_sas,
        N_sa=N_sa,
        R_sum=R_sum,
        states=new_states,
        J_per_env=state.J_per_env,
        prev_score=state.prev_score,
        rng_key=rng_key,
        r_k=state.r_k,
        Q=Q_final,
        bar_r=state.bar_r,
        h_s=state.h_s,
        R_E_factor=state.R_E_factor,
        V_checkpoint=state.V_checkpoint,
        steps_since_check=state.steps_since_check,
        working_V=state.working_V,
        policy_cache=state.policy_cache,
        waterfall_h=state.waterfall_h,
        waterfall_g_best=state.waterfall_g_best,
        waterfall_g_candidate=state.waterfall_g_candidate,
        waterfall_reward_abs_max=state.waterfall_reward_abs_max,
        waterfall_inner_depth=state.waterfall_inner_depth,
        waterfall_last_status=state.waterfall_last_status,
        waterfall_d_base=state.waterfall_d_base,
        waterfall_d_refine=state.waterfall_d_refine,
        defect_r=state.defect_r,
        defect_avg=state.defect_avg,
        defect_init=state.defect_init,
        defect_clip_fraction=state.defect_clip_fraction,
        defect_gamma_bound=state.defect_gamma_bound,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=advance_replay_sample_index(
            state.replay_sample_index,
            state.replay_states,
        ),
    )
    return new_state, jnp.mean(V_final[:, 0])


@jax.jit
def run_n_steps_dyna_pure(state, mdp, params, start_k, n_steps):
    """Run N steps of Dyna in a single JIT-compiled loop."""
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = dyna_step_pure(bst, mdp, params, curr_k)
        return (nst, val)
    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


# =============================================================================
# Dyna Class
# =============================================================================

class JaxBatchedDyna:
    """
    JAX implementation of Batched Dyna.
    
    Supports two planning modes:
    - Full surrogate solve (planning_steps=-1): runs the maintained fixed-step
      policy-iteration solve on the current model
    - Sample-based (planning_steps>0): uses k random samples from the model per step
    """
    
    def __init__(self, mdp_P, mdp_R, gamma, num_envs, lambda_val, planning_steps=-1,
                 lr_params=(0.5, 100), lr_schedule="constant_delay",
                 vi_tol=1e-6, vi_max_iters=PI_DEFAULT_ITERS, seed=0, trajectory_mode=False):
        """
        Initialize BatchedDyna.
        """
        self.num_envs = num_envs
        self.gamma = gamma
        self.lambda_val = lambda_val
        self.planning_steps = planning_steps
        
        # MDP
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        
        # Init State
        self.mdp_data = MDPData(P=self.mdp_P, R=self.mdp_R, gamma=gamma)
        self.mdp_data, self.state = create_initial_state(
            self.mdp_P, self.mdp_R, num_envs, 0.0, seed=seed
        )
        
        # Params
        # LR
        start_lr, delay = lr_params
        start_lr = as_column_parameter(start_lr, num_envs, dtype=jnp.float64)
        delay = as_column_parameter(delay, num_envs, dtype=jnp.float64)
        sched_int = 0 if lr_schedule == "constant_delay" else 1
        
        # Lambda
        lam_arr = as_column_parameter(lambda_val, num_envs, dtype=jnp.float64)
        
        # Gamma
        gam_arr = as_column_parameter(gamma, num_envs, dtype=jnp.float64)
        
        # Planning Steps
        if jnp.asarray(planning_steps).ndim == 0:
            plan_arr = jnp.full((num_envs,), planning_steps, dtype=jnp.int32)
            use_vi = (planning_steps == -1)
            max_j = planning_steps if not use_vi else 0
        else:
            plan_arr = as_vector_parameter(planning_steps, num_envs, dtype=jnp.int32)
            use_vi = bool(jnp.all(plan_arr == -1))
            max_j = int(jnp.max(plan_arr)) if not use_vi else 0
            
        self.params = AlgorithmParams(
            gamma=gam_arr,
            lambda_val=lam_arr,
            h=1.0, J_init=0, 
            surrogate_tol=vi_tol, surrogate_iters=vi_max_iters,
            max_J=max_j,
            n_s=self.n_s, n_a=self.n_a,
            disc=gam_arr.reshape(-1, 1, 1),
            inv_1ph=0.0, h_inv_1ph=0.0,
            lr_start=start_lr, lr_delay=delay, lr_schedule=sched_int,
            planning_steps=plan_arr,
            use_vi_solve=use_vi
        )
        
    def step(self, k: int) -> float:
        self.state, mean_val = dyna_step_pure(self.state, self.mdp_data, self.params, k)
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        """Return the current checkpoint policy π_k from the current model-control solve."""
        return _compute_policy_from_q(self.state.Q)
        
    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_dyna_pure(self.state, self.mdp_data, self.params, start_k, n_steps)
        return float(mean_val)


# =============================================================================
# ISP-Dyna Step Function
# =============================================================================

@jax.jit
def isp_dyna_update_with_sample_pure(
    state: AgentState,
    params: AlgorithmParams,
    s: jnp.ndarray,
    a: jnp.ndarray,
    r: jnp.ndarray,
    ns: jnp.ndarray,
    k: int,
) -> AgentState:
    """Apply one manuscript-faithful ISP-Dyna physical sample to batched state."""
    b_idx = jnp.arange(state.V.shape[0])
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    h_vec = jnp.asarray(params.h, dtype=jnp.float64).reshape(-1)
    inv_1ph = (1.0 / (1.0 + h_vec)).reshape(-1)
    h_inv_1ph = (h_vec / (1.0 + h_vec)).reshape(-1)
    disc = ((h_vec * gamma_vec) / (1.0 + h_vec)).reshape(-1, 1, 1)
    j_vec = jnp.asarray(params.J_init, dtype=jnp.int32).reshape(-1)
    batch_vec = jnp.asarray(params.batch_size, dtype=jnp.int32).reshape(-1)
    cycle_vec = jnp.maximum(j_vec * batch_vec, 1)
    sample_in_epoch = jnp.mod(k, cycle_vec)
    batch_offset = jnp.mod(sample_in_epoch, batch_vec)
    outer_start = sample_in_epoch == 0
    batch_start = batch_offset == 0
    batch_end = batch_offset == (batch_vec - 1)

    anchor_V = jnp.where(outer_start[:, None], state.V, state.V_checkpoint)
    flushed_bar_r = jnp.where(batch_start[:, None, None], jnp.zeros_like(state.bar_r), state.bar_r)

    prior_row = state.P_hat[b_idx, s, a]
    future_sample = state.V[b_idx, ns]
    surrogate_expectation = jnp.sum(prior_row * state.V, axis=1)
    target = (
        inv_1ph * anchor_V[b_idx, s]
        + h_inv_1ph * r.reshape(-1)
        + (h_inv_1ph * gamma_vec) * future_sample
        - (h_inv_1ph * gamma_vec) * surrogate_expectation
    )

    lr = compute_lr_pure(k, params.lr_start, params.lr_delay, params.lr_schedule).reshape(-1)
    tracker_delta = lr * (target - flushed_bar_r[b_idx, s, a])
    bar_r = flushed_bar_r.at[b_idx, s, a].add(tracker_delta)

    N_sas = state.N_sas.at[b_idx, s, a, ns].add(1.0)
    N_sa = state.N_sa.at[b_idx, s, a].add(1.0)
    R_sum = state.R_sum.at[b_idx, s, a].add(r.reshape(-1))

    row_counts = N_sas[b_idx, s, a]
    _, _, row_p = _compute_model_error_noisy_row_pure(
        row_counts=row_counts,
        row_states=s,
        n_s=params.n_s,
        lambda_val=params.lambda_val,
        model_error_mode=params.model_error_mode,
    )
    P_hat = state.P_hat.at[b_idx, s, a].set(row_p)

    planned_or_current_V = lax.cond(
        jnp.any(batch_end),
        lambda _: surrogate_vi_solve(
            P_hat,
            bar_r,
            state.V,
            disc,
            tol=params.surrogate_tol,
            max_iters=params.surrogate_iters,
        ),
        lambda _: state.V,
        operand=None,
    )
    next_V = jnp.where(batch_end[:, None], planned_or_current_V, state.V)

    return AgentState(
        V=next_V,
        P_hat=P_hat,
        R_hat=state.R_hat,
        N_sas=N_sas,
        N_sa=N_sa,
        R_sum=R_sum,
        states=state.states,
        J_per_env=state.J_per_env,
        prev_score=state.prev_score,
        rng_key=state.rng_key,
        r_k=state.r_k,
        Q=state.Q,
        bar_r=bar_r,
        h_s=state.h_s,
        R_E_factor=state.R_E_factor,
        V_checkpoint=anchor_V,
        steps_since_check=state.steps_since_check,
        working_V=state.working_V,
        policy_cache=state.policy_cache,
        waterfall_h=state.waterfall_h,
        waterfall_g_best=state.waterfall_g_best,
        waterfall_g_candidate=state.waterfall_g_candidate,
        waterfall_reward_abs_max=state.waterfall_reward_abs_max,
        waterfall_inner_depth=state.waterfall_inner_depth,
        waterfall_last_status=state.waterfall_last_status,
        waterfall_d_base=state.waterfall_d_base,
        waterfall_d_refine=state.waterfall_d_refine,
        defect_r=state.defect_r,
        defect_avg=state.defect_avg,
        defect_init=state.defect_init,
        defect_clip_fraction=state.defect_clip_fraction,
        defect_gamma_bound=state.defect_gamma_bound,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )


@jax.jit
def isp_dyna_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """Pure JAX step for manuscript ISP-Dyna under generative-model sampling."""
    s, a, r, ns, rng_key = sample_generative_transition_pure(
        mdp.P,
        mdp.R,
        state.rng_key,
        state.V.shape[0],
        params.n_s,
        params.n_a,
        params.sampling_distribution,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )
    updated = isp_dyna_update_with_sample_pure(state, params, s, a, r, ns, k)
    updated = updated._replace(
        states=s,
        rng_key=rng_key,
        replay_sample_index=advance_replay_sample_index(
            state.replay_sample_index,
            state.replay_states,
        ),
    )
    return updated, jnp.mean(updated.V[:, 0])


@jax.jit
def run_n_steps_isp_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    """Run N steps of ISP-Dyna in a single JIT-compiled loop."""
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = isp_dyna_step_pure(bst, mdp, params, curr_k)
        return (nst, val)

    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


class JaxBatchedISPDyna:
    """JAX implementation of manuscript ISP-Dyna with exact surrogate planning."""

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        h,
        J,
        batch_size,
        lambda_val=0.0,
        lr_params=(0.5, 0),
        lr_schedule="constant_delay",
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
        seed=0,
        sampling_distribution="uniform_sa",
        trajectory_mode=False,
    ):
        del trajectory_mode
        if str(sampling_distribution).lower() != "uniform_sa":
            raise ValueError("ISP-Dyna currently supports only sampling_distribution='uniform_sa'.")

        self.num_envs = num_envs
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        self.mdp_data = MDPData(P=self.mdp_P, R=self.mdp_R, gamma=gamma)
        self.mdp_data, self.state = create_initial_state(
            self.mdp_P,
            self.mdp_R,
            num_envs,
            0.0,
            J_init=int(jnp.asarray(J).reshape(-1)[0]) if jnp.asarray(J).ndim == 0 else 0,
            seed=seed,
        )

        start_lr, delay = lr_params
        lr_start = as_column_parameter(start_lr, num_envs, dtype=jnp.float64)
        lr_delay = as_column_parameter(delay, num_envs, dtype=jnp.float64)
        sched_int = 0 if str(lr_schedule).lower() in {"constant_delay", "constantanddelay"} else 1
        gam_arr = as_vector_parameter(gamma, num_envs, dtype=jnp.float64)
        lam_arr = as_vector_parameter(lambda_val, num_envs, dtype=jnp.float64)
        h_arr = as_vector_parameter(h, num_envs, dtype=jnp.float64)
        j_arr = as_vector_parameter(J, num_envs, dtype=jnp.int32)
        batch_arr = as_vector_parameter(batch_size, num_envs, dtype=jnp.int32)

        self.params = AlgorithmParams(
            gamma=gam_arr,
            lambda_val=lam_arr,
            h=h_arr,
            J_init=j_arr,
            surrogate_tol=vi_tol,
            surrogate_iters=vi_max_iters,
            max_J=int(jnp.max(j_arr)),
            n_s=self.n_s,
            n_a=self.n_a,
            disc=(h_arr * gam_arr) / (1.0 + h_arr),
            inv_1ph=1.0 / (1.0 + h_arr),
            h_inv_1ph=h_arr / (1.0 + h_arr),
            lr_start=lr_start,
            lr_delay=lr_delay,
            lr_schedule=sched_int,
            planning_steps=-1,
            use_vi_solve=True,
            batch_size=batch_arr,
            sampling_distribution=0,
        )

    def step(self, k: int) -> float:
        self.state, mean_val = isp_dyna_step_pure(self.state, self.mdp_data, self.params, k)
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        """Return the current checkpoint policy π_k induced by the ISP surrogate solve."""
        return _compute_policy_isp_dyna(
            self.state.bar_r,
            self.state.P_hat,
            self.state.V,
            self.params.gamma,
            self.params.h,
        )

    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_isp_dyna_pure(
            self.state,
            self.mdp_data,
            self.params,
            start_k,
            n_steps,
        )
        return float(mean_val)


@jax.jit
def _derive_alt_empirical_mdp_from_state(
    state: AgentState,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Construct the empirical SAA MDP with surrogate-prior fallback."""
    visits = jnp.asarray(state.N_sa, dtype=jnp.float64)
    empirical_P = jnp.where(
        visits[..., None] > 0.0,
        state.N_sas / jnp.maximum(visits[..., None], 1.0),
        state.P_hat,
    )
    empirical_R = jnp.where(
        visits > 0.0,
        state.R_sum / jnp.maximum(visits, 1.0),
        state.R_hat,
    )
    visited_fraction = jnp.mean((visits > 0.0).astype(jnp.float64), axis=(1, 2))
    return empirical_P, empirical_R, visited_fraction


@jax.jit
def _compute_alt_isp_dynamic_h_pure(
    P_emp: jnp.ndarray,
    P_prior: jnp.ndarray,
    gamma: Union[float, jnp.ndarray],
    h_target: Union[float, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the contraction-safe effective step size for Alt-ISP-Dyna."""
    gamma_vec = jnp.asarray(gamma, dtype=jnp.float64).reshape(-1)
    h_target_vec = jnp.asarray(h_target, dtype=jnp.float64).reshape(-1)
    empirical_error = gamma_vec * jnp.max(jnp.sum(jnp.abs(P_emp - P_prior), axis=3), axis=(1, 2))
    delta = 1.0 - gamma_vec
    denom = (2.0 * empirical_error) - delta
    safe_inverse = 1.0 / jnp.maximum(denom, 1e-12)
    effective_h = jnp.where(denom > 0.0, jnp.minimum(h_target_vec, safe_inverse), h_target_vec)
    return effective_h, empirical_error


@jax.jit
def _compute_policy_alt_isp_dyna(
    R_emp: jnp.ndarray,
    P_emp: jnp.ndarray,
    V: jnp.ndarray,
    gamma: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Greedy policy induced by the empirical Bellman operator."""
    gamma_vec = jnp.asarray(gamma, dtype=jnp.float64).reshape(-1, 1, 1)
    q_values = R_emp + gamma_vec * jnp.einsum("bzas,bs->bza", P_emp, V)
    return jnp.argmax(q_values, axis=2).astype(jnp.int32)


@jax.jit
def _compute_model_error_noisy_row_pure(
    row_counts: jnp.ndarray,
    row_states: jnp.ndarray,
    n_s: int,
    lambda_val: Union[float, jnp.ndarray],
    model_error_mode: Union[int, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the maintained stochastic noisy row from realized counts."""
    del n_s
    n_s_scalar = jnp.asarray(row_counts.shape[-1], dtype=jnp.float64)
    row_tot = jnp.sum(row_counts, axis=1, keepdims=True)
    safe_row_tot = jnp.where(row_tot > 0.0, row_tot, 1.0)
    row_emp = jnp.where(
        row_tot > 0.0,
        row_counts / safe_row_tot,
        jnp.ones_like(row_counts) / n_s_scalar,
    )
    row_pos = (row_counts > 0.0).astype(jnp.float64)
    row_pos_tot = jnp.sum(row_pos, axis=1, keepdims=True)
    safe_row_pos_tot = jnp.where(row_pos_tot > 0.0, row_pos_tot, 1.0)
    support_uniform = jnp.where(
        row_pos_tot > 0.0,
        row_pos / safe_row_pos_tot,
        jnp.ones_like(row_counts) / n_s_scalar,
    )
    self_loop = jax.nn.one_hot(
        jnp.asarray(row_states, dtype=jnp.int32).reshape(-1),
        row_counts.shape[-1],
        dtype=jnp.float64,
    )
    mode_arr = jnp.asarray(model_error_mode, dtype=jnp.int32).reshape(-1)
    if mode_arr.size == 1:
        mode_arr = jnp.broadcast_to(mode_arr, (row_counts.shape[0],))
    row_prior = jnp.where(mode_arr[:, None] == 1, self_loop, support_uniform)
    lam_vec = jnp.asarray(lambda_val, dtype=jnp.float64).reshape(-1, 1)
    row_p = row_emp * (1.0 - lam_vec) + row_prior * lam_vec
    return row_emp, row_prior, row_p


@jax.jit
def _compute_support_uniform_noisy_row_pure(
    row_counts: jnp.ndarray,
    n_s: int,
    lambda_val: Union[float, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the OS-Dyna support-uniform noisy row model from realized counts."""
    return _compute_model_error_noisy_row_pure(
        row_counts=row_counts,
        row_states=jnp.zeros((row_counts.shape[0],), dtype=jnp.int32),
        n_s=n_s,
        lambda_val=lambda_val,
        model_error_mode=0,
    )


@jax.jit
def _empirical_alt_isp_update_with_sample_pure(
    state: AgentState,
    params: AlgorithmParams,
    s: jnp.ndarray,
    a: jnp.ndarray,
    r: jnp.ndarray,
    ns: jnp.ndarray,
) -> AgentState:
    """Apply one realized physical sample to the empirical SAA ISP core."""
    b_idx = jnp.arange(state.V.shape[0])
    N_sas = state.N_sas.at[b_idx, s, a, ns].add(1.0)
    N_sa = state.N_sa.at[b_idx, s, a].add(1.0)
    R_sum = state.R_sum.at[b_idx, s, a].add(r.reshape(-1))
    _, _, row_p = _compute_model_error_noisy_row_pure(
        row_counts=N_sas[b_idx, s, a],
        row_states=s,
        n_s=params.n_s,
        lambda_val=params.lambda_val,
        model_error_mode=params.model_error_mode,
    )
    P_hat = state.P_hat.at[b_idx, s, a].set(row_p)
    counted_state = state._replace(N_sas=N_sas, N_sa=N_sa, R_sum=R_sum, P_hat=P_hat)

    P_emp, R_emp, visited_fraction = _derive_alt_empirical_mdp_from_state(counted_state)
    h_eff, empirical_error = _compute_alt_isp_dynamic_h_pure(
        P_emp,
        P_hat,
        gamma=params.gamma,
        h_target=params.h,
    )

    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    disc = ((h_eff * gamma_vec) / (1.0 + h_eff)).reshape(-1, 1, 1)
    inv_1ph = (1.0 / (1.0 + h_eff)).reshape(-1, 1, 1)
    h_inv_1ph = (h_eff / (1.0 + h_eff)).reshape(-1, 1, 1)
    j_vec = jnp.asarray(params.J_init, dtype=jnp.int32).reshape(-1)

    def body_fn(i: int, carry: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        U, shifted_last = carry
        correction = jnp.einsum("bzas,bs->bza", P_emp - P_hat, U)
        shifted = inv_1ph * state.V[:, :, None] + h_inv_1ph * R_emp + disc * correction
        candidate = surrogate_vi_solve(
            P_hat,
            shifted,
            U,
            disc,
            tol=params.surrogate_tol,
            max_iters=params.surrogate_iters,
        )
        active = i < j_vec
        next_U = jnp.where(active[:, None], candidate, U)
        next_shifted = jnp.where(active[:, None, None], shifted, shifted_last)
        return next_U, next_shifted

    initial_shifted = inv_1ph * state.V[:, :, None] + h_inv_1ph * R_emp
    next_V, shifted_R = lax.fori_loop(0, params.max_J, body_fn, (state.V, initial_shifted))
    greedy_policy = _compute_policy_alt_isp_dyna(R_emp, P_emp, next_V, gamma_vec)

    return AgentState(
        V=next_V,
        P_hat=P_hat,
        N_sas=N_sas,
        R_hat=state.R_hat,
        N_sa=N_sa,
        R_sum=R_sum,
        states=state.states,
        J_per_env=state.J_per_env,
        prev_score=state.prev_score,
        rng_key=state.rng_key,
        r_k=state.r_k,
        Q=state.Q,
        bar_r=shifted_R,
        h_s=state.h_s,
        R_E_factor=visited_fraction,
        V_checkpoint=next_V,
        steps_since_check=state.steps_since_check,
        working_V=next_V,
        policy_cache=greedy_policy,
        waterfall_h=h_eff,
        waterfall_g_best=visited_fraction,
        waterfall_g_candidate=empirical_error,
        waterfall_reward_abs_max=jnp.max(jnp.abs(shifted_R), axis=(1, 2)),
        waterfall_inner_depth=j_vec,
        waterfall_last_status=jnp.ones_like(j_vec),
        waterfall_d_base=state.waterfall_d_base,
        waterfall_d_refine=state.waterfall_d_refine,
        defect_r=state.defect_r,
        defect_avg=state.defect_avg,
        defect_init=state.defect_init,
        defect_clip_fraction=state.defect_clip_fraction,
        defect_gamma_bound=state.defect_gamma_bound,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )


@jax.jit
def alt_isp_dyna_update_with_sample_pure(
    state: AgentState,
    params: AlgorithmParams,
    s: jnp.ndarray,
    a: jnp.ndarray,
    r: jnp.ndarray,
    ns: jnp.ndarray,
) -> AgentState:
    """Apply one physical trajectory sample to Alt-ISP-Dyna."""
    return _empirical_alt_isp_update_with_sample_pure(state, params, s, a, r, ns)


@jax.jit
def gen_alt_isp_dyna_update_with_sample_pure(
    state: AgentState,
    params: AlgorithmParams,
    s: jnp.ndarray,
    a: jnp.ndarray,
    r: jnp.ndarray,
    ns: jnp.ndarray,
) -> AgentState:
    """Apply one physical generative-model sample to Gen-Alt-ISP-Dyna."""
    return _empirical_alt_isp_update_with_sample_pure(state, params, s, a, r, ns)


@jax.jit
def alt_isp_dyna_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """Pure JAX step for trajectory-only Alt-ISP-Dyna."""
    del k
    P_emp, R_emp, _ = _derive_alt_empirical_mdp_from_state(state)
    greedy_policy = _compute_policy_alt_isp_dyna(R_emp, P_emp, state.V, params.gamma)
    current_states = jnp.asarray(state.states, dtype=jnp.int32)
    greedy_actions = greedy_policy[jnp.arange(state.V.shape[0]), current_states]
    s, a, r, ns, new_states, rng_key = sample_trajectory_transition_pure(
        mdp.P,
        mdp.R,
        current_states,
        greedy_actions,
        params.epsilon,
        state.rng_key,
        params.n_a,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )
    updated = alt_isp_dyna_update_with_sample_pure(state, params, s, a, r, ns)
    updated = updated._replace(
        states=new_states,
        rng_key=rng_key,
        replay_sample_index=advance_replay_sample_index(
            state.replay_sample_index,
            state.replay_states,
        ),
    )
    return updated, jnp.mean(updated.V[:, 0])


@jax.jit
def gen_alt_isp_dyna_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """Pure JAX step for generative-only Gen-Alt-ISP-Dyna."""
    del k
    s, a, r, ns, rng_key = sample_generative_transition_pure(
        mdp.P,
        mdp.R,
        state.rng_key,
        state.V.shape[0],
        params.n_s,
        params.n_a,
        0,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )
    updated = gen_alt_isp_dyna_update_with_sample_pure(state, params, s, a, r, ns)
    updated = updated._replace(
        states=ns,
        rng_key=rng_key,
        replay_sample_index=advance_replay_sample_index(
            state.replay_sample_index,
            state.replay_states,
        ),
    )
    return updated, jnp.mean(updated.V[:, 0])


@jax.jit
def run_n_steps_alt_isp_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    """Run N physical trajectory steps of Alt-ISP-Dyna in one compiled loop."""
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = alt_isp_dyna_step_pure(bst, mdp, params, curr_k)
        return (nst, val)

    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


@jax.jit
def run_n_steps_gen_alt_isp_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    """Run N physical generative queries of Gen-Alt-ISP-Dyna in one compiled loop."""
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = gen_alt_isp_dyna_step_pure(bst, mdp, params, curr_k)
        return (nst, val)

    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


def _build_alt_isp_common_runtime(
    mdp_P,
    mdp_R,
    gamma,
    num_envs,
    h_target,
    J,
    *,
    lambda_val,
    epsilon,
    seed,
    vi_tol,
    vi_max_iters,
) -> tuple[MDPData, AgentState, AlgorithmParams]:
    mdp_P_jax = jnp.asarray(mdp_P, dtype=jnp.float64)
    mdp_R_jax = jnp.asarray(mdp_R, dtype=jnp.float64)
    n_s, n_a, _ = mdp_P_jax.shape
    mdp_data = MDPData(P=mdp_P_jax, R=mdp_R_jax, gamma=gamma)
    mdp_data, state = create_initial_state(
        mdp_P_jax,
        mdp_R_jax,
        num_envs,
        0.0,
        J_init=int(jnp.asarray(J).reshape(-1)[0]) if jnp.asarray(J).ndim == 0 else 0,
        seed=seed,
    )

    gam_arr = as_vector_parameter(gamma, num_envs, dtype=jnp.float64)
    lam_arr = as_vector_parameter(lambda_val, num_envs, dtype=jnp.float64)
    h_arr = as_vector_parameter(h_target, num_envs, dtype=jnp.float64)
    j_arr = as_vector_parameter(J, num_envs, dtype=jnp.int32)
    epsilon_arr = as_vector_parameter(epsilon, num_envs, dtype=jnp.float64)
    unit_batch_arr = jnp.ones((num_envs,), dtype=jnp.int32)

    state = state._replace(
        V_checkpoint=state.V,
        working_V=state.V,
        R_E_factor=jnp.zeros((num_envs,), dtype=jnp.float64),
        waterfall_h=h_arr,
        waterfall_g_best=jnp.zeros((num_envs,), dtype=jnp.float64),
        waterfall_g_candidate=jnp.zeros((num_envs,), dtype=jnp.float64),
        waterfall_reward_abs_max=jnp.zeros((num_envs,), dtype=jnp.float64),
        waterfall_inner_depth=j_arr,
        waterfall_last_status=jnp.zeros((num_envs,), dtype=jnp.int32),
    )
    params = AlgorithmParams(
        gamma=gam_arr,
        lambda_val=lam_arr,
        h=h_arr,
        J_init=j_arr,
        surrogate_tol=vi_tol,
        surrogate_iters=vi_max_iters,
        max_J=int(jnp.max(j_arr)),
        n_s=n_s,
        n_a=n_a,
        disc=(h_arr * gam_arr) / (1.0 + h_arr),
        inv_1ph=1.0 / (1.0 + h_arr),
        h_inv_1ph=h_arr / (1.0 + h_arr),
        lr_start=jnp.zeros((num_envs,), dtype=jnp.float64),
        lr_delay=jnp.zeros((num_envs,), dtype=jnp.float64),
        lr_schedule=0,
        planning_steps=-1,
        use_vi_solve=True,
        batch_size=unit_batch_arr,
        epsilon=epsilon_arr,
    )
    return mdp_data, state, params


class JaxBatchedAltISPDyna:
    """JAX implementation of the trajectory-only Alt-ISP-Dyna core."""

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        h_target,
        J,
        epsilon=0.1,
        seed=0,
        trajectory_mode=True,
        sampling_distribution=None,
        lambda_val=0.0,
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
    ):
        del trajectory_mode, sampling_distribution
        self.num_envs = num_envs
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        self.mdp_data, self.state, self.params = _build_alt_isp_common_runtime(
            self.mdp_P,
            self.mdp_R,
            gamma,
            num_envs,
            h_target,
            J,
            lambda_val=lambda_val,
            epsilon=epsilon,
            seed=seed,
            vi_tol=vi_tol,
            vi_max_iters=vi_max_iters,
        )

    def step(self, k: int) -> float:
        self.state, mean_val = alt_isp_dyna_step_pure(self.state, self.mdp_data, self.params, k)
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        P_emp, R_emp, _ = _derive_alt_empirical_mdp_from_state(self.state)
        return _compute_policy_alt_isp_dyna(R_emp, P_emp, self.state.V, self.params.gamma)

    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_alt_isp_dyna_pure(
            self.state,
            self.mdp_data,
            self.params,
            start_k,
            n_steps,
        )
        return float(mean_val)


class JaxBatchedGenAltISPDyna:
    """JAX implementation of the generative-only Gen-Alt-ISP-Dyna core."""

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        h_target,
        J,
        seed=0,
        sampling_distribution=None,
        epsilon=None,
        lambda_val=0.0,
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
    ):
        del sampling_distribution, epsilon
        self.num_envs = num_envs
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        self.mdp_data, self.state, self.params = _build_alt_isp_common_runtime(
            self.mdp_P,
            self.mdp_R,
            gamma,
            num_envs,
            h_target,
            J,
            lambda_val=lambda_val,
            epsilon=0.0,
            seed=seed,
            vi_tol=vi_tol,
            vi_max_iters=vi_max_iters,
        )

    def step(self, k: int) -> float:
        self.state, mean_val = gen_alt_isp_dyna_step_pure(self.state, self.mdp_data, self.params, k)
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        P_emp, R_emp, _ = _derive_alt_empirical_mdp_from_state(self.state)
        return _compute_policy_alt_isp_dyna(R_emp, P_emp, self.state.V, self.params.gamma)

    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_gen_alt_isp_dyna_pure(
            self.state,
            self.mdp_data,
            self.params,
            start_k,
            n_steps,
        )
        return float(mean_val)


@jax.jit
def _rm_isp_shifted_reward_pure(
    anchor_V: jnp.ndarray,
    defect_r: jnp.ndarray,
    h: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Affine reward map induced by the rm-ISP defect tracker."""
    h_vec = jnp.asarray(h, dtype=jnp.float64).reshape(-1)
    inv_1ph = (1.0 / (1.0 + h_vec)).reshape(-1, 1, 1)
    h_inv_1ph = (h_vec / (1.0 + h_vec)).reshape(-1, 1, 1)
    return inv_1ph * anchor_V[:, :, None] + h_inv_1ph * defect_r


@jax.jit
def _compute_policy_rm_isp_dyna(
    defect_r: jnp.ndarray,
    P_hat: jnp.ndarray,
    V: jnp.ndarray,
    gamma: Union[float, jnp.ndarray],
    h: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Argmax policy for maintained RM-ISP-Dyna using defect-space extraction."""
    gamma_vec = jnp.asarray(gamma, dtype=jnp.float64).reshape(-1)
    h_vec = jnp.asarray(h, dtype=jnp.float64).reshape(-1)
    disc = ((h_vec * gamma_vec) / (1.0 + h_vec)).reshape(-1, 1, 1)
    shifted_R = _rm_isp_shifted_reward_pure(V, defect_r, h_vec)
    term_fut = jnp.einsum("bzas,bs->bza", P_hat, V)
    q_values = shifted_R + disc * term_fut
    return jnp.argmax(q_values, axis=2)


@jax.jit
def _compute_policy_nr_rm_isp_dyna(
    defect_r: jnp.ndarray,
    P_hat: jnp.ndarray,
    V: jnp.ndarray,
    gamma: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Argmax policy for the exact no-reset reward-modified limit."""
    gamma_vec = jnp.asarray(gamma, dtype=jnp.float64).reshape(-1, 1, 1)
    term_fut = jnp.einsum("bzas,bs->bza", P_hat, V)
    q_values = defect_r + gamma_vec * term_fut
    return jnp.argmax(q_values, axis=2)


@jax.jit
def _compute_policy_nr_rm_isp_dispatch(
    defect_r: jnp.ndarray,
    P_hat: jnp.ndarray,
    V: jnp.ndarray,
    gamma: Union[float, jnp.ndarray],
    h: Union[float, jnp.ndarray],
    nr_mode: Union[int, jnp.ndarray],
) -> jnp.ndarray:
    mode_code = jnp.asarray(nr_mode, dtype=jnp.int32)
    generalized = jnp.any(mode_code != 0)
    return lax.cond(
        generalized,
        lambda _: _compute_policy_rm_isp_dyna(defect_r, P_hat, V, gamma, h),
        lambda _: _compute_policy_nr_rm_isp_dyna(defect_r, P_hat, V, gamma),
        operand=None,
    )


@jax.jit
def _compute_policy_arch_isp_dyna(
    r_k: jnp.ndarray,
    P_hat: jnp.ndarray,
    V: jnp.ndarray,
    gamma: Union[float, jnp.ndarray],
    h: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Argmax policy for the archived ISP-Dyna surrogate extraction rule."""
    gamma_vec = jnp.asarray(gamma, dtype=jnp.float64).reshape(-1)
    h_vec = jnp.asarray(h, dtype=jnp.float64).reshape(-1)
    inv_1ph = (1.0 / (1.0 + h_vec)).reshape(-1, 1, 1)
    h_inv_1ph = (h_vec / (1.0 + h_vec)).reshape(-1, 1, 1)
    disc = ((h_vec * gamma_vec) / (1.0 + h_vec)).reshape(-1, 1, 1)
    shifted_R = inv_1ph * V[:, :, None] + h_inv_1ph * r_k
    term_fut = jnp.einsum("bzas,bs->bza", P_hat, V)
    q_values = shifted_R + disc * term_fut
    return jnp.argmax(q_values, axis=2)


@jax.jit
def arch_isp_dyna_update_with_sample_pure(
    state: AgentState,
    params: AlgorithmParams,
    s: jnp.ndarray,
    a: jnp.ndarray,
    r: jnp.ndarray,
    ns: jnp.ndarray,
    k: int,
) -> AgentState:
    """Apply one archived ISP-Dyna physical sample to batched state."""
    b_idx = jnp.arange(state.V.shape[0])
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    h_vec = jnp.asarray(params.h, dtype=jnp.float64).reshape(-1)
    inv_1ph = (1.0 / (1.0 + h_vec)).reshape(-1)
    h_inv_1ph = (h_vec / (1.0 + h_vec)).reshape(-1)
    disc = ((h_vec * gamma_vec) / (1.0 + h_vec)).reshape(-1, 1, 1)
    j_vec = jnp.asarray(params.J_init, dtype=jnp.int32).reshape(-1)

    if state.working_V is None:
        current_inner = state.V
    else:
        current_inner = jnp.asarray(state.working_V, dtype=jnp.float64)
    if state.waterfall_inner_depth is None:
        current_depth = jnp.zeros((state.V.shape[0],), dtype=jnp.int32)
    else:
        current_depth = jnp.asarray(state.waterfall_inner_depth, dtype=jnp.int32).reshape(-1)

    N_sas = state.N_sas.at[b_idx, s, a, ns].add(1.0)
    N_sa = state.N_sa.at[b_idx, s, a].add(1.0)
    R_sum = state.R_sum.at[b_idx, s, a].add(r.reshape(-1))

    row_counts = N_sas[b_idx, s, a]
    _, _, row_p = _compute_model_error_noisy_row_pure(
        row_counts=row_counts,
        row_states=s,
        n_s=params.n_s,
        lambda_val=params.lambda_val,
        model_error_mode=params.model_error_mode,
    )
    P_hat = state.P_hat.at[b_idx, s, a].set(row_p)

    row_visits = N_sa[b_idx, s, a]
    row_reward_mean = jnp.where(
        row_visits > 0.0,
        R_sum[b_idx, s, a] / row_visits,
        jnp.zeros_like(row_visits),
    )
    R_hat = state.R_hat.at[b_idx, s, a].set(row_reward_mean)

    future_sample = current_inner[b_idx, ns]
    surrogate_expectation = jnp.sum(row_p * current_inner, axis=1)
    target = row_reward_mean + gamma_vec * future_sample - gamma_vec * surrogate_expectation

    lr = compute_lr_pure(k, params.lr_start, params.lr_delay, params.lr_schedule).reshape(-1)
    tracker_delta = lr * (target - state.r_k[b_idx, s, a])
    r_k = state.r_k.at[b_idx, s, a].add(tracker_delta)

    shifted_R = inv_1ph[:, None, None] * state.V[:, :, None] + h_inv_1ph[:, None, None] * r_k
    candidate_inner = surrogate_vi_solve(
        P_hat,
        shifted_R,
        current_inner,
        disc,
        tol=params.surrogate_tol,
        max_iters=params.surrogate_iters,
    )

    next_depth = current_depth + 1
    commit_mask = next_depth >= j_vec
    committed_V = jnp.where(commit_mask[:, None], candidate_inner, state.V)
    next_inner = jnp.where(commit_mask[:, None], committed_V, candidate_inner)
    reset_depth = jnp.where(commit_mask, 0, next_depth)

    return AgentState(
        V=committed_V,
        P_hat=P_hat,
        R_hat=R_hat,
        N_sas=N_sas,
        N_sa=N_sa,
        R_sum=R_sum,
        states=state.states,
        J_per_env=state.J_per_env,
        prev_score=state.prev_score,
        rng_key=state.rng_key,
        r_k=r_k,
        Q=state.Q,
        bar_r=shifted_R,
        h_s=state.h_s,
        R_E_factor=state.R_E_factor,
        V_checkpoint=committed_V,
        steps_since_check=reset_depth,
        working_V=next_inner,
        policy_cache=state.policy_cache,
        waterfall_h=h_vec,
        waterfall_g_best=state.waterfall_g_best,
        waterfall_g_candidate=state.waterfall_g_candidate,
        waterfall_reward_abs_max=state.waterfall_reward_abs_max,
        waterfall_inner_depth=reset_depth,
        waterfall_last_status=state.waterfall_last_status,
        waterfall_d_base=state.waterfall_d_base,
        waterfall_d_refine=state.waterfall_d_refine,
        defect_r=state.defect_r,
        defect_avg=state.defect_avg,
        defect_init=state.defect_init,
        defect_clip_fraction=state.defect_clip_fraction,
        defect_gamma_bound=state.defect_gamma_bound,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )


@jax.jit
def arch_isp_dyna_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """Pure JAX step for archived ISP-Dyna under generative-model sampling."""
    s, a, r, ns, rng_key = sample_generative_transition_pure(
        mdp.P,
        mdp.R,
        state.rng_key,
        state.V.shape[0],
        params.n_s,
        params.n_a,
        params.sampling_distribution,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )
    updated = arch_isp_dyna_update_with_sample_pure(state, params, s, a, r, ns, k)
    updated = updated._replace(
        states=s,
        rng_key=rng_key,
        replay_sample_index=advance_replay_sample_index(
            state.replay_sample_index,
            state.replay_states,
        ),
    )
    return updated, jnp.mean(updated.V[:, 0])


@jax.jit
def run_n_steps_arch_isp_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    """Run N steps of archived ISP-Dyna in a single JIT-compiled loop."""
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = arch_isp_dyna_step_pure(bst, mdp, params, curr_k)
        return (nst, val)

    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


class JaxBatchedArchISPDyna:
    """JAX implementation of the archived ISP-Dyna core as a maintained selector."""

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        h,
        J,
        lambda_val=0.0,
        lr_params=(0.5, 0),
        lr_schedule="constant_delay",
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
        seed=0,
        sampling_distribution="uniform_sa",
        trajectory_mode=False,
    ):
        del trajectory_mode
        if str(sampling_distribution).lower() != "uniform_sa":
            raise ValueError("Arch-ISP-Dyna currently supports only sampling_distribution='uniform_sa'.")

        self.num_envs = num_envs
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        self.mdp_data = MDPData(P=self.mdp_P, R=self.mdp_R, gamma=gamma)
        self.mdp_data, self.state = create_initial_state(
            self.mdp_P,
            self.mdp_R,
            num_envs,
            0.0,
            J_init=int(jnp.asarray(J).reshape(-1)[0]) if jnp.asarray(J).ndim == 0 else 0,
            seed=seed,
        )

        start_lr, delay = lr_params
        lr_start = as_column_parameter(start_lr, num_envs, dtype=jnp.float64)
        lr_delay = as_column_parameter(delay, num_envs, dtype=jnp.float64)
        sched_int = 0 if str(lr_schedule).lower() in {"constant_delay", "constantanddelay"} else 1
        gam_arr = as_vector_parameter(gamma, num_envs, dtype=jnp.float64)
        lam_arr = as_vector_parameter(lambda_val, num_envs, dtype=jnp.float64)
        h_arr = as_vector_parameter(h, num_envs, dtype=jnp.float64)
        j_arr = as_vector_parameter(J, num_envs, dtype=jnp.int32)
        unit_batch_arr = jnp.ones((num_envs,), dtype=jnp.int32)

        self.state = self.state._replace(
            V_checkpoint=self.state.V,
            working_V=self.state.V,
            waterfall_h=h_arr,
            waterfall_inner_depth=jnp.zeros((num_envs,), dtype=jnp.int32),
        )
        self.params = AlgorithmParams(
            gamma=gam_arr,
            lambda_val=lam_arr,
            h=h_arr,
            J_init=j_arr,
            surrogate_tol=vi_tol,
            surrogate_iters=vi_max_iters,
            max_J=int(jnp.max(j_arr)),
            n_s=self.n_s,
            n_a=self.n_a,
            disc=(h_arr * gam_arr) / (1.0 + h_arr),
            inv_1ph=1.0 / (1.0 + h_arr),
            h_inv_1ph=h_arr / (1.0 + h_arr),
            lr_start=lr_start,
            lr_delay=lr_delay,
            lr_schedule=sched_int,
            planning_steps=-1,
            use_vi_solve=True,
            batch_size=unit_batch_arr,
            sampling_distribution=0,
        )

    def step(self, k: int) -> float:
        self.state, mean_val = arch_isp_dyna_step_pure(self.state, self.mdp_data, self.params, k)
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        return _compute_policy_arch_isp_dyna(
            self.state.r_k,
            self.state.P_hat,
            self.state.V,
            self.params.gamma,
            self.params.h,
        )

    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_arch_isp_dyna_pure(
            self.state,
            self.mdp_data,
            self.params,
            start_k,
            n_steps,
        )
        return float(mean_val)


def _nr_rm_isp_h_for_step(
    params: AlgorithmParams,
    k: int,
    batch_size: int,
) -> jnp.ndarray:
    nr_mode = jnp.asarray(params.nr_mode, dtype=jnp.int32)
    nr_mode_vec = (
        jnp.full((batch_size,), nr_mode, dtype=jnp.int32)
        if nr_mode.ndim == 0
        else nr_mode.reshape(-1)
    )
    base_h = jnp.asarray(params.h_init, dtype=jnp.float64)
    base_h_vec = (
        jnp.full((batch_size,), base_h, dtype=jnp.float64)
        if base_h.ndim == 0
        else base_h.reshape(-1)
    )
    h_schedule_shape = jnp.asarray(getattr(params, "h_schedule_shape", 0), dtype=jnp.int32)
    h_schedule_shape_vec = (
        jnp.full((batch_size,), h_schedule_shape, dtype=jnp.int32)
        if h_schedule_shape.ndim == 0
        else h_schedule_shape.reshape(-1)
    )
    h_schedule_finish_frac = jnp.asarray(
        getattr(params, "h_schedule_finish_frac", 1.0),
        dtype=jnp.float64,
    )
    h_schedule_finish_frac_vec = (
        jnp.full((batch_size,), h_schedule_finish_frac, dtype=jnp.float64)
        if h_schedule_finish_frac.ndim == 0
        else h_schedule_finish_frac.reshape(-1)
    )
    horizon = jnp.asarray(getattr(params, "h_schedule_horizon", 0), dtype=jnp.float64)
    terminal_h = jnp.asarray(params.h_terminal, dtype=jnp.float64)
    terminal_h_vec = (
        jnp.full((batch_size,), terminal_h, dtype=jnp.float64)
        if terminal_h.ndim == 0
        else terminal_h.reshape(-1)
    )
    h_floor = jnp.asarray(params.base_h, dtype=jnp.float64)
    h_floor_vec = (
        jnp.full((batch_size,), h_floor, dtype=jnp.float64)
        if h_floor.ndim == 0
        else h_floor.reshape(-1)
    )
    numerator = jnp.maximum(base_h_vec - h_floor_vec, 1e-12)
    denominator = jnp.maximum(terminal_h_vec - h_floor_vec, 1e-12)
    safe_horizon = jnp.maximum(1.0, horizon)
    progress = jnp.clip(jnp.asarray(k, dtype=jnp.float64) / safe_horizon, 0.0, 1.0)
    decayed = h_floor_vec + numerator * jnp.power(denominator / numerator, progress)
    exponential_h = jnp.maximum(h_floor_vec, decayed)
    linear_horizon = jnp.maximum(1.0, safe_horizon * jnp.maximum(h_schedule_finish_frac_vec, 1e-12))
    linear_progress = jnp.clip(jnp.asarray(k, dtype=jnp.float64) / linear_horizon, 0.0, 1.0)
    linear_h = jnp.maximum(
        h_floor_vec,
        base_h_vec + (terminal_h_vec - base_h_vec) * linear_progress,
    )
    scheduled_h = jnp.where(h_schedule_shape_vec == 1, linear_h, exponential_h)
    return jnp.where(
        horizon > 0.0,
        jnp.where(nr_mode_vec == 2, scheduled_h, base_h_vec),
        base_h_vec,
    )


def _nr_rm_isp_normalize_state(
    state: AgentState,
) -> AgentState:
    batch_size, n_states = state.V.shape
    _, _, n_actions = state.R_hat.shape
    return state._replace(
        working_V=state.working_V if state.working_V is not None else state.V,
        policy_cache=(
            state.policy_cache
            if state.policy_cache is not None
            else jnp.zeros((batch_size, n_states), dtype=jnp.int32)
        ),
        waterfall_h=(
            state.waterfall_h
            if state.waterfall_h is not None
            else jnp.ones((batch_size,), dtype=jnp.float64)
        ),
        waterfall_g_best=(
            state.waterfall_g_best
            if state.waterfall_g_best is not None
            else jnp.zeros((batch_size,), dtype=jnp.float64)
        ),
        waterfall_g_candidate=(
            state.waterfall_g_candidate
            if state.waterfall_g_candidate is not None
            else jnp.zeros((batch_size,), dtype=jnp.float64)
        ),
        waterfall_reward_abs_max=(
            state.waterfall_reward_abs_max
            if state.waterfall_reward_abs_max is not None
            else jnp.zeros((batch_size,), dtype=jnp.float64)
        ),
        waterfall_inner_depth=(
            state.waterfall_inner_depth
            if state.waterfall_inner_depth is not None
            else jnp.zeros((batch_size,), dtype=jnp.int32)
        ),
        waterfall_last_status=(
            state.waterfall_last_status
            if state.waterfall_last_status is not None
            else jnp.zeros((batch_size,), dtype=jnp.int32)
        ),
        waterfall_d_base=(
            state.waterfall_d_base
            if state.waterfall_d_base is not None
            else jnp.zeros((batch_size,), dtype=jnp.float64)
        ),
        waterfall_d_refine=(
            state.waterfall_d_refine
            if state.waterfall_d_refine is not None
            else jnp.zeros((batch_size,), dtype=jnp.float64)
        ),
        defect_avg=(
            state.defect_avg
            if state.defect_avg is not None
            else jnp.zeros((batch_size, n_states, n_actions), dtype=jnp.float64)
        ),
        defect_init=(
            state.defect_init
            if state.defect_init is not None
            else jnp.zeros((batch_size, n_states, n_actions), dtype=jnp.float64)
        ),
        defect_clip_fraction=(
            state.defect_clip_fraction
            if state.defect_clip_fraction is not None
            else jnp.zeros((batch_size,), dtype=jnp.float64)
        ),
        defect_gamma_bound=(
            state.defect_gamma_bound
            if state.defect_gamma_bound is not None
            else jnp.zeros((batch_size,), dtype=jnp.float64)
        ),
    )


def _nr_rm_isp_limit_update_with_sample_pure(
    state: AgentState,
    params: AlgorithmParams,
    s: jnp.ndarray,
    a: jnp.ndarray,
    r: jnp.ndarray,
    ns: jnp.ndarray,
    k: int,
) -> AgentState:
    b_idx = jnp.arange(state.V.shape[0])
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    current_V = jnp.asarray(state.V, dtype=jnp.float64)
    current_defect = (
        jnp.asarray(state.defect_r, dtype=jnp.float64)
        if state.defect_r is not None
        else jnp.asarray(state.r_k, dtype=jnp.float64)
    )
    defect_init = (
        jnp.asarray(state.defect_init, dtype=jnp.float64)
        if state.defect_init is not None
        else jnp.zeros_like(current_defect)
    )

    N_sas = state.N_sas.at[b_idx, s, a, ns].add(1.0)
    N_sa = state.N_sa.at[b_idx, s, a].add(1.0)
    R_sum = state.R_sum.at[b_idx, s, a].add(r.reshape(-1))

    _, _, row_p = _compute_model_error_noisy_row_pure(
        row_counts=N_sas[b_idx, s, a],
        row_states=s,
        n_s=params.n_s,
        lambda_val=params.lambda_val,
        model_error_mode=params.model_error_mode,
    )
    P_hat = state.P_hat.at[b_idx, s, a].set(row_p)

    future_sample = current_V[b_idx, ns]
    surrogate_expectation = jnp.sum(row_p * current_V, axis=1)
    target = r.reshape(-1) + gamma_vec * future_sample - gamma_vec * surrogate_expectation

    lr = compute_lr_pure(k, params.lr_start, params.lr_delay, params.lr_schedule).reshape(-1)
    updated_defect = current_defect.at[b_idx, s, a].add(
        lr * (target - current_defect[b_idx, s, a])
    )

    committed_V = surrogate_vi_solve(
        P_hat,
        updated_defect,
        current_V,
        gamma_vec.reshape(-1, 1, 1),
        tol=params.surrogate_tol,
        max_iters=params.surrogate_iters,
    )

    return AgentState(
        V=committed_V,
        P_hat=P_hat,
        R_hat=state.R_hat,
        N_sas=N_sas,
        N_sa=N_sa,
        R_sum=R_sum,
        states=state.states,
        J_per_env=state.J_per_env,
        prev_score=state.prev_score,
        rng_key=state.rng_key,
        r_k=updated_defect,
        Q=state.Q,
        bar_r=updated_defect,
        h_s=state.h_s,
        R_E_factor=state.R_E_factor,
        V_checkpoint=committed_V,
        steps_since_check=jnp.zeros_like(state.steps_since_check),
        working_V=committed_V,
        policy_cache=state.policy_cache,
        waterfall_h=state.waterfall_h,
        waterfall_g_best=state.waterfall_g_best,
        waterfall_g_candidate=state.waterfall_g_candidate,
        waterfall_reward_abs_max=jnp.max(jnp.abs(updated_defect), axis=(1, 2)),
        waterfall_inner_depth=jnp.zeros((state.V.shape[0],), dtype=jnp.int32),
        waterfall_last_status=state.waterfall_last_status,
        waterfall_d_base=state.waterfall_d_base,
        waterfall_d_refine=state.waterfall_d_refine,
        defect_r=updated_defect,
        defect_avg=state.defect_avg,
        defect_init=defect_init,
        defect_clip_fraction=state.defect_clip_fraction,
        defect_gamma_bound=state.defect_gamma_bound,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )


def _nr_rm_isp_generalized_update_with_sample_pure(
    state: AgentState,
    params: AlgorithmParams,
    s: jnp.ndarray,
    a: jnp.ndarray,
    r: jnp.ndarray,
    ns: jnp.ndarray,
    k: int,
) -> AgentState:
    b_idx = jnp.arange(state.V.shape[0])
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    current_h = _nr_rm_isp_h_for_step(params, k, state.V.shape[0])
    disc = ((current_h * gamma_vec) / (1.0 + current_h)).reshape(-1, 1, 1)
    j_vec = jnp.asarray(params.J_init, dtype=jnp.int32).reshape(-1)

    current_inner = (
        jnp.asarray(state.working_V, dtype=jnp.float64)
        if state.working_V is not None
        else jnp.asarray(state.V, dtype=jnp.float64)
    )
    current_depth = (
        jnp.asarray(state.waterfall_inner_depth, dtype=jnp.int32).reshape(-1)
        if state.waterfall_inner_depth is not None
        else jnp.zeros((state.V.shape[0],), dtype=jnp.int32)
    )
    anchor_V = jnp.asarray(state.V_checkpoint, dtype=jnp.float64)
    current_defect = (
        jnp.asarray(state.defect_r, dtype=jnp.float64)
        if state.defect_r is not None
        else jnp.asarray(state.r_k, dtype=jnp.float64)
    )
    defect_init = (
        jnp.asarray(state.defect_init, dtype=jnp.float64)
        if state.defect_init is not None
        else jnp.zeros_like(current_defect)
    )

    N_sas = state.N_sas.at[b_idx, s, a, ns].add(1.0)
    N_sa = state.N_sa.at[b_idx, s, a].add(1.0)
    R_sum = state.R_sum.at[b_idx, s, a].add(r.reshape(-1))

    _, _, row_p = _compute_model_error_noisy_row_pure(
        row_counts=N_sas[b_idx, s, a],
        row_states=s,
        n_s=params.n_s,
        lambda_val=params.lambda_val,
        model_error_mode=params.model_error_mode,
    )
    P_hat = state.P_hat.at[b_idx, s, a].set(row_p)

    future_sample = current_inner[b_idx, ns]
    surrogate_expectation = jnp.sum(row_p * current_inner, axis=1)
    target = r.reshape(-1) + gamma_vec * future_sample - gamma_vec * surrogate_expectation

    lr = compute_lr_pure(k, params.lr_start, params.lr_delay, params.lr_schedule).reshape(-1)
    updated_defect = current_defect.at[b_idx, s, a].add(
        lr * (target - current_defect[b_idx, s, a])
    )
    shifted_reward = _rm_isp_shifted_reward_pure(anchor_V, updated_defect, current_h)
    candidate_inner = surrogate_vi_solve(
        P_hat,
        shifted_reward,
        current_inner,
        disc,
        tol=params.surrogate_tol,
        max_iters=params.surrogate_iters,
    )

    next_depth = current_depth + 1
    commit_mask = next_depth >= j_vec
    committed_V = jnp.where(commit_mask[:, None], candidate_inner, state.V)
    committed_shifted_reward = _rm_isp_shifted_reward_pure(committed_V, updated_defect, current_h)
    next_inner = jnp.where(commit_mask[:, None], committed_V, candidate_inner)
    reset_depth = jnp.where(commit_mask, 0, next_depth)
    next_shifted = jnp.where(commit_mask[:, None, None], committed_shifted_reward, shifted_reward)

    return AgentState(
        V=committed_V,
        P_hat=P_hat,
        R_hat=state.R_hat,
        N_sas=N_sas,
        N_sa=N_sa,
        R_sum=R_sum,
        states=state.states,
        J_per_env=state.J_per_env,
        prev_score=state.prev_score,
        rng_key=state.rng_key,
        r_k=updated_defect,
        Q=state.Q,
        bar_r=next_shifted,
        h_s=state.h_s,
        R_E_factor=state.R_E_factor,
        V_checkpoint=committed_V,
        steps_since_check=reset_depth,
        working_V=next_inner,
        policy_cache=state.policy_cache,
        waterfall_h=current_h,
        waterfall_g_best=state.waterfall_g_best,
        waterfall_g_candidate=state.waterfall_g_candidate,
        waterfall_reward_abs_max=jnp.max(jnp.abs(updated_defect), axis=(1, 2)),
        waterfall_inner_depth=reset_depth,
        waterfall_last_status=state.waterfall_last_status,
        waterfall_d_base=state.waterfall_d_base,
        waterfall_d_refine=state.waterfall_d_refine,
        defect_r=updated_defect,
        defect_avg=state.defect_avg,
        defect_init=defect_init,
        defect_clip_fraction=state.defect_clip_fraction,
        defect_gamma_bound=state.defect_gamma_bound,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )


@jax.jit
def _nr1_rm_isp_limit_update_with_sample_pure(
    state: AgentState,
    params: AlgorithmParams,
    s: jnp.ndarray,
    a: jnp.ndarray,
    r: jnp.ndarray,
    ns: jnp.ndarray,
    k: int,
) -> AgentState:
    b_idx = jnp.arange(state.V.shape[0])
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    current_V = jnp.asarray(state.V, dtype=jnp.float64)
    current_defect = (
        jnp.asarray(state.defect_r, dtype=jnp.float64)
        if state.defect_r is not None
        else jnp.asarray(state.r_k, dtype=jnp.float64)
    )
    defect_init = (
        jnp.asarray(state.defect_init, dtype=jnp.float64)
        if state.defect_init is not None
        else jnp.zeros_like(current_defect)
    )
    N_sas = state.N_sas.at[b_idx, s, a, ns].add(1.0)
    N_sa = state.N_sa.at[b_idx, s, a].add(1.0)
    R_sum = state.R_sum.at[b_idx, s, a].add(r.reshape(-1))

    _, _, row_p = _compute_model_error_noisy_row_pure(
        row_counts=N_sas[b_idx, s, a],
        row_states=s,
        n_s=params.n_s,
        lambda_val=params.lambda_val,
        model_error_mode=params.model_error_mode,
    )
    P_hat = state.P_hat.at[b_idx, s, a].set(row_p)

    future_sample = current_V[b_idx, ns]
    surrogate_expectation = jnp.sum(row_p * current_V, axis=1)
    target = r.reshape(-1) + gamma_vec * future_sample - gamma_vec * surrogate_expectation

    lr = compute_lr_pure(k, params.lr_start, params.lr_delay, params.lr_schedule).reshape(-1)
    updated_defect = current_defect.at[b_idx, s, a].add(
        lr * (target - current_defect[b_idx, s, a])
    )

    committed_V = surrogate_vi_solve(
        P_hat,
        updated_defect,
        current_V,
        gamma_vec.reshape(-1, 1, 1),
        tol=params.surrogate_tol,
        max_iters=params.surrogate_iters,
    )

    return AgentState(
        V=committed_V,
        P_hat=P_hat,
        R_hat=state.R_hat,
        N_sas=N_sas,
        N_sa=N_sa,
        R_sum=R_sum,
        states=state.states,
        J_per_env=state.J_per_env,
        prev_score=state.prev_score,
        rng_key=state.rng_key,
        r_k=updated_defect,
        Q=state.Q,
        bar_r=updated_defect,
        h_s=state.h_s,
        R_E_factor=state.R_E_factor,
        V_checkpoint=committed_V,
        steps_since_check=jnp.zeros_like(state.steps_since_check),
        working_V=committed_V,
        policy_cache=state.policy_cache,
        waterfall_h=state.waterfall_h,
        waterfall_g_best=state.waterfall_g_best,
        waterfall_g_candidate=state.waterfall_g_candidate,
        waterfall_reward_abs_max=jnp.max(jnp.abs(updated_defect), axis=(1, 2)),
        waterfall_inner_depth=jnp.zeros((state.V.shape[0],), dtype=jnp.int32),
        waterfall_last_status=state.waterfall_last_status,
        waterfall_d_base=state.waterfall_d_base,
        waterfall_d_refine=state.waterfall_d_refine,
        defect_r=updated_defect,
        defect_avg=state.defect_avg,
        defect_init=defect_init,
        defect_clip_fraction=state.defect_clip_fraction,
        defect_gamma_bound=state.defect_gamma_bound,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )


@jax.jit
def nr_rm_isp_dyna_update_with_sample_pure(
    state: AgentState,
    params: AlgorithmParams,
    s: jnp.ndarray,
    a: jnp.ndarray,
    r: jnp.ndarray,
    ns: jnp.ndarray,
    k: int,
) -> AgentState:
    """Apply one maintained NR-RM-ISP-Dyna physical sample to batched state."""
    normalized_state = _nr_rm_isp_normalize_state(state)
    mode_code = jnp.asarray(params.nr_mode, dtype=jnp.int32)
    generalized = jnp.any(mode_code != 0)
    return lax.cond(
        generalized,
        lambda _: _nr_rm_isp_generalized_update_with_sample_pure(
            normalized_state,
            params,
            s,
            a,
            r,
            ns,
            k,
        ),
        lambda _: _nr_rm_isp_limit_update_with_sample_pure(
            normalized_state,
            params,
            s,
            a,
            r,
            ns,
            k,
        ),
        operand=None,
    )


@jax.jit
def nr_rm_isp_dyna_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """Pure JAX step for the exact no-reset reward-modified limit under generative sampling."""
    s, a, r, ns, rng_key = sample_generative_transition_pure(
        mdp.P,
        mdp.R,
        state.rng_key,
        state.V.shape[0],
        params.n_s,
        params.n_a,
        params.sampling_distribution,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )
    updated = nr_rm_isp_dyna_update_with_sample_pure(state, params, s, a, r, ns, k)
    updated = updated._replace(
        states=s,
        rng_key=rng_key,
        replay_sample_index=advance_replay_sample_index(
            state.replay_sample_index,
            state.replay_states,
        ),
    )
    return updated, jnp.mean(updated.V[:, 0])


@jax.jit
def run_n_steps_nr_rm_isp_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    """Run N steps of the exact no-reset reward-modified limit in one JIT loop."""

    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = nr_rm_isp_dyna_step_pure(bst, mdp, params, curr_k)
        return (nst, val)

    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


@jax.jit
def _nr1_rm_isp_generalized_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    batch_size = state.V.shape[0]
    b_idx = jnp.arange(batch_size)
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    current_h = _nr_rm_isp_h_for_step(params, k, batch_size)
    disc = ((current_h * gamma_vec) / (1.0 + current_h)).reshape(-1, 1, 1)
    j_scalar = jnp.asarray(params.J_init, dtype=jnp.int32).reshape(-1)[0]
    anchor_V = jnp.asarray(state.V, dtype=jnp.float64)
    initial_inner = (
        jnp.asarray(state.working_V, dtype=jnp.float64)
        if state.working_V is not None
        else anchor_V
    )
    initial_defect = (
        jnp.asarray(state.defect_r, dtype=jnp.float64)
        if state.defect_r is not None
        else jnp.asarray(state.r_k, dtype=jnp.float64)
    )
    defect_init = (
        jnp.asarray(state.defect_init, dtype=jnp.float64)
        if state.defect_init is not None
        else jnp.zeros_like(initial_defect)
    )
    outer_start_sample = jnp.asarray(k, dtype=jnp.int32) * j_scalar

    def body_fn(i: int, carry: tuple[AgentState, jnp.ndarray, jnp.ndarray]):
        carry_state, carry_inner, carry_defect = carry
        sample_index = jnp.asarray(carry_state.replay_sample_index, dtype=jnp.int32) + i
        s, a, r, ns, rng_key = sample_generative_transition_pure(
            mdp.P,
            mdp.R,
            carry_state.rng_key,
            batch_size,
            params.n_s,
            params.n_a,
            params.sampling_distribution,
            replay_states=carry_state.replay_states,
            replay_actions=carry_state.replay_actions,
            replay_next_states=carry_state.replay_next_states,
            replay_row_index=carry_state.replay_row_index,
            replay_sample_index=sample_index,
        )
        N_sas = carry_state.N_sas.at[b_idx, s, a, ns].add(1.0)
        N_sa = carry_state.N_sa.at[b_idx, s, a].add(1.0)
        R_sum = carry_state.R_sum.at[b_idx, s, a].add(r.reshape(-1))
        _, _, row_p = _compute_model_error_noisy_row_pure(
            row_counts=N_sas[b_idx, s, a],
            row_states=s,
            n_s=params.n_s,
            lambda_val=params.lambda_val,
            model_error_mode=params.model_error_mode,
        )
        P_hat = carry_state.P_hat.at[b_idx, s, a].set(row_p)
        future_sample = carry_inner[b_idx, ns]
        surrogate_expectation = jnp.sum(row_p * carry_inner, axis=1)
        target = r.reshape(-1) + gamma_vec * future_sample - gamma_vec * surrogate_expectation

        lr = compute_lr_pure(
            outer_start_sample + i,
            params.lr_start,
            params.lr_delay,
            params.lr_schedule,
        ).reshape(-1)
        updated_defect = carry_defect.at[b_idx, s, a].add(
            lr * (target - carry_defect[b_idx, s, a])
        )
        shifted_reward = _rm_isp_shifted_reward_pure(anchor_V, updated_defect, current_h)
        candidate_inner = surrogate_vi_solve(
            P_hat,
            shifted_reward,
            carry_inner,
            disc,
            tol=params.surrogate_tol,
            max_iters=params.surrogate_iters,
        )
        next_state = carry_state._replace(
            P_hat=P_hat,
            N_sas=N_sas,
            N_sa=N_sa,
            R_sum=R_sum,
            states=s,
            rng_key=rng_key,
        )
        return next_state, candidate_inner, updated_defect

    stepped_state, final_inner, final_defect = lax.fori_loop(
        0,
        j_scalar,
        body_fn,
        (state, initial_inner, initial_defect),
    )
    committed_shifted_reward = _rm_isp_shifted_reward_pure(final_inner, final_defect, current_h)
    updated = stepped_state._replace(
        V=final_inner,
        r_k=final_defect,
        bar_r=committed_shifted_reward,
        V_checkpoint=final_inner,
        steps_since_check=jnp.zeros_like(state.steps_since_check),
        working_V=final_inner,
        waterfall_h=current_h,
        waterfall_reward_abs_max=jnp.max(jnp.abs(final_defect), axis=(1, 2)),
        waterfall_inner_depth=jnp.zeros((batch_size,), dtype=jnp.int32),
        defect_r=final_defect,
        defect_init=defect_init,
        replay_sample_index=advance_replay_sample_index(
            state.replay_sample_index,
            state.replay_states,
            delta=j_scalar,
        ),
    )
    return updated, jnp.mean(updated.V[:, 0])


@jax.jit
def nr1_rm_isp_dyna_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    normalized_state = _nr_rm_isp_normalize_state(state)
    mode_code = jnp.asarray(params.nr_mode, dtype=jnp.int32)
    generalized = jnp.any(mode_code != 0)

    def _generalized(_: None) -> Tuple[AgentState, jax.Array]:
        return _nr1_rm_isp_generalized_step_pure(normalized_state, mdp, params, k)

    def _limit(_: None) -> Tuple[AgentState, jax.Array]:
        s, a, r, ns, rng_key = sample_generative_transition_pure(
            mdp.P,
            mdp.R,
            normalized_state.rng_key,
            normalized_state.V.shape[0],
            params.n_s,
            params.n_a,
            params.sampling_distribution,
            replay_states=normalized_state.replay_states,
            replay_actions=normalized_state.replay_actions,
            replay_next_states=normalized_state.replay_next_states,
            replay_row_index=normalized_state.replay_row_index,
            replay_sample_index=normalized_state.replay_sample_index,
        )
        updated_state = _nr1_rm_isp_limit_update_with_sample_pure(
            normalized_state,
            params,
            s,
            a,
            r,
            ns,
            k,
        )
        updated_state = updated_state._replace(
            states=s,
            rng_key=rng_key,
            replay_sample_index=advance_replay_sample_index(
                normalized_state.replay_sample_index,
                normalized_state.replay_states,
            ),
        )
        return updated_state, jnp.mean(updated_state.V[:, 0])

    return lax.cond(generalized, _generalized, _limit, operand=None)


@jax.jit
def run_n_steps_nr1_rm_isp_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = nr1_rm_isp_dyna_step_pure(bst, mdp, params, curr_k)
        return (nst, val)

    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


class JaxBatchedNRRMISPDyna:
    """Maintained exact no-reset reward-modified limit with persistent defect reuse."""

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        lambda_val=0.0,
        h=None,
        J=None,
        h_init=None,
        h_terminal=None,
        h_floor=1.0,
        h_schedule_horizon=None,
        h_schedule_shape="exponential",
        h_schedule_finish_frac=1.0,
        lr_params=(0.5, 0),
        lr_schedule="constant_delay",
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
        seed=0,
        sampling_distribution="uniform_sa",
        trajectory_mode=False,
    ):
        del trajectory_mode
        if str(sampling_distribution).lower() != "uniform_sa":
            raise ValueError("NR-RM-ISP-Dyna currently supports only sampling_distribution='uniform_sa'.")

        self.num_envs = num_envs
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        self.mdp_data = MDPData(P=self.mdp_P, R=self.mdp_R, gamma=gamma)
        self.mdp_data, self.state = create_initial_state(
            self.mdp_P,
            self.mdp_R,
            num_envs,
            0.0,
            J_init=(
                int(jnp.asarray(J).reshape(-1)[0])
                if J is not None and jnp.asarray(J).ndim == 0
                else (1 if J is None else 0)
            ),
            seed=seed,
        )

        start_lr, delay = lr_params
        lr_start = as_column_parameter(start_lr, num_envs, dtype=jnp.float64)
        lr_delay = as_column_parameter(delay, num_envs, dtype=jnp.float64)
        sched_int = 0 if str(lr_schedule).lower() in {"constant_delay", "constantanddelay"} else 1
        gam_arr = as_vector_parameter(gamma, num_envs, dtype=jnp.float64)
        lam_arr = as_vector_parameter(lambda_val, num_envs, dtype=jnp.float64)
        # Defect is always zero-initialized; no oracle reward prior is permitted.
        initial_defect = jnp.zeros_like(self.state.R_hat)
        # J participates in both generalized NR modes, so it cannot be used to
        # distinguish fixed-h from scheduled-h at construction time.
        has_fixed_mode = h is not None
        has_scheduled_mode = h_init is not None or h_terminal is not None
        if has_fixed_mode and has_scheduled_mode:
            raise ValueError(
                "NR-RM-ISP-Dyna must define exactly one mode: limit, fixed_h, or scheduled_h."
            )
        if has_fixed_mode:
            if h is None or J is None:
                raise ValueError("NR-RM-ISP-Dyna fixed_h mode requires both h and J.")
            if h_schedule_shape != "exponential" or float(h_schedule_finish_frac) != 1.0:
                raise ValueError("NR-RM-ISP-Dyna fixed_h mode must not define schedule-shape fields.")
            nr_mode = 1
            h_arr = as_vector_parameter(h, num_envs, dtype=jnp.float64)
            j_arr = as_vector_parameter(J, num_envs, dtype=jnp.int32)
            h_terminal_arr = h_arr
            h_floor_arr = jnp.full((num_envs,), float(h_floor), dtype=jnp.float64)
            schedule_horizon = 0
            schedule_shape_code = 0
            schedule_finish_frac = 1.0
        elif has_scheduled_mode:
            if J is None or h_init is None or h_terminal is None:
                raise ValueError(
                    "NR-RM-ISP-Dyna scheduled_h mode requires J, h_init, and h_terminal."
                )
            nr_mode = 2
            h_arr = as_vector_parameter(h_init, num_envs, dtype=jnp.float64)
            j_arr = as_vector_parameter(J, num_envs, dtype=jnp.int32)
            h_terminal_arr = as_vector_parameter(h_terminal, num_envs, dtype=jnp.float64)
            h_floor_arr = as_vector_parameter(h_floor, num_envs, dtype=jnp.float64)
            schedule_horizon = max(1, int(h_schedule_horizon or 1))
            normalized_schedule_shape = str(h_schedule_shape).strip().lower().replace("-", "_")
            if normalized_schedule_shape not in {"exponential", "linear"}:
                raise ValueError("NR-RM-ISP-Dyna scheduled_h mode requires h_schedule_shape in {'exponential', 'linear'}.")
            schedule_shape_code = 0 if normalized_schedule_shape == "exponential" else 1
            schedule_finish_frac = float(h_schedule_finish_frac)
            if not (0.0 < schedule_finish_frac <= 1.0):
                raise ValueError(
                    "NR-RM-ISP-Dyna scheduled_h mode requires 0 < h_schedule_finish_frac <= 1."
                )
        else:
            if h_schedule_shape != "exponential" or float(h_schedule_finish_frac) != 1.0:
                raise ValueError("NR-RM-ISP-Dyna limit mode must not define schedule-shape fields.")
            nr_mode = 0
            h_arr = jnp.ones((num_envs,), dtype=jnp.float64)
            j_arr = jnp.ones((num_envs,), dtype=jnp.int32)
            h_terminal_arr = h_arr
            h_floor_arr = jnp.ones((num_envs,), dtype=jnp.float64)
            schedule_horizon = 0
            schedule_shape_code = 0
            schedule_finish_frac = 1.0

        if nr_mode == 0:
            initial_value = surrogate_vi_solve(
                self.state.P_hat,
                initial_defect,
                self.state.V,
                gam_arr.reshape(-1, 1, 1),
                tol=vi_tol,
                max_iters=vi_max_iters,
            )

            self.state = self.state._replace(
                V=initial_value,
                r_k=initial_defect,
                bar_r=initial_defect,
                V_checkpoint=initial_value,
                working_V=initial_value,
                defect_r=initial_defect,
                defect_init=initial_defect,
                waterfall_h=jnp.ones((num_envs,), dtype=jnp.float64),
                waterfall_inner_depth=jnp.zeros((num_envs,), dtype=jnp.int32),
                waterfall_reward_abs_max=jnp.max(jnp.abs(initial_defect), axis=(1, 2)),
            )
            self.params = AlgorithmParams(
                gamma=gam_arr,
                lambda_val=lam_arr,
                h=jnp.ones((num_envs,), dtype=jnp.float64),
                J_init=jnp.ones((num_envs,), dtype=jnp.int32),
                surrogate_tol=vi_tol,
                surrogate_iters=vi_max_iters,
                max_J=1,
                n_s=self.n_s,
                n_a=self.n_a,
                disc=gam_arr,
                inv_1ph=jnp.zeros((num_envs,), dtype=jnp.float64),
                h_inv_1ph=jnp.ones((num_envs,), dtype=jnp.float64),
                lr_start=lr_start,
                lr_delay=lr_delay,
                lr_schedule=sched_int,
                planning_steps=-1,
                use_vi_solve=True,
                batch_size=jnp.ones((num_envs,), dtype=jnp.int32),
                sampling_distribution=0,
                nr_mode=0,
            )
            return

        disc = (h_arr * gam_arr) / (1.0 + h_arr)
        shifted_reward = _rm_isp_shifted_reward_pure(self.state.V, initial_defect, h_arr)
        initial_value = surrogate_vi_solve(
            self.state.P_hat,
            shifted_reward,
            self.state.V,
            disc.reshape(-1, 1, 1),
            tol=vi_tol,
            max_iters=vi_max_iters,
        )
        self.state = self.state._replace(
            V=initial_value,
            r_k=initial_defect,
            bar_r=shifted_reward,
            V_checkpoint=initial_value,
            working_V=initial_value,
            defect_r=initial_defect,
            defect_init=initial_defect,
            waterfall_h=h_arr,
            waterfall_inner_depth=jnp.zeros((num_envs,), dtype=jnp.int32),
            waterfall_reward_abs_max=jnp.max(jnp.abs(initial_defect), axis=(1, 2)),
        )
        self.params = AlgorithmParams(
            gamma=gam_arr,
            lambda_val=lam_arr,
            h=h_arr,
            J_init=j_arr,
            surrogate_tol=vi_tol,
            surrogate_iters=vi_max_iters,
            max_J=int(jnp.max(j_arr)),
            n_s=self.n_s,
            n_a=self.n_a,
            disc=disc,
            inv_1ph=1.0 / (1.0 + h_arr),
            h_inv_1ph=h_arr / (1.0 + h_arr),
            lr_start=lr_start,
            lr_delay=lr_delay,
            lr_schedule=sched_int,
            planning_steps=-1,
            use_vi_solve=True,
            batch_size=jnp.ones((num_envs,), dtype=jnp.int32),
            sampling_distribution=0,
            base_h=h_floor_arr,
            h_init=h_arr,
            h_terminal=h_terminal_arr,
            h_schedule_horizon=schedule_horizon,
            h_schedule_shape=schedule_shape_code,
            h_schedule_finish_frac=schedule_finish_frac,
            nr_mode=nr_mode,
        )

    def step(self, k: int) -> float:
        self.state, mean_val = nr_rm_isp_dyna_step_pure(self.state, self.mdp_data, self.params, k)
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        defect_r = self.state.defect_r if self.state.defect_r is not None else self.state.r_k
        current_h = self.state.waterfall_h if self.state.waterfall_h is not None else self.params.h
        return _compute_policy_nr_rm_isp_dispatch(
            defect_r,
            self.state.P_hat,
            self.state.V,
            self.params.gamma,
            current_h,
            self.params.nr_mode,
        )

    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_nr_rm_isp_dyna_pure(
            self.state,
            self.mdp_data,
            self.params,
            start_k,
            n_steps,
        )
        return float(mean_val)


class JaxBatchedNR1RMISPDyna:
    """Maintained NR1 variant with J physical samples consumed per committed outer step."""

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        lambda_val=0.0,
        h=None,
        J=None,
        h_init=None,
        h_terminal=None,
        h_floor=1.0,
        h_schedule_horizon=None,
        h_schedule_shape="exponential",
        h_schedule_finish_frac=1.0,
        lr_params=(0.5, 0),
        lr_schedule="constant_delay",
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
        seed=0,
        sampling_distribution="uniform_sa",
        trajectory_mode=False,
    ):
        delegate = JaxBatchedNRRMISPDyna(
            mdp_P=mdp_P,
            mdp_R=mdp_R,
            gamma=gamma,
            num_envs=num_envs,
            lambda_val=lambda_val,
            h=h,
            J=J,
            h_init=h_init,
            h_terminal=h_terminal,
            h_floor=h_floor,
            h_schedule_horizon=h_schedule_horizon,
            h_schedule_shape=h_schedule_shape,
            h_schedule_finish_frac=h_schedule_finish_frac,
            lr_params=lr_params,
            lr_schedule=lr_schedule,
            vi_tol=vi_tol,
            vi_max_iters=vi_max_iters,
            seed=seed,
            sampling_distribution=sampling_distribution,
            trajectory_mode=trajectory_mode,
        )
        self.num_envs = delegate.num_envs
        self.mdp_P = delegate.mdp_P
        self.mdp_R = delegate.mdp_R
        self.n_s = delegate.n_s
        self.n_a = delegate.n_a
        self.mdp_data = delegate.mdp_data
        self.state = delegate.state
        self.params = delegate.params

    def step(self, k: int) -> float:
        self.state, mean_val = nr1_rm_isp_dyna_step_pure(self.state, self.mdp_data, self.params, k)
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        defect_r = self.state.defect_r if self.state.defect_r is not None else self.state.r_k
        current_h = self.state.waterfall_h if self.state.waterfall_h is not None else self.params.h
        return _compute_policy_nr_rm_isp_dispatch(
            defect_r,
            self.state.P_hat,
            self.state.V,
            self.params.gamma,
            current_h,
            self.params.nr_mode,
        )

    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_nr1_rm_isp_dyna_pure(
            self.state,
            self.mdp_data,
            self.params,
            start_k,
            n_steps,
        )
        return float(mean_val)


def _scheduled_waterfall_h_for_step(
    params: AlgorithmParams,
    k: int,
) -> jnp.ndarray:
    h_max = jnp.asarray(params.h_init, dtype=jnp.float64).reshape(-1)
    h_min = jnp.asarray(params.base_h, dtype=jnp.float64).reshape(-1)
    delay = jnp.asarray(params.lr_delay, dtype=jnp.float64).reshape(-1)
    schedule_decay_rate = jnp.asarray(params.decay_factor, dtype=jnp.float64)
    ratio = jnp.asarray(k, dtype=jnp.float64) / jnp.maximum(1.0, delay)
    return jnp.maximum(h_min, h_max * jnp.exp(-schedule_decay_rate * ratio))


@jax.jit
def scheduled_waterfall_isp_dyna_update_with_sample_pure(
    state: AgentState,
    params: AlgorithmParams,
    s: jnp.ndarray,
    a: jnp.ndarray,
    r: jnp.ndarray,
    ns: jnp.ndarray,
    k: int,
) -> AgentState:
    """Apply one archived-style scheduled-waterfall wrapper step to manuscript ISP-Dyna."""
    current_h = _scheduled_waterfall_h_for_step(params, k)
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    effective_params = params._replace(
        h=current_h,
        disc=(current_h * gamma_vec) / (1.0 + current_h),
        inv_1ph=1.0 / (1.0 + current_h),
        h_inv_1ph=current_h / (1.0 + current_h),
        batch_size=jnp.ones_like(jnp.asarray(params.J_init, dtype=jnp.int32).reshape(-1)),
    )
    updated = isp_dyna_update_with_sample_pure(state, effective_params, s, a, r, ns, k)
    return updated._replace(waterfall_h=current_h)


@jax.jit
def scheduled_waterfall_isp_dyna_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """Pure JAX step for the archived-schedule waterfall wrapper over manuscript ISP-Dyna."""
    s, a, r, ns, rng_key = sample_generative_transition_pure(
        mdp.P,
        mdp.R,
        state.rng_key,
        state.V.shape[0],
        params.n_s,
        params.n_a,
        params.sampling_distribution,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )
    updated = scheduled_waterfall_isp_dyna_update_with_sample_pure(state, params, s, a, r, ns, k)
    updated = updated._replace(
        states=s,
        rng_key=rng_key,
        replay_sample_index=advance_replay_sample_index(
            state.replay_sample_index,
            state.replay_states,
        ),
    )
    return updated, jnp.mean(updated.V[:, 0])


@jax.jit
def run_n_steps_scheduled_waterfall_isp_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    """Run N steps of scheduled-waterfall ISP-Dyna in a single JIT-compiled loop."""
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = scheduled_waterfall_isp_dyna_step_pure(bst, mdp, params, curr_k)
        return (nst, val)

    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


class JaxBatchedScheduledWaterfallISPDyna:
    """Archived-schedule waterfall wrapper over the maintained manuscript ISP-Dyna core."""

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        h_max,
        h_min,
        J,
        lambda_val=0.0,
        lr_params=(0.02, 35000),
        lr_schedule="constant_delay",
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
        seed=0,
        schedule_decay_rate=5.0,
        sampling_distribution="uniform_sa",
        trajectory_mode=False,
    ):
        del trajectory_mode
        if str(sampling_distribution).lower() != "uniform_sa":
            raise ValueError(
                "Scheduled-Waterfall-ISP-Dyna currently supports only sampling_distribution='uniform_sa'."
            )

        self.num_envs = num_envs
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        self.mdp_data = MDPData(P=self.mdp_P, R=self.mdp_R, gamma=gamma)
        self.mdp_data, self.state = create_initial_state(
            self.mdp_P,
            self.mdp_R,
            num_envs,
            0.0,
            J_init=int(jnp.asarray(J).reshape(-1)[0]) if jnp.asarray(J).ndim == 0 else 0,
            seed=seed,
        )

        start_lr, delay = lr_params
        lr_start = as_column_parameter(start_lr, num_envs, dtype=jnp.float64)
        lr_delay = as_column_parameter(delay, num_envs, dtype=jnp.float64)
        sched_int = 0 if str(lr_schedule).lower() in {"constant_delay", "constantanddelay"} else 1
        gam_arr = as_vector_parameter(gamma, num_envs, dtype=jnp.float64)
        lam_arr = as_vector_parameter(lambda_val, num_envs, dtype=jnp.float64)
        h_max_arr = as_vector_parameter(h_max, num_envs, dtype=jnp.float64)
        h_min_arr = as_vector_parameter(h_min, num_envs, dtype=jnp.float64)
        j_arr = as_vector_parameter(J, num_envs, dtype=jnp.int32)
        unit_batch_arr = jnp.ones((num_envs,), dtype=jnp.int32)

        self.state = self.state._replace(waterfall_h=h_max_arr)
        self.params = AlgorithmParams(
            gamma=gam_arr,
            lambda_val=lam_arr,
            h=h_max_arr,
            J_init=j_arr,
            surrogate_tol=vi_tol,
            surrogate_iters=vi_max_iters,
            max_J=int(jnp.max(j_arr)),
            n_s=self.n_s,
            n_a=self.n_a,
            disc=(h_max_arr * gam_arr) / (1.0 + h_max_arr),
            inv_1ph=1.0 / (1.0 + h_max_arr),
            h_inv_1ph=h_max_arr / (1.0 + h_max_arr),
            lr_start=lr_start,
            lr_delay=lr_delay,
            lr_schedule=sched_int,
            planning_steps=-1,
            use_vi_solve=True,
            batch_size=unit_batch_arr,
            base_h=h_min_arr,
            sampling_distribution=0,
            decay_factor=schedule_decay_rate,
            h_init=h_max_arr,
        )

    def step(self, k: int) -> float:
        self.state, mean_val = scheduled_waterfall_isp_dyna_step_pure(
            self.state, self.mdp_data, self.params, k
        )
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        """Return the current policy using the live scheduled h value."""
        return _compute_policy_isp_dyna(
            self.state.bar_r,
            self.state.P_hat,
            self.state.V,
            self.params.gamma,
            self.state.waterfall_h,
        )

    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_scheduled_waterfall_isp_dyna_pure(
            self.state,
            self.mdp_data,
            self.params,
            start_k,
            n_steps,
        )
        return float(mean_val)


@jax.jit
def arch_waterfall_isp_dyna_update_with_sample_pure(
    state: AgentState,
    params: AlgorithmParams,
    s: jnp.ndarray,
    a: jnp.ndarray,
    r: jnp.ndarray,
    ns: jnp.ndarray,
    k: int,
) -> AgentState:
    """Apply one archived waterfall step to the archived ISP-Dyna core."""
    current_h = _scheduled_waterfall_h_for_step(params, k)
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    effective_params = params._replace(
        h=current_h,
        disc=(current_h * gamma_vec) / (1.0 + current_h),
        inv_1ph=1.0 / (1.0 + current_h),
        h_inv_1ph=current_h / (1.0 + current_h),
        batch_size=jnp.ones_like(jnp.asarray(params.J_init, dtype=jnp.int32).reshape(-1)),
    )
    updated = arch_isp_dyna_update_with_sample_pure(state, effective_params, s, a, r, ns, k)
    return updated._replace(waterfall_h=current_h)


@jax.jit
def arch_waterfall_isp_dyna_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """Pure JAX step for the archived waterfall wrapper over archived ISP-Dyna."""
    s, a, r, ns, rng_key = sample_generative_transition_pure(
        mdp.P,
        mdp.R,
        state.rng_key,
        state.V.shape[0],
        params.n_s,
        params.n_a,
        params.sampling_distribution,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index,
    )
    updated = arch_waterfall_isp_dyna_update_with_sample_pure(state, params, s, a, r, ns, k)
    updated = updated._replace(
        states=s,
        rng_key=rng_key,
        replay_sample_index=advance_replay_sample_index(
            state.replay_sample_index,
            state.replay_states,
        ),
    )
    return updated, jnp.mean(updated.V[:, 0])


@jax.jit
def run_n_steps_arch_waterfall_isp_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    """Run N steps of archived waterfall ISP-Dyna in a single JIT loop."""
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = arch_waterfall_isp_dyna_step_pure(bst, mdp, params, curr_k)
        return (nst, val)

    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


class JaxBatchedArchWaterfallISPDyna:
    """Archived waterfall wrapper over the maintained archived ISP-Dyna core."""

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        h_max,
        h_min,
        J,
        lambda_val=0.0,
        lr_params=(0.02, 35000),
        lr_schedule="constant_delay",
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
        seed=0,
        schedule_decay_rate=5.0,
        sampling_distribution="uniform_sa",
        trajectory_mode=False,
    ):
        del trajectory_mode
        if str(sampling_distribution).lower() != "uniform_sa":
            raise ValueError(
                "Arch-Waterfall-ISP-Dyna currently supports only sampling_distribution='uniform_sa'."
            )

        self.num_envs = num_envs
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        self.mdp_data = MDPData(P=self.mdp_P, R=self.mdp_R, gamma=gamma)
        self.mdp_data, self.state = create_initial_state(
            self.mdp_P,
            self.mdp_R,
            num_envs,
            0.0,
            J_init=int(jnp.asarray(J).reshape(-1)[0]) if jnp.asarray(J).ndim == 0 else 0,
            seed=seed,
        )

        start_lr, delay = lr_params
        lr_start = as_column_parameter(start_lr, num_envs, dtype=jnp.float64)
        lr_delay = as_column_parameter(delay, num_envs, dtype=jnp.float64)
        sched_int = 0 if str(lr_schedule).lower() in {"constant_delay", "constantanddelay"} else 1
        gam_arr = as_vector_parameter(gamma, num_envs, dtype=jnp.float64)
        lam_arr = as_vector_parameter(lambda_val, num_envs, dtype=jnp.float64)
        h_max_arr = as_vector_parameter(h_max, num_envs, dtype=jnp.float64)
        h_min_arr = as_vector_parameter(h_min, num_envs, dtype=jnp.float64)
        j_arr = as_vector_parameter(J, num_envs, dtype=jnp.int32)
        unit_batch_arr = jnp.ones((num_envs,), dtype=jnp.int32)

        self.state = self.state._replace(
            V_checkpoint=self.state.V,
            working_V=self.state.V,
            waterfall_h=h_max_arr,
            waterfall_inner_depth=jnp.zeros((num_envs,), dtype=jnp.int32),
        )
        self.params = AlgorithmParams(
            gamma=gam_arr,
            lambda_val=lam_arr,
            h=h_max_arr,
            J_init=j_arr,
            surrogate_tol=vi_tol,
            surrogate_iters=vi_max_iters,
            max_J=int(jnp.max(j_arr)),
            n_s=self.n_s,
            n_a=self.n_a,
            disc=(h_max_arr * gam_arr) / (1.0 + h_max_arr),
            inv_1ph=1.0 / (1.0 + h_max_arr),
            h_inv_1ph=h_max_arr / (1.0 + h_max_arr),
            lr_start=lr_start,
            lr_delay=lr_delay,
            lr_schedule=sched_int,
            planning_steps=-1,
            use_vi_solve=True,
            batch_size=unit_batch_arr,
            base_h=h_min_arr,
            sampling_distribution=0,
            decay_factor=schedule_decay_rate,
            h_init=h_max_arr,
        )

    def step(self, k: int) -> float:
        self.state, mean_val = arch_waterfall_isp_dyna_step_pure(
            self.state,
            self.mdp_data,
            self.params,
            k,
        )
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        return _compute_policy_arch_isp_dyna(
            self.state.r_k,
            self.state.P_hat,
            self.state.V,
            self.params.gamma,
            self.state.waterfall_h,
        )

    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_arch_waterfall_isp_dyna_pure(
            self.state,
            self.mdp_data,
            self.params,
            start_k,
            n_steps,
        )
        return float(mean_val)


DD_GEO_WATERFALL_STATUS_NONE = 0
DD_GEO_WATERFALL_STATUS_PROGRESS = 1
DD_GEO_WATERFALL_STATUS_INVALID = 2
DD_GEO_WATERFALL_STATUS_EXHAUSTED = 3
DD_GEO_WATERFALL_STATUS_CONTINUE = 4


@jax.jit
def dd_geo_waterfall_isp_dyna_batch_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """One DD-Geo-Waterfall ISP-Dyna atomic batch step."""
    batch_size_vec = jnp.asarray(params.batch_size, dtype=jnp.int32).reshape(-1)
    batch_size_scalar = jnp.asarray(params.inner_batch_size, dtype=jnp.int32)
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    lam_vec = jnp.asarray(params.lambda_val, dtype=jnp.float64).reshape(-1)
    decay_factor = jnp.asarray(params.decay_factor, dtype=jnp.float64)
    max_j = jnp.asarray(params.max_J, dtype=jnp.int32)

    current_h = jnp.asarray(state.waterfall_h, dtype=jnp.float64).reshape(-1)
    current_g_best = jnp.asarray(state.waterfall_g_best, dtype=jnp.float64).reshape(-1)
    current_reward_abs_max = jnp.asarray(state.waterfall_reward_abs_max, dtype=jnp.float64).reshape(-1)
    current_inner_depth = jnp.asarray(state.waterfall_inner_depth, dtype=jnp.int32).reshape(-1)
    starting_new_outer = current_inner_depth == 0

    anchor_V = jnp.where(starting_new_outer[:, None], state.V, state.V_checkpoint)
    working_V = jnp.where(starting_new_outer[:, None], state.V, state.working_V)
    bar_r = jnp.zeros_like(state.bar_r)
    P_hat = state.P_hat
    N_sas = state.N_sas
    N_sa = state.N_sa
    R_sum = state.R_sum
    rng_key = state.rng_key
    reward_abs_max = current_reward_abs_max
    last_sampled_states = state.states

    start_sample = jnp.asarray(k, dtype=jnp.int32) * jnp.asarray(batch_size_scalar, dtype=jnp.int32)

    def sample_body(i: int, carry):
        bar_r_i, P_hat_i, N_sas_i, N_sa_i, R_sum_i, rng_key_i, reward_abs_i, sampled_states_i = carry
        active = i < batch_size_vec
        s, a, r, ns, next_rng = sample_generative_transition_pure(
            mdp.P,
            mdp.R,
            rng_key_i,
            state.V.shape[0],
            params.n_s,
            params.n_a,
            params.sampling_distribution,
            replay_states=state.replay_states,
            replay_actions=state.replay_actions,
            replay_next_states=state.replay_next_states,
            replay_row_index=state.replay_row_index,
            replay_sample_index=state.replay_sample_index + i,
        )
        b_idx = jnp.arange(state.V.shape[0])
        inv_1ph = 1.0 / (1.0 + current_h)
        h_inv_1ph = current_h / (1.0 + current_h)
        prior_row = P_hat_i[b_idx, s, a]
        future_sample = working_V[b_idx, ns]
        surrogate_expectation = jnp.sum(prior_row * working_V, axis=1)
        target = (
            inv_1ph * anchor_V[b_idx, s]
            + h_inv_1ph * r.reshape(-1)
            + (h_inv_1ph * gamma_vec) * future_sample
            - (h_inv_1ph * gamma_vec) * surrogate_expectation
        )
        lr = compute_lr_pure(start_sample + i, params.lr_start, params.lr_delay, params.lr_schedule).reshape(-1)
        tracker_delta = lr * (target - bar_r_i[b_idx, s, a]) * active.astype(jnp.float64)
        next_bar_r = bar_r_i.at[b_idx, s, a].add(tracker_delta)

        next_N_sas = N_sas_i.at[b_idx, s, a, ns].add(active.astype(jnp.float64))
        next_N_sa = N_sa_i.at[b_idx, s, a].add(active.astype(jnp.float64))
        next_R_sum = R_sum_i.at[b_idx, s, a].add(r.reshape(-1) * active.astype(jnp.float64))

        row_counts = next_N_sas[b_idx, s, a]
        _, _, row_p = _compute_model_error_noisy_row_pure(
            row_counts=row_counts,
            row_states=s,
            n_s=params.n_s,
            lambda_val=lam_vec,
            model_error_mode=params.model_error_mode,
        )
        prior_update_row = P_hat_i[b_idx, s, a]
        updated_row = jnp.where(active[:, None], row_p, prior_update_row)
        next_P_hat = P_hat_i.at[b_idx, s, a].set(updated_row)

        next_reward_abs = jnp.maximum(reward_abs_i, jnp.abs(r.reshape(-1)) * active.astype(jnp.float64))
        next_sampled_states = jnp.where(active, s, sampled_states_i)
        return (
            next_bar_r,
            next_P_hat,
            next_N_sas,
            next_N_sa,
            next_R_sum,
            next_rng,
            next_reward_abs,
            next_sampled_states,
        )

    (
        bar_r,
        P_hat,
        N_sas,
        N_sa,
        R_sum,
        rng_key,
        reward_abs_max,
        last_sampled_states,
    ) = lax.fori_loop(
        0,
        batch_size_scalar,
        sample_body,
        (bar_r, P_hat, N_sas, N_sa, R_sum, rng_key, reward_abs_max, last_sampled_states),
    )

    disc = ((current_h * gamma_vec) / (1.0 + current_h)).reshape(-1, 1, 1)
    candidate_V = surrogate_vi_solve(
        P_hat,
        bar_r,
        working_V,
        disc,
        tol=params.surrogate_tol,
        max_iters=params.surrogate_iters,
    )

    finite_mask = jnp.all(jnp.isfinite(candidate_V), axis=1)
    guard = 1.0e4 * jnp.maximum(reward_abs_max / jnp.maximum(1.0 - gamma_vec, 1.0e-12), 1.0)
    invalid_mask = (~finite_mask) | (jnp.max(jnp.abs(candidate_V), axis=1) > guard)
    g_candidate = jnp.max(jnp.abs(candidate_V - anchor_V), axis=1) / current_h
    progress_mask = (~invalid_mask) & (g_candidate < current_g_best)
    exhausted_mask = (~invalid_mask) & (~progress_mask) & ((current_inner_depth + 1) >= max_j)
    continue_mask = (~invalid_mask) & (~progress_mask) & (~exhausted_mask)

    decayed_h = jnp.maximum(1.0, current_h * decay_factor)
    next_h = jnp.where(invalid_mask | exhausted_mask, decayed_h, current_h)
    next_g_best = jnp.where(progress_mask, g_candidate, current_g_best)
    committed_V = jnp.where(progress_mask[:, None], candidate_V, state.V)
    next_working_V = jnp.where(continue_mask[:, None], candidate_V, committed_V)
    next_anchor_V = jnp.where(continue_mask[:, None], anchor_V, committed_V)
    next_inner_depth = jnp.where(continue_mask, current_inner_depth + 1, 0)
    status = jnp.where(
        progress_mask,
        DD_GEO_WATERFALL_STATUS_PROGRESS,
        jnp.where(
            invalid_mask,
            DD_GEO_WATERFALL_STATUS_INVALID,
            jnp.where(
                exhausted_mask,
                DD_GEO_WATERFALL_STATUS_EXHAUSTED,
                DD_GEO_WATERFALL_STATUS_CONTINUE,
            ),
        ),
    ).astype(jnp.int32)
    candidate_policy = _compute_policy_isp_dyna(
        bar_r,
        P_hat,
        candidate_V,
        gamma_vec,
        current_h,
    ).astype(jnp.int32)
    next_policy_cache = jnp.where(
        progress_mask[:, None],
        candidate_policy,
        state.policy_cache,
    )

    next_state = AgentState(
        V=committed_V,
        P_hat=P_hat,
        R_hat=state.R_hat,
        N_sas=N_sas,
        N_sa=N_sa,
        R_sum=R_sum,
        states=last_sampled_states,
        J_per_env=state.J_per_env,
        prev_score=state.prev_score,
        rng_key=rng_key,
        r_k=state.r_k,
        Q=state.Q,
        bar_r=bar_r,
        h_s=state.h_s,
        R_E_factor=state.R_E_factor,
        V_checkpoint=next_anchor_V,
        steps_since_check=state.steps_since_check,
        working_V=next_working_V,
        policy_cache=next_policy_cache,
        waterfall_h=next_h,
        waterfall_g_best=next_g_best,
        waterfall_g_candidate=g_candidate,
        waterfall_reward_abs_max=reward_abs_max,
        waterfall_inner_depth=next_inner_depth,
        waterfall_last_status=status,
        waterfall_d_base=state.waterfall_d_base,
        waterfall_d_refine=state.waterfall_d_refine,
        defect_r=state.defect_r,
        defect_avg=state.defect_avg,
        defect_init=state.defect_init,
        defect_clip_fraction=state.defect_clip_fraction,
        defect_gamma_bound=state.defect_gamma_bound,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index + batch_size_scalar,
    )
    return next_state, jnp.mean(committed_V[:, 0])


@jax.jit
def run_n_steps_dd_geo_waterfall_isp_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = dd_geo_waterfall_isp_dyna_batch_step_pure(bst, mdp, params, curr_k)
        return (nst, val)

    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


class JaxBatchedDDGeoWaterfallISPDyna:
    """Adaptive-depth DD-Geo-Waterfall ISP-Dyna with batch-atomic stepping."""

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        h_init,
        J_max,
        batch_size,
        lambda_val=0.0,
        lr_params=(0.5, 0),
        lr_schedule="constant_delay",
        decay_factor=0.5,
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
        seed=0,
        sampling_distribution="uniform_sa",
        trajectory_mode=False,
    ):
        del trajectory_mode
        if str(sampling_distribution).lower() != "uniform_sa":
            raise ValueError(
                "DD-Geo-Waterfall-ISP-Dyna currently supports only sampling_distribution='uniform_sa'."
            )

        self.num_envs = num_envs
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        self.mdp_data = MDPData(P=self.mdp_P, R=self.mdp_R, gamma=gamma)
        self.mdp_data, self.state = create_initial_state(
            self.mdp_P,
            self.mdp_R,
            num_envs,
            0.0,
            seed=seed,
        )

        start_lr, delay = lr_params
        lr_start = as_column_parameter(start_lr, num_envs, dtype=jnp.float64)
        lr_delay = as_column_parameter(delay, num_envs, dtype=jnp.float64)
        sched_int = 0 if str(lr_schedule).lower() in {"constant_delay", "constantanddelay"} else 1
        gam_arr = as_vector_parameter(gamma, num_envs, dtype=jnp.float64)
        lam_arr = as_vector_parameter(lambda_val, num_envs, dtype=jnp.float64)
        h_init_arr = as_vector_parameter(h_init, num_envs, dtype=jnp.float64)
        j_max_arr = as_vector_parameter(J_max, num_envs, dtype=jnp.int32)
        batch_arr = as_vector_parameter(batch_size, num_envs, dtype=jnp.int32)

        initial_policy = jnp.zeros((num_envs, self.n_s), dtype=jnp.int32)
        self.state = self.state._replace(
            working_V=self.state.V,
            policy_cache=initial_policy,
            waterfall_h=h_init_arr,
            waterfall_g_best=jnp.full((num_envs,), jnp.inf, dtype=jnp.float64),
            waterfall_g_candidate=jnp.zeros((num_envs,), dtype=jnp.float64),
            waterfall_reward_abs_max=jnp.zeros((num_envs,), dtype=jnp.float64),
            waterfall_inner_depth=jnp.zeros((num_envs,), dtype=jnp.int32),
            waterfall_last_status=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

        self.params = AlgorithmParams(
            gamma=gam_arr,
            lambda_val=lam_arr,
            h=h_init_arr,
            J_init=jnp.ones((num_envs,), dtype=jnp.int32),
            surrogate_tol=vi_tol,
            surrogate_iters=vi_max_iters,
            max_J=int(jnp.max(j_max_arr)),
            n_s=self.n_s,
            n_a=self.n_a,
            disc=(h_init_arr * gam_arr) / (1.0 + h_init_arr),
            inv_1ph=1.0 / (1.0 + h_init_arr),
            h_inv_1ph=h_init_arr / (1.0 + h_init_arr),
            lr_start=lr_start,
            lr_delay=lr_delay,
            lr_schedule=sched_int,
            planning_steps=-1,
            use_vi_solve=True,
            inner_batch_size=int(jnp.max(batch_arr)),
            batch_size=batch_arr,
            sampling_distribution=0,
            decay_factor=decay_factor,
            h_init=h_init_arr,
        )

    def step(self, k: int) -> float:
        self.state, mean_val = dd_geo_waterfall_isp_dyna_batch_step_pure(
            self.state,
            self.mdp_data,
            self.params,
            k,
        )
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        return self.state.policy_cache

    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_dd_geo_waterfall_isp_dyna_pure(
            self.state,
            self.mdp_data,
            self.params,
            start_k,
            n_steps,
        )
        return float(mean_val)


@jax.jit
def elastic_dd_geo_waterfall_isp_dyna_batch_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """One Elastic DD-Geo-Waterfall ISP-Dyna atomic batch step."""
    batch_size_vec = jnp.asarray(params.batch_size, dtype=jnp.int32).reshape(-1)
    batch_size_scalar = jnp.asarray(params.inner_batch_size, dtype=jnp.int32)
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    lam_vec = jnp.asarray(params.lambda_val, dtype=jnp.float64).reshape(-1)
    decay_factor = jnp.asarray(params.decay_factor, dtype=jnp.float64)
    max_j = jnp.asarray(params.max_J, dtype=jnp.int32)
    h_init_vec = jnp.asarray(params.h_init, dtype=jnp.float64).reshape(-1)

    current_h = jnp.asarray(state.waterfall_h, dtype=jnp.float64).reshape(-1)
    current_reward_abs_max = jnp.asarray(state.waterfall_reward_abs_max, dtype=jnp.float64).reshape(-1)
    current_inner_depth = jnp.asarray(state.waterfall_inner_depth, dtype=jnp.int32).reshape(-1)
    current_d_base = jnp.asarray(state.waterfall_d_base, dtype=jnp.float64).reshape(-1)
    current_g_best = jnp.asarray(state.waterfall_g_best, dtype=jnp.float64).reshape(-1)
    starting_new_outer = current_inner_depth == 0
    epoch_g_best = jnp.where(starting_new_outer, jnp.inf, current_g_best)

    anchor_V = jnp.where(starting_new_outer[:, None], state.V, state.V_checkpoint)
    working_V = jnp.where(starting_new_outer[:, None], state.V, state.working_V)
    dense_prior = (
        (1.0 / (1.0 + current_h))[:, None, None] * anchor_V[:, :, None]
        + (current_h / (1.0 + current_h))[:, None, None] * state.R_hat
    )
    previous_bar_r = jnp.asarray(state.bar_r, dtype=jnp.float64)
    bar_r = dense_prior + lam_vec[:, None, None] * (previous_bar_r - dense_prior)
    P_hat = state.P_hat
    R_hat = state.R_hat
    N_sas = state.N_sas
    N_sa = state.N_sa
    R_sum = state.R_sum
    rng_key = state.rng_key
    reward_abs_max = current_reward_abs_max
    last_sampled_states = state.states

    start_sample = jnp.asarray(k, dtype=jnp.int32) * jnp.asarray(batch_size_scalar, dtype=jnp.int32)

    def sample_body(i: int, carry):
        bar_r_i, P_hat_i, R_hat_i, N_sas_i, N_sa_i, R_sum_i, rng_key_i, reward_abs_i, sampled_states_i = carry
        active = i < batch_size_vec
        active_f = active.astype(jnp.float64)
        s, a, r, ns, next_rng = sample_generative_transition_pure(
            mdp.P,
            mdp.R,
            rng_key_i,
            state.V.shape[0],
            params.n_s,
            params.n_a,
            params.sampling_distribution,
            replay_states=state.replay_states,
            replay_actions=state.replay_actions,
            replay_next_states=state.replay_next_states,
            replay_row_index=state.replay_row_index,
            replay_sample_index=state.replay_sample_index + i,
        )
        b_idx = jnp.arange(state.V.shape[0])
        inv_1ph = 1.0 / (1.0 + current_h)
        h_inv_1ph = current_h / (1.0 + current_h)
        prior_row = P_hat_i[b_idx, s, a]
        future_sample = working_V[b_idx, ns]
        surrogate_expectation = jnp.sum(prior_row * working_V, axis=1)

        next_N_sa = N_sa_i.at[b_idx, s, a].add(active_f)
        next_R_sum = R_sum_i.at[b_idx, s, a].add(r.reshape(-1) * active_f)
        row_visits = next_N_sa[b_idx, s, a]
        row_reward_sum = next_R_sum[b_idx, s, a]
        row_reward_mean = jnp.where(
            row_visits > 0.0,
            row_reward_sum / row_visits,
            jnp.zeros_like(row_visits),
        )
        prior_reward_row = R_hat_i[b_idx, s, a]
        updated_reward_row = jnp.where(active, row_reward_mean, prior_reward_row)
        next_R_hat = R_hat_i.at[b_idx, s, a].set(updated_reward_row)

        next_N_sas = N_sas_i.at[b_idx, s, a, ns].add(active_f)
        row_counts = next_N_sas[b_idx, s, a]
        row_emp, row_prior, _ = _compute_model_error_noisy_row_pure(
            row_counts=row_counts,
            row_states=s,
            n_s=params.n_s,
            lambda_val=lam_vec,
            model_error_mode=params.model_error_mode,
        )
        reuse_lambda = 1.0 - jnp.square(1.0 - lam_vec)
        reuse_mid_boost = reuse_lambda - lam_vec
        visit_confidence = row_visits / (row_visits + current_h)
        row_total = jnp.sum(row_counts, axis=1, keepdims=True)
        row_confidence = row_total / (row_total + current_h.reshape(-1, 1))
        row_lambda = lam_vec.reshape(-1, 1) * (1.0 - row_confidence)
        row_p = row_emp * (1.0 - row_lambda) + row_prior * row_lambda
        row_future_expectation = jnp.sum(row_p * working_V, axis=1)
        reuse_weight = (
            lam_vec * visit_confidence
            + reuse_mid_boost * jnp.sqrt(visit_confidence)
        )
        future_bridge = (
            (1.0 - reuse_weight) * future_sample
            + reuse_weight * row_future_expectation
        )
        future_correction = future_bridge - surrogate_expectation
        sample_reward = r.reshape(-1)
        reward_bridge = (
            (1.0 - reuse_weight) * sample_reward
            + reuse_weight * row_reward_mean
        )
        target = (
            inv_1ph * anchor_V[b_idx, s]
            + h_inv_1ph * reward_bridge
            + (h_inv_1ph * gamma_vec) * future_correction
        )
        prior_update_row = P_hat_i[b_idx, s, a]
        updated_row = jnp.where(active[:, None], row_p, prior_update_row)
        next_P_hat = P_hat_i.at[b_idx, s, a].set(updated_row)

        lr = compute_lr_pure(start_sample + i, params.lr_start, params.lr_delay, params.lr_schedule).reshape(-1)
        tracker_delta = lr * (target - bar_r_i[b_idx, s, a]) * active_f
        next_bar_r = bar_r_i.at[b_idx, s, a].add(tracker_delta)

        next_reward_abs = jnp.maximum(reward_abs_i, jnp.abs(r.reshape(-1)) * active_f)
        next_sampled_states = jnp.where(active, s, sampled_states_i)
        return (
            next_bar_r,
            next_P_hat,
            next_R_hat,
            next_N_sas,
            next_N_sa,
            next_R_sum,
            next_rng,
            next_reward_abs,
            next_sampled_states,
        )

    (
        bar_r,
        P_hat,
        R_hat,
        N_sas,
        N_sa,
        R_sum,
        rng_key,
        reward_abs_max,
        last_sampled_states,
    ) = lax.fori_loop(
        0,
        batch_size_scalar,
        sample_body,
        (bar_r, P_hat, R_hat, N_sas, N_sa, R_sum, rng_key, reward_abs_max, last_sampled_states),
    )

    disc = ((current_h * gamma_vec) / (1.0 + current_h)).reshape(-1, 1, 1)
    candidate_V = surrogate_vi_solve(
        P_hat,
        bar_r,
        working_V,
        disc,
        tol=params.surrogate_tol,
        max_iters=params.surrogate_iters,
    )

    finite_mask = jnp.all(jnp.isfinite(candidate_V), axis=1)
    guard = 1.0e4 * jnp.maximum(reward_abs_max / jnp.maximum(1.0 - gamma_vec, 1.0e-12), 1.0)
    invalid_mask = (~finite_mask) | (jnp.max(jnp.abs(candidate_V), axis=1) > guard)

    safe_candidate = jnp.where(jnp.isfinite(candidate_V), candidate_V, working_V)
    base_distance = jnp.max(jnp.abs(safe_candidate - anchor_V), axis=1)
    refine_distance = jnp.max(jnp.abs(safe_candidate - working_V), axis=1)
    has_base = current_inner_depth > 0
    next_d_base = jnp.where(has_base, current_d_base, base_distance)
    next_d_refine = jnp.where(has_base, refine_distance, jnp.zeros_like(refine_distance))
    g_candidate = base_distance / current_h

    progress_mask = (~invalid_mask) & has_base & (next_d_refine <= next_d_base)
    exhausted_mask = (~invalid_mask) & (~progress_mask) & ((current_inner_depth + 1) >= max_j)
    continue_mask = (~invalid_mask) & (~progress_mask) & (~exhausted_mask)
    best_candidate_mask = (~invalid_mask) & (g_candidate < epoch_g_best)
    best_epoch_candidate = jnp.where(best_candidate_mask[:, None], safe_candidate, working_V)
    next_g_best = jnp.where(
        invalid_mask,
        epoch_g_best,
        jnp.where(has_base, jnp.minimum(epoch_g_best, g_candidate), g_candidate),
    )

    decayed_h = jnp.maximum(1.0, current_h * decay_factor)
    recovered_h = jnp.minimum(h_init_vec, current_h / decay_factor)
    next_h = jnp.where(
        progress_mask,
        recovered_h,
        jnp.where(invalid_mask | exhausted_mask, decayed_h, current_h),
    )

    committed_V = jnp.where(progress_mask[:, None], best_epoch_candidate, state.V)
    next_working_V = jnp.where(continue_mask[:, None], best_epoch_candidate, committed_V)
    next_anchor_V = jnp.where(continue_mask[:, None], anchor_V, committed_V)
    next_inner_depth = jnp.where(continue_mask, current_inner_depth + 1, 0)
    status = jnp.where(
        progress_mask,
        DD_GEO_WATERFALL_STATUS_PROGRESS,
        jnp.where(
            invalid_mask,
            DD_GEO_WATERFALL_STATUS_INVALID,
            jnp.where(
                exhausted_mask,
                DD_GEO_WATERFALL_STATUS_EXHAUSTED,
                DD_GEO_WATERFALL_STATUS_CONTINUE,
            ),
        ),
    ).astype(jnp.int32)
    candidate_policy = _compute_policy_isp_dyna(
        bar_r,
        P_hat,
        best_epoch_candidate,
        gamma_vec,
        current_h,
    ).astype(jnp.int32)
    next_policy_cache = jnp.where(
        progress_mask[:, None],
        candidate_policy,
        state.policy_cache,
    )

    next_state = AgentState(
        V=committed_V,
        P_hat=P_hat,
        R_hat=R_hat,
        N_sas=N_sas,
        N_sa=N_sa,
        R_sum=R_sum,
        states=last_sampled_states,
        J_per_env=state.J_per_env,
        prev_score=state.prev_score,
        rng_key=rng_key,
        r_k=state.r_k,
        Q=state.Q,
        bar_r=bar_r,
        h_s=state.h_s,
        R_E_factor=state.R_E_factor,
        V_checkpoint=next_anchor_V,
        steps_since_check=state.steps_since_check,
        working_V=next_working_V,
        policy_cache=next_policy_cache,
        waterfall_h=next_h,
        waterfall_g_best=next_g_best,
        waterfall_g_candidate=g_candidate,
        waterfall_reward_abs_max=reward_abs_max,
        waterfall_inner_depth=next_inner_depth,
        waterfall_last_status=status,
        waterfall_d_base=next_d_base,
        waterfall_d_refine=next_d_refine,
        defect_r=state.defect_r,
        defect_avg=state.defect_avg,
        defect_init=state.defect_init,
        defect_clip_fraction=state.defect_clip_fraction,
        defect_gamma_bound=state.defect_gamma_bound,
        replay_states=state.replay_states,
        replay_actions=state.replay_actions,
        replay_next_states=state.replay_next_states,
        replay_row_index=state.replay_row_index,
        replay_sample_index=state.replay_sample_index + batch_size_scalar,
    )
    return next_state, jnp.mean(committed_V[:, 0])


@jax.jit
def run_n_steps_elastic_dd_geo_waterfall_isp_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = elastic_dd_geo_waterfall_isp_dyna_batch_step_pure(bst, mdp, params, curr_k)
        return (nst, val)

    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


class JaxBatchedElasticDDGeoWaterfallISPDyna:
    """Elastic memoryless DD-Geo-Waterfall ISP-Dyna with batch-atomic stepping."""

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        h_init,
        J_max,
        batch_size,
        lambda_val=0.0,
        lr_params=(0.5, 0),
        lr_schedule="constant_delay",
        decay_factor=0.5,
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
        seed=0,
        sampling_distribution="uniform_sa",
        trajectory_mode=False,
    ):
        del trajectory_mode
        if str(sampling_distribution).lower() != "uniform_sa":
            raise ValueError(
                "Elastic-DD-Geo-Waterfall-ISP-Dyna currently supports only sampling_distribution='uniform_sa'."
            )

        self.num_envs = num_envs
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        self.mdp_data = MDPData(P=self.mdp_P, R=self.mdp_R, gamma=gamma)
        self.mdp_data, self.state = create_initial_state(
            self.mdp_P,
            self.mdp_R,
            num_envs,
            0.0,
            seed=seed,
        )

        start_lr, delay = lr_params
        lr_start = as_column_parameter(start_lr, num_envs, dtype=jnp.float64)
        lr_delay = as_column_parameter(delay, num_envs, dtype=jnp.float64)
        sched_int = 0 if str(lr_schedule).lower() in {"constant_delay", "constantanddelay"} else 1
        gam_arr = as_vector_parameter(gamma, num_envs, dtype=jnp.float64)
        lam_arr = as_vector_parameter(lambda_val, num_envs, dtype=jnp.float64)
        h_init_arr = as_vector_parameter(h_init, num_envs, dtype=jnp.float64)
        j_max_arr = as_vector_parameter(J_max, num_envs, dtype=jnp.int32)
        batch_arr = as_vector_parameter(batch_size, num_envs, dtype=jnp.int32)

        initial_policy = jnp.zeros((num_envs, self.n_s), dtype=jnp.int32)
        zero_r_hat = jnp.zeros((num_envs, self.n_s, self.n_a), dtype=jnp.float64)
        self.state = self.state._replace(
            R_hat=zero_r_hat,
            working_V=self.state.V,
            policy_cache=initial_policy,
            waterfall_h=h_init_arr,
            waterfall_g_best=jnp.full((num_envs,), jnp.inf, dtype=jnp.float64),
            waterfall_g_candidate=jnp.zeros((num_envs,), dtype=jnp.float64),
            waterfall_reward_abs_max=jnp.zeros((num_envs,), dtype=jnp.float64),
            waterfall_inner_depth=jnp.zeros((num_envs,), dtype=jnp.int32),
            waterfall_last_status=jnp.zeros((num_envs,), dtype=jnp.int32),
            waterfall_d_base=jnp.zeros((num_envs,), dtype=jnp.float64),
            waterfall_d_refine=jnp.zeros((num_envs,), dtype=jnp.float64),
        )

        self.params = AlgorithmParams(
            gamma=gam_arr,
            lambda_val=lam_arr,
            h=h_init_arr,
            J_init=jnp.ones((num_envs,), dtype=jnp.int32),
            surrogate_tol=vi_tol,
            surrogate_iters=vi_max_iters,
            max_J=int(jnp.max(j_max_arr)),
            n_s=self.n_s,
            n_a=self.n_a,
            disc=(h_init_arr * gam_arr) / (1.0 + h_init_arr),
            inv_1ph=1.0 / (1.0 + h_init_arr),
            h_inv_1ph=h_init_arr / (1.0 + h_init_arr),
            lr_start=lr_start,
            lr_delay=lr_delay,
            lr_schedule=sched_int,
            planning_steps=-1,
            use_vi_solve=True,
            inner_batch_size=int(jnp.max(batch_arr)),
            batch_size=batch_arr,
            sampling_distribution=0,
            decay_factor=decay_factor,
            h_init=h_init_arr,
        )

    def step(self, k: int) -> float:
        self.state, mean_val = elastic_dd_geo_waterfall_isp_dyna_batch_step_pure(
            self.state,
            self.mdp_data,
            self.params,
            k,
        )
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        return self.state.policy_cache

    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_elastic_dd_geo_waterfall_isp_dyna_pure(
            self.state,
            self.mdp_data,
            self.params,
            start_k,
            n_steps,
        )
        return float(mean_val)


# =============================================================================
# NR1 Waterfall (adaptive h) — Staircase / Geo
# =============================================================================

_WATERFALL_NR1_G_BEST_SENTINEL = 1e30
_WATERFALL_NR1_SCHEDULE_STAIRCASE = 0
_WATERFALL_NR1_SCHEDULE_GEO = 1


def _nr1_waterfall_run_inner_j_samples(state, mdp, params, k, current_h, j_scalar):
    """Mirror of the J-sample fori_loop body used by `_nr1_rm_isp_generalized_step_pure`
    but with `current_h` supplied externally rather than derived from schedule."""
    batch_size = state.V.shape[0]
    b_idx = jnp.arange(batch_size)
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    disc = ((current_h * gamma_vec) / (1.0 + current_h)).reshape(-1, 1, 1)
    anchor_V = jnp.asarray(state.V, dtype=jnp.float64)
    initial_inner = (
        jnp.asarray(state.working_V, dtype=jnp.float64)
        if state.working_V is not None
        else anchor_V
    )
    initial_defect = (
        jnp.asarray(state.defect_r, dtype=jnp.float64)
        if state.defect_r is not None
        else jnp.asarray(state.r_k, dtype=jnp.float64)
    )
    defect_init = (
        jnp.asarray(state.defect_init, dtype=jnp.float64)
        if state.defect_init is not None
        else jnp.zeros_like(initial_defect)
    )
    outer_start_sample = jnp.asarray(k, dtype=jnp.int32) * j_scalar

    def body_fn(i, carry):
        carry_state, carry_inner, carry_defect = carry
        sample_index = jnp.asarray(carry_state.replay_sample_index, dtype=jnp.int32) + i
        s, a, r, ns, rng_key = sample_generative_transition_pure(
            mdp.P,
            mdp.R,
            carry_state.rng_key,
            batch_size,
            params.n_s,
            params.n_a,
            params.sampling_distribution,
            replay_states=carry_state.replay_states,
            replay_actions=carry_state.replay_actions,
            replay_next_states=carry_state.replay_next_states,
            replay_row_index=carry_state.replay_row_index,
            replay_sample_index=sample_index,
        )
        N_sas = carry_state.N_sas.at[b_idx, s, a, ns].add(1.0)
        N_sa = carry_state.N_sa.at[b_idx, s, a].add(1.0)
        R_sum = carry_state.R_sum.at[b_idx, s, a].add(r.reshape(-1))
        _, _, row_p = _compute_model_error_noisy_row_pure(
            row_counts=N_sas[b_idx, s, a],
            row_states=s,
            n_s=params.n_s,
            lambda_val=params.lambda_val,
            model_error_mode=params.model_error_mode,
        )
        P_hat = carry_state.P_hat.at[b_idx, s, a].set(row_p)
        future_sample = carry_inner[b_idx, ns]
        surrogate_expectation = jnp.sum(row_p * carry_inner, axis=1)
        target = r.reshape(-1) + gamma_vec * future_sample - gamma_vec * surrogate_expectation
        lr = compute_lr_pure(
            outer_start_sample + i,
            params.lr_start,
            params.lr_delay,
            params.lr_schedule,
        ).reshape(-1)
        updated_defect = carry_defect.at[b_idx, s, a].add(
            lr * (target - carry_defect[b_idx, s, a])
        )
        shifted_reward = _rm_isp_shifted_reward_pure(anchor_V, updated_defect, current_h)
        candidate_inner = surrogate_vi_solve(
            P_hat,
            shifted_reward,
            carry_inner,
            disc,
            tol=params.surrogate_tol,
            max_iters=params.surrogate_iters,
        )
        next_state = carry_state._replace(
            P_hat=P_hat,
            N_sas=N_sas,
            N_sa=N_sa,
            R_sum=R_sum,
            states=s,
            rng_key=rng_key,
        )
        return next_state, candidate_inner, updated_defect

    stepped_state, final_inner, final_defect = lax.fori_loop(
        0,
        j_scalar,
        body_fn,
        (state, initial_inner, initial_defect),
    )
    return stepped_state, final_inner, final_defect, anchor_V, defect_init


def _nr1_waterfall_finalize_state(
    stepped_state,
    final_inner,
    final_defect,
    defect_init,
    current_h,
    h_new,
    g_best_new,
    stall_new,
    j_scalar,
):
    batch_size = stepped_state.V.shape[0]
    committed_shifted_reward = _rm_isp_shifted_reward_pure(final_inner, final_defect, current_h)
    return stepped_state._replace(
        V=final_inner,
        r_k=final_defect,
        bar_r=committed_shifted_reward,
        V_checkpoint=final_inner,
        steps_since_check=jnp.zeros_like(stepped_state.steps_since_check),
        working_V=final_inner,
        waterfall_h=h_new,
        waterfall_g_best=g_best_new,
        waterfall_last_status=stall_new.astype(jnp.int32),
        waterfall_reward_abs_max=jnp.max(jnp.abs(final_defect), axis=(1, 2)),
        waterfall_inner_depth=jnp.zeros((batch_size,), dtype=jnp.int32),
        defect_r=final_defect,
        defect_init=defect_init,
        replay_sample_index=advance_replay_sample_index(
            stepped_state.replay_sample_index,
            stepped_state.replay_states,
            delta=j_scalar,
        ),
    )


@jax.jit
def nr1_staircase_waterfall_rm_isp_dyna_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """One adaptive Waterfall-NR1 outer step using the staircase h-cascade rule."""
    normalized_state = _nr_rm_isp_normalize_state(state)
    current_h = jnp.asarray(normalized_state.waterfall_h, dtype=jnp.float64).reshape(-1)
    j_scalar = jnp.asarray(params.J_init, dtype=jnp.int32).reshape(-1)[0]
    stepped, final_inner, final_defect, anchor_V, defect_init = _nr1_waterfall_run_inner_j_samples(
        normalized_state, mdp, params, k, current_h, j_scalar,
    )
    g_new = jnp.max(jnp.abs(final_inner - anchor_V), axis=1)
    g_best_prev = jnp.asarray(normalized_state.waterfall_g_best, dtype=jnp.float64).reshape(-1)
    stall_prev = jnp.asarray(normalized_state.waterfall_last_status, dtype=jnp.int32).reshape(-1)

    first_step = g_best_prev >= (_WATERFALL_NR1_G_BEST_SENTINEL - 1.0)
    improved = g_new < g_best_prev
    reset_best_or_improved = improved | first_step
    g_best = jnp.where(reset_best_or_improved, g_new, g_best_prev)
    stall = jnp.where(reset_best_or_improved, 0, stall_prev + 1)

    sp = jnp.asarray(params.waterfall_stall_patience, dtype=jnp.int32)
    df = jnp.asarray(params.waterfall_decay_factor, dtype=jnp.float64)
    hm = jnp.asarray(params.waterfall_h_min, dtype=jnp.float64).reshape(-1)
    dm = jnp.asarray(params.waterfall_divergence_mult, dtype=jnp.float64)
    vm = jnp.asarray(params.waterfall_v_max, dtype=jnp.float64).reshape(-1)

    envelope_violate = jnp.max(jnp.abs(final_inner), axis=1) > dm * vm
    cascade_stall = stall >= sp
    cascade = cascade_stall | envelope_violate

    h_new = jnp.where(cascade, jnp.maximum(hm, current_h * df), current_h)
    stall_final = jnp.where(cascade, 0, stall)
    g_best_final = jnp.where(
        cascade,
        jnp.full_like(g_best, _WATERFALL_NR1_G_BEST_SENTINEL),
        g_best,
    )

    updated = _nr1_waterfall_finalize_state(
        stepped, final_inner, final_defect, defect_init,
        current_h, h_new, g_best_final, stall_final, j_scalar,
    )
    return updated, jnp.mean(updated.V[:, 0])


@jax.jit
def nr1_geo_waterfall_rm_isp_dyna_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int,
) -> Tuple[AgentState, jax.Array]:
    """One adaptive Waterfall-NR1 outer step with pure geometric h decay."""
    normalized_state = _nr_rm_isp_normalize_state(state)
    current_h = jnp.asarray(normalized_state.waterfall_h, dtype=jnp.float64).reshape(-1)
    j_scalar = jnp.asarray(params.J_init, dtype=jnp.int32).reshape(-1)[0]
    stepped, final_inner, final_defect, anchor_V, defect_init = _nr1_waterfall_run_inner_j_samples(
        normalized_state, mdp, params, k, current_h, j_scalar,
    )
    al = jnp.asarray(params.waterfall_alpha, dtype=jnp.float64)
    hm = jnp.asarray(params.waterfall_h_min, dtype=jnp.float64).reshape(-1)
    h_new = jnp.maximum(hm, current_h * al)

    g_best_prev = jnp.asarray(normalized_state.waterfall_g_best, dtype=jnp.float64).reshape(-1)
    stall_prev = jnp.asarray(normalized_state.waterfall_last_status, dtype=jnp.int32).reshape(-1)
    updated = _nr1_waterfall_finalize_state(
        stepped, final_inner, final_defect, defect_init,
        current_h, h_new, g_best_prev, stall_prev, j_scalar,
    )
    return updated, jnp.mean(updated.V[:, 0])


@jax.jit
def run_n_steps_nr1_staircase_waterfall_rm_isp_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = nr1_staircase_waterfall_rm_isp_dyna_step_pure(bst, mdp, params, curr_k)
        return (nst, val)

    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


@jax.jit
def run_n_steps_nr1_geo_waterfall_rm_isp_dyna_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    start_k: int,
    n_steps: int,
) -> Tuple[AgentState, jnp.ndarray]:
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = nr1_geo_waterfall_rm_isp_dyna_step_pure(bst, mdp, params, curr_k)
        return (nst, val)

    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


class _JaxBatchedNR1WaterfallBase:
    """Shared scaffolding for Waterfall-NR1 classes (Staircase, Geo)."""

    _SCHEDULE_KIND: int = -1

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        *,
        J,
        h_init,
        h_min,
        lr_params=(0.5, 0),
        lr_schedule="constant_delay",
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
        seed=0,
        sampling_distribution="uniform_sa",
        lambda_val=0.0,
        divergence_mult=2.5,
        v_max=0.0,
        decay_factor=0.5,
        stall_patience=0,
        alpha=1.0,
        trajectory_mode=False,
    ):
        del trajectory_mode
        if str(sampling_distribution).lower() != "uniform_sa":
            raise ValueError(
                "NR1 Waterfall variants currently support only sampling_distribution='uniform_sa'."
            )
        if J is None:
            raise ValueError("NR1 Waterfall variants require J >= 1.")
        j_min_value = int(jnp.min(jnp.asarray(J).reshape(-1)))
        if j_min_value < 1:
            raise ValueError("NR1 Waterfall variants require J >= 1.")
        if h_init is None:
            raise ValueError("NR1 Waterfall variants require h_init > 0.")
        h_init_min_value = float(jnp.min(jnp.asarray(h_init).reshape(-1)))
        if h_init_min_value <= 0.0:
            raise ValueError("NR1 Waterfall variants require h_init > 0.")
        if h_min is None:
            raise ValueError("NR1 Waterfall variants require h_min > 0.")
        h_min_min_value = float(jnp.min(jnp.asarray(h_min).reshape(-1)))
        if h_min_min_value <= 0.0:
            raise ValueError("NR1 Waterfall variants require h_min > 0.")
        h_min_max_value = float(jnp.max(jnp.asarray(h_min).reshape(-1)))
        h_init_max_value = float(jnp.max(jnp.asarray(h_init).reshape(-1)))
        if h_min_max_value > h_init_max_value:
            raise ValueError("NR1 Waterfall variants require h_min <= h_init.")

        self.num_envs = num_envs
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        self.mdp_data = MDPData(P=self.mdp_P, R=self.mdp_R, gamma=gamma)
        j_scalar_for_init = int(jnp.max(jnp.asarray(J).reshape(-1)))
        self.mdp_data, self.state = create_initial_state(
            self.mdp_P,
            self.mdp_R,
            num_envs,
            0.0,
            J_init=j_scalar_for_init,
            seed=seed,
        )

        start_lr, delay = lr_params
        lr_start = as_column_parameter(start_lr, num_envs, dtype=jnp.float64)
        lr_delay = as_column_parameter(delay, num_envs, dtype=jnp.float64)
        sched_int = 0 if str(lr_schedule).lower() in {"constant_delay", "constantanddelay"} else 1
        gam_arr = as_vector_parameter(gamma, num_envs, dtype=jnp.float64)
        lam_arr = as_vector_parameter(lambda_val, num_envs, dtype=jnp.float64)
        # Defect is always zero-initialized; no oracle reward prior is permitted.
        initial_defect = jnp.zeros_like(self.state.R_hat)
        h_arr = as_vector_parameter(h_init, num_envs, dtype=jnp.float64)
        j_arr = as_vector_parameter(J, num_envs, dtype=jnp.int32)
        h_min_arr = as_vector_parameter(h_min, num_envs, dtype=jnp.float64)
        v_max_arr = as_vector_parameter(v_max, num_envs, dtype=jnp.float64)

        disc = (h_arr * gam_arr) / (1.0 + h_arr)
        shifted_reward = _rm_isp_shifted_reward_pure(self.state.V, initial_defect, h_arr)
        initial_value = surrogate_vi_solve(
            self.state.P_hat,
            shifted_reward,
            self.state.V,
            disc.reshape(-1, 1, 1),
            tol=vi_tol,
            max_iters=vi_max_iters,
        )
        self.state = self.state._replace(
            V=initial_value,
            r_k=initial_defect,
            bar_r=shifted_reward,
            V_checkpoint=initial_value,
            working_V=initial_value,
            defect_r=initial_defect,
            defect_init=initial_defect,
            waterfall_h=h_arr,
            waterfall_g_best=jnp.full(
                (num_envs,), _WATERFALL_NR1_G_BEST_SENTINEL, dtype=jnp.float64,
            ),
            waterfall_last_status=jnp.zeros((num_envs,), dtype=jnp.int32),
            waterfall_inner_depth=jnp.zeros((num_envs,), dtype=jnp.int32),
            waterfall_reward_abs_max=jnp.max(jnp.abs(initial_defect), axis=(1, 2)),
        )

        self.params = AlgorithmParams(
            gamma=gam_arr,
            lambda_val=lam_arr,
            h=h_arr,
            J_init=j_arr,
            surrogate_tol=vi_tol,
            surrogate_iters=vi_max_iters,
            max_J=int(jnp.max(j_arr)),
            n_s=self.n_s,
            n_a=self.n_a,
            disc=disc,
            inv_1ph=1.0 / (1.0 + h_arr),
            h_inv_1ph=h_arr / (1.0 + h_arr),
            lr_start=lr_start,
            lr_delay=lr_delay,
            lr_schedule=sched_int,
            planning_steps=-1,
            use_vi_solve=True,
            batch_size=jnp.ones((num_envs,), dtype=jnp.int32),
            sampling_distribution=0,
            base_h=h_min_arr,
            h_init=h_arr,
            h_terminal=h_min_arr,
            h_schedule_horizon=0,
            h_schedule_shape=0,
            h_schedule_finish_frac=1.0,
            nr_mode=1,
            waterfall_h_init=h_arr,
            waterfall_h_min=h_min_arr,
            waterfall_decay_factor=float(decay_factor),
            waterfall_stall_patience=int(stall_patience),
            waterfall_alpha=float(alpha),
            waterfall_divergence_mult=float(divergence_mult),
            waterfall_v_max=v_max_arr,
            waterfall_schedule_kind=int(self._SCHEDULE_KIND),
        )

    def get_policy(self) -> jnp.ndarray:
        defect_r = self.state.defect_r if self.state.defect_r is not None else self.state.r_k
        current_h = self.state.waterfall_h if self.state.waterfall_h is not None else self.params.h
        return _compute_policy_nr_rm_isp_dispatch(
            defect_r,
            self.state.P_hat,
            self.state.V,
            self.params.gamma,
            current_h,
            self.params.nr_mode,
        )


class JaxBatchedNR1StaircaseWaterfallRMISPDyna(_JaxBatchedNR1WaterfallBase):
    """Waterfall-NR1 variant with adaptive staircase h cascade."""

    _SCHEDULE_KIND = _WATERFALL_NR1_SCHEDULE_STAIRCASE

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        *,
        J,
        h_init,
        h_min,
        stall_patience,
        decay_factor,
        divergence_mult=2.5,
        v_max=0.0,
        lambda_val=0.0,
        lr_params=(0.5, 0),
        lr_schedule="constant_delay",
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
        seed=0,
        sampling_distribution="uniform_sa",
        trajectory_mode=False,
    ):
        if stall_patience is None or int(stall_patience) < 1:
            raise ValueError("Staircase Waterfall-NR1 requires stall_patience >= 1.")
        if decay_factor is None or not (0.0 < float(decay_factor) < 1.0):
            raise ValueError("Staircase Waterfall-NR1 requires 0 < decay_factor < 1.")
        super().__init__(
            mdp_P=mdp_P,
            mdp_R=mdp_R,
            gamma=gamma,
            num_envs=num_envs,
            J=J,
            h_init=h_init,
            h_min=h_min,
            lr_params=lr_params,
            lr_schedule=lr_schedule,
            vi_tol=vi_tol,
            vi_max_iters=vi_max_iters,
            seed=seed,
            sampling_distribution=sampling_distribution,
            lambda_val=lambda_val,
            divergence_mult=divergence_mult,
            v_max=v_max,
            decay_factor=decay_factor,
            stall_patience=stall_patience,
            alpha=1.0,
            trajectory_mode=trajectory_mode,
        )

    def step(self, k: int) -> float:
        self.state, mean_val = nr1_staircase_waterfall_rm_isp_dyna_step_pure(
            self.state, self.mdp_data, self.params, k,
        )
        return float(mean_val)

    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_nr1_staircase_waterfall_rm_isp_dyna_pure(
            self.state, self.mdp_data, self.params, start_k, n_steps,
        )
        return float(mean_val)


class JaxBatchedNR1GeoWaterfallRMISPDyna(_JaxBatchedNR1WaterfallBase):
    """Waterfall-NR1 variant with pure geometric h decay."""

    _SCHEDULE_KIND = _WATERFALL_NR1_SCHEDULE_GEO

    def __init__(
        self,
        mdp_P,
        mdp_R,
        gamma,
        num_envs,
        *,
        J,
        h_init,
        h_min,
        alpha,
        lambda_val=0.0,
        lr_params=(0.5, 0),
        lr_schedule="constant_delay",
        vi_tol=1e-6,
        vi_max_iters=PI_DEFAULT_ITERS,
        seed=0,
        sampling_distribution="uniform_sa",
        trajectory_mode=False,
    ):
        if alpha is None or not (0.0 < float(alpha) <= 1.0):
            raise ValueError("Geo Waterfall-NR1 requires 0 < alpha <= 1.")
        super().__init__(
            mdp_P=mdp_P,
            mdp_R=mdp_R,
            gamma=gamma,
            num_envs=num_envs,
            J=J,
            h_init=h_init,
            h_min=h_min,
            lr_params=lr_params,
            lr_schedule=lr_schedule,
            vi_tol=vi_tol,
            vi_max_iters=vi_max_iters,
            seed=seed,
            sampling_distribution=sampling_distribution,
            lambda_val=lambda_val,
            divergence_mult=2.5,
            v_max=0.0,
            decay_factor=0.5,
            stall_patience=0,
            alpha=alpha,
            trajectory_mode=trajectory_mode,
        )

    def step(self, k: int) -> float:
        self.state, mean_val = nr1_geo_waterfall_rm_isp_dyna_step_pure(
            self.state, self.mdp_data, self.params, k,
        )
        return float(mean_val)

    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_nr1_geo_waterfall_rm_isp_dyna_pure(
            self.state, self.mdp_data, self.params, start_k, n_steps,
        )
        return float(mean_val)
