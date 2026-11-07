"""
JAX-accelerated Q-Learning Implementation.

This module contains the pure JAX implementation of Q-Learning for batched
parallel environments.

"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import Tuple

from .types import AgentState, MDPData, AlgorithmParams
from .base import (
    advance_replay_sample_index,
    as_column_parameter,
    create_initial_state,
    sample_step_pure,
    compute_lr_pure,
    _compute_policy_from_q,
)


# =============================================================================
# Q-Learning Step Function
# =============================================================================

@jax.jit
def q_learning_step_pure(
    state: AgentState,
    mdp: MDPData,
    params: AlgorithmParams,
    k: int
) -> Tuple[AgentState, jax.Array]:
    """Pure JAX step for Q-Learning."""
    
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
    
    # 2. LR
    lr = compute_lr_pure(k, params.lr_start, params.lr_delay, params.lr_schedule)
    lr = lr.reshape(-1)
    
    # 3. Q Update
    b_idx = jnp.arange(state.V.shape[0])
    next_vals = jnp.max(state.Q[b_idx, ns], axis=1).reshape(-1)
    rng_vec = r.reshape(-1)
    
    gamma_vec = jnp.asarray(params.gamma, dtype=jnp.float64).reshape(-1)
    target = rng_vec + gamma_vec * next_vals
    diff = target - state.Q[b_idx, s, a]
    
    Q = state.Q.at[b_idx, s, a].add(lr * diff)
    V_final = jnp.max(Q, axis=2)
    
    new_state = AgentState(
        V=V_final,
        # Pytree-required passthroughs (Q-Learning is model-free).
        P_hat=state.P_hat,
        R_hat=state.R_hat,
        N_sas=state.N_sas,
        N_sa=state.N_sa,
        R_sum=state.R_sum,
        states=new_states,
        J_per_env=state.J_per_env,
        prev_score=state.prev_score,
        rng_key=rng_key,
        r_k=state.r_k,
        Q=Q,
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
def run_n_steps_q_pure(state, mdp, params, start_k, n_steps):
    """Run N steps of Q-Learning in a single JIT-compiled loop."""
    def body_fn(i, carry):
        bst, _ = carry
        curr_k = start_k + i
        nst, val = q_learning_step_pure(bst, mdp, params, curr_k)
        return (nst, val)
    init_metric = jnp.zeros((), dtype=state.V.dtype)
    return lax.fori_loop(0, n_steps, body_fn, (state, init_metric))


# =============================================================================
# Q-Learning Class
# =============================================================================

class JaxBatchedQLearning:
    """
    JAX implementation of Batched Q-Learning.
    
    Uses pure JAX functions for JIT compilation and GPU acceleration.
    """
    
    def __init__(self, mdp_P, mdp_R, gamma, num_envs,
                 lr_params=(0.02, 68000), lr_schedule="constant_delay",
                 seed=0, trajectory_mode=False):
        
        self.num_envs = num_envs
        self.gamma = gamma
        
        self.mdp_P = jnp.asarray(mdp_P, dtype=jnp.float64)
        self.mdp_R = jnp.asarray(mdp_R, dtype=jnp.float64)
        self.n_s, self.n_a, _ = self.mdp_P.shape
        
        self.mdp_data = MDPData(P=self.mdp_P, R=self.mdp_R, gamma=gamma)
        self.mdp_data, self.state = create_initial_state(
            self.mdp_P, self.mdp_R, num_envs, 0.0, seed=seed
        )
        
        start_lr, delay = lr_params
        start_lr = as_column_parameter(start_lr, num_envs, dtype=jnp.float64)
        delay = as_column_parameter(delay, num_envs, dtype=jnp.float64)
        sched_int = 0 if lr_schedule == "constant_delay" else 1
        
        gam_arr = as_column_parameter(gamma, num_envs, dtype=jnp.float64)
        
        self.params = AlgorithmParams(
            gamma=gam_arr,
            lambda_val=0.0,
            h=1.0, J_init=0, 
            surrogate_tol=0.0, surrogate_iters=0,
            max_J=0,
            n_s=self.n_s, n_a=self.n_a,
            disc=gam_arr.reshape(-1, 1, 1),
            inv_1ph=0.0, h_inv_1ph=0.0,
            lr_start=start_lr, lr_delay=delay, lr_schedule=sched_int
        )
        
    def step(self, k: int) -> float:
        self.state, mean_val = q_learning_step_pure(self.state, self.mdp_data, self.params, k)
        return float(mean_val)

    def get_policy(self) -> jnp.ndarray:
        """Return the current checkpoint policy π_k as the greedy policy of Q_k."""
        return _compute_policy_from_q(self.state.Q)
        
    def run_steps(self, start_k: int, n_steps: int) -> float:
        self.state, mean_val = run_n_steps_q_pure(self.state, self.mdp_data, self.params, start_k, n_steps)
        return float(mean_val)
