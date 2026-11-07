"""OS-Dyna, IE-Dyna with a fixed h, Q-learning and plain Dyna under corruption that varies by state.

The tape, model rows, surrogate solver, reward shift and policy extraction are defined in environment.
Each run has its own weight lam_x and prior mode per state (0 support-uniform, 1 self-loop, 2 the env's fixed PRIOR kernel).
With a uniform support-uniform corruption, OS and IE (h fixed at 1e6) must match environment.run_group.
"""
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import environment as L
from environment import (EVAL_EVERY, GAMMA, N_A, N_S, P_TRUE, PI_STEPS, PRIOR, R_TRUE, _compute_model_error_noisy_row_pure,
                      _compute_policy_os_dyna, _compute_policy_rm_isp_dyna, _rm_isp_shifted_reward_pure, learning_rate,
                      start_value, surrogate_vi_solve)

METHODS = ("os", "ie", "q", "dyna")
# name: (kind, a, b) in environment.learning_rate; constants use kind 0 with an unreachable delay.
SCHEDULES = {**{f"c{c:g}": (0, c, 1e12) for c in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)},
             "0.2/(1+0.02n)": (1, 0.2, 0.98), "0.5/(1+0.005n)": (1, 0.5, 0.995), "n^-0.6": (2, 0.6, 0.0), "10/(10+n)": (3, 10.0, 0.0)}


def hazard_mask():
    """Non-absorbing states that can enter a negative-reward absorbing state under some action."""
    P, R = np.asarray(P_TRUE), np.asarray(R_TRUE)
    absorbing = np.array([np.allclose(P[x, :, x], 1.0) for x in range(N_S)])
    hazard = absorbing & (R.max(1) < 0)
    return ~absorbing & (P[:, :, hazard].sum(2).max(1) > 0)


def corruption(family, lam):
    """(lam_x, mode_x), each of shape (N_S,)."""
    if family == "su":
        return np.full(N_S, lam), np.zeros(N_S, int)
    if family == "sl":
        return np.full(N_S, lam), np.ones(N_S, int)
    if family == "sticky":
        m = hazard_mask()
        return np.where(m, lam, 0.1), m.astype(int)
    if family == "prior":  # mode 2: the env's fixed PRIOR kernel (npz envs only)
        if PRIOR is None:
            raise ValueError("family 'prior' needs DYNA_ENV=npz:<name> (no PRIOR kernel in this env)")
        return np.full(N_S, lam), np.full(N_S, 2, int)
    raise ValueError(family)


class Pm(NamedTuple):
    lamx: jax.Array
    modex: jax.Array
    kind: jax.Array
    a: jax.Array
    b: jax.Array
    visit: jax.Array
    row: jax.Array
    ts: jax.Array
    ta: jax.Array
    tn: jax.Array


class St(NamedTuple):
    V: jax.Array
    D: jax.Array  # the defect tracker, or Q for Q-learning
    P: jax.Array
    Nsas: jax.Array
    Nsa: jax.Array
    h: jax.Array
    t: jax.Array


def absorb_batch(B, U, st, p):
    """B samples one at a time with U held fixed, as environment.absorb_batch but with per-state corruption."""
    idx = jnp.arange(U.shape[0])

    def one(i, c):
        D, Pc, Nsas, Nsa = c
        t = st.t + i
        s, a, ns = p.ts[p.row, t], p.ta[p.row, t], p.tn[p.row, t]
        r = R_TRUE[s, a]
        Nsas = Nsas.at[idx, s, a, ns].add(1.0)
        Nsa = Nsa.at[idx, s, a].add(1.0)
        row_emp, _, row_p = _compute_model_error_noisy_row_pure(row_counts=Nsas[idx, s, a], row_states=s, n_s=N_S,
                                                                lambda_val=p.lamx[idx, s], model_error_mode=p.modex[idx, s])
        if PRIOR is not None:  # mode 2: (1-lam) * empirical + lam * PRIOR[s, a]; modes 0/1 keep row_p untouched
            lam_ = p.lamx[idx, s][:, None]
            row_p = jnp.where((p.modex[idx, s] == 2)[:, None], (1.0 - lam_) * row_emp + lam_ * PRIOR[s, a], row_p)
        Pc = Pc.at[idx, s, a].set(row_p)
        target = r + GAMMA * U[idx, ns] - GAMMA * jnp.sum(row_p * U, axis=1)
        index = jnp.where(p.visit, Nsa[idx, s, a] - 1.0, t.astype(jnp.float64))
        return D.at[idx, s, a].add(learning_rate(index, p) * (target - D[idx, s, a])), Pc, Nsas, Nsa

    D, Pc, Nsas, Nsa = lax.fori_loop(0, B, one, (st.D, st.P, st.Nsas, st.Nsa))
    return st._replace(D=D, P=Pc, Nsas=Nsas, Nsa=Nsa, t=st.t + B)


def q_batch(B, st, p):
    idx = jnp.arange(st.D.shape[0])

    def one(i, c):
        Q, Nsa = c
        t = st.t + i
        s, a, ns = p.ts[p.row, t], p.ta[p.row, t], p.tn[p.row, t]
        Nsa = Nsa.at[idx, s, a].add(1.0)
        target = R_TRUE[s, a] + GAMMA * Q[idx, ns].max(axis=1)
        index = jnp.where(p.visit, Nsa[idx, s, a] - 1.0, t.astype(jnp.float64))
        return Q.at[idx, s, a].add(learning_rate(index, p) * (target - Q[idx, s, a])), Nsa

    Q, Nsa = lax.fori_loop(0, B, one, (st.D, st.Nsa))
    return st._replace(D=Q, Nsa=Nsa, t=st.t + B)


def disc(st, g):
    return jnp.full((st.V.shape[0], 1, 1), g)


def os_outer(B, st, p):
    st = absorb_batch(B, st.V, st, p)
    return st._replace(V=surrogate_vi_solve(st.P, st.D, st.V, disc(st, GAMMA), max_iters=PI_STEPS))


def ie_outer(J, B, st, p):
    """J inner rounds against the anchor V_k with h fixed for the whole run."""
    anchor = st.V
    d = (st.h * GAMMA / (1.0 + st.h)).reshape(-1, 1, 1)

    def inner(_, c):
        U, s_ = c
        s_ = absorb_batch(B, U, s_, p)
        return surrogate_vi_solve(s_.P, _rm_isp_shifted_reward_pure(anchor, s_.D, st.h), U, d, max_iters=PI_STEPS), s_

    U, s_ = lax.fori_loop(0, J, inner, (anchor, st))
    return s_._replace(V=U)


def dyna_outer(B, st, p):
    """Plain Dyna: the model is learned the same way, then planned with the true reward and no correction."""
    st = absorb_batch(B, st.V, st, p)
    R = jnp.broadcast_to(R_TRUE, st.D.shape)
    return st._replace(V=surrogate_vi_solve(st.P, R, st.V, disc(st, GAMMA), max_iters=PI_STEPS))


@lru_cache(maxsize=None)
def chunk_fn(method, J, B):
    step = {"os": lambda st, p: os_outer(B, st, p), "ie": lambda st, p: ie_outer(J, B, st, p),
            "q": lambda st, p: q_batch(B, st, p), "dyna": lambda st, p: dyna_outer(B, st, p)}[method]

    @jax.jit
    def run(st, p, n):
        return lax.fori_loop(0, n, lambda _, c: step(c, p), st)

    return run


def policy(method, st):
    E = st.V.shape[0]
    if method == "os":
        return _compute_policy_os_dyna(st.D, st.P, st.V, jnp.full((E,), GAMMA))
    if method == "ie":
        return _compute_policy_rm_isp_dyna(st.D, st.P, st.V, jnp.full((E,), GAMMA), st.h)
    if method == "dyna":
        return _compute_policy_os_dyna(jnp.broadcast_to(R_TRUE, st.D.shape), st.P, st.V, jnp.full((E,), GAMMA))
    return jnp.argmax(st.D, axis=2)


def initial_model(lamx, modex):
    """As initial_model_estimate_pure, per state: uniform rows blended toward uniform (mode 0), staying put (mode 1) or PRIOR[s, a] (mode 2)."""
    uniform = np.full((N_S, N_S), 1.0 / N_S)
    prior = np.where(modex[..., None] == 1, np.eye(N_S)[None], uniform[None])  # (E, S, S)
    rows = (1.0 - lamx[..., None]) * uniform[None] + lamx[..., None] * prior
    full = np.broadcast_to(rows[:, :, None, :], (lamx.shape[0], N_S, N_A, N_S)).copy()
    if (modex == 2).any():  # mode 2: the unvisited empirical row is uniform, the prior is the env's PRIOR[s, a]
        fixed = (1.0 - lamx[:, :, None, None]) / N_S + lamx[:, :, None, None] * np.asarray(PRIOR)[None]
        full = np.where((modex == 2)[:, :, None, None], fixed, full)
    return jnp.asarray(full)


def run(method, J, B, lamx, modex, kinds, a, b, visit, h, row, tape, budget, every=EVAL_EVERY):
    """Arrays have one entry (lamx, modex: one row) per run. Returns (samples used, start-state returns (evals, E))."""
    E = len(row)
    ts, ta, tn = (jnp.asarray(x, dtype=jnp.int32) for x in tape[:3])
    lamx, modex = np.asarray(lamx, dtype=float).reshape(E, N_S), np.asarray(modex, dtype=int).reshape(E, N_S)
    p = Pm(lamx=jnp.asarray(lamx), modex=jnp.asarray(modex, dtype=jnp.int32), kind=jnp.asarray(kinds, dtype=jnp.int32),
           a=jnp.asarray(a, dtype=jnp.float64), b=jnp.asarray(b, dtype=jnp.float64), visit=jnp.asarray(visit, dtype=bool),
           row=jnp.asarray(row, dtype=jnp.int32), ts=ts, ta=ta, tn=tn)
    st = St(V=jnp.zeros((E, N_S)), D=jnp.zeros((E, N_S, N_A)), P=initial_model(lamx, modex),
            Nsas=jnp.zeros((E, N_S, N_A, N_S)), Nsa=jnp.zeros((E, N_S, N_A)), h=jnp.asarray(h, dtype=jnp.float64).reshape(E),
            t=jnp.asarray(0, dtype=jnp.int32))
    fn = chunk_fn(method, int(J), int(B))
    per_outer = (int(J) if method == "ie" else 1) * int(B)
    used, values, done, last = [], [], 0, None
    for x in range(0, budget + 1, every):
        outer = x // per_outer
        if outer > done:
            st = fn(st, p, outer - done)
            done, last = outer, None
        if last is None:
            last = np.asarray(start_value(policy(method, st)))
        used.append(done * per_outer)
        values.append(last)
    return np.asarray(used), np.stack(values)
