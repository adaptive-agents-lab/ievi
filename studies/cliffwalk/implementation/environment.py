"""OS-Dyna and IE-Dyna with a learning-rate index choice, J and a batch size.

Sampling, model rows, the surrogate solver, the reward shift and policy extraction use the
reference implementation. With B=1 and the global index this reproduces its update convention.
"""
import os

os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import sys
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SRC = str(Path(__file__).resolve().parents[3] / "methods/planning/implementation/src")
sys.path.insert(0, SRC)

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from besp.algorithms.jax.base import initial_model_estimate_pure, surrogate_vi_solve
from besp.algorithms.jax.dyna import (_compute_model_error_noisy_row_pure, _compute_policy_os_dyna,
                                      _compute_policy_rm_isp_dyna, _rm_isp_shifted_reward_pure)
from besp.domains.jax_rl.replay_tape import generate_uniform_sa_replay_tape
from besp.runners.factory import create_environment_jax

GAMMA = 0.99
TARGET = 819.1545470513453
PI_STEPS = 5  # the class default that the plotted runs use
H_INIT, DECAY, PATIENCE, KAPPA, V_MAX = 1e6, 0.1, 3, 2.5, 0.0
SENTINEL = 1e30
MODE_SUPPORT_UNIFORM = 0
BUDGET = 40000
EVAL_EVERY = 2000

# kind: 0 constant_delay, 1 rescaled_linear, 2 n^-a, 3 a/(a+n), with n the 1-based index.
SCHEDULES = {
    "const_0.02": (0, 0.02, 40000.0), "const_0.1": (0, 0.1, 160000.0), "const_0.15": (0, 0.15, 80000.0),
    "rl_0.2_0.98": (1, 0.2, 0.98), "rl_0.5_0.995": (1, 0.5, 0.995), "rl_1_0.995": (1, 1.0, 0.995),
    "rl_0.4_0.9995": (1, 0.4, 0.9995), "rl_0.5_0.9995": (1, 0.5, 0.9995), "rl_0.5_0.99999": (1, 0.5, 0.99999),
    "rl_0.8_0.9999": (1, 0.8, 0.9999), "rl_0.8_0.99995": (1, 0.8, 0.99995),
    "1/n": (2, 1.0, 0.0), "n^-0.6": (2, 0.6, 0.0), "n^-0.8": (2, 0.8, 0.0),
    "10/(10+n)": (3, 10.0, 0.0), "50/(50+n)": (3, 50.0, 0.0),
}

ENV = os.environ.get("DYNA_ENV", "cliffwalk")  # one domain per process, since the constants below are module-level
ENV_KWARGS = {"garnet": {"seed": 0}}
if ENV.startswith("npz:"):  # custom environment from envs/<name>.npz (see envs5.py); PRIOR is its fixed surrogate prior kernel
    _z = np.load(ROOT / "envs" / (ENV[4:] + ".npz"))
    P_TRUE, R_TRUE, PRIOR = (jnp.asarray(_z[k], dtype=jnp.float64) for k in ("P", "R", "PRIOR"))
else:
    _env = create_environment_jax(ENV, gamma=GAMMA, **ENV_KWARGS.get(ENV, {}))
    P_TRUE, R_TRUE = jnp.asarray(_env.P, dtype=jnp.float64), jnp.asarray(_env.R, dtype=jnp.float64)
    PRIOR = None
N_S, N_A = P_TRUE.shape[0], P_TRUE.shape[1]


def optimal_values():
    P, R = np.asarray(P_TRUE), np.asarray(R_TRUE)
    idx, pi = np.arange(N_S), np.zeros(N_S, int)
    for _ in range(500):
        V = np.linalg.solve(np.eye(N_S) - GAMMA * P[idx, pi], R[idx, pi])
        new = (R + GAMMA * P @ V).argmax(1)
        if (new == pi).all():
            return V
        pi = new
    raise RuntimeError("policy iteration did not converge")


def load_tape(seeds, steps=BUDGET):
    """Tape prefix for these seeds; the generator is per-seed, and 400k matches the retained length."""
    prefix = "" if ENV == "cliffwalk" else f"{ENV.replace(':', '_')}_"  # npz:<name> -> npz_<name>_
    cache = ROOT / "tapes" / (f"{prefix}tape_{seeds[0]}_{seeds[-1]}.npz" if steps == BUDGET else f"{prefix}tape_{seeds[0]}_{seeds[-1]}_{steps}.npz")
    if cache.exists():
        z = np.load(cache)
        return z["s"], z["a"], z["ns"], str(z["checksum"])
    tape = generate_uniform_sa_replay_tape(mdp_P=np.asarray(P_TRUE), mdp_R=np.asarray(R_TRUE), seed_values=tuple(seeds),
                                           tape_steps=max(400000, steps), env_name=ENV, env_params={"gamma": GAMMA}, gamma=GAMMA)
    s, a, ns = (np.asarray(x[:, :steps], dtype=np.int16) for x in (tape.states, tape.actions, tape.next_states))
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, s=s, a=a, ns=ns, checksum=tape.manifest.tape_checksum)
    return s, a, ns, tape.manifest.tape_checksum


class St(NamedTuple):
    V: jax.Array
    D: jax.Array
    P: jax.Array
    Nsas: jax.Array
    Nsa: jax.Array
    h: jax.Array
    gbest: jax.Array
    stall: jax.Array
    t: jax.Array


class Pm(NamedTuple):
    lam: jax.Array
    kind: jax.Array
    a: jax.Array
    b: jax.Array
    visit: jax.Array
    hmin: jax.Array
    row: jax.Array
    ts: jax.Array
    ta: jax.Array
    tn: jax.Array


def learning_rate(i, p):
    """i is the 0-based index, the convention of compute_lr_pure."""
    i = i.astype(jnp.float64)
    n = i + 1.0
    constant = p.a / jnp.where(i < p.b, 1.0, i - p.b + 1.0)
    rescaled = p.a / (1.0 + (1.0 - p.b) * n)
    power = n ** (-p.a)
    harmonic = p.a / (p.a + n)
    return jnp.select([p.kind == 0, p.kind == 1, p.kind == 2], [constant, rescaled, power], harmonic)


def absorb_batch(B, U, st, p):
    """Apply B samples one at a time with U held fixed. Each is one physical sample."""
    idx = jnp.arange(U.shape[0])

    def one(i, c):
        D, Pc, Nsas, Nsa = c
        t = st.t + i
        s, a, ns = p.ts[p.row, t], p.ta[p.row, t], p.tn[p.row, t]
        r = R_TRUE[s, a]
        Nsas = Nsas.at[idx, s, a, ns].add(1.0)
        Nsa = Nsa.at[idx, s, a].add(1.0)
        _, _, row_p = _compute_model_error_noisy_row_pure(row_counts=Nsas[idx, s, a], row_states=s, n_s=N_S,
                                                          lambda_val=p.lam, model_error_mode=MODE_SUPPORT_UNIFORM)
        Pc = Pc.at[idx, s, a].set(row_p)
        target = r + GAMMA * U[idx, ns] - GAMMA * jnp.sum(row_p * U, axis=1)
        index = jnp.where(p.visit, Nsa[idx, s, a] - 1.0, t.astype(jnp.float64))
        lr = learning_rate(index, p)
        return D.at[idx, s, a].add(lr * (target - D[idx, s, a])), Pc, Nsas, Nsa

    D, Pc, Nsas, Nsa = lax.fori_loop(0, B, one, (st.D, st.P, st.Nsas, st.Nsa))
    return st._replace(D=D, P=Pc, Nsas=Nsas, Nsa=Nsa, t=st.t + B)


def os_outer(B, st, p):
    st = absorb_batch(B, st.V, st, p)
    disc = jnp.full((st.V.shape[0], 1, 1), GAMMA)
    return st._replace(V=surrogate_vi_solve(st.P, st.D, st.V, disc, max_iters=PI_STEPS))


def ie_outer(J, B, st, p):
    anchor = st.V
    disc = (st.h * GAMMA / (1.0 + st.h)).reshape(-1, 1, 1)

    def inner(_, c):
        U, s_ = c
        s_ = absorb_batch(B, U, s_, p)
        shifted = _rm_isp_shifted_reward_pure(anchor, s_.D, st.h)
        return surrogate_vi_solve(s_.P, shifted, U, disc, max_iters=PI_STEPS), s_

    U, s_ = lax.fori_loop(0, J, inner, (anchor, st))
    g = jnp.max(jnp.abs(U - anchor), axis=1)
    first = st.gbest >= SENTINEL - 1.0
    better = (g < st.gbest) | first
    gbest = jnp.where(better, g, st.gbest)
    stall = jnp.where(better, 0, st.stall + 1)
    cascade = (stall >= PATIENCE) | (jnp.max(jnp.abs(U), axis=1) > KAPPA * V_MAX)
    h = jnp.where(cascade, jnp.maximum(p.hmin, st.h * DECAY), st.h)
    return s_._replace(V=U, h=h, gbest=jnp.where(cascade, SENTINEL, gbest), stall=jnp.where(cascade, 0, stall))


@lru_cache(maxsize=None)
def chunk_fn(method, J, B):
    step = (lambda st, p: os_outer(B, st, p)) if method == "os" else (lambda st, p: ie_outer(J, B, st, p))

    @jax.jit
    def run(st, p, n):
        return lax.fori_loop(0, n, lambda _, c: step(c, p), st)

    return run


@jax.jit
def all_values(pi):
    E = pi.shape[0]
    b, s = jnp.arange(E)[:, None], jnp.arange(N_S)[None, :]
    A = jnp.eye(N_S) - GAMMA * P_TRUE[s, pi]
    return jnp.linalg.solve(A, R_TRUE[s, pi][..., None])[:, :, 0]


def start_value(pi):
    return all_values(pi)[:, 0]


def run_group(method, J, B, lam, kinds, a, b, visit, hmin, row, tape, budget=BUDGET):
    """Run E settings at once. Returns (samples_used, start-state returns of shape (evals, E), final h)."""
    E = len(row)
    ts, ta, tn = (jnp.asarray(x, dtype=jnp.int32) for x in tape[:3])
    p = Pm(lam=jnp.asarray(lam, dtype=jnp.float64), kind=jnp.asarray(kinds, dtype=jnp.int32), a=jnp.asarray(a, dtype=jnp.float64),
           b=jnp.asarray(b, dtype=jnp.float64), visit=jnp.asarray(visit, dtype=bool), hmin=jnp.asarray(hmin, dtype=jnp.float64),
           row=jnp.asarray(row, dtype=jnp.int32), ts=ts, ta=ta, tn=tn)
    st = St(V=jnp.zeros((E, N_S)), D=jnp.zeros((E, N_S, N_A)),
            P=initial_model_estimate_pure(num_envs=E, n_s=N_S, n_a=N_A, lambda_val=p.lam, model_error_mode=MODE_SUPPORT_UNIFORM),
            Nsas=jnp.zeros((E, N_S, N_A, N_S)), Nsa=jnp.zeros((E, N_S, N_A)), h=jnp.full((E,), H_INIT),
            gbest=jnp.full((E,), SENTINEL), stall=jnp.zeros((E,), dtype=jnp.int32), t=jnp.asarray(0, dtype=jnp.int32))
    run = chunk_fn(method, int(J), int(B))
    per_outer = (1 if method == "os" else int(J)) * int(B)

    def policy(s):
        if method == "os":
            return _compute_policy_os_dyna(s.D, s.P, s.V, jnp.full((E,), GAMMA))
        return _compute_policy_rm_isp_dyna(s.D, s.P, s.V, jnp.full((E,), GAMMA), s.h)

    used, values, done, last = [], [], 0, None
    for x in range(0, budget + 1, EVAL_EVERY):
        outer = x // per_outer
        if outer > done:
            st = run(st, p, outer - done)
            done = outer
            last = None
        if last is None:
            last = np.asarray(start_value(policy(st)))
        used.append(done * per_outer)
        values.append(last)
    return np.asarray(used), np.stack(values), np.asarray(st.h)
