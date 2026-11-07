"""learning.run with multi-metric recording for the matched-norm study.

The dynamics are learning's own (same chunk_fn, policy, initial_model, Pm/St); only the recording differs. At each evaluation
run() records, per run e:
  Vpi[e, :]  the greedy policy's true value at every state (environment.all_values, the function learning scores through start_value)
  Vest[e, :] the method's own value estimate V_k (st.V for os/ie/dyna; max_a Q for q)
  pi[e, :]   the greedy policy itself
so that any norm / scope can be computed offline. cells() turns these into the specified metric cells.
"""
import numpy as np
import jax.numpy as jnp

import environment as L
import learning
from learning import Pm, St, chunk_fn, initial_model, policy
from environment import EVAL_EVERY, GAMMA, N_A, N_S, P_TRUE


def estimate(method, st):
    return jnp.max(st.D, axis=2) if method == "q" else st.V


def run(method, J, B, lamx, modex, kinds, a, b, visit, h, row, tape, budget, every=EVAL_EVERY):
    """As learning.run. Returns (used (evals,), Vpi (evals, E, S), Vest (evals, E, S), pi (evals, E, S) int16)."""
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
    used, vpi, vest, pis, done, last = [], [], [], [], 0, None
    for x in range(0, budget + 1, every):
        outer = x // per_outer
        if outer > done:
            st = fn(st, p, outer - done)
            done, last = outer, None
        if last is None:
            pi = policy(method, st)
            last = (np.asarray(L.all_values(pi)), np.asarray(estimate(method, st)), np.asarray(pi, dtype=np.int16))
        used.append(done * per_outer)
        vpi.append(last[0])
        vest.append(last[1])
        pis.append(last[2])
    return np.asarray(used), np.stack(vpi), np.stack(vest), np.stack(pis)


# ---------------------------------------------------------------- metric cells
def occupancy(pi, x0=0):
    """Normalised start-state discounted occupancy d = (1-g) e_x0^T (I - g P_pi)^-1 of a fixed policy pi (S,)."""
    P = np.asarray(P_TRUE)
    Ppi = P[np.arange(N_S), pi]
    d = np.linalg.solve((np.eye(N_S) - GAMMA * Ppi).T, np.eye(N_S)[x0]) * (1 - GAMMA)
    return np.maximum(d, 0.0)


def optimal_policy():
    """(V*, greedy pi*). environment.optimal_values first (used for cliffwalk and maze); if its policy iteration cycles between
    tied policies, policy iteration that switches an action only on a strict improvement (> 1e-9 relative)."""
    P, R = np.asarray(P_TRUE), np.asarray(L.R_TRUE)
    try:
        V = L.optimal_values()
    except RuntimeError:
        idx, pi = np.arange(N_S), np.zeros(N_S, int)
        for _ in range(2000):
            V = np.linalg.solve(np.eye(N_S) - GAMMA * P[idx, pi], R[idx, pi])
            Q = R + GAMMA * P @ V
            new = Q.argmax(1)
            new = np.where(Q[idx, new] > Q[idx, pi] + 1e-9 * max(1.0, np.abs(V).max()), new, pi)
            if (new == pi).all():
                break
            pi = new
        else:
            raise RuntimeError("policy iteration did not converge")
    return V, (R + GAMMA * P @ V).argmax(1)


SCOPES = ("start", "all", "occ")  # all = uniform over states = the sampling distribution (uniform over pairs, equal A per state)
NORMS = ("L1", "L2", "Linf")
OBJECTS = ("pol", "est")


def cell_names():
    out = []
    for o in OBJECTS:
        out.append(f"{o}|start")
        out += [f"{o}|{s}|{n}" for s in ("all", "occ") for n in NORMS]
        out.append(f"{o}|all|span")
    return out


def cells(Vpi, Vest, Vstar, d, x0=0):
    """Relative errors in percent. Vpi, Vest: (..., S). d: occupancy weights (S,) (of pi*). Returns (..., n_cells) float32
    in the order of cell_names(). Each cell is ||X - V*||_{norm, scope} / ||V*||_{norm, scope}; the start cell uses
    max(1, |V*(x0)|) as the panel does."""
    out = []
    sup = d > 1e-12
    for X in (Vpi, Vest):
        e = X - Vstar
        out.append(np.abs(e[..., x0]) / max(1.0, abs(Vstar[x0])))
        a = np.abs(e)
        out.append(a.sum(-1) / np.abs(Vstar).sum())
        out.append(np.sqrt((e ** 2).sum(-1)) / np.sqrt((Vstar ** 2).sum()))
        out.append(a.max(-1) / np.abs(Vstar).max())
        out.append((a * d).sum(-1) / (np.abs(Vstar) * d).sum())
        out.append(np.sqrt((e ** 2 * d).sum(-1)) / np.sqrt((Vstar ** 2 * d).sum()))
        out.append(a[..., sup].max(-1) / np.abs(Vstar[sup]).max())
        out.append((e.max(-1) - e.min(-1)) / (Vstar.max() - Vstar.min()))
    return (100.0 * np.stack(out, -1)).astype(np.float32)
