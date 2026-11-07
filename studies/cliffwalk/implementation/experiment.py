"""Tuning candidates and grouped execution for the Cliffwalk paper panel."""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTROOT = ROOT / "results"
TAPE_LEN = 400000
SEEDS = {"tune": list(range(20, 30)), "test": list(range(250, 310))}
# Areas use nominal evaluation checkpoints when CLIFFWALK_AXIS=nominal.
NOM = os.environ.get("CLIFFWALK_AXIS") == "nominal"
SFX = "" if NOM else "-actual"
ENVS = {"cw": ("cliffwalk", ("su",), (0.8,))}
HORIZONS = {}

def result_directory(env):
    return OUTROOT / ("cliffwalk" if env == "cw" else env)


# Learning-rate schedule grid. Encoding as
# environment.learning_rate: kind 0 constant (unreachable delay), 1 a/(1+(1-b)n), 2 n^-a, 3 a/(a+n). "v:" = index by the
# pair's own visit count (learning visit=True), "g:" and constants = global sample index.
CONST2 = {f"c{c:g}": (0, c, 1e12) for c in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)}
VISIT = {**{f"v:{c}/({c}+n)": (3, float(c), 0.0) for c in (1, 3, 10, 30)},
         **{f"v:n^-{p}": (2, p, 0.0) for p in (0.5, 0.6, 0.8)},
         **{f"v:{a}/(1+{1 - b:.2f}n)": (1, float(a), b) for a in (0.5, 1) for b in (0.9, 0.97, 0.99)}}
LOW = {**{f"{p}:{a}/(1+{1 - b:.3f}n)": (1, a, b) for p in ("v", "g") for a in (0.02, 0.05, 0.1) for b in (0.99, 0.999)},
       **{f"v:{a}*30/(30+n)": (1, a, 1 - 1 / 30) for a in (0.05, 0.1)}}
SCHED = {**CONST2, **VISIT, **LOW}
H_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
IE_J = (1, 3)


def setup(env):
    """Sets DYNA_ENV (must run before environment is imported in this process) and returns the scaled constants."""
    de, fams, lams = ENVS[env]
    os.environ["DYNA_ENV"] = de
    if de.startswith("npz:"):
        P = np.load(ROOT / "envs" / f"{de[4:]}.npz")["P"]
        M = P.shape[0] * P.shape[1]
    else:
        M = {"cliffwalk": 144, "maze": None, "frozen_lake": 256, "slippery_cliffwalk": 192}[de]
        if M is None:
            import environment as L
            M = L.N_S * L.N_A
    sc = lambda n: max(1, round(n * M / 144))
    bud, hor, ev = HORIZONS.get(env, (80000, 40000, 2000))
    return dict(M=M, sc=sc, fams=fams, lams=lams, BUDGET=sc(bud), H40=sc(hor), EVERY=sc(ev),
                OS_B=tuple(sorted({sc(10), sc(30), sc(100)})), IE_B=tuple(sorted({sc(10), sc(30)})))


def jfam(fam, lam):
    """lambda = 0 is the true model for every family; it is run once, under 'su'."""
    return "su" if lam == 0 else fam


def tune_jobs(env):
    c = setup(env)
    out = [("q", "-", 0.0, 1, c["sc"](10), 0.0)]
    cells = sorted({(jfam(f, l), l) for f in c["fams"] for l in c["lams"]})
    for fam, lam in cells:
        out += [("os", fam, lam, 1, B, 0.0) for B in c["OS_B"]]
        out += [("ie", fam, lam, J, B, h) for J in IE_J for B in c["IE_B"] for h in H_GRID]
    return sorted(out, key=lambda j: (-(c["BUDGET"] / j[4]) * (0.1 if j[0] == "q" else 1.0), j))


def test_jobs(env):
    # Preserve source batch dimensions; only the reported curve is exported.
    groups = json.loads((result_directory(env) / "execution-groups.json").read_text())
    c = setup(env)
    return sorted(((*j[:6], tuple(tuple(v) for v in j[6])) for j in groups),
                  key=lambda j: (-(c["BUDGET"] / j[4]), j[:6]))


def jobs_for(env, stage):
    return [(*j, None) for j in tune_jobs(env)] if stage == "tune" else test_jobs(env)


def fname(env, stage, job):
    method, fam, lam, J, B, h = job[:6]
    return result_directory(env) / (("heldout" + SFX) if stage == "test" else "tuning") / f"{method}_{fam}_l{lam}_J{J}_B{B}_h{h}.npz"


def idle():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)


def load_tape(env, seeds):
    """Generate the same replay tape as environment.load_tape; reuse an existing tape read-only when available."""
    import environment as L
    prefix = "" if L.ENV == "cliffwalk" else f"{L.ENV.replace(':', '_')}_"
    shared = ROOT / "tapes" / f"{prefix}tape_{seeds[0]}_{seeds[-1]}_{TAPE_LEN}.npz"
    mine = OUTROOT / "tapes" / f"{prefix}tape_{seeds[0]}_{seeds[-1]}_{TAPE_LEN}.npz"
    for f in (mine, shared):
        if f.exists():
            z = np.load(f)
            return z["s"], z["a"], z["ns"], str(z["checksum"])
    from besp.domains.jax_rl.replay_tape import generate_uniform_sa_replay_tape
    tape = generate_uniform_sa_replay_tape(mdp_P=np.asarray(L.P_TRUE), mdp_R=np.asarray(L.R_TRUE), seed_values=tuple(seeds),
                                           tape_steps=TAPE_LEN, env_name=L.ENV, env_params={"gamma": L.GAMMA}, gamma=L.GAMMA)
    s, a, ns = (np.asarray(x[:, :TAPE_LEN], dtype=np.int16) for x in (tape.states, tape.actions, tape.next_states))
    mine.parent.mkdir(parents=True, exist_ok=True)
    tmp = mine.with_name(mine.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, s=s, a=a, ns=ns, checksum=tape.manifest.tape_checksum)
    tmp.replace(mine)
    return s, a, ns, tape.manifest.tape_checksum


def reference(env):
    """(V*, pi*, d) with d the start-state discounted occupancy of pi*."""
    import metrics
    V, pi = metrics.optimal_policy()
    return V, pi, metrics.occupancy(pi)


def run_job(args):
    env, stage, job = args
    out = fname(env, stage, job)
    if out.exists():
        return job[:6], 0.0
    idle()
    c = setup(env)
    import learning
    import metrics
    method, fam, lam, J, B, h, combos = job
    t0 = time.perf_counter()
    seeds = SEEDS[stage]
    if combos is None:
        scheds = tuple(SCHED) if method in ("os", "ie", "q") else ("c0.02",)
        combos = [(s, h) for s in scheds]
    S, C = len(seeds), len(combos)
    E = S * C
    kinds, a, b = (np.repeat([SCHED[s][k] for s, _ in combos], S) for k in range(3))
    visit = np.repeat([s.startswith("v:") for s, _ in combos], S)
    hh = np.repeat([x if x else 1e6 for _, x in combos], S)
    lx, mx = learning.corruption("su" if fam == "-" else fam, lam)
    tape = load_tape(env, seeds)
    u, vpi, vest, pi = metrics.run(method, J, B, np.tile(lx, (E, 1)), np.tile(mx, (E, 1)), kinds, a, b, visit, hh,
                                np.tile(np.arange(S), C), tape, c["BUDGET"], every=c["EVERY"])
    V, _, d = reference(env)
    cells = metrics.cells(vpi, vest, V, d)  # (evals, E, n_cells)
    T, N = len(u), vpi.shape[-1]
    extra = {}
    if stage == "test":
        extra = dict(Vpi=vpi.reshape(T, C, S, N).astype(np.float32), Vest=vest.reshape(T, C, S, N).astype(np.float32),
                     pi=pi.reshape(T, C, S, N))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, used=u, cells=cells.reshape(T, C, S, -1), cell_names=np.asarray(metrics.cell_names()),
                        scheds=np.asarray([s for s, _ in combos]), hs=np.asarray([float(x) for _, x in combos]),
                        seeds=np.asarray(seeds), **extra)
    tmp.replace(out)
    return job[:6], time.perf_counter() - t0


def run_local(env, stage, workers):
    todo = [(env, stage, j) for j in jobs_for(env, stage) if not fname(env, stage, j).exists()]
    if not todo:
        return
    setup(env)
    load_tape(env, SEEDS[stage])
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for n, (k, el) in enumerate(pool.imap_unordered(run_job, todo, chunksize=1)):
            print(f"{n + 1}/{len(todo)} {k} {el:.0f}s elapsed {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- selection and report
def xgrid(used, c):
    """x positions of the evaluations: actual samples used or nominal budgets (CLIFFWALK_AXIS=nominal)."""
    return np.arange(0, c["BUDGET"] + 1, c["EVERY"])[:len(used)] if NOM else used


def summarize(u, e, horizon):
    """e (..., evals) -> (final = mean of the last 3 evals up to the horizon, area to the horizon / horizon)."""
    k = np.flatnonzero(u <= horizon)
    return e[..., k[-3:]].mean(-1), np.trapezoid(e[..., k], u[k], axis=-1) / horizon


def settings(env, method):
    c = setup(env)
    if method == "os":
        return [(1, B, 0.0) for B in c["OS_B"]]
    if method == "ie":
        return [(J, B, h) for J in IE_J for B in c["IE_B"] for h in H_GRID]
    return [(1, c["sc"](10), 0.0)]


def tune_table(env, method, fam, lam):
    """{(sched, J, B, h): area40 per cell (n_cells,)} on the seed-mean curve of each cell (tuning seeds)."""
    c = setup(env)
    out = {}
    f, l = ("-", 0.0) if method == "q" else (jfam(fam, lam), lam)
    for J, B, h in settings(env, method):
        z = np.load(fname(env, "tune", (method, f, l, J, B, h)))
        u, cells = xgrid(z["used"], c), z["cells"]  # (T, C, S, K)
        _, area = summarize(u, cells.mean(2).transpose(1, 2, 0), c["H40"])  # (C, K)
        for i, s in enumerate(z["scheds"]):
            out[(str(s), J, B, h)] = area[i]
    return out, [str(x) for x in z["cell_names"]]
