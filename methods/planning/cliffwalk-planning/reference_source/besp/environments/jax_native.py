"""JAX-native maintained environment constructors."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp

from ..core.types import MDP

if TYPE_CHECKING:
    import numpy as np

FloatArray = jax.Array
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _coerce_seed(seed: int | None = None, rng: Any | None = None) -> int:
    if seed is not None:
        return int(seed)
    if rng is not None and hasattr(rng, "integers"):
        return int(rng.integers(0, 2**31 - 1))
    return 0


def _empty_mdp(n_s: int, n_a: int) -> tuple[FloatArray, FloatArray]:
    return (
        jnp.zeros((n_s, n_a, n_s), dtype=jnp.float64),
        jnp.zeros((n_s, n_a), dtype=jnp.float64),
    )


def _default_branching(n_s: int) -> int:
    return max(2, min(8, int(math.log(max(n_s, 2)))))


def _default_reward_states(n_s: int) -> int:
    return max(1, int(math.ceil(0.1 * n_s)))


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def _load_frozen_garnet_arrays(instance_manifest: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    manifest_path = _resolve_repo_path(instance_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid frozen Garnet manifest at {manifest_path}.")
    arrays_rel = manifest.get("artifacts", {}).get("environment_arrays_npz")
    if not isinstance(arrays_rel, str) or not arrays_rel:
        raise ValueError(f"Frozen Garnet manifest missing environment_arrays_npz at {manifest_path}.")

    suite_dir = manifest_path.parents[2]
    candidate_paths = [
        suite_dir / arrays_rel,
        manifest_path.parent / arrays_rel,
        _resolve_repo_path(arrays_rel),
    ]
    arrays_path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if arrays_path is None:
        raise FileNotFoundError(
            f"Frozen Garnet arrays not found for manifest {manifest_path}; checked {candidate_paths}."
        )

    with np.load(arrays_path, allow_pickle=False) as payload:
        p = np.asarray(payload["P"], dtype=np.float64)
        r = np.asarray(payload["R"], dtype=np.float64)
    return p, r


def generate_modified_cliffwalk_jax(gamma: float = 0.9, **_: Any) -> MDP:
    n = 6
    n_s = n * n
    n_a = 4

    def idx(row: int, col: int) -> int:
        return row * n + col

    goal = idx(0, n - 1)
    holes: list[int] = []
    r_state = jnp.full((n_s,), -1.0, dtype=jnp.float64)
    r_state = r_state.at[goal].set(20.0)
    for row, penalty in zip((0, 2, 4), (-32.0, -16.0, -8.0), strict=True):
        for col in range(1, n - 1):
            state = idx(row, col)
            holes.append(state)
            r_state = r_state.at[state].set(penalty)
    absorbing = set(holes + [goal])

    p, _unused_r = _empty_mdp(n_s, n_a)
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for row in range(n):
        for col in range(n):
            state = idx(row, col)
            if state in absorbing:
                p = p.at[state, :, state].set(1.0)
                continue
            for action, (dr, dc) in enumerate(actions):
                nr, nc = row + dr, col + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    nr, nc = row, col
                succ = idx(nr, nc)
                prob = jnp.zeros((n_s,), dtype=jnp.float64)
                prob = prob.at[succ].add(0.9)
                fail_prob = 0.1 / 3.0
                for alt_action, (adr, adc) in enumerate(actions):
                    if alt_action == action:
                        continue
                    rr, cc = row + adr, col + adc
                    if not (0 <= rr < n and 0 <= cc < n):
                        rr, cc = row, col
                    prob = prob.at[idx(rr, cc)].add(fail_prob)
                p = p.at[state, action].set(prob)
    r = jnp.tile(r_state[:, None], (1, n_a))
    return MDP(P=p, R=r, gamma=gamma)


def generate_maze_jax(gamma: float = 0.9, **_: Any) -> MDP:
    n_rows = 3
    n_columns = 3
    n_s = n_rows * n_columns
    n_a = 4
    success_prob = 0.9
    unif_prob = (1.0 - success_prob) / 3.0
    goal_state = 2
    walls = [{0, 1}, {3, 4}, {1, 2}, {7, 8}]

    def get_target(state: int, action: int) -> int:
        col = state % n_columns
        row = state // n_columns
        if action == 0:
            target_r, target_c = max(row - 1, 0), col
        elif action == 1:
            target_r, target_c = row, min(col + 1, n_columns - 1)
        elif action == 2:
            target_r, target_c = min(row + 1, n_rows - 1), col
        elif action == 3:
            target_r, target_c = row, max(col - 1, 0)
        else:
            raise ValueError("Illegal action")
        target = target_r * n_columns + target_c
        if {state, target} in walls:
            return state
        return target

    p, r = _empty_mdp(n_s, n_a)
    for state in range(n_s):
        for action in range(n_a):
            for direction in range(n_a):
                target = get_target(state, direction)
                prob = success_prob if direction == action else unif_prob
                p = p.at[state, action, target].add(prob)
        for action in range(n_a):
            if get_target(state, action) == goal_state:
                r = r.at[state, action].set(1.0)
    return MDP(P=p, R=r, gamma=gamma)


def generate_garnet_jax(
    n_s: int = 50,
    n_a: int = 4,
    branching: int | None = None,
    reward_states: int | None = None,
    gamma: float = 0.99,
    seed: int | None = None,
    rng: Any | None = None,
    **_: Any,
) -> MDP:
    branching = branching if branching is not None else _default_branching(n_s)
    reward_states = reward_states if reward_states is not None else _default_reward_states(n_s)
    key = jax.random.PRNGKey(_coerce_seed(seed, rng))
    p, r = _empty_mdp(n_s, n_a)

    for state in range(n_s):
        for action in range(n_a):
            key, key_choice, key_cut = jax.random.split(key, 3)
            next_states = jax.random.choice(
                key_choice,
                n_s,
                shape=(branching,),
                replace=False,
            )
            cuts = jnp.sort(jax.random.uniform(key_cut, (branching - 1,), dtype=jnp.float64))
            probs = jnp.diff(
                jnp.concatenate(
                    (
                        jnp.array([0.0], dtype=jnp.float64),
                        cuts,
                        jnp.array([1.0], dtype=jnp.float64),
                    )
                )
            )
            p = p.at[state, action, next_states].set(probs)

    key, key_reward_states, key_rewards = jax.random.split(key, 3)
    reward_idx = jax.random.choice(
        key_reward_states,
        n_s,
        shape=(reward_states,),
        replace=False,
    )
    reward_values = jax.random.uniform(
        key_rewards,
        (reward_states, n_a),
        dtype=jnp.float64,
    )
    r = r.at[reward_idx, :].set(reward_values)
    return MDP(P=p, R=r, gamma=gamma)


def generate_frozen_garnet_jax(
    *,
    instance_manifest: str | Path,
    gamma: float = 0.99,
    **_: Any,
) -> MDP:
    p, r = _load_frozen_garnet_arrays(instance_manifest)
    return MDP(
        P=jnp.asarray(p, dtype=jnp.float64),
        R=jnp.asarray(r, dtype=jnp.float64),
        gamma=gamma,
    )


def generate_river_swim_jax(
    n_s: int = 6,
    small_reward: float = 0.005,
    big_reward: float = 1.0,
    gamma: float = 0.99,
    **_: Any,
) -> MDP:
    n_a = 2
    p, r = _empty_mdp(n_s, n_a)
    for state in range(n_s):
        left = max(0, state - 1)
        p = p.at[state, 0, left].set(1.0)
        if state == 0:
            r = r.at[state, 0].set(small_reward)
        move_left = max(0, state - 1)
        move_right = min(n_s - 1, state + 1)
        p = p.at[state, 1, move_left].add(0.6)
        p = p.at[state, 1, state].add(0.35)
        p = p.at[state, 1, move_right].add(0.05)
        if state == n_s - 1:
            r = r.at[state, 1].set(big_reward)
    return MDP(P=p, R=r, gamma=gamma)


def generate_chain_jax(
    n_s: int = 5,
    p_forward: float = 0.8,
    small_reward: float = 0.01,
    big_reward: float = 1.0,
    gamma: float = 0.99,
    **_: Any,
) -> MDP:
    n_a = 2
    p, r = _empty_mdp(n_s, n_a)
    for state in range(n_s):
        left = max(0, state - 1)
        right = min(n_s - 1, state + 1)
        p = p.at[state, 0, left].add(p_forward)
        p = p.at[state, 0, right].add(1.0 - p_forward)
        if state == 0:
            r = r.at[state, 0].set(small_reward)
        p = p.at[state, 1, right].add(p_forward)
        p = p.at[state, 1, left].add(1.0 - p_forward)
        if state == n_s - 1:
            r = r.at[state, 1].set(big_reward)
    return MDP(P=p, R=r, gamma=gamma)


def generate_stochastic_grid_jax(
    n: int = 10,
    slip: float = 0.1,
    wall_prob: float = 0.1,
    goal_reward: float = 1.0,
    trap_reward: float = -1.0,
    gamma: float = 0.99,
    seed: int | None = None,
    rng: Any | None = None,
    **_: Any,
) -> MDP:
    n_s = n * n
    n_a = 4
    key = jax.random.PRNGKey(_coerce_seed(seed, rng))
    key_walls, key_traps = jax.random.split(key)
    wall_draws = jax.random.uniform(key_walls, (n, n), dtype=jnp.float64)
    trap_draws = jax.random.uniform(key_traps, (n, n), dtype=jnp.float64)

    def idx(row: int, col: int) -> int:
        return row * n + col

    walls: set[tuple[int, int]] = set()
    start = (0, 0)
    goal = (n - 1, n - 1)
    for row in range(n):
        for col in range(n):
            if (row, col) in {start, goal}:
                continue
            if float(wall_draws[row, col]) < wall_prob:
                walls.add((row, col))

    traps: set[tuple[int, int]] = set()
    for row in range(n):
        for col in range(n):
            if (row, col) in walls or (row, col) in {start, goal}:
                continue
            if float(trap_draws[row, col]) < wall_prob:
                traps.add((row, col))

    p, r = _empty_mdp(n_s, n_a)
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for row in range(n):
        for col in range(n):
            state = idx(row, col)
            if (row, col) == goal:
                p = p.at[state, :, state].set(1.0)
                r = r.at[state, :].set(goal_reward)
                continue
            if (row, col) in traps:
                p = p.at[state, :, state].set(1.0)
                r = r.at[state, :].set(trap_reward)
                continue
            for action, (dr, dc) in enumerate(actions):
                prob = jnp.zeros((n_s,), dtype=jnp.float64)
                nr, nc = row + dr, col + dc
                if not (0 <= nr < n and 0 <= nc < n) or (nr, nc) in walls:
                    nr, nc = row, col
                prob = prob.at[idx(nr, nc)].add(1.0 - slip)
                slip_prob = slip / (n_a - 1)
                for alt_action, (adr, adc) in enumerate(actions):
                    if alt_action == action:
                        continue
                    rr, cc = row + adr, col + adc
                    if not (0 <= rr < n and 0 <= cc < n) or (rr, cc) in walls:
                        rr, cc = row, col
                    prob = prob.at[idx(rr, cc)].add(slip_prob)
                p = p.at[state, action].set(prob)
                r = r.at[state, action].set(0.0)
    return MDP(P=p, R=r, gamma=gamma)


def generate_four_rooms_jax(
    size: int = 15,
    slip: float = 0.1,
    gamma: float = 0.99,
    goal_reward: float = 1.0,
    trap_reward: float = -1.0,
    seed: int | None = None,
    rng: Any | None = None,
    **_: Any,
) -> MDP:
    n_s = size * size
    n_a = 4
    key = jax.random.PRNGKey(_coerce_seed(seed, rng))
    key_door_v, key_door_h, key_traps = jax.random.split(key, 3)

    def idx(row: int, col: int) -> int:
        return row * size + col

    mid = size // 2
    door_v = int(jax.random.randint(key_door_v, (), 1, size - 1))
    door_h = int(jax.random.randint(key_door_h, (), 1, size - 1))
    walls: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for row in range(size):
        if row != door_v:
            walls.add(((row, mid), (row, mid - 1)))
    for col in range(size):
        if col != door_h:
            walls.add(((mid, col), (mid - 1, col)))

    def blocked(r1: int, c1: int, r2: int, c2: int) -> bool:
        return ((r1, c1), (r2, c2)) in walls or ((r2, c2), (r1, c1)) in walls

    traps: set[tuple[int, int]] = set()
    room_ranges = [
        (range(0, mid), range(0, mid)),
        (range(0, mid), range(mid, size)),
        (range(mid, size), range(0, mid)),
        (range(mid, size), range(mid, size)),
    ]
    room_keys = iter(jax.random.split(key_traps, len(room_ranges)))
    for rows_range, cols_range in room_ranges:
        candidates = [
            (row, col)
            for row in rows_range
            for col in cols_range
            if (row, col) not in {(0, 0), (size - 1, size - 1)}
        ]
        if not candidates:
            continue
        room_key = next(room_keys)
        chosen_idx = int(jax.random.randint(room_key, (), 0, len(candidates)))
        traps.add(candidates[chosen_idx])

    goal = (size - 1, size - 1)
    p, r = _empty_mdp(n_s, n_a)
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for row in range(size):
        for col in range(size):
            state = idx(row, col)
            if (row, col) == goal:
                p = p.at[state, :, state].set(1.0)
                r = r.at[state, :].set(goal_reward)
                continue
            if (row, col) in traps:
                p = p.at[state, :, state].set(1.0)
                r = r.at[state, :].set(trap_reward)
                continue
            for action, (dr, dc) in enumerate(actions):
                prob = jnp.zeros((n_s,), dtype=jnp.float64)
                nr, nc = row + dr, col + dc
                if not (0 <= nr < size and 0 <= nc < size) or blocked(row, col, nr, nc):
                    nr, nc = row, col
                prob = prob.at[idx(nr, nc)].add(1.0 - slip)
                slip_prob = slip / (n_a - 1)
                for alt_action, (adr, adc) in enumerate(actions):
                    if alt_action == action:
                        continue
                    rr, cc = row + adr, col + adc
                    if not (0 <= rr < size and 0 <= cc < size) or blocked(row, col, rr, cc):
                        rr, cc = row, col
                    prob = prob.at[idx(rr, cc)].add(slip_prob)
                p = p.at[state, action].set(prob)
                r = r.at[state, action].set(0.0)
    return MDP(P=p, R=r, gamma=gamma)


def generate_dense_reward_grid_jax(
    n: int = 10,
    slip: float = 0.1,
    gamma: float = 0.99,
    seed: int | None = None,
    rng: Any | None = None,
    **_: Any,
) -> MDP:
    n_s = n * n
    n_a = 4
    key = jax.random.PRNGKey(_coerce_seed(seed, rng))

    def idx(row: int, col: int) -> int:
        return row * n + col

    p, _unused_r = _empty_mdp(n_s, n_a)
    r = jax.random.uniform(key, (n_s, n_a), dtype=jnp.float64)
    goal = (n - 1, n - 1)
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    for row in range(n):
        for col in range(n):
            state = idx(row, col)
            if (row, col) == goal:
                p = p.at[state, :, state].set(1.0)
                r = r.at[state, :].set(1.0)
                continue
            for action, (dr, dc) in enumerate(actions):
                prob = jnp.zeros((n_s,), dtype=jnp.float64)
                nr, nc = row + dr, col + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    nr, nc = row, col
                prob = prob.at[idx(nr, nc)].add(1.0 - slip)
                slip_prob = slip / (n_a - 1)
                for alt_action, (adr, adc) in enumerate(actions):
                    if alt_action == action:
                        continue
                    rr, cc = row + adr, col + adc
                    if not (0 <= rr < n and 0 <= cc < n):
                        rr, cc = row, col
                    prob = prob.at[idx(rr, cc)].add(slip_prob)
                p = p.at[state, action].set(prob)
    return MDP(P=p, R=r, gamma=gamma)


def generate_frozen_lake_jax(
    size: int = 8,
    slip_prob: float = 1 / 3,
    gamma: float = 0.99,
    hole_positions: list[tuple[int, int]] | None = None,
    **_: Any,
) -> MDP:
    n_s = size * size
    n_a = 4

    def idx(row: int, col: int) -> int:
        return row * size + col

    if hole_positions is None and size == 8:
        hole_positions = [
            (2, 3),
            (3, 5),
            (4, 3),
            (5, 1),
            (5, 2),
            (5, 6),
            (6, 1),
            (6, 4),
            (6, 6),
            (7, 3),
        ]
    elif hole_positions is None:
        hole_positions = []

    holes = set(hole_positions)
    goal = (size - 1, size - 1)
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    p, r = _empty_mdp(n_s, n_a)
    for row in range(size):
        for col in range(size):
            state = idx(row, col)
            if (row, col) in holes or (row, col) == goal:
                p = p.at[state, :, state].set(1.0)
                if (row, col) == goal:
                    r = r.at[state, :].set(1.0)
                continue
            for action, (dr, dc) in enumerate(actions):
                prob = jnp.zeros((n_s,), dtype=jnp.float64)
                nr, nc = row + dr, col + dc
                if not (0 <= nr < size and 0 <= nc < size):
                    nr, nc = row, col
                prob = prob.at[idx(nr, nc)].add(1.0 - slip_prob)
                for perp_action in ((action + 1) % 4, (action + 3) % 4):
                    pr, pc = row + actions[perp_action][0], col + actions[perp_action][1]
                    if not (0 <= pr < size and 0 <= pc < size):
                        pr, pc = row, col
                    prob = prob.at[idx(pr, pc)].add(slip_prob / 2.0)
                p = p.at[state, action].set(prob)
    return MDP(P=p, R=r, gamma=gamma)


def generate_slippery_cliffwalk_jax(
    slip_prob: float = 0.1,
    gamma: float = 0.99,
    **_: Any,
) -> MDP:
    n_rows = 4
    n_cols = 12
    n_s = n_rows * n_cols
    n_a = 4

    def idx(row: int, col: int) -> int:
        return row * n_cols + col

    start = (3, 0)
    goal = (3, 11)
    cliff = {(3, col) for col in range(1, 11)}
    start_idx = idx(*start)
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    p, r = _empty_mdp(n_s, n_a)

    def get_next_state(action_idx: int, row: int, col: int) -> tuple[int, float]:
        dr, dc = actions[action_idx]
        nr, nc = row + dr, col + dc
        if not (0 <= nr < n_rows and 0 <= nc < n_cols):
            nr, nc = row, col
        if (nr, nc) in cliff:
            return start_idx, -100.0
        return idx(nr, nc), -1.0

    for row in range(n_rows):
        for col in range(n_cols):
            state = idx(row, col)
            if (row, col) == goal:
                p = p.at[state, :, state].set(1.0)
                continue
            for action in range(n_a):
                prob = jnp.zeros((n_s,), dtype=jnp.float64)
                intended_next, intended_reward = get_next_state(action, row, col)
                perp1_next, perp1_reward = get_next_state((action + 1) % 4, row, col)
                perp2_next, perp2_reward = get_next_state((action + 3) % 4, row, col)
                p_intended = 1.0 - slip_prob
                p_perp = slip_prob / 2.0
                prob = prob.at[intended_next].add(p_intended)
                prob = prob.at[perp1_next].add(p_perp)
                prob = prob.at[perp2_next].add(p_perp)
                p = p.at[state, action].set(prob)
                r = r.at[state, action].set(
                    p_intended * intended_reward + p_perp * perp1_reward + p_perp * perp2_reward
                )
    return MDP(P=p, R=r, gamma=gamma)


def generate_taxi_jax(gamma: float = 0.99, **_: Any) -> MDP:
    rows, cols = 5, 5
    passenger_locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
    destinations = passenger_locs
    n_pass = len(passenger_locs)
    n_dest = len(destinations)
    n_s = rows * cols * n_pass * n_dest
    n_a = 6

    def state_idx(row: int, col: int, p_idx: int, d_idx: int) -> int:
        return ((row * cols + col) * n_pass + p_idx) * n_dest + d_idx

    walls = set()
    walls.update({((row, 0), (row, 1)) for row in range(rows)})
    walls.update({((row, 2), (row, 3)) for row in range(rows)})
    walls.update({((row, 3), (row, 4)) for row in range(rows) if row != 0})

    def blocked(r1: int, c1: int, r2: int, c2: int) -> bool:
        return ((r1, c1), (r2, c2)) in walls or ((r2, c2), (r1, c1)) in walls

    p, _unused_r = _empty_mdp(n_s, n_a)
    r = jnp.full((n_s, n_a), -1.0, dtype=jnp.float64)
    for row in range(rows):
        for col in range(cols):
            for p_idx, p_loc in enumerate(passenger_locs):
                for d_idx, d_loc in enumerate(destinations):
                    state = state_idx(row, col, p_idx, d_idx)
                    for action, (dr, dc) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                        nr, nc = row + dr, col + dc
                        if not (0 <= nr < rows and 0 <= nc < cols) or blocked(row, col, nr, nc):
                            nr, nc = row, col
                        next_state = state_idx(nr, nc, p_idx, d_idx)
                        p = p.at[state, action, next_state].set(1.0)
                    if (row, col) == p_loc:
                        p = p.at[state, 4, state].set(1.0)
                    else:
                        p = p.at[state, 4, state].set(1.0)
                        r = r.at[state, 4].set(-10.0)
                    if (row, col) == d_loc and (row, col) == p_loc:
                        p = p.at[state, 5, state_idx(row, col, p_idx, d_idx)].set(1.0)
                        r = r.at[state, 5].set(20.0)
                    else:
                        p = p.at[state, 5, state].set(1.0)
                        r = r.at[state, 5].set(-10.0)
    return MDP(P=p, R=r, gamma=gamma)


def generate_mini_breakout_jax(
    n_paddle: int = 3,
    n_ball_x: int = 3,
    n_ball_y: int = 3,
    gamma: float = 0.99,
    hit_reward: float = 1.0,
    miss_reward: float = -1.0,
    slip: float = 0.1,
    **_: Any,
) -> MDP:
    n_a = 3
    n_states = n_paddle * n_ball_x * n_ball_y

    def state_idx(paddle: int, ball_x: int, ball_y: int) -> int:
        return (paddle * n_ball_x + ball_x) * n_ball_y + ball_y

    p, r = _empty_mdp(n_states, n_a)
    top_y = n_ball_y - 1
    reset_bx = n_ball_x // 2
    reset_by = top_y
    for paddle in range(n_paddle):
        for ball_x in range(n_ball_x):
            for ball_y in range(n_ball_y):
                state = state_idx(paddle, ball_x, ball_y)
                for action in range(n_a):
                    prob_paddle = jnp.zeros((n_paddle,), dtype=jnp.float64)
                    intended_paddle = paddle
                    if action == 1:
                        intended_paddle = max(0, paddle - 1)
                    elif action == 2:
                        intended_paddle = min(n_paddle - 1, paddle + 1)
                    prob_paddle = prob_paddle.at[intended_paddle].add(1.0 - slip)
                    for alt_action in (candidate for candidate in (0, 1, 2) if candidate != action):
                        alt = paddle
                        if alt_action == 1:
                            alt = max(0, paddle - 1)
                        elif alt_action == 2:
                            alt = min(n_paddle - 1, paddle + 1)
                        prob_paddle = prob_paddle.at[alt].add(slip / 2.0)

                    prob_ball = jnp.zeros((n_ball_x, n_ball_y), dtype=jnp.float64)
                    intended_y = ball_y - 1
                    intended_x = ball_x
                    prob_ball = prob_ball.at[intended_x, max(intended_y, 0)].add(1.0 - slip)
                    for dx in (-1, 1):
                        next_x = min(max(0, ball_x + dx), n_ball_x - 1)
                        next_y = max(0, ball_y)
                        prob_ball = prob_ball.at[next_x, next_y].add(slip / 2.0)

                    reward_value = 0.0
                    for next_paddle in range(n_paddle):
                        paddle_prob = float(prob_paddle[next_paddle])
                        if paddle_prob == 0.0:
                            continue
                        for next_ball_x in range(n_ball_x):
                            for next_ball_y in range(n_ball_y):
                                ball_prob = float(prob_ball[next_ball_x, next_ball_y])
                                if ball_prob == 0.0:
                                    continue
                                reward_value = 0.0
                                final_x, final_y = next_ball_x, next_ball_y
                                if next_ball_y == 0:
                                    reward_value = hit_reward if next_ball_x == next_paddle else miss_reward
                                    final_x, final_y = reset_bx, reset_by
                                next_state = state_idx(next_paddle, final_x, final_y)
                                p = p.at[state, action, next_state].add(paddle_prob * ball_prob)
                                r = r.at[state, action].set(reward_value)
    return MDP(P=p, R=r, gamma=gamma)


_NATIVE_BUILDERS: dict[str, Callable[..., MDP]] = {
    "cliffwalk": generate_modified_cliffwalk_jax,
    "garnet": generate_garnet_jax,
    "maze": generate_maze_jax,
    "chain": generate_chain_jax,
    "river_swim": generate_river_swim_jax,
    "four_rooms": generate_four_rooms_jax,
    "stochastic_grid": generate_stochastic_grid_jax,
    "dense_reward_grid": generate_dense_reward_grid_jax,
    "taxi": generate_taxi_jax,
    "mini_breakout": generate_mini_breakout_jax,
    "frozen_lake": generate_frozen_lake_jax,
    "slippery_cliffwalk": generate_slippery_cliffwalk_jax,
}


def create_environment_jax_native(env_name: str, **kwargs: Any) -> MDP:
    """Create a maintained environment directly on JAX arrays."""

    gamma = kwargs.get("gamma", 0.99)
    rng = kwargs.get("rng")
    seed = kwargs.get("seed")

    if env_name == "cliffwalk":
        return generate_modified_cliffwalk_jax(gamma=gamma)
    if env_name == "garnet":
        return generate_garnet_jax(
            n_s=kwargs.get("n_states", 30),
            n_a=kwargs.get("n_actions", 4),
            branching=kwargs.get("branching"),
            reward_states=kwargs.get("reward_states"),
            gamma=gamma,
            seed=seed,
            rng=rng,
        )
    if env_name == "garnet_frozen":
        return generate_frozen_garnet_jax(
            instance_manifest=kwargs["instance_manifest"],
            gamma=gamma,
        )
    if env_name == "maze":
        return generate_maze_jax(gamma=gamma)
    if env_name == "chain":
        return generate_chain_jax(n_s=kwargs.get("n_states", 20), gamma=gamma)
    if env_name == "river_swim":
        return generate_river_swim_jax(n_s=kwargs.get("n_states", 6), gamma=gamma)
    if env_name == "four_rooms":
        return generate_four_rooms_jax(
            size=kwargs.get("size", 15),
            gamma=gamma,
            seed=seed,
            rng=rng,
        )
    if env_name == "stochastic_grid":
        return generate_stochastic_grid_jax(
            n=kwargs.get("size", 5),
            gamma=gamma,
            seed=seed,
            rng=rng,
        )
    if env_name == "dense_reward_grid":
        return generate_dense_reward_grid_jax(
            n=kwargs.get("size", 5),
            gamma=gamma,
            seed=seed,
            rng=rng,
        )
    if env_name == "taxi":
        return generate_taxi_jax(gamma=gamma)
    if env_name == "mini_breakout":
        return generate_mini_breakout_jax(gamma=gamma)
    if env_name == "frozen_lake":
        return generate_frozen_lake_jax(
            size=kwargs.get("size", 8),
            slip_prob=kwargs.get("slip_prob", 1 / 3),
            gamma=gamma,
            hole_positions=kwargs.get("hole_positions"),
        )
    if env_name == "slippery_cliffwalk":
        return generate_slippery_cliffwalk_jax(
            slip_prob=kwargs.get("slip_prob", 0.1),
            gamma=gamma,
        )

    raise ValueError(f"Unknown environment: {env_name}")
