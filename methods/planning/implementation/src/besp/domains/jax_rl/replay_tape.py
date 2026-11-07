"""Immutable replay-tape artifacts for maintained stochastic-control runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...core.artifact_io import save_json_payload


DEFAULT_REPLAY_TAPE_CONTRACT_SCOPE = "transition_law"


def _canonical_env_params(env_params: Mapping[str, Any] | None) -> dict[str, Any]:
    if env_params is None:
        return {}
    return json.loads(json.dumps(dict(env_params), sort_keys=True))


def normalize_replay_tape_contract_scope(contract_scope: str | None) -> str:
    resolved = str(contract_scope or DEFAULT_REPLAY_TAPE_CONTRACT_SCOPE).strip().lower().replace("-", "_")
    if resolved not in {"transition_law", "discount_specific"}:
        raise ValueError(
            "Replay tape contract_scope must be one of: transition_law, discount_specific."
        )
    return resolved


def replay_tape_contract_env_params(
    env_params: Mapping[str, Any] | None,
    *,
    contract_scope: str = DEFAULT_REPLAY_TAPE_CONTRACT_SCOPE,
) -> dict[str, Any]:
    resolved_scope = normalize_replay_tape_contract_scope(contract_scope)
    canonical = _canonical_env_params(env_params)
    if resolved_scope == "transition_law":
        canonical.pop("gamma", None)
    return canonical


def _manifest_path(bundle_root: Path | str) -> Path:
    return Path(bundle_root) / "manifest.json"


def _states_path(bundle_root: Path | str, relative_path: str) -> Path:
    return Path(bundle_root) / relative_path


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class ReplayTapeManifest:
    env_name: str
    env_params: Mapping[str, Any]
    gamma: float
    sampling_distribution: str
    seed_values: tuple[int, ...]
    tape_steps: int
    storage_shape: tuple[int, int]
    tape_checksum: str
    schema_version: int
    states_path: str
    actions_path: str
    next_states_path: str
    created_at: str
    contract_scope: str = DEFAULT_REPLAY_TAPE_CONTRACT_SCOPE

    def as_mapping(self) -> dict[str, Any]:
        return {
            "env_name": self.env_name,
            "env_params": dict(self.env_params),
            "gamma": self.gamma,
            "sampling_distribution": self.sampling_distribution,
            "contract_scope": self.contract_scope,
            "seed_values": list(self.seed_values),
            "tape_steps": self.tape_steps,
            "storage_shape": list(self.storage_shape),
            "tape_checksum": self.tape_checksum,
            "schema_version": self.schema_version,
            "states_path": self.states_path,
            "actions_path": self.actions_path,
            "next_states_path": self.next_states_path,
            "created_at": self.created_at,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ReplayTapeManifest":
        return cls(
            env_name=str(mapping["env_name"]),
            env_params=_canonical_env_params(mapping.get("env_params")),
            gamma=float(mapping["gamma"]),
            sampling_distribution=str(mapping["sampling_distribution"]),
            contract_scope=normalize_replay_tape_contract_scope(mapping.get("contract_scope")),
            seed_values=tuple(int(value) for value in mapping["seed_values"]),
            tape_steps=int(mapping["tape_steps"]),
            storage_shape=tuple(int(value) for value in mapping["storage_shape"]),
            tape_checksum=str(mapping["tape_checksum"]),
            schema_version=int(mapping.get("schema_version", 1)),
            states_path=str(mapping["states_path"]),
            actions_path=str(mapping["actions_path"]),
            next_states_path=str(mapping["next_states_path"]),
            created_at=str(mapping["created_at"]),
        )


@dataclass(frozen=True)
class ReplayTapeBundle:
    manifest: ReplayTapeManifest
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray


def _validate_manifest_contract(
    *,
    manifest: ReplayTapeManifest,
    env_name: str,
    env_params: Mapping[str, Any] | None,
    gamma: float,
    sampling_distribution: str,
) -> None:
    resolved_scope = normalize_replay_tape_contract_scope(manifest.contract_scope)
    requested_env_params = replay_tape_contract_env_params(
        env_params,
        contract_scope=resolved_scope,
    )
    if (
        manifest.env_name != str(env_name)
        or replay_tape_contract_env_params(
            manifest.env_params,
            contract_scope=resolved_scope,
        ) != requested_env_params
        or str(manifest.sampling_distribution) != str(sampling_distribution)
    ):
        raise ValueError("Existing replay tape does not match the requested environment/seed contract.")
    if resolved_scope == "discount_specific" and not math.isclose(
        float(manifest.gamma),
        float(gamma),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Existing replay tape does not match the requested discount-specific contract."
        )


def select_replay_tape_bundle(
    bundle: ReplayTapeBundle,
    *,
    seed_values: Sequence[int],
    tape_steps: int,
) -> ReplayTapeBundle:
    requested_seed_values, requested_tape_steps, row_indices = _replay_tape_selection_spec(
        manifest=bundle.manifest,
        seed_values=seed_values,
        tape_steps=tape_steps,
    )
    states = np.asarray(bundle.states[row_indices, :requested_tape_steps], dtype=np.int32)
    actions = np.asarray(bundle.actions[row_indices, :requested_tape_steps], dtype=np.int32)
    next_states = np.asarray(bundle.next_states[row_indices, :requested_tape_steps], dtype=np.int32)
    return ReplayTapeBundle(
        manifest=ReplayTapeManifest(
            env_name=bundle.manifest.env_name,
            env_params=bundle.manifest.env_params,
            gamma=bundle.manifest.gamma,
            sampling_distribution=bundle.manifest.sampling_distribution,
            contract_scope=bundle.manifest.contract_scope,
            seed_values=requested_seed_values,
            tape_steps=requested_tape_steps,
            storage_shape=states.shape,
            tape_checksum=compute_replay_tape_checksum(
                states=states,
                actions=actions,
                next_states=next_states,
            ),
            schema_version=bundle.manifest.schema_version,
            states_path=bundle.manifest.states_path,
            actions_path=bundle.manifest.actions_path,
            next_states_path=bundle.manifest.next_states_path,
            created_at=bundle.manifest.created_at,
        ),
        states=states,
        actions=actions,
        next_states=next_states,
    )


def _replay_tape_selection_spec(
    *,
    manifest: ReplayTapeManifest,
    seed_values: Sequence[int],
    tape_steps: int,
) -> tuple[tuple[int, ...], int, list[int]]:
    requested_seed_values = tuple(int(value) for value in seed_values)
    requested_tape_steps = int(tape_steps)
    if requested_tape_steps > int(manifest.tape_steps):
        raise ValueError(
            "Requested replay tape prefix exceeds the stored tape horizon; "
            f"requested={requested_tape_steps}, stored={manifest.tape_steps}."
        )
    row_lookup = {int(seed): index for index, seed in enumerate(manifest.seed_values)}
    missing = [seed for seed in requested_seed_values if seed not in row_lookup]
    if missing:
        raise ValueError(
            "Requested replay-tape seed subset is not present in the stored tape; "
            f"missing={missing}, stored={list(manifest.seed_values)}."
        )
    row_indices = [row_lookup[seed] for seed in requested_seed_values]
    return requested_seed_values, requested_tape_steps, row_indices


def compute_replay_tape_checksum(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
) -> str:
    arrays = (
        np.asarray(states, dtype=np.int32),
        np.asarray(actions, dtype=np.int32),
        np.asarray(next_states, dtype=np.int32),
    )
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def persist_replay_tape(
    *,
    bundle_root: Path | str,
    manifest: ReplayTapeManifest,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
) -> ReplayTapeManifest:
    root = Path(bundle_root)
    root.mkdir(parents=True, exist_ok=True)
    state_array = np.asarray(states, dtype=np.int32)
    action_array = np.asarray(actions, dtype=np.int32)
    next_state_array = np.asarray(next_states, dtype=np.int32)
    expected_shape = tuple(int(value) for value in manifest.storage_shape)
    for name, array in (
        ("states", state_array),
        ("actions", action_array),
        ("next_states", next_state_array),
    ):
        if array.shape != expected_shape:
            raise ValueError(f"{name} array shape must be {expected_shape}, received {array.shape}.")
    checksum = compute_replay_tape_checksum(
        states=state_array,
        actions=action_array,
        next_states=next_state_array,
    )
    if checksum != manifest.tape_checksum:
        raise ValueError(
            "Replay tape checksum mismatch while persisting artifact; "
            f"manifest={manifest.tape_checksum}, computed={checksum}."
        )
    np.save(_states_path(root, manifest.states_path), state_array, allow_pickle=False)
    np.save(_states_path(root, manifest.actions_path), action_array, allow_pickle=False)
    np.save(_states_path(root, manifest.next_states_path), next_state_array, allow_pickle=False)
    save_json_payload(_manifest_path(root), manifest.as_mapping())
    return manifest


def load_replay_tape_manifest(*, bundle_root: Path | str) -> ReplayTapeManifest:
    payload = json.loads(_manifest_path(bundle_root).read_text(encoding="utf-8"))
    return ReplayTapeManifest.from_mapping(payload)


def load_replay_tape_bundle(*, bundle_root: Path | str) -> ReplayTapeBundle:
    manifest = load_replay_tape_manifest(bundle_root=bundle_root)
    states = np.asarray(np.load(_states_path(bundle_root, manifest.states_path), mmap_mode=None), dtype=np.int32)
    actions = np.asarray(np.load(_states_path(bundle_root, manifest.actions_path), mmap_mode=None), dtype=np.int32)
    next_states = np.asarray(
        np.load(_states_path(bundle_root, manifest.next_states_path), mmap_mode=None),
        dtype=np.int32,
    )
    checksum = compute_replay_tape_checksum(states=states, actions=actions, next_states=next_states)
    if checksum != manifest.tape_checksum:
        raise ValueError(
            "Replay tape checksum mismatch while loading artifact; "
            f"manifest={manifest.tape_checksum}, computed={checksum}."
        )
    return ReplayTapeBundle(
        manifest=manifest,
        states=states,
        actions=actions,
        next_states=next_states,
    )


def load_replay_tape_selection(
    *,
    bundle_root: Path | str,
    seed_values: Sequence[int],
    tape_steps: int,
) -> ReplayTapeBundle:
    manifest = load_replay_tape_manifest(bundle_root=bundle_root)
    requested_seed_values, requested_tape_steps, row_indices = _replay_tape_selection_spec(
        manifest=manifest,
        seed_values=seed_values,
        tape_steps=tape_steps,
    )
    states_mmap = np.load(_states_path(bundle_root, manifest.states_path), mmap_mode="r")
    actions_mmap = np.load(_states_path(bundle_root, manifest.actions_path), mmap_mode="r")
    next_states_mmap = np.load(_states_path(bundle_root, manifest.next_states_path), mmap_mode="r")
    states = np.asarray(states_mmap[row_indices, :requested_tape_steps], dtype=np.int32)
    actions = np.asarray(actions_mmap[row_indices, :requested_tape_steps], dtype=np.int32)
    next_states = np.asarray(next_states_mmap[row_indices, :requested_tape_steps], dtype=np.int32)
    return ReplayTapeBundle(
        manifest=ReplayTapeManifest(
            env_name=manifest.env_name,
            env_params=manifest.env_params,
            gamma=manifest.gamma,
            sampling_distribution=manifest.sampling_distribution,
            contract_scope=manifest.contract_scope,
            seed_values=requested_seed_values,
            tape_steps=requested_tape_steps,
            storage_shape=states.shape,
            tape_checksum=compute_replay_tape_checksum(
                states=states,
                actions=actions,
                next_states=next_states,
            ),
            schema_version=manifest.schema_version,
            states_path=manifest.states_path,
            actions_path=manifest.actions_path,
            next_states_path=manifest.next_states_path,
            created_at=manifest.created_at,
        ),
        states=states,
        actions=actions,
        next_states=next_states,
    )


def _sample_uniform_sa_next_states(
    *,
    mdp_P: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    draws: np.ndarray,
    chunk_size: int = 200_000,
) -> np.ndarray:
    P = np.asarray(mdp_P, dtype=np.float64)
    state_array = np.asarray(states, dtype=np.int32)
    action_array = np.asarray(actions, dtype=np.int32)
    draw_array = np.asarray(draws, dtype=np.float64)
    if state_array.shape != action_array.shape or state_array.shape != draw_array.shape:
        raise ValueError("states, actions, and draws must share the same shape.")
    next_states = np.empty(state_array.shape, dtype=np.int32)
    n_s = int(P.shape[2])
    normalized_chunk_size = max(1, int(chunk_size))
    for start in range(0, int(state_array.shape[0]), normalized_chunk_size):
        stop = min(int(state_array.shape[0]), start + normalized_chunk_size)
        probs = P[state_array[start:stop], action_array[start:stop]]
        cdf = np.cumsum(probs, axis=1)
        chunk_next_states = np.sum(
            draw_array[start:stop, None] >= cdf,
            axis=1,
            dtype=np.int32,
        )
        next_states[start:stop] = np.minimum(chunk_next_states, n_s - 1)
    return next_states


def _generate_uniform_sa_row(
    *,
    mdp_P: np.ndarray,
    seed_value: int,
    tape_steps: int,
    n_states: int,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(int(seed_value)))
    states = rng.integers(0, int(n_states), size=int(tape_steps), dtype=np.int32)
    actions = rng.integers(0, int(n_actions), size=int(tape_steps), dtype=np.int32)
    draws = rng.random(size=int(tape_steps))
    next_states = _sample_uniform_sa_next_states(
        mdp_P=mdp_P,
        states=states,
        actions=actions,
        draws=draws,
    )
    return states, actions, next_states


def _extend_uniform_sa_row_from_prefix(
    *,
    mdp_P: np.ndarray,
    seed_value: int,
    total_steps: int,
    n_states: int,
    n_actions: int,
    prefix_states: np.ndarray,
    prefix_actions: np.ndarray,
    prefix_next_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix_state_array = np.asarray(prefix_states, dtype=np.int32)
    prefix_action_array = np.asarray(prefix_actions, dtype=np.int32)
    prefix_next_state_array = np.asarray(prefix_next_states, dtype=np.int32)
    prefix_steps = int(prefix_state_array.shape[0])
    if prefix_steps > int(total_steps):
        raise ValueError("Reference prefix exceeds the requested total_steps.")

    rng = np.random.Generator(np.random.PCG64(int(seed_value)))
    generated_prefix_states = rng.integers(0, int(n_states), size=prefix_steps, dtype=np.int32)
    generated_prefix_actions = rng.integers(0, int(n_actions), size=prefix_steps, dtype=np.int32)
    generated_prefix_draws = rng.random(size=prefix_steps)
    generated_prefix_next_states = _sample_uniform_sa_next_states(
        mdp_P=mdp_P,
        states=generated_prefix_states,
        actions=generated_prefix_actions,
        draws=generated_prefix_draws,
    )
    if (
        not np.array_equal(generated_prefix_states, prefix_state_array)
        or not np.array_equal(generated_prefix_actions, prefix_action_array)
        or not np.array_equal(generated_prefix_next_states, prefix_next_state_array)
    ):
        raise ValueError(
            "Reference replay prefix is not compatible with the maintained uniform_sa generation contract."
        )

    remaining_steps = int(total_steps) - prefix_steps
    if remaining_steps <= 0:
        return (
            prefix_state_array.copy(),
            prefix_action_array.copy(),
            prefix_next_state_array.copy(),
        )
    tail_states = rng.integers(0, int(n_states), size=remaining_steps, dtype=np.int32)
    tail_actions = rng.integers(0, int(n_actions), size=remaining_steps, dtype=np.int32)
    tail_draws = rng.random(size=remaining_steps)
    tail_next_states = _sample_uniform_sa_next_states(
        mdp_P=mdp_P,
        states=tail_states,
        actions=tail_actions,
        draws=tail_draws,
    )
    return (
        np.concatenate([prefix_state_array, tail_states]),
        np.concatenate([prefix_action_array, tail_actions]),
        np.concatenate([prefix_next_state_array, tail_next_states]),
    )


def generate_uniform_sa_replay_tape(
    *,
    mdp_P: np.ndarray,
    mdp_R: np.ndarray,
    seed_values: Sequence[int],
    tape_steps: int,
    env_name: str = "unknown_env",
    env_params: Mapping[str, Any] | None = None,
    gamma: float = 0.0,
    sampling_distribution: str = "uniform_sa",
    contract_scope: str = DEFAULT_REPLAY_TAPE_CONTRACT_SCOPE,
) -> ReplayTapeBundle:
    if str(sampling_distribution).strip().lower().replace("-", "_") != "uniform_sa":
        raise ValueError("Replay-tape generation currently supports only sampling_distribution='uniform_sa'.")
    resolved_contract_scope = normalize_replay_tape_contract_scope(contract_scope)
    P = np.asarray(mdp_P, dtype=np.float64)
    R = np.asarray(mdp_R, dtype=np.float64)
    del R
    n_s, n_a, _ = P.shape
    resolved_seed_values = tuple(int(value) for value in seed_values)
    steps = int(tape_steps)
    states = np.empty((len(resolved_seed_values), steps), dtype=np.int32)
    actions = np.empty((len(resolved_seed_values), steps), dtype=np.int32)
    next_states = np.empty((len(resolved_seed_values), steps), dtype=np.int32)
    for row_index, seed_value in enumerate(resolved_seed_values):
        row_states, row_actions, row_next_states = _generate_uniform_sa_row(
            mdp_P=P,
            seed_value=int(seed_value),
            tape_steps=steps,
            n_states=n_s,
            n_actions=n_a,
        )
        states[row_index] = row_states
        actions[row_index] = row_actions
        next_states[row_index] = row_next_states
    manifest = ReplayTapeManifest(
        env_name=str(env_name),
        env_params=replay_tape_contract_env_params(
            env_params,
            contract_scope=resolved_contract_scope,
        ),
        gamma=float(gamma),
        sampling_distribution="uniform_sa",
        contract_scope=resolved_contract_scope,
        seed_values=resolved_seed_values,
        tape_steps=steps,
        storage_shape=(len(resolved_seed_values), steps),
        tape_checksum=compute_replay_tape_checksum(
            states=states,
            actions=actions,
            next_states=next_states,
        ),
        schema_version=1,
        states_path="states.npy",
        actions_path="actions.npy",
        next_states_path="next_states.npy",
        created_at=_utc_timestamp(),
    )
    return ReplayTapeBundle(
        manifest=manifest,
        states=states,
        actions=actions,
        next_states=next_states,
    )


def ensure_uniform_sa_replay_tape(
    *,
    bundle_root: Path | str,
    mdp_P: np.ndarray,
    mdp_R: np.ndarray,
    env_name: str,
    env_params: Mapping[str, Any] | None,
    gamma: float,
    seed_values: Sequence[int],
    tape_steps: int,
    contract_scope: str = DEFAULT_REPLAY_TAPE_CONTRACT_SCOPE,
) -> ReplayTapeBundle:
    root = Path(bundle_root)
    if _manifest_path(root).exists():
        manifest = load_replay_tape_manifest(bundle_root=root)
        _validate_manifest_contract(
            manifest=manifest,
            env_name=env_name,
            env_params=env_params,
            gamma=gamma,
            sampling_distribution="uniform_sa",
        )
        return load_replay_tape_selection(
            bundle_root=root,
            seed_values=seed_values,
            tape_steps=tape_steps,
        )
    bundle = generate_uniform_sa_replay_tape(
        mdp_P=mdp_P,
        mdp_R=mdp_R,
        seed_values=seed_values,
        tape_steps=tape_steps,
        env_name=env_name,
        env_params=env_params,
        gamma=gamma,
        sampling_distribution="uniform_sa",
        contract_scope=contract_scope,
    )
    persist_replay_tape(
        bundle_root=root,
        manifest=bundle.manifest,
        states=bundle.states,
        actions=bundle.actions,
        next_states=bundle.next_states,
    )
    return bundle


__all__ = [
    "DEFAULT_REPLAY_TAPE_CONTRACT_SCOPE",
    "ReplayTapeBundle",
    "ReplayTapeManifest",
    "compute_replay_tape_checksum",
    "ensure_uniform_sa_replay_tape",
    "generate_uniform_sa_replay_tape",
    "load_replay_tape_bundle",
    "load_replay_tape_manifest",
    "normalize_replay_tape_contract_scope",
    "persist_replay_tape",
    "replay_tape_contract_env_params",
    "select_replay_tape_bundle",
]
