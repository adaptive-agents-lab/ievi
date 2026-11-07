"""Canonical replay-bank selection for maintained stochastic-control runs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...core.artifact_io import save_json_payload
from ...core.config import canonical_algorithm_name
from .replay_tape import (
    DEFAULT_REPLAY_TAPE_CONTRACT_SCOPE,
    ReplayTapeBundle,
    ReplayTapeManifest,
    _extend_uniform_sa_row_from_prefix,
    _generate_uniform_sa_row,
    compute_replay_tape_checksum,
    ensure_uniform_sa_replay_tape,
    load_replay_tape_manifest,
    load_replay_tape_selection,
    normalize_replay_tape_contract_scope,
    persist_replay_tape,
    replay_tape_contract_env_params,
)


CANONICAL_REPLAY_BANK_SEED_VALUES = tuple(range(20, 40))
CANONICAL_REPLAY_BANK_STEPS = 1_000_000


@dataclass(frozen=True)
class ReplayBankSelection:
    bank_root: Path
    bank_manifest_path: Path
    bank_manifest: ReplayTapeManifest
    selected_bundle: ReplayTapeBundle
    compatibility_payload: dict[str, Any] | None = None
    compatibility_certification_path: Path | None = None


def canonical_replay_bank_bundle_root(
    *,
    replay_root: Path | str,
    env_name: str,
    env_params: Mapping[str, Any] | None,
    sampling_distribution: str = "uniform_sa",
    contract_scope: str = DEFAULT_REPLAY_TAPE_CONTRACT_SCOPE,
    bank_seed_values: Sequence[int] = CANONICAL_REPLAY_BANK_SEED_VALUES,
    bank_steps: int = CANONICAL_REPLAY_BANK_STEPS,
) -> Path:
    contract_env_params = replay_tape_contract_env_params(
        env_params,
        contract_scope=contract_scope,
    )
    params_json = json.dumps(contract_env_params, sort_keys=True, separators=(",", ":"))
    params_hash = hashlib.sha1(params_json.encode("utf-8")).hexdigest()[:12]
    params_token = "no_env_params" if not contract_env_params else f"env_{params_hash}"
    seed_token = "seeds_" + "_".join(str(int(value)) for value in bank_seed_values)
    return (
        Path(replay_root)
        / canonical_algorithm_name(env_name)
        / str(contract_scope)
        / str(sampling_distribution).strip().lower().replace("-", "_")
        / params_token
        / seed_token
        / f"bank_{int(bank_steps)}"
    )


def _certification_path(
    *,
    bank_root: Path,
    reference_manifest_path: Path,
    seed_values: Sequence[int],
    tape_steps: int,
) -> Path:
    payload = {
        "reference_manifest_path": str(reference_manifest_path.resolve()),
        "seed_values": [int(value) for value in seed_values],
        "tape_steps": int(tape_steps),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return bank_root / "certifications" / f"{digest}.json"


def certify_replay_bank_prefix_compatibility(
    *,
    bank_root: Path | str,
    reference_manifest_path: Path | str,
    seed_values: Sequence[int],
    tape_steps: int,
) -> tuple[dict[str, Any], Path]:
    resolved_bank_root = Path(bank_root).expanduser().resolve()
    resolved_reference_manifest = Path(reference_manifest_path).expanduser().resolve()
    reference_root = (
        resolved_reference_manifest.parent
        if resolved_reference_manifest.name == "manifest.json"
        else resolved_reference_manifest
    )
    bank_selection = load_replay_tape_selection(
        bundle_root=resolved_bank_root,
        seed_values=seed_values,
        tape_steps=tape_steps,
    )
    reference_selection = load_replay_tape_selection(
        bundle_root=reference_root,
        seed_values=seed_values,
        tape_steps=tape_steps,
    )
    if (
        not np.array_equal(bank_selection.states, reference_selection.states)
        or not np.array_equal(bank_selection.actions, reference_selection.actions)
        or not np.array_equal(bank_selection.next_states, reference_selection.next_states)
    ):
        raise ValueError(
            "Canonical replay bank does not exactly match the requested certified prefix reference."
        )
    certification_path = _certification_path(
        bank_root=resolved_bank_root,
        reference_manifest_path=resolved_reference_manifest,
        seed_values=seed_values,
        tape_steps=tape_steps,
    )
    payload = {
        "status": "exact_match",
        "bank_manifest_path": str((resolved_bank_root / "manifest.json").resolve()),
        "reference_manifest_path": str(resolved_reference_manifest),
        "seed_values": [int(value) for value in seed_values],
        "tape_steps": int(tape_steps),
        "bank_tape_checksum": str(bank_selection.manifest.tape_checksum),
        "reference_tape_checksum": str(reference_selection.manifest.tape_checksum),
    }
    certification_path.parent.mkdir(parents=True, exist_ok=True)
    save_json_payload(certification_path, payload)
    return payload, certification_path


def _validate_bank_manifest(
    *,
    bank_manifest: ReplayTapeManifest,
    env_name: str,
    env_params: Mapping[str, Any] | None,
    sampling_distribution: str,
    contract_scope: str,
    bank_seed_values: Sequence[int],
    bank_steps: int,
) -> None:
    requested_env_params = replay_tape_contract_env_params(
        env_params,
        contract_scope=contract_scope,
    )
    if bank_manifest.env_name != str(env_name):
        raise ValueError("Existing replay bank env_name does not match the requested transition-law identity.")
    if replay_tape_contract_env_params(
        bank_manifest.env_params,
        contract_scope=contract_scope,
    ) != requested_env_params:
        raise ValueError("Existing replay bank env_params do not match the requested transition-law identity.")
    if str(bank_manifest.sampling_distribution) != str(sampling_distribution):
        raise ValueError("Existing replay bank sampling_distribution does not match the requested contract.")
    if str(bank_manifest.contract_scope) != str(contract_scope):
        raise ValueError("Existing replay bank contract_scope does not match the requested contract.")
    if tuple(int(value) for value in bank_manifest.seed_values) != tuple(int(value) for value in bank_seed_values):
        raise ValueError("Existing replay bank seed bank does not match the canonical bank seed values.")
    if int(bank_manifest.tape_steps) != int(bank_steps):
        raise ValueError("Existing replay bank horizon does not match the canonical bank horizon.")


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _materialize_uniform_sa_replay_bank_from_reference_prefix(
    *,
    bundle_root: Path | str,
    mdp_P: np.ndarray,
    mdp_R: np.ndarray,
    env_name: str,
    env_params: Mapping[str, Any] | None,
    gamma: float,
    bank_seed_values: Sequence[int],
    bank_steps: int,
    contract_scope: str,
    reference_manifest_path: Path | str,
    reference_seed_values: Sequence[int],
    reference_tape_steps: int,
) -> ReplayTapeBundle:
    P = np.asarray(mdp_P, dtype=np.float64)
    R = np.asarray(mdp_R, dtype=np.float64)
    del R
    n_s, n_a, _ = P.shape
    resolved_seed_values = tuple(int(value) for value in bank_seed_values)
    resolved_bank_steps = int(bank_steps)
    resolved_reference_steps = int(reference_tape_steps)
    if resolved_reference_steps > resolved_bank_steps:
        raise ValueError("reference_tape_steps cannot exceed bank_steps.")

    reference_root = (
        Path(reference_manifest_path).expanduser().resolve().parent
        if Path(reference_manifest_path).expanduser().resolve().name == "manifest.json"
        else Path(reference_manifest_path).expanduser().resolve()
    )
    reference_bundle = load_replay_tape_selection(
        bundle_root=reference_root,
        seed_values=reference_seed_values,
        tape_steps=resolved_reference_steps,
    )
    reference_lookup = {
        int(seed): index for index, seed in enumerate(reference_bundle.manifest.seed_values)
    }

    states = np.empty((len(resolved_seed_values), resolved_bank_steps), dtype=np.int32)
    actions = np.empty((len(resolved_seed_values), resolved_bank_steps), dtype=np.int32)
    next_states = np.empty((len(resolved_seed_values), resolved_bank_steps), dtype=np.int32)
    for row_index, seed_value in enumerate(resolved_seed_values):
        if int(seed_value) in reference_lookup:
            reference_index = int(reference_lookup[int(seed_value)])
            row_states, row_actions, row_next_states = _extend_uniform_sa_row_from_prefix(
                mdp_P=P,
                seed_value=int(seed_value),
                total_steps=resolved_bank_steps,
                n_states=n_s,
                n_actions=n_a,
                prefix_states=reference_bundle.states[reference_index],
                prefix_actions=reference_bundle.actions[reference_index],
                prefix_next_states=reference_bundle.next_states[reference_index],
            )
        else:
            row_states, row_actions, row_next_states = _generate_uniform_sa_row(
                mdp_P=P,
                seed_value=int(seed_value),
                tape_steps=resolved_bank_steps,
                n_states=n_s,
                n_actions=n_a,
            )
        states[row_index] = row_states
        actions[row_index] = row_actions
        next_states[row_index] = row_next_states

    resolved_contract_scope = normalize_replay_tape_contract_scope(contract_scope)
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
        tape_steps=resolved_bank_steps,
        storage_shape=states.shape,
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
    persist_replay_tape(
        bundle_root=bundle_root,
        manifest=manifest,
        states=states,
        actions=actions,
        next_states=next_states,
    )
    return load_replay_tape_selection(
        bundle_root=bundle_root,
        seed_values=resolved_seed_values,
        tape_steps=resolved_bank_steps,
    )


def ensure_transition_law_replay_bank_selection(
    *,
    replay_root: Path | str,
    env_name: str,
    env_params: Mapping[str, Any] | None,
    gamma: float,
    mdp_P: np.ndarray,
    mdp_R: np.ndarray,
    requested_seed_values: Sequence[int],
    requested_tape_steps: int,
    sampling_distribution: str = "uniform_sa",
    contract_scope: str = DEFAULT_REPLAY_TAPE_CONTRACT_SCOPE,
    bank_seed_values: Sequence[int] = CANONICAL_REPLAY_BANK_SEED_VALUES,
    bank_steps: int = CANONICAL_REPLAY_BANK_STEPS,
    reference_manifest_path: Path | str | None = None,
    reference_seed_values: Sequence[int] | None = None,
    reference_tape_steps: int | None = None,
) -> ReplayBankSelection:
    if str(sampling_distribution).strip().lower().replace("-", "_") != "uniform_sa":
        raise ValueError("Canonical replay-bank selection currently supports only sampling_distribution='uniform_sa'.")
    bank_root = canonical_replay_bank_bundle_root(
        replay_root=replay_root,
        env_name=env_name,
        env_params=env_params,
        sampling_distribution=sampling_distribution,
        contract_scope=contract_scope,
        bank_seed_values=bank_seed_values,
        bank_steps=bank_steps,
    ).resolve()
    bank_manifest_path = bank_root / "manifest.json"
    if not bank_manifest_path.exists():
        if reference_manifest_path is not None:
            _materialize_uniform_sa_replay_bank_from_reference_prefix(
                bundle_root=bank_root,
                mdp_P=mdp_P,
                mdp_R=mdp_R,
                env_name=env_name,
                env_params=env_params,
                gamma=float(gamma),
                bank_seed_values=bank_seed_values,
                bank_steps=bank_steps,
                contract_scope=contract_scope,
                reference_manifest_path=reference_manifest_path,
                reference_seed_values=(
                    requested_seed_values
                    if reference_seed_values is None
                    else reference_seed_values
                ),
                reference_tape_steps=(
                    int(requested_tape_steps)
                    if reference_tape_steps is None
                    else int(reference_tape_steps)
                ),
            )
        else:
            ensure_uniform_sa_replay_tape(
                bundle_root=bank_root,
                mdp_P=mdp_P,
                mdp_R=mdp_R,
                env_name=env_name,
                env_params=env_params,
                gamma=float(gamma),
                seed_values=bank_seed_values,
                tape_steps=bank_steps,
                contract_scope=contract_scope,
            )
    bank_manifest = load_replay_tape_manifest(bundle_root=bank_root)
    _validate_bank_manifest(
        bank_manifest=bank_manifest,
        env_name=env_name,
        env_params=env_params,
        sampling_distribution=sampling_distribution,
        contract_scope=contract_scope,
        bank_seed_values=bank_seed_values,
        bank_steps=bank_steps,
    )
    selected_bundle = load_replay_tape_selection(
        bundle_root=bank_root,
        seed_values=requested_seed_values,
        tape_steps=requested_tape_steps,
    )
    compatibility_payload = None
    compatibility_certification_path = None
    if reference_manifest_path is not None:
        compatibility_payload, compatibility_certification_path = certify_replay_bank_prefix_compatibility(
            bank_root=bank_root,
            reference_manifest_path=reference_manifest_path,
            seed_values=(
                requested_seed_values
                if reference_seed_values is None
                else [int(value) for value in reference_seed_values]
            ),
            tape_steps=(
                int(requested_tape_steps)
                if reference_tape_steps is None
                else int(reference_tape_steps)
            ),
        )
    return ReplayBankSelection(
        bank_root=bank_root,
        bank_manifest_path=bank_manifest_path,
        bank_manifest=bank_manifest,
        selected_bundle=selected_bundle,
        compatibility_payload=compatibility_payload,
        compatibility_certification_path=compatibility_certification_path,
    )


__all__ = [
    "CANONICAL_REPLAY_BANK_SEED_VALUES",
    "CANONICAL_REPLAY_BANK_STEPS",
    "ReplayBankSelection",
    "canonical_replay_bank_bundle_root",
    "certify_replay_bank_prefix_compatibility",
    "ensure_transition_law_replay_bank_selection",
]
