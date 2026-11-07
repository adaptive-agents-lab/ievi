"""Shared offline policy capture and exact evaluation helpers."""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..algorithms.jax.base import (
    evaluate_policy_start_values_batched_pure,
    evaluate_policy_vectors_batched_pure,
)
from ..core.artifact_io import save_json_payload
from ..core.types import OfflineEvalManifest, OfflineEvalPlan


def _bundle_path(bundle_root: Path | str, relative_path: str) -> Path:
    return Path(bundle_root) / relative_path


def _policy_steps_relative_path() -> str:
    return "policy_steps.npy"


def _evaluated_mask_relative_path() -> str:
    return "evaluated_mask.npy"


def _summary_relative_path() -> str:
    return "summary.json"


def _manifest_relative_path() -> str:
    return "manifest.json"


def _phase_manifest_relative_path() -> str:
    return "phase_manifest.json"


def _stream_checkpoint_relative_path() -> str:
    return "stream_checkpoint.pkl"


def _policy_slab_relative_path(slab_index: int) -> str:
    return f"policy_slabs/slab_{slab_index:03d}.npy"


def _value_slab_relative_path(slab_index: int) -> str:
    return f"value_slabs/slab_{slab_index:03d}.npy"


def _start_state_values_relative_path() -> str:
    return "start_state_values.npy"


def write_offline_eval_manifest(
    *,
    bundle_root: Path | str,
    manifest: OfflineEvalManifest,
) -> None:
    save_json_payload(_bundle_path(bundle_root, _manifest_relative_path()), manifest.as_mapping())


def load_offline_eval_manifest(*, bundle_root: Path | str) -> OfflineEvalManifest:
    with _bundle_path(bundle_root, _manifest_relative_path()).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return OfflineEvalManifest.from_mapping(payload)


def write_offline_eval_phase_manifest(
    *,
    bundle_root: Path | str,
    payload: Mapping[str, Any],
) -> None:
    save_json_payload(_bundle_path(bundle_root, _phase_manifest_relative_path()), dict(payload))


def save_stream_checkpoint(
    *,
    bundle_root: Path | str,
    manifest: OfflineEvalManifest,
    state: Any,
    current_sample: int,
    completed_slabs: Sequence[int],
) -> OfflineEvalManifest:
    resolved_completed = tuple(sorted({int(value) for value in completed_slabs}))
    checkpoint_payload = {
        "state": jax.device_get(state),
        "current_sample": int(current_sample),
        "completed_slabs": resolved_completed,
    }
    with _bundle_path(bundle_root, _stream_checkpoint_relative_path()).open("wb") as handle:
        pickle.dump(checkpoint_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    updated_manifest = replace(manifest, completed_slabs=resolved_completed)
    write_offline_eval_manifest(bundle_root=bundle_root, manifest=updated_manifest)
    return updated_manifest


def load_stream_checkpoint(*, bundle_root: Path | str) -> dict[str, Any] | None:
    checkpoint_path = _bundle_path(bundle_root, _stream_checkpoint_relative_path())
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open("rb") as handle:
        return pickle.load(handle)


def clear_stream_checkpoint(*, bundle_root: Path | str) -> None:
    checkpoint_path = _bundle_path(bundle_root, _stream_checkpoint_relative_path())
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def persist_policy_capture_bundle(
    *,
    bundle_root: Path | str,
    algorithm: str,
    problem_type: str,
    lambda_value: float | None,
    seed: int | None,
    n_seeds: int,
    plan: OfflineEvalPlan,
    sample_axis: Sequence[int],
    policy_history: np.ndarray,
    value_storage_mode: str = "full_vectors",
    config_labels: Sequence[str] | None = None,
    seed_values: Sequence[int] | None = None,
    sampling_contract: Mapping[str, Any] | None = None,
) -> OfflineEvalManifest:
    root = Path(bundle_root)
    policy_array = np.asarray(policy_history, dtype=np.uint8)
    if policy_array.shape != (plan.total_capture_steps, plan.env_batch_size, plan.n_states):
        raise ValueError(
            "policy_history shape must match "
            f"({plan.total_capture_steps}, {plan.env_batch_size}, {plan.n_states}), "
            f"received {policy_array.shape}."
        )
    manifest = initialize_policy_capture_bundle(
        bundle_root=root,
        algorithm=algorithm,
        problem_type=problem_type,
        lambda_value=lambda_value,
        seed=seed,
        n_seeds=n_seeds,
        plan=plan,
        sample_axis=sample_axis,
        value_storage_mode=value_storage_mode,
        config_labels=config_labels,
        seed_values=seed_values,
        sampling_contract=sampling_contract,
    )
    for slab in plan.slabs:
        write_policy_capture_slab(
            bundle_root=root,
            manifest=manifest,
            slab_index=slab.slab_index,
            policy_slab=policy_array[slab.capture_start : slab.capture_stop],
        )
    save_json_payload(
        root / _summary_relative_path(),
        {
            "phase": "captured",
            "algorithm": algorithm,
            "problem_type": problem_type,
            "lambda_value": lambda_value,
            "seed": seed,
            "n_seeds": n_seeds,
            "plan": plan.as_mapping(),
            "value_storage_mode": manifest.value_storage_mode,
            "sampling_contract": (
                dict(sampling_contract)
                if sampling_contract is not None
                else None
            ),
        },
    )
    return manifest


def initialize_policy_capture_bundle(
    *,
    bundle_root: Path | str,
    algorithm: str,
    problem_type: str,
    lambda_value: float | None,
    seed: int | None,
    n_seeds: int,
    plan: OfflineEvalPlan,
    sample_axis: Sequence[int],
    value_storage_mode: str = "full_vectors",
    config_labels: Sequence[str] | None = None,
    seed_values: Sequence[int] | None = None,
    sampling_contract: Mapping[str, Any] | None = None,
) -> OfflineEvalManifest:
    resolved_storage_mode = str(value_storage_mode).strip().lower().replace("-", "_")
    if resolved_storage_mode not in {"full_vectors", "start_state_only"}:
        raise ValueError("value_storage_mode must be one of: full_vectors, start_state_only.")
    root = Path(bundle_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "policy_slabs").mkdir(parents=True, exist_ok=True)
    if resolved_storage_mode == "full_vectors":
        (root / "value_slabs").mkdir(parents=True, exist_ok=True)
    resolved_config_labels = tuple(str(value) for value in (config_labels or ()))
    resolved_seed_values = tuple(int(value) for value in (seed_values or tuple(range(int(n_seeds)))))
    resolved_config_count = (
        len(resolved_config_labels)
        if resolved_config_labels
        else max(1, plan.env_batch_size // max(1, int(n_seeds)))
    )
    if resolved_config_count * int(n_seeds) != plan.env_batch_size:
        raise ValueError(
            "config_count * n_seeds must match plan.env_batch_size; "
            f"received config_count={resolved_config_count}, n_seeds={n_seeds}, "
            f"env_batch_size={plan.env_batch_size}."
        )
    if len(resolved_seed_values) != int(n_seeds):
        raise ValueError(
            f"seed_values must contain exactly {n_seeds} entries, received {len(resolved_seed_values)}."
        )
    if resolved_config_labels and len(resolved_config_labels) != resolved_config_count:
        raise ValueError(
            "config_labels length must match resolved config_count; "
            f"received {len(resolved_config_labels)} and {resolved_config_count}."
        )
    storage_layout_version = 2 if resolved_storage_mode == "start_state_only" else 1
    storage_shape = (
        (plan.total_capture_steps, resolved_config_count, int(n_seeds))
        if storage_layout_version >= 2 and resolved_storage_mode == "start_state_only"
        else (
            (plan.total_capture_steps, plan.env_batch_size, plan.n_states)
            if resolved_storage_mode == "full_vectors"
            else (plan.total_capture_steps, plan.env_batch_size)
        )
    )

    policy_steps = np.asarray(tuple(int(value) for value in sample_axis), dtype=np.int32)
    if policy_steps.shape != (plan.total_capture_steps,):
        raise ValueError("sample_axis length must match plan.total_capture_steps")

    np.save(root / _policy_steps_relative_path(), policy_steps, allow_pickle=False)
    np.save(
        root / _evaluated_mask_relative_path(),
        np.zeros((plan.total_capture_steps,), dtype=np.bool_),
        allow_pickle=False,
    )

    manifest = OfflineEvalManifest(
        evaluation_pipeline="train_then_offline",
        algorithm=algorithm,
        problem_type=problem_type,
        lambda_value=lambda_value,
        seed=seed,
        n_seeds=n_seeds,
        plan=plan,
        value_storage_mode=resolved_storage_mode,
        policy_steps_path=_policy_steps_relative_path(),
        evaluated_mask_path=_evaluated_mask_relative_path(),
        policy_slab_paths=tuple(_policy_slab_relative_path(slab.slab_index) for slab in plan.slabs),
        value_slab_paths=(
            tuple(_value_slab_relative_path(slab.slab_index) for slab in plan.slabs)
            if resolved_storage_mode == "full_vectors"
            else ()
        ),
        start_state_values_path=(
            _start_state_values_relative_path()
            if resolved_storage_mode == "start_state_only"
            else None
        ),
        summary_path=_summary_relative_path(),
        config_labels=resolved_config_labels,
        seed_values=resolved_seed_values,
        config_count=resolved_config_count,
        storage_layout_version=storage_layout_version,
        storage_shape=storage_shape,
        completed_slabs=(),
        sampling_contract=(None if sampling_contract is None else dict(sampling_contract)),
    )
    if manifest.start_state_values_path is not None:
        np.lib.format.open_memmap(
            _bundle_path(root, manifest.start_state_values_path),
            mode="w+",
            dtype=np.float64,
            shape=manifest.resolved_storage_shape,
        )
    save_json_payload(root / _manifest_relative_path(), manifest.as_mapping())
    return manifest


def write_policy_capture_slab(
    *,
    bundle_root: Path | str,
    manifest: OfflineEvalManifest,
    slab_index: int,
    policy_slab: np.ndarray,
) -> None:
    slab = manifest.plan.slabs[slab_index]
    policy_array = np.asarray(policy_slab, dtype=np.uint8)
    expected_shape = (slab.capture_count, manifest.plan.env_batch_size, manifest.plan.n_states)
    if policy_array.shape != expected_shape:
        raise ValueError(f"policy_slab shape must be {expected_shape}, received {policy_array.shape}.")
    np.save(
        _bundle_path(bundle_root, manifest.policy_slab_paths[slab_index]),
        policy_array,
        allow_pickle=False,
    )


def _evaluate_policy_rows(
    *,
    mdp_P: jnp.ndarray,
    mdp_R: jnp.ndarray,
    gamma: float | jnp.ndarray,
    policy_rows: jax.Array,
    chunk_size: int,
) -> jax.Array:
    if policy_rows.shape[0] == 0:
        return jnp.empty((0, mdp_P.shape[-1]), dtype=jnp.float64)
    gamma_array = jnp.asarray(gamma, dtype=jnp.float64)
    if gamma_array.ndim == 0:
        gamma_vector = jnp.full((policy_rows.shape[0],), gamma_array, dtype=jnp.float64)
    else:
        if policy_rows.shape[0] % gamma_array.shape[0] != 0:
            raise ValueError(
                "Vector gamma payload must divide the flattened policy row count; "
                f"received {gamma_array.shape[0]} and {policy_rows.shape[0]}."
            )
        gamma_vector = jnp.tile(gamma_array, policy_rows.shape[0] // gamma_array.shape[0])
    outputs: list[jax.Array] = []
    for chunk_start in range(0, policy_rows.shape[0], chunk_size):
        chunk_stop = min(policy_rows.shape[0], chunk_start + chunk_size)
        outputs.append(
            evaluate_policy_vectors_batched_pure(
                mdp_P,
                mdp_R,
                policy_rows[chunk_start:chunk_stop],
                gamma_vector[chunk_start:chunk_stop],
            )
        )
    return jnp.concatenate(outputs, axis=0)


def _evaluate_policy_rows_start_state(
    *,
    mdp_P: jnp.ndarray,
    mdp_R: jnp.ndarray,
    gamma: float | jnp.ndarray,
    policy_rows: jax.Array,
    chunk_size: int,
    start_state_idx: int = 0,
) -> jax.Array:
    if policy_rows.shape[0] == 0:
        return jnp.empty((0,), dtype=jnp.float64)
    gamma_array = jnp.asarray(gamma, dtype=jnp.float64)
    if gamma_array.ndim == 0:
        gamma_vector = jnp.full((policy_rows.shape[0],), gamma_array, dtype=jnp.float64)
    else:
        if policy_rows.shape[0] % gamma_array.shape[0] != 0:
            raise ValueError(
                "Vector gamma payload must divide the flattened policy row count; "
                f"received {gamma_array.shape[0]} and {policy_rows.shape[0]}."
            )
        gamma_vector = jnp.tile(gamma_array, policy_rows.shape[0] // gamma_array.shape[0])
    outputs: list[jax.Array] = []
    for chunk_start in range(0, policy_rows.shape[0], chunk_size):
        chunk_stop = min(policy_rows.shape[0], chunk_start + chunk_size)
        outputs.append(
            evaluate_policy_start_values_batched_pure(
                mdp_P,
                mdp_R,
                policy_rows[chunk_start:chunk_stop],
                gamma_vector[chunk_start:chunk_stop],
                start_state_idx=start_state_idx,
            )
        )
    return jnp.concatenate(outputs, axis=0)


def evaluate_policy_capture_slab_start_state(
    *,
    policy_slab: jax.Array | np.ndarray,
    mdp_P: jnp.ndarray,
    mdp_R: jnp.ndarray,
    gamma: float | jnp.ndarray,
    chunk_size: int,
    start_state_idx: int = 0,
) -> jax.Array:
    slab = jnp.asarray(policy_slab, dtype=jnp.int32)
    if slab.ndim != 3:
        raise ValueError(
            "policy_slab must have shape (capture_count, env_batch_size, n_states); "
            f"received {slab.shape}."
        )
    flat_rows = slab.reshape((-1, slab.shape[-1]))
    start_state_values = _evaluate_policy_rows_start_state(
        mdp_P=mdp_P,
        mdp_R=mdp_R,
        gamma=gamma,
        policy_rows=flat_rows,
        chunk_size=chunk_size,
        start_state_idx=start_state_idx,
    )
    return jnp.reshape(start_state_values, (slab.shape[0], slab.shape[1]))


def extract_start_state_history_from_value_history(
    *,
    value_history: np.ndarray,
    start_state_idx: int = 0,
) -> np.ndarray:
    values = np.asarray(value_history, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("value_history must have shape (captures, envs, states).")
    return np.asarray(values[:, :, int(start_state_idx)], dtype=np.float64)


def write_start_state_history_slab(
    *,
    bundle_root: Path | str,
    manifest: OfflineEvalManifest,
    slab_index: int,
    start_state_slab: np.ndarray,
) -> None:
    if manifest.start_state_values_path is None:
        raise ValueError("Manifest does not define a start-state values artifact.")
    slab = manifest.plan.slabs[slab_index]
    start_state_array = np.asarray(start_state_slab, dtype=np.float64)
    if start_state_array.ndim == 2:
        start_state_array = start_state_array.reshape(
            slab.capture_count,
            manifest.resolved_config_count,
            int(manifest.n_seeds),
        )
    expected_shape = (
        slab.capture_count,
        manifest.resolved_config_count,
        int(manifest.n_seeds),
    )
    if start_state_array.shape != expected_shape:
        raise ValueError(
            f"start_state_slab shape must be {expected_shape}, received {start_state_array.shape}."
        )
    target = np.lib.format.open_memmap(
        _bundle_path(bundle_root, manifest.start_state_values_path),
        mode="r+",
        dtype=np.float64,
        shape=manifest.resolved_storage_shape,
    )
    target[slab.capture_start : slab.capture_stop] = start_state_array
    target.flush()


def evaluate_offline_policy_bundle(
    *,
    bundle_root: Path | str,
    manifest: OfflineEvalManifest,
    mdp_P: jnp.ndarray,
    mdp_R: jnp.ndarray,
    gamma: float | jnp.ndarray,
    mode: str,
    resident_policy_slabs: Sequence[jax.Array] | None = None,
) -> dict[str, Any]:
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"resident", "reload"}:
        raise ValueError("mode must be one of: resident, reload.")
    root = Path(bundle_root)
    evaluated_mask = np.zeros((manifest.plan.total_capture_steps,), dtype=np.bool_)
    slab_summaries: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    completed_slabs: list[int] = list(int(value) for value in manifest.completed_slabs)

    for slab in manifest.plan.slabs:
        slab_started = time.perf_counter()
        if resident_policy_slabs is not None and resolved_mode == "resident":
            host_policy_slab = None
            policy_slab = resident_policy_slabs[slab.slab_index]
        else:
            host_policy_slab = np.load(
                _bundle_path(root, manifest.policy_slab_paths[slab.slab_index]),
                mmap_mode=None,
            )
            policy_slab = jax.device_put(np.asarray(host_policy_slab, dtype=np.uint8))
        flat_policy_rows = jnp.reshape(policy_slab, (-1, manifest.plan.n_states))
        if manifest.value_storage_mode == "full_vectors":
            values = _evaluate_policy_rows(
                mdp_P=mdp_P,
                mdp_R=mdp_R,
                gamma=gamma,
                policy_rows=flat_policy_rows,
                chunk_size=manifest.plan.solve_chunk_size,
            )
            host_values = np.asarray(
                jax.device_get(
                    jnp.reshape(
                        values,
                        (slab.capture_count, manifest.plan.env_batch_size, manifest.plan.n_states),
                    )
                ),
                dtype=np.float64,
            )
            np.save(
                _bundle_path(root, manifest.value_slab_paths[slab.slab_index]),
                host_values,
                allow_pickle=False,
            )
            del values
            del host_values
        elif manifest.value_storage_mode == "start_state_only":
            start_state_values = _evaluate_policy_rows_start_state(
                mdp_P=mdp_P,
                mdp_R=mdp_R,
                gamma=gamma,
                policy_rows=flat_policy_rows,
                chunk_size=manifest.plan.solve_chunk_size,
                start_state_idx=0,
            )
            write_start_state_history_slab(
                bundle_root=root,
                manifest=manifest,
                slab_index=slab.slab_index,
                start_state_slab=np.asarray(
                    jax.device_get(
                        jnp.reshape(
                            start_state_values,
                            (slab.capture_count, manifest.plan.env_batch_size),
                        )
                    ),
                    dtype=np.float64,
                ),
            )
            del start_state_values
        else:
            raise ValueError(
                "Unsupported manifest.value_storage_mode: "
                f"{manifest.value_storage_mode}"
            )
        evaluated_mask[slab.capture_start : slab.capture_stop] = True
        completed_slabs.append(int(slab.slab_index))
        slab_summaries.append(
            {
                "slab_index": slab.slab_index,
                "capture_count": slab.capture_count,
                "row_count": slab.row_count,
                "seconds": time.perf_counter() - slab_started,
            }
        )
        if host_policy_slab is not None:
            del host_policy_slab

    np.save(_bundle_path(root, manifest.evaluated_mask_path), evaluated_mask, allow_pickle=False)
    summary = {
        "phase": "evaluated",
        "mode": resolved_mode,
        "value_storage_mode": manifest.value_storage_mode,
        "total_seconds": time.perf_counter() - total_started,
        "slabs": slab_summaries,
        "completed_slabs": sorted(set(completed_slabs)),
        "sampling_contract": (
            dict(manifest.sampling_contract)
            if manifest.sampling_contract is not None
            else None
        ),
    }
    save_json_payload(_bundle_path(root, manifest.summary_path), summary)
    return summary


def load_value_slabs(*, bundle_root: Path | str, manifest: OfflineEvalManifest) -> np.ndarray:
    if manifest.value_storage_mode != "full_vectors":
        raise ValueError("Full value slabs are not available for this manifest.")
    root = Path(bundle_root)
    slabs = [
        np.load(_bundle_path(root, relative_path), mmap_mode=None)
        for relative_path in manifest.value_slab_paths
    ]
    return np.concatenate(slabs, axis=0)


def load_start_state_history(*, bundle_root: Path | str, manifest: OfflineEvalManifest) -> np.ndarray:
    if manifest.start_state_values_path is None:
        raise ValueError("Manifest does not define a start-state values artifact.")
    values = np.asarray(
        np.load(_bundle_path(bundle_root, manifest.start_state_values_path), mmap_mode=None),
        dtype=np.float64,
    )
    if values.ndim == 3:
        return values.reshape(values.shape[0], values.shape[1] * values.shape[2])
    return values


def load_policy_steps(*, bundle_root: Path | str, manifest: OfflineEvalManifest) -> np.ndarray:
    return np.asarray(
        np.load(_bundle_path(bundle_root, manifest.policy_steps_path), mmap_mode=None),
        dtype=np.int32,
    )


def load_dense_start_state_history(*, bundle_root: Path | str, manifest: OfflineEvalManifest) -> np.ndarray:
    values = np.asarray(
        np.load(_bundle_path(bundle_root, manifest.start_state_values_path), mmap_mode=None),
        dtype=np.float64,
    )
    if values.ndim == 3:
        return values
    if values.ndim != 2:
        raise ValueError("start_state_values must have either 2 or 3 dimensions.")
    if values.shape[1] != manifest.plan.env_batch_size:
        raise ValueError(
            "Legacy dense history second axis must equal env_batch_size; "
            f"received {values.shape[1]} and {manifest.plan.env_batch_size}."
        )
    return values.reshape(
        manifest.plan.total_capture_steps,
        manifest.resolved_config_count,
        int(manifest.n_seeds),
    )


def load_config_start_state_history(
    *,
    bundle_root: Path | str,
    manifest: OfflineEvalManifest,
    config_label: str,
) -> np.ndarray:
    dense_values = load_dense_start_state_history(bundle_root=bundle_root, manifest=manifest)
    labels = list(manifest.config_labels)
    if not labels:
        raise ValueError("Manifest does not define config_labels.")
    try:
        config_index = labels.index(str(config_label))
    except ValueError as exc:
        raise KeyError(f"Unknown config_label: {config_label}") from exc
    return np.asarray(dense_values[:, config_index, :], dtype=np.float64)


def materialize_start_state_trace_from_history(
    *,
    start_state_history: np.ndarray,
    sample_axis: Sequence[int],
    eval_freq: int,
) -> Mapping[str, np.ndarray]:
    values = np.asarray(start_state_history, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("start_state_history must have shape (captures, envs).")
    sample_values = np.asarray(tuple(int(value) for value in sample_axis), dtype=np.int32)
    selected_indices: list[int] = []
    for index, sample in enumerate(sample_values.tolist()):
        is_final = index == len(sample_values) - 1
        if sample == 0 or sample % int(eval_freq) == 0 or is_final:
            if not selected_indices or selected_indices[-1] != index:
                selected_indices.append(index)
    selected = values[selected_indices]
    return {
        "x_axis": sample_values[selected_indices],
        "seeds": selected,
        "mean": np.mean(selected, axis=1),
        "std": np.std(selected, axis=1),
    }


def materialize_start_state_trace_from_values(
    *,
    value_history: np.ndarray,
    sample_axis: Sequence[int],
    eval_freq: int,
    start_state_idx: int = 0,
) -> Mapping[str, np.ndarray]:
    return materialize_start_state_trace_from_history(
        start_state_history=extract_start_state_history_from_value_history(
            value_history=value_history,
            start_state_idx=start_state_idx,
        ),
        sample_axis=sample_axis,
        eval_freq=eval_freq,
    )


__all__ = [
    "clear_stream_checkpoint",
    "evaluate_offline_policy_bundle",
    "evaluate_policy_capture_slab_start_state",
    "extract_start_state_history_from_value_history",
    "initialize_policy_capture_bundle",
    "load_offline_eval_manifest",
    "load_config_start_state_history",
    "load_dense_start_state_history",
    "load_policy_steps",
    "load_start_state_history",
    "load_stream_checkpoint",
    "load_value_slabs",
    "materialize_start_state_trace_from_history",
    "materialize_start_state_trace_from_values",
    "persist_policy_capture_bundle",
    "save_stream_checkpoint",
    "write_offline_eval_manifest",
    "write_offline_eval_phase_manifest",
    "write_start_state_history_slab",
    "write_policy_capture_slab",
]
