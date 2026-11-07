"""Explicit environment-variant expansion for maintained deterministic configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json

import yaml


@dataclass(frozen=True)
class ResolvedEnvVariant:
    variant_id: str
    env_name: str
    env_params: Mapping[str, Any]
    deterministic_target_manifest: str | None
    metadata: Mapping[str, Any]


def _decode_manifest(path: Path) -> Mapping[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    try:
        decoded = yaml.safe_load(raw_text)
    except Exception:
        decoded = json.loads(raw_text)
    if not isinstance(decoded, Mapping):
        raise ValueError(f"Environment-variant manifest at {path} did not decode to a mapping.")
    return decoded


def _merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(dict(override))
    return merged


def load_env_variants_manifest(
    manifest_path: str | Path,
    *,
    default_env_name: str,
    base_env_params: Mapping[str, Any],
    default_target_manifest: str | None = None,
) -> list[ResolvedEnvVariant]:
    path = Path(manifest_path)
    payload = _decode_manifest(path)
    raw_variants = payload.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError(
            f"Environment-variant manifest at {path} must contain a non-empty 'variants' list."
        )

    manifest_default_env_name = payload.get("env_name")
    manifest_default_env_name = (
        str(manifest_default_env_name) if isinstance(manifest_default_env_name, str) else default_env_name
    )
    manifest_default_env_params = payload.get("env_params", {})
    if not isinstance(manifest_default_env_params, Mapping):
        raise ValueError(f"Environment-variant manifest env_params must be a mapping: {path}")
    manifest_default_target = payload.get("deterministic_target_manifest")
    manifest_default_target = (
        str(manifest_default_target)
        if isinstance(manifest_default_target, str) and manifest_default_target
        else default_target_manifest
    )
    manifest_metadata = payload.get("metadata", {})
    if not isinstance(manifest_metadata, Mapping):
        raise ValueError(f"Environment-variant manifest metadata must be a mapping: {path}")

    resolved: list[ResolvedEnvVariant] = []
    for index, raw_variant in enumerate(raw_variants):
        if not isinstance(raw_variant, Mapping):
            raise ValueError(
                f"Environment-variant manifest variants[{index}] must be a mapping: {path}"
            )
        variant_env_name_raw = raw_variant.get("env_name")
        variant_env_name = (
            str(variant_env_name_raw)
            if isinstance(variant_env_name_raw, str) and variant_env_name_raw
            else manifest_default_env_name
        )

        variant_env_params = raw_variant.get("env_params", {})
        if not isinstance(variant_env_params, Mapping):
            raise ValueError(
                f"Environment-variant manifest variants[{index}].env_params must be a mapping: {path}"
            )
        env_params = _merge_mappings(
            _merge_mappings(base_env_params, manifest_default_env_params),
            variant_env_params,
        )

        variant_target_manifest = raw_variant.get("deterministic_target_manifest")
        if variant_target_manifest in (None, ""):
            deterministic_target_manifest = manifest_default_target
        else:
            deterministic_target_manifest = str(variant_target_manifest)

        variant_metadata = raw_variant.get("metadata", {})
        if not isinstance(variant_metadata, Mapping):
            raise ValueError(
                f"Environment-variant manifest variants[{index}].metadata must be a mapping: {path}"
            )
        variant_id_raw = raw_variant.get("variant_id")
        variant_id = (
            str(variant_id_raw)
            if isinstance(variant_id_raw, str) and variant_id_raw
            else f"variant_{index:03d}"
        )
        metadata = _merge_mappings(dict(manifest_metadata), dict(variant_metadata))
        metadata.setdefault("variant_id", variant_id)
        metadata.setdefault("env_name", variant_env_name)
        metadata.setdefault("env_params", dict(env_params))
        if deterministic_target_manifest:
            metadata.setdefault("deterministic_target_manifest", deterministic_target_manifest)

        resolved.append(
            ResolvedEnvVariant(
                variant_id=variant_id,
                env_name=variant_env_name,
                env_params=env_params,
                deterministic_target_manifest=deterministic_target_manifest,
                metadata=metadata,
            )
        )

    return resolved


__all__ = ["ResolvedEnvVariant", "load_env_variants_manifest"]
