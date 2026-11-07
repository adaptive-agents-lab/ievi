"""Runner-local helpers for deterministic JAX execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..core.types import MDP
from ..model_error_runtime import apply_model_error_for_task


def _as_jax(array: Any, *, dtype: Any = jnp.float64) -> jnp.ndarray:
    return jnp.asarray(array, dtype=dtype)


def _as_numpy(array: Any, *, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(array, dtype=dtype)


@dataclass(frozen=True)
class LoadedDeterministicTarget:
    target_id: str
    manifest_path: Path
    lane: str
    env_name: str
    env_params: dict[str, Any]
    gamma: float
    mdp_fingerprint: str
    pi_star: np.ndarray
    v_star_control: np.ndarray
    v_pi_star: np.ndarray
    p_pi: np.ndarray
    r_pi: np.ndarray
    certification: dict[str, Any]


def compute_mdp_fingerprint(mdp: MDP) -> str:
    p = np.ascontiguousarray(_as_numpy(mdp.P, dtype=np.float64))
    r = np.ascontiguousarray(_as_numpy(mdp.R, dtype=np.float64))
    gamma = np.asarray([float(mdp.gamma)], dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(p.shape).encode("utf-8"))
    digest.update(p.tobytes())
    digest.update(str(r.shape).encode("utf-8"))
    digest.update(r.tobytes())
    digest.update(gamma.tobytes())
    return digest.hexdigest()


def load_deterministic_target_bundle(
    manifest_path: str | Path,
    *,
    mdp: MDP,
) -> LoadedDeterministicTarget:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid deterministic target manifest at {manifest_path}.")
    if not bool(manifest.get("certified", False)):
        raise ValueError(f"Deterministic target manifest is not certified: {manifest_path}")

    expected_fingerprint = str(manifest.get("true_mdp_fingerprint"))
    actual_fingerprint = compute_mdp_fingerprint(mdp)
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "Deterministic target fingerprint mismatch: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}."
        )

    arrays_rel = manifest.get("artifacts", {}).get("target_arrays_npz")
    if not isinstance(arrays_rel, str) or not arrays_rel:
        raise ValueError(f"Deterministic target manifest missing target_arrays_npz: {manifest_path}")
    arrays_path = (Path(manifest_path).parent / arrays_rel).resolve()
    if not arrays_path.exists():
        raise FileNotFoundError(arrays_path)

    with np.load(arrays_path, allow_pickle=False) as payload:
        pi_star = np.asarray(payload["pi_star"], dtype=np.int32)
        v_star_control = np.asarray(payload["v_star_control"], dtype=np.float64)
        v_pi_star = np.asarray(payload["v_pi_star"], dtype=np.float64)
        p_pi = np.asarray(payload["p_pi"], dtype=np.float64)
        r_pi = np.asarray(payload["r_pi"], dtype=np.float64)

    return LoadedDeterministicTarget(
        target_id=str(manifest["target_id"]),
        manifest_path=Path(manifest_path),
        lane=str(manifest["lane"]),
        env_name=str(manifest["env_name"]),
        env_params=dict(manifest.get("env_params", {})),
        gamma=float(manifest["gamma"]),
        mdp_fingerprint=expected_fingerprint,
        pi_star=pi_star,
        v_star_control=v_star_control,
        v_pi_star=v_pi_star,
        p_pi=p_pi,
        r_pi=r_pi,
        certification=dict(manifest.get("certification", {})),
    )


def build_control_residual_fn(mdp: MDP) -> Callable[[Any], float]:
    p = _as_jax(mdp.P)
    r = _as_jax(mdp.R)
    gamma = mdp.gamma

    def compute_resid(v: Any) -> float:
        v_arr = _as_jax(v)
        q = r + gamma * jnp.einsum("sak,k->sa", p, v_arr)
        t_v = jnp.max(q, axis=1)
        return float(jnp.max(jnp.abs(t_v - v_arr)))

    return compute_resid


def evaluate_control_policy_value(mdp: MDP, policy: Any) -> jnp.ndarray:
    policy_arr = jnp.asarray(policy, dtype=jnp.int32)
    p = _as_jax(mdp.P)
    r = _as_jax(mdp.R)
    rows = jnp.arange(p.shape[0], dtype=jnp.int32)
    p_pi = p[rows, policy_arr, :]
    r_pi = r[rows, policy_arr]
    identity = jnp.eye(p.shape[0], dtype=jnp.float64)
    system = identity - float(mdp.gamma) * p_pi
    return jnp.linalg.solve(system, r_pi)


def _value_norm(vec: jnp.ndarray, *, norm_name: str) -> float:
    norm_key = norm_name.upper()
    if norm_key == "L1":
        return float(jnp.sum(jnp.abs(vec)))
    if norm_key == "L2":
        return float(jnp.linalg.norm(vec))
    if norm_key in {"LINF", "INF", "INFINITY"}:
        return float(jnp.max(jnp.abs(vec)))
    raise ValueError(f"Unsupported value norm {norm_name!r}")


def compute_value_error(
    value: Any,
    target: Any,
    *,
    norm_name: str,
    normalized: bool,
    eps: float = 1e-12,
) -> float:
    value_arr = _as_jax(value)
    target_arr = _as_jax(target)
    numer = _value_norm(value_arr - target_arr, norm_name=norm_name)
    if not normalized:
        return numer
    denom = _value_norm(target_arr, norm_name=norm_name) + eps
    return numer / denom


def control_target_value(
    mdp: MDP,
    *,
    target_bundle: LoadedDeterministicTarget | None = None,
) -> jnp.ndarray:
    if target_bundle is not None:
        return _as_jax(target_bundle.v_star_control)

    p = _as_jax(mdp.P)
    r = _as_jax(mdp.R)
    v = jnp.zeros(p.shape[0], dtype=jnp.float64)
    for _ in range(10000):
        q = r + float(mdp.gamma) * jnp.einsum("sak,k->sa", p, v)
        v_next = jnp.max(q, axis=1)
        if bool(jnp.max(jnp.abs(v_next - v)) <= 1e-12):
            return v_next
        v = v_next
    return v


def reconstruct_policy_value_errors(
    mdp: MDP,
    policy_trace: Any,
    *,
    target_bundle: LoadedDeterministicTarget | None = None,
    norm_name: str = "L1",
    normalized: bool = True,
) -> list[float]:
    target_value = control_target_value(mdp, target_bundle=target_bundle)
    policies = _as_numpy(policy_trace, dtype=np.int32)
    if policies.ndim == 1:
        policies = policies[None, :]
    errors: list[float] = []
    for policy in policies:
        v_pi = evaluate_control_policy_value(mdp, policy)
        errors.append(
            compute_value_error(
                v_pi,
                target_value,
                norm_name=norm_name,
                normalized=normalized,
            )
        )
    return errors


def make_zero_value(mdp: MDP) -> jnp.ndarray:
    return jnp.zeros(mdp.P.shape[0], dtype=jnp.float64)


def seeded_shadow_init(n_states: int, seed: int) -> jnp.ndarray:
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (n_states,), dtype=jnp.float64)


__all__ = [
    "LoadedDeterministicTarget",
    "apply_model_error_for_task",
    "build_control_residual_fn",
    "compute_mdp_fingerprint",
    "compute_value_error",
    "control_target_value",
    "evaluate_control_policy_value",
    "load_deterministic_target_bundle",
    "make_zero_value",
    "reconstruct_policy_value_errors",
    "seeded_shadow_init",
]
