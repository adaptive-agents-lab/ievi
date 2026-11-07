"""Shared JAX-backed model-error helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from jax import random

SUPPORTED_MODEL_ERROR_MODES = frozenset(
    {
        "support_uniform",
        "smooth",
        "self_loop",
        "identity",
        "temperature",
        "temp",
        "softmax",
        "dirichlet",
        "dirichlet_noise",
        "low_rank",
        "lowrank",
        "svd",
        "gaussian",
        "gauss",
        "blur",
    }
)


def _as_jax(array: Any, *, dtype: Any = jnp.float64) -> jnp.ndarray:
    return jnp.asarray(array, dtype=dtype)


def _normalize_mode(mode: str) -> str:
    return mode.lower().replace("-", "_")


def _smooth_model_jax(p: Any, lam: float) -> jnp.ndarray:
    p_arr = _as_jax(p)
    support = p_arr > 0
    support_size = jnp.sum(support, axis=2, keepdims=True)
    safe_support_size = jnp.where(support_size > 0, support_size, 1.0)
    uniform = jnp.where(support, 1.0 / safe_support_size, 0.0)
    mixed = (1.0 - lam) * p_arr + lam * uniform
    return jnp.where(support_size > 0, mixed, p_arr)


def _self_loop_model_jax(p: Any, lam: float) -> jnp.ndarray:
    p_arr = _as_jax(p)
    identity = jnp.eye(p_arr.shape[0], dtype=p_arr.dtype)
    return (1.0 - lam) * p_arr + lam * identity[:, None, :]


def _temperature_model_jax(p: Any, lam: Any) -> jnp.ndarray:
    p_arr = _as_jax(p)
    lam_clipped = jnp.clip(_as_jax(lam), 0.0, 0.999)
    temperature = 1.0 / (1.0 - lam_clipped)
    support = p_arr > 0
    logits = jnp.where(support, jnp.log(p_arr + 1e-20), -jnp.inf)
    scaled = logits / temperature
    scaled_max = jnp.max(scaled, axis=2, keepdims=True)
    exp_vals = jnp.where(support, jnp.exp(scaled - scaled_max), 0.0)
    denom = jnp.sum(exp_vals, axis=2, keepdims=True)
    return jnp.where(denom > 0, exp_vals / denom, p_arr)


def _dirichlet_row(key: jax.Array, row: jnp.ndarray, lam: Any) -> jnp.ndarray:
    support = row > 0
    support_count = jnp.sum(support)
    lam_clipped = jnp.clip(_as_jax(lam), 0.0, 1.0)
    eps = 1e-6
    k = (1.0 - lam_clipped) / (lam_clipped + eps)
    alpha = jnp.where(support, jnp.maximum(row * k, 1e-8), 1.0)
    min_alpha = jnp.min(jnp.where(support, alpha, jnp.inf))

    def no_support() -> jnp.ndarray:
        return row

    def sample_row() -> jnp.ndarray:
        def sample_logspace() -> jnp.ndarray:
            u = random.uniform(
                key,
                row.shape,
                minval=jnp.finfo(jnp.float64).tiny,
                maxval=1.0,
                dtype=jnp.float64,
            )
            log_gammas = jnp.where(support, jnp.log(u) / (alpha + 1e-20), -jnp.inf)
            probs = jax.nn.softmax(log_gammas, axis=0)
            return jnp.where(support, probs, 0.0)

        def sample_gamma() -> jnp.ndarray:
            gammas = random.gamma(key, alpha, dtype=jnp.float64)
            gammas = jnp.where(support, gammas, 0.0)
            total = jnp.sum(gammas)
            return jnp.where(total > 0, gammas / total, row)

        return jax.lax.cond(min_alpha < 0.01, sample_logspace, sample_gamma)

    return jax.lax.cond(support_count == 0, no_support, sample_row)


def _dirichlet_model_jax(p: Any, lam: Any, seed: Any) -> jnp.ndarray:
    p_arr = _as_jax(p)
    flat = p_arr.reshape(-1, p_arr.shape[-1])
    keys = random.split(random.PRNGKey(seed), flat.shape[0])
    sampled = jax.vmap(lambda key, row: _dirichlet_row(key, row, lam))(keys, flat)
    return sampled.reshape(p_arr.shape)


def _low_rank_model_jax(p: Any, lam: Any, min_rank: int = 1) -> jnp.ndarray:
    p_arr = _as_jax(p)
    n_s, n_a, _ = p_arr.shape
    matrix = p_arr.reshape(n_s * n_a, n_s)
    max_rank = min(matrix.shape)
    lam_clipped = float(jnp.clip(_as_jax(lam), 0.0, 1.0))
    target_rank = int(max_rank - lam_clipped * (max_rank - 1))
    target_rank = max(int(min_rank), target_rank)
    u, sigma, vt = jnp.linalg.svd(matrix, full_matrices=False)
    sigma_trunc = jnp.where(jnp.arange(sigma.shape[0]) < target_rank, sigma, 0.0)
    approx = (u * sigma_trunc) @ vt
    approx = jnp.maximum(approx, 0.0)
    row_sums = jnp.sum(approx, axis=1, keepdims=True)
    uniform_rows = jnp.full_like(approx, 1.0 / n_s)
    safe = jnp.where(row_sums < 1e-10, uniform_rows, approx)
    normalized = safe / jnp.sum(safe, axis=1, keepdims=True)
    return normalized.reshape(p_arr.shape)


def _resolve_grid_shape(
    p: Any,
    *,
    env_name: str | None = None,
    env_params: Mapping[str, Any] | None = None,
    grid_shape: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    if grid_shape is not None:
        return (int(grid_shape[0]), int(grid_shape[1]))

    env_params = env_params or {}
    if env_name in {"stochastic_grid", "dense_reward_grid", "four_rooms", "frozen_lake"}:
        size = int(env_params.get("size", 5 if env_name != "four_rooms" else 15))
        return (size, size)
    if env_name == "cliffwalk":
        return (6, 6)
    if env_name == "maze":
        return (3, 3)
    if env_name == "slippery_cliffwalk":
        return (4, 12)

    n_states = int(_as_jax(p).shape[0])
    side = int(math.isqrt(n_states))
    if side * side == n_states:
        return (side, side)
    return None


def _gaussian_kernel_jax(sigma: float) -> tuple[jnp.ndarray, int]:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    axis = jnp.arange(-radius, radius + 1, dtype=jnp.float64)
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    kernel = jnp.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / jnp.sum(kernel)
    return kernel[:, :, None, None], radius


def _gaussian_model_jax(
    p: Any,
    lam: Any,
    *,
    grid_shape: tuple[int, int],
    state_mask: Any | None = None,
    blur_scale: float = 1.0,
) -> jnp.ndarray:
    p_arr = _as_jax(p)
    height, width = int(grid_shape[0]), int(grid_shape[1])
    n_s, _, next_s = p_arr.shape
    if height * width != n_s or next_s != n_s:
        raise ValueError(f"grid_shape {grid_shape} does not match number of states {n_s}")

    lam_clipped = float(jnp.clip(_as_jax(lam), 0.0, 0.999))
    if lam_clipped == 0.0:
        return p_arr

    sigma = float(blur_scale) * (lam_clipped / (1.0 - lam_clipped))
    if sigma <= 0.0:
        return p_arr

    kernel, radius = _gaussian_kernel_jax(sigma)
    flat = p_arr.reshape(-1, height, width, 1)
    blurred = jax.lax.conv_general_dilated(
        flat,
        kernel,
        window_strides=(1, 1),
        padding=((radius, radius), (radius, radius)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )[..., 0]

    if state_mask is not None:
        mask = _as_jax(state_mask).reshape(height, width)
        blurred = blurred * mask

    blurred_flat = blurred.reshape(-1, n_s)
    original_flat = p_arr.reshape(-1, n_s)
    totals = jnp.sum(blurred_flat, axis=1, keepdims=True)
    normalized = jnp.where(totals > 1e-12, blurred_flat / totals, original_flat)
    return normalized.reshape(p_arr.shape)


def apply_model_error_for_task(
    p: Any,
    lam: Any,
    mode: str,
    seed: Any,
    env_name: str | None = None,
    env_params: Mapping[str, Any] | None = None,
    grid_shape: tuple[int, int] | None = None,
    state_mask: Any | None = None,
    blur_scale: float = 1.0,
    min_rank: int = 1,
) -> Any:
    """Apply documented maintained model-error modes using JAX-native helpers only."""
    key = _normalize_mode(mode)
    if key in {"support_uniform", "smooth"}:
        return _smooth_model_jax(p, lam)
    if key in {"self_loop", "identity"}:
        return _self_loop_model_jax(p, lam)
    if key in {"temperature", "temp", "softmax"}:
        return _temperature_model_jax(p, lam)
    if key in {"dirichlet", "dirichlet_noise"}:
        return _dirichlet_model_jax(p, lam, seed)
    if key in {"low_rank", "lowrank", "svd"}:
        return _low_rank_model_jax(p, lam, min_rank=min_rank)
    if key in {"gaussian", "gauss", "blur"}:
        resolved_grid_shape = _resolve_grid_shape(
            p, env_name=env_name, env_params=env_params, grid_shape=grid_shape
        )
        if resolved_grid_shape is None:
            raise ValueError("grid_shape is required for gaussian model error")
        return _gaussian_model_jax(
            p,
            lam,
            grid_shape=resolved_grid_shape,
            state_mask=state_mask,
            blur_scale=blur_scale,
        )

    raise ValueError(
        "Unsupported maintained model error mode "
        f"{mode!r}. Supported modes: {', '.join(sorted(SUPPORTED_MODEL_ERROR_MODES))}."
    )


__all__ = ["SUPPORTED_MODEL_ERROR_MODES", "apply_model_error_for_task"]
