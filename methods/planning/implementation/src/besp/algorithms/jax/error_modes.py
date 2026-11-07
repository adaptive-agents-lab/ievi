"""Canonical stochastic model-error modes for maintained JAX RL control."""

from __future__ import annotations

from typing import Final


JAX_RL_SUPPORT_UNIFORM_ERROR_MODE: Final[str] = "support_uniform"
JAX_RL_SELF_LOOP_ERROR_MODE: Final[str] = "self_loop"

JAX_RL_STOCHASTIC_ERROR_MODE_ALIASES: Final[dict[str, str]] = {
    JAX_RL_SUPPORT_UNIFORM_ERROR_MODE: JAX_RL_SUPPORT_UNIFORM_ERROR_MODE,
    "smooth": JAX_RL_SUPPORT_UNIFORM_ERROR_MODE,
    JAX_RL_SELF_LOOP_ERROR_MODE: JAX_RL_SELF_LOOP_ERROR_MODE,
    "identity": JAX_RL_SELF_LOOP_ERROR_MODE,
}

JAX_RL_STOCHASTIC_MODEL_ERROR_CODES: Final[dict[str, int]] = {
    JAX_RL_SUPPORT_UNIFORM_ERROR_MODE: 0,
    JAX_RL_SELF_LOOP_ERROR_MODE: 1,
}


def _normalize_error_mode_key(mode: str | None) -> str:
    if mode is None:
        return JAX_RL_SUPPORT_UNIFORM_ERROR_MODE
    return str(mode).strip().lower().replace("-", "_")


def canonical_jax_rl_stochastic_error_mode(mode: str | None) -> str:
    key = _normalize_error_mode_key(mode)
    try:
        return JAX_RL_STOCHASTIC_ERROR_MODE_ALIASES[key]
    except KeyError as exc:  # pragma: no cover - exercised via callers
        allowed = ", ".join(sorted(JAX_RL_STOCHASTIC_ERROR_MODE_ALIASES))
        raise ValueError(
            "Maintained stochastic JAX RL supports only error_mode values "
            f"{allowed}; received {mode!r}."
        ) from exc


def jax_rl_stochastic_model_error_code(mode: str | None) -> int:
    canonical = canonical_jax_rl_stochastic_error_mode(mode)
    return JAX_RL_STOCHASTIC_MODEL_ERROR_CODES[canonical]


def jax_rl_stochastic_error_mode_from_code(code: int) -> str:
    resolved = int(code)
    for name, mode_code in JAX_RL_STOCHASTIC_MODEL_ERROR_CODES.items():
        if mode_code == resolved:
            return name
    raise ValueError(f"Unsupported stochastic JAX RL model_error_mode code: {code}.")


def jax_rl_stochastic_error_mode_token(mode: str | None) -> str | None:
    canonical = canonical_jax_rl_stochastic_error_mode(mode)
    if canonical == JAX_RL_SUPPORT_UNIFORM_ERROR_MODE:
        return None
    return canonical.replace("_", "-")


__all__ = [
    "JAX_RL_SELF_LOOP_ERROR_MODE",
    "JAX_RL_STOCHASTIC_ERROR_MODE_ALIASES",
    "JAX_RL_STOCHASTIC_MODEL_ERROR_CODES",
    "JAX_RL_SUPPORT_UNIFORM_ERROR_MODE",
    "canonical_jax_rl_stochastic_error_mode",
    "jax_rl_stochastic_error_mode_from_code",
    "jax_rl_stochastic_error_mode_token",
    "jax_rl_stochastic_model_error_code",
]
