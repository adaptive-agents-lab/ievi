"""Compatibility re-exports for canonical A-BESP helpers."""

from ..algorithms.jax.deterministic import (
    _as_jax,
    _compute_value_error as _compute_value_error_local,
    _discount_gap_depth_pe_impl,
    _get_bellman_residual_norm as get_bellman_residual_norm,
)
from ..domains.policy_evaluation import abesp_pe, besp_precompute

__all__ = [
    "_as_jax",
    "_compute_value_error_local",
    "_discount_gap_depth_pe_impl",
    "abesp_pe",
    "besp_precompute",
    "get_bellman_residual_norm",
]
