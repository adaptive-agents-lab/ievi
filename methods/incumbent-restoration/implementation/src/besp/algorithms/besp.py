"""Compatibility re-exports for canonical BESP helpers."""

from ..algorithms.jax.deterministic import _discount_gap_depth_pe_impl, _discount_gap_depth_theory
from ..core.types import MDP
from ..domains.control import pick_isp_params_control
from ..domains.policy_evaluation import BESPParams, besp_pe, besp_precompute, model_error_stats, pick_besp_rho_half

__all__ = [
    "BESPParams",
    "_discount_gap_depth_pe_impl",
    "_discount_gap_depth_theory",
    "besp_pe",
    "besp_precompute",
    "model_error_stats",
    "pick_besp_rho_half",
    "pick_isp_params_control",
    "MDP",
]
