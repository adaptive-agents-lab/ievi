"""Canonical parameter helpers for policy-evaluation algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...algorithms.jax.deterministic import pick_besp_rho_half as _pick_besp_rho_half_impl


@dataclass(frozen=True)
class BESPParams:
    """Canonical BESP PE parameters with manuscript-aligned ``m`` semantics."""

    # ``m`` is the manuscript inner defect-correction count, not a series degree.
    m: int
    h: float
    delta: float
    E_star: float
    RE: float
    rho: float
    source: str


def pick_besp_rho_half(p_pi: Any, p_hat_pi: Any, gamma: float, norm_name: str = "L1") -> BESPParams:
    """Return maintained BESP defaults with manuscript-aligned ``m`` semantics."""

    m, h, delta, e_star, re, rho, source = _pick_besp_rho_half_impl(
        p_pi,
        p_hat_pi,
        gamma,
        norm_name=norm_name,
    )
    return BESPParams(m=m, h=h, delta=delta, E_star=e_star, RE=re, rho=rho, source=source)
