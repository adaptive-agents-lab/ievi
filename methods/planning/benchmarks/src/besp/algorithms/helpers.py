"""Compatibility re-exports for canonical deterministic control helpers."""

from ..domains.control import build_policy_matrices, solve_value, value_iteration_control

__all__ = ["build_policy_matrices", "solve_value", "value_iteration_control"]
