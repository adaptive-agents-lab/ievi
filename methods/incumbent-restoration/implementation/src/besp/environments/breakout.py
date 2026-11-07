"""Compatibility wrapper for the JAX-native mini-breakout environment."""

from .jax_native import generate_mini_breakout_jax as generate_mini_breakout

__all__ = ["generate_mini_breakout"]
