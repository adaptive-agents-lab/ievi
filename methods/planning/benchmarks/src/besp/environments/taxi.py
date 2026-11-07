"""Compatibility wrapper for the JAX-native taxi environment."""

from .jax_native import generate_taxi_jax as generate_taxi

__all__ = ["generate_taxi"]
