"""Internal helpers for the maintained JAX hyper-batch runner."""

from .algorithms import group_configs_by_algorithm, resolve_jax_algorithm_class
from .planning import (
    StochasticBatchCompatibilityKey,
    compute_packable_config_count,
    stochastic_batch_compatibility_key,
    validate_packed_stochastic_group,
)
from .results import assemble_algorithm_results, build_trace
from .specs import HyperBatchConfig, normalize_hyper_batch_configs

__all__ = [
    "HyperBatchConfig",
    "StochasticBatchCompatibilityKey",
    "assemble_algorithm_results",
    "build_trace",
    "compute_packable_config_count",
    "group_configs_by_algorithm",
    "normalize_hyper_batch_configs",
    "resolve_jax_algorithm_class",
    "stochastic_batch_compatibility_key",
    "validate_packed_stochastic_group",
]
