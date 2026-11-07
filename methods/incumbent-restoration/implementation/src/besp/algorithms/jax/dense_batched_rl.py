"""JAX-accelerated batched stochastic baseline algorithms for BESP.

These implementations use JAX control flow primitives and XLA compilation to
run the stochastic baselines on one or many devices.


Shared utilities are provided by:
- base.py: Shared utilities (PI solver, sampling, LR computation)
- q_learning.py: Q-Learning implementation
- dyna.py: Dyna and OS-Dyna implementations
"""

import jax

from .base import benchmark_vi

if __name__ == "__main__":
    # Quick test
    print("JAX Backend:", jax.default_backend())
    print("Devices:", jax.devices())

    # Benchmark
    result = benchmark_vi(n_s=48, n_a=4, num_envs=100, num_iters=1000)
    print(f"\nBenchmark: {result['elapsed_ms']:.2f}ms for {result['iters']} iters")
    print(f"  {result['ms_per_iter']:.4f}ms per iteration")
