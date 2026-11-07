"""Matrix-free fine/coarse Markov simulators for the runtime comparison.

The true macro-transition operator is a composition of many fine Euler
microsteps of a time-varying, spatially varying advection-diffusion generator.
The surrogate uses the same continuous-time dynamics but far fewer coarse
microsteps.  Neither macro-transition matrix is assembled.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numba import njit


@njit(cache=True)
def _apply_steps(
    x: np.ndarray,
    rate_right: np.ndarray,
    rate_left: np.ndarray,
    rate_up: np.ndarray,
    rate_down: np.ndarray,
    dt: float,
    n: int,
) -> np.ndarray:
    """Apply B_0 ... B_{K-1} to a value vector, without forming the product.

    A row-stochastic transition product P=B_0...B_{K-1} acts on a value
    vector from right to left, hence the reverse microstep order below.
    Reflecting boundaries are implemented by mapping an outward transition
    back to the same state.
    """
    value = x.copy()
    output = np.empty_like(value)
    steps = rate_right.shape[0]

    for step in range(steps - 1, -1, -1):
        for i in range(n):
            i_down = i - 1 if i > 0 else i
            i_up = i + 1 if i < n - 1 else i
            row = i * n
            for j in range(n):
                j_left = j - 1 if j > 0 else j
                j_right = j + 1 if j < n - 1 else j
                idx = row + j
                centre = value[idx]
                output[idx] = centre + dt * (
                    rate_right[step, idx] * (value[row + j_right] - centre)
                    + rate_left[step, idx] * (value[row + j_left] - centre)
                    + rate_up[step, idx] * (value[i_up * n + j] - centre)
                    + rate_down[step, idx] * (value[i_down * n + j] - centre)
                )
        temporary = value
        value = output
        output = temporary
    return value


@njit(cache=True)
def _propagate_distribution(
    distribution: np.ndarray,
    rate_right: np.ndarray,
    rate_left: np.ndarray,
    rate_up: np.ndarray,
    rate_down: np.ndarray,
    dt: float,
    n: int,
) -> np.ndarray:
    """Propagate a row distribution through the microsteps in forward time."""
    mass = distribution.copy()
    output = np.empty_like(mass)
    steps = rate_right.shape[0]

    for step in range(steps):
        output[:] = 0.0
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                current_mass = mass[idx]
                p_right = dt * rate_right[step, idx]
                p_left = dt * rate_left[step, idx]
                p_up = dt * rate_up[step, idx]
                p_down = dt * rate_down[step, idx]
                p_stay = 1.0 - p_right - p_left - p_up - p_down

                output[idx] += current_mass * p_stay
                output[i * n + (j + 1 if j < n - 1 else j)] += current_mass * p_right
                output[i * n + (j - 1 if j > 0 else j)] += current_mass * p_left
                output[(i + 1 if i < n - 1 else i) * n + j] += current_mass * p_up
                output[(i - 1 if i > 0 else i) * n + j] += current_mass * p_down

        temporary = mass
        mass = output
        output = temporary
    return mass


@dataclass(frozen=True)
class DynamicsParameters:
    diffusion_rate: float = 0.12
    advection_amplitude: float = 0.30
    secondary_amplitude: float = 0.08
    phase_seed: int = 20260729


class MicroSimulator:
    """A matrix-free macro-transition operator built from local microsteps."""

    def __init__(
        self,
        grid_side: int,
        microsteps: int,
        parameters: DynamicsParameters,
    ) -> None:
        if grid_side < 2:
            raise ValueError("grid_side must be at least 2")
        if microsteps < 1:
            raise ValueError("microsteps must be positive")

        start = perf_counter()
        self.grid_side = int(grid_side)
        self.state_count = self.grid_side * self.grid_side
        self.microsteps_per_call = int(microsteps)
        self.dt = 1.0 / self.microsteps_per_call
        self.parameters = parameters
        self.calls = 0
        self.microstep_count = 0

        coordinates = (np.arange(self.grid_side, dtype=np.float64) + 0.5) / self.grid_side
        x, y = np.meshgrid(2.0 * coordinates - 1.0, 2.0 * coordinates - 1.0, indexing="ij")

        rng = np.random.default_rng(parameters.phase_seed)
        phase_x, phase_y, phase_secondary_x, phase_secondary_y = rng.uniform(0.0, 2.0 * np.pi, size=4)

        right: list[np.ndarray] = []
        left: list[np.ndarray] = []
        up: list[np.ndarray] = []
        down: list[np.ndarray] = []

        for step in range(self.microsteps_per_call):
            time_point = (step + 0.5) / self.microsteps_per_call
            velocity_x = (
                parameters.advection_amplitude
                * np.sin(np.pi * y + 2.0 * np.pi * time_point + phase_x)
                + parameters.secondary_amplitude
                * np.sin(2.0 * np.pi * x - 2.0 * np.pi * time_point + phase_secondary_x)
            )
            velocity_y = (
                parameters.advection_amplitude
                * np.cos(np.pi * x - 2.0 * np.pi * time_point + phase_y)
                + parameters.secondary_amplitude
                * np.cos(2.0 * np.pi * y + 2.0 * np.pi * time_point + phase_secondary_y)
            )

            right.append((parameters.diffusion_rate + np.maximum(velocity_x, 0.0)).ravel())
            left.append((parameters.diffusion_rate + np.maximum(-velocity_x, 0.0)).ravel())
            up.append((parameters.diffusion_rate + np.maximum(velocity_y, 0.0)).ravel())
            down.append((parameters.diffusion_rate + np.maximum(-velocity_y, 0.0)).ravel())

        self.rate_right = np.asarray(right, dtype=np.float64)
        self.rate_left = np.asarray(left, dtype=np.float64)
        self.rate_up = np.asarray(up, dtype=np.float64)
        self.rate_down = np.asarray(down, dtype=np.float64)

        exit_rates = self.rate_right + self.rate_left + self.rate_up + self.rate_down
        self.maximum_microstep_exit_probability = float(self.dt * np.max(exit_rates))
        if self.maximum_microstep_exit_probability > 1.0 + 1e-12:
            raise ValueError(
                "Euler microstep is not stochastic: maximum exit probability "
                f"{self.maximum_microstep_exit_probability:.6g} exceeds one"
            )

        # Compile the JIT kernels before any timed run.
        zero = np.zeros(self.state_count, dtype=np.float64)
        _apply_steps(
            zero,
            self.rate_right,
            self.rate_left,
            self.rate_up,
            self.rate_down,
            self.dt,
            self.grid_side,
        )
        _propagate_distribution(
            zero,
            self.rate_right,
            self.rate_left,
            self.rate_up,
            self.rate_down,
            self.dt,
            self.grid_side,
        )
        self.setup_seconds = perf_counter() - start

    def reset_counters(self) -> None:
        self.calls = 0
        self.microstep_count = 0

    def apply(self, value: np.ndarray, *, count: bool = True) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (self.state_count,):
            raise ValueError(f"Expected shape {(self.state_count,)}, received {vector.shape}")
        if count:
            self.calls += 1
            self.microstep_count += self.microsteps_per_call
        return _apply_steps(
            vector,
            self.rate_right,
            self.rate_left,
            self.rate_up,
            self.rate_down,
            self.dt,
            self.grid_side,
        )

    def row_distribution(self, state_index: int) -> np.ndarray:
        if not 0 <= state_index < self.state_count:
            raise IndexError(state_index)
        initial = np.zeros(self.state_count, dtype=np.float64)
        initial[state_index] = 1.0
        return _propagate_distribution(
            initial,
            self.rate_right,
            self.rate_left,
            self.rate_up,
            self.rate_down,
            self.dt,
            self.grid_side,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "grid_side": self.grid_side,
            "state_count": self.state_count,
            "microsteps_per_call": self.microsteps_per_call,
            "dt": self.dt,
            "maximum_microstep_exit_probability": self.maximum_microstep_exit_probability,
            "setup_seconds": self.setup_seconds,
            "dynamics_parameters": {
                "diffusion_rate": self.parameters.diffusion_rate,
                "advection_amplitude": self.parameters.advection_amplitude,
                "secondary_amplitude": self.parameters.secondary_amplitude,
                "phase_seed": self.parameters.phase_seed,
            },
        }


def make_reward(grid_side: int) -> np.ndarray:
    """A smooth, nontrivial reward used by all methods."""
    coordinates = (np.arange(grid_side, dtype=np.float64) + 0.5) / grid_side
    x, y = np.meshgrid(2.0 * coordinates - 1.0, 2.0 * coordinates - 1.0, indexing="ij")
    reward = (
        np.exp(-((x - 0.45) ** 2 + (y + 0.30) ** 2) / 0.08)
        - 0.70 * np.exp(-((x + 0.40) ** 2 + (y - 0.35) ** 2) / 0.06)
        + 0.20 * np.sin(3.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    )
    reward -= np.mean(reward)
    scale = np.max(np.abs(reward))
    if scale == 0.0:
        raise RuntimeError("Degenerate reward")
    return (reward / scale).ravel()


def anchor_states(grid_side: int, random_count: int = 8, seed: int = 1729) -> list[int]:
    """Deterministic boundary/interior states plus fixed random anchors."""
    axis = np.linspace(0, grid_side - 1, 5, dtype=int)
    states = {int(i * grid_side + j) for i in axis for j in axis}
    rng = np.random.default_rng(seed + grid_side)
    for index in rng.choice(grid_side * grid_side, size=min(random_count, grid_side * grid_side), replace=False):
        states.add(int(index))
    return sorted(states)
