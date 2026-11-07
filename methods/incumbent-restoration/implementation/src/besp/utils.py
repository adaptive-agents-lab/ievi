"""Compatibility numerical helpers backed directly by JAX array ops."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional
import warnings

import jax
import jax.numpy as jnp

from .core.types import ValueFunction

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
else:
    NDArray = Any


# Type alias for norm names
NormType = Literal["L1", "L2", "Linf", "inf", "spectral", "fro", "frob", "entry_inf"]


def _as_jax(array: Any, *, dtype: Any | None = None) -> jax.Array:
    return jnp.asarray(array, dtype=dtype)


def _return_array(array: Any, *, like: Any) -> Any:
    if isinstance(like, jax.Array):
        return array
    return jax.device_get(array)


# =============================================================================
# Matrix Norms
# =============================================================================

def max_row_sum_norm(mat: NDArray[np.floating[Any]]) -> float:
    """
    Compute the induced infinity norm (maximum absolute row sum).
    
    This is the standard operator norm for matrices when considering
    the infinity norm on vectors: ||A||_∞ = max_i Σ_j |A_ij|
    
    Args:
        mat: Input matrix of any shape (will be treated as 2D)
        
    Returns:
        Maximum absolute row sum
        
    Example:
        >>> A = np.array([[1, -2], [3, 4]])
        >>> max_row_sum_norm(A)
        7.0
    """
    mat_arr = _as_jax(mat, dtype=jnp.float64)
    return float(jnp.max(jnp.sum(jnp.abs(mat_arr), axis=1)))


def model_error_norm(
    P: NDArray[np.floating[Any]], 
    P_hat: NDArray[np.floating[Any]], 
    norm_name: str = "Linf"
) -> float:
    """
    Compute the model error norm ||P_hat - P||_* for transition matrices.
    
    This function computes various matrix norms of the difference between
    an approximate transition model and the true model. Used extensively
    in BESP theory for bounding convergence rates.
    
    Args:
        P: True transition matrix, shape (n_s, n_a, n_s) or (n_s*n_a, n_s)
        P_hat: Approximate transition matrix, same shape as P
        norm_name: Norm type to use:
            - "Linf", "inf", "row_sum": Induced L∞ norm (max row sum)
            - "L1", "col_sum": Induced L1 norm (max column sum)
            - "L2", "spectral": Induced L2 norm (largest singular value)
            - "entry_inf", "maxabs": Max absolute entry (element-wise)
            - "fro", "frob": Frobenius norm
            
    Returns:
        The computed norm value
        
    Example:
        >>> P = np.random.dirichlet(np.ones(10), size=(5, 4)).reshape(5, 4, 10)
        >>> P_hat = np.random.dirichlet(np.ones(10), size=(5, 4)).reshape(5, 4, 10)
        >>> err = model_error_norm(P, P_hat, "Linf")
    """
    p_arr = _as_jax(P, dtype=jnp.float64)
    p_hat_arr = _as_jax(P_hat, dtype=jnp.float64)

    diff = p_hat_arr - p_arr
    key = norm_name.lower()

    if key in ("l2", "spectral"):
        # Spectral norm via largest singular value
        svals = jnp.linalg.svd(diff.reshape(-1, diff.shape[-1]), compute_uv=False)
        return float(jnp.max(svals))

    if key in ("l1", "col_sum", "column_sum"):
        mat = diff.reshape(-1, diff.shape[-1])
        return float(jnp.max(jnp.sum(jnp.abs(mat), axis=0)))

    if key in ("entry_inf", "entryinf", "maxabs"):
        return float(jnp.max(jnp.abs(diff)))

    if key in ("fro", "frob", "frobenius"):
        return float(jnp.linalg.norm(diff))

    # Default: induced Linf (row-sum)
    mat = diff.reshape(-1, diff.shape[-1])
    return float(jnp.max(jnp.sum(jnp.abs(mat), axis=1)))


# =============================================================================
# Value Function Errors
# =============================================================================

def compute_value_error(
    v: ValueFunction, 
    v_true: ValueFunction, 
    norm_name: str = "L1", 
    normalized: bool = True, 
    eps: float = 1e-12
) -> float:
    """
    Compute the error between value function estimates.
    
    This is the primary metric for evaluating Policy Evaluation algorithms.
    Supports multiple norm types and optional normalization.
    
    Args:
        v: Estimated value function, shape (n_states,)
        v_true: True value function, shape (n_states,)
        norm_name: Norm type: "L1", "L2", or "inf"
        normalized: If True, normalize by ||v_true||
        eps: Small constant to prevent division by zero
        
    Returns:
        Error value (normalized if requested)
        
    Example:
        >>> v_true = np.array([1.0, 2.0, 3.0])
        >>> v_est = np.array([1.1, 1.9, 3.2])
        >>> compute_value_error(v_est, v_true, "L1", normalized=True)
        0.066...
    """
    v_arr = _as_jax(v, dtype=jnp.float64)
    v_true_arr = _as_jax(v_true, dtype=jnp.float64)
    diff = v_arr - v_true_arr
    key = norm_name.lower()

    if key == "l2":
        num = float(jnp.linalg.norm(diff))
        denom = float(jnp.linalg.norm(v_true_arr)) + eps
    elif key in ("inf", "linf"):
        num = float(jnp.max(jnp.abs(diff)))
        denom = float(jnp.max(jnp.abs(v_true_arr))) + eps
    else:  # L1
        num = float(jnp.sum(jnp.abs(diff)))
        denom = float(jnp.sum(jnp.abs(v_true_arr))) + eps

    return num / denom if normalized else num


def get_bellman_residual_norm(
    V: ValueFunction,
    P_pi: NDArray[np.floating[Any]],
    r_pi: NDArray[np.floating[Any]],
    gamma: float,
    norm_name: str = "Linf",
) -> float:
    """
    Compute the Bellman residual norm ||T^π V - V||_*.
    
    The Bellman residual measures how far V is from being a fixed point
    of the Bellman operator. For optimal V^π, this should be zero.
    
    Args:
        V: Current value function estimate, shape (n_states,)
        P_pi: Policy transition matrix, shape (n_states, n_states)
        r_pi: Policy reward vector, shape (n_states,)
        gamma: Discount factor
        norm_name: Norm type for computing residual
        
    Returns:
        Bellman residual norm
        
    Note:
        BR(V) = r_π + γ P_π V - V
    """
    v_arr = _as_jax(V, dtype=jnp.float64)
    p_pi_arr = _as_jax(P_pi, dtype=jnp.float64)
    r_pi_arr = _as_jax(r_pi, dtype=jnp.float64)

    br = r_pi_arr + gamma * (p_pi_arr @ v_arr) - v_arr
    return compute_value_error(br, jnp.zeros_like(br), norm_name=norm_name, normalized=False)


def normalized_l1_error(
    v: ValueFunction, 
    v_true: ValueFunction, 
    eps: float = 1e-12
) -> float:
    """
    Compute normalized L1 error between value functions.
    
    .. deprecated::
        Use `compute_value_error(v, v_true, norm_name="L1")` instead.
    
    Args:
        v: Estimated value function
        v_true: True value function
        eps: Stability constant
        
    Returns:
        Normalized L1 error
    """
    return compute_value_error(v, v_true, norm_name="L1", normalized=True, eps=eps)


# =============================================================================
# Linear System Solvers
# =============================================================================

def _solve_inverse_device(mat):
    """Internal: Solve for matrix inverse, keeping result on device."""
    mat_arr = _as_jax(mat, dtype=jnp.float64)
    return jnp.linalg.solve(mat_arr, jnp.eye(mat_arr.shape[0], dtype=mat_arr.dtype))


def solve_inverse(mat: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """
    Compute matrix inverse via linear solve for improved numerical stability.
    
    Uses linear solver instead of explicit inversion: A^{-1} = solve(A, I).
    This is more numerically stable than np.linalg.inv().
    
    Args:
        mat: Square matrix to invert, shape (n, n)
        
    Returns:
        Matrix inverse, shape (n, n), materialized on host for non-JAX inputs
        
    Raises:
        LinAlgError: If matrix is singular
    """
    result = _solve_inverse_device(mat)
    return _return_array(result, like=mat)


def solve_iterative(
    P_pi: NDArray[np.floating[Any]],
    r_pi: NDArray[np.floating[Any]],
    gamma: float,
    tol: float = 1e-10,
    max_iter: int = 100000,
) -> NDArray[np.floating[Any]]:
    """
    Solve for value function using iterative policy evaluation.
    
    Solves the linear system V = r_π + γ P_π V using fixed-point iteration.
    Equivalent to running Value Iteration for a fixed policy until convergence.
    
    Args:
        P_pi: Policy transition matrix, shape (n_states, n_states)
        r_pi: Policy reward vector, shape (n_states,)
        gamma: Discount factor
        tol: Convergence tolerance (max change in V)
        max_iter: Maximum iterations
        
    Returns:
        Value function V^π, shape (n_states,)
        
    Note:
        For well-conditioned problems, consider using direct solve:
        V = solve(I - γP_π, r_π) which is faster but less stable.
    """
    p_pi_arr = _as_jax(P_pi, dtype=jnp.float64)
    r_pi_arr = _as_jax(r_pi, dtype=jnp.float64)
    v = jnp.zeros(p_pi_arr.shape[0], dtype=r_pi_arr.dtype)

    for _ in range(max_iter):
        v_new = r_pi_arr + gamma * (p_pi_arr @ v)
        if float(jnp.max(jnp.abs(v_new - v))) < tol:
            return _return_array(v_new, like=r_pi)
        v = v_new

    return _return_array(v, like=r_pi)


# =============================================================================
# Numerical Stability
# =============================================================================

def ensure_finite(
    arr: NDArray[np.floating[Any]], 
    name: str = "array", 
    replace_with: Optional[float] = None,
    warn: bool = True
) -> NDArray[np.floating[Any]]:
    """
    Check and optionally fix non-finite values in an array.
    
    Useful for catching numerical instabilities in iterative algorithms.
    
    Args:
        arr: Array to check
        name: Name for error/warning messages
        replace_with: If provided, replace non-finite values with this.
                     If None, raises ValueError on non-finite values.
        warn: If True, print warning when replacing values
        
    Returns:
        The (possibly modified) array
        
    Raises:
        ValueError: If non-finite values found and replace_with is None
        
    Example:
        >>> x = np.array([1.0, np.nan, 3.0])
        >>> ensure_finite(x, "values", replace_with=0.0)
        array([1., 0., 3.])
    """
    arr_jax = _as_jax(arr)
    finite_mask = jnp.isfinite(arr_jax)
    if bool(jnp.all(finite_mask)):
        return arr

    if replace_with is not None:
        if warn:
            warnings.warn(
                f"Non-finite values found in {name}. Replacing with {replace_with}."
            )
        arr_jax = jnp.where(finite_mask, arr_jax, replace_with)
    else:
        raise ValueError(f"Non-finite values found in {name}.")
    return _return_array(arr_jax, like=arr)


# =============================================================================
# Learning Rate Scheduling
# =============================================================================

class LearningRateScheduler:
    """
    Learning rate scheduler for RL algorithms.
    
    Provides various learning rate decay schedules commonly used in
    reinforcement learning, especially for online algorithms.
    
    Supported schedules:
        - "ConstantAndDelay": Constant for `delay` steps, then 1/t decay
        - "RescaledLinear": Scaled linear decay
    
    Attributes:
        lr_type: Type of schedule
        start_lr: Initial learning rate
        delay: Number of steps before decay begins
        decay_rate: Decay rate parameter
        
    Example:
        >>> scheduler = LearningRateScheduler("ConstantAndDelay", 0.5, delay=100)
        >>> [scheduler.get_lr(t) for t in [0, 50, 100, 200]]
        [0.5, 0.5, 0.5, 0.005]  # Constant then 1/t decay
    """
    
    def __init__(
        self, 
        lr_type: str, 
        start_lr: float, 
        delay: int = 0, 
        decay_rate: float = 0.9
    ) -> None:
        """
        Initialize the scheduler.
        
        Args:
            lr_type: Schedule type ("ConstantAndDelay" or "RescaledLinear")
            start_lr: Initial learning rate (alpha in papers)
            delay: Steps before decay starts (N in papers)
            decay_rate: Rate parameter for decay
        """
        self.lr_type = lr_type
        self.start_lr = start_lr
        self.delay = delay
        self.decay_rate = decay_rate

    def get_lr(self, current_iter: int) -> float:
        """
        Get learning rate for the current iteration.
        
        Args:
            current_iter: Current time step (0-indexed)
            
        Returns:
            Learning rate for this step
            
        Raises:
            ValueError: If lr_type is unknown
        """
        if self.lr_type == "RescaledLinear":
            return self.start_lr / (1.0 + (1.0 - self.decay_rate) * (current_iter + 1))
        
        elif self.lr_type == "ConstantAndDelay":
            if current_iter <= self.delay:
                return self.start_lr
            # After delay: α / (t - N)
            denom = current_iter - self.delay
            if denom <= 0:
                return self.start_lr  # Edge case guard
            return self.start_lr / float(denom)
        
        else:
            raise ValueError(f"Unknown LR type: {self.lr_type}")
    
    def __repr__(self) -> str:
        return (
            f"LearningRateScheduler(type='{self.lr_type}', "
            f"start={self.start_lr}, delay={self.delay})"
        )
