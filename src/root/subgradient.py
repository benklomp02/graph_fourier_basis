import numpy as np
from scipy.linalg import null_space
from enum import Enum


def compute_exact_null_space(weights: np.ndarray) -> np.ndarray:
    """
    For a convex quadratic J(x) = ||L x||_1 with L the graph Laplacian,
    the exact null space of L is the minimizer manifold.
    """
    # Build graph Laplacian L
    degrees = np.sum(weights, axis=1)
    L = np.diag(degrees) - weights
    # Return basis for null-space of L
    return null_space(L)


class sequences(Enum):
    """A simle enum of sequences for easy access."""

    HARMONIC = "harmonic"
    LOG_HARMONIC = "log-harmonic"
    POWER = "power"


def compute_greedy_subgradient_basis(
    weights: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-3,
    seq: sequences = sequences.HARMONIC,
) -> np.ndarray:
    """
    Run a greedy subgradient descent to find a minimizer x* of J(x) = sum_i |(Lx)_i|,
    then compute the tangent-space basis (null-space of the constraint matrix).
    """
    x_star, _ = _compute_greedy_subgradient(weights, x0, max_iter, tol, seq)
    return _null_space_basis(weights, x_star, tol)


def _calculate_step_size(seq: sequences, k: int) -> float:
    """
    Get the step size for the k-th iteration based on the specified sequence.
    """
    assert k > 0, "k must be a positive integer."
    if seq == sequences.HARMONIC:
        return 1 / k
    elif seq == sequences.LOG_HARMONIC:
        return 1 / (k * np.log(k + 1))
    elif seq == sequences.POWER:
        p = np.random.uniform(0.5, 1)
        return 1 / (k**p)
    else:
        raise ValueError(f"Unknown sequence: {seq}.")


def _compute_greedy_subgradient(
    weights: np.ndarray,
    x: np.ndarray,
    max_iter: int,
    tol: float,
    seq: sequences,
) -> np.ndarray:
    """
    Greedy subgradient: at each step, pick the largest |S_i| and step along its subgradient.
    """
    n = weights.shape[0]
    active = np.arange(n, dtype=int)
    x = x.copy()
    k = 0
    while True:
        k += 1
        # Compute all S_i(x), i=1..n for the active indices
        S = np.zeros(n)
        S[active] = np.array([_compute_Si(weights, x, i) for i in active])
        # Filter out close to zero entries
        mask = np.abs(S[active]) >= tol
        active = active[mask]
        if active.size == 0:
            break
        # Find the index with the largest |S_i|
        i_max = active[np.argmax(np.abs(S[active]))]
        # Compute the subgradient and step
        g = _compute_grad_Si(weights, i_max)
        subgrad = np.sign(S[i_max]) * g
        step = _calculate_step_size(seq, k)
        # Update x^(k+1) = x^k - step * subgrad
        x_new = x - step * subgrad
        # Check convergence
        if np.linalg.norm(x_new - x) < tol or k >= max_iter:
            x = x_new
            break
        x = x_new
    return x, k


def _compute_Si(weights: np.ndarray, x: np.ndarray, i: int) -> float:
    """
    S_i(x) = sum_j w_ij (x_i - x_j) = (L x)_i
    """
    return np.dot(weights[i], x[i] - x)


def _compute_grad_Si(weights: np.ndarray, i: int) -> np.ndarray:
    """
    Gradient of S_i(x) = (L x)_i is the i-th row of Laplacian L. The gradient
    is constant and therefore does not depend on x.
    """
    w = weights[i]
    grad = -w.copy()
    grad[i] = np.sum(w)
    return grad


def _null_space_basis(
    weights: np.ndarray,
    x_star: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Build the active/inactive set constraints and compute their null-space.
    """
    n = weights.shape[0]
    S = np.array([_compute_Si(weights, x_star, i) for i in range(n)])
    signs = np.sign(S)
    A_idx = np.where(np.abs(S) < tol)[0]
    N_idx = np.where(np.abs(S) >= tol)[0]
    A_rows = []
    for i in A_idx:
        A_rows.append(_compute_grad_Si(weights, i))
    if len(N_idx) > 0:
        grad_sum = np.sum(
            [signs[j] * _compute_grad_Si(weights, j) for j in N_idx], axis=0
        )
        A_rows.append(grad_sum)
    A = np.vstack(A_rows)
    return null_space(A)
