import numpy as np
from line_profiler import profile

from src.graph_fourier_transform.api import compute_l1_norm_basis_fast, __objective_t__


def compute_greedy_basis(
    n: int,
    weights: np.ndarray,
    rank_fn: __objective_t__,
) -> np.ndarray:
    """
    Computes a greedy basis for the L1 norm.

    Parameters:
        n : int
            The number of vertices.
        weights : np.ndarray
            A simple graph represented as a numpy array.
        rank_fn : objective_t
            A function that computes 2 indices to be merged by the underlying objective.

    Returns:
        np.ndarray: The greedy basis.
    """
    return _compute_greedy_basis(n, weights, rank_fn)


@profile
def _compute_greedy_basis(n, weights, rank_fn):
    tau = np.eye(n, dtype=bool)
    memo = weights.copy()
    basis = []
    for _ in range(n - 1):
        gi, gj = rank_fn(tau.shape[0], tau, memo)
        memo[gi] += memo[gj]
        memo[:, gi] += memo[:, gj]
        memo[gi, gi] = 0.0
        mask_clusters = np.arange(memo.shape[0]) != gj
        memo = memo[np.ix_(mask_clusters, mask_clusters)]
        ni, nj = tau[gi].sum(), tau[gj].sum()
        t = 1.0 / np.sqrt((ni + nj) * ni * nj)
        u = t * (nj * tau[gi] - ni * tau[gj])
        basis.append(u)
        union_vec = tau[gi] | tau[gj]
        tau[gi] = union_vec
        tau = tau[np.arange(tau.shape[0]) != gj]

    # add the global constant basis
    basis.append(np.ones(n, dtype=np.float64) / np.sqrt(n))
    return np.column_stack(basis[::-1])


def compute_l1_norm_basis(n: int, weights: np.ndarray) -> np.ndarray:
    """
    Computes a the l1 norm basis using a C++ implementation for efficiency.

    Parameters
    ----------
    n : int
        The number of vertices.
    weights : np.ndarray
        An directed graph represented as a numpy array.

    Returns
    -------
    np.ndarray
        The exact l1 norm basis basis.
    """
    return compute_l1_norm_basis_fast(n, weights)
