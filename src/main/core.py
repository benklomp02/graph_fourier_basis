import numpy as np
from line_profiler import profile

from src.main.api import compute_l1_norm_basis_fast, __objective_t__


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


def greedy_rank_fn(n: int, tau: np.ndarray, weights: np.ndarray):
    sizes = tau.sum(axis=1)
    R = weights / (sizes[:, None] * sizes[None, :])
    np.fill_diagonal(R, -np.inf)  # prevent self-merges
    i, j = np.unravel_index(np.argmax(R), R.shape)
    return (min(i, j), max(i, j))


@profile
def _compute_greedy_basis(n, weights, rank_fn):
    # Initialize the tau matrix as an identity matrix
    tau = np.eye(n, dtype=bool)
    # Initialize the memoization matrix to store W(A_i, A_j)
    memo = weights.copy()
    basis = []
    for _ in range(n - 1):
        # Find the two groups to merge
        gi, gj = rank_fn(tau.shape[0], tau, memo)  # Fast C implementation
        # We merge gj into gi and update the memoization matrix
        memo[gi] += memo[gj]
        memo[:, gi] += memo[:, gj]
        memo[gi, gi] = 0.0
        mask_clusters = np.arange(memo.shape[0]) != gj
        memo = memo[np.ix_(mask_clusters, mask_clusters)]
        # We compute the new basis vector
        ni, nj = tau[gi].sum(), tau[gj].sum()  # Number of vertices in the two groups
        t = 1.0 / np.sqrt((ni + nj) * ni * nj)  # Normalization factor
        u = t * (nj * tau[gi] - ni * tau[gj])  # New basis vector
        basis.append(u)
        # Update the groups in tau
        union_vec = tau[gi] | tau[gj]
        tau[gi] = union_vec
        tau = tau[np.arange(tau.shape[0]) != gj]
    first_vec = np.ones(n, dtype=float) / np.sqrt(n)  # First basis vector
    basis.append(first_vec)
    return np.column_stack(basis[::-1])


def compute_greedy_basis_py(n: int, weights: np.ndarray) -> np.ndarray:
    """
    Computes the greedy basis of an undirected graph using a Python implementation.

    Parameters:
        weights : np.ndarray
            An (n x n) symmetric weight matrix of the graph.

    Returns:
        np.ndarray: An (n x n) matrix where each column is a basis vector.
    """
    W = weights.copy()
    n = W.shape[0]

    labels = np.arange(n)
    s = np.ones(n, dtype=int)
    basis = []

    for _ in range(n - 1):
        z = np.outer(s, s)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(z > 0, W / z, -np.inf)
        np.fill_diagonal(R, -np.inf)

        mi, mj = divmod(np.argmax(R), n)
        if mi > mj:
            mi, mj = mj, mi  # Enforce mi < mj

        A = np.where(labels == mi)[0]
        B = np.where(labels == mj)[0]
        ni, nj = len(A), len(B)
        z = ni + nj
        t = 1.0 / np.sqrt(ni * nj * z)

        u = np.zeros(n)
        u[A] = -nj * t
        u[B] = ni * t
        basis.append(u)

        W[mi, :] += W[mj, :]
        W[:, mi] += W[:, mj]
        W[mi, mi] = 0
        W[mj, :] = 0
        W[:, mj] = 0

        s[mi] += s[mj]
        s[mj] = -1
        labels[labels == mj] = mi

    # Add final constant vector
    u1 = np.ones(n) / np.sqrt(n)
    basis.append(u1)

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
    return compute_l1_norm_basis_fast(n, weights)  # Fast C++ implementation


from src.main.tools.linalg import (
    get_all_partition_matrices,
    solve_minimisation_problem,
)


# Sole purpose of this function is to compute the Laplacian l1 norm cost basis
def compute_exact_basis_py(n: int, weights: np.ndarray, obj) -> np.ndarray:
    """
    Computes the l1 norm basis using a Python implementation.

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
    u1 = np.ones(n) / np.sqrt(n)
    basis = [u1]
    _expand_first_basis_vector(basis, n, weights, obj)
    for k in range(3, 1 + n):
        _expand_basis_set(basis, k, weights, n, obj)
    return np.column_stack(basis)


def _expand_first_basis_vector(basis, n, weights, obj):
    """
    Expand the basis set by finding the best first vector.
    """
    best_u = None
    best_score = np.inf
    for M in get_all_partition_matrices(n, 2):
        c1 = M[:, 0].sum()
        c2 = M[:, 1].sum()
        a = np.array([1.0, -c1 / c2])
        x = M @ a
        x = x / np.linalg.norm(x)
        score = obj(weights=weights, x=x)
        if score < best_score:
            best_score = score
            best_u = x
    basis.append(best_u)


def _expand_basis_set(basis, k, weights, n, obj):
    """
    Expand the basis set by finding the best new vector.
    """
    U = np.column_stack(basis)
    best_u = None
    best_score = np.inf

    for j in range(2, k + 1):
        for M in get_all_partition_matrices(n, j):
            A = U.T @ M
            rank = np.linalg.matrix_rank(A)
            if A.shape[1] - rank != 1:
                continue
            uk = solve_minimisation_problem(M, U)
            score = obj(weights=weights, x=uk)
            if score < best_score:
                best_score = score
                best_u = uk

    basis.append(best_u)
