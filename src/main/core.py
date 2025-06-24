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


def matlab_style_rank_fn(n: int, tau: np.ndarray, weights: np.ndarray):
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


import numpy as np


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


if __name__ == "__main__":
    # Example usage
    n = 10
    import networkx as nx
    from src.main.api import arg_max_greedy_undirected
    from src.main.tools.io import load_graph_from_file, load_basis_vectors_from_file

    G = load_graph_from_file(n, is_directed=False)
    weights = nx.to_numpy_array(G)
    exact_basis = load_basis_vectors_from_file(n, is_directed=False)
    basis = compute_greedy_basis_py(n, weights)
    # print the basis in a readable format
    np.set_printoptions(precision=3, suppress=True)
    from src.main.tools.errors import total_variation

    print("Greedy Basis:")
    xtv = np.apply_along_axis(lambda x: total_variation(weights, x), 0, basis)
    xtv = np.sort(xtv)
    xtv_exact = np.apply_along_axis(
        lambda x: total_variation(weights, x), 0, exact_basis
    )

    for i, b in enumerate(basis.T):
        print(f"TV1: {xtv[i]:.3f}, Exact TV1: {xtv_exact[i]:.3f}")
