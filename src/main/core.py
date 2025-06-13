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


def arg_max(n: int, tau: np.ndarray, memo: np.ndarray) -> tuple[int, int]:
    val = -np.inf
    gi, gj = -1, -1
    sizes = tau.sum(axis=0)
    for i in range(n):
        for j in range(n):
            v = memo[i, j] / (sizes[i] * sizes[j])
            if v > val:
                val = v
                gi, gj = i, j
    return gi, gj


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


if __name__ == "__main__":
    # Example usage
    n = 10
    weights = np.random.rand(n, n)
    weights = (weights + weights.T) / 2  # Make it symmetric
    weights[np.diag_indices(n)] = 0  # Set diagonal to zero
    from src.main.api import arg_max_greedy_undirected

    basis = compute_greedy_basis(n, weights, arg_max_greedy_undirected)
    # print the basis in a readable format
    np.set_printoptions(precision=3, suppress=True)
    from src.main.tools.errors import total_variation

    print("Greedy Basis:")
    for t in basis.T:
        print(f"Total Variation: {total_variation(weights, t):.3f}")
