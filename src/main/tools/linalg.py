import numpy as np

# import null space computation
from scipy.linalg import null_space


def solve_minimisation_problem(M, U):
    X = U.T @ M
    ns = null_space(X)
    if ns.shape[1] != 1:
        raise ValueError(f"Null space has dimension {ns.shape[1]}")
    null_vec = ns[:, 0]
    x = M @ null_vec
    x /= np.linalg.norm(x)
    return x


def build_partition(x: np.ndarray):
    """
    Build a partition from vector x: x = M @ a.
    """
    x = x.ravel()
    n = x.shape[0]
    a = np.unique(x)
    a.sort()
    m = len(a)
    M = np.zeros((n, m))
    for i in range(n):
        j = np.where(a == x[i])[0][0]
        M[i, j] = 1
    return M, a


def _compute_partition_matrices(n, m, i, free, to_be_used, M):
    """
    Recursively yield all partition matrices.
    """
    if i == n:
        yield M.copy()
        return

    for j in range(m):
        if (to_be_used >> j) & 1:
            M[i, j] = 1
            yield from _compute_partition_matrices(
                n, m, i + 1, free ^ (1 << j), to_be_used ^ (1 << j), M
            )
            M[i, j] = 0

    deg_freedom = to_be_used.bit_count()
    if n - i > deg_freedom:
        for j in range(m):
            if (free >> j) & 1:
                M[i, j] = 1
                yield from _compute_partition_matrices(n, m, i + 1, free, to_be_used, M)
                M[i, j] = 0


def get_all_partition_matrices(n, m):
    """
    Generate all partition matrices of size n x m.
    """
    M = np.zeros((n, m))
    yield from _compute_partition_matrices(n, m, 0, 0, (1 << m) - 1, M)
