import numpy as np
from typing import Callable, Optional
from src.main.tools.linalg import build_partition

objective_t = Callable[[np.ndarray, np.ndarray, np.ndarray, Optional[float]], float]


def laplacian_cost(weights, M, a, scaling=1.0) -> float:
    # Computes the exact Laplacian cost
    x = M @ a
    D = weights.sum(axis=1)
    diff = x * D - weights @ x
    return np.abs(diff).sum() * scaling


def rank_fn_lap(weights, x):
    M, a = build_partition(x)
    return laplacian_cost(weights, M, a)


def laplacian_cost_approx_by_triangle_inequality(weights, M, a, scaling=1.0):
    # Computes an upper bound for the Laplacian cost using triangle inequality
    Wm = M.T @ weights @ M
    W_sym = Wm + Wm.T
    np.fill_diagonal(W_sym, 0)
    lower = np.tril(W_sym, -1).sum(axis=1)
    upper = np.triu(W_sym, +1).sum(axis=1)
    f_vec = lower - upper
    return float(f_vec @ a) * scaling


def rank_fn_ub(weights, x):
    M, a = build_partition(x)
    return laplacian_cost_approx_by_triangle_inequality(weights, M, a)


def laplacian_cost_approx_by_median_split(weights, M, a, scaling=1.0):
    # Approximates Laplacian cost by splitting at the median of a
    m = M.shape[1]
    Wm = M.T @ weights @ M
    t = m // 2
    row_sum = Wm.sum(axis=1)
    sp = a[t + 1 :].dot(row_sum[t + 1 :]) - (Wm[t + 1 :, :] @ a).sum()
    sm = a[:t].dot(row_sum[:t]) - (Wm[:t, :] @ a).sum()
    return (sp - sm) * scaling


def rank_fn_median(weights, x):
    M, a = build_partition(x)
    return laplacian_cost_approx_by_median_split(weights, M, a)


def laplacian_cost_approx_by_mean_split(weights, M, a, scaling=1.0):
    # Approximates Laplacian cost by splitting at the mean of a
    Wm = M.T @ weights @ M
    t = np.abs(a - a.mean()).argmin()
    row_sum = Wm.sum(axis=1)
    # sp = a[:t].dot(row_sum[:t]) - (Wm[:t, :] @ a).sum()
    # sm = a[t + 1 :].dot(row_sum[t + 1 :]) - (Wm[t + 1 :, :] @ a).sum()
    sp = a[t + 1 :].dot(row_sum[t + 1 :]) - (Wm[t + 1 :, :] @ a).sum()
    sm = a[: t + 1].dot(row_sum[: t + 1]) - (Wm[: t + 1, :] @ a).sum()
    return (sp - sm) * scaling


def rank_fn_mean(weights, x):
    M, a = build_partition(x)
    return laplacian_cost_approx_by_mean_split(weights, M, a)


def laplacian_cost_approx_by_majority(weights, M, a, scaling=1.0):
    # Approximates Laplacian cost by a weighted sign heuristic
    M_inv = np.argmax(M, axis=1)
    W_out = weights.sum(axis=1)
    A = a[M_inv] * W_out
    B = weights.dot(a[M_inv])
    beta = np.where(A > B, 1, -1)
    cluster_size = M.sum(axis=0)
    alpha = np.where(M.T.dot(beta) > cluster_size / 2, 1, -1)
    Wm = M.T @ weights @ M
    diff = a[:, None] - a[None, :]
    return float((alpha[:, None] * diff * Wm).sum()) * scaling


def rank_fn_majority(weights, x):
    M, a = build_partition(x)
    return laplacian_cost_approx_by_majority(weights, M, a)


__xrank_fn_sur__ = {
    "Triangle Inequality": rank_fn_ub,
    "Majority Heuristic": rank_fn_majority,
    "Median Split": rank_fn_median,
    "Mean Split": rank_fn_mean,
}
