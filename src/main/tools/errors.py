import numpy as np
from statistics import mean
from typing import Callable

from src.main.utils import create_random_basis_vector


def smoothness(A: np.ndarray, x: np.ndarray) -> float:
    n = x.shape[0]
    return sum(A[i, j] * max(x[i] - x[j], 0) for i in range(n) for j in range(n))


def sparsity(x: np.ndarray) -> float:
    """
    Computes the sparsity of a vector x as the ratio of its non-zero elements
    to the total number of elements in x.
    """
    zero_thresh = 1e-6
    return np.sum(np.abs(x) < zero_thresh) / len(x)


def total_variation(weights: np.ndarray, x: np.ndarray) -> float:
    """
    Computes the total variation l1 norm variation of a vector x.
    """
    return 0.5 * np.sum(np.abs(x[:, None] - x[None, :]) * weights)


def total_l2_variation(weights: np.ndarray, x: np.ndarray) -> float:
    """
    Computes the total variation l2 norm variation of a vector x.
    """
    n = x.shape[0]
    return sum((x[i] - x[j]) ** 2 * weights[i, j] for i in range(n) for j in range(n))


def sum_of_total_variation(weights: np.ndarray, x: np.ndarray) -> float:
    return sum(total_variation(weights, xi) for xi in x.T)


def sum_of_total_l2_variation(x: np.ndarray, weights: np.ndarray) -> float:
    return sum(total_l2_variation(weights, xi) for xi in x.T)


def average_objective_fn(
    basis: np.ndarray, objective_fn: Callable[[np.ndarray], float], trials: int = 100
) -> float:
    """
    Computes the average value of an objective function over a random basis vector.
    The function generates a random vector from the basis span and evaluates the objective
    function for a specified number of trials. The average value is returned.
    """
    return mean(objective_fn(create_random_basis_vector(basis)) for _ in range(trials))


def laplacian_l1_cost(weights: np.ndarray, x: np.ndarray):
    """
    Computes the cost of a vector x with respect to the Laplacian matrix defined by weights.
    The cost is defined as the l1 norm of the Laplacian applied to x.
    """
    laplacian = np.diag(np.sum(weights, axis=1)) - weights
    return np.linalg.norm(laplacian @ x, ord=1)


def laplacian_l1_cost_mean(weights: np.ndarray, basis: np.ndarray, trials: int = 100):
    """
    Computes the mean cost of a vector x with respect to the Laplacian matrix defined by weights.
    The cost is defined as the l1 norm of the Laplacian applied to x, averaged over a number of trials.
    """
    return mean(
        laplacian_l1_cost(weights, create_random_basis_vector(basis))
        for _ in range(trials)
    )
