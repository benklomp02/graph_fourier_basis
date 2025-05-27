import tqdm
from typing import Callable, List, Dict, Tuple

from src.graph_fourier_transform.core import compute_greedy_basis
from src.graph_fourier_transform.utils import create_random_graph
import src.graph_fourier_transform.tools.errors as errors
from src.graph_fourier_transform.api import (
    __xrank_fn_directed__,
    __xrank_fn_undirected__,
)
from src.graph_fourier_transform.tools.plotting import (
    plot_total_variation,
)

"""
This script evaluates the different greedy-rank functions on random directed graphs.
The goal is to find a good apprximation of the Laplacian cost function.
"""


def _run(
    n: int,
    num_trials: int,
    show_progress: bool,
    xrank_fn: Dict[str, List[Callable]],
    is_directed: bool,
) -> Tuple[Dict[str, float], Dict[str, List[List[float]]]]:
    """
    Evaluate each greedy-rank function on random directed graphs.

    Returns
    -------
    avg_variation: Dict[str, float]
        Average total L1 variation per method.
    traces: Dict[str, List[List[float]]]
        Per-trial variation traces (cluster-count vs variation) per method.
    """
    # Initialize accumulators
    variation_sum: Dict[str, float] = {name: 0.0 for name in xrank_fn.keys()}

    trials = range(num_trials)
    if show_progress:
        trials = tqdm.tqdm(trials, desc=f"Analysis N={n}")

    for trial in trials:
        weights = create_random_graph(num_nodes=n, seed=trial, is_directed=is_directed)
        for name, fn in xrank_fn.items():
            basis = compute_greedy_basis(n, weights, fn)
            variation_sum[name] += errors.l1_variation(basis)

    avg_variation = {name: variation_sum[name] / num_trials for name in variation_sum}
    return avg_variation


def run_analysis(rank_fn: Dict[str, Callable], is_directed: bool, file_name: str):
    xnum_verts = list(range(20, 201, 20))
    num_trials = 3
    show_progress = True
    # Collect average variation per method across different graph sizes
    total_outputs: Dict[str, List[float]] = {name: [] for name in rank_fn.keys()}

    for N in xnum_verts:
        avg_var = _run(
            n=N,
            num_trials=num_trials,
            show_progress=show_progress,
            xrank_fn=rank_fn,
            is_directed=is_directed,
        )
        # Append for total variation vs N plot
        for name, val in avg_var.items():
            total_outputs[name].append(val)

    plot_total_variation(total_outputs, xnum_verts, file_name=file_name)
    print(f"Analysis complete. Figure saved to '{file_name}'.\n")


if __name__ == "__main__":
    run_analysis(
        __xrank_fn_directed__,
        is_directed=True,
        file_name="objective_analysis_directed.png",
    )
    run_analysis(
        __xrank_fn_undirected__,
        is_directed=False,
        file_name="objective_analysis_undirected.png",
    )
