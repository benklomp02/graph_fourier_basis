import tqdm
from typing import Callable, List, Dict, Tuple
import matplotlib.pyplot as plt

from src.root.core import compute_greedy_basis
from src.root.utils import create_random_graph
import src.root.tools.errors as errors
from src.root.api import (
    __xrank_fn_directed__,
    __xrank_fn_undirected__,
)


"""
This script evaluates the different greedy-rank functions on random directed graphs.
The goal is to find a good apprximation of the Laplacian cost function.
"""


def _plot_total_variation(
    outputs: Dict[str, List[float]], xnum_verts: List[int], is_directed: bool
):
    plt.figure(figsize=(10, 5))
    for name, vals in outputs.items():
        plt.plot(xnum_verts, vals, marker="o", label=name)
    plt.title("Total Variation vs Number of Vertices")
    plt.xlabel("N")
    plt.ylabel("L1 Norm Variation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    file_name = f"plots/temp/objective_analysis_{'directed' if is_directed else 'undirected'}.png"
    plt.savefig(
        f"plots/temp/objective_analysis_{"directed" if is_directed else "undirected"}.png",
        bbox_inches="tight",
    )
    print(f"Analysis complete. Figure saved to '{file_name}'.\n")
    plt.show()
    plt.close()


def _run(
    n: int,
    num_trials: int,
    show_progress: bool,
    xrank_fn: Dict[str, List[Callable]],
    is_directed: bool,
) -> Tuple[Dict[str, float], Dict[str, List[List[float]]]]:
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


def main():
    print("Starting objective analysis for greedy-rank functions...")
    for rank_fn, is_directed in [
        (__xrank_fn_directed__, True),
        (__xrank_fn_undirected__, False),
    ]:
        if is_directed:
            print("Running analysis for directed graphs...")
        else:
            print("Running analysis for undirected graphs...")
    xnum_verts = list(range(20, 201, 20))
    num_trials = 3
    show_progress = True
    total_outputs: Dict[str, List[float]] = {name: [] for name in rank_fn.keys()}
    for N in xnum_verts:
        avg_var = _run(
            n=N,
            num_trials=num_trials,
            show_progress=show_progress,
            xrank_fn=rank_fn,
            is_directed=is_directed,
        )
        for name, val in avg_var.items():
            total_outputs[name].append(val)
    _plot_total_variation(total_outputs, xnum_verts, is_directed)
    print("Objective analysis completed.")


if __name__ == "__main__":
    main()
