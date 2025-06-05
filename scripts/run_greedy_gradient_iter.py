import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import os

from src.main.utils import create_random_geometric_graph
from src.main.subgradient import (
    _compute_greedy_subgradient,
    sequences,
)

"""This script runs a greedy subgradient method for different sequences and plots the 
number of iterations required as a function of the number of nodes in the graph."""

if not os.path.exists("plots/temp"):
    os.makedirs("plots/temp")


def _plot_num_iterations_vs_n(xiters: Dict[str, List[int]], xaxis: List[int]):
    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax = plt.subplots(figsize=(6, 4))
    for method_name, iters in xiters.items():
        ax.plot(xaxis, iters, label=method_name.replace("-", " ").title(), marker="o")

    ax.set_xlabel("N", fontsize=11)
    ax.set_ylabel("Number of Iterations", fontsize=11)
    ax.set_title("Greedy Subgradient Iterations vs. Number of Nodes", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(xaxis, rotation=45)
    plt.xlim(min(xaxis), max(xaxis))
    plt.ylim(0, max(max(iters) for iters in xiters.values()) + 10)
    plt.gca().set_aspect("auto", adjustable="box")
    plt.tight_layout()
    file_name = "plots/temp/greedy_subgradient_iterations_vs_n.png"
    plt.savefig(file_name, dpi=300)
    print(f"Plot saved to '{file_name}'")
    plt.show()
    plt.close()


def main():
    print("Running greedy subgradient iterations analysis...")
    methods = {
        "harmonic": sequences.HARMONIC,
        "log-harmonic": sequences.LOG_HARMONIC,
        "power": sequences.POWER,
    }
    xiters_means = {method_name: [] for method_name in methods}
    xaxis = list(range(20, 201, 10))
    for n in xaxis:
        print(f"Running analysis for N={n}")
        for _ in range(3):
            x0 = np.random.rand(n)
            xiters = {method_name: [] for method_name in methods}
            weights = create_random_geometric_graph(
                n, is_weighted=True, is_directed=True, is_connected=False
            )
            for method_name, seq in methods.items():
                _, k = _compute_greedy_subgradient(
                    weights=weights, x=x0, max_iter=1000000, tol=1e-4, seq=seq
                )
                xiters[method_name].append(k)
        for method_name in methods:
            xiters_means[method_name].append(np.mean(xiters[method_name]))
    _plot_num_iterations_vs_n(xiters_means, xaxis=xaxis)
    print("Greedy subgradient iterations analysis completed.")


if __name__ == "__main__":
    main()
