import numpy as np
import networkx as nx
from typing import Dict
import matplotlib.pyplot as plt

from src.main.core import compute_greedy_basis
from src.main.tools.io import load_graph_from_file, load_basis_vectors_from_file
from src.main.tools.errors import (
    smoothness,
    total_variation,
    sparsity,
)
from src.main.api import __xrank_fn_directed__, __xrank_fn_undirected__

"""
This script computes the smoothness, sparsity, and total variation errors
for different surrogate bases compared to an exact basis, and generates
a scatter plot of smoothness vs. sparsity trade-offs. The results are saved
to CSV files.
"""

def _compute_errors(
    exact_basis: np.ndarray,
    xbasis: Dict[str, np.ndarray],
    weights: np.ndarray,
    laplacian: np.ndarray,
) -> Dict[str, float]:
    # prepare per‐vector baseline values
    total_var_fn = lambda x: total_variation(weights, x)
    smooth_fn = lambda x: smoothness(laplacian, x)

    xtv_exact = np.apply_along_axis(total_var_fn, 0, exact_basis)
    xsmooth_exact = np.apply_along_axis(smooth_fn, 0, exact_basis)
    xsparsity_exact = np.apply_along_axis(sparsity, 0, exact_basis)

    xmerrors = {
        "L1 Norm": {
            "rms_smoothness": 0.0,
            "max_tv_deviation": 0.0,
            "sum_tv": np.round(np.sum(xtv_exact), 3),
            "sparsity": np.round(np.mean(xsparsity_exact), 3),
            "max_sparsity_dev": 0.0,
            "rms_sparsity_error": 0.0,
        }
    }

    for rank_name, basis in xbasis.items():
        # compute per‐vector metrics
        xtv = np.apply_along_axis(total_var_fn, 0, basis)
        xsmooth = np.apply_along_axis(smooth_fn, 0, basis)
        xsparsity = np.apply_along_axis(sparsity, 0, basis)

        # smoothness errors
        rms_smoothness = float(np.sqrt(np.mean((xsmooth - xsmooth_exact) ** 2)))
        max_tv_deviation = float(np.max(np.abs(xtv_exact - xtv)))
        sum_tv = float(np.sum(xtv))

        # sparsity errors
        deltas = np.abs(xsparsity - xsparsity_exact)
        max_sparsity_dev = float(np.max(deltas))
        rms_sparsity_err = float(np.sqrt(np.mean(deltas**2)))
        sparsity_mean = float(np.mean(xsparsity))

        xmerrors[rank_name] = {
            "rms_smoothness": np.round(rms_smoothness, 3),
            "max_tv_deviation": np.round(max_tv_deviation, 3),
            "sum_tv": np.round(sum_tv, 3),
            "sparsity": np.round(sparsity_mean, 3),
            "max_sparsity_dev": np.round(max_sparsity_dev, 3),
            "rms_sparsity_error": np.round(rms_sparsity_err, 3),
        }

    return xmerrors


def main():
    N = 10
    for is_directed in [False, True]:
        dom = "directed" if is_directed else "undirected"
        xrank_fn = __xrank_fn_directed__ if is_directed else __xrank_fn_undirected__
        exact_basis = load_basis_vectors_from_file(N, is_directed=is_directed)
        G = load_graph_from_file(N, is_directed=is_directed)
        laplacian = nx.laplacian_matrix(G).toarray()
        weights = nx.to_numpy_array(G, weight="weight")

        # compute all bases
        xbasis = {
            rank_name: compute_greedy_basis(N, weights, rank_fn)
            for rank_name, rank_fn in xrank_fn.items()
        }

        # compute errors
        errors = _compute_errors(exact_basis, xbasis, weights, laplacian)

        # write summary CSV including new sparsity‐error columns
        filename = f"plots/visualizations/{dom}/summary_{N}.csv"
        with open(filename, "w") as f:
            f.write(
                "rank_name,rms_smoothness,max_tv_deviation,sum_tv,"
                "sparsity,max_sparsity_dev,rms_sparsity_error\n"
            )
            for rank_name, vals in errors.items():
                f.write(
                    f"{rank_name},"
                    f"{vals['rms_smoothness']},"
                    f"{vals['max_tv_deviation']},"
                    f"{vals['sum_tv']},"
                    f"{vals['sparsity']},"
                    f"{vals['max_sparsity_dev']},"
                    f"{vals['rms_sparsity_error']}\n"
                )

        # generate smoothness‐vs‐sparsity Pareto scatter
        xsmooth_exact = np.apply_along_axis(
            lambda x: smoothness(laplacian, x), 0, exact_basis
        )
        xsparsity_exact = np.apply_along_axis(sparsity, 0, exact_basis)
        plt.scatter(
            xsmooth_exact, xsparsity_exact, marker="x", label="L1 Norm", color="black"
        )

        # plot each surrogate
        for rank_name, basis in xbasis.items():
            xsmooth = np.apply_along_axis(lambda x: smoothness(laplacian, x), 0, basis)
            xsparsity = np.apply_along_axis(sparsity, 0, basis)
            plt.scatter(xsmooth, xsparsity, label=rank_name)

        plt.xlabel("Smoothness")
        plt.ylabel("Sparsity")
        plt.title(f"Smoothness vs Sparsity Trade‐Off ({dom.capitalize()})")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"plots/visualizations/{dom}/pareto_{N}.png")
        plt.show()


if __name__ == "__main__":
    main()
