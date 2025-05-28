import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Callable, List, Dict
import os

from src.root.utils import create_random_graph
from src.root.api import __xrank_fn_undirected__
from src.root.core import compute_greedy_basis
import src.root.tools.errors as errors

"""This script visualizes the basis vectors of a random graph for multiple objectives.
For each iteration, it computes the basis vectors using a greedy algorithm and shows 
the basis vectors as a heatmap on the graph, with color coding for the values."""

if not os.path.exists("plots/temp"):
    os.makedirs("plots/temp")


def _run(
    compute_basis: Callable[[int, np.ndarray, Callable], np.ndarray],
    xrank_fn: Dict[str, Callable],
    num_nodes: int,
    basis_indices: List[int],
    layout: str = "kamada",
    annotate_values: bool = False,
    save_path: str = None,
    mode: str = "grid",  # "grid" or "row"
    share_colorbar: bool = False,  # whether to use one common colorbar
    single_title: bool = False,
) -> None:
    """
    Visualize basis vectors for multiple objectives, either in a grid or a single row.
    """
    weights = create_random_graph(num_nodes, is_directed=False)
    G = nx.from_numpy_array(weights)
    pos = {
        "spring": nx.spring_layout(G, seed=42),
        "kamada": nx.kamada_kawai_layout(G),
        "circular": nx.circular_layout(G),
        "spectral": nx.spectral_layout(G),
    }[layout]
    for i in basis_indices:
        n_obj = len(xrank_fn)
        if mode == "grid":
            n_cols = 2
            n_rows = int(np.ceil(n_obj / n_cols))
        elif mode == "row":
            n_rows = 1
            n_cols = n_obj
        else:
            raise ValueError("mode must be 'grid' or 'row'")
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False
        )
        axs = axs.flatten()
        all_values = []
        results = []
        for name, rank_fn in xrank_fn.items():
            basis = compute_basis(num_nodes, weights, rank_fn)
            b_i = basis[:, i]
            smooth = errors.smoothness(nx.laplacian_matrix(G).toarray(), b_i)
            spars = errors.sparsity(b_i)
            l1n = np.linalg.norm(b_i, 1)
            results.append((name, b_i, smooth, spars, l1n))
            if share_colorbar:
                all_values.append(b_i)
        if share_colorbar:
            all_stack = np.hstack(all_values)
            vmin, vmax = -np.max(np.abs(all_stack)), np.max(np.abs(all_stack))
        for j, (name, b_i, smooth, spars, l1n) in enumerate(results):
            ax = axs[j]
            if not share_colorbar:
                vmin, vmax = -np.max(np.abs(b_i)), np.max(np.abs(b_i))
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
            nodes = nx.draw_networkx_nodes(
                G,
                pos,
                node_color=b_i,
                cmap=plt.cm.coolwarm,
                node_size=200,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
            )
            if annotate_values:
                for node, val in enumerate(b_i):
                    ax.text(
                        pos[node][0],
                        pos[node][1] + 0.03,
                        f"{val:.2f}",
                        ha="center",
                        fontsize=7,
                    )
            info = f"Smooth: {smooth:.2f}\nSpars: {spars:.2f}\nL1-norm: {l1n:.2f}"
            ax.text(
                0.05,
                0.95,
                info,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7),
            )
            if not single_title or i == 0:
                ax.set_title(name, fontsize=10)
            ax.axis("off")

            if not share_colorbar:
                fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
        if share_colorbar:
            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # x, y, width, height
            fig.colorbar(nodes, cax=cax)
        for k in range(n_obj, len(axs)):
            axs[k].axis("off")
        fig.suptitle(f"Basis Vector {i+1}", fontsize=16)
        fig.subplots_adjust(
            left=0.05,
            right=0.9 if share_colorbar else 0.98,
            top=0.92,
            bottom=0.08,
            wspace=0.3,
            hspace=0.3,
        )
        if save_path:
            fname = f"basis_vector_{i+1}_{mode}"
            fig.savefig(os.path.join(save_path, fname + ".png"), dpi=300)


def main(num_vertices: int = 10):
    print("Running visualization of basis vectors for multiple objectives...")
    SAVE_PATH = "plots/temp"

    _run(
        compute_basis=compute_greedy_basis,
        xrank_fn=__xrank_fn_undirected__,
        num_nodes=num_vertices,
        basis_indices=range(num_vertices),
        layout="kamada",
        annotate_values=False,
        save_path=SAVE_PATH,
        mode="grid",  # "grid" or "row"
        share_colorbar=True,
        single_title=True,
    )
    print("Visualization completed. Check the 'plots/temp' directory for results.")


if __name__ == "__main__":
    main()
