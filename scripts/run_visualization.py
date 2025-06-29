import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Callable, List, Dict, Optional

from src.main.utils import (
    create_random_geometric_graph,
    convert_to_directed,
    visualize_graph,
)
from src.main.api import __xrank_fn_undirected__, __xrank_fn_directed__
from src.main.core import compute_greedy_basis, compute_l1_norm_basis
from src.main.tools.io import load_basis_vectors_from_file, load_graph_from_file
import src.main.tools.errors as errors

"""
This script visualizes basis vectors on a random geometric graph (or toy graph)
to compare different greedy methods to the exact L1 basis for both directed and 
undirected graphs.
"""

# Ensure output folder exists (we’ll create subfolders under here)
SAVE_DIR = "plots/temp"
os.makedirs(SAVE_DIR, exist_ok=True)


def _plot_single(
    ax: plt.Axes,
    G: nx.Graph,
    pos: Dict[int, np.ndarray],
    data: Dict,
    vmin: float,
    vmax: float,
    annotate_vals: bool,
    single_title: bool,
    font_sizes: Dict[str, int],
):
    b_i = data["vector"]
    method_name = data["name"]
    smooth = data["smoothness"]
    spars = data["sparsity"]
    l1n = data["l1norm"]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999", alpha=0.4, width=0.8)

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=b_i,
        cmap=plt.cm.seismic,
        node_size=300,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.5,
        ax=ax,
    )

    if annotate_vals:
        for node_idx, val in enumerate(b_i):
            x, y = pos[node_idx]
            ax.text(
                x,
                y + 0.03,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=font_sizes["annotate"],
                color="black",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none"),
            )

    info_text = (
        f"{'TV2:':<8}   {smooth:8.2f}\n"
        f"{'Spars:':<8}  {spars:8.2f}\n"
        f"{'L1-norm:':<8}{l1n:8.2f}"
    )
    ax.text(
        -0.18,  # left
        1.1,  # up
        info_text,
        transform=ax.transAxes,
        fontsize=font_sizes["metrics"],
        va="top",
        ha="left",
        color="#111111",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="#f7f7f7",
            edgecolor="#cccccc",
            linewidth=1,
        ),
        zorder=10,
    )

    if not single_title:
        if method_name == "L1-Norm":
            ax.set_title(
                method_name,
                fontsize=font_sizes["method_title"],
                fontweight="bold",
                fontfamily="sans-serif",
                color="#388e3c",
            )
        else:
            ax.set_title(
                method_name,
                fontsize=font_sizes["method_title"],
                fontweight="bold",
                fontfamily="sans-serif",
                color="#1a237e",
            )

    ax.axis("off")
    return nodes


def _choose_random_geometric_graph(num_nodes, is_directed, layout):
    while True:
        weights = create_random_geometric_graph(num_nodes, is_directed=is_directed)
        G = nx.from_numpy_array(weights)

        pos = {
            "spring": nx.spring_layout(G, seed=42),
            "kamada": nx.kamada_kawai_layout(G),
            "circular": nx.circular_layout(G),
            "spectral": nx.spectral_layout(G),
        }.get(layout, nx.kamada_kawai_layout(G))

        plt.figure(figsize=(6, 6))
        nx.draw(
            G,
            pos,
            node_size=50,
            node_color="skyblue",
            edge_color="gray",
            with_labels=False,
            alpha=0.8,
        )
        plt.title(
            f"Random Geometric Graph ({num_nodes} nodes)\n" f"Layout = '{layout}'\n",
            fontsize=10,
        )
        plt.tight_layout()
        plt.show()

        resp = input("\nDo you accept this graph? (y/n): ").strip().lower()
        while resp not in ("y", "n"):
            resp = input("Please type 'y' to accept or 'n' to reject: ").strip().lower()

        if resp == "y":
            plt.close()
            print("Graph accepted. Proceeding with visualization…\n")
            break
        else:
            plt.close()
            print("Graph rejected. Generating a new random geometric graph…\n")
            continue

    return weights, G, pos


def _plot_on_graph(
    compute_basis: Callable[[int, np.ndarray, Callable], np.ndarray],
    xrank_fn: Dict[str, Callable],
    num_nodes: int,
    basis_indices: List[int],
    G: nx.Graph,
    weights: np.ndarray,
    pos: Dict[int, np.ndarray],
    experiment_label: str,
    annotate_values: bool = False,
    save_path: Optional[str] = None,
    mode: str = "grid",
    share_colorbar: bool = False,
    single_title: bool = False,
):
    method_names = list(xrank_fn.keys())
    n_methods = len(method_names)
    font_sizes = {
        "method_title": 14,
        "metrics": 14,
        "annotate": 8,
        "suptitle": 20,
    }

    first_basis_idx = basis_indices[0]

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    all_bases: Dict[str, np.ndarray] = {}
    lap = nx.laplacian_matrix(G).toarray()
    for method_name, rank_fn in xrank_fn.items():
        if method_name == "L1-Norm":
            all_bases[method_name] = load_basis_vectors_from_file(
                N=num_nodes, is_directed=G.is_directed()
            )
        else:
            all_bases[method_name] = compute_basis(num_nodes, weights, rank_fn=rank_fn)
            # Ensure similar basis vectors have the same sign for comparison
            # TODO: Remove this if not needed
            if method_name in ["Product", "Sum", "Max Size"]:
                print("Changing sign for method:", method_name)
                all_bases[method_name][
                    :, 2
                ] *= -1  # Forcing the sign to match the L1-Norm basis

    for idx in basis_indices:
        if mode == "grid":
            ncols = 2
            nrows = int(np.ceil(n_methods / ncols))
        elif mode == "row":
            nrows, ncols = 1, n_methods
        else:
            raise ValueError("mode must be either 'grid' or 'row'")

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 5 * nrows),
            squeeze=False,
            facecolor="#f7f7f7",
        )
        axes_flat = axes.flatten()

        # Build results by extracting column `idx` from each precomputed basis
        results = []
        for method_name in method_names:
            basis = all_bases[method_name]
            b_i = basis[:, idx]
            smooth = errors.smoothness(lap, b_i)
            spars = errors.sparsity(b_i)
            l1n = np.linalg.norm(b_i, 1)
            results.append(
                {
                    "name": method_name,
                    "vector": b_i,
                    "smoothness": smooth,
                    "sparsity": spars,
                    "l1norm": l1n,
                }
            )

        if share_colorbar:
            all_vectors = np.hstack([r["vector"] for r in results])
            vmin, vmax = -np.max(np.abs(all_vectors)), np.max(np.abs(all_vectors))

        last_nodes = None
        is_first_figure = idx == first_basis_idx

        for i, data in enumerate(results):
            ax = axes_flat[i]

            if not share_colorbar:
                vec = data["vector"]
                vmin, vmax = -np.max(np.abs(vec)), np.max(np.abs(vec))

            last_nodes = _plot_single(
                ax=ax,
                G=G,
                pos=pos,
                data=data,
                vmin=vmin,
                vmax=vmax,
                annotate_vals=annotate_values,
                single_title=single_title,
                font_sizes=font_sizes,
            )

            if is_first_figure:
                method_name = method_names[i]
                if method_name == "L1-Norm":
                    # Set red for L1-Norm title
                    ax.set_title(
                        method_name,
                        fontsize=font_sizes["method_title"],
                        fontweight="bold",
                        fontfamily="sans-serif",
                        color="#d32f2f",  # Red color
                    )
                else:
                    # Set blue color for other methods
                    ax.set_title(
                        method_name,
                        fontsize=font_sizes["method_title"],
                        fontweight="bold",
                        fontfamily="sans-serif",
                        color="#1a237e",  # Dark blue color
                    )
            elif single_title:
                ax.set_title("", fontsize=0)

        for j in range(n_methods, nrows * ncols):
            axes_flat[j].axis("off")

        if share_colorbar and last_nodes is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(last_nodes, cax=cbar_ax, label="Basis value")

        fig.suptitle(
            f"{experiment_label} | Basis Vector {idx + 1}",
            fontsize=font_sizes["suptitle"],
            fontweight="bold",
            y=0.95,
        )
        fig.subplots_adjust(
            left=0.03,
            right=0.9 if share_colorbar else 0.98,
            top=0.80,
            bottom=0.05,
            wspace=0.25,
            hspace=0.25,
        )

        if save_path:
            fname = f"basis_vector_{idx + 1}_{mode}.png"
            fullpath = os.path.join(save_path, fname)
            fig.savefig(fullpath, dpi=300)
            print(f"Saved: {fullpath}")

        plt.close(fig)


def get_positions(G: nx.Graph, layout: str) -> Dict[int, np.ndarray]:
    return {
        "spring": nx.spring_layout(G, seed=42),
        "kamada": nx.kamada_kawai_layout(G),
        "circular": nx.circular_layout(G),
        "spectral": nx.spectral_layout(G),
    }.get(layout, nx.kamada_kawai_layout(G))


def main(num_vertices: int = 10):
    layout = "spring"  # "spring", "circular", "spectral" or "kamada"

    print("1) Generating and previewing an undirected random‐geometric graph…")
    weights_und, G_und, pos = _choose_random_geometric_graph(
        num_nodes=num_vertices,
        is_directed=False,
        layout=layout,
    )

    for v in G_und.nodes():
        G_und.nodes[v]["pos"] = pos[v]

    und_dir = os.path.join(SAVE_DIR, "undirected")
    os.makedirs(und_dir, exist_ok=True)

    print(
        "2) Plotting all basis‐vectors on the undirected graph (xrank_fn_undirected)…"
    )
    xrank_fn_undirected = {"L1-Norm": None} | __xrank_fn_undirected__

    _plot_on_graph(
        compute_basis=compute_greedy_basis,
        xrank_fn=xrank_fn_undirected,
        num_nodes=num_vertices,
        basis_indices=list(range(num_vertices)),
        G=G_und,
        weights=weights_und,
        pos=pos,
        experiment_label="UNDIRECTED",
        annotate_values=False,
        save_path=und_dir,
        mode="row",  # "grid" or "row"
        share_colorbar=True,
        single_title=True,
    )

    print("3) Converting to directed, then plotting with __xrank_fn_directed__…")
    G_dir = convert_to_directed(G_und)
    weights_dir = nx.to_numpy_array(G_dir, weight="weight")

    dir_dir = os.path.join(SAVE_DIR, "directed")
    os.makedirs(dir_dir, exist_ok=True)
    xrank_fn_directed = {"L1-Norm": None} | __xrank_fn_directed__
    _plot_on_graph(
        compute_basis=compute_greedy_basis,
        xrank_fn=xrank_fn_directed,
        num_nodes=num_vertices,
        basis_indices=list(range(num_vertices)),
        G=G_dir,
        weights=weights_dir,
        pos=pos,
        experiment_label="DIRECTED",
        annotate_values=False,
        save_path=dir_dir,
        mode="row",
        share_colorbar=True,
        single_title=True,
    )

    print("Done! Check:")
    print(f"  – Undirected plots: {und_dir}/")
    print(f"  – Directed plots: {dir_dir}/")


def toy_graph():
    """An example of how to visualize basis vectors on a toy graph."""
    num_vertices = 10
    G_und = load_graph_from_file(N=num_vertices, is_directed=False)
    weights_und = nx.to_numpy_array(G_und, weight="weight")
    print("Loaded undirected graph from file.")
    while True:
        visualize_graph(
            G_und,
            "toy_graph",
        )
        if (
            input("Do you want to use the current layout? (y/n): ").strip().lower()
            == "y"
        ):
            break
        pos = {
            v: ((1 + x) / 2, (1 + y) / 2)
            for v, (x, y) in nx.spring_layout(G_und).items()
        }
        nx.set_node_attributes(G_und, pos, "pos")

    und_dir = os.path.join(SAVE_DIR, "undirected")
    os.makedirs(und_dir, exist_ok=True)

    xrank_fn_undirected = {"L1-Norm": None} | __xrank_fn_undirected__
    print("Plotting all basis-vectors on the undirected graph...")
    _plot_on_graph(
        compute_basis=compute_greedy_basis,
        xrank_fn=xrank_fn_undirected,
        num_nodes=num_vertices,
        basis_indices=list(range(num_vertices)),
        G=G_und,
        weights=weights_und,
        pos=nx.get_node_attributes(G_und, "pos"),
        experiment_label="UNDIRECTED",
        annotate_values=False,
        save_path=und_dir,
        mode="row",
        share_colorbar=True,
        single_title=True,
    )

    G_dir = load_graph_from_file(N=num_vertices, is_directed=True)
    weights_dir = nx.to_numpy_array(G_dir, weight="weight")
    print("Loaded directed graph from file.")

    dir_dir = os.path.join(SAVE_DIR, "directed")
    os.makedirs(dir_dir, exist_ok=True)
    xrank_fn_directed = {"L1-Norm": None} | __xrank_fn_directed__
    print("Plotting all basis-vectors on the directed graph...")
    _plot_on_graph(
        compute_basis=compute_greedy_basis,
        xrank_fn=xrank_fn_directed,
        num_nodes=num_vertices,
        basis_indices=list(range(num_vertices)),
        G=G_dir,
        weights=weights_dir,
        pos=nx.get_node_attributes(G_dir, "pos"),
        experiment_label="DIRECTED",
        annotate_values=False,
        save_path=dir_dir,
        mode="row",
        share_colorbar=True,
        single_title=True,
    )

    print("Done! Check:")
    print(f" - Undirected plots: {und_dir}/")
    print(f" - Directed plots: {dir_dir}/")


if __name__ == "__main__":
    toy_graph()
