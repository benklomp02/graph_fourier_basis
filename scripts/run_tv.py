import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Dict, Optional
from tqdm import tqdm

from src.main.tools.errors import total_variation
from src.main.tools.io import load_graph_from_file, load_basis_vectors_from_file
from src.main.utils import (
    create_random_geometric_graph,
    create_random_erdos_renyi_graph,
)
from src.main.core import compute_greedy_basis
from src.main.api import __xrank_fn_directed__, __xrank_fn_undirected__

__nexact = "L1-Norm Basis"
__sN = 10
__lN = 200
__mcreate_random_graphs = {
    "geometric-0.2": lambda n: create_random_geometric_graph(n, 0.2),
    "geometric-0.5": lambda n: create_random_geometric_graph(n, 0.5),
    "erdos_renyi-0.2": lambda n: create_random_erdos_renyi_graph(n, 0.2),
    "erdos_renyi-0.5": lambda n: create_random_erdos_renyi_graph(n, 0.5),
    "erdos_renyi-0.8": lambda n: create_random_erdos_renyi_graph(n, 0.8),
}

__mcreate_random_graphs_directed = {
    "geometric-0.2": lambda n: create_random_geometric_graph(n, 0.2, is_directed=True),
    "geometric-0.5": lambda n: create_random_geometric_graph(n, 0.5, is_directed=True),
    "erdos_renyi-0.2": lambda n: create_random_erdos_renyi_graph(
        n, 0.2, is_directed=True
    ),
    "erdos_renyi-0.5": lambda n: create_random_erdos_renyi_graph(
        n, 0.5, is_directed=True
    ),
    "erdos_renyi-0.8": lambda n: create_random_erdos_renyi_graph(
        n, 0.8, is_directed=True
    ),
}


def _ensure_folder(is_directed: bool) -> str:
    """Ensure that `plots/graph/` or `plots/digraph/` exists, then return its path."""
    folder = f"plots/{'digraph' if is_directed else 'graph'}/"
    os.makedirs(folder, exist_ok=True)
    return folder


def _filename(
    name_suffix: str,
    is_directed: bool,
    gtype: Optional[str],
    is_mean: bool,
) -> str:
    folder = _ensure_folder(is_directed)
    if gtype is None:
        direction = "directed" if is_directed else "undirected"
        return folder + f"tv_small_{direction}.png"
    else:
        direction = "directed" if is_directed else "undirected"
        name, p = gtype.split("-")
        name = name.replace("_", "-").title()
        return folder + f"tv_{name}_{int(float(p) * 100)}_{direction}.png".lower()


def _plot(
    tv_dict: Dict[str, np.ndarray],
    is_directed: bool,
    title_tag: str,
    ntarget: Optional[str] = None,
    gtype: Optional[str] = None,
    is_mean: bool = False,
    step: int = 1,
    save_fig: bool = False,
):
    if not tv_dict:
        return

    if ntarget is not None and ntarget not in tv_dict:
        raise ValueError(
            f"Target method '{ntarget}' not found; available keys: {list(tv_dict.keys())}"
        )

    N = next(iter(tv_dict.values())).shape[0]
    full_indices = np.arange(1, N + 1)
    subsampled_indices = full_indices[::step]

    plt.style.use("seaborn-v0_8-talk")

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#f7f7f7")
    ax.set_facecolor("#fafafa")
    ax.set_axisbelow(True)

    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.minorticks_on()

    cmap = plt.get_cmap("tab10")
    non_target_methods = [m for m in tv_dict.keys() if m != ntarget]

    color_map: Dict[str, str] = {}
    for idx, method in enumerate(non_target_methods):
        color_map[method] = cmap(idx % 10)

    if ntarget is not None:
        color_map[ntarget] = "black"

    for method, tv_array in tv_dict.items():
        tv_sub = tv_array[::step]

        if method == ntarget:
            ax.plot(
                subsampled_indices,
                tv_sub,
                marker="o",
                markersize=7,
                linestyle="-",
                linewidth=2.5,
                color="black",
                alpha=0.95,
                zorder=5,
                label=method,
            )
        else:
            ax.plot(
                subsampled_indices,
                tv_sub,
                marker="o",
                markersize=6,
                linestyle="-",
                linewidth=1.8,
                color=color_map[method],
                alpha=0.85,
                label=method,
            )

    all_y = np.concatenate([tv_array[::step] for tv_array in tv_dict.values()])
    y_min, y_max = all_y.min(), all_y.max()
    y_range = y_max - y_min
    # Increase the top by 20% of the range. Leave bottom at y_min (or adjust if needed).
    ax.set_ylim(y_min, y_max + 0.30 * y_range)

    prefix = "Directed" if is_directed else "Undirected"
    mean_tag = " (mean)" if is_mean else ""
    if gtype is None:
        gtype_tag = f"({title_tag})"
    else:
        name, p = gtype.split("-")
        name = name.replace("_", "-").title()
        gtype_tag = f"[{name}({p})]"
    ax.set_title(
        f"{prefix} TV {gtype_tag}",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Basis Vector Index", fontsize=14, labelpad=10)
    ax.set_ylabel("Total Variation", fontsize=14, labelpad=10)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", labelsize=10, length=4, width=1.0)

    leg = ax.legend(loc="upper right", fontsize=11, frameon=True)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()

    fname = _filename(title_tag, is_directed, gtype, is_mean)
    if save_fig:
        plt.savefig(fname, dpi=300)
        print(f"Saved combined TV-vs-index plot: {fname}")
        plt.close(fig)
    else:
        plt.show()


def _run_greedy_vs_exact_directed(save_fig=False):
    N = __sN
    G = load_graph_from_file(f"data/graphs/random_digraph_{N}.graphml")
    weights = nx.to_numpy_array(G, dtype=np.float64)

    exact_basis = load_basis_vectors_from_file(
        f"data/basis_vectors/digraph_{N}_basis.npy"
    )
    tv_exact = np.apply_along_axis(total_variation, 0, exact_basis)

    print("Computing greedy bases for directed XRank methods (N=10)…")
    tv_dict: Dict[str, np.ndarray] = {__nexact: tv_exact}
    for method, rank_fn in tqdm(
        __xrank_fn_directed__.items(), desc="Directed Greedy Methods", unit="method"
    ):
        greedy_basis = compute_greedy_basis(N, weights, rank_fn=rank_fn)
        tv_greedy = np.apply_along_axis(total_variation, 0, greedy_basis)
        tv_dict[method] = tv_greedy

    _plot(
        tv_dict=tv_dict,
        is_directed=True,
        title_tag="N=10",
        ntarget=__nexact,
        gtype=None,
        is_mean=False,
        step=1,
        save_fig=save_fig,
    )


def _run_exact_only(save_fig=False):
    N = __sN
    exact_basis = load_basis_vectors_from_file(
        f"data/basis_vectors/digraph_{N}_basis.npy"
    )
    tv_exact = np.apply_along_axis(total_variation, 0, exact_basis)
    _plot(
        tv_dict={__nexact: tv_exact},
        is_directed=True,
        title_tag="N=10",
        ntarget=__nexact,
        gtype=None,
        is_mean=False,
        step=1,
        save_fig=save_fig,
    )
    exact_basis = load_basis_vectors_from_file(
        f"data/basis_vectors/graph_{N}_basis.npy"
    )
    tv_exact = np.apply_along_axis(total_variation, 0, exact_basis)
    _plot(
        tv_dict={__nexact: tv_exact},
        is_directed=False,
        title_tag="N=10",
        ntarget=__nexact,
        gtype=None,
        is_mean=False,
        step=1,
        save_fig=save_fig,
    )


def _run_greedy_comparison_directed(save_fig=False):
    N = __lN
    trials = 10

    print(
        f"Running total variation experiment on random directed graphs (N={N}, trials={trials})..."
    )
    for gtype, create_graph in tqdm(
        __mcreate_random_graphs_directed.items(), desc="Graph Types", unit="graph"
    ):
        methods = list(__xrank_fn_directed__.items())
        tv_sum: Dict[str, np.ndarray] = {
            method: np.zeros(N, dtype=np.float64) for method, _ in methods
        }

        for _ in range(trials):
            weights = create_graph(N)
            for method, rank_fn in methods:
                basis = compute_greedy_basis(N, weights, rank_fn=rank_fn)
                tv_vals = np.apply_along_axis(total_variation, 0, basis)
                tv_sum[method] += tv_vals

        tv_mean: Dict[str, np.ndarray] = {
            method: tv_sum[method] / trials for method, _ in methods
        }

        _plot(
            tv_dict=tv_mean,
            is_directed=True,
            title_tag=gtype,
            gtype=gtype,
            is_mean=True,
            step=5,
            save_fig=save_fig,
        )


def _run_greedy_vs_exact_undirected(save_fig=False):
    N = __sN
    G = load_graph_from_file(f"data/graphs/random_graph_{N}.graphml")
    weights = nx.to_numpy_array(G, dtype=np.float64)

    exact_basis = load_basis_vectors_from_file(
        f"data/basis_vectors/digraph_{N}_basis.npy"
    )
    tv_exact = np.apply_along_axis(total_variation, 0, exact_basis)

    print("Computing greedy bases for undirected XRank methods (N=10)…")
    tv_dict: Dict[str, np.ndarray] = {__nexact: tv_exact}
    for method, rank_fn in tqdm(
        __xrank_fn_undirected__.items(), desc="Undirected Greedy Methods", unit="method"
    ):
        greedy_basis = compute_greedy_basis(N, weights, rank_fn=rank_fn)
        tv_greedy = np.apply_along_axis(total_variation, 0, greedy_basis)
        tv_dict[method] = tv_greedy

    _plot(
        tv_dict=tv_dict,
        is_directed=False,
        title_tag="N=10",
        ntarget=__nexact,
        gtype=None,
        is_mean=False,
        step=1,
        save_fig=save_fig,
    )


def _run_greedy_comparison_undirected(save_fig=False):
    N = __lN
    trials = 10

    print(
        f"Running total variation experiment on random undirected graphs (N={N}, trials={trials})..."
    )
    for gtype, create_graph in tqdm(
        __mcreate_random_graphs.items(), desc="Graph Types", unit="graph"
    ):
        methods = list(__xrank_fn_undirected__.items())
        tv_sum: Dict[str, np.ndarray] = {
            method: np.zeros(N, dtype=np.float64) for method, _ in methods
        }

        for _ in range(trials):
            weights = create_graph(N)
            for method, rank_fn in methods:
                basis = compute_greedy_basis(N, weights, rank_fn=rank_fn)
                tv_vals = np.apply_along_axis(total_variation, 0, basis)
                tv_sum[method] += tv_vals

        tv_mean: Dict[str, np.ndarray] = {
            method: tv_sum[method] / trials for method, _ in methods
        }

        _plot(
            tv_dict=tv_mean,
            is_directed=False,
            title_tag=gtype,
            gtype=gtype,
            is_mean=True,
            step=5,
            save_fig=save_fig,
        )


if __name__ == "__main__":
    # Run all experiments in sequence
    save_fig = True
    _run_exact_only(save_fig=save_fig)
    _run_greedy_vs_exact_directed(save_fig=save_fig)
    _run_greedy_vs_exact_undirected(save_fig=save_fig)
    _run_greedy_comparison_directed(save_fig=save_fig)
    _run_greedy_comparison_undirected(save_fig=save_fig)
