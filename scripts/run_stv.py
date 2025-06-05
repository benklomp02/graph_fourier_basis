import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Dict, Optional, List
from tqdm import tqdm

from src.main.tools.errors import total_variation
from src.main.tools.io import load_graph_from_file, load_basis_vectors_from_file
from src.main.utils import (
    create_random_erdos_renyi_graph,
    create_random_geometric_graph,
)
from src.main.core import compute_greedy_basis
from src.main.api import __xrank_fn_directed__, __xrank_fn_undirected__

__nexact = "L1-Norm Basis"
__sxaxis = list(range(3, 11))
__lxaxis = list(range(5, 201, 5))

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
    is_directed: bool,
    gtype: Optional[str],
    is_mean: bool = False,
) -> str:
    folder = _ensure_folder(is_directed)
    if gtype is None:
        direction = "directed" if is_directed else "undirected"
        return folder + f"sumTV_{direction}.png"
    else:
        direction = "directed" if is_directed else "undirected"
        name, p = gtype.split("-")
        name = name.replace(".", "_").replace("-", "_")
        return folder + f"sumTV_{name}_{int(float(p) * 100)}_{direction}.png".lower()


def _plot(
    sumtv_dict: Dict[str, np.ndarray],
    N_list: np.ndarray,
    is_directed: bool,
    ntarget: Optional[str] = None,
    gtype: Optional[str] = None,
    save_fig: bool = False,
):
    if not sumtv_dict:
        return

    if ntarget is not None and ntarget not in sumtv_dict:
        raise ValueError(
            f"Target method '{ntarget}' not found; available keys: {list(sumtv_dict.keys())}"
        )

    plt.style.use("seaborn-v0_8-talk")

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#f7f7f7")
    ax.set_facecolor("#fafafa")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.minorticks_on()

    cmap = plt.get_cmap("tab10")
    non_target = [m for m in sumtv_dict.keys() if m != ntarget]
    color_map: Dict[str, str] = {}
    for idx, m in enumerate(non_target):
        color_map[m] = cmap(idx % 10)
    if ntarget is not None:
        color_map[ntarget] = "black"

    for method, sumtv_array in sumtv_dict.items():
        if method == ntarget:
            ax.plot(
                N_list,
                sumtv_array,
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
                N_list,
                sumtv_array,
                marker="o",
                markersize=6,
                linestyle="-",
                linewidth=1.8,
                color=color_map[method],
                alpha=0.85,
                label=method,
            )

    all_y = np.concatenate([arr for arr in sumtv_dict.values()])
    y_min, y_max = all_y.min(), all_y.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + 0.30 * y_range)

    prefix = "Directed" if is_directed else "Undirected"
    if gtype is None:
        ax.set_title(
            f"{prefix} Sum TV vs Graph Size",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
    else:
        name, p = gtype.split("-")
        nice_name = name.replace("_", "-").title()
        ax.set_title(
            f"{prefix} Sum TV vs Graph Size {nice_name}({p})",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )

    ax.set_xlabel("N", fontsize=14, labelpad=10)
    ax.set_ylabel("Sum of Total Variations", fontsize=14, labelpad=10)

    # Customize ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", labelsize=10, length=4, width=1.0)

    # Legend with a light frame
    leg = ax.legend(loc="upper left", fontsize=11, frameon=True)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()

    fname = _filename(is_directed, gtype, is_mean=False)
    if save_fig:
        plt.savefig(fname, dpi=300)
        print(f"Saved sum‐TV plot: {fname}")
        plt.close(fig)
    else:
        plt.show()


def _run_exact_only(save_fig=False):
    """Run the exact sum TV calculation for small graphs."""
    Ns: List[int] = __sxaxis
    for is_directed in [True, False]:
        exact_sums: List[float] = []
        for N in Ns:
            if is_directed:
                exact_basis = load_basis_vectors_from_file(
                    f"data/basis_vectors/digraph_{N}_basis.npy"
                )
            else:
                exact_basis = load_basis_vectors_from_file(
                    f"data/basis_vectors/graph_{N}_basis.npy"
                )
            tv_vals = np.apply_along_axis(total_variation, 0, exact_basis)
            exact_sums.append(tv_vals.sum())
        N_array = np.array(Ns)
        _plot(
            sumtv_dict={__nexact: np.array(exact_sums)},
            N_list=N_array,
            is_directed=is_directed,
            ntarget=__nexact,
            gtype=None,
            save_fig=save_fig,
        )


def _run_sum_tv_small_directed(save_fig=False):
    Ns: List[int] = __sxaxis
    methods = list(__xrank_fn_directed__.items())

    sumtv: Dict[str, np.ndarray] = {}

    exact_sums: List[float] = []
    for N in Ns:
        G = load_graph_from_file(f"data/graphs/random_digraph_{N}.graphml")
        weights = nx.to_numpy_array(G, dtype=np.float64)

        exact_basis = load_basis_vectors_from_file(
            f"data/basis_vectors/digraph_{N}_basis.npy"
        )
        tv_vals = np.apply_along_axis(total_variation, 0, exact_basis)
        exact_sums.append(tv_vals.sum())

    sumtv[__nexact] = np.array(exact_sums)

    for method_name, rank_fn in tqdm(
        methods, desc="Directed Greedy (small‐N)", unit="method"
    ):
        sums_for_method: List[float] = []
        for N in Ns:
            G = load_graph_from_file(f"data/graphs/random_digraph_{N}.graphml")
            weights = nx.to_numpy_array(G, dtype=np.float64)

            greedy_basis = compute_greedy_basis(N, weights, rank_fn=rank_fn)
            tv_vals = np.apply_along_axis(total_variation, 0, greedy_basis)
            sums_for_method.append(tv_vals.sum())
        sumtv[method_name] = np.array(sums_for_method)

    N_array = np.array(Ns)
    _plot(
        sumtv_dict=sumtv,
        N_list=N_array,
        is_directed=True,
        ntarget=__nexact,
        gtype=None,
        save_fig=save_fig,
    )


def _run_sum_tv_small_undirected(save_fig=False):
    Ns: List[int] = __sxaxis
    methods = list(__xrank_fn_undirected__.items())

    sumtv: Dict[str, np.ndarray] = {}

    exact_sums: List[float] = []
    for N in Ns:
        G = load_graph_from_file(f"data/graphs/random_graph_{N}.graphml")
        weights = nx.to_numpy_array(G, dtype=np.float64)

        exact_basis = load_basis_vectors_from_file(
            f"data/basis_vectors/digraph_{N}_basis.npy"
        )
        tv_vals = np.apply_along_axis(total_variation, 0, exact_basis)
        exact_sums.append(tv_vals.sum())

    sumtv[__nexact] = np.array(exact_sums)

    for method_name, rank_fn in tqdm(
        methods, desc="Undirected Greedy (small‐N)", unit="method"
    ):
        sums_for_method: List[float] = []
        for N in Ns:
            G = load_graph_from_file(f"data/graphs/random_graph_{N}.graphml")
            weights = nx.to_numpy_array(G, dtype=np.float64)

            greedy_basis = compute_greedy_basis(N, weights, rank_fn=rank_fn)
            tv_vals = np.apply_along_axis(total_variation, 0, greedy_basis)
            sums_for_method.append(tv_vals.sum())
        sumtv[method_name] = np.array(sums_for_method)

    N_array = np.array(Ns)
    _plot(
        sumtv_dict=sumtv,
        N_list=N_array,
        is_directed=False,
        ntarget=__nexact,
        gtype=None,
        save_fig=save_fig,
    )


def _run_sum_tv_large_directed(save_fig=False):
    Ns: List[int] = __lxaxis
    methods = list(__xrank_fn_directed__.items())

    for gtype, create_graph in __mcreate_random_graphs_directed.items():
        sumtv: Dict[str, List[float]] = {method_name: [] for method_name, _ in methods}

        for N in tqdm(Ns, desc=f"Directed large‐N ({gtype})", unit="N"):
            weights = create_graph(N)
            for method_name, rank_fn in methods:
                basis = compute_greedy_basis(N, weights, rank_fn=rank_fn)
                tv_vals = np.apply_along_axis(total_variation, 0, basis)
                sumtv[method_name].append(tv_vals.sum())

        sumtv_arr: Dict[str, np.ndarray] = {
            m: np.array(vals) for m, vals in sumtv.items()
        }

        N_array = np.array(Ns)
        _plot(
            sumtv_dict=sumtv_arr,
            N_list=N_array,
            is_directed=True,
            ntarget=None,
            gtype=gtype,
            save_fig=save_fig,
        )


def _run_sum_tv_large_undirected(save_fig: False):
    Ns: List[int] = __lxaxis
    methods = list(__xrank_fn_undirected__.items())

    for gtype, create_graph in __mcreate_random_graphs.items():
        sumtv: Dict[str, List[float]] = {method_name: [] for method_name, _ in methods}

        for N in tqdm(Ns, desc=f"Undirected large‐N ({gtype})", unit="N"):
            weights = create_graph(N)
            for method_name, rank_fn in methods:
                basis = compute_greedy_basis(N, weights, rank_fn=rank_fn)
                tv_vals = np.apply_along_axis(total_variation, 0, basis)
                sumtv[method_name].append(tv_vals.sum())

        sumtv_arr: Dict[str, np.ndarray] = {
            m: np.array(vals) for m, vals in sumtv.items()
        }

        N_array = np.array(Ns)
        _plot(
            sumtv_dict=sumtv_arr,
            N_list=N_array,
            is_directed=False,
            ntarget=None,
            gtype=gtype,
            save_fig=save_fig,
        )


if __name__ == "__main__":
    # Run all four experiments in sequence:
    save_fig = True
    _run_exact_only(save_fig=save_fig)
    _run_sum_tv_small_directed(save_fig=save_fig)
    _run_sum_tv_small_undirected(save_fig=save_fig)
    _run_sum_tv_large_directed(save_fig=save_fig)
    _run_sum_tv_large_undirected(save_fig=save_fig)
