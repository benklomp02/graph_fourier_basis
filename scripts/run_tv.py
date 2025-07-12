import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Dict, Optional
from tqdm import tqdm
from scipy.interpolate import make_interp_spline

from src.main.tools.errors import total_variation
from src.main.tools.io import load_graph_from_file, load_basis_vectors_from_file
from src.main.utils import (
    create_random_geometric_graph,
    create_random_erdos_renyi_graph,
)
from src.main.core import (
    compute_greedy_basis,
    compute_l1_norm_basis,
    compute_greedy_basis_py,
)
from src.main.api import __xrank_fn_directed__, __xrank_fn_undirected__

__sN = 10
__lN = 200
__NAME_EXACT = "L1-Norm Basis"
__SORT_BY_TV = False
__IS_STORED = True
__STEP_SM = 1
__STEP_LG = 1
__SMOOTH = False

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


def _recover_l1_norm_basis(
    N: int, weights: np.ndarray, is_directed: bool
) -> np.ndarray:
    if __IS_STORED:
        basis = load_basis_vectors_from_file(N, is_directed)
        if basis.shape[0] != N:
            raise ValueError(
                f"Loaded basis has shape {basis.shape}, expected ({N}, ...)"
            )
        return basis
    else:
        basis = compute_l1_norm_basis(N, weights)
        return basis


def _ensure_folder(is_directed: bool) -> str:
    """Ensure that `plots/graph/` or `plots/digraph/` exists, then return its path."""
    folder = f"plots/{'digraph' if is_directed else 'graph'}/"
    os.makedirs(folder, exist_ok=True)
    return folder


def _filename(
    is_directed: bool,
    gtype: Optional[str],
) -> str:
    folder = _ensure_folder(is_directed)
    if gtype is None:
        return folder + f"tv_small{"_sorted" if __SORT_BY_TV else ""}.png"
    else:
        name, p = gtype.split("-")
        name = name.replace("_", "-").title()
        return (
            folder
            + f"tv_{name}_{int(float(p) * 100)}_N{__lN}{"_sorted" if __SORT_BY_TV else ""}.png".lower()
        )


def _plot(
    tv_dict: Dict[str, np.ndarray],
    is_directed: bool,
    title_tag: str,
    ntarget: Optional[str] = None,
    gtype: Optional[str] = None,
    is_mean: bool = False,
    step: int = 1,
    save_fig: bool = False,
    fname: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    if not tv_dict:
        return

    if ntarget is not None and ntarget not in tv_dict:
        raise ValueError(
            f"Target method '{ntarget}' not found; available keys: {list(tv_dict.keys())}"
        )

    if __SORT_BY_TV:
        # Sort the TV values by their total variation
        tv_dict = {method: np.sort(tv_dict[method]) for method in tv_dict.keys()}

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
        if __SMOOTH:
            x = subsampled_indices
            y = tv_sub
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=3)  # cubic spline
            y_smooth = spline(x_smooth)

            ax.plot(
                x_smooth,
                y_smooth,
                linestyle="-",
                linewidth=2.0 if method == ntarget else 1.5,
                color=color_map[method],
                alpha=0.95 if method == ntarget else 0.85,
                label=method,
            )
        else:
            if method == ntarget:
                ax.plot(
                    subsampled_indices,
                    tv_sub,
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
    ax.set_ylim(y_min, y_max + 0.40 * y_range)

    prefix = "Directed" if is_directed else "Undirected"
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
    xlabel = (
        "Basis Vector Index k" if not __SORT_BY_TV else "Sorted Basis Vector Index k"
    )
    ax.set_xlabel(xlabel, fontsize=14, labelpad=10)
    ax.set_ylabel(
        "Total Variation" if ylabel is None else ylabel, fontsize=14, labelpad=10
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", labelsize=10, length=4, width=1.0)

    leg = ax.legend(loc="upper right", fontsize=11, frameon=True)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()

    if fname is None:
        fname = _filename(is_directed, gtype)
    if save_fig:
        plt.savefig(fname, dpi=300)
        print(f"Saved combined TV-vs-index plot: {fname}")
        plt.close(fig)
    else:
        plt.show()


def _run_greedy_vs_exact_directed(save_fig=False):
    N = __sN
    G = load_graph_from_file(N, is_directed=True)
    weights = nx.to_numpy_array(G, dtype=np.float64)

    exact_basis = _recover_l1_norm_basis(N, weights, is_directed=True)
    f = lambda x: total_variation(weights.copy(), x)
    tv_exact = np.apply_along_axis(f, 0, exact_basis)

    print("Computing greedy bases for directed XRank methods (N=10)…")
    tv_dict: Dict[str, np.ndarray] = {__NAME_EXACT: tv_exact}
    for method, rank_fn in tqdm(
        __xrank_fn_directed__.items(), desc="Directed Greedy Methods", unit="method"
    ):
        greedy_basis = compute_greedy_basis(N, weights.copy(), rank_fn=rank_fn)
        tv_greedy = np.apply_along_axis(f, 0, greedy_basis)
        tv_dict[method] = tv_greedy

    _plot(
        tv_dict=tv_dict,
        is_directed=True,
        title_tag="N=10",
        ntarget=__NAME_EXACT,
        gtype=None,
        is_mean=False,
        step=__STEP_SM,
        save_fig=save_fig,
    )


def _run_exact_only(save_fig=False):
    N = __sN
    G = load_graph_from_file(N, is_directed=True)
    weights_dir = nx.to_numpy_array(G, dtype=np.float64)
    exact_basis = _recover_l1_norm_basis(N, weights_dir, is_directed=True)
    f_dir = lambda x: total_variation(weights_dir, x)
    tv_exact = np.apply_along_axis(f_dir, 0, exact_basis)
    _plot(
        tv_dict={__NAME_EXACT: tv_exact},
        is_directed=True,
        title_tag="N=10",
        ntarget=__NAME_EXACT,
        gtype=None,
        is_mean=False,
        step=__STEP_SM,
        save_fig=save_fig,
        fname=f"plots/digraph/tv_exact{"_sorted" if __SORT_BY_TV else ""}.png",
    )
    G = load_graph_from_file(N, is_directed=False)
    weights = nx.to_numpy_array(G, dtype=np.float64)
    f = lambda x: total_variation(weights, x)
    exact_basis = _recover_l1_norm_basis(N, weights, is_directed=False)
    tv_exact = np.apply_along_axis(f, 0, exact_basis)
    _plot(
        tv_dict={__NAME_EXACT: tv_exact},
        is_directed=False,
        title_tag="N=10",
        ntarget=__NAME_EXACT,
        gtype=None,
        is_mean=False,
        step=__STEP_SM,
        save_fig=save_fig,
        fname=f"plots/graph/tv_exact{"_sorted" if __SORT_BY_TV else ""}.png",
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
            G = create_graph(N)
            weights = nx.to_numpy_array(G, dtype=np.float64)
            f = lambda x: total_variation(weights, x)
            for method, rank_fn in methods:
                basis = compute_greedy_basis(N, weights, rank_fn=rank_fn)
                tv_vals = np.apply_along_axis(f, 0, basis)
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
            step=__STEP_LG,
            save_fig=save_fig,
        )


def _run_greedy_vs_exact_undirected(save_fig=False):
    N = __sN
    G = load_graph_from_file(N, is_directed=False)
    weights = nx.to_numpy_array(G, dtype=np.float64)
    f = lambda x: total_variation(weights, x)
    exact_basis = _recover_l1_norm_basis(N, weights, is_directed=False)
    tv_exact = np.apply_along_axis(f, 0, exact_basis)

    print("Computing greedy bases for undirected XRank methods (N=10)…")
    tv_dict: Dict[str, np.ndarray] = {__NAME_EXACT: tv_exact}
    for method, rank_fn in tqdm(
        __xrank_fn_undirected__.items(), desc="Undirected Greedy Methods", unit="method"
    ):
        greedy_basis = compute_greedy_basis(N, weights, rank_fn=rank_fn)
        tv_greedy = np.apply_along_axis(f, 0, greedy_basis)
        tv_dict[method] = tv_greedy

    _plot(
        tv_dict=tv_dict,
        is_directed=False,
        title_tag="N=10",
        ntarget=__NAME_EXACT,
        gtype=None,
        is_mean=False,
        step=__STEP_SM,
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
            G = create_graph(N)
            weights = nx.to_numpy_array(G, dtype=np.float64)
            f = lambda x: total_variation(weights, x)
            for method, rank_fn in methods:
                basis = compute_greedy_basis(N, weights, rank_fn=rank_fn)
                tv_vals = np.apply_along_axis(f, 0, basis)
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
            step=__STEP_LG,
            save_fig=save_fig,
        )


# ------ Main functions to run the experiments ------


def _experiment_exact(save_fig):
    global __SORT_BY_TV
    for sorted in [False, True]:
        __SORT_BY_TV = sorted
        _run_exact_only(save_fig=save_fig)
        _run_greedy_vs_exact_directed(save_fig=save_fig)
        _run_greedy_vs_exact_undirected(save_fig=save_fig)


def _experiment_greedy(num_vert, save_fig):
    global __SORT_BY_TV
    for sorted in [False, True]:
        __SORT_BY_TV = sorted
        global __lN
        __lN = num_vert
        _run_greedy_comparison_directed(save_fig=save_fig)
        _run_greedy_comparison_undirected(save_fig=save_fig)


def _run_experiment_laplacian_cost(N, save_fig):
    """Run the Laplacian cost experiment for a specific number of vertices."""
    from src.main.tools.surrogates import __xrank_fn_sur__ as xrank_fn_sur
    from src.main.tools.surrogates import rank_fn_lap
    from src.main.core import compute_exact_basis_py

    global __SORT_BY_TV
    for is_directed in [True, False]:
        print(
            f"Running Laplacian cost experiment for {"directed" if is_directed else "undirected"} graphs (N={N})..."
        )
        G = load_graph_from_file(N, is_directed=is_directed)
        weights = nx.to_numpy_array(G, dtype=np.float64)
        f = lambda x: rank_fn_lap(weights, x)
        tv_dict = {}
        for method, rank_fn in xrank_fn_sur.items():
            basis = compute_exact_basis_py(N, weights, obj=rank_fn)
            tv_vals = np.apply_along_axis(f, 0, basis)
            tv_dict[method] = tv_vals
        for sorted in [False]:
            __SORT_BY_TV = sorted
            fname = f"plots/surrogates/lapcost_{"dir" if is_directed else "undir"}{"_sorted" if __SORT_BY_TV else ""}_N{N}.png"
            _plot(
                tv_dict=tv_dict,
                is_directed=is_directed,
                title_tag=f"N={N}",
                gtype=None,
                is_mean=False,
                step=__STEP_SM,
                save_fig=save_fig,
                fname=fname,
                ylabel="Laplacian Cost",
            )


def main():
    """Main function to run the experiments."""
    save_fig = True
    _experiment_exact(save_fig=save_fig)
    for num_vert in [20, 50, 80, 150, 200]:
        _experiment_greedy(num_vert, save_fig=save_fig)
    print("All experiments completed successfully.")


if __name__ == "__main__":
    _experiment_exact(True)
