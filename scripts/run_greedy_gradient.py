import time
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List

from src.graph_fourier_transform.subgradient import (
    compute_greedy_subgradient_basis,
    compute_exact_null_space,
)
from src.graph_fourier_transform.utils import create_random_graph
from src.graph_fourier_transform.tools.errors import laplacian_l1_cost_mean
from src.graph_fourier_transform.subgradient import sequences


import matplotlib.pyplot as plt


import itertools
import matplotlib.pyplot as plt


def plot_error_vs_time(
    xcosts: Dict[str, float], xtimes: Dict[str, float], N: int, graph_type: str
) -> None:
    """
    Scatter error vs. runtime for each method, at fixed graph size N.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = [k for k in xcosts if k != "exact"]

    # cycle through markers and linestyles
    markers = itertools.cycle(["o", "s", "D", "^", "v"])
    linestyles = itertools.cycle(["-", "--", "-.", ":"])
    for m in methods:
        ax.scatter(
            xtimes[m],
            xcosts[m],
            label=m.replace("greedy-", "").replace("-", " ").title(),
            marker=next(markers),
            linestyle=next(linestyles),
            s=80,
            alpha=0.8,
        )

    # exact point (zero error)
    ax.scatter(
        xtimes["exact"],
        xcosts["exact"],
        label="Exact",
        marker="X",
        linestyle="--",
        s=100,
        color="black",
        zorder=5,
    )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Laplacian l1 Cost", fontsize=11)
    ax.set_title(f"Laplacian l1 Cost vs Time (N={N})", fontsize=12)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    plt.savefig(f"plots/temp/greedy_descent_error_vs_time_{graph_type}.png", dpi=150)
    plt.close()


def plot_error_vs_num_vertices(
    xcosts: Dict[str, List[float]], xaxis: List[int], graph_type: str
) -> None:
    """
    Line plot of mean cost vs. number of vertices for each method.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))

    methods = list(xcosts.keys())
    markers = itertools.cycle(["o", "s", "D", "^", "v"])
    linestyles = itertools.cycle(["-", "--", "-.", ":"])

    for m in methods:
        ax.plot(
            xaxis,
            xcosts[m],
            label=m.replace("greedy-", "").replace("-", " ").title(),
            marker=next(markers),
            linestyle=next(linestyles),
            linewidth=1.8,
            markersize=6,
        )

    ax.set_xlabel("N", fontsize=11)
    ax.set_ylabel("Laplacian l1 Cost", fontsize=11)
    ax.set_title("Laplacian l1 vs. Graph Size", fontsize=12)
    ax.legend(frameon=True, fontsize=9, ncol=2)
    fig.tight_layout()
    plt.savefig(
        f"plots/temp/greedy_descent_error_vs_num_vert_{graph_type}.png", dpi=150
    )
    plt.close()


def get_distance_threshold(graph_type: str, num_nodes: int) -> float:
    if graph_type == "sparse":
        return 1 / num_nodes
    elif graph_type == "dense":
        return 0.15
    elif graph_type == "complete":
        return 1.0
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def run_experiment(num_nodes: int, num_trials: int, graph_type: str = "dense"):
    weights = create_random_graph(
        num_nodes,
        is_weighted=True,
        is_directed=False,
        is_connected=False,
        distance_threshold=get_distance_threshold(graph_type, num_nodes),
    )
    xcosts = {
        f"greedy-{s.value}": []
        for s in (sequences.HARMONIC, sequences.LOG_HARMONIC, sequences.POWER)
    }
    xcosts["exact"] = []
    xtimes = {k: [] for k in xcosts}

    for _ in tqdm.tqdm(range(num_trials), desc=f"Trials (n={num_nodes})"):
        for seq in (sequences.HARMONIC, sequences.LOG_HARMONIC, sequences.POWER):
            t0 = time.time()
            x0 = np.random.rand(num_nodes)
            basis = compute_greedy_subgradient_basis(
                weights, x0, seq=seq, max_iter=20, c_impl=True
            )
            xtimes[f"greedy-{seq.value}"].append(time.time() - t0)
            cost = laplacian_l1_cost_mean(weights, basis)
            if np.isnan(cost):
                raise ValueError(
                    f"NaN cost encountered for sequence {seq.value} with {num_nodes} nodes."
                )
            xcosts[f"greedy-{seq.value}"].append(cost)

        # exact
        t0 = time.time()
        basis_e = compute_exact_null_space(weights)
        xtimes["exact"].append(time.time() - t0)
        cost = laplacian_l1_cost_mean(weights, basis_e)
        if np.isnan(cost):
            raise ValueError(
                f"NaN cost encountered for exact basis with {num_nodes} nodes."
            )
        xcosts["exact"].append(cost)

    # average out
    for k in xcosts:
        xcosts[k] = np.mean(xcosts[k])
        xtimes[k] = np.mean(xtimes[k])
    return xcosts, xtimes


def main(min_nodes=50, max_nodes=1000, step=50, trials=3):
    node_counts = list(range(min_nodes, max_nodes + 1, step))

    # We’ll keep track of the “final” run at n = max_nodes
    for graph_type in ["sparse", "dense", "complete"]:
        print(f"Running experiments for graph type: {graph_type}")
        cost_curves = {
            k: []
            for k in ["greedy-harmonic", "greedy-log-harmonic", "greedy-power", "exact"]
        }
        time_curves = {k: [] for k in cost_curves}
        final_costs = final_times = None
        for n in node_counts:
            xcosts, xtimes = run_experiment(n, trials, graph_type=graph_type)
            for k in cost_curves:
                cost_curves[k].append(xcosts[k])
                time_curves[k].append(xtimes[k])

            if n == max_nodes:
                final_costs = xcosts
                final_times = xtimes
        # now plot
        plot_error_vs_num_vertices(cost_curves, node_counts, graph_type=graph_type)
        plot_error_vs_time(final_costs, final_times, max_nodes, graph_type=graph_type)


if __name__ == "__main__":
    main()
