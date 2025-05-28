import numpy as np
import matplotlib.pyplot as plt
import os

from src.root.core import compute_greedy_basis, compute_l1_norm_basis
from src.root.utils import create_random_graph
import src.root.tools.errors as errors
from src.root.api import arg_max_greedy_undirected

_xtimes = {
    3: 0.0001,
    4: 0.0002,
    5: 0.0009,
    6: 0.008,
    7: 0.1,
    8: 1.4,
    9: 21.5,
}  # Estimated times for each number of nodes

"""
This script compares the performance of greedy and exact L1-norm basis 
by analyzing the total l1 norm variation for small graphs.
"""

if not os.path.exists("plots/temp"):
    os.makedirs("plots/temp")


def _format(seconds: float) -> str:
    """
    Convert a duration in seconds to a human-readable string, e.g. "1h 2m 3.20s".
    """
    secs = int(seconds)
    hours, rem = divmod(secs, 3600)
    mins, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if mins:
        parts.append(f"{mins}m")
    # Include seconds with fractional part
    frac = seconds - int(seconds)
    total_secs = secs + frac
    parts.append(f"{total_secs:.2f}s")
    return " ".join(parts)


def main(
    min_nodes: int = 3,
    max_nodes: int = 7,
    num_samples: int = 100,
    gap: int = 1,
    save_path: str = "plots/temp/greedy_vs_exact_mean.png",
):
    """
    Compare greedy vs exact L1-norm basis over multiple samples,
    printing a human-friendly runtime estimate.
    """
    print("Running greedy vs exact L1-norm basis comparison...")
    # Validations
    assert min_nodes > 1, "Minimum nodes >= 2."
    assert max_nodes >= min_nodes, "Max nodes >= min nodes."
    assert gap > 0, "Gap > 0."
    assert num_samples > 0, "Samples > 0."
    assert max_nodes <= 9, "Max nodes <= 9."

    # Estimate total duration
    est_sec = num_samples * sum(
        _xtimes[i] for i in range(min_nodes, max_nodes + 1, gap)
    )
    print(f"Estimated total runtime: {_format(est_sec)}")

    xvals = np.arange(min_nodes, max_nodes + 1, gap)
    greedy_vars = []
    exact_vars = []
    for n in xvals:
        gv, ev = [], []
        for _ in range(num_samples):
            w = create_random_graph(n, is_directed=False)
            gb = compute_greedy_basis(n, w, rank_fn=arg_max_greedy_undirected)
            eb = compute_l1_norm_basis(n, w)
            gv.append(errors.l1_variation(gb))
            ev.append(errors.l1_variation(eb))
        greedy_vars.append(gv)
        exact_vars.append(ev)

    greedy_arr = np.array(greedy_vars)
    exact_arr = np.array(exact_vars)
    mean_g = greedy_arr.mean(axis=1)
    sem_g = greedy_arr.std(axis=1, ddof=1) / np.sqrt(num_samples)
    mean_e = exact_arr.mean(axis=1)
    sem_e = exact_arr.std(axis=1, ddof=1) / np.sqrt(num_samples)

    # Plot styling
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.figsize": (8, 5),
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )
    fig, ax = plt.subplots()
    ax.errorbar(
        xvals, mean_g, yerr=sem_g, marker="o", capsize=4, linestyle="-", label="Greedy"
    )
    ax.errorbar(
        xvals, mean_e, yerr=sem_e, marker="s", capsize=4, linestyle="--", label="Exact"
    )
    ax.set_title("Greedy vs Exact L1-Norm Basis")
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("L1 Variation (mean ± SEM)")
    ax.set_xticks(xvals)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.show()
    print("Greedy vs Exact L1-norm basis comparison completed.")


if __name__ == "__main__":
    stats = main(3, 7)
