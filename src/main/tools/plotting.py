import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_performance(
    times: Dict[str, List[float]], n: int, file_name: str = None
) -> None:
    names = list(times.keys())
    data = [times[nm] for nm in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        data,
        labels=names,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "firebrick"},
        boxprops={"color": "navy"},
        medianprops={"color": "green"},
    )
    for i, nm in enumerate(names, start=1):
        y = times[nm]
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.6, s=20, color="gray")
        μ, m = np.mean(y), np.median(y)
        ax.text(i + 0.15, μ, f"μ={μ:.1e}", fontsize=9, color="firebrick")
        ax.text(i + 0.15, m, f"m={m:.1e}", fontsize=9, color="green")
    ax.set_title(f"Evaluation Time (N={n})")
    ax.set_xlabel("Function")
    ax.set_ylabel("Time per Partition (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, bbox_inches="tight")
    else:
        plt.show()
        plt.close()


def plot_output_distributions(
    outputs: Dict[str, List[float]], n: int, file_name: str = None
) -> None:
    plt.figure(figsize=(10, 5))
    for name, vals in outputs.items():
        plt.hist(vals, bins=60, density=True, histtype="step", linewidth=2, label=name)
    plt.title(f"Output Distributions (N={n})")
    plt.yscale("log")
    plt.xlabel("Objective Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, bbox_inches="tight")
    else:
        plt.show()
        plt.close()


def plot_error_distributions(
    outputs: Dict[str, List[float]], baseline: str, n: int, file_name: str = None
) -> None:
    base = np.array(outputs[baseline])
    plt.figure(figsize=(12, 4))
    for name, vals in outputs.items():
        if name == baseline:
            continue
        rel_err = 100 * (np.array(vals) - base) / base
        plt.hist(rel_err, bins=50, alpha=0.6, label=f"{name} vs {baseline}")
    plt.axvline(0, color="k", linestyle="--")
    plt.title(f"Relative Error vs {baseline} (N={n})")
    plt.xlabel("Relative Error (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, bbox_inches="tight")
    else:
        plt.show()
        plt.close()


def plot_time_vs_error(
    times: Dict[str, List[float]],
    outputs: Dict[str, List[float]],
    baseline: str,
    n: int,
    file_name: str = None,
) -> None:
    base_out = np.array(outputs[baseline])
    mean_times = {nm: np.mean(t) for nm, t in times.items()}
    mean_errs = {
        nm: np.mean(np.abs(np.array(vals) - base_out) / base_out * 100)
        for nm, vals in outputs.items()
        if nm != baseline
    }
    plt.figure(figsize=(6, 5))
    bt = mean_times[baseline]
    plt.scatter(bt, 0, marker="x", color="red", s=100, label=f"{baseline} (0%)")
    plt.axvline(bt, linestyle="--", color="red")
    for nm, err in mean_errs.items():
        t = mean_times[nm]
        plt.scatter(t, err, s=80, label=nm)
        plt.text(
            t,
            err,
            f"{err:.1f}%",
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
        )
    plt.xlabel("Mean Eval Time per Partition (s)")
    plt.ylabel("Mean Relative Error (%)")
    plt.title(f"Speed–Accuracy Tradeoff (N={n})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, bbox_inches="tight")
    else:
        plt.show()
        plt.close()
