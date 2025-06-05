import os
import logging
import numpy as np
import time
from tqdm import tqdm

from src.main.tools.surrogates import (
    laplacian_cost,
    laplacian_cost_approx_by_triangle_inequality,
    laplacian_cost_approx_by_median_split,
    laplacian_cost_approx_by_majority,
    laplacian_cost_approx_by_mean_split,
)
from src.main.tools.surrogates import objective_t
from src.main.tools.plotting import (
    plot_performance,
    plot_output_distributions,
    plot_error_distributions,
    plot_time_vs_error,
)
from typing import Dict, List, Optional, Tuple, Dict
from src.main.utils import (
    create_random_partition_matrix,
    create_random_geometric_graph,
)

logger = logging.getLogger(__name__)

if not os.path.exists("plots/temp"):
    os.makedirs("plots/temp")


def _eval(
    funcs: Dict[str, objective_t],
    num_partitions: int,
    weights: np.ndarray,
    show_progress: bool,
    scaling: Optional[Dict[str, float]] = None,
) -> Dict[str, List[float]]:
    outputs = {name: [] for name in funcs}
    iterator = tqdm(range(num_partitions), desc="Partitions", disable=not show_progress)
    n = weights.shape[0]
    for _ in iterator:
        m = np.random.randint(2, n + 1)
        M = create_random_partition_matrix(n, m)
        a = np.sort(np.random.rand(m))
        for name, func in funcs.items():
            if scaling and name in scaling:
                outputs[name].append(
                    func(weights=weights, M=M, a=a, scaling=scaling[name])
                )
            else:
                outputs[name].append(func(weights=weights, M=M, a=a))
    return outputs


def _run(
    funcs: Dict[str, objective_t],
    n: int,
    num_trials: int,
    show_progress: bool,
    seed: Optional[int],
    show_graph: bool,
    is_weighted: bool,
    num_partitions: int,
    scaling: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    logger.info(
        "Starting benchmark: N=%d, trials=%d, partitions=%d",
        n,
        num_trials,
        num_partitions,
    )
    rng = np.random.default_rng(seed)
    times_dict = {name: [] for name in funcs}
    outputs_dict = {name: [] for name in funcs}

    for trial in range(1, num_trials + 1):
        logger.info("Trial %d/%d", trial, num_trials)
        W = create_random_geometric_graph(
            n, seed=rng, visualize=show_graph, is_weighted=is_weighted
        )
        trial_out = _eval(funcs, num_partitions, W, show_progress, scaling=scaling)
        for name, vals in trial_out.items():
            outputs_dict[name].extend(vals)
        for name, func in funcs.items():
            start = time.perf_counter()
            for _ in range(num_partitions):
                m = rng.integers(2, n + 1)
                M = create_random_partition_matrix(n, m)
                a = np.sort(rng.random(m))
                if scaling and name in scaling:
                    func(weights=W, M=M, a=a, scaling=scaling[name])
                else:
                    func(weights=W, M=M, a=a)
            times_dict[name].append(time.perf_counter() - start)
    means = {name: np.mean(ts) for name, ts in times_dict.items()}
    logger.info("Avg eval times: %s", means)
    return times_dict, outputs_dict


def main(save_fig=False):
    print("Running surrogates performance analysis...")
    # Define objective functions
    funcs = {
        "original": laplacian_cost,
        "triangle": laplacian_cost_approx_by_triangle_inequality,
        "median-split": laplacian_cost_approx_by_median_split,
        "mean-split": laplacian_cost_approx_by_mean_split,
        "majority": laplacian_cost_approx_by_majority,
    }
    # Run
    num_vertices = 200
    times_dict, outputs_dict = _run(
        funcs,
        n=num_vertices,
        num_trials=10,
        show_progress=True,
        seed=42,
        show_graph=False,
        is_weighted=True,
        num_partitions=100,
    )
    # Summarize results
    orig = times_dict.get("original", [])
    if orig:
        print(
            f"Original timing: μ={np.mean(orig):.2e}s, min={np.min(orig):.2e}s, max={np.max(orig):.2e}s"
        )
    # Create the save_path if it doesn't exist
    save_path_prefix = "plots/temp/"
    # Create the save path if it doesn't exist
    if save_path_prefix and not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    file_name = None
    if save_fig:
        file_name = os.path.join(save_path_prefix, f"surrogates_performance.pdf")
    plot_performance(times_dict, num_vertices, file_name=file_name)
    if save_fig:
        file_name = os.path.join(
            save_path_prefix, f"surrogates_output_distributions.png"
        )
    plot_output_distributions(outputs_dict, num_vertices, file_name=file_name)
    if save_fig:
        file_name = os.path.join(
            save_path_prefix, f"surrogates_error_distributions.png"
        )
    plot_error_distributions(
        outputs_dict, "original", num_vertices, file_name=file_name
    )
    if file_name:
        file_name = os.path.join(save_path_prefix, f"surrogates_time_vs_error.png")
    plot_time_vs_error(
        times_dict,
        outputs_dict,
        "original",
        num_vertices,
        file_name=file_name,
    )
    print("Surrogates performance analysis completed.")


if __name__ == "__main__":
    main(save_fig=True)
