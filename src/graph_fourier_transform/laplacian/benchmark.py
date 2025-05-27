import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Dict
from tqdm.auto import tqdm

from src.graph_fourier_transform.laplacian.objectives import objective_t
from src.graph_fourier_transform.utils import (
    create_random_partition_matrix,
    create_random_graph,
)

logger = logging.getLogger(__name__)


def _evaluate_objectives(
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


def benchmark_and_evaluate(
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
        W = create_random_graph(
            n, seed=rng, visualize=show_graph, is_weighted=is_weighted
        )
        # collect outputs
        trial_out = _evaluate_objectives(
            funcs, num_partitions, W, show_progress, scaling=scaling
        )
        for name, vals in trial_out.items():
            outputs_dict[name].extend(vals)

        # timing
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
