import numpy as np
import os
import shutil
import networkx as nx
import numpy as np
from itertools import combinations
from typing import Optional, Union
import matplotlib.pyplot as plt


def create_fresh_directory(dir_path: str):
    """
    Create a fresh directory for saving figures.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def create_random_basis_vector(
    basis: np.ndarray, zero_prob: float = 0.5, weight_dist: str = "uniform"
) -> np.ndarray:
    """
    Generate a random unit‐norm vector in the span of the given orthonormal basis.
    """
    _, d = basis.shape
    mask = np.random.rand(d) > zero_prob
    if not mask.any():
        mask[np.random.randint(d)] = True
    if weight_dist == "uniform":
        weights = np.random.uniform(-1, 1, size=d)
    elif weight_dist == "normal":
        weights = np.random.randn(d)
    else:
        raise ValueError("weight_dist must be 'uniform' or 'normal'")
    coefs = weights * mask
    norm = np.linalg.norm(coefs)
    if norm == 0:
        idx = np.random.randint(d)
        coefs = np.zeros(d)
        coefs[idx] = 1.0
    x = basis @ coefs
    x /= np.linalg.norm(x)
    return x


def create_random_partition_matrix(n: int, m: int) -> np.ndarray:
    """Generate an `n × m` one-hot partition assignment matrix."""
    assert n >= m >= 2
    base = np.arange(m)
    extras = np.random.choice(base, n - m)
    labels = np.concatenate([base, extras])
    np.random.shuffle(labels)
    M = np.zeros((n, m), dtype=int)
    for i, lbl in enumerate(labels):
        M[i, lbl] = 1
    return M


def _validate_inputs(num_nodes: int) -> None:
    """Validate the inputs for the graph generation function."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer.")


def _add_edges_to_graph(
    graph: nx.DiGraph,
    coords: np.ndarray,
    distance_threshold: float,
    rng: np.random.Generator,
    is_directed: bool,
) -> None:
    """
    Add directed edges based on spatial distance and random orientation.
    """
    n = coords.shape[0]
    for i, j in combinations(range(n), 2):
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < distance_threshold:
            # random orientation
            if is_directed:
                if rng.random() < 0.5:
                    graph.add_edge(i, j)
                else:
                    graph.add_edge(j, i)
            else:
                # undirected graph
                graph.add_edge(i, j)
                graph.add_edge(j, i)


def _visualize_graph(
    graph: nx.DiGraph,
    coords: np.ndarray,
) -> None:
    """Visualize the directed graph using matplotlib."""
    pos = {i: tuple(coords[i]) for i in range(coords.shape[0])}
    nx.draw(
        graph,
        pos=pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        font_size=10,
        arrowsize=15,
    )
    if nx.is_weighted(graph):
        edge_labels = nx.get_edge_attributes(graph, "weight")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.show()


def _add_random_weights_to_graph(graph: nx.DiGraph, is_directed: bool) -> None:
    """
    Add weights to the edges of the graph.
    """
    if is_directed:
        for u, v in graph.edges():
            w_uv = np.random.rand()
            graph[u][v]["weight"] = w_uv
    else:
        for u in graph.nodes():
            for v in graph.neighbors(u):
                if u < v:
                    w_uv = np.random.rand()
                    w_uv = np.round(w_uv, 2)
                    graph[u][v]["weight"] = w_uv
                    graph[v][u]["weight"] = w_uv


def create_random_graph(
    num_nodes: int,
    distance_threshold: Optional[float] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
    visualize: bool = False,
    is_weighted: bool = True,
    is_directed: bool = True,
    is_connected: bool = True,
) -> np.ndarray:
    """
    Generate a weakly connected random directed graph.

    Args:
        num_nodes: number of nodes in the graph.
        distance_threshold: initial distance cutoff (default: 3/num_nodes).
        seed: random seed (int) or np.random.Generator for reproducibility.
        visualize: if True, display a plot of the generated graph.
        is_weighted: if True, add random weights to the edges.
        is_directed: if True, create a directed graph.

    Returns:
        Adjacency matrix (np.ndarray) of shape (num_nodes, num_nodes).
    """
    _validate_inputs(num_nodes)

    # Initialize RNG
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # Generate random 2D coordinates
    coords = rng.random((num_nodes, 2))

    # Initialize directed graph
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))

    # Set default threshold
    if distance_threshold is None:
        distance_threshold = 3 / num_nodes

    # Try until weakly connected or timeout
    max_iter = 100
    thr = distance_threshold
    for _ in range(max_iter):
        graph.clear_edges()
        _add_edges_to_graph(graph, coords, thr, rng, is_directed=is_directed)
        if nx.is_weakly_connected(graph) or not is_connected:
            break
        thr *= 1.1
    else:
        raise RuntimeError(
            f"Failed to generate weakly connected graph after {max_iter} iterations"
        )
    if is_weighted:
        _add_random_weights_to_graph(graph, is_directed=is_directed)
    # Optional visualization
    if visualize:
        _visualize_graph(graph, coords)

    # Return adjacency matrix
    return nx.to_numpy_array(graph)
