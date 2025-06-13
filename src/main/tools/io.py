import numpy as np
import networkx as nx
import time
from typing import Union, Tuple, Optional, Dict
import json
import os

from src.main.utils import create_random_geometric_graph, convert_to_directed
from src.main.core import compute_l1_norm_basis


def save_graph_to_file(G: Union[nx.Graph, nx.DiGraph]) -> None:
    gtype = "digraph" if nx.is_directed(G) else "graph"
    prefix = "data/graphs/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # Save the adjacency matrix
    N = G.number_of_nodes()
    weights = nx.to_numpy_array(G, weight="weight")
    np.save(f"{prefix}random_{gtype}_{N}.npy", weights)
    # Save the positions of the nodes
    with open(f"{prefix}random_pos_{N}.json", "w") as f:
        json.dump(nx.get_node_attributes(G, "pos"), f)


def load_graph_from_file(N: int, is_directed: bool) -> Union[nx.Graph, nx.DiGraph]:
    """
    Load a graph from a file and return the graph object and its weights.

    Args:
        filename: Path to the file containing the graph data.

    Returns:
        A tuple containing the graph object and the weights as a numpy array.
    """
    filename = f"data/graphs/random_{'digraph' if is_directed else 'graph'}_{N}.npy"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Graph file {filename} not found.")
    weights = np.load(filename)
    G = nx.from_numpy_array(
        weights, create_using=nx.DiGraph if "digraph" in filename else nx.Graph
    )
    filename_pos = f"data/graphs/random_pos_{N}.json"
    if not os.path.exists(filename_pos):
        raise FileNotFoundError(f"Position file {filename_pos} not found.")
    pos = json.load(open(filename_pos, "r"))
    npos = {int(k): v for k, v in pos.items()}
    nx.set_node_attributes(G, npos, "pos")
    return G


def save_basis_vectors_to_file(basis: np.ndarray, is_directed: bool) -> None:
    N = basis.shape[0]
    prefix = "data/basis_vectors/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    filename = f"{prefix}{'digraph' if is_directed else 'graph'}_{N}_basis.npy"
    np.save(filename, basis)


def load_basis_vectors_from_file(N: int, is_directed: bool) -> np.ndarray:
    filename = (
        f"data/basis_vectors/{'digraph' if is_directed else 'graph'}_{N}_basis.npy"
    )
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Basis vectors file {filename} not found.")
    return np.load(filename)


def save_random_graph_l1_basis(N: int, visualize: Optional[bool] = False) -> None:
    """
    Create a random graph and save its L1 norm basis to a file. Afterwards,
    it also creates a directed version of the graph and saves its L1 norm basis.

    Args:
        N: Number of nodes in the graph.
        visualize: Whether to visualize the graph.
    """
    print(f"Creating random graph with {N} nodes...")
    thres = 0.4
    seed = None
    tstart = time.time()
    # Undirected graph
    G = create_random_geometric_graph(
        num_nodes=N,
        distance_threshold=thres,
        seed=seed,
        is_weighted=True,
        is_directed=False,
        is_connected=True,
        visualize=visualize,
    )
    weights = nx.to_numpy_array(G, weight="weight")
    basis = compute_l1_norm_basis(N, weights)
    save_graph_to_file(G)
    save_basis_vectors_to_file(basis, is_directed=False)

    print("Undirected graph L1 norm basis saved.")
    # Directed graph
    G_dir = convert_to_directed(G)
    weights = nx.to_numpy_array(G_dir, weight="weight")
    basis = compute_l1_norm_basis(N, weights)

    save_basis_vectors_to_file(basis, is_directed=True)
    save_graph_to_file(G_dir)
    print("Directed graph L1 norm basis saved.")

    tend = time.time()
    print(f"Time taken: {tend - tstart:.2f} seconds")


def save_graph_l1_basis(G: Union[nx.Graph, nx.DiGraph]) -> None:
    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    N = G.number_of_nodes()
    basis = compute_l1_norm_basis(N, weights)
    if isinstance(G, nx.DiGraph):
        save_basis_vectors_to_file(basis, f"data/basis_vectors/digraph_{N}_basis.npy")
    else:
        save_basis_vectors_to_file(basis, f"data/basis_vectors/graph_{N}_basis.npy")


if __name__ == "__main__":
    from src.main.utils import visualize_graph

    for N in range(3, 11):
        G = load_graph_from_file(N, is_directed=False)
        visualize_graph(G, f"data/graphs/random_graph_{N}.png")
        G_dir = load_graph_from_file(N, is_directed=True)
        visualize_graph(G_dir, f"data/graphs/random_digraph_{N}.png")
