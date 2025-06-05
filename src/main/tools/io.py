import numpy as np
import networkx as nx
import time
from typing import Union

from src.main.utils import create_random_geometric_graph
from src.main.core import compute_l1_norm_basis


def save_graph_to_file(G: Union[nx.Graph, nx.DiGraph], file_path: str) -> None:
    """
    Save the graph to a file in GraphML format.

    Args:
        G: The graph to save.
        file_path: Path to the output file.
    """
    nx.write_graphml(G, file_path)


def load_graph_from_file(
    file_path: str, suppress: bool = True
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Load a graph from a file in GraphML format.

    Args:
        file_path: Path to the input file.

    Returns:
        The loaded graph.
    """
    G = nx.read_graphml(file_path)
    return G


def save_basis_vectors_to_file(basis_vectors: np.ndarray, file_path: str) -> None:
    """
    Save the basis vectors to a file in NumPy format.

    Args:
        basis_vectors: The basis vectors to save.
        file_path: Path to the output file.
    """
    np.save(file_path, basis_vectors)


def load_basis_vectors_from_file(file_path: str) -> np.ndarray:
    """
    Load the basis vectors from a file in NumPy format.

    Args:
        file_path: Path to the input file.

    Returns:
        The loaded basis vectors.
    """
    return np.load(file_path)


def save_random_graph_l1_basis(
    N: int, thres: float, seed: int, visualize: bool = False
) -> None:
    """
    Create a random graph and save its L1 norm basis to a file. Afterwards,
    it also creates a directed version of the graph and saves its L1 norm basis.

    Args:
        N: Number of nodes in the graph.
        thres: Distance threshold for edge creation.
        seed: Random seed for reproducibility.
        file_path: Path to save the graph and basis vectors.
    """
    tstart = time.time()
    weights = create_random_geometric_graph(
        N,
        thres,
        seed,
        is_weighted=True,
        is_directed=False,
        is_connected=True,
        visualize=visualize,
    )
    G = nx.from_numpy_array(weights)
    save_graph_to_file(G, f"data/graphs/random_graph_{N}.graphml")

    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    basis = compute_l1_norm_basis(N, weights)
    save_basis_vectors_to_file(basis, f"data/basis_vectors/graph_{N}_basis.npy")

    weights = create_random_geometric_graph(
        N,
        thres,
        seed,
        is_weighted=True,
        is_directed=True,
        is_connected=True,
        visualize=True,
    )
    DiG = nx.from_numpy_array(weights, create_using=nx.DiGraph)
    save_graph_to_file(DiG, f"data/graphs/random_digraph_{N}.graphml")

    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    basis = compute_l1_norm_basis(N, weights)
    save_basis_vectors_to_file(basis, f"data/basis_vectors/digraph_{N}_basis.npy")
    tend = time.time()
    print(f"Time taken: {tend - tstart:.2f} seconds")
