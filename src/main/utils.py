import numpy as np
import os
import shutil
import networkx as nx
import numpy as np
from itertools import combinations
from typing import Optional, Union
import matplotlib.pyplot as plt
import warnings



def pretty_time(seconds: float) -> str:
    """
    Convert seconds to a human-readable string format.
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
    else:
        days = seconds / 86400
        return f"{days:.2f} days"


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


def _draw_arrows_with_gap(
    ax,
    G: nx.Graph,
    pos,
    gap=0.03,
    arrowsize=30,
    edge_color="#555555",
    edge_widths=None,
    alpha=0.6,
):
    """
    Draw directed edges with a fixed gap between the arrow tip and the node center.
    gap: float, in data coordinates (e.g., 0.05 for 5% of axis)
    """
    if edge_widths is None:
        edge_widths = [1.5] * G.number_of_edges()

    for idx, (u, v) in enumerate(G.edges()):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        dist = np.hypot(dx, dy)
        if dist == 0:
            continue
        # Shorten the arrow by 'gap' at the end
        gapx = gap * dx / dist
        gapy = gap * dy / dist
        ax.annotate(
            "",
            xy=(x1 - gapx, y1 - gapy),
            xytext=(x0 + gapx, y0 + gapy),
            arrowprops=dict(
                arrowstyle="->" if G.is_directed() else "-",
                color=edge_color,
                lw=edge_widths[idx] if edge_widths else 1.5,
                alpha=alpha,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=arrowsize,
            ),
            zorder=1,
        )


def visualize_graph(
    G: nx.Graph, file_name: str, show_edge_weights: bool = False
) -> None:
    """
    Visualize a (small) graph G whose node positions are stored in "pos".

    - Edges are drawn with thickness ∝ weight (if weighted), otherwise uniform.
    - Nodes are colored by their distance to the center (0.5,0.5) and labeled.
    - Edge‐weight labels are drawn last with a small white box for readability.
    """
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        # Assign each node a random (x,y) in [0,1] so that _visualize_graph can use it.
        pos = nx.random_layout(G)
        for v in G.nodes():
            G.nodes[v]["pos"] = pos[v]
            warnings.warn(
                f"Node {v} has no 'pos' attribute, assigning random position."
            )

    centroid = np.mean(np.array(list(pos.values())), axis=0)
    offset = np.array([0.5, 0.5]) - centroid
    for n in pos:
        pos[n] = np.array(pos[n]) + offset
    p = {n: np.linalg.norm(np.array(pos[n]) - 0.5) for n in pos}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    if nx.is_weighted(G):
        weights = nx.get_edge_attributes(G, "weight")
        wvals = np.array(list(weights.values()))
        minw, maxw = wvals.min(), wvals.max()

        def _scaled_lw(w):
            if maxw == minw:
                return 2.0
            return 0.5 + 3.5 * ((w - minw) / (maxw - minw))

        edge_widths = [_scaled_lw(weights[e]) for e in G.edges()]
    else:
        edge_widths = [1.5] * G.number_of_edges()

    # Draw edges with fixed gap
    _draw_arrows_with_gap(
        ax,
        G,
        pos,
        gap=0.04,
        arrowsize=20,
        edge_color="#777777",
        edge_widths=edge_widths,
        alpha=1,
    )

    node_sizes = 1000
    node_colors = [p[n] for n in G.nodes()]
    node_collection = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(G.nodes()),
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.Reds,
        edgecolors="black",
        linewidths=1.2,
        alpha=1,
        ax=ax,
    )
    node_collection.set_zorder(2)
    for n in G.nodes():
        color = "black" if p[n] < 0.2 else "white"
        ax.text(
            pos[n][0],
            pos[n][1],
            str(n),
            fontsize=24,
            fontweight="bold",
            color=color,
            horizontalalignment="center",
            verticalalignment="center",
            zorder=3,
        )

    if nx.is_weighted(G):
        edge_labels = nx.get_edge_attributes(G, "weight")
        edge_labels = {e: f"{w:.3f}" for e, w in edge_labels.items()}

        if show_edge_weights:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_size=8,
                bbox=dict(boxstyle="round,pad=0.1", fc="none", ec="none", alpha=0.75),
                label_pos=0.5,
                rotate=False,
                clip_on=True,
                ax=ax,
            )

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, bbox_inches="tight")
        print(f"Graph visualization saved to {file_name}")
    plt.show()
    plt.close(fig)


def convert_to_directed(G: nx.Graph) -> nx.DiGraph:
    pos = nx.get_node_attributes(G, "pos")
    DiG = nx.DiGraph()
    DiG.add_nodes_from(G.nodes())
    if not pos is None:
        for v in G.nodes():
            DiG.nodes[v]["pos"] = pos[v]

    # For each undirected edge (u,v), pick one orientation based on x‐coordinate,
    # and copy over its weight if is_weighted, otherwise leave weight unset (or set to 1).
    is_weighted = nx.is_weighted(G)
    for u, v in G.edges():
        if pos[u][0] < pos[v][0]:
            DiG.add_edge(u, v)
            if is_weighted:
                DiG[u][v]["weight"] = G[u][v]["weight"]
        else:
            DiG.add_edge(v, u)
            if is_weighted:
                DiG[v][u]["weight"] = G[u][v]["weight"]
    return DiG


def create_random_geometric_graph(
    num_nodes: int,
    distance_threshold: Optional[float] = None,
    seed: Optional[int] = None,
    is_weighted: Optional[bool] = True,
    is_directed: Optional[bool] = True,
    is_connected: Optional[bool] = True,
    visualize: Optional[bool] = False,
) -> Union[nx.Graph, nx.DiGraph]:
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
    if distance_threshold is None:
        distance_threshold = 0.4
    if num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer.")
    if distance_threshold is not None and not (0 <= distance_threshold <= 1):
        raise ValueError("distance_threshold must be in [0, 1]")

    # Build the undirected random geometric graph
    G = nx.random_geometric_graph(num_nodes, radius=distance_threshold, seed=seed)
    if is_connected and not nx.is_connected(G):
        if seed is not None:
            raise ValueError(
                "Cannot enforce connectivity with a fixed seed. "
                "Please set seed=None to allow randomization."
            )
        # Keep drawing until the graph is connected, but without seed
        while is_connected and not nx.is_connected(G):
            G = nx.random_geometric_graph(num_nodes, radius=distance_threshold)

    # If weighted, draw exactly one random weight per undirected edge
    if is_weighted:
        rng = np.random.default_rng(seed)
        for u, v in G.edges():
            wt = float(np.round(rng.uniform(0.1, 1.0), 3))
            G[u][v]["weight"] = wt

    # If directed, create a DiGraph but carry over the exact same weight.
    if is_directed:
        G = convert_to_directed(G)

    # Visualize if requested
    if visualize:
        gtype = "digraph" if G.is_directed() else "graph"
        visualize_graph(
            G,
            file_name=f"data/temp/random_{gtype}_{num_nodes}.png",
        )
    return G


def create_random_erdos_renyi_graph(
    num_nodes: int,
    edge_probability: float = 0.5,
    seed: Optional[int] = None,
    is_weighted: Optional[bool] = True,
    is_directed: Optional[bool] = True,
    is_connected: Optional[bool] = True,
    visualize: Optional[bool] = False,
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Generate a random Erdos-Renyi graph, assign each node a random (x,y) ∈ [0,1]^2,
    and then (optionally) add weights and turn it into a directed graph. If visualize=True,
    _visualize_graph will find those 'pos' attributes and draw everything.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer.")
    if not (0 <= edge_probability <= 1):
        raise ValueError("edge_probability must be in [0, 1]")

    # Build the (undirected) Erdos-Renyi graph
    G = nx.erdos_renyi_graph(num_nodes, edge_probability, seed=seed)

    # Enforce connectivity if requested
    if is_connected and not nx.is_connected(G):
        if seed is not None:
            raise ValueError(
                "Cannot enforce connectivity with a fixed seed. "
                "Please set seed=None to allow randomization."
            )
        # Keep drawing until the graph is connected, but wihtout seed
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(num_nodes, edge_probability)

    # If weighted, draw exactly one random weight per undirected edge
    if is_weighted:
        rng = np.random.default_rng(seed)
        for u, v in G.edges():
            wt = float(np.round(rng.uniform(0.1, 1.0), 3))
            G[u][v]["weight"] = wt
    # Assign random (x,y) positions to each node in [0,1]^2
    pos = nx.random_layout(G)
    for v in G.nodes():
        G.nodes[v]["pos"] = pos[v]

    # If directed, build a DiGraph and copy over “pos” and “weight”
    if is_directed:
        DiG = nx.DiGraph()
        DiG.add_nodes_from(G.nodes())

        # Copy node positions into the directed graph
        for v in G.nodes():
            DiG.nodes[v]["pos"] = G.nodes[v]["pos"]

        # For each undirected (u,v), orient by x‐coordinate and copy weight if present:
        for u, v in G.edges():
            pu, pv = G.nodes[u]["pos"][0], G.nodes[v]["pos"][0]
            if pu < pv:
                DiG.add_edge(u, v)
                if is_weighted:
                    DiG[u][v]["weight"] = G[u][v]["weight"]
            else:
                DiG.add_edge(v, u)
                if is_weighted:
                    DiG[v][u]["weight"] = G[u][v]["weight"]

        G = DiG
    pos = nx.random_layout(G)
    for v in G.nodes():
        G.nodes[v]["pos"] = pos[v]
    # Visualize, if requested
    if visualize:
        visualize_graph(
            G,
        )
    return G


def main():
    from src.main.tools.io import load_graph_from_file

    for N in range(3, 11):
        G = load_graph_from_file(N, is_directed=False)
        visualize_graph(G, f"data/graphs/random_graph_{N}.png")
        G_dir = load_graph_from_file(N, is_directed=True)
        visualize_graph(G_dir, f"data/graphs/random_digraph_{N}.png")


if __name__ == "__main__":
    # Uncomment if you want to visualize all graphs in data/graphs
    main()
