"""Alternative Contact component visualizations from saved latent graphs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection

from contact_component_report import graph_from_adjacency
from seasurface_atlas_component_report import top_k_adjacency


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.titlesize": 24,
        "axes.labelsize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def union_community_order(binary_components: list[np.ndarray]) -> tuple[list[int], dict[int, int]]:
    union = np.maximum.reduce(binary_components)
    graph = graph_from_adjacency(union)
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    communities = sorted(communities, key=lambda community: (-len(community), min(community)))
    labels: dict[int, int] = {}
    order: list[int] = []
    for idx, community in enumerate(communities):
        sub = graph.subgraph(community)
        nodes = sorted(community, key=lambda node: (-sub.degree(node), node))
        for node in nodes:
            labels[int(node)] = idx
        order.extend(int(node) for node in nodes)
    return order, labels


def weighted_community_order(weighted_components: list[np.ndarray]) -> tuple[list[int], dict[int, int]]:
    union = np.maximum.reduce(weighted_components)
    graph = graph_from_adjacency(union)
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
    communities = sorted(communities, key=lambda community: (-len(community), min(community)))
    labels: dict[int, int] = {}
    order: list[int] = []
    for idx, community in enumerate(communities):
        sub = graph.subgraph(community)
        nodes = sorted(
            community,
            key=lambda node: (-sub.degree(node, weight="weight"), node),
        )
        for node in nodes:
            labels[int(node)] = idx
        order.extend(int(node) for node in nodes)
    return order, labels


def plot_sorted_adjacency_grid(
    binary_components: list[np.ndarray],
    order: list[int],
    output_pdf: Path,
    output_png: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    for idx, ax in enumerate(axes):
        matrix = binary_components[idx][np.ix_(order, order)]
        ax.imshow(1 - matrix, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(rf"Contact $\mathbf{{A}}_{idx + 1}$")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_matrix(
    binary_components: list[np.ndarray],
    order: list[int],
    output_pdf: Path,
    output_png: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    overlap = np.sum(np.stack(binary_components, axis=0), axis=0)
    matrix = overlap[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    image = ax.imshow(matrix, cmap="viridis", interpolation="nearest", vmin=0, vmax=3)
    ax.set_title("Contact Edge Sharing")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of components")
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def circular_positions(order: list[int]) -> dict[int, np.ndarray]:
    n_nodes = len(order)
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 - 2.0 * np.pi, n_nodes, endpoint=False)
    return {
        node: np.array([np.cos(angle), np.sin(angle)], dtype=float)
        for node, angle in zip(order, angles)
    }


def plot_circular_component(
    graph_matrix: np.ndarray,
    binary: np.ndarray,
    order: list[int],
    community_labels: dict[int, int],
    component: int,
    output_pdf: Path,
    output_png: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    positions = circular_positions(order)
    graph = graph_from_adjacency(binary)
    rows, cols = np.triu_indices_from(binary, k=1)
    keep = binary[rows, cols] > 0
    rows, cols = rows[keep], cols[keep]
    weights = 0.5 * (graph_matrix + graph_matrix.T)[rows, cols]
    norm = weights / max(float(np.max(weights)), 1e-12) if weights.size else weights
    points = np.array([positions[node] for node in range(binary.shape[0])])
    segments = np.stack([points[rows], points[cols]], axis=1) if len(rows) else np.empty((0, 2, 2))

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    if len(segments):
        collection = LineCollection(
            segments,
            colors=[(0.05, 0.12, 0.22, 0.08 + 0.55 * float(value)) for value in norm],
            linewidths=0.25 + 2.0 * norm,
            zorder=1,
        )
        ax.add_collection(collection)
    palette = plt.get_cmap("tab20")
    xy = np.array([positions[node] for node in order])
    colors = [palette(community_labels[node] % 20) for node in order]
    sizes = [18 + 5 * graph.degree(node) for node in order]
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=colors, edgecolors="white", linewidths=0.5, zorder=2)
    ax.set_title(rf"Contact $\mathbf{{A}}_{component}$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def weighted_metric_rows(
    weighted_components: list[np.ndarray],
    *,
    observed_percent: int,
    top_edges: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for component, weighted in enumerate(weighted_components, start=1):
        graph = graph_from_adjacency(weighted)
        degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
        strengths = np.array([degree for _, degree in graph.degree(weight="weight")], dtype=float)
        if graph.number_of_edges() > 0:
            communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
            modularity = nx.algorithms.community.modularity(graph, communities, weight="weight")
        else:
            communities = []
            modularity = float("nan")
        try:
            assortativity = nx.degree_assortativity_coefficient(graph, weight="weight")
        except Exception:
            assortativity = float("nan")
        clustering = nx.average_clustering(graph, weight="weight") if graph.number_of_nodes() else float("nan")
        transitivity = nx.transitivity(graph)
        rows.append(
            {
                "dataset": "Contact",
                "observed_percent": observed_percent,
                "rank": 3,
                "component": component,
                "top_edges": top_edges,
                "community_weighting": "weighted",
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
                "degree_mean": float(np.mean(degrees)),
                "degree_std": float(np.std(degrees, ddof=1)),
                "degree_max": int(np.max(degrees)),
                "degree_min": int(np.min(degrees)),
                "strength_mean": float(np.mean(strengths)),
                "strength_max": float(np.max(strengths)),
                "average_weighted_clustering_coefficient": float(clustering),
                "transitivity_binary_support": float(transitivity),
                "community_count": len(communities),
                "weighted_modularity": float(modularity),
                "weighted_assortativity": float(assortativity),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latents",
        type=Path,
        default=Path("results/data/contact_report_R3_obs90_outer50_latents.npz"),
    )
    parser.add_argument("--top-edges", type=int, default=150)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results/plots/atlas/contact_report_R3_obs90_outer50/alternative"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latents = np.load(args.latents)
    latent_graphs = latents["latent_graphs"]
    top_k_components = [top_k_adjacency(latent_graphs[idx], args.top_edges) for idx in range(3)]
    binary_components = [pair[0] for pair in top_k_components]
    weighted_components = [pair[1] for pair in top_k_components]
    order, labels = union_community_order(binary_components)
    plot_sorted_adjacency_grid(
        binary_components,
        order,
        args.plot_dir / "contact_sorted_adjacency_components.pdf",
        args.plot_dir / "contact_sorted_adjacency_components.png",
    )
    plot_overlap_matrix(
        binary_components,
        order,
        args.plot_dir / "contact_component_edge_sharing_matrix.pdf",
        args.plot_dir / "contact_component_edge_sharing_matrix.png",
    )
    for component in range(1, 4):
        plot_circular_component(
            latent_graphs[component - 1],
            binary_components[component - 1],
            order,
            labels,
            component,
            args.plot_dir / f"contact_A{component}_circular_communities.pdf",
            args.plot_dir / f"contact_A{component}_circular_communities.png",
        )
    weighted_plot_dir = args.plot_dir / "weighted_communities"
    weighted_order, weighted_labels = weighted_community_order(weighted_components)
    plot_sorted_adjacency_grid(
        binary_components,
        weighted_order,
        weighted_plot_dir / "contact_sorted_adjacency_components_weighted_communities.pdf",
        weighted_plot_dir / "contact_sorted_adjacency_components_weighted_communities.png",
    )
    for component in range(1, 4):
        plot_circular_component(
            latent_graphs[component - 1],
            binary_components[component - 1],
            weighted_order,
            weighted_labels,
            component,
            weighted_plot_dir / f"contact_A{component}_circular_weighted_communities.pdf",
            weighted_plot_dir / f"contact_A{component}_circular_weighted_communities.png",
        )
    write_csv(
        Path("results/csv/Graph data/contact_report_R3_obs90_outer50_weighted_component_metrics.csv"),
        weighted_metric_rows(
            weighted_components,
            observed_percent=90,
            top_edges=args.top_edges,
        ),
    )
    print(f"wrote {args.plot_dir / 'contact_sorted_adjacency_components.pdf'}")
    print(f"wrote {args.plot_dir / 'contact_component_edge_sharing_matrix.pdf'}")
    for component in range(1, 4):
        print(f"wrote {args.plot_dir / f'contact_A{component}_circular_communities.pdf'}")
        print(f"wrote {weighted_plot_dir / f'contact_A{component}_circular_weighted_communities.pdf'}")


if __name__ == "__main__":
    main()
