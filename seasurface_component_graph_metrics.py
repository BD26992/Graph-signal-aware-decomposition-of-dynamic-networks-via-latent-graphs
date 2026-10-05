"""Compare graph metrics for SeaSurface DGD latent components."""

from __future__ import annotations

import argparse
import csv
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = {
    "obs90": Path("results/data/seasurface_dgd_R3_obs90_outer50_latents.npz"),
    "obs50": Path("results/data/seasurface_dgd_R3_obs50_outer50_latents.npz"),
    "obs90_beta10": Path("results/data/seasurface_dgd_R3_obs90_outer50_beta10_latents.npz"),
    "obs90_beta100": Path("results/data/seasurface_dgd_R3_obs90_outer50_beta100_latents.npz"),
}


def top_k_binary_graph(graph: np.ndarray, top_edges: int) -> tuple[np.ndarray, np.ndarray]:
    graph = 0.5 * (graph + graph.T)
    rows, cols = np.triu_indices_from(graph, k=1)
    weights = graph[rows, cols]
    positive = weights > 0.0
    rows, cols, weights = rows[positive], cols[positive], weights[positive]
    adjacency = np.zeros_like(graph, dtype=int)
    weighted = np.zeros_like(graph, dtype=float)
    if weights.size == 0:
        return adjacency, weighted
    n_edges = min(top_edges, weights.size)
    idx = np.argpartition(weights, -n_edges)[-n_edges:]
    chosen_rows, chosen_cols, chosen_weights = rows[idx], cols[idx], weights[idx]
    adjacency[chosen_rows, chosen_cols] = 1
    adjacency[chosen_cols, chosen_rows] = 1
    weighted[chosen_rows, chosen_cols] = chosen_weights
    weighted[chosen_cols, chosen_rows] = chosen_weights
    return adjacency, weighted


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    n_nodes = adjacency.shape[0]
    seen = np.zeros(n_nodes, dtype=bool)
    components: list[list[int]] = []
    neighbors = [np.flatnonzero(adjacency[i]).tolist() for i in range(n_nodes)]
    for start in range(n_nodes):
        if seen[start]:
            continue
        seen[start] = True
        queue: deque[int] = deque([start])
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for nbr in neighbors[node]:
                if not seen[nbr]:
                    seen[nbr] = True
                    queue.append(nbr)
        components.append(component)
    return components


def average_shortest_path_largest_component(adjacency: np.ndarray, components: list[list[int]]) -> float:
    largest = max(components, key=len)
    if len(largest) <= 1:
        return float("nan")
    sub_nodes = set(largest)
    neighbors = {
        node: [nbr for nbr in np.flatnonzero(adjacency[node]).tolist() if nbr in sub_nodes]
        for node in largest
    }
    total_distance = 0
    n_pairs = 0
    for source in largest:
        dist = {source: 0}
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            for nbr in neighbors[node]:
                if nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        for target, distance in dist.items():
            if target > source:
                total_distance += distance
                n_pairs += 1
    return float(total_distance / n_pairs) if n_pairs else float("nan")


def local_clustering_coefficients(adjacency: np.ndarray) -> np.ndarray:
    degrees = adjacency.sum(axis=1)
    coefficients = np.zeros(adjacency.shape[0], dtype=float)
    for node, degree in enumerate(degrees):
        if degree < 2:
            continue
        nbrs = np.flatnonzero(adjacency[node])
        subgraph = adjacency[np.ix_(nbrs, nbrs)]
        neighbor_edges = float(np.sum(np.triu(subgraph, k=1)))
        coefficients[node] = 2.0 * neighbor_edges / (degree * (degree - 1))
    return coefficients


def transitivity(adjacency: np.ndarray) -> float:
    degrees = adjacency.sum(axis=1)
    triples = float(np.sum(degrees * (degrees - 1) / 2))
    if triples == 0.0:
        return 0.0
    triangles = float(np.trace(adjacency @ adjacency @ adjacency) / 6)
    return 3.0 * triangles / triples


def graph_metric_row(
    variant: str,
    component: int,
    adjacency: np.ndarray,
    weighted: np.ndarray,
) -> dict[str, object]:
    n_nodes = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    strengths = weighted.sum(axis=1)
    n_edges = int(np.sum(degrees) // 2)
    components = connected_components(adjacency)
    component_sizes = sorted((len(c) for c in components), reverse=True)
    clustering = local_clustering_coefficients(adjacency)
    return {
        "variant": variant,
        "component": component,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": 2.0 * n_edges / (n_nodes * (n_nodes - 1)),
        "degree_mean": float(np.mean(degrees)),
        "degree_std": float(np.std(degrees, ddof=1)),
        "degree_min": int(np.min(degrees)),
        "degree_max": int(np.max(degrees)),
        "strength_mean": float(np.mean(strengths)),
        "strength_std": float(np.std(strengths, ddof=1)),
        "average_clustering": float(np.mean(clustering)),
        "transitivity": transitivity(adjacency),
        "n_connected_components": len(components),
        "largest_component_size": component_sizes[0],
        "second_largest_component_size": component_sizes[1] if len(component_sizes) > 1 else 0,
        "average_shortest_path_largest_component": average_shortest_path_largest_component(
            adjacency,
            components,
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_degree_histograms(
    degree_rows: list[dict[str, object]],
    output_pdf: Path,
    output_png: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    variants = list(VARIANTS)
    colors = ["#1f77b4", "#d95f02", "#2ca02c"]
    fig, axes = plt.subplots(len(variants), 3, figsize=(12.0, 11.0), sharex=True, sharey=True)
    for row_idx, variant in enumerate(variants):
        for component in range(1, 4):
            ax = axes[row_idx, component - 1]
            degrees = [
                int(row["degree"])
                for row in degree_rows
                if row["variant"] == variant and int(row["component"]) == component
            ]
            ax.hist(
                degrees,
                bins=np.arange(-0.5, max(degrees) + 1.5, 1.0),
                color=colors[component - 1],
                alpha=0.78,
                edgecolor="white",
            )
            ax.grid(True, axis="y", alpha=0.28)
            if row_idx == 0:
                ax.set_title(f"Component {component}", fontsize=12, weight="bold")
            if component == 1:
                ax.set_ylabel(f"{variant}\ncount")
            if row_idx == len(variants) - 1:
                ax.set_xlabel("Degree")
    fig.suptitle("SeaSurface latent component degree histograms, top-150 edges", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(metric_rows: list[dict[str, object]], output_pdf: Path, output_png: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("average_clustering", "Avg. clustering"),
        ("transitivity", "Transitivity"),
        ("largest_component_size", "Largest CC size"),
        ("average_shortest_path_largest_component", "Avg. path, largest CC"),
    ]
    variants = list(VARIANTS)
    colors = ["#1f77b4", "#d95f02", "#2ca02c"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        x = np.arange(len(variants))
        width = 0.24
        for component in range(1, 4):
            values = []
            for variant in variants:
                row = next(
                    row
                    for row in metric_rows
                    if row["variant"] == variant and int(row["component"]) == component
                )
                values.append(float(row[metric]))
            ax.bar(
                x + (component - 2) * width,
                values,
                width=width,
                color=colors[component - 1],
                label=f"C{component}",
            )
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.28)
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.suptitle("SeaSurface latent component global graph metrics", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-edges", type=int, default=150)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/csv/seasurface_component_graph_metrics_top150.csv"),
    )
    parser.add_argument(
        "--degree-output",
        type=Path,
        default=Path("results/csv/seasurface_component_degrees_top150.csv"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results/plots/atlas"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_rows: list[dict[str, object]] = []
    degree_rows: list[dict[str, object]] = []
    for variant, path in VARIANTS.items():
        latents = np.load(path)
        latent_graphs = latents["latent_graphs"]
        for component_idx in range(latent_graphs.shape[0]):
            adjacency, weighted = top_k_binary_graph(latent_graphs[component_idx], args.top_edges)
            component = component_idx + 1
            metric_rows.append(graph_metric_row(variant, component, adjacency, weighted))
            degrees = adjacency.sum(axis=1)
            strengths = weighted.sum(axis=1)
            for node, (degree, strength) in enumerate(zip(degrees, strengths), start=1):
                degree_rows.append(
                    {
                        "variant": variant,
                        "component": component,
                        "node": node,
                        "degree": int(degree),
                        "strength": float(strength),
                    }
                )
    write_csv(args.metrics_output, metric_rows)
    write_csv(args.degree_output, degree_rows)
    plot_degree_histograms(
        degree_rows,
        args.plot_dir / "seasurface_component_degree_histograms_top150.pdf",
        args.plot_dir / "seasurface_component_degree_histograms_top150.png",
    )
    plot_metric_bars(
        metric_rows,
        args.plot_dir / "seasurface_component_global_metrics_top150.pdf",
        args.plot_dir / "seasurface_component_global_metrics_top150.png",
    )
    print(f"wrote {args.metrics_output}")
    print(f"wrote {args.degree_output}")
    print(f"wrote {args.plot_dir / 'seasurface_component_degree_histograms_top150.pdf'}")
    print(f"wrote {args.plot_dir / 'seasurface_component_global_metrics_top150.pdf'}")


if __name__ == "__main__":
    main()
