"""Contact component graph-layout plots and graph-metric report."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection

from convergence_observed_curves import REAL_DATASETS
from dgd_rank_sweep_observed_error import params_for_rank
from dynamic_graph_decomposition import DynamicGraphDecomposer
from plot_seasurface_latent_atlas import strongest_edges
from seasurf_paper_experiments import load_matlab_adjacency_signals, matlab_mask_type2
from seasurface_atlas_component_report import (
    graph_metrics,
    jaccard_rows,
    top_k_adjacency,
    write_csv,
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
    }
)


def fit_contact(
    *,
    observed_percent: int,
    mask_seed: int,
    outer_iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = REAL_DATASETS["contact"]
    adjacency, signals = load_matlab_adjacency_signals(Path(spec["path"]))
    mask = matlab_mask_type2(adjacency, observed_percent, seed=mask_seed)
    n_nodes, n_signals, _ = signals.shape
    scaled_signals = signals / np.sqrt(n_nodes * n_signals)
    params = params_for_rank(spec, rank=3, seed=seed)
    params.outer_iterations = outer_iterations
    solver = DynamicGraphDecomposer(params)
    result = solver.fit(adjacency, scaled_signals, mask=mask)
    reconstruction = solver.reconstruct(result.latent_graphs, result.temporal_signatures)
    return adjacency, result.latent_graphs, result.temporal_signatures, reconstruction


def graph_from_adjacency(adjacency: np.ndarray) -> nx.Graph:
    graph = nx.from_numpy_array(adjacency)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def community_labels(graph: nx.Graph) -> dict[int, int]:
    if graph.number_of_edges() == 0:
        return {node: 0 for node in graph.nodes}
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    labels: dict[int, int] = {}
    for idx, community in enumerate(communities):
        for node in community:
            labels[int(node)] = idx
    return labels


def union_layout(binary_components: list[np.ndarray], seed: int) -> dict[int, np.ndarray]:
    union = np.zeros_like(binary_components[0], dtype=int)
    for adjacency in binary_components:
        union = np.maximum(union, adjacency)
    graph = graph_from_adjacency(union)
    return nx.spring_layout(graph, seed=seed, k=0.42, iterations=300)


def plot_component_graph(
    graph_matrix: np.ndarray,
    binary: np.ndarray,
    component: int,
    positions: dict[int, np.ndarray],
    output_pdf: Path,
    output_png: Path,
    *,
    top_edges: int,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    rows, cols, weights = strongest_edges(graph_matrix, top_edges)
    binary_graph = graph_from_adjacency(binary)
    labels = community_labels(binary_graph)
    palette = plt.get_cmap("tab20")
    node_colors = [palette(labels.get(node, 0) % 20) for node in binary_graph.nodes]

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.set_facecolor("#f7f8fb")
    if weights.size:
        pos_array = np.array([positions[node] for node in range(graph_matrix.shape[0])])
        segments = np.stack([pos_array[rows], pos_array[cols]], axis=1)
        norm = weights / max(float(np.max(weights)), 1e-12)
        collection = LineCollection(
            segments,
            colors=[(0.05, 0.12, 0.22, 0.10 + 0.62 * float(value)) for value in norm],
            linewidths=0.35 + 2.4 * norm,
            zorder=1,
        )
        ax.add_collection(collection)
    xy = np.array([positions[node] for node in binary_graph.nodes])
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=34,
        c=node_colors,
        edgecolors="white",
        linewidths=0.6,
        zorder=2,
    )
    ax.set_title(rf"Contact $\mathbf{{A}}_{component}$", fontsize=28, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_signatures(temporal_signatures: np.ndarray, output_pdf: Path, output_png: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#d95f02", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    time = np.arange(1, temporal_signatures.shape[0] + 1)
    for component in range(3):
        ax.plot(
            time,
            temporal_signatures[:, component],
            linewidth=2.1,
            color=colors[component],
            label=f"c{component + 1}",
        )
    ax.set_title("Contact Temporal Signatures", fontsize=28, pad=12)
    ax.set_xlabel("Time index", fontsize=24)
    ax.set_ylabel("Coefficient", fontsize=24)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-percent", type=int, default=90)
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--mask-seed", type=int, default=91)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout-seed", type=int, default=12)
    parser.add_argument("--top-edges", type=int, default=150)
    parser.add_argument(
        "--latent-output",
        type=Path,
        default=Path("results/data/contact_report_R3_obs90_outer50_latents.npz"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/csv/contact_report_R3_obs90_outer50_component_metrics.csv"),
    )
    parser.add_argument(
        "--jaccard-output",
        type=Path,
        default=Path("results/csv/contact_report_R3_obs90_outer50_component_jaccard.csv"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results/plots/atlas/contact_report_R3_obs90_outer50"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adjacency, latent_graphs, temporal_signatures, reconstruction = fit_contact(
        observed_percent=args.observed_percent,
        mask_seed=args.mask_seed,
        outer_iterations=args.outer_iterations,
        seed=args.seed,
    )
    args.latent_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.latent_output,
        adjacency=adjacency,
        latent_graphs=latent_graphs,
        temporal_signatures=temporal_signatures,
        reconstruction=reconstruction,
        observed_percent=args.observed_percent,
        outer_iterations=args.outer_iterations,
        mask_seed=args.mask_seed,
        seed=args.seed,
    )
    metric_rows: list[dict[str, object]] = []
    binary_components: list[np.ndarray] = []
    weighted_components: list[np.ndarray] = []
    for component in range(1, 4):
        binary, weighted = top_k_adjacency(latent_graphs[component - 1], args.top_edges)
        binary_components.append(binary)
        weighted_components.append(weighted)
        metric_rows.append(
            graph_metrics(
                "Contact",
                component,
                binary,
                weighted,
                observed_percent=args.observed_percent,
                top_edges=args.top_edges,
            )
        )
    positions = union_layout(binary_components, args.layout_seed)
    for component in range(1, 4):
        plot_component_graph(
            latent_graphs[component - 1],
            binary_components[component - 1],
            component,
            positions,
            args.plot_dir / f"contact_A{component}_spring_communities.pdf",
            args.plot_dir / f"contact_A{component}_spring_communities.png",
            top_edges=args.top_edges,
        )
    write_csv(args.metrics_output, metric_rows)
    write_csv(
        args.jaccard_output,
        jaccard_rows(
            "Contact",
            binary_components,
            observed_percent=args.observed_percent,
            top_edges=args.top_edges,
        ),
    )
    plot_signatures(
        temporal_signatures,
        args.plot_dir / "contact_temporal_signatures.pdf",
        args.plot_dir / "contact_temporal_signatures.png",
    )
    print(f"wrote {args.latent_output}")
    print(f"wrote {args.metrics_output}")
    print(f"wrote {args.jaccard_output}")
    for component in range(1, 4):
        print(f"wrote {args.plot_dir / f'contact_A{component}_spring_communities.pdf'}")
    print(f"wrote {args.plot_dir / 'contact_temporal_signatures.pdf'}")


if __name__ == "__main__":
    main()
