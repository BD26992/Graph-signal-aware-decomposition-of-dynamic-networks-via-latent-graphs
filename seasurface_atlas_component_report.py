"""SeaSurface component atlas plots and graph-metric report."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

import contextily as ctx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection

from plot_seasurface_latent_atlas import (
    fit_seasurface,
    load_surface_positions,
    lonlat_to_webmercator,
    strongest_edges,
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


def top_k_adjacency(graph: np.ndarray, top_edges: int) -> tuple[np.ndarray, np.ndarray]:
    graph = 0.5 * (graph + graph.T)
    rows, cols, weights = strongest_edges(graph, top_edges)
    adjacency = np.zeros_like(graph, dtype=int)
    weighted = np.zeros_like(graph, dtype=float)
    adjacency[rows, cols] = 1
    adjacency[cols, rows] = 1
    weighted[rows, cols] = weights
    weighted[cols, rows] = weights
    return adjacency, weighted


def edge_set(adjacency: np.ndarray) -> set[tuple[int, int]]:
    rows, cols = np.triu_indices_from(adjacency, k=1)
    keep = adjacency[rows, cols] > 0
    return set(zip(rows[keep].astype(int).tolist(), cols[keep].astype(int).tolist()))


def degree_histogram_string(degrees: np.ndarray) -> str:
    values, counts = np.unique(degrees.astype(int), return_counts=True)
    return ";".join(f"{int(value)}:{int(count)}" for value, count in zip(values, counts))


def graph_metrics(
    dataset: str,
    component: int,
    adjacency: np.ndarray,
    weighted: np.ndarray,
    *,
    observed_percent: int,
    top_edges: int,
) -> dict[str, object]:
    graph = nx.from_numpy_array(adjacency)
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    strengths = weighted.sum(axis=1)
    if graph.number_of_edges() > 0:
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
        modularity = nx.algorithms.community.modularity(graph, communities)
    else:
        communities = []
        modularity = float("nan")
    try:
        assortativity = nx.degree_assortativity_coefficient(graph)
    except Exception:
        assortativity = float("nan")
    clustering = nx.average_clustering(graph) if graph.number_of_nodes() else float("nan")
    transitivity = nx.transitivity(graph)
    return {
        "dataset": dataset,
        "observed_percent": observed_percent,
        "rank": 3,
        "component": component,
        "top_edges": top_edges,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "degree_histogram": degree_histogram_string(degrees),
        "degree_mean": float(np.mean(degrees)),
        "degree_std": float(np.std(degrees, ddof=1)),
        "degree_max": int(np.max(degrees)),
        "degree_min": int(np.min(degrees)),
        "strength_mean": float(np.mean(strengths)),
        "strength_max": float(np.max(strengths)),
        "average_clustering_coefficient": float(clustering),
        "transitivity": float(transitivity),
        "community_count": len(communities),
        "modularity": float(modularity),
        "assortativity": float(assortativity),
    }


def jaccard_rows(
    dataset: str,
    adjacencies: list[np.ndarray],
    *,
    observed_percent: int,
    top_edges: int,
) -> list[dict[str, object]]:
    sets = [edge_set(adjacency) for adjacency in adjacencies]
    rows: list[dict[str, object]] = []
    for i in range(3):
        for j in range(i + 1, 3):
            intersection = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            rows.append(
                {
                    "dataset": dataset,
                    "observed_percent": observed_percent,
                    "rank": 3,
                    "component_i": i + 1,
                    "component_j": j + 1,
                    "top_edges": top_edges,
                    "edges_i": len(sets[i]),
                    "edges_j": len(sets[j]),
                    "intersection": intersection,
                    "union": union,
                    "jaccard": float(intersection / union) if union else float("nan"),
                    "overlap_over_min": float(intersection / min(len(sets[i]), len(sets[j])))
                    if min(len(sets[i]), len(sets[j]))
                    else float("nan"),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_component_atlas(
    lon: np.ndarray,
    lat: np.ndarray,
    graph: np.ndarray,
    component: int,
    output_pdf: Path,
    output_png: Path,
    *,
    top_edges: int,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    x_coord, y_coord = lonlat_to_webmercator(lon, lat)
    rows, cols, weights = strongest_edges(graph, top_edges)
    fig, ax = plt.subplots(figsize=(6.5, 6.6))
    pad_x = 600000.0
    pad_y = 600000.0
    ax.set_xlim(float(np.min(x_coord) - pad_x), float(np.max(x_coord) + pad_x))
    ax.set_ylim(float(np.min(y_coord) - pad_y), float(np.max(y_coord) + pad_y))
    ctx.add_basemap(
        ax,
        source=ctx.providers.Esri.OceanBasemap,
        zoom=3,
        attribution_size=5,
        reset_extent=False,
        alpha=0.92,
    )
    if weights.size:
        points = np.column_stack([x_coord, y_coord])
        segments = np.stack([points[rows], points[cols]], axis=1)
        norm = weights / max(float(np.max(weights)), 1e-12)
        collection = LineCollection(
            segments,
            colors=[(0.06, 0.14, 0.24, 0.13 + 0.62 * float(v)) for v in norm],
            linewidths=0.4 + 2.6 * norm,
            zorder=2,
        )
        ax.add_collection(collection)
    colors = ["#1f77b4", "#d95f02", "#2ca02c"]
    ax.scatter(
        x_coord,
        y_coord,
        s=30,
        c=colors[component - 1],
        edgecolors="white",
        linewidths=0.65,
        zorder=3,
    )
    lon_ticks = np.arange(-170, -80, 20)
    lat_ticks = np.arange(-60, 20, 20)
    x_ticks, _ = lonlat_to_webmercator(lon_ticks, np.zeros_like(lon_ticks))
    _, y_ticks = lonlat_to_webmercator(np.zeros_like(lat_ticks), lat_ticks)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{value:g}" for value in lon_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{value:g}" for value in lat_ticks])
    ax.set_xlabel("Longitude", fontsize=24)
    ax.set_ylabel("Latitude", fontsize=24)
    ax.grid(True, color="white", linewidth=0.75, alpha=0.68)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_title(rf"SeaSurf $\mathbf{{A}}_{component}$", fontsize=28, pad=12)
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_signatures(temporal_signatures: np.ndarray, output_pdf: Path, output_png: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#d95f02", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    time = np.arange(1, temporal_signatures.shape[0] + 1)
    for component in range(3):
        ax.plot(
            time,
            temporal_signatures[:, component],
            marker="o",
            linewidth=2.4,
            color=colors[component],
            label=f"c{component + 1}",
        )
    ax.set_title(r"SeaSurf Temporal Signatures", fontsize=28, pad=12)
    ax.set_xlabel("Time index", fontsize=24)
    ax.set_ylabel("Coefficient", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
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
    parser.add_argument("--top-edges", type=int, default=150)
    parser.add_argument(
        "--latent-output",
        type=Path,
        default=Path("results/data/seasurface_report_R3_obs90_outer50_latents.npz"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/csv/seasurface_report_R3_obs90_outer50_component_metrics.csv"),
    )
    parser.add_argument(
        "--jaccard-output",
        type=Path,
        default=Path("results/csv/seasurface_report_R3_obs90_outer50_component_jaccard.csv"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results/plots/atlas/seasurface_report_R3_obs90_outer50"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adjacency, latent_graphs, temporal_signatures, reconstruction = fit_seasurface(
        observed_percent=args.observed_percent,
        mask_seed=args.mask_seed,
        outer_iterations=args.outer_iterations,
        seed=args.seed,
        beta_factor=1.0,
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

    lon, lat = load_surface_positions()
    metric_rows: list[dict[str, object]] = []
    binary_components: list[np.ndarray] = []
    for component in range(1, 4):
        binary, weighted = top_k_adjacency(latent_graphs[component - 1], args.top_edges)
        binary_components.append(binary)
        metric_rows.append(
            graph_metrics(
                "SeaSurface",
                component,
                binary,
                weighted,
                observed_percent=args.observed_percent,
                top_edges=args.top_edges,
            )
        )
        plot_component_atlas(
            lon,
            lat,
            latent_graphs[component - 1],
            component,
            args.plot_dir / f"seasurface_A{component}_atlas.pdf",
            args.plot_dir / f"seasurface_A{component}_atlas.png",
            top_edges=args.top_edges,
        )
    j_rows = jaccard_rows(
        "SeaSurface",
        binary_components,
        observed_percent=args.observed_percent,
        top_edges=args.top_edges,
    )
    write_csv(args.metrics_output, metric_rows)
    write_csv(args.jaccard_output, j_rows)
    plot_signatures(
        temporal_signatures,
        args.plot_dir / "seasurface_temporal_signatures.pdf",
        args.plot_dir / "seasurface_temporal_signatures.png",
    )
    print(f"wrote {args.latent_output}")
    print(f"wrote {args.metrics_output}")
    print(f"wrote {args.jaccard_output}")
    for component in range(1, 4):
        print(f"wrote {args.plot_dir / f'seasurface_A{component}_atlas.pdf'}")
    print(f"wrote {args.plot_dir / 'seasurface_temporal_signatures.pdf'}")


if __name__ == "__main__":
    main()
