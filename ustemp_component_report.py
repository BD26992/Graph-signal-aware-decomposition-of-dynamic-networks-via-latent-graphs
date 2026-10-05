"""USTemp component-coordinate plots and graph-metric report."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.io import loadmat

from convergence_observed_curves import REAL_DATASETS
from dgd_rank_sweep_observed_error import params_for_rank
from dynamic_graph_decomposition import DynamicGraphDecomposer
from seasurf_paper_experiments import load_matlab_adjacency_signals, matlab_mask_type2
from seasurface_atlas_component_report import (
    graph_metrics,
    jaccard_rows,
    top_k_adjacency,
    write_csv,
)
from plot_seasurface_latent_atlas import strongest_edges


REPO_ROOT = Path(__file__).resolve().parent
COORD_PATH = REPO_ROOT / "data/raw/UStemp/UStemp_hXY.mat"

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


def load_ustemp_coordinates(path: Path = COORD_PATH) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(path)
    x = np.asarray(mat["h1hX"], dtype=float).squeeze()
    y = np.asarray(mat["h1hY"], dtype=float).squeeze()
    return x, y


def fit_ustemp(
    *,
    observed_percent: int,
    mask_seed: int,
    outer_iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = REAL_DATASETS["ustemp"]
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


def plot_component_layout(
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    graph: np.ndarray,
    component: int,
    output_pdf: Path,
    output_png: Path,
    *,
    top_edges: int,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    rows, cols, weights = strongest_edges(graph, top_edges)
    fig, ax = plt.subplots(figsize=(6.3, 6.2))
    ax.set_facecolor("#f6f8fb")
    if weights.size:
        points = np.column_stack([x_coord, y_coord])
        segments = np.stack([points[rows], points[cols]], axis=1)
        norm = weights / max(float(np.max(weights)), 1e-12)
        collection = LineCollection(
            segments,
            colors=[(0.07, 0.14, 0.24, 0.12 + 0.62 * float(value)) for value in norm],
            linewidths=0.4 + 2.6 * norm,
            zorder=1,
        )
        ax.add_collection(collection)
    colors = ["#1f77b4", "#d95f02", "#2ca02c"]
    ax.scatter(
        x_coord,
        y_coord,
        s=34,
        c=colors[component - 1],
        edgecolors="white",
        linewidths=0.7,
        zorder=2,
    )
    pad_x = 0.8
    pad_y = 0.8
    ax.set_xlim(float(np.min(x_coord) - pad_x), float(np.max(x_coord) + pad_x))
    ax.set_ylim(float(np.min(y_coord) - pad_y), float(np.max(y_coord) + pad_y))
    ax.grid(True, color="white", linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Coordinate 1", fontsize=24)
    ax.set_ylabel("Coordinate 2", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_title(rf"USTemp $\mathbf{{A}}_{component}$", fontsize=28, pad=12)
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
    ax.set_title("USTemp Temporal Signatures", fontsize=28, pad=12)
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
        default=Path("results/data/ustemp_report_R3_obs90_outer50_latents.npz"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/csv/ustemp_report_R3_obs90_outer50_component_metrics.csv"),
    )
    parser.add_argument(
        "--jaccard-output",
        type=Path,
        default=Path("results/csv/ustemp_report_R3_obs90_outer50_component_jaccard.csv"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results/plots/atlas/ustemp_report_R3_obs90_outer50"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adjacency, latent_graphs, temporal_signatures, reconstruction = fit_ustemp(
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
    x_coord, y_coord = load_ustemp_coordinates()
    metric_rows: list[dict[str, object]] = []
    binary_components: list[np.ndarray] = []
    for component in range(1, 4):
        binary, weighted = top_k_adjacency(latent_graphs[component - 1], args.top_edges)
        binary_components.append(binary)
        metric_rows.append(
            graph_metrics(
                "USTemp",
                component,
                binary,
                weighted,
                observed_percent=args.observed_percent,
                top_edges=args.top_edges,
            )
        )
        plot_component_layout(
            x_coord,
            y_coord,
            latent_graphs[component - 1],
            component,
            args.plot_dir / f"ustemp_A{component}_layout.pdf",
            args.plot_dir / f"ustemp_A{component}_layout.png",
            top_edges=args.top_edges,
        )
    write_csv(args.metrics_output, metric_rows)
    write_csv(
        args.jaccard_output,
        jaccard_rows(
            "USTemp",
            binary_components,
            observed_percent=args.observed_percent,
            top_edges=args.top_edges,
        ),
    )
    plot_signatures(
        temporal_signatures,
        args.plot_dir / "ustemp_temporal_signatures.pdf",
        args.plot_dir / "ustemp_temporal_signatures.png",
    )
    print(f"wrote {args.latent_output}")
    print(f"wrote {args.metrics_output}")
    print(f"wrote {args.jaccard_output}")
    for component in range(1, 4):
        print(f"wrote {args.plot_dir / f'ustemp_A{component}_layout.pdf'}")
    print(f"wrote {args.plot_dir / 'ustemp_temporal_signatures.pdf'}")


if __name__ == "__main__":
    main()
