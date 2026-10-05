"""Plot SeaSurface recovered DGD latent graphs in longitude/latitude space."""

from __future__ import annotations

import argparse
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


REPO_ROOT = Path(__file__).resolve().parent
SURFACE_POSITION_PATH = REPO_ROOT / "data/raw/surface/seasurfacetemperature.mat"


def load_surface_positions(path: Path = SURFACE_POSITION_PATH) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(path)
    if "Position" in mat:
        position = np.asarray(mat["Position"], dtype=float)
        if position.shape[1] != 2:
            raise ValueError(f"Position must have two columns, got {position.shape}")
        lat = position[:, 0]
        lon = position[:, 1]
        return lon, lat
    raise KeyError(f"{path} does not contain Position")


def strongest_edges(graph: np.ndarray, n_edges: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graph = 0.5 * (graph + graph.T)
    rows, cols = np.triu_indices_from(graph, k=1)
    weights = graph[rows, cols]
    positive = weights > 0.0
    rows, cols, weights = rows[positive], cols[positive], weights[positive]
    if weights.size == 0:
        return rows, cols, weights
    n_edges = min(n_edges, weights.size)
    idx = np.argpartition(weights, -n_edges)[-n_edges:]
    idx = idx[np.argsort(weights[idx])]
    return rows[idx], cols[idx], weights[idx]


def lonlat_to_webmercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = 6378137.0
    x = radius * np.deg2rad(lon)
    lat_clip = np.clip(lat, -85.05112878, 85.05112878)
    y = radius * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat_clip) / 2.0))
    return x, y


def webmercator_to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = 6378137.0
    lon = np.rad2deg(x / radius)
    lat = np.rad2deg(2.0 * np.arctan(np.exp(y / radius)) - np.pi / 2.0)
    return lon, lat


def fit_seasurface(
    *,
    observed_percent: int,
    mask_seed: int,
    outer_iterations: int,
    seed: int,
    beta_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = REAL_DATASETS["seasurface"]
    adjacency, signals = load_matlab_adjacency_signals(Path(spec["path"]))
    mask = matlab_mask_type2(adjacency, observed_percent, seed=mask_seed)
    n_nodes, n_signals, _ = signals.shape
    scaled_signals = signals / np.sqrt(n_nodes * n_signals)
    params = params_for_rank(spec, rank=3, seed=seed)
    params.beta *= beta_factor
    params.outer_iterations = outer_iterations
    solver = DynamicGraphDecomposer(params)
    result = solver.fit(adjacency, scaled_signals, mask=mask)
    reconstruction = solver.reconstruct(result.latent_graphs, result.temporal_signatures)
    return adjacency, result.latent_graphs, result.temporal_signatures, reconstruction


def plot_latent_atlas(
    lon: np.ndarray,
    lat: np.ndarray,
    latent_graphs: np.ndarray,
    temporal_signatures: np.ndarray,
    output_pdf: Path,
    output_png: Path,
    *,
    top_edges: int,
    basemap: str,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#d95f02", "#2ca02c"]
    extent_pad_lon = 5.0
    extent_pad_lat = 5.0
    use_basemap = basemap != "none"
    if use_basemap:
        x_coord, y_coord = lonlat_to_webmercator(lon, lat)
    else:
        x_coord, y_coord = lon, lat

    fig = plt.figure(figsize=(12.0, 8.0))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.55], hspace=0.32, wspace=0.2)

    for r in range(3):
        ax = fig.add_subplot(grid[0, r])
        ax.set_facecolor("#eef6fb")
        if use_basemap:
            try:
                import contextily as ctx

                provider = {
                    "ocean": ctx.providers.Esri.OceanBasemap,
                    "topo": ctx.providers.Esri.WorldTopoMap,
                    "terrain": ctx.providers.Esri.WorldTerrain,
                    "physical": ctx.providers.Esri.WorldPhysical,
                }[basemap]
                pad_x = 600000.0
                pad_y = 600000.0
                ax.set_xlim(float(np.min(x_coord) - pad_x), float(np.max(x_coord) + pad_x))
                ax.set_ylim(float(np.min(y_coord) - pad_y), float(np.max(y_coord) + pad_y))
                ctx.add_basemap(
                    ax,
                    source=provider,
                    zoom=3,
                    attribution_size=5,
                    reset_extent=False,
                    alpha=0.92,
                )
            except Exception as exc:
                print(f"warning: could not add basemap ({exc}); using plain coordinates", flush=True)
                use_basemap = False
        rows, cols, weights = strongest_edges(latent_graphs[r], top_edges)
        if weights.size:
            points = np.column_stack([x_coord, y_coord])
            segments = np.stack([points[rows], points[cols]], axis=1)
            norm = weights / max(float(np.max(weights)), 1e-12)
            widths = 0.35 + 2.4 * norm
            edge_colors = [(0.08, 0.16, 0.28, 0.12 + 0.55 * float(v)) for v in norm]
            collection = LineCollection(
                segments,
                colors=edge_colors,
                linewidths=widths,
                zorder=1,
            )
            ax.add_collection(collection)
        ax.scatter(
            x_coord,
            y_coord,
            s=24,
            c=colors[r],
            edgecolors="white",
            linewidths=0.55,
            zorder=2,
        )
        ax.set_title(f"Latent graph {r + 1}", fontsize=13, weight="bold")
        if use_basemap:
            lon_ticks = np.arange(-170, -80, 20)
            lat_ticks = np.arange(-60, 20, 20)
            x_ticks, _ = lonlat_to_webmercator(lon_ticks, np.zeros_like(lon_ticks))
            _, y_ticks = lonlat_to_webmercator(np.zeros_like(lat_ticks), lat_ticks)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{v:g}" for v in lon_ticks])
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"{v:g}" for v in lat_ticks])
            ax.set_xlabel("Longitude")
            if r == 0:
                ax.set_ylabel("Latitude")
            ax.grid(True, color="white", linewidth=0.7, alpha=0.65)
        else:
            ax.set_xlabel("Longitude")
            if r == 0:
                ax.set_ylabel("Latitude")
            ax.set_xlim(float(np.min(lon) - extent_pad_lon), float(np.max(lon) + extent_pad_lon))
            ax.set_ylim(float(np.min(lat) - extent_pad_lat), float(np.max(lat) + extent_pad_lat))
            ax.grid(True, color="white", linewidth=1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.02,
            0.03,
            f"top {top_edges} edges",
            transform=ax.transAxes,
            fontsize=9,
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 3},
        )

    ax_c = fig.add_subplot(grid[1, :])
    time = np.arange(1, temporal_signatures.shape[0] + 1)
    for r in range(3):
        ax_c.plot(
            time,
            temporal_signatures[:, r],
            color=colors[r],
            marker="o",
            linewidth=2.2,
            label=f"c{r + 1}",
        )
    ax_c.set_title("Temporal signatures", fontsize=13, weight="bold")
    ax_c.set_xlabel("Time index")
    ax_c.set_ylabel("Coefficient")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(ncol=3, frameon=False)

    title_suffix = "with ocean basemap" if basemap == "ocean" else "in atlas coordinates"
    fig.suptitle(f"SeaSurface DGD latent graphs {title_suffix}", fontsize=15, weight="bold")
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-percent", type=int, default=90)
    parser.add_argument("--mask-seed", type=int, default=91)
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-edges", type=int, default=400)
    parser.add_argument("--beta-factor", type=float, default=1.0)
    parser.add_argument(
        "--basemap",
        choices=["none", "ocean", "topo", "terrain", "physical"],
        default="none",
    )
    parser.add_argument(
        "--reuse-latents",
        action="store_true",
        help="Load latent graphs from --latent-output instead of fitting DGD again.",
    )
    parser.add_argument(
        "--latent-output",
        type=Path,
        default=Path("results/data/seasurface_dgd_R3_obs90_outer50_latents.npz"),
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=Path("results/plots/atlas/seasurface_dgd_R3_obs90_outer50_atlas.pdf"),
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=Path("results/plots/atlas/seasurface_dgd_R3_obs90_outer50_atlas.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse_latents:
        latents = np.load(args.latent_output)
        latent_graphs = latents["latent_graphs"]
        temporal_signatures = latents["temporal_signatures"]
    else:
        adjacency, latent_graphs, temporal_signatures, reconstruction = fit_seasurface(
            observed_percent=args.observed_percent,
            mask_seed=args.mask_seed,
            outer_iterations=args.outer_iterations,
            seed=args.seed,
            beta_factor=args.beta_factor,
        )
        args.latent_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.latent_output,
            adjacency=adjacency,
            latent_graphs=latent_graphs,
            temporal_signatures=temporal_signatures,
            reconstruction=reconstruction,
            observed_percent=args.observed_percent,
            mask_seed=args.mask_seed,
            outer_iterations=args.outer_iterations,
            seed=args.seed,
            beta_factor=args.beta_factor,
        )
    lon, lat = load_surface_positions()
    plot_latent_atlas(
        lon,
        lat,
        latent_graphs,
        temporal_signatures,
        args.pdf_output,
        args.png_output,
        top_edges=args.top_edges,
        basemap=args.basemap,
    )
    print(f"wrote {args.latent_output}")
    print(f"wrote {args.pdf_output}")
    print(f"wrote {args.png_output}")


if __name__ == "__main__":
    main()
