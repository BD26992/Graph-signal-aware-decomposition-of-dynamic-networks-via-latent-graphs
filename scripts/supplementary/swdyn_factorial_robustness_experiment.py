"""Factorial SwDyn robustness experiment for nonlinear topology and white signals.

The four variants are:

- linear_smooth: original bilinear SwDyn topology and graph-smooth signals
- nonlinear_smooth: sin(.) applied to the bilinear topology and graph-smooth signals
- linear_white: original bilinear topology and white-noise signals
- nonlinear_white: sin(.) topology and white-noise signals
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
from dataclasses import dataclass
from itertools import permutations
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dynamic_graph_decomposition import (
    DecompositionParams,
    DynamicGraphDecomposer,
    generate_graph_smooth_signals,
    graph_laplacian,
    signal_distance_matrices,
)
from swdyn_paper_experiments import (
    f1_score_hidden,
    hidden_mask,
    relative_error_squared,
    stochastic_block_model,
    symmetric_observation_mask,
)


VARIANTS = [
    (False, False, "linear_smooth", "Linear topology, smooth signals"),
    (True, False, "nonlinear_smooth", "Sin topology, smooth signals"),
    (False, True, "linear_white", "Linear topology, white signals"),
    (True, True, "nonlinear_white", "Sin topology, white signals"),
]

VARIANT_BY_SLUG = {slug: (nonlinear, white, slug, label) for nonlinear, white, slug, label in VARIANTS}


@dataclass(frozen=True)
class VariantConfig:
    nonlinear: bool
    white_signals: bool
    slug: str
    label: str
    args: argparse.Namespace


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_true_components(
    *,
    n_nodes: int,
    n_times: int,
    seed: int,
    p4_in: float,
    p4_out: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels_2 = np.arange(n_nodes) * 2 // n_nodes
    graph_2 = stochastic_block_model(
        rng,
        n_nodes=n_nodes,
        n_communities=2,
        p_in=0.28,
        p_out=0.04,
    )
    labels_4 = np.arange(n_nodes) * 4 // n_nodes
    rng.shuffle(labels_4)
    probabilities = np.where(labels_4[:, None] == labels_4[None, :], p4_in, p4_out)
    upper = rng.random((n_nodes, n_nodes)) < probabilities
    graph_4 = np.triu(upper.astype(float), k=1)
    graph_4 = graph_4 + graph_4.T
    true_graphs = np.stack([(graph_2 > 0).astype(float), (graph_4 > 0).astype(float)])
    for r in range(true_graphs.shape[0]):
        np.fill_diagonal(true_graphs[r], 0.0)

    c1 = np.linspace(1.0, 0.0, n_times)
    c2 = 1.0 - c1
    temporal = np.column_stack([c1, c2])
    return true_graphs, temporal, labels_2, labels_4


def make_variant_data(config: VariantConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    args = config.args
    true_graphs, temporal, labels_2, labels_4 = make_true_components(
        n_nodes=args.n_nodes,
        n_times=args.n_times,
        seed=args.seed,
        p4_in=args.p4_in,
        p4_out=args.p4_out,
    )
    if args.nonlinear_power_target == "component":
        powered_graphs = np.power(true_graphs, args.nonlinear_power)
        linear_adjacency = DynamicGraphDecomposer.reconstruct(powered_graphs, temporal)
    else:
        linear_adjacency = DynamicGraphDecomposer.reconstruct(true_graphs, temporal)
        if config.nonlinear:
            linear_adjacency = np.power(linear_adjacency, args.nonlinear_power)
    if config.nonlinear:
        adjacency = np.sin(args.nonlinear_scale * linear_adjacency)
    else:
        adjacency = linear_adjacency

    if config.white_signals:
        rng = np.random.default_rng(args.seed + 5000)
        signals = rng.standard_normal((args.n_nodes, args.n_signals, args.n_times))
    else:
        signals = np.empty((args.n_nodes, args.n_signals, args.n_times), dtype=float)
        for t in range(args.n_times):
            laplacian = graph_laplacian(adjacency[:, :, t])
            signals[:, :, t] = generate_graph_smooth_signals(
                laplacian,
                n_signals=args.n_signals,
                smoothness=args.signal_smoothness,
                noise_power=0.0,
                random_state=args.seed + t,
            )

    return adjacency, signals, true_graphs, labels_2, labels_4


def make_params(args: argparse.Namespace, *, iteration_seed: int) -> DecompositionParams:
    return DecompositionParams(
        rank=2,
        alpha=1.0,
        gamma=args.gamma,
        beta=args.beta,
        delta=args.delta,
        delta_A=args.delta,
        mu=args.mu,
        rho=args.rho,
        eta_graph=args.eta_graph,
        zeta=1e-4,
        outer_iterations=1,
        inner_iterations_A=5,
        inner_iterations_C=5,
        random_state=iteration_seed,
    )


def upper_values(matrix: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices_from(matrix, k=1)
    return np.asarray(matrix[rows, cols], dtype=float)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def correlation_matrix(true_graphs: np.ndarray, recovered_graphs: np.ndarray) -> np.ndarray:
    matrix = np.zeros((true_graphs.shape[0], recovered_graphs.shape[0]), dtype=float)
    for i in range(true_graphs.shape[0]):
        truth = upper_values(true_graphs[i])
        for j in range(recovered_graphs.shape[0]):
            matrix[i, j] = safe_correlation(truth, upper_values(recovered_graphs[j]))
    return matrix


def best_matching(correlations: np.ndarray) -> dict[int, int]:
    best_perm: tuple[int, ...] | None = None
    best_score = -math.inf
    scored = np.nan_to_num(correlations, nan=-math.inf, posinf=math.inf, neginf=-math.inf)
    for perm in permutations(range(correlations.shape[1]), correlations.shape[0]):
        score = sum(scored[i, perm[i]] for i in range(scored.shape[0]))
        if best_perm is None or score > best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None
    return {true_idx: recovered_idx for true_idx, recovered_idx in enumerate(best_perm)}


def community_stats(graph: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    rows, cols = np.triu_indices_from(graph, k=1)
    weights = graph[rows, cols]
    within = labels[rows] == labels[cols]
    within_mean = float(np.mean(weights[within])) if np.any(within) else float("nan")
    between_mean = float(np.mean(weights[~within])) if np.any(~within) else float("nan")
    ratio = float(within_mean / between_mean) if between_mean != 0.0 else float("inf")
    return within_mean, between_mean, ratio


def plot_metric_curves(rows: list[dict[str, object]], output: Path, *, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("observed_relative_error", "Observed RE"),
        ("unobserved_relative_error", "Unobserved RE"),
        ("observed_f1", "Observed F1"),
        ("unobserved_f1", "Unobserved F1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=True)
    x = np.array([int(row["iteration"]) for row in rows])
    for ax, (key, ylabel) in zip(axes.ravel(), metrics):
        y = np.array([float(row[key]) for row in rows])
        ax.plot(x, y, marker="o", markersize=3.2, linewidth=1.4)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Iterations")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_components(
    true_graphs: np.ndarray,
    recovered: np.ndarray,
    match: dict[int, int],
    output: Path,
    *,
    title: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("white_to_blue", ["#ffffff", "#08306b"])
    panels = [
        (true_graphs[0], "True 2-community"),
        (recovered[match[0]], "Matched recovered"),
        (true_graphs[1], "True 4-community"),
        (recovered[match[1]], "Matched recovered"),
    ]
    vmax = max(float(np.max(matrix)) for matrix, _ in panels)
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0))
    for ax, (matrix, subtitle) in zip(axes, panels):
        im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(subtitle, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_variant(config: VariantConfig) -> tuple[str, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    args = config.args
    adjacency, signals, true_graphs, labels_2, labels_4 = make_variant_data(config)
    scaled_signals = signals / np.sqrt(args.n_nodes * args.n_signals)
    distances = signal_distance_matrices(scaled_signals)
    mask = symmetric_observation_mask(
        adjacency.shape,
        args.observed_percent / 100.0,
        seed=args.mask_seed,
    )
    hmask = hidden_mask(mask)

    latent_graphs = None
    temporal_signatures = None
    metric_rows: list[dict[str, object]] = []
    solver = DynamicGraphDecomposer(make_params(args, iteration_seed=args.seed))

    for iteration in range(1, args.outer_iterations + 1):
        result = solver.fit(
            adjacency,
            scaled_signals,
            mask=mask,
            initial_latent_graphs=latent_graphs,
            initial_temporal_signatures=temporal_signatures,
            signal_distances=distances,
        )
        latent_graphs = result.latent_graphs
        temporal_signatures = result.temporal_signatures
        reconstruction = solver.reconstruct(latent_graphs, temporal_signatures)
        metric_rows.append(
            {
                "variant": config.slug,
                "variant_label": config.label,
                "nonlinear": config.nonlinear,
                "white_signals": config.white_signals,
                "iteration": iteration,
                "observed_percent": args.observed_percent,
                "rank": 2,
                "observed_relative_error": relative_error_squared(adjacency, reconstruction, mask),
                "unobserved_relative_error": relative_error_squared(adjacency, reconstruction, hmask),
                "observed_f1": f1_score_hidden(adjacency, reconstruction, mask),
                "unobserved_f1": f1_score_hidden(adjacency, reconstruction, hmask),
                "alpha": 1.0,
                "gamma": args.gamma,
                "beta": args.beta,
                "delta": args.delta,
                "delta_A": args.delta,
                "mu": args.mu,
                "rho": args.rho,
                "eta_graph": args.eta_graph,
            }
        )

    assert latent_graphs is not None
    corr = correlation_matrix(true_graphs, latent_graphs)
    match = best_matching(corr)
    correlation_rows: list[dict[str, object]] = []
    stat_rows: list[dict[str, object]] = []
    for true_idx in range(true_graphs.shape[0]):
        for rec_idx in range(latent_graphs.shape[0]):
            correlation_rows.append(
                {
                    "variant": config.slug,
                    "variant_label": config.label,
                    "true_component": true_idx + 1,
                    "recovered_component": rec_idx + 1,
                    "matched": rec_idx == match[true_idx],
                    "correlation": corr[true_idx, rec_idx],
                }
            )
        rec_idx = match[true_idx]
        labels = labels_2 if true_idx == 0 else labels_4
        within, between, ratio = community_stats(latent_graphs[rec_idx], labels)
        stat_rows.append(
            {
                "variant": config.slug,
                "variant_label": config.label,
                "true_component": true_idx + 1,
                "matched_recovered_component": rec_idx + 1,
                "matched_correlation": corr[true_idx, rec_idx],
                "within_mean": within,
                "between_mean": between,
                "within_between_ratio": ratio,
            }
        )

    variant_plot_dir = args.plot_dir / config.slug
    plot_metric_curves(
        metric_rows,
        variant_plot_dir / f"{config.slug}_convergence_metrics.pdf",
        title=config.label,
    )
    plot_components(
        true_graphs,
        latent_graphs,
        match,
        variant_plot_dir / f"{config.slug}_matched_components.pdf",
        title=config.label,
    )
    return config.slug, metric_rows, correlation_rows, stat_rows


def plot_all_variants(metric_rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("observed_relative_error", "Observed RE"),
        ("unobserved_relative_error", "Unobserved RE"),
        ("observed_f1", "Observed F1"),
        ("unobserved_f1", "Unobserved F1"),
    ]
    labels = {
        str(row["variant"]): str(row["variant_label"])
        for row in sorted(metric_rows, key=lambda item: str(item["variant"]))
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=True)
    for ax, (key, ylabel) in zip(axes.ravel(), metrics):
        for slug in labels:
            rows = [row for row in metric_rows if row["variant"] == slug]
            rows = sorted(rows, key=lambda row: int(row["iteration"]))
            ax.plot(
                [int(row["iteration"]) for row in rows],
                [float(row[key]) for row in rows],
                marker="o",
                markersize=2.8,
                linewidth=1.25,
                label=labels[slug],
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Iterations")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("SwDyn factorial robustness", fontsize=14)
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-nodes", type=int, default=40)
    parser.add_argument("--n-signals", type=int, default=1000)
    parser.add_argument("--n-times", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mask-seed", type=int, default=790)
    parser.add_argument("--observed-percent", type=int, default=90)
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--p4-in", type=float, default=0.65)
    parser.add_argument("--p4-out", type=float, default=0.06)
    parser.add_argument("--signal-smoothness", type=float, default=10.0)
    parser.add_argument("--nonlinear-scale", type=float, default=1.0)
    parser.add_argument("--nonlinear-power", type=float, default=1.0)
    parser.add_argument(
        "--nonlinear-power-target",
        choices=["mixture", "component"],
        default="mixture",
        help="Apply the power to the dynamic mixture or to each latent component before mixing.",
    )
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--delta", type=float, default=100.0)
    parser.add_argument(
        "--delta-grid",
        type=float,
        nargs="+",
        default=None,
        help="Optional delta sweep. If provided, overrides --delta and appends delta to variant slugs.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[slug for _, _, slug, _ in VARIANTS],
        default=[slug for _, _, slug, _ in VARIANTS],
        help="Subset of variants to run.",
    )
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=0.001)
    parser.add_argument("--eta-graph", type=float, default=0.0)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/csv/swdyn_factorial_robustness_metrics.csv"),
    )
    parser.add_argument(
        "--correlation-output",
        type=Path,
        default=Path("results/csv/swdyn_factorial_robustness_correlations.csv"),
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("results/csv/swdyn_factorial_robustness_component_stats.csv"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results/plots/swdyn_factorial_robustness"),
    )
    return parser.parse_args()


def delta_slug(delta: float) -> str:
    return f"delta_{delta:g}".replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    deltas = args.delta_grid if args.delta_grid is not None else [args.delta]
    configs = []
    for variant_slug in args.variants:
        nonlinear, white, slug, label = VARIANT_BY_SLUG[variant_slug]
        for delta in deltas:
            local_args = copy.copy(args)
            local_args.delta = delta
            if len(deltas) > 1:
                config_slug = f"{slug}_{delta_slug(delta)}"
                config_label = fr"{label}, $\delta={delta:g}$"
            else:
                config_slug = slug
                config_label = label
            configs.append(
                VariantConfig(
                    nonlinear=nonlinear,
                    white_signals=white,
                    slug=config_slug,
                    label=config_label,
                    args=local_args,
                )
            )
    if args.workers == 1:
        results = [run_variant(config) for config in configs]
    else:
        with Pool(processes=min(args.workers, len(configs))) as pool:
            results = pool.map(run_variant, configs)

    metric_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    stat_rows: list[dict[str, object]] = []
    for _slug, metrics, correlations, stats in results:
        metric_rows.extend(metrics)
        correlation_rows.extend(correlations)
        stat_rows.extend(stats)

    metric_rows.sort(key=lambda row: (str(row["variant"]), int(row["iteration"])))
    correlation_rows.sort(
        key=lambda row: (
            str(row["variant"]),
            int(row["true_component"]),
            int(row["recovered_component"]),
        )
    )
    stat_rows.sort(key=lambda row: (str(row["variant"]), int(row["true_component"])))

    write_csv(args.metrics_output, metric_rows)
    write_csv(args.correlation_output, correlation_rows)
    write_csv(args.stats_output, stat_rows)
    plot_all_variants(metric_rows, args.plot_dir / "swdyn_factorial_all_metrics.pdf")

    print(f"wrote {args.metrics_output}")
    print(f"wrote {args.correlation_output}")
    print(f"wrote {args.stats_output}")
    print(f"wrote {args.plot_dir / 'swdyn_factorial_all_metrics.pdf'}")


if __name__ == "__main__":
    main()
