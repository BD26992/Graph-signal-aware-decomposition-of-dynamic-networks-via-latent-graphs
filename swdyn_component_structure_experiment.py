"""SwDyn DGD component-structure diagnostic.

This experiment fits DGD to the synthetic SwDyn data under three beta/delta
settings, matches recovered latent graphs to the true two- and four-community
SBM graphs by correlation, and reports within/between community statistics.
"""

from __future__ import annotations

import argparse
import csv
import os
from itertools import permutations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

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
)
from swdyn_paper_experiments import stochastic_block_model


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    }
)


def make_swdyn_with_labels(
    *,
    n_nodes: int = 40,
    n_signals: int = 1000,
    n_times: int = 50,
    seed: int = 7,
    p4_in: float = 0.36,
    p4_out: float = 0.035,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    graph_2 = positive_support_graph(graph_2)
    graph_4 = positive_support_graph(graph_4)

    true_graphs = np.stack([graph_2, graph_4], axis=0)
    c1 = np.linspace(1.0, 0.0, n_times)
    c2 = 1.0 - c1
    temporal = np.column_stack([c1, c2])
    adjacency = DynamicGraphDecomposer.reconstruct(true_graphs, temporal)

    signals = np.empty((n_nodes, n_signals, n_times), dtype=float)
    for t in range(n_times):
        laplacian = graph_laplacian(adjacency[:, :, t])
        signals[:, :, t] = generate_graph_smooth_signals(
            laplacian,
            n_signals=n_signals,
            smoothness=10.0,
            noise_power=0.0,
            random_state=seed + t,
        )
    return adjacency, signals, true_graphs, labels_2, labels_4


def upper_values(matrix: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices_from(matrix, k=1)
    return np.asarray(matrix[rows, cols], dtype=float)


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def correlation_matrix(true_graphs: np.ndarray, recovered_graphs: np.ndarray) -> np.ndarray:
    matrix = np.zeros((true_graphs.shape[0], recovered_graphs.shape[0]), dtype=float)
    for i in range(true_graphs.shape[0]):
        truth = upper_values(true_graphs[i])
        for j in range(recovered_graphs.shape[0]):
            matrix[i, j] = safe_correlation(truth, upper_values(recovered_graphs[j]))
    return matrix


def best_matching(correlations: np.ndarray) -> dict[int, int]:
    best_perm: tuple[int, ...] | None = None
    best_score = -np.inf
    for perm in permutations(range(correlations.shape[1]), correlations.shape[0]):
        score = sum(correlations[i, perm[i]] for i in range(correlations.shape[0]))
        if score > best_score:
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


def positive_support_graph(graph: np.ndarray) -> np.ndarray:
    graph = 0.5 * (graph + graph.T)
    binary = (graph > 0.0).astype(float)
    np.fill_diagonal(binary, 0.0)
    return binary


def params_for_setting(
    *,
    rank: int,
    beta: float,
    delta: float,
    outer_iterations: int,
    seed: int,
) -> DecompositionParams:
    return DecompositionParams(
        rank=rank,
        alpha=1.0,
        gamma=0.01,
        beta=beta,
        delta=delta,
        delta_A=delta,
        mu=0.1,
        rho=0.001,
        eta_graph=0.0,
        zeta=1e-4,
        outer_iterations=outer_iterations,
        inner_iterations_A=5,
        inner_iterations_C=5,
        random_state=seed,
    )


def run_setting(
    adjacency: np.ndarray,
    signals: np.ndarray,
    *,
    rank: int,
    beta: float,
    delta: float,
    outer_iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes, n_signals, _ = signals.shape
    scaled_signals = signals / np.sqrt(n_nodes * n_signals)
    mask = np.ones_like(adjacency, dtype=float)
    solver = DynamicGraphDecomposer(
        params_for_setting(
            rank=rank,
            beta=beta,
            delta=delta,
            outer_iterations=outer_iterations,
            seed=seed,
        )
    )
    result = solver.fit(adjacency, scaled_signals, mask=mask)
    return result.latent_graphs, result.temporal_signatures


def plot_results(
    true_graphs: np.ndarray,
    recovered_by_setting: dict[str, np.ndarray],
    matches: dict[str, dict[int, int]],
    setting_params: dict[str, tuple[float, float]],
    output_pdf: Path,
    output_png: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(recovered_by_setting)
    fig, axes = plt.subplots(n_rows, 4, figsize=(11.5, 3.0 * n_rows))
    cmap = LinearSegmentedColormap.from_list("white_to_blue", ["#ffffff", "#08306b"])
    if n_rows == 1:
        axes = axes[None, :]
    for row_idx, (setting_name, recovered) in enumerate(recovered_by_setting.items()):
        beta, delta = setting_params[setting_name]
        setting_label = fr"$\beta = {beta:g}$, $\delta = {delta:g}$"
        pairs = [
            (true_graphs[0], "True 2-community"),
            (recovered[matches[setting_name][0]], "Matched recovered"),
            (true_graphs[1], "True 4-community"),
            (recovered[matches[setting_name][1]], "Matched recovered"),
        ]
        for col_idx, (matrix, title) in enumerate(pairs):
            ax = axes[row_idx, col_idx]
            vmax = max(float(np.max(matrix)), 1e-12)
            ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=22)
            if col_idx == 0:
                ax.set_ylabel(setting_label, fontsize=22)
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def safe_slug(text: str) -> str:
    return (
        text.lower()
        .replace(",", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def plot_single_setting(
    true_graphs: np.ndarray,
    recovered: np.ndarray,
    match: dict[int, int],
    setting_name: str,
    beta: float,
    delta: float,
    output_pdf: Path,
    output_png: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("white_to_blue", ["#ffffff", "#08306b"])
    matched_indices = {match[0], match[1]}
    extra_indices = [idx for idx in range(recovered.shape[0]) if idx not in matched_indices]
    matrices = [
        (true_graphs[0], "True 2-community"),
        (recovered[match[0]], "Matched recovered"),
        (true_graphs[1], "True 4-community"),
        (recovered[match[1]], "Matched recovered"),
    ]
    matrices.extend((recovered[idx], f"Recovered component {idx + 1}") for idx in extra_indices)
    vmax = max(float(np.max(matrix)) for matrix, _ in matrices)
    vmax = max(vmax, 1e-12)
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.7 * len(matrices), 3.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    image = None
    for ax, (matrix, title) in zip(axes, matrices):
        image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=22)
        ax.set_xticks([])
        ax.set_yticks([])
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Edge weight", fontsize=22)
    cbar.ax.tick_params(labelsize=22)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=7)
    parser.add_argument("--high-beta", type=float, default=0.1)
    parser.add_argument("--low-beta", type=float, default=0.001)
    parser.add_argument("--high-delta", type=float, default=100.0)
    parser.add_argument("--low-delta", type=float, default=0.001)
    parser.add_argument("--p4-in", type=float, default=0.36)
    parser.add_argument("--p4-out", type=float, default=0.035)
    parser.add_argument(
        "--only-setting",
        choices=["high_high", "low_high", "high_low"],
        default=None,
        help="Run only one beta/delta scenario.",
    )
    parser.add_argument(
        "--binary-recovered",
        action="store_true",
        help="Deprecated compatibility flag; recovered matrices are kept continuous.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/csv/swdyn_dgd_component_structure_metrics.csv"),
    )
    parser.add_argument(
        "--correlation-output",
        type=Path,
        default=Path("results/csv/swdyn_dgd_component_structure_correlations.csv"),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("results/plots/swdyn_component_structure/swdyn_true_vs_recovered_components.pdf"),
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=Path("results/plots/swdyn_component_structure/swdyn_true_vs_recovered_components.png"),
    )
    parser.add_argument(
        "--separate-plot-dir",
        type=Path,
        default=Path("results/plots/swdyn_component_structure/separate"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adjacency, signals, true_graphs, labels_2, labels_4 = make_swdyn_with_labels(
        seed=args.data_seed,
        p4_in=args.p4_in,
        p4_out=args.p4_out,
    )
    settings = {
        "high beta, high delta": (args.high_beta, args.high_delta),
        "low beta, high delta": (args.low_beta, args.high_delta),
        "high beta, low delta": (args.high_beta, args.low_delta),
    }
    if args.only_setting == "high_high":
        settings = {"high beta, high delta": (args.high_beta, args.high_delta)}
    elif args.only_setting == "low_high":
        settings = {"low beta, high delta": (args.low_beta, args.high_delta)}
    elif args.only_setting == "high_low":
        settings = {"high beta, low delta": (args.high_beta, args.low_delta)}
    metric_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    recovered_by_setting: dict[str, np.ndarray] = {}
    matches: dict[str, dict[int, int]] = {}
    for setting_name, (beta, delta) in settings.items():
        print(f"running {setting_name}: beta={beta:g} delta={delta:g}", flush=True)
        recovered, temporal = run_setting(
            adjacency,
            signals,
            rank=args.rank,
            beta=beta,
            delta=delta,
            outer_iterations=args.outer_iterations,
            seed=args.seed,
        )
        recovered_by_setting[setting_name] = recovered
        corr = correlation_matrix(true_graphs, recovered)
        match = best_matching(corr)
        matches[setting_name] = match
        for true_idx in range(2):
            for recovered_idx in range(recovered.shape[0]):
                correlation_rows.append(
                    {
                        "setting": setting_name,
                        "beta": beta,
                        "delta": delta,
                        "true_component": true_idx + 1,
                        "true_structure": "2-community" if true_idx == 0 else "4-community",
                        "recovered_component": recovered_idx + 1,
                        "correlation": corr[true_idx, recovered_idx],
                        "matched": int(match[true_idx] == recovered_idx),
                    }
                )
        for true_idx, labels in [(0, labels_2), (1, labels_4)]:
            recovered_idx = match[true_idx]
            within, between, ratio = community_stats(recovered[recovered_idx], labels)
            metric_rows.append(
                {
                    "setting": setting_name,
                    "beta": beta,
                    "delta": delta,
                    "true_component": true_idx + 1,
                    "true_structure": "2-community" if true_idx == 0 else "4-community",
                    "matched_recovered_component": recovered_idx + 1,
                    "matched_correlation": corr[true_idx, recovered_idx],
                    "within_mean": within,
                    "between_mean": between,
                    "within_between_ratio": ratio,
                }
            )
    write_csv(args.metrics_output, metric_rows)
    write_csv(args.correlation_output, correlation_rows)
    plot_results(true_graphs, recovered_by_setting, matches, settings, args.plot_output, args.png_output)
    for setting_name, recovered in recovered_by_setting.items():
        slug = safe_slug(setting_name)
        plot_single_setting(
            true_graphs,
            recovered,
            matches[setting_name],
            setting_name,
            settings[setting_name][0],
            settings[setting_name][1],
            args.separate_plot_dir / f"swdyn_{slug}_adjacency_matrices.pdf",
            args.separate_plot_dir / f"swdyn_{slug}_adjacency_matrices.png",
        )
    print(f"wrote {args.metrics_output}")
    print(f"wrote {args.correlation_output}")
    print(f"wrote {args.plot_output}")
    print(f"wrote {args.png_output}")
    for setting_name in recovered_by_setting:
        slug = safe_slug(setting_name)
        print(f"wrote {args.separate_plot_dir / f'swdyn_{slug}_adjacency_matrices.pdf'}")


if __name__ == "__main__":
    main()
