"""Run SwDyn experiments from the DGD paper in Python.

The paper's synthetic setting uses N=40 nodes, Q=1000 graph-signal samples,
and T=50 time instants.  The dynamic topology transitions from a 2-community
SBM graph to a 4-community SBM graph with two linear temporal signatures.

This script writes CSV files under ``results/csv``.  Use ``--mode verify`` for
a fast sanity run and ``--mode full`` for the complete SwDyn sweeps.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dynamic_graph_decomposition import (
    DecompositionParams,
    DynamicGraphDecomposer,
    fit_btd_ll1_als,
    fit_cpd_als,
    fit_proposed_without_signals,
    fit_sica_baseline,
    fit_smooth_topology_baseline,
    fit_unconstrained_observation_als,
    generate_graph_smooth_signals,
    graph_laplacian,
    signal_distance_matrices,
)


@dataclass
class SwDynData:
    adjacency: np.ndarray
    signals: np.ndarray
    true_graphs: np.ndarray
    true_temporal_signatures: np.ndarray


def make_swdyn(
    *,
    n_nodes: int = 40,
    n_signals: int = 1000,
    n_times: int = 50,
    seed: int = 7,
) -> SwDynData:
    rng = np.random.default_rng(seed)
    graph_2 = stochastic_block_model(
        rng,
        n_nodes=n_nodes,
        n_communities=2,
        p_in=0.28,
        p_out=0.04,
    )
    graph_4 = stochastic_block_model(
        rng,
        n_nodes=n_nodes,
        n_communities=4,
        p_in=0.36,
        p_out=0.035,
        shuffle_labels=True,
    )
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

    return SwDynData(
        adjacency=adjacency,
        signals=signals,
        true_graphs=true_graphs,
        true_temporal_signatures=temporal,
    )


def stochastic_block_model(
    rng: np.random.Generator,
    *,
    n_nodes: int,
    n_communities: int,
    p_in: float,
    p_out: float,
    shuffle_labels: bool = False,
) -> np.ndarray:
    labels = np.arange(n_nodes) * n_communities // n_nodes
    if shuffle_labels:
        labels = labels.copy()
        rng.shuffle(labels)
    probabilities = np.where(labels[:, None] == labels[None, :], p_in, p_out)
    upper = rng.random((n_nodes, n_nodes)) < probabilities
    graph = np.triu(upper.astype(float), k=1)
    return graph + graph.T


def symmetric_observation_mask(
    shape: tuple[int, int, int],
    observed_fraction: float,
    *,
    seed: int,
) -> np.ndarray:
    n_nodes, _, n_times = shape
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=float)
    for t in range(n_times):
        upper = rng.random((n_nodes, n_nodes)) < observed_fraction
        upper = np.triu(upper.astype(float), k=1)
        mask[:, :, t] = upper + upper.T
    return mask


def hidden_mask(observed_mask: np.ndarray) -> np.ndarray:
    mask = 1.0 - observed_mask
    for t in range(mask.shape[2]):
        np.fill_diagonal(mask[:, :, t], 0.0)
    return mask


def relative_error_squared(
    truth: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray,
) -> float:
    denominator = np.sum((mask * truth) ** 2)
    if denominator == 0.0:
        return float("nan")
    return float(np.sum((mask * (estimate - truth)) ** 2) / denominator)


def f1_score_hidden(
    truth: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray,
) -> float:
    truth_binary = two_means_binary(mask * truth)
    estimate_binary = two_means_binary(mask * estimate)
    true_pos = np.sum((truth_binary == 1) & (estimate_binary == 1))
    predicted_pos = np.sum(estimate_binary == 1)
    relevant = np.sum(truth_binary == 1)
    if predicted_pos == 0 or relevant == 0:
        return 0.0
    precision = true_pos / predicted_pos
    recall = true_pos / relevant
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def two_means_binary(values: np.ndarray, n_iterations: int = 50) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=int)
    if np.allclose(values, values[0]):
        return (values > 0.0).astype(int)
    centers = np.array([np.min(values), np.max(values)], dtype=float)
    labels = np.zeros(values.shape, dtype=int)
    for _ in range(n_iterations):
        distances = np.abs(values[:, None] - centers[None, :])
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for k in range(2):
            if np.any(new_labels == k):
                new_centers[k] = np.mean(values[new_labels == k])
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            break
        labels = new_labels
        centers = new_centers
    high_label = int(np.argmax(centers))
    return (labels == high_label).astype(int)


def proposed_params(rank: int, *, outer_iterations: int = 20) -> DecompositionParams:
    return DecompositionParams(
        rank=rank,
        gamma=1e-3,
        delta=1e-3,
        beta=1e-3,
        mu=1e-2,
        rho=1e-3,
        zeta=1e-4,
        outer_iterations=outer_iterations,
        inner_iterations_A=5,
        inner_iterations_C=5,
        random_state=0,
    )


def run_dgd(
    data: SwDynData,
    observed_fraction: float,
    rank: int,
    *,
    outer_iterations: int,
    seed: int,
) -> dict[str, object]:
    mask = symmetric_observation_mask(data.adjacency.shape, observed_fraction, seed=seed)
    hmask = hidden_mask(mask)
    n_nodes, n_signals, _ = data.signals.shape
    scaled_signals = data.signals / np.sqrt(n_nodes * n_signals)
    params = proposed_params(rank, outer_iterations=outer_iterations)
    solver = DynamicGraphDecomposer(params)
    start = time.perf_counter()
    result = solver.fit(data.adjacency, scaled_signals, mask=mask)
    elapsed = time.perf_counter() - start
    reconstruction = solver.reconstruct(result.latent_graphs, result.temporal_signatures)
    return {
        "result": result,
        "reconstruction": reconstruction,
        "mask": mask,
        "hidden_mask": hmask,
        "elapsed_seconds": elapsed,
        "relative_error": relative_error_squared(data.adjacency, reconstruction, hmask),
        "f1": f1_score_hidden(data.adjacency, reconstruction, hmask),
    }


def run_convergence(data: SwDynData, output_dir: Path, *, mode: str) -> None:
    outer_iterations = 20
    run = run_dgd(
        data,
        observed_fraction=0.90,
        rank=10,
        outer_iterations=outer_iterations,
        seed=100,
    )
    rows = []
    total_elapsed = float(run["elapsed_seconds"])
    history = run["result"].objective_history
    initial = history[0] if history else 1.0
    for idx, objective in enumerate(history, start=1):
        rows.append(
            {
                "dataset": "SwDyn",
                "iteration": idx,
                "rank": 10,
                "observed_percent": 90,
                "normalized_objective": objective / initial,
                "objective": objective,
                "final_relative_error": run["relative_error"],
                "final_f1": run["f1"],
                "elapsed_seconds_estimate": total_elapsed * idx / max(len(history), 1),
            }
        )
    write_csv(output_dir / "exp1_swdyn_convergence.csv", rows)
    print(
        "convergence:",
        f"RE={run['relative_error']:.4f}",
        f"F1={run['f1']:.4f}",
        f"seconds={run['elapsed_seconds']:.2f}",
        flush=True,
    )


def run_rank_sweep(data: SwDynData, output_dir: Path, *, mode: str) -> None:
    ranks = list(range(1, 11)) if mode == "full" else [1, 2, 10]
    rows = []
    for rank in ranks:
        run = run_dgd(
            data,
            observed_fraction=0.90,
            rank=rank,
            outer_iterations=20 if mode == "full" else 10,
            seed=200,
        )
        rows.append(
            {
                "dataset": "SwDyn",
                "rank": rank,
                "observed_percent": 90,
                "relative_error": run["relative_error"],
                "f1": run["f1"],
                "elapsed_seconds": run["elapsed_seconds"],
            }
        )
        print(
            "rank sweep:",
            f"R={rank}",
            f"RE={run['relative_error']:.4f}",
            f"F1={run['f1']:.4f}",
            flush=True,
        )
    write_csv(output_dir / "exp2_swdyn_rank_sweep.csv", rows)


def run_observation_sweep(data: SwDynData, output_dir: Path, *, mode: str) -> None:
    observed_percents = [20, 40, 60, 80, 90] if mode == "full" else [20, 60, 90]
    rows = []
    for observed_percent in observed_percents:
        run = run_dgd(
            data,
            observed_fraction=observed_percent / 100.0,
            rank=10,
            outer_iterations=20 if mode == "full" else 10,
            seed=300 + observed_percent,
        )
        rows.append(
            {
                "dataset": "SwDyn",
                "rank": 10,
                "observed_percent": observed_percent,
                "relative_error": run["relative_error"],
                "f1": run["f1"],
                "elapsed_seconds": run["elapsed_seconds"],
            }
        )
        print(
            "observation sweep:",
            f"obs={observed_percent}",
            f"RE={run['relative_error']:.4f}",
            f"F1={run['f1']:.4f}",
            flush=True,
        )
    write_csv(output_dir / "exp3_swdyn_observation_sweep.csv", rows)


def run_baseline_check(data: SwDynData, output_dir: Path, *, mode: str) -> None:
    observed_percents = [20, 40, 60, 80] if mode == "full" else [90]
    rows = []
    n_nodes, n_signals, n_times = data.signals.shape
    proposed_rank = 10
    cpd_rank = matlab_parameter_matched_cpd_rank(n_nodes, n_times, proposed_rank)
    btd_blocks = matlab_parameter_matched_btd_blocks(n_nodes, n_times, proposed_rank)
    btd_block_rank = btd_blocks
    sica_rank = min(4, n_times)
    scaled_signals = data.signals / np.sqrt(n_nodes * n_signals)
    distances = signal_distance_matrices(scaled_signals)
    for observed_percent in observed_percents:
        mask = symmetric_observation_mask(
            data.adjacency.shape,
            observed_percent / 100.0,
            seed=400 + observed_percent,
        )
        hmask = hidden_mask(mask)
        methods = [
            (
                "DGD",
                lambda: _wrap_dgd_result(
                    run_dgd(
                        data,
                        observed_fraction=observed_percent / 100.0,
                        rank=proposed_rank,
                        outer_iterations=10 if mode == "verify" else 20,
                        seed=400 + observed_percent,
                    )
                ),
            ),
            (
                "NSDGD",
                lambda: _reconstruct_proposed_without_signals(
                    data.adjacency,
                    mask,
                    proposed_params(proposed_rank, outer_iterations=10 if mode == "verify" else 20),
                ),
            ),
            (
                "UNC",
                lambda: fit_unconstrained_observation_als(
                    data.adjacency,
                    rank=proposed_rank,
                    mask=mask,
                    max_iterations=10 if mode == "verify" else 30,
                    random_state=0,
                ).reconstruction,
            ),
            (
                "CPD",
                lambda: fit_cpd_als(
                    data.adjacency,
                    rank=min(cpd_rank, 30) if mode == "verify" else cpd_rank,
                    mask=mask,
                    max_iterations=10 if mode == "verify" else 50,
                    random_state=0,
                ).reconstruction,
            ),
            (
                "BTD",
                lambda: fit_btd_ll1_als(
                    data.adjacency,
                    n_blocks=min(btd_blocks, 4) if mode == "verify" else btd_blocks,
                    block_rank=min(btd_block_rank, 4) if mode == "verify" else btd_block_rank,
                    mask=mask,
                    max_iterations=5 if mode == "verify" else 25,
                    random_state=0,
                ).reconstruction,
            ),
            (
                "SICA",
                lambda: fit_sica_baseline(
                    data.adjacency,
                    rank=sica_rank,
                    mask=mask,
                    max_iterations=50 if mode == "verify" else 300,
                    random_state=0,
                ).reconstruction,
            ),
            (
                "SGL",
                lambda: fit_smooth_topology_baseline(
                    distances,
                    max_iterations=50 if mode == "verify" else 300,
                ).reconstruction,
            ),
        ]
        for method, factory in methods:
            start = time.perf_counter()
            reconstruction = factory()
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "dataset": "SwDyn",
                    "method": method,
                    "observed_percent": observed_percent,
                    "relative_error": relative_error_squared(
                        data.adjacency, reconstruction, hmask
                    ),
                    "f1": f1_score_hidden(data.adjacency, reconstruction, hmask),
                    "elapsed_seconds": elapsed,
                }
            )
            print(
                "baseline:",
                method,
                f"obs={observed_percent}",
                f"RE={rows[-1]['relative_error']:.4f}",
                f"F1={rows[-1]['f1']:.4f}",
                flush=True,
            )
    write_csv(output_dir / "exp6_swdyn_baseline_check.csv", rows)


def matlab_parameter_matched_cpd_rank(
    n_nodes: int,
    n_times: int,
    proposed_rank: int,
) -> int:
    tgd_parameters = (n_nodes * n_nodes + n_times) * proposed_rank
    return max(1, int(round(tgd_parameters / (2 * n_nodes + n_times + 1))))


def matlab_parameter_matched_btd_blocks(
    n_nodes: int,
    n_times: int,
    proposed_rank: int,
) -> int:
    tgd_parameters = (n_nodes * n_nodes + n_times) * proposed_rank
    value = (-n_times + np.sqrt(n_times * n_times + 8 * n_nodes * tgd_parameters)) / (
        4 * n_nodes
    )
    return max(1, int(round(value)))


def _wrap_dgd_result(run: dict[str, object]) -> np.ndarray:
    return run["reconstruction"]


def _reconstruct_proposed_without_signals(
    adjacency: np.ndarray,
    mask: np.ndarray,
    params: DecompositionParams,
) -> np.ndarray:
    result = fit_proposed_without_signals(adjacency, mask, params)
    return DynamicGraphDecomposer.reconstruct(
        result.latent_graphs,
        result.temporal_signatures,
    )


def write_dataset_summary(data: SwDynData, output_dir: Path) -> None:
    adjacency = data.adjacency
    n_nodes, _, n_times = adjacency.shape
    n_signals = data.signals.shape[1]
    edge_counts = [np.sum(two_means_binary(adjacency[:, :, t].reshape(-1))) / 2 for t in range(n_times)]
    temporal_variations = []
    for t in range(n_times - 1):
        diff = adjacency[:, :, t + 1] - adjacency[:, :, t]
        denom = max(np.linalg.norm(adjacency[:, :, t], "fro") ** 2, 1e-12)
        temporal_variations.append(np.linalg.norm(diff, "fro") ** 2 / denom)
    scaled_signals = data.signals / np.sqrt(n_nodes * n_signals)
    distances = signal_distance_matrices(scaled_signals)
    qv = 0.0
    for t in range(n_times):
        qv += 0.5 * np.sum(adjacency[:, :, t] * distances[t])
    rows = [
        {
            "dataset": "SwDyn",
            "N": n_nodes,
            "average_edges": float(np.mean(edge_counts)),
            "Q": n_signals,
            "T": n_times,
            "temporal_variation_mean": float(np.mean(temporal_variations)),
            "temporal_variation_std": float(np.std(temporal_variations)),
            "quadratic_variation": float(qv),
        }
    ]
    write_csv(output_dir / "exp0_swdyn_dataset_summary.csv", rows)
    print("dataset summary:", rows[0], flush=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["verify", "full"],
        default="verify",
        help="verify runs a smaller SwDyn check; full runs the complete SwDyn grids.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "summary", "convergence", "rank", "observed", "baselines"],
        default="all",
    )
    parser.add_argument("--output-dir", default="results/csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    print("generating SwDyn N=40, Q=1000, T=50", flush=True)
    data = make_swdyn()
    if args.only in {"all", "summary"}:
        write_dataset_summary(data, output_dir)
    if args.only in {"all", "convergence"}:
        run_convergence(data, output_dir, mode=args.mode)
    if args.only in {"all", "rank"}:
        run_rank_sweep(data, output_dir, mode=args.mode)
    if args.only in {"all", "observed"}:
        run_observation_sweep(data, output_dir, mode=args.mode)
    if args.only in {"all", "baselines"}:
        run_baseline_check(data, output_dir, mode=args.mode)


if __name__ == "__main__":
    main()
