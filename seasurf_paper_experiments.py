"""Run real SeaSurf mask-sweep experiments with manuscript-style settings.

This runner is intentionally separate from the earlier debug CSVs. It uses
the paper/MATLAB conventions for SeaSurf:

- real data file ``surface_8_data.mat``: A=(100, 100, 8), X=(100, 108, 8)
- mask type 2: observe p% of edges and p% of non-edges per time slice
- hidden metrics over ``1 - Mask``
- CPD, BTD, SICA, SGL, UNC see only the masked observation or signals, not the
  hidden test mask

The MATLAB file ``experiments/journal/exp4/surface_results_mask2.mat`` is
misleadingly named: its saved arrays are A=(40, 40, 50), X=(40, 1000, 50), so
it is not the real SeaSurf data used by this runner.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio

from dynamic_graph_decomposition import (
    DecompositionParams,
    DynamicGraphDecomposer,
    fit_btd_ll1_als,
    fit_cpd_als,
    fit_proposed_without_signals,
    fit_sica_baseline,
    fit_smooth_topology_baseline,
    fit_unconstrained_observation_als,
    signal_distance_matrices,
)
from swdyn_paper_experiments import f1_score_hidden, hidden_mask, relative_error_squared


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = REPO_ROOT / "data/processed/surface_8_data.mat"
METHOD_ORDER = ["DGD", "NSDGD", "SICA", "BTD", "CPD", "SGL", "UNC"]
DEFAULT_DATASET_NAME = "SeaSurfReal"


def matlab_mask_type2(
    adjacency: np.ndarray,
    observed_percent: int,
    *,
    seed: int,
) -> np.ndarray:
    """Replicate MATLAB ``mask_equally(..., op=2)``."""

    adjacency = np.asarray(adjacency, dtype=float)
    n_nodes, _, n_times = adjacency.shape
    rows, cols = np.triu_indices(n_nodes, k=1)
    rng = np.random.default_rng(seed)
    mask = np.zeros_like(adjacency, dtype=float)
    for t in range(n_times):
        upper_values = adjacency[:, :, t][rows, cols]
        one_idx = np.flatnonzero(upper_values == 1.0)
        zero_idx = np.flatnonzero(upper_values == 0.0)
        n_ones = int(round(len(one_idx) * observed_percent / 100.0))
        n_zeros = int(round(len(zero_idx) * observed_percent / 100.0))
        selected = []
        if n_ones:
            selected.append(rng.choice(one_idx, n_ones, replace=False))
        if n_zeros:
            selected.append(rng.choice(zero_idx, n_zeros, replace=False))
        selected_idx = np.concatenate(selected) if selected else np.array([], dtype=int)
        upper_mask = np.zeros_like(upper_values, dtype=float)
        upper_mask[selected_idx] = 1.0
        slice_mask = np.zeros((n_nodes, n_nodes), dtype=float)
        slice_mask[rows, cols] = upper_mask
        mask[:, :, t] = slice_mask + slice_mask.T
    return mask


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


def load_matlab_adjacency_signals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(path)
    adjacency = (np.asarray(data["A"], dtype=float) > 0.0).astype(float)
    signals = np.asarray(data["X"], dtype=float)
    return adjacency, signals


def compute_matlab_signal_distances(signals: np.ndarray) -> np.ndarray:
    """Replicate MATLAB ``compute_W(X)`` with ``smooth_topoID`` scaling.

    MATLAB computes squared row distances per time slice and then divides by
    ``N*M`` inside ``smooth_topoID``.
    """

    n_nodes, n_signals, _ = signals.shape
    return signal_distance_matrices(signals) / float(n_nodes * n_signals)


def paper_dgd_params(
    rank: int,
    *,
    alpha: float,
    gamma: float,
    beta: float,
    delta: float,
    delta_a: float | None,
    mu: float,
    rho: float,
    eta_graph: float,
    outer_iterations: int,
) -> DecompositionParams:
    return DecompositionParams(
        rank=rank,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        delta=delta,
        delta_A=delta,
        mu=mu,
        rho=rho,
        eta_graph=eta_graph,
        zeta=1e-4,
        outer_iterations=outer_iterations,
        inner_iterations_A=5,
        inner_iterations_C=5,
        random_state=0,
    )


def run_methods(
    adjacency: np.ndarray,
    signals: np.ndarray,
    mask: np.ndarray,
    *,
    dataset_name: str,
    data_source: str,
    rank: int,
    outer_iterations: int,
    cpd_iterations: int,
    btd_iterations: int,
    seed: int,
    dgd_gamma: float,
    dgd_alpha: float,
    dgd_beta: float,
    dgd_delta: float,
    dgd_delta_a: float | None,
    dgd_mu: float,
    dgd_rho: float,
    dgd_eta: float,
    sgl_a: float,
    sgl_b: float,
) -> list[dict[str, object]]:
    n_nodes, n_signals, n_times = signals.shape
    hmask = hidden_mask(mask)
    observed = adjacency * mask
    scaled_signals = signals / np.sqrt(n_nodes * n_signals)
    matlab_distances = compute_matlab_signal_distances(signals)
    cpd_rank = matlab_parameter_matched_cpd_rank(n_nodes, n_times, rank)
    btd_blocks = matlab_parameter_matched_btd_blocks(n_nodes, n_times, rank)
    sica_rank = min(rank, n_times)
    unc_rank = rank

    methods = [
        (
            "DGD",
            lambda: _run_dgd(
                adjacency,
                scaled_signals,
                mask,
                paper_dgd_params(
                    rank,
                    alpha=dgd_alpha,
                    gamma=dgd_gamma,
                    beta=dgd_beta,
                    delta=dgd_delta,
                    delta_a=dgd_delta_a,
                    mu=dgd_mu,
                    rho=dgd_rho,
                    eta_graph=dgd_eta,
                    outer_iterations=outer_iterations,
                ),
            ),
            rank,
        ),
        (
            "NSDGD",
            lambda: _run_nsdgd(
                adjacency,
                mask,
                paper_dgd_params(
                    rank,
                    alpha=dgd_alpha,
                    gamma=dgd_gamma,
                    beta=dgd_beta,
                    delta=dgd_delta,
                    delta_a=dgd_delta_a,
                    mu=dgd_mu,
                    rho=dgd_rho,
                    eta_graph=dgd_eta,
                    outer_iterations=outer_iterations,
                ),
            ),
            rank,
        ),
        (
            "SICA",
            lambda: fit_sica_baseline(
                observed,
                rank=sica_rank,
                mask=None,
                max_iterations=500,
                random_state=seed,
                backend="sklearn",
            ).reconstruction,
            sica_rank,
        ),
        (
            "BTD",
            lambda: fit_btd_ll1_als(
                observed,
                n_blocks=btd_blocks,
                block_rank=btd_blocks,
                mask=None,
                max_iterations=btd_iterations,
                random_state=seed,
            ).reconstruction,
            btd_blocks,
        ),
        (
            "CPD",
            lambda: fit_cpd_als(
                observed,
                rank=cpd_rank,
                mask=None,
                max_iterations=cpd_iterations,
                random_state=seed,
                backend="tensorly",
            ).reconstruction,
            cpd_rank,
        ),
        (
            "SGL",
            lambda: fit_smooth_topology_baseline(
                matlab_distances,
                a=sgl_a,
                b=sgl_b,
                max_iterations=1000,
            ).reconstruction,
            rank,
        ),
        (
            "UNC",
            lambda: fit_unconstrained_observation_als(
                observed,
                rank=unc_rank,
                mask=None,
                max_iterations=30,
                random_state=seed,
            ).reconstruction,
            unc_rank,
        ),
    ]

    rows = []
    for method, factory, method_rank in methods:
        start = time.perf_counter()
        reconstruction = factory()
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "dataset": dataset_name,
                "source": "python",
                "data_source": data_source,
                "adjacency_shape": "x".join(map(str, adjacency.shape)),
                "signal_shape": "x".join(map(str, signals.shape)),
                "mask_type": 2,
                "proposed_rank": rank,
                "dgd_alpha": dgd_alpha,
                "dgd_delta": dgd_delta,
                "dgd_delta_a": dgd_delta,
                "method_rank": method_rank,
                "method": method,
                "relative_error": relative_error_squared(adjacency, reconstruction, hmask),
                "f1": f1_score_hidden(adjacency, reconstruction, hmask),
                "elapsed_seconds": elapsed,
            }
        )
        print(
            method,
            f"rank={method_rank}",
            f"RE={rows[-1]['relative_error']:.4f}",
            f"F1={rows[-1]['f1']:.4f}",
            flush=True,
        )
    return rows


def _run_dgd(
    adjacency: np.ndarray,
    signals: np.ndarray,
    mask: np.ndarray,
    params: DecompositionParams,
) -> np.ndarray:
    solver = DynamicGraphDecomposer(params)
    result = solver.fit(adjacency, signals, mask=mask)
    return solver.reconstruct(result.latent_graphs, result.temporal_signatures)


def _run_nsdgd(
    adjacency: np.ndarray,
    mask: np.ndarray,
    params: DecompositionParams,
) -> np.ndarray:
    result = fit_proposed_without_signals(adjacency, mask, params)
    return DynamicGraphDecomposer.reconstruct(
        result.latent_graphs,
        result.temporal_signatures,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["observed_percent"]), str(row["method"]))].append(row)

    summary_rows = []
    for observed_percent in sorted({key[0] for key in groups}):
        for method in METHOD_ORDER:
            group = groups.get((observed_percent, method), [])
            if not group:
                continue
            relative_errors = np.array([float(row["relative_error"]) for row in group])
            f1_values = np.array([float(row["f1"]) for row in group])
            elapsed = np.array([float(row["elapsed_seconds"]) for row in group])
            summary_rows.append(
                {
                    "dataset": group[0]["dataset"],
                    "source": "python",
                    "data_source": group[0]["data_source"],
                    "adjacency_shape": group[0]["adjacency_shape"],
                    "signal_shape": group[0]["signal_shape"],
                    "mask_type": 2,
                    "proposed_rank": group[0]["proposed_rank"],
                    "dgd_alpha": group[0].get("dgd_alpha", ""),
                    "dgd_delta": group[0].get("dgd_delta", ""),
                    "dgd_delta_a": group[0].get("dgd_delta_a", ""),
                    "method_rank": group[0]["method_rank"],
                    "observed_percent": observed_percent,
                    "method": method,
                    "n_runs": len(group),
                    "relative_error_mean": float(np.mean(relative_errors)),
                    "relative_error_std": float(np.std(relative_errors, ddof=1))
                    if len(group) > 1
                    else 0.0,
                    "f1_mean": float(np.mean(f1_values)),
                    "f1_std": float(np.std(f1_values, ddof=1)) if len(group) > 1 else 0.0,
                    "elapsed_seconds_mean": float(np.mean(elapsed)),
                }
            )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output", type=Path, default=Path("results/csv/exp8_seasurf_python_mask2.csv"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/csv/exp8_seasurf_python_mask2_summary.csv"),
    )
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--observed", type=int, nargs="+", default=[90, 80, 70, 60, 50, 40, 30, 20, 10])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--outer-iterations", type=int, default=20)
    parser.add_argument("--dgd-alpha", type=float, default=1.0)
    parser.add_argument("--dgd-gamma", type=float, default=1.0)
    parser.add_argument("--dgd-beta", type=float, default=1e-3)
    parser.add_argument(
        "--dgd-delta",
        type=float,
        default=1.0,
        help=(
            "Python-equivalent signal smoothness weight for scaled signals. "
            "The MATLAB surface parameter is delta=22 before this scaling."
        ),
    )
    parser.add_argument("--dgd-mu", type=float, default=1e-4)
    parser.add_argument(
        "--dgd-delta-a",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dgd-rho", type=float, default=10.0)
    parser.add_argument("--dgd-eta", type=float, default=0.0)
    parser.add_argument("--sgl-a", type=float, default=1e-6)
    parser.add_argument("--sgl-b", type=float, default=1e-1)
    parser.add_argument("--cpd-iterations", type=int, default=10)
    parser.add_argument("--btd-iterations", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adjacency, signals = load_matlab_adjacency_signals(args.data)
    print(
        f"dataset={args.dataset_name} data={args.data} "
        f"A={adjacency.shape} X={signals.shape} R={args.rank}",
        flush=True,
    )
    rows = []
    for observed_percent in args.observed:
        for repeat in range(args.repeats):
            run_seed = args.seed + 1000 * repeat + observed_percent
            print(f"observed={observed_percent} repeat={repeat} seed={run_seed}", flush=True)
            mask = matlab_mask_type2(
                adjacency,
                observed_percent,
                seed=run_seed,
            )
            for row in run_methods(
                adjacency,
                signals,
                mask,
                dataset_name=args.dataset_name,
                data_source=str(args.data),
                rank=args.rank,
                outer_iterations=args.outer_iterations,
                cpd_iterations=args.cpd_iterations,
                btd_iterations=args.btd_iterations,
                seed=run_seed,
                dgd_alpha=args.dgd_alpha,
                dgd_gamma=args.dgd_gamma,
                dgd_beta=args.dgd_beta,
                dgd_delta=args.dgd_delta,
                dgd_delta_a=args.dgd_delta_a,
                dgd_mu=args.dgd_mu,
                dgd_rho=args.dgd_rho,
                dgd_eta=args.dgd_eta,
                sgl_a=args.sgl_a,
                sgl_b=args.sgl_b,
            ):
                row["observed_percent"] = observed_percent
                row["repeat"] = repeat
                row["seed"] = run_seed
                rows.append(row)
    write_csv(args.output, rows)
    write_csv(args.summary_output, summarize(rows))
    print(f"wrote {args.output}")
    print(f"wrote {args.summary_output}")


if __name__ == "__main__":
    main()
