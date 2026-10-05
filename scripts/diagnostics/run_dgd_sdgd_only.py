"""Run only DGD and no-signal DGD for a MATLAB adjacency-signal dataset."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

from dynamic_graph_decomposition import DecompositionParams, DynamicGraphDecomposer, fit_proposed_without_signals
from seasurf_paper_experiments import matlab_mask_type2
from swdyn_paper_experiments import f1_score_hidden, hidden_mask, relative_error_squared


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(path)
    adjacency = (np.asarray(data["A"], dtype=float) > 0.0).astype(float)
    signals = np.asarray(data["X"], dtype=float)
    return adjacency, signals


def make_params(args: argparse.Namespace, eta: float) -> DecompositionParams:
    return DecompositionParams(
        rank=args.rank,
        alpha=args.alpha,
        gamma=args.gamma,
        beta=args.beta,
        delta=args.delta,
        delta_A=args.delta,
        mu=args.mu,
        rho=args.rho,
        eta_graph=eta,
        zeta=1e-4,
        outer_iterations=args.outer_iterations,
        inner_iterations_A=5,
        inner_iterations_C=5,
        random_state=0,
    )


def reconstruct_dgd(
    adjacency: np.ndarray,
    signals: np.ndarray,
    mask: np.ndarray,
    params: DecompositionParams,
) -> tuple[np.ndarray, float]:
    n_nodes, n_signals, _ = signals.shape
    scaled_signals = signals / np.sqrt(n_nodes * n_signals)
    solver = DynamicGraphDecomposer(params)
    start = time.perf_counter()
    result = solver.fit(adjacency, scaled_signals, mask=mask)
    elapsed = time.perf_counter() - start
    return solver.reconstruct(result.latent_graphs, result.temporal_signatures), elapsed


def reconstruct_sdgd(
    adjacency: np.ndarray,
    mask: np.ndarray,
    params: DecompositionParams,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    result = fit_proposed_without_signals(adjacency, mask, params)
    elapsed = time.perf_counter() - start
    return DynamicGraphDecomposer.reconstruct(
        result.latent_graphs,
        result.temporal_signatures,
    ), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--observed", type=int, nargs="+", default=[10, 30, 50, 70, 90])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outer-iterations", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--delta-a", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=1e3)
    parser.add_argument("--etas", type=float, nargs="+", default=[1e3])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adjacency, signals = load_data(args.data)
    rows: list[dict[str, object]] = []
    print(
        f"dataset={args.dataset_name} A={adjacency.shape} X={signals.shape} "
        f"R={args.rank} etas={args.etas}",
        flush=True,
    )
    for observed_percent in args.observed:
        run_seed = args.seed + observed_percent
        mask = matlab_mask_type2(adjacency, observed_percent, seed=run_seed)
        hmask = hidden_mask(mask)
        for eta in args.etas:
            params = make_params(args, eta)
            for method, factory in [
                ("DGD", lambda: reconstruct_dgd(adjacency, signals, mask, params)),
                ("SDGD", lambda: reconstruct_sdgd(adjacency, mask, params)),
            ]:
                reconstruction, elapsed = factory()
                row = {
                    "dataset": args.dataset_name,
                    "data_source": str(args.data),
                    "adjacency_shape": "x".join(map(str, adjacency.shape)),
                    "signal_shape": "x".join(map(str, signals.shape)),
                    "observed_percent": observed_percent,
                    "method": method,
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "gamma": args.gamma,
                    "beta": args.beta,
                    "delta": args.delta if method == "DGD" else 0.0,
                    "delta_a": args.delta if method == "DGD" else "",
                    "mu": args.mu,
                    "rho": args.rho,
                    "eta": eta,
                    "relative_error": relative_error_squared(adjacency, reconstruction, hmask),
                    "f1": f1_score_hidden(adjacency, reconstruction, hmask),
                    "elapsed_seconds": elapsed,
                }
                rows.append(row)
                print(
                    f"obs={observed_percent} eta={eta:g} {method} "
                    f"RE={row['relative_error']:.4f} F1={row['f1']:.4f}",
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
