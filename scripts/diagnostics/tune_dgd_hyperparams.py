"""Grid-search DGD hyperparameters on a MATLAB adjacency-signal dataset."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

from dynamic_graph_decomposition import DecompositionParams, DynamicGraphDecomposer
from seasurf_paper_experiments import matlab_mask_type2
from swdyn_paper_experiments import f1_score_hidden, hidden_mask, relative_error_squared


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(path)
    adjacency = (np.asarray(data["A"], dtype=float) > 0.0).astype(float)
    signals = np.asarray(data["X"], dtype=float)
    return adjacency, signals


def run_one(
    adjacency: np.ndarray,
    signals: np.ndarray,
    *,
    observed_percent: int,
    rank: int,
    alpha: float,
    gamma: float,
    beta: float,
    delta: float,
    delta_a: float | None,
    mu: float,
    rho: float,
    eta_graph: float,
    seed: int,
    outer_iterations: int,
) -> dict[str, object]:
    n_nodes, n_signals, _ = signals.shape
    mask = matlab_mask_type2(adjacency, observed_percent, seed=seed)
    scaled_signals = signals / np.sqrt(n_nodes * n_signals)
    params = DecompositionParams(
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
    solver = DynamicGraphDecomposer(params)
    start = time.perf_counter()
    result = solver.fit(adjacency, scaled_signals, mask=mask)
    elapsed = time.perf_counter() - start
    reconstruction = solver.reconstruct(result.latent_graphs, result.temporal_signatures)
    hmask = hidden_mask(mask)
    return {
        "observed_percent": observed_percent,
        "rank": rank,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "delta": delta,
        "delta_a": delta,
        "mu": mu,
        "rho": rho,
        "eta_graph": eta_graph,
        "seed": seed,
        "relative_error": relative_error_squared(adjacency, reconstruction, hmask),
        "f1": f1_score_hidden(adjacency, reconstruction, hmask),
        "elapsed_seconds": elapsed,
        "final_objective": result.objective_history[-1] if result.objective_history else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--observed", type=int, nargs="+", default=[10, 90])
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--gammas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--betas", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.1, 1.0, 2.0, 5.0])
    parser.add_argument("--delta-as", type=float, nargs="+", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mus", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--rhos", type=float, nargs="+", default=[10.0])
    parser.add_argument("--eta-graphs", type=float, nargs="+", default=[0.0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outer-iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adjacency, signals = load_data(args.data)
    rows: list[dict[str, object]] = []
    print(
        f"dataset={args.dataset_name} A={adjacency.shape} X={signals.shape} R={args.rank}",
        flush=True,
    )
    for observed_percent in args.observed:
        for alpha in args.alphas:
            for gamma in args.gammas:
                for beta in args.betas:
                    for delta in args.deltas:
                        delta_a = delta
                        for mu in args.mus:
                            for rho in args.rhos:
                                for eta_graph in args.eta_graphs:
                                    seed = args.seed + observed_percent
                                    row = run_one(
                                        adjacency,
                                        signals,
                                        observed_percent=observed_percent,
                                        rank=args.rank,
                                        alpha=alpha,
                                        gamma=gamma,
                                        beta=beta,
                                        delta=delta,
                                        delta_a=delta_a,
                                        mu=mu,
                                        rho=rho,
                                        eta_graph=eta_graph,
                                        seed=seed,
                                        outer_iterations=args.outer_iterations,
                                    )
                                    row.update(
                                        {
                                            "dataset": args.dataset_name,
                                            "data_source": str(args.data),
                                            "adjacency_shape": "x".join(map(str, adjacency.shape)),
                                            "signal_shape": "x".join(map(str, signals.shape)),
                                        }
                                    )
                                    rows.append(row)
                                    print(
                                        f"obs={observed_percent} alpha={alpha:g} "
                                        f"gamma={gamma:g} beta={beta:g} "
                                        f"delta={delta:g} delta_a={delta_a:g} "
                                        f"mu={mu:g} rho={rho:g} eta={eta_graph:g} "
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
