"""Run real-data mask experiments with per-percentage DGD/SDGD tuning."""

from __future__ import annotations

import argparse
import csv
import time
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
)
from seasurf_paper_experiments import (
    compute_matlab_signal_distances,
    load_matlab_adjacency_signals,
    matlab_mask_type2,
    matlab_parameter_matched_btd_blocks,
    matlab_parameter_matched_cpd_rank,
)
from swdyn_paper_experiments import f1_score_hidden, hidden_mask, relative_error_squared


METHOD_ORDER = ["DGD", "SDGD", "SICA", "BTD", "CPD", "SGL", "UNC", "UNC_masked"]


def make_params(row: dict[str, float], *, rank: int, outer_iterations: int) -> DecompositionParams:
    return DecompositionParams(
        rank=rank,
        alpha=row["alpha"],
        gamma=row["gamma"],
        beta=row["beta"],
        delta=row["delta"],
        delta_A=row["delta_a"],
        mu=row["mu"],
        rho=row["rho"],
        eta_graph=row["eta_graph"],
        zeta=1e-4,
        outer_iterations=outer_iterations,
        inner_iterations_A=5,
        inner_iterations_C=5,
        random_state=0,
    )


def reconstruct_dgd(
    adjacency: np.ndarray,
    signals: np.ndarray,
    mask: np.ndarray,
    params: DecompositionParams,
) -> np.ndarray:
    n_nodes, n_signals, _ = signals.shape
    solver = DynamicGraphDecomposer(params)
    result = solver.fit(adjacency, signals / np.sqrt(n_nodes * n_signals), mask=mask)
    return solver.reconstruct(result.latent_graphs, result.temporal_signatures)


def reconstruct_sdgd(
    adjacency: np.ndarray,
    mask: np.ndarray,
    params: DecompositionParams,
) -> np.ndarray:
    result = fit_proposed_without_signals(adjacency, mask, params)
    return DynamicGraphDecomposer.reconstruct(result.latent_graphs, result.temporal_signatures)


def evaluate(
    adjacency: np.ndarray,
    reconstruction: np.ndarray,
    hmask: np.ndarray,
) -> tuple[float, float]:
    return (
        relative_error_squared(adjacency, reconstruction, hmask),
        f1_score_hidden(adjacency, reconstruction, hmask),
    )


def validation_split_mask(
    observed_mask: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split observed upper-triangular pairs into train/validation masks."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train_mask = observed_mask.copy()
    validation_mask = np.zeros_like(observed_mask)
    n_nodes, _, n_times = observed_mask.shape
    rows, cols = np.triu_indices(n_nodes, k=1)
    for t in range(n_times):
        observed_pairs = np.flatnonzero(observed_mask[:, :, t][rows, cols] > 0.0)
        n_validation = int(round(validation_fraction * len(observed_pairs)))
        if n_validation == 0:
            continue
        selected = rng.choice(observed_pairs, n_validation, replace=False)
        r = rows[selected]
        c = cols[selected]
        train_mask[r, c, t] = 0.0
        train_mask[c, r, t] = 0.0
        validation_mask[r, c, t] = 1.0
        validation_mask[c, r, t] = 1.0
    return train_mask, validation_mask


def tune_method(
    adjacency: np.ndarray,
    signals: np.ndarray,
    fit_mask: np.ndarray,
    score_mask: np.ndarray,
    *,
    method: str,
    rank: int,
    outer_iterations: int,
    alphas: list[float],
    gammas: list[float],
    betas: list[float],
    deltas: list[float],
    mus: list[float],
    rhos: list[float],
    eta_graphs: list[float],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    best: dict[str, float] | None = None
    best_score = float("inf")
    for alpha in alphas:
        for gamma in gammas:
            for beta in betas:
                for delta in ([0.0] if method == "SDGD" else deltas):
                    delta_a = delta
                    for mu in mus:
                        for rho in rhos:
                            for eta_graph in eta_graphs:
                                row = {
                                    "alpha": alpha,
                                    "gamma": gamma,
                                    "beta": beta,
                                    "delta": delta,
                                    "delta_a": delta_a,
                                    "mu": mu,
                                    "rho": rho,
                                    "eta_graph": eta_graph,
                                }
                                params = make_params(
                                    row,
                                    rank=rank,
                                    outer_iterations=outer_iterations,
                                )
                                start = time.perf_counter()
                                if method == "DGD":
                                    reconstruction = reconstruct_dgd(
                                        adjacency,
                                        signals,
                                        fit_mask,
                                        params,
                                    )
                                else:
                                    reconstruction = reconstruct_sdgd(
                                        adjacency,
                                        fit_mask,
                                        params,
                                    )
                                elapsed = time.perf_counter() - start
                                relative_error, f1 = evaluate(adjacency, reconstruction, score_mask)
                                out_row = {
                                    "method": method,
                                    "selection_split": "validation",
                                    **row,
                                    "relative_error": relative_error,
                                    "f1": f1,
                                    "elapsed_seconds": elapsed,
                                }
                                rows.append(out_row)
                                if relative_error < best_score:
                                    best_score = relative_error
                                    best = row
                                print(
                                    f"tune {method} alpha={alpha:g} gamma={gamma:g} "
                                    f"beta={beta:g} delta={delta:g} delta_a={delta_a:g} "
                                    f"mu={mu:g} rho={rho:g} eta={eta_graph:g} "
                                    f"RE={relative_error:.4f} F1={f1:.4f}",
                                    flush=True,
                                )
    if best is None:
        raise RuntimeError(f"No candidates for {method}")
    return best, rows


def run_baselines(
    adjacency: np.ndarray,
    signals: np.ndarray,
    mask: np.ndarray,
    *,
    rank: int,
    cpd_iterations: int,
    btd_iterations: int,
    sgl_iterations: int,
    sgl_a: float,
    sgl_b: float,
    seed: int,
    unc_mode: str = "zero_filled",
) -> list[tuple[str, np.ndarray, int]]:
    n_nodes, _, n_times = adjacency.shape
    observed = adjacency * mask
    cpd_rank = matlab_parameter_matched_cpd_rank(n_nodes, n_times, rank)
    btd_blocks = matlab_parameter_matched_btd_blocks(n_nodes, n_times, rank)
    distances = compute_matlab_signal_distances(signals)
    results = [
        (
            "SICA",
            fit_sica_baseline(
                observed,
                rank=min(rank, n_times),
                mask=None,
                max_iterations=500,
                random_state=seed,
                backend="sklearn",
            ).reconstruction,
            min(rank, n_times),
        ),
        (
            "BTD",
            fit_btd_ll1_als(
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
            fit_cpd_als(
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
            fit_smooth_topology_baseline(
                distances,
                a=sgl_a,
                b=sgl_b,
                max_iterations=sgl_iterations,
                tolerance=1e-5,
            ).reconstruction,
            rank,
        ),
    ]
    if unc_mode in {"zero_filled", "both"}:
        results.append(
            (
                "UNC",
                fit_unconstrained_observation_als(
                    observed,
                    rank=rank,
                    mask=None,
                    max_iterations=30,
                    random_state=seed,
                ).reconstruction,
                rank,
            )
        )
    if unc_mode in {"masked", "both"}:
        results.append(
            (
                "UNC_masked",
                fit_unconstrained_observation_als(
                    adjacency,
                    rank=rank,
                    mask=mask,
                    max_iterations=30,
                    random_state=seed,
                ).reconstruction,
                rank,
            )
        )
    return results


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning-output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--observed", type=int, nargs="+", default=[10, 30, 50, 70, 90])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outer-iterations", type=int, default=20)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--gammas", type=float, nargs="+", default=[1.0, 3.0])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.01, 0.1])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.0, 2.0])
    parser.add_argument("--delta-as", type=float, nargs="+", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mus", type=float, nargs="+", default=[0.1])
    parser.add_argument("--rhos", type=float, nargs="+", default=[1.0])
    parser.add_argument("--eta-graphs", type=float, nargs="+", default=[0.0])
    parser.add_argument("--sgl-a", type=float, default=2.85e3)
    parser.add_argument("--sgl-b", type=float, default=0.1)
    parser.add_argument("--sgl-iterations", type=int, default=300)
    parser.add_argument("--cpd-iterations", type=int, default=10)
    parser.add_argument("--btd-iterations", type=int, default=25)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--skip-baselines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adjacency, signals = load_matlab_adjacency_signals(args.data)
    summary_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    print(
        f"dataset={args.dataset_name} data={args.data} "
        f"A={adjacency.shape} X={signals.shape} R={args.rank}",
        flush=True,
    )
    for observed_percent in args.observed:
        seed = args.seed + observed_percent
        mask = matlab_mask_type2(adjacency, observed_percent, seed=seed)
        hmask = hidden_mask(mask)
        train_mask, validation_mask = validation_split_mask(
            mask,
            validation_fraction=args.validation_fraction,
            seed=args.seed + 1000 + observed_percent,
        )
        best_params: dict[str, dict[str, float]] = {}
        for method in ["DGD", "SDGD"]:
            best, rows = tune_method(
                adjacency,
                signals,
                train_mask,
                validation_mask,
                method=method,
                rank=args.rank,
                outer_iterations=args.outer_iterations,
                alphas=args.alphas,
                gammas=args.gammas,
                betas=args.betas,
                deltas=args.deltas,
                mus=args.mus,
                rhos=args.rhos,
                eta_graphs=args.eta_graphs,
            )
            best_params[method] = best
            for row in rows:
                tuning_rows.append(
                    {
                        "dataset": args.dataset_name,
                        "data_source": str(args.data),
                        "observed_percent": observed_percent,
                        "rank": args.rank,
                        "validation_fraction": args.validation_fraction,
                        **row,
                    }
                )
        for method in ["DGD", "SDGD"]:
            params = make_params(
                best_params[method],
                rank=args.rank,
                outer_iterations=args.outer_iterations,
            )
            start = time.perf_counter()
            if method == "DGD":
                reconstruction = reconstruct_dgd(adjacency, signals, mask, params)
            else:
                reconstruction = reconstruct_sdgd(adjacency, mask, params)
            elapsed = time.perf_counter() - start
            relative_error, f1 = evaluate(adjacency, reconstruction, hmask)
            summary_rows.append(
                {
                    "dataset": args.dataset_name,
                    "data_source": str(args.data),
                    "observed_percent": observed_percent,
                    "rank": args.rank,
                    "validation_fraction": args.validation_fraction,
                    "method": method,
                    "method_rank": args.rank,
                    **best_params[method],
                    "relative_error": relative_error,
                    "f1": f1,
                    "elapsed_seconds": elapsed,
                }
            )
            print(
                f"best {method} obs={observed_percent} RE={relative_error:.4f} "
                f"F1={f1:.4f} params={best_params[method]}",
                flush=True,
            )
        if args.skip_baselines:
            continue
        for method, reconstruction, method_rank in run_baselines(
            adjacency,
            signals,
            mask,
            rank=args.rank,
            cpd_iterations=args.cpd_iterations,
            btd_iterations=args.btd_iterations,
            sgl_iterations=args.sgl_iterations,
            sgl_a=args.sgl_a,
            sgl_b=args.sgl_b,
            seed=seed,
        ):
            relative_error, f1 = evaluate(adjacency, reconstruction, hmask)
            summary_rows.append(
                {
                    "dataset": args.dataset_name,
                    "data_source": str(args.data),
                    "observed_percent": observed_percent,
                    "rank": args.rank,
                    "validation_fraction": args.validation_fraction,
                    "method": method,
                    "method_rank": method_rank,
                    "alpha": "",
                    "gamma": "",
                    "beta": "",
                    "delta": "",
                    "delta_a": "",
                    "mu": "",
                    "rho": "",
                    "eta_graph": "",
                    "relative_error": relative_error,
                    "f1": f1,
                    "elapsed_seconds": "",
                }
            )
            print(
                f"baseline {method} obs={observed_percent} RE={relative_error:.4f} F1={f1:.4f}",
                flush=True,
            )
    order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    summary_rows.sort(key=lambda row: (int(row["observed_percent"]), order[str(row["method"])]))
    write_csv(args.tuning_output, tuning_rows)
    write_csv(args.output, summary_rows)
    print(f"wrote {args.tuning_output}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
