"""Run SwDyn mask experiments with per-percentage DGD/SDGD tuning."""

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
    signal_distance_matrices,
)
from swdyn_paper_experiments import (
    SwDynData,
    f1_score_hidden,
    hidden_mask,
    make_swdyn,
    matlab_parameter_matched_btd_blocks,
    matlab_parameter_matched_cpd_rank,
    relative_error_squared,
    symmetric_observation_mask,
)


METHOD_ORDER = ["DGD", "SDGD", "SICA", "BTD", "CPD", "SGL", "UNC", "UNC_masked"]


def params_from_row(row: dict[str, float], *, rank: int, outer_iterations: int) -> DecompositionParams:
    return DecompositionParams(
        rank=rank,
        alpha=row["alpha"],
        gamma=row["gamma"],
        beta=row["beta"],
        delta=row["delta"],
        delta_A=row["delta_A"],
        mu=row["mu"],
        rho=row["rho"],
        zeta=1e-4,
        outer_iterations=outer_iterations,
        inner_iterations_A=5,
        inner_iterations_C=5,
        random_state=0,
    )


def reconstruct_dgd(
    data: SwDynData,
    mask: np.ndarray,
    params: DecompositionParams,
) -> np.ndarray:
    n_nodes, n_signals, _ = data.signals.shape
    scaled_signals = data.signals / np.sqrt(n_nodes * n_signals)
    solver = DynamicGraphDecomposer(params)
    result = solver.fit(data.adjacency, scaled_signals, mask=mask)
    return solver.reconstruct(result.latent_graphs, result.temporal_signatures)


def reconstruct_sdgd(
    data: SwDynData,
    mask: np.ndarray,
    params: DecompositionParams,
) -> np.ndarray:
    result = fit_proposed_without_signals(data.adjacency, mask, params)
    return DynamicGraphDecomposer.reconstruct(result.latent_graphs, result.temporal_signatures)


def evaluate(
    data: SwDynData,
    reconstruction: np.ndarray,
    hmask: np.ndarray,
) -> tuple[float, float]:
    return (
        relative_error_squared(data.adjacency, reconstruction, hmask),
        f1_score_hidden(data.adjacency, reconstruction, hmask),
    )


def validation_split_mask(
    observed_mask: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split observed entries into train/validation masks symmetrically."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train_mask = observed_mask.copy()
    validation_mask = np.zeros_like(observed_mask)
    n_nodes, _, n_times = observed_mask.shape
    rows, cols = np.triu_indices(n_nodes, k=1)
    for t in range(n_times):
        observed_edges = np.flatnonzero(observed_mask[:, :, t][rows, cols] > 0)
        n_validation = int(round(validation_fraction * len(observed_edges)))
        if n_validation == 0:
            continue
        selected = rng.choice(observed_edges, n_validation, replace=False)
        r = rows[selected]
        c = cols[selected]
        train_mask[r, c, t] = 0.0
        train_mask[c, r, t] = 0.0
        validation_mask[r, c, t] = 1.0
        validation_mask[c, r, t] = 1.0
    return train_mask, validation_mask


def tune_method(
    data: SwDynData,
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
) -> tuple[dict[str, float], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    best: dict[str, float] | None = None
    best_score = float("inf")
    for alpha in alphas:
        for gamma in gammas:
            for beta in betas:
                for delta in ([0.0] if method == "SDGD" else deltas):
                    delta_A = delta
                    for mu in mus:
                        for rho in rhos:
                            row = {
                                "alpha": alpha,
                                "gamma": gamma,
                                "beta": beta,
                                "delta": delta,
                                "delta_A": delta_A,
                                "mu": mu,
                                "rho": rho,
                            }
                            params = params_from_row(
                                row,
                                rank=rank,
                                outer_iterations=outer_iterations,
                            )
                            start = time.perf_counter()
                            if method == "DGD":
                                reconstruction = reconstruct_dgd(data, fit_mask, params)
                            else:
                                reconstruction = reconstruct_sdgd(data, fit_mask, params)
                            elapsed = time.perf_counter() - start
                            relative_error, f1 = evaluate(data, reconstruction, score_mask)
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
                                f"beta={beta:g} delta={delta:g} delta_A={delta_A:g} "
                                f"mu={mu:g} rho={rho:g} RE={relative_error:.4f} "
                                f"F1={f1:.4f}",
                                flush=True,
                            )
    if best is None:
        raise RuntimeError(f"No tuning candidates for {method}")
    return best, rows


def run_baselines(
    data: SwDynData,
    mask: np.ndarray,
    *,
    rank: int,
    max_iterations: int,
    unc_mode: str = "zero_filled",
) -> list[tuple[str, np.ndarray, int]]:
    n_nodes, n_signals, n_times = data.signals.shape
    observed = data.adjacency * mask
    distances = signal_distance_matrices(data.signals / np.sqrt(n_nodes * n_signals))
    cpd_rank = matlab_parameter_matched_cpd_rank(n_nodes, n_times, rank)
    btd_blocks = matlab_parameter_matched_btd_blocks(n_nodes, n_times, rank)
    results = [
        (
            "SICA",
            fit_sica_baseline(
                observed,
                rank=min(rank, n_times),
                mask=None,
                max_iterations=max_iterations,
                random_state=0,
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
                max_iterations=max_iterations,
                random_state=0,
            ).reconstruction,
            btd_blocks,
        ),
        (
            "CPD",
            fit_cpd_als(
                observed,
                rank=cpd_rank,
                mask=None,
                max_iterations=max_iterations,
                random_state=0,
                backend="tensorly",
            ).reconstruction,
            cpd_rank,
        ),
        (
            "SGL",
            fit_smooth_topology_baseline(
                distances,
                a=3e4,
                b=10.0,
                max_iterations=max_iterations,
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
                    max_iterations=max_iterations,
                    random_state=0,
                ).reconstruction,
                rank,
            )
        )
    if unc_mode in {"masked", "both"}:
        results.append(
            (
                "UNC_masked",
                fit_unconstrained_observation_als(
                    data.adjacency,
                    rank=rank,
                    mask=mask,
                    max_iterations=max_iterations,
                    random_state=0,
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


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: (int(row["observed_percent"]), METHOD_ORDER.index(str(row["method"]))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/csv/swdyn_tuned_mask_summary.csv"))
    parser.add_argument("--tuning-output", type=Path, default=Path("results/csv/swdyn_tuned_mask_grid.csv"))
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--observed", type=int, nargs="+", default=[20, 60, 90])
    parser.add_argument("--outer-iterations", type=int, default=15)
    parser.add_argument("--baseline-iterations", type=int, default=15)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--gammas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--betas", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.1, 1.0])
    parser.add_argument("--delta-as", type=float, nargs="+", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mus", type=float, nargs="+", default=[1e-2])
    parser.add_argument("--rhos", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--skip-baselines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = make_swdyn()
    summary_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    for observed_percent in args.observed:
        mask = symmetric_observation_mask(
            data.adjacency.shape,
            observed_percent / 100.0,
            seed=700 + observed_percent,
        )
        hmask = hidden_mask(mask)
        train_mask, validation_mask = validation_split_mask(
            mask,
            validation_fraction=args.validation_fraction,
            seed=900 + observed_percent,
        )
        best_params: dict[str, dict[str, float]] = {}
        for method in ["DGD", "SDGD"]:
            best, rows = tune_method(
                data,
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
            )
            best_params[method] = best
            for row in rows:
                tuning_rows.append(
                    {
                        "dataset": "SwDyn",
                        "observed_percent": observed_percent,
                        "rank": args.rank,
                        "validation_fraction": args.validation_fraction,
                        **row,
                    }
                )

        for method in ["DGD", "SDGD"]:
            params = params_from_row(
                best_params[method],
                rank=args.rank,
                outer_iterations=args.outer_iterations,
            )
            start = time.perf_counter()
            if method == "DGD":
                reconstruction = reconstruct_dgd(data, mask, params)
            else:
                reconstruction = reconstruct_sdgd(data, mask, params)
            elapsed = time.perf_counter() - start
            relative_error, f1 = evaluate(data, reconstruction, hmask)
            summary_rows.append(
                {
                    "dataset": "SwDyn",
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
                f"best {method} obs={observed_percent} RE={relative_error:.4f} F1={f1:.4f} "
                f"params={best_params[method]}",
                flush=True,
            )

        if args.skip_baselines:
            continue

        for method, reconstruction, method_rank in run_baselines(
            data,
            mask,
            rank=args.rank,
            max_iterations=args.baseline_iterations,
        ):
            relative_error, f1 = evaluate(data, reconstruction, hmask)
            summary_rows.append(
                {
                    "dataset": "SwDyn",
                    "observed_percent": observed_percent,
                    "rank": args.rank,
                    "method": method,
                    "method_rank": method_rank,
                    "alpha": "",
                    "gamma": "",
                    "beta": "",
                    "delta": "",
                    "delta_A": "",
                    "mu": "",
                    "rho": "",
                    "relative_error": relative_error,
                    "f1": f1,
                    "elapsed_seconds": "",
                }
            )
            print(
                f"baseline {method} obs={observed_percent} RE={relative_error:.4f} F1={f1:.4f}",
                flush=True,
            )

    write_csv(args.tuning_output, tuning_rows)
    write_csv(args.output, summarize(summary_rows))
    print(f"wrote {args.tuning_output}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
