"""Rerun Contact DGD/NSDGD once with revised hyperparameter grids."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import statistics
import time
from pathlib import Path

import numpy as np

from dynamic_graph_decomposition import DynamicGraphDecomposer, signal_distance_matrices
from real_repeated_mask_experiments import load_adjacency_signals
from real_tuned_mask_experiments import (
    METHOD_ORDER,
    evaluate,
    make_params,
    reconstruct_dgd,
    reconstruct_sdgd,
    validation_split_mask,
)
from seasurf_paper_experiments import matlab_mask_type2
from swdyn_paper_experiments import hidden_mask


ROOT = Path(__file__).resolve().parent
DATA = Path("/Users/bishwadeep/Downloads/tensors_decomp-2/real_data/contact/contact_50_data.mat")
_WORKER_CONTEXT: dict[str, object] = {}


def _init_worker(
    adjacency: np.ndarray,
    scaled_signals: np.ndarray,
    signal_distances: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    rank: int,
    outer_iterations: int,
) -> None:
    _WORKER_CONTEXT.update(
        {
            "adjacency": adjacency,
            "scaled_signals": scaled_signals,
            "signal_distances": signal_distances,
            "train_mask": train_mask,
            "validation_mask": validation_mask,
            "rank": rank,
            "outer_iterations": outer_iterations,
        }
    )


def _evaluate_candidate(job: tuple[str, dict[str, float]]) -> dict[str, object]:
    method, params_row = job
    adjacency = _WORKER_CONTEXT["adjacency"]
    params = make_params(
        params_row,
        rank=int(_WORKER_CONTEXT["rank"]),
        outer_iterations=int(_WORKER_CONTEXT["outer_iterations"]),
    )
    start = time.perf_counter()
    if method == "DGD":
        reconstruction = reconstruct_dgd_cached(
            adjacency,
            _WORKER_CONTEXT["scaled_signals"],
            _WORKER_CONTEXT["signal_distances"],
            _WORKER_CONTEXT["train_mask"],
            params,
        )
    else:
        reconstruction = reconstruct_sdgd_fast(
            adjacency,
            _WORKER_CONTEXT["train_mask"],
            params,
        )
    elapsed = time.perf_counter() - start
    relative_error, f1 = evaluate(
        adjacency,
        reconstruction,
        _WORKER_CONTEXT["validation_mask"],
    )
    return {
        "method": method,
        **params_row,
        "relative_error": relative_error,
        "f1": f1,
        "elapsed_seconds": elapsed,
    }


def reconstruct_dgd_cached(
    adjacency: np.ndarray,
    scaled_signals: np.ndarray,
    signal_distances: np.ndarray,
    mask: np.ndarray,
    params,
) -> np.ndarray:
    params.record_objective = False
    solver = DynamicGraphDecomposer(params)
    result = solver.fit(
        adjacency,
        scaled_signals,
        mask=mask,
        signal_distances=signal_distances,
    )
    return solver.reconstruct(result.latent_graphs, result.temporal_signatures)


def reconstruct_sdgd_fast(adjacency: np.ndarray, mask: np.ndarray, params) -> np.ndarray:
    params.record_objective = False
    return reconstruct_sdgd(adjacency, mask, params)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["observed_percent"]), str(row["method"])), []).append(row)

    out = []
    for (observed_percent, method), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], METHOD_ORDER.index(item[0][1])),
    ):
        re_values = [float(row["relative_error"]) for row in group]
        f1_values = [float(row["f1"]) for row in group]
        elapsed_values = [float(row["elapsed_seconds"]) for row in group]
        out.append(
            {
                "dataset": "Contact",
                "data_source": str(DATA),
                "observed_percent": observed_percent,
                "rank": 3,
                "method": method,
                "method_rank": 3,
                "n_runs": len(group),
                "relative_error_mean": statistics.fmean(re_values),
                "relative_error_std": statistics.stdev(re_values) if len(group) > 1 else 0.0,
                "relative_error_min": min(re_values),
                "relative_error_max": max(re_values),
                "f1_mean": statistics.fmean(f1_values),
                "f1_std": statistics.stdev(f1_values) if len(group) > 1 else 0.0,
                "f1_min": min(f1_values),
                "f1_max": max(f1_values),
                "elapsed_seconds_mean": statistics.fmean(elapsed_values),
            }
        )
    return out


def candidate_rows(method: str, args: argparse.Namespace) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if method == "DGD":
        for delta in args.dgd_deltas:
            for mu in args.mus:
                for gamma in args.dgd_gammas:
                    for beta in args.dgd_betas:
                        rows.append(
                            {
                                "alpha": 1.0,
                                "gamma": gamma,
                                "beta": beta,
                                "delta": delta,
                                "delta_a": delta,
                                "mu": mu,
                                "rho": 0.001,
                                "eta_graph": 0.0,
                            }
                        )
    else:
        for gamma in args.sdgd_gammas:
            for beta in args.sdgd_betas:
                for mu in args.mus:
                    rows.append(
                        {
                            "alpha": 1.0,
                            "gamma": gamma,
                            "beta": beta,
                            "delta": 0.0,
                            "delta_a": 0.0,
                            "mu": mu,
                            "rho": 0.001,
                            "eta_graph": 0.0,
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--observed", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60, 70, 80, 90])
    parser.add_argument("--dgd-deltas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--dgd-gammas", type=float, nargs="+", default=[0.0001, 0.001, 0.01])
    parser.add_argument("--dgd-betas", type=float, nargs="+", default=[0.0, 0.001, 0.01])
    parser.add_argument("--sdgd-gammas", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 2.0])
    parser.add_argument("--sdgd-betas", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.1])
    parser.add_argument("--mus", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=ROOT
        / "results/csv/diagnostics/contact_dgd_nsdgd_revised_1repeat_detail_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT
        / "results/csv/diagnostics/contact_dgd_nsdgd_revised_1repeat_summary_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv",
    )
    parser.add_argument(
        "--tuning-output",
        type=Path,
        default=ROOT
        / "results/csv/diagnostics/contact_dgd_nsdgd_revised_1repeat_tuning_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv",
    )
    args = parser.parse_args()

    adjacency, signals = load_adjacency_signals(DATA)
    n_nodes, n_signals, _ = signals.shape
    scaled_signals = signals / np.sqrt(n_nodes * n_signals)
    signal_distances = signal_distance_matrices(scaled_signals)
    detail_rows: list[dict[str, object]] = read_existing_rows(args.detail_output) if args.resume else []
    tuning_rows: list[dict[str, object]] = read_existing_rows(args.tuning_output) if args.resume else []
    completed = {
        (int(row["observed_percent"]), str(row["method"]))
        for row in detail_rows
    }

    total_jobs = len(args.observed) * 2
    job = 0
    for observed_percent in args.observed:
        mask_seed = args.seed + 1000 * args.repeat + observed_percent
        validation_seed = args.seed + 1000 * args.repeat + 1000 + observed_percent
        mask = matlab_mask_type2(adjacency, observed_percent, seed=mask_seed)
        train_mask, validation_mask = validation_split_mask(
            mask,
            validation_fraction=args.validation_fraction,
            seed=validation_seed,
        )
        test_mask = hidden_mask(mask)

        for method in ["DGD", "SDGD"]:
            job += 1
            if args.resume and (observed_percent, method) in completed:
                print(f"{job}/{total_jobs} obs={observed_percent} {method} already complete; skipping")
                continue
            if args.resume:
                tuning_rows = [
                    row
                    for row in tuning_rows
                    if not (
                        int(row["observed_percent"]) == observed_percent
                        and str(row["method"]) == method
                    )
                ]
                write_csv(args.tuning_output, tuning_rows)
            best_row = None
            best_score = float("inf")
            candidates = candidate_rows(method, args)
            candidate_jobs = [(method, params_row) for params_row in candidates]
            if args.workers > 1:
                with ProcessPoolExecutor(
                    max_workers=args.workers,
                    initializer=_init_worker,
                    initargs=(
                        adjacency,
                        scaled_signals,
                        signal_distances,
                        train_mask,
                        validation_mask,
                        3,
                        args.outer_iterations,
                    ),
                ) as pool:
                    futures = [pool.submit(_evaluate_candidate, item) for item in candidate_jobs]
                    candidate_results = []
                    for future in as_completed(futures):
                        result_row = future.result()
                        candidate_results.append(result_row)
                        params_row = {
                            key: result_row[key]
                            for key in [
                                "alpha",
                                "gamma",
                                "beta",
                                "delta",
                                "delta_a",
                                "mu",
                                "rho",
                                "eta_graph",
                            ]
                        }
                        val_re = float(result_row["relative_error"])
                        val_f1 = float(result_row["f1"])
                        elapsed = float(result_row["elapsed_seconds"])
                        tuning_rows.append(
                            {
                                "dataset": "Contact",
                                "data_source": str(DATA),
                                "observed_percent": observed_percent,
                                "repeat": args.repeat,
                                "mask_seed": mask_seed,
                                "validation_seed": validation_seed,
                                "rank": 3,
                                "validation_fraction": args.validation_fraction,
                                "method": method,
                                "selection_split": "validation",
                                **params_row,
                                "relative_error": val_re,
                                "f1": val_f1,
                                "elapsed_seconds": elapsed,
                            }
                        )
                        write_csv(args.tuning_output, tuning_rows)
                        if val_re < best_score:
                            best_score = val_re
                            best_row = params_row
                        params_preview = {
                            key: result_row[key]
                            for key in [
                                "gamma",
                                "beta",
                                "delta",
                                "delta_a",
                                "mu",
                            ]
                        }
                        print(
                            f"done obs={observed_percent} {method} "
                            f"{len(candidate_results)}/{len(candidate_jobs)} "
                            f"params={params_preview} "
                            f"val_RE={float(result_row['relative_error']):.4g}",
                            flush=True,
                        )
            else:
                _init_worker(
                    adjacency,
                    scaled_signals,
                    signal_distances,
                    train_mask,
                    validation_mask,
                    3,
                    args.outer_iterations,
                )
                candidate_results = [_evaluate_candidate(item) for item in candidate_jobs]

            for result_row in candidate_results:
                params_row = {
                    key: result_row[key]
                    for key in ["alpha", "gamma", "beta", "delta", "delta_a", "mu", "rho", "eta_graph"]
                }
                val_re = float(result_row["relative_error"])
                val_f1 = float(result_row["f1"])
                elapsed = float(result_row["elapsed_seconds"])
                if args.workers > 1:
                    continue
                tuning_rows.append(
                    {
                        "dataset": "Contact",
                        "data_source": str(DATA),
                        "observed_percent": observed_percent,
                        "repeat": args.repeat,
                        "mask_seed": mask_seed,
                        "validation_seed": validation_seed,
                        "rank": 3,
                        "validation_fraction": args.validation_fraction,
                        "method": method,
                        "selection_split": "validation",
                        **params_row,
                        "relative_error": val_re,
                        "f1": val_f1,
                        "elapsed_seconds": elapsed,
                    }
                )
                if val_re < best_score:
                    best_score = val_re
                    best_row = params_row
                write_csv(args.tuning_output, tuning_rows)
                print(
                    f"tune obs={observed_percent} {method} "
                    f"params={params_row} val_RE={val_re:.4g} F1={val_f1:.4g} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if best_row is None:
                raise RuntimeError(f"No candidates evaluated for {method}")

            params = make_params(best_row, rank=3, outer_iterations=args.outer_iterations)
            start = time.perf_counter()
            if method == "DGD":
                reconstruction = reconstruct_dgd_cached(
                    adjacency,
                    scaled_signals,
                    signal_distances,
                    mask,
                    params,
                )
            else:
                reconstruction = reconstruct_sdgd_fast(adjacency, mask, params)
            elapsed = time.perf_counter() - start
            hidden_re, hidden_f1 = evaluate(adjacency, reconstruction, test_mask)
            detail_rows.append(
                {
                    "dataset": "Contact",
                    "data_source": str(DATA),
                    "observed_percent": observed_percent,
                    "repeat": args.repeat,
                    "mask_seed": mask_seed,
                    "validation_seed": validation_seed,
                    "rank": 3,
                    "validation_fraction": args.validation_fraction,
                    "method": method,
                    "method_rank": 3,
                    "selection_metric": "validation_relative_error",
                    **best_row,
                    "relative_error": hidden_re,
                    "f1": hidden_f1,
                    "elapsed_seconds": elapsed,
                }
            )
            write_csv(args.detail_output, detail_rows)
            write_csv(args.tuning_output, tuning_rows)
            write_csv(args.summary_output, summarize(detail_rows))
            print(
                f"{job}/{total_jobs} obs={observed_percent} {method} "
                f"best={best_row} hidden_RE={hidden_re:.4g} F1={hidden_f1:.4g}",
                flush=True,
            )

    print(f"wrote {args.detail_output}")
    print(f"wrote {args.summary_output}")
    print(f"wrote {args.tuning_output}")


if __name__ == "__main__":
    main()
