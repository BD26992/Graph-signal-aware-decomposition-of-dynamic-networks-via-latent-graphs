"""Rerun USTemp NSDGD with a broader gamma grid."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from real_repeated_mask_experiments import load_adjacency_signals
from real_tuned_mask_experiments import (
    evaluate,
    make_params,
    reconstruct_sdgd,
    validation_split_mask,
)
from seasurf_paper_experiments import matlab_mask_type2
from swdyn_paper_experiments import hidden_mask


ROOT = Path(__file__).resolve().parent
DATA = Path("/Users/bishwadeep/Downloads/tensors_decomp-2/real_data/UStemp/UStemp_15_data.mat")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["observed_percent"]), []).append(row)

    summary = []
    for observed_percent, group in sorted(grouped.items()):
        re_values = [float(row["relative_error"]) for row in group]
        f1_values = [float(row["f1"]) for row in group]
        summary.append(
            {
                "dataset": "USTemp",
                "data_source": str(DATA),
                "observed_percent": observed_percent,
                "rank": 3,
                "method": "SDGD",
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
                "elapsed_seconds_mean": statistics.fmean(
                    float(row["elapsed_seconds"]) for row in group
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--observed", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60, 70, 80, 90])
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 2.0])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.1])
    parser.add_argument("--eta-graphs", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=ROOT
        / "results/csv/diagnostics/ustemp_nsdgd_expanded_gamma_detail_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT
        / "results/csv/diagnostics/ustemp_nsdgd_expanded_gamma_summary_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv",
    )
    parser.add_argument(
        "--tuning-output",
        type=Path,
        default=ROOT
        / "results/csv/diagnostics/ustemp_nsdgd_expanded_gamma_tuning_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv",
    )
    args = parser.parse_args()

    adjacency, signals = load_adjacency_signals(DATA)
    del signals

    detail_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    total_jobs = args.repeats * len(args.observed)
    job = 0

    for repeat in range(args.repeats):
        for observed_percent in args.observed:
            job += 1
            mask_seed = args.seed + 1000 * repeat + observed_percent
            validation_seed = args.seed + 1000 * repeat + 1000 + observed_percent
            mask = matlab_mask_type2(adjacency, observed_percent, seed=mask_seed)
            train_mask, validation_mask = validation_split_mask(
                mask,
                validation_fraction=args.validation_fraction,
                seed=validation_seed,
            )
            test_mask = hidden_mask(mask)

            best_row = None
            best_score = float("inf")
            for gamma in args.gammas:
                for beta in args.betas:
                    for eta_graph in args.eta_graphs:
                        params_row = {
                            "alpha": 1.0,
                            "gamma": gamma,
                            "beta": beta,
                            "delta": 0.0,
                            "delta_a": 0.0,
                            "mu": 0.1,
                            "rho": 0.001,
                            "eta_graph": eta_graph,
                        }
                        params = make_params(params_row, rank=3, outer_iterations=args.outer_iterations)
                        start = time.perf_counter()
                        reconstruction = reconstruct_sdgd(adjacency, train_mask, params)
                        elapsed = time.perf_counter() - start
                        val_re, val_f1 = evaluate(adjacency, reconstruction, validation_mask)
                        row = {
                            "dataset": "USTemp",
                            "data_source": str(DATA),
                            "observed_percent": observed_percent,
                            "repeat": repeat,
                            "mask_seed": mask_seed,
                            "validation_seed": validation_seed,
                            "rank": 3,
                            "validation_fraction": args.validation_fraction,
                            "method": "SDGD",
                            "selection_split": "validation",
                            **params_row,
                            "relative_error": val_re,
                            "f1": val_f1,
                            "elapsed_seconds": elapsed,
                        }
                        tuning_rows.append(row)
                        if val_re < best_score:
                            best_score = val_re
                            best_row = params_row

            if best_row is None:
                raise RuntimeError("No NSDGD tuning candidates evaluated")

            params = make_params(best_row, rank=3, outer_iterations=args.outer_iterations)
            start = time.perf_counter()
            reconstruction = reconstruct_sdgd(adjacency, mask, params)
            elapsed = time.perf_counter() - start
            hidden_re, hidden_f1 = evaluate(adjacency, reconstruction, test_mask)
            detail_rows.append(
                {
                    "dataset": "USTemp",
                    "data_source": str(DATA),
                    "observed_percent": observed_percent,
                    "repeat": repeat,
                    "mask_seed": mask_seed,
                    "validation_seed": validation_seed,
                    "rank": 3,
                    "validation_fraction": args.validation_fraction,
                    "method": "SDGD",
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
                f"{job}/{total_jobs} repeat={repeat} obs={observed_percent} "
                f"best={best_row} hidden_RE={hidden_re:.4g} F1={hidden_f1:.4g}",
                flush=True,
            )

    print(f"wrote {args.detail_output}")
    print(f"wrote {args.summary_output}")
    print(f"wrote {args.tuning_output}")


if __name__ == "__main__":
    main()
