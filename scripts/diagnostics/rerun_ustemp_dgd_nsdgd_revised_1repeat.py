"""Rerun USTemp DGD/NSDGD once with revised hyperparameter grids."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

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
                "dataset": "USTemp",
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
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=ROOT
        / "results/csv/diagnostics/ustemp_dgd_nsdgd_revised_1repeat_detail_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT
        / "results/csv/diagnostics/ustemp_dgd_nsdgd_revised_1repeat_summary_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv",
    )
    parser.add_argument(
        "--tuning-output",
        type=Path,
        default=ROOT
        / "results/csv/diagnostics/ustemp_dgd_nsdgd_revised_1repeat_tuning_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv",
    )
    args = parser.parse_args()

    adjacency, signals = load_adjacency_signals(DATA)
    detail_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

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
            best_row = None
            best_score = float("inf")
            candidates = candidate_rows(method, args)
            for params_row in candidates:
                params = make_params(params_row, rank=3, outer_iterations=args.outer_iterations)
                start = time.perf_counter()
                if method == "DGD":
                    reconstruction = reconstruct_dgd(adjacency, signals, train_mask, params)
                else:
                    reconstruction = reconstruct_sdgd(adjacency, train_mask, params)
                elapsed = time.perf_counter() - start
                val_re, val_f1 = evaluate(adjacency, reconstruction, validation_mask)
                tuning_rows.append(
                    {
                        "dataset": "USTemp",
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

            if best_row is None:
                raise RuntimeError(f"No candidates evaluated for {method}")

            params = make_params(best_row, rank=3, outer_iterations=args.outer_iterations)
            start = time.perf_counter()
            if method == "DGD":
                reconstruction = reconstruct_dgd(adjacency, signals, mask, params)
            else:
                reconstruction = reconstruct_sdgd(adjacency, mask, params)
            elapsed = time.perf_counter() - start
            hidden_re, hidden_f1 = evaluate(adjacency, reconstruction, test_mask)
            detail_rows.append(
                {
                    "dataset": "USTemp",
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
