"""Run repeated SwDyn train/validation/test experiments.

This keeps the same method definitions as ``swdyn_tuned_mask_experiments.py``
and adds repeat-level CSV output for plotting mean/std curves.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import statistics
import time
from pathlib import Path

from swdyn_paper_experiments import hidden_mask, make_swdyn, symmetric_observation_mask
from swdyn_tuned_mask_experiments import (
    METHOD_ORDER,
    evaluate,
    params_from_row,
    reconstruct_dgd,
    reconstruct_sdgd,
    run_baselines,
    tune_method,
    validation_split_mask,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def blank_params() -> dict[str, object]:
    return {
        "alpha": "",
        "gamma": "",
        "beta": "",
        "delta": "",
        "delta_A": "",
        "mu": "",
        "rho": "",
    }


def summarize(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in detail_rows:
        groups.setdefault((int(row["observed_percent"]), str(row["method"])), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (observed_percent, method), rows in sorted(
        groups.items(),
        key=lambda item: (item[0][0], METHOD_ORDER.index(item[0][1])),
    ):
        re_values = [float(row["relative_error"]) for row in rows]
        f1_values = [float(row["f1"]) for row in rows]
        elapsed_values = [
            float(row["elapsed_seconds"])
            for row in rows
            if str(row.get("elapsed_seconds", "")) != ""
        ]
        n_runs = len(rows)
        summary_rows.append(
            {
                "dataset": "SwDyn",
                "observed_percent": observed_percent,
                "rank": rows[0]["rank"],
                "method": method,
                "method_rank": rows[0]["method_rank"],
                "n_runs": n_runs,
                "relative_error_mean": statistics.fmean(re_values),
                "relative_error_std": statistics.stdev(re_values) if n_runs > 1 else 0.0,
                "relative_error_min": min(re_values),
                "relative_error_max": max(re_values),
                "f1_mean": statistics.fmean(f1_values),
                "f1_std": statistics.stdev(f1_values) if n_runs > 1 else 0.0,
                "f1_min": min(f1_values),
                "f1_max": max(f1_values),
                "elapsed_seconds_mean": (
                    statistics.fmean(elapsed_values) if elapsed_values else ""
                ),
            }
        )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--tuning-output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--observed", type=int, nargs="+", default=[20, 40, 60, 80, 90])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outer-iterations", type=int, default=15)
    parser.add_argument("--baseline-iterations", type=int, default=15)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--mask-seed-base", type=int, default=700)
    parser.add_argument("--validation-seed-base", type=int, default=900)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 5.0, 10.0, 20.0])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.01])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0001, 0.001])
    parser.add_argument("--deltas", type=float, nargs="+", default=[10.0, 50.0, 100.0])
    parser.add_argument("--mus", type=float, nargs="+", default=[0.1])
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.001])
    parser.add_argument(
        "--unc-mode",
        choices=["zero_filled", "masked", "both"],
        default="zero_filled",
        help=(
            "UNC baseline variant: zero_filled fits the zero-filled observed "
            "tensor; masked passes the observation mask to UNC; both reports both."
        ),
    )
    parser.add_argument("--skip-baselines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = make_swdyn()
    detail_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    print(
        f"dataset=SwDyn A={data.adjacency.shape} X={data.signals.shape} "
        f"R={args.rank} repeats={args.repeats}",
        flush=True,
    )
    for repeat in range(args.repeats):
        for observed_percent in args.observed:
            mask_seed = args.mask_seed_base + 1000 * repeat + observed_percent
            validation_seed = args.validation_seed_base + 1000 * repeat + observed_percent
            mask = symmetric_observation_mask(
                data.adjacency.shape,
                observed_percent / 100.0,
                seed=mask_seed,
            )
            hmask = hidden_mask(mask)
            train_mask, validation_mask = validation_split_mask(
                mask,
                validation_fraction=args.validation_fraction,
                seed=validation_seed,
            )

            best_params: dict[str, dict[str, float]] = {}
            for method in ["DGD", "SDGD"]:
                with contextlib.redirect_stdout(io.StringIO()):
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
                            "repeat": repeat,
                            "mask_seed": mask_seed,
                            "validation_seed": validation_seed,
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
                detail_rows.append(
                    {
                        "dataset": "SwDyn",
                        "observed_percent": observed_percent,
                        "repeat": repeat,
                        "mask_seed": mask_seed,
                        "validation_seed": validation_seed,
                        "rank": args.rank,
                        "validation_fraction": args.validation_fraction,
                        "method": method,
                        "method_rank": args.rank,
                        "selection_metric": "validation_relative_error",
                        **best_params[method],
                        "relative_error": relative_error,
                        "f1": f1,
                        "elapsed_seconds": elapsed,
                    }
                )
                print(
                    f"repeat={repeat} obs={observed_percent} {method} "
                    f"RE={relative_error:.4g} F1={f1:.4g} params={best_params[method]}",
                    flush=True,
                )

            if args.skip_baselines:
                continue

            with contextlib.redirect_stdout(io.StringIO()):
                baseline_results = run_baselines(
                    data,
                    mask,
                    rank=args.rank,
                    max_iterations=args.baseline_iterations,
                    unc_mode=args.unc_mode,
                )
            for method, reconstruction, method_rank in baseline_results:
                relative_error, f1 = evaluate(data, reconstruction, hmask)
                detail_rows.append(
                    {
                        "dataset": "SwDyn",
                        "observed_percent": observed_percent,
                        "repeat": repeat,
                        "mask_seed": mask_seed,
                        "validation_seed": validation_seed,
                        "rank": args.rank,
                        "validation_fraction": args.validation_fraction,
                        "method": method,
                        "method_rank": method_rank,
                        "selection_metric": "",
                        **blank_params(),
                        "relative_error": relative_error,
                        "f1": f1,
                        "elapsed_seconds": "",
                    }
                )
                print(
                    f"repeat={repeat} obs={observed_percent} {method} "
                    f"RE={relative_error:.4g} F1={f1:.4g}",
                    flush=True,
                )

    detail_rows.sort(
        key=lambda row: (
            int(row["observed_percent"]),
            int(row["repeat"]),
            METHOD_ORDER.index(str(row["method"])),
        )
    )
    tuning_rows.sort(
        key=lambda row: (
            int(row["observed_percent"]),
            int(row["repeat"]),
            METHOD_ORDER.index(str(row["method"])),
        )
    )
    summary_rows = summarize(detail_rows)
    write_csv(args.detail_output, detail_rows)
    write_csv(args.summary_output, summary_rows)
    write_csv(args.tuning_output, tuning_rows)
    print(f"wrote {args.detail_output}")
    print(f"wrote {args.summary_output}")
    print(f"wrote {args.tuning_output}")


if __name__ == "__main__":
    main()
