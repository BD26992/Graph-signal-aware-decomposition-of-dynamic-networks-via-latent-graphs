"""Run repeated real-data train/validation/test mask experiments."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import statistics
import time
from pathlib import Path

import numpy as np

from real_tuned_mask_experiments import (
    METHOD_ORDER,
    evaluate,
    make_params,
    reconstruct_dgd,
    reconstruct_sdgd,
    run_baselines,
    tune_method,
    validation_split_mask,
)
from seasurf_paper_experiments import load_matlab_adjacency_signals, matlab_mask_type2
from swdyn_paper_experiments import hidden_mask, symmetric_observation_mask


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
        "delta_a": "",
        "mu": "",
        "rho": "",
        "eta_graph": "",
    }


def load_adjacency_signals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return np.asarray(data["adjacency"], dtype=float), np.asarray(data["signals"], dtype=float)
    return load_matlab_adjacency_signals(path)


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
                "dataset": rows[0]["dataset"],
                "data_source": rows[0]["data_source"],
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
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--detail-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--tuning-output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--observed", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60, 70, 80, 90])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.001, 0.01])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0001, 0.001])
    parser.add_argument("--deltas", type=float, nargs="+", default=[10.0, 50.0, 100.0])
    parser.add_argument("--mus", type=float, nargs="+", default=[0.1])
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.001])
    parser.add_argument("--eta-graphs", type=float, nargs="+", default=[0.0])
    parser.add_argument("--sgl-a", type=float, default=2.85e3)
    parser.add_argument("--sgl-b", type=float, default=0.1)
    parser.add_argument("--sgl-iterations", type=int, default=300)
    parser.add_argument("--cpd-iterations", type=int, default=15)
    parser.add_argument("--btd-iterations", type=int, default=15)
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
    parser.add_argument("--only-baselines", action="store_true")
    parser.add_argument(
        "--mask-mode",
        choices=["matlab_type2", "random"],
        default="matlab_type2",
        help="Use MATLAB edge/non-edge mask or a random symmetric mask over all pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adjacency, signals = load_adjacency_signals(args.data)
    detail_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    print(
        f"dataset={args.dataset_name} data={args.data} "
        f"A={adjacency.shape} X={signals.shape} R={args.rank} repeats={args.repeats}",
        flush=True,
    )

    for repeat in range(args.repeats):
        for observed_percent in args.observed:
            mask_seed = args.seed + 1000 * repeat + observed_percent
            validation_seed = args.seed + 1000 * repeat + 1000 + observed_percent
            if args.mask_mode == "random":
                mask = symmetric_observation_mask(
                    adjacency.shape,
                    observed_percent / 100.0,
                    seed=mask_seed,
                )
            else:
                mask = matlab_mask_type2(adjacency, observed_percent, seed=mask_seed)
            hmask = hidden_mask(mask)
            train_mask, validation_mask = validation_split_mask(
                mask,
                validation_fraction=args.validation_fraction,
                seed=validation_seed,
            )

            if not args.only_baselines:
                best_params: dict[str, dict[str, float]] = {}
                for method in ["DGD", "SDGD"]:
                    with contextlib.redirect_stdout(io.StringIO()):
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
                                "repeat": repeat,
                                "mask_seed": mask_seed,
                                "validation_seed": validation_seed,
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
                    detail_rows.append(
                        {
                            "dataset": args.dataset_name,
                            "data_source": str(args.data),
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
                    adjacency,
                    signals,
                    mask,
                    rank=args.rank,
                    cpd_iterations=args.cpd_iterations,
                    btd_iterations=args.btd_iterations,
                    sgl_iterations=args.sgl_iterations,
                    sgl_a=args.sgl_a,
                    sgl_b=args.sgl_b,
                    seed=mask_seed,
                    unc_mode=args.unc_mode,
                )
            for method, reconstruction, method_rank in baseline_results:
                relative_error, f1 = evaluate(adjacency, reconstruction, hmask)
                detail_rows.append(
                    {
                        "dataset": args.dataset_name,
                        "data_source": str(args.data),
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
