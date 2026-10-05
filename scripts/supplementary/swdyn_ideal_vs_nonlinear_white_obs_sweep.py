"""Observed-percentage sweep for ideal SwDyn and nonlinear-white SwDyn.

This compares:

- ideal: linear dynamic topology with graph-smooth signals
- nonlinear_white: sinusoidal dynamic topology with white-noise signals

For each observed percentage, the script stores per-iteration observed and
unobserved relative errors and plots the final-iteration errors.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swdyn_factorial_robustness_experiment import (
    VARIANT_BY_SLUG,
    VariantConfig,
    run_variant,
)


@dataclass(frozen=True)
class SweepJob:
    observed_percent: int
    variant_slug: str
    args: argparse.Namespace


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_job(job: SweepJob) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    nonlinear, white, _slug, label = VARIANT_BY_SLUG[job.variant_slug]
    local_args = copy.copy(job.args)
    local_args.observed_percent = job.observed_percent
    local_args.mask_seed = job.args.mask_seed + job.observed_percent
    config = VariantConfig(
        nonlinear=nonlinear,
        white_signals=white,
        slug=job.variant_slug,
        label="Ideal" if job.variant_slug == "linear_smooth" else label,
        args=local_args,
    )
    _slug, metrics, correlations, stats = run_variant(config)
    return metrics, correlations, stats


def final_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[tuple[str, int, float], dict[str, object]] = {}
    for row in rows:
        key = (str(row["variant"]), int(row["observed_percent"]), float(row["delta"]))
        if key not in best or int(row["iteration"]) > int(best[key]["iteration"]):
            best[key] = row
    return [
        best[key]
        for key in sorted(best, key=lambda item: (item[0], item[1], item[2]))
    ]


def plot_error(
    rows: list[dict[str, object]],
    output: Path,
    *,
    metric: str,
    ylabel: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = {
        "linear_smooth": "Ideal",
        "nonlinear_white": "Nonlinear + white noise",
    }
    colors = {
        "linear_smooth": "#1f77b4",
        "nonlinear_white": "#d62728",
    }
    markers = {
        "linear_smooth": "o",
        "nonlinear_white": "s",
    }
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for slug in ["linear_smooth", "nonlinear_white"]:
        selected = sorted(
            [row for row in rows if row["variant"] == slug],
            key=lambda row: int(row["observed_percent"]),
        )
        ax.plot(
            [int(row["observed_percent"]) for row in selected],
            [float(row[metric]) for row in selected],
            marker=markers[slug],
            markersize=5.2,
            linewidth=1.7,
            color=colors[slug],
            label=labels[slug],
        )
    ax.set_xlabel("Percentage of observed data")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(10, 100, 10))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_error_by_delta(
    rows: list[dict[str, object]],
    output: Path,
    *,
    variant: str,
    metric: str,
    ylabel: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    variant_rows = [row for row in rows if row["variant"] == variant]
    deltas = sorted({float(row["delta"]) for row in variant_rows})
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for idx, delta in enumerate(deltas):
        selected = sorted(
            [row for row in variant_rows if np.isclose(float(row["delta"]), delta)],
            key=lambda row: int(row["observed_percent"]),
        )
        ax.plot(
            [int(row["observed_percent"]) for row in selected],
            [float(row[metric]) for row in selected],
            marker=markers[idx % len(markers)],
            markersize=5.0,
            linewidth=1.5,
            label=fr"$\delta={delta:g}$",
        )
    ax.set_title("SwDyn", fontsize=20)
    ax.set_xlabel("Percentage of observed data", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xticks([10, 30, 50, 70, 90])
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(0.98, 0.68), fontsize=16)
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=int, nargs="+", default=list(range(10, 100, 10)))
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-nodes", type=int, default=40)
    parser.add_argument("--n-signals", type=int, default=1000)
    parser.add_argument("--n-times", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mask-seed", type=int, default=790)
    parser.add_argument("--p4-in", type=float, default=0.65)
    parser.add_argument("--p4-out", type=float, default=0.06)
    parser.add_argument("--signal-smoothness", type=float, default=10.0)
    parser.add_argument("--nonlinear-scale", type=float, default=1.0)
    parser.add_argument("--nonlinear-power", type=float, default=1.0)
    parser.add_argument(
        "--nonlinear-power-target",
        choices=["mixture", "component"],
        default="mixture",
    )
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--delta", type=float, default=10.0)
    parser.add_argument("--delta-grid", type=float, nargs="+", default=None)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=0.001)
    parser.add_argument("--eta-graph", type=float, default=0.0)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/csv/swdyn_ideal_vs_nonlinear_white_obs_sweep_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/csv/swdyn_ideal_vs_nonlinear_white_obs_sweep_summary.csv"),
    )
    parser.add_argument(
        "--correlation-output",
        type=Path,
        default=Path("results/csv/swdyn_ideal_vs_nonlinear_white_obs_sweep_correlations.csv"),
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("results/csv/swdyn_ideal_vs_nonlinear_white_obs_sweep_component_stats.csv"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results/plots/swdyn_ideal_vs_nonlinear_white_obs_sweep"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    delta_values = args.delta_grid if args.delta_grid is not None else [args.delta]
    jobs = []
    for observed in args.observed:
        for variant in ["linear_smooth", "nonlinear_white"]:
            for delta in delta_values:
                local_args = copy.copy(args)
                local_args.delta = delta
                jobs.append(SweepJob(observed_percent=observed, variant_slug=variant, args=local_args))
    if args.workers == 1:
        outputs = [run_job(job) for job in jobs]
    else:
        with Pool(processes=min(args.workers, len(jobs))) as pool:
            outputs = pool.map(run_job, jobs)

    metrics: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []
    stats: list[dict[str, object]] = []
    for metric_rows, correlation_rows, stat_rows in outputs:
        metrics.extend(metric_rows)
        correlations.extend(correlation_rows)
        stats.extend(stat_rows)

    metrics.sort(
        key=lambda row: (
            str(row["variant"]),
            int(row["observed_percent"]),
            int(row["iteration"]),
        )
    )
    summary = final_rows(metrics)
    correlations.sort(
        key=lambda row: (
            str(row["variant"]),
            int(row.get("observed_percent", 0)),
            int(row["true_component"]),
            int(row["recovered_component"]),
        )
    )
    stats.sort(
        key=lambda row: (
            str(row["variant"]),
            int(row.get("observed_percent", 0)),
            int(row["true_component"]),
        )
    )

    write_csv(args.metrics_output, metrics)
    write_csv(args.summary_output, summary)
    write_csv(args.correlation_output, correlations)
    write_csv(args.stats_output, stats)

    plot_error(
        summary,
        args.plot_dir / "observed_relative_error.pdf",
        metric="observed_relative_error",
        ylabel="Observed relative error",
    )
    plot_error(
        summary,
        args.plot_dir / "unobserved_relative_error.pdf",
        metric="unobserved_relative_error",
        ylabel="Unobserved relative error",
    )
    if len(delta_values) > 1:
        for variant in ["linear_smooth", "nonlinear_white"]:
            plot_error_by_delta(
                summary,
                args.plot_dir / f"{variant}_observed_relative_error_by_delta.pdf",
                variant=variant,
                metric="observed_relative_error",
                ylabel="Observed relative error",
            )
            plot_error_by_delta(
                summary,
                args.plot_dir / f"{variant}_unobserved_relative_error_by_delta.pdf",
                variant=variant,
                metric="unobserved_relative_error",
                ylabel="Unobserved relative error",
            )
    print(f"wrote {args.metrics_output}")
    print(f"wrote {args.summary_output}")
    print(f"wrote {args.plot_dir / 'observed_relative_error.pdf'}")
    print(f"wrote {args.plot_dir / 'unobserved_relative_error.pdf'}")


if __name__ == "__main__":
    main()
