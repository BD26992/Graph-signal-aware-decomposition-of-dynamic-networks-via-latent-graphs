"""Append SICA baseline rows to an existing repeat-detail experiment CSV."""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from dynamic_graph_decomposition import fit_sica_baseline
from matlab_v5_loader import loadmat_numeric
from swdyn_paper_experiments import (
    f1_score_hidden,
    hidden_mask,
    relative_error_squared,
    symmetric_observation_mask,
    write_csv,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(float(row["observed_percent"])))].append(row)

    summary = []
    dataset = str(rows[0]["dataset"]) if rows else ""
    for (method, observed_percent), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        relative_errors = np.array([float(row["relative_error"]) for row in group])
        f1_values = np.array([float(row["f1"]) for row in group])
        elapsed = np.array([float(row["elapsed_seconds"]) for row in group])
        first = group[0]
        summary.append(
            {
                "dataset": dataset,
                "method": method,
                "observed_percent": observed_percent,
                "n_runs": len(group),
                "rank": first.get("rank", ""),
                "outer_iterations": first.get("outer_iterations", ""),
                "relative_error_mean": float(np.mean(relative_errors)),
                "relative_error_std": float(np.std(relative_errors, ddof=1))
                if len(relative_errors) > 1
                else 0.0,
                "f1_mean": float(np.mean(f1_values)),
                "f1_std": float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else 0.0,
                "elapsed_seconds_mean": float(np.mean(elapsed)),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--mat-file", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing SICA rows instead of leaving them untouched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.detail_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.detail_csv}")

    if args.overwrite:
        rows = [row for row in rows if row["method"] != "SICA"]

    existing_keys = {
        (row["method"], int(float(row["observed_percent"])), int(row["repeat"]))
        for row in rows
    }
    mat = loadmat_numeric(args.mat_file, names={"A"})
    adjacency = np.asarray(mat["A"], dtype=float)
    rank_by_key = {
        (int(float(row["observed_percent"])), int(row["repeat"])): int(float(row["rank"]))
        for row in rows
        if row.get("rank") and row["method"] == "SICA"
    }

    additions: list[dict[str, object]] = []
    seeds = sorted(
        {
            (
                int(float(row["observed_percent"])),
                int(row["repeat"]),
                int(row["seed"]),
            )
            for row in rows
        }
    )
    for observed_percent, repeat, seed in seeds:
        key = ("SICA", observed_percent, repeat)
        if key in existing_keys:
            continue
        mask = symmetric_observation_mask(
            adjacency.shape,
            observed_percent / 100.0,
            seed=seed,
        )
        hmask = hidden_mask(mask)
        rank = rank_by_key.get((observed_percent, repeat), args.rank)
        start = time.perf_counter()
        result = fit_sica_baseline(
            adjacency,
            rank=rank,
            mask=mask,
            max_iterations=args.max_iterations,
            random_state=args.random_state,
        )
        elapsed = time.perf_counter() - start
        additions.append(
            {
                "dataset": rows[0]["dataset"],
                "method": "SICA",
                "observed_percent": observed_percent,
                "repeat": repeat,
                "seed": seed,
                "rank": rank,
                "outer_iterations": "",
                "delta": "",
                "relative_error": relative_error_squared(
                    adjacency,
                    result.reconstruction,
                    hmask,
                ),
                "f1": f1_score_hidden(adjacency, result.reconstruction, hmask),
                "elapsed_seconds": elapsed,
            }
        )
        print(
            "SICA:",
            f"obs={observed_percent}",
            f"repeat={repeat}",
            f"RE={additions[-1]['relative_error']:.4f}",
            f"F1={additions[-1]['f1']:.4f}",
            flush=True,
        )

    merged = [dict(row) for row in rows] + additions
    write_csv(args.detail_csv, merged)
    write_csv(args.summary_csv, summarize(merged))
    print(f"added {len(additions)} SICA rows")


if __name__ == "__main__":
    main()
