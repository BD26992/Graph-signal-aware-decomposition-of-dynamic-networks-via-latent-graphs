"""Append or replace manuscript-matched CPD rows in a repeat-detail CSV."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from add_sica_to_experiment_csv import read_rows, summarize
from dynamic_graph_decomposition import fit_cpd_als
from matlab_v5_loader import loadmat_numeric
from swdyn_paper_experiments import (
    f1_score_hidden,
    hidden_mask,
    relative_error_squared,
    symmetric_observation_mask,
    write_csv,
)


def matlab_parameter_matched_cpd_rank(
    n_nodes: int,
    n_times: int,
    proposed_rank: int,
) -> int:
    tgd_parameters = (n_nodes * n_nodes + n_times) * proposed_rank
    return max(1, int(round(tgd_parameters / (2 * n_nodes + n_times + 1))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--mat-file", type=Path, required=True)
    parser.add_argument("--proposed-rank", type=int, default=10)
    parser.add_argument("--cpd-rank", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=["auto", "numpy", "tensorly", "tensorly_masked"],
        default="tensorly",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.detail_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.detail_csv}")
    if args.overwrite:
        rows = [row for row in rows if row["method"] != "CPD"]

    existing_keys = {
        (row["method"], int(float(row["observed_percent"])), int(row["repeat"]))
        for row in rows
    }
    adjacency = np.asarray(
        loadmat_numeric(args.mat_file, names={"A"})["A"],
        dtype=float,
    )
    n_nodes, _, n_times = adjacency.shape
    cpd_rank = args.cpd_rank or matlab_parameter_matched_cpd_rank(
        n_nodes,
        n_times,
        args.proposed_rank,
    )

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
        key = ("CPD", observed_percent, repeat)
        if key in existing_keys:
            continue
        mask = symmetric_observation_mask(
            adjacency.shape,
            observed_percent / 100.0,
            seed=seed,
        )
        hmask = hidden_mask(mask)
        start = time.perf_counter()
        result = fit_cpd_als(
            adjacency,
            rank=cpd_rank,
            mask=mask,
            max_iterations=args.max_iterations,
            random_state=args.random_state,
            backend=args.backend,
        )
        elapsed = time.perf_counter() - start
        additions.append(
            {
                "dataset": rows[0]["dataset"],
                "method": "CPD",
                "observed_percent": observed_percent,
                "repeat": repeat,
                "seed": seed,
                "rank": cpd_rank,
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
            "CPD:",
            f"backend={result.name}",
            f"rank={cpd_rank}",
            f"obs={observed_percent}",
            f"repeat={repeat}",
            f"RE={additions[-1]['relative_error']:.4f}",
            f"F1={additions[-1]['f1']:.4f}",
            flush=True,
        )

    merged = [dict(row) for row in rows] + additions
    write_csv(args.detail_csv, merged)
    write_csv(args.summary_csv, summarize(merged))
    print(f"added {len(additions)} CPD rows")


if __name__ == "__main__":
    main()
