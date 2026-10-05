"""Repeat Appendix E graph metrics over multiple DGD reruns."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

from contact_component_report import fit_contact
from seasurface_atlas_component_report import (
    fit_seasurface,
    graph_metrics,
    jaccard_rows,
    top_k_adjacency,
)
from ustemp_component_report import fit_ustemp


FitFn = Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


DATASETS: dict[str, FitFn] = {
    "SeaSurf": fit_seasurface,
    "USTemp": fit_ustemp,
    "Contact": fit_contact,
}


METRIC_COLUMNS = [
    "average_clustering_coefficient",
    "modularity",
    "assortativity",
    "degree_mean",
    "degree_std",
    "degree_max",
    "strength_mean",
    "strength_max",
]


JACCARD_COLUMNS = ["intersection", "union", "jaccard", "overlap_over_min"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: list[dict[str, object]],
    group_columns: list[str],
    value_columns: list[str],
    *,
    repeats: int,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[column] for column in group_columns)
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        out = {column: value for column, value in zip(group_columns, key)}
        out["n_runs"] = len(group)
        out["requested_repeats"] = repeats
        for column in value_columns:
            values = np.array([float(row[column]) for row in group], dtype=float)
            out[f"{column}_mean"] = float(np.nanmean(values))
            out[f"{column}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            out[f"{column}_var"] = float(np.nanvar(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(out)
    return summary_rows


def run_repeats(
    *,
    datasets: list[str],
    repeats: int,
    observed_percent: int,
    outer_iterations: int,
    top_edges: int,
    seed_base: int,
    mask_seed_base: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    jaccard_detail_rows: list[dict[str, object]] = []

    for dataset in datasets:
        fit_fn = DATASETS[dataset]
        for repeat in range(repeats):
            seed = seed_base + repeat
            mask_seed = mask_seed_base + 1000 * repeat
            start = time.perf_counter()
            fit_kwargs = {
                "observed_percent": observed_percent,
                "mask_seed": mask_seed,
                "outer_iterations": outer_iterations,
                "seed": seed,
            }
            if dataset == "SeaSurf":
                fit_kwargs["beta_factor"] = 1.0
            _, latent_graphs, _, _ = fit_fn(**fit_kwargs)
            elapsed = time.perf_counter() - start
            binary_components: list[np.ndarray] = []
            for component in range(1, 4):
                binary, weighted = top_k_adjacency(latent_graphs[component - 1], top_edges)
                binary_components.append(binary)
                row = graph_metrics(
                    dataset,
                    component,
                    binary,
                    weighted,
                    observed_percent=observed_percent,
                    top_edges=top_edges,
                )
                row["repeat"] = repeat
                row["seed"] = seed
                row["mask_seed"] = mask_seed
                row["elapsed_seconds"] = elapsed
                metric_rows.append(row)

            for row in jaccard_rows(
                dataset,
                binary_components,
                observed_percent=observed_percent,
                top_edges=top_edges,
            ):
                row["repeat"] = repeat
                row["seed"] = seed
                row["mask_seed"] = mask_seed
                row["elapsed_seconds"] = elapsed
                jaccard_detail_rows.append(row)

            print(
                f"{dataset} repeat {repeat + 1}/{repeats}: "
                f"{elapsed:.2f}s",
                flush=True,
            )
    return metric_rows, jaccard_detail_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--observed-percent", type=int, default=90)
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--top-edges", type=int, default=150)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--mask-seed-base", type=int, default=91)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/csv/Graph data/repeats"),
    )
    args = parser.parse_args()

    metric_rows, j_rows = run_repeats(
        datasets=args.datasets,
        repeats=args.repeats,
        observed_percent=args.observed_percent,
        outer_iterations=args.outer_iterations,
        top_edges=args.top_edges,
        seed_base=args.seed_base,
        mask_seed_base=args.mask_seed_base,
    )

    prefix = (
        f"graph_metrics_R3_obs{args.observed_percent}_outer{args.outer_iterations}"
        f"_top{args.top_edges}_{args.repeats}runs"
    )
    metric_detail = args.output_dir / f"{prefix}_component_detail.csv"
    jaccard_detail = args.output_dir / f"{prefix}_jaccard_detail.csv"
    metric_summary = args.output_dir / f"{prefix}_component_summary.csv"
    jaccard_summary = args.output_dir / f"{prefix}_jaccard_summary.csv"

    write_csv(metric_detail, metric_rows)
    write_csv(jaccard_detail, j_rows)
    write_csv(
        metric_summary,
        summarize(
            metric_rows,
            ["dataset", "observed_percent", "rank", "component", "top_edges"],
            METRIC_COLUMNS,
            repeats=args.repeats,
        ),
    )
    write_csv(
        jaccard_summary,
        summarize(
            j_rows,
            [
                "dataset",
                "observed_percent",
                "rank",
                "component_i",
                "component_j",
                "top_edges",
            ],
            JACCARD_COLUMNS,
            repeats=args.repeats,
        ),
    )

    print(f"wrote {metric_detail}")
    print(f"wrote {jaccard_detail}")
    print(f"wrote {metric_summary}")
    print(f"wrote {jaccard_summary}")


if __name__ == "__main__":
    main()
