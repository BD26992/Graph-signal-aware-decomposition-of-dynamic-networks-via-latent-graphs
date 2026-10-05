"""Compare zero-filled UNC with explicitly masked UNC on stored experiment splits."""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from convergence_observed_curves import REAL_DATASETS
from dynamic_graph_decomposition import fit_unconstrained_observation_als
from seasurf_paper_experiments import load_matlab_adjacency_signals, matlab_mask_type2
from swdyn_paper_experiments import (
    f1_score_hidden,
    hidden_mask,
    make_swdyn,
    relative_error_squared,
    symmetric_observation_mask,
)


DATASETS = [
    {
        "key": "swdyn",
        "label": "SwDyn",
        "detail": Path("results/csv/swdyn_alpha1_5runs_with_baselines_detail_R2_obs10_20_30_40_50_60_70_80_90_outer50.csv"),
        "summary": Path("results/csv/swdyn_alpha1_5runs_with_baselines_summary_R2_obs10_20_30_40_50_60_70_80_90_outer50.csv"),
    },
    {
        "key": "seasurface",
        "label": "SeaSurf",
        "detail": Path("results/csv/seasurface_alpha1_5runs_with_baselines_detail_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv"),
        "summary": Path("results/csv/seasurface_alpha1_5runs_with_baselines_summary_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv"),
    },
    {
        "key": "ustemp",
        "label": "USTemp",
        "detail": Path("results/csv/ustemp_alpha1_5runs_with_baselines_detail_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv"),
        "summary": Path("results/csv/ustemp_alpha1_5runs_with_baselines_summary_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv"),
    },
    {
        "key": "contact",
        "label": "Contact",
        "detail": Path("results/csv/contact_alpha1_5runs_baselines_only_detail_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv"),
        "summary": Path("results/csv/contact_alpha1_5runs_baselines_only_summary_R3_obs10_20_30_40_50_60_70_80_90_outer50.csv"),
    },
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_dataset(key: str) -> tuple[np.ndarray, int]:
    if key == "swdyn":
        data = make_swdyn()
        return data.adjacency, 0
    spec = REAL_DATASETS[key]
    adjacency, _signals = load_matlab_adjacency_signals(Path(spec["path"]))
    return adjacency, int(spec["mask_seed"])


def make_mask(key: str, adjacency: np.ndarray, observed_percent: int, mask_seed: int) -> np.ndarray:
    if key == "swdyn":
        return symmetric_observation_mask(adjacency.shape, observed_percent / 100.0, seed=mask_seed)
    return matlab_mask_type2(adjacency, observed_percent, seed=mask_seed)


def summarize(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in detail_rows:
        groups.setdefault((str(row["dataset"]), int(row["observed_percent"])), []).append(row)
    summary_rows: list[dict[str, object]] = []
    for (dataset, observed_percent), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        re_values = [float(row["relative_error"]) for row in rows]
        f1_values = [float(row["f1"]) for row in rows]
        elapsed_values = [float(row["elapsed_seconds"]) for row in rows]
        summary_rows.append(
            {
                "dataset": dataset,
                "observed_percent": observed_percent,
                "rank": rows[0]["rank"],
                "method": "UNC_masked",
                "method_rank": rows[0]["method_rank"],
                "n_runs": len(rows),
                "relative_error_mean": statistics.fmean(re_values),
                "relative_error_std": statistics.stdev(re_values) if len(rows) > 1 else 0.0,
                "relative_error_min": min(re_values),
                "relative_error_max": max(re_values),
                "f1_mean": statistics.fmean(f1_values),
                "f1_std": statistics.stdev(f1_values) if len(rows) > 1 else 0.0,
                "f1_min": min(f1_values),
                "f1_max": max(f1_values),
                "elapsed_seconds_mean": statistics.fmean(elapsed_values),
            }
        )
    return summary_rows


def compare_with_zero_filled(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_dataset = {spec["label"]: spec["summary"] for spec in DATASETS}
    rows: list[dict[str, object]] = []
    for masked in summary_rows:
        dataset = str(masked["dataset"])
        observed_percent = int(masked["observed_percent"])
        zero_summary = pd.read_csv(by_dataset[dataset])
        zero = zero_summary[
            (zero_summary["method"] == "UNC")
            & (zero_summary["observed_percent"] == observed_percent)
        ].iloc[0]
        zero_re = float(zero["relative_error_mean"])
        zero_f1 = float(zero["f1_mean"])
        masked_re = float(masked["relative_error_mean"])
        masked_f1 = float(masked["f1_mean"])
        rows.append(
            {
                "dataset": dataset,
                "observed_percent": observed_percent,
                "rank": masked["rank"],
                "n_runs": masked["n_runs"],
                "zero_filled_relative_error_mean": zero_re,
                "masked_relative_error_mean": masked_re,
                "relative_error_change_masked_minus_zero": masked_re - zero_re,
                "relative_error_percent_change": 100.0 * (masked_re - zero_re) / zero_re,
                "zero_filled_f1_mean": zero_f1,
                "masked_f1_mean": masked_f1,
                "f1_change_masked_minus_zero": masked_f1 - zero_f1,
            }
        )
    return rows


def main() -> None:
    detail_rows: list[dict[str, object]] = []
    for spec in DATASETS:
        key = str(spec["key"])
        label = str(spec["label"])
        adjacency, _base_seed = load_dataset(key)
        detail = pd.read_csv(spec["detail"])
        unc_rows = detail[detail["method"] == "UNC"].copy()
        for row in unc_rows.itertuples(index=False):
            observed_percent = int(getattr(row, "observed_percent"))
            repeat = int(getattr(row, "repeat"))
            mask_seed = int(getattr(row, "mask_seed"))
            rank = int(getattr(row, "rank"))
            method_rank = int(getattr(row, "method_rank"))
            mask = make_mask(key, adjacency, observed_percent, mask_seed)
            hmask = hidden_mask(mask)
            start = time.perf_counter()
            result = fit_unconstrained_observation_als(
                adjacency,
                rank=method_rank,
                mask=mask,
                max_iterations=30,
                random_state=mask_seed,
            )
            elapsed = time.perf_counter() - start
            relative_error = relative_error_squared(adjacency, result.reconstruction, hmask)
            f1 = f1_score_hidden(adjacency, result.reconstruction, hmask)
            detail_rows.append(
                {
                    "dataset": label,
                    "observed_percent": observed_percent,
                    "repeat": repeat,
                    "mask_seed": mask_seed,
                    "rank": rank,
                    "method": "UNC_masked",
                    "method_rank": method_rank,
                    "relative_error": relative_error,
                    "f1": f1,
                    "elapsed_seconds": elapsed,
                }
            )
            print(
                f"{label} obs={observed_percent} repeat={repeat} "
                f"UNC_masked RE={relative_error:.4f} F1={f1:.4f}",
                flush=True,
            )

    summary_rows = summarize(detail_rows)
    comparison_rows = compare_with_zero_filled(summary_rows)
    write_csv(Path("results/csv/diagnostics/unc_masked_detail_all_datasets.csv"), detail_rows)
    write_csv(Path("results/csv/diagnostics/unc_masked_summary_all_datasets.csv"), summary_rows)
    write_csv(Path("results/csv/diagnostics/unc_masked_vs_zero_filled_comparison_all_datasets.csv"), comparison_rows)
    print("wrote results/csv/diagnostics/unc_masked_detail_all_datasets.csv")
    print("wrote results/csv/diagnostics/unc_masked_summary_all_datasets.csv")
    print("wrote results/csv/diagnostics/unc_masked_vs_zero_filled_comparison_all_datasets.csv")


if __name__ == "__main__":
    main()
