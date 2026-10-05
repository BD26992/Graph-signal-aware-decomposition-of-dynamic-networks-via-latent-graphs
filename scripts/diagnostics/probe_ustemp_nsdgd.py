"""Focused USTemp NSDGD hyperparameter probe at low observation rate."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

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
OUT = ROOT / "results/csv/diagnostics/ustemp_nsdgd_obs10_high_gamma_probe.csv"


def main() -> None:
    adjacency, signals = load_adjacency_signals(DATA)
    del signals

    observed_percent = 10
    repeat = 0
    mask_seed = 1 + 1000 * repeat + observed_percent
    validation_seed = 1 + 1000 * repeat + 1000 + observed_percent
    mask = matlab_mask_type2(adjacency, observed_percent, seed=mask_seed)
    train_mask, validation_mask = validation_split_mask(mask, validation_fraction=0.2, seed=validation_seed)
    test_mask = hidden_mask(mask)

    grid = []
    for gamma in [1.0, 2.0, 5.0, 10.0, 20.0]:
        for beta in [0.0, 1e-3, 1e-2, 1e-1]:
            for eta_graph in [0.0, 1e-4]:
                grid.append((gamma, beta, eta_graph))

    rows = []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for idx, (gamma, beta, eta_graph) in enumerate(grid, start=1):
        row = {
            "alpha": 1.0,
            "gamma": gamma,
            "beta": beta,
            "delta": 0.0,
            "delta_a": 0.0,
            "mu": 0.1,
            "rho": 0.001,
            "eta_graph": eta_graph,
        }
        params = make_params(row, rank=3, outer_iterations=50)
        start = time.perf_counter()
        reconstruction = reconstruct_sdgd(adjacency, train_mask, params)
        elapsed = time.perf_counter() - start
        val_re, val_f1 = evaluate(adjacency, reconstruction, validation_mask)

        final_reconstruction = reconstruct_sdgd(adjacency, mask, params)
        hidden_re, hidden_f1 = evaluate(adjacency, final_reconstruction, test_mask)
        out_row = {
            "dataset": "USTemp",
            "observed_percent": observed_percent,
            "repeat": repeat,
            "mask_seed": mask_seed,
            "validation_seed": validation_seed,
            "rank": 3,
            **row,
            "validation_relative_error": val_re,
            "validation_f1": val_f1,
            "hidden_relative_error": hidden_re,
            "hidden_f1": hidden_f1,
            "elapsed_seconds_validation_fit": elapsed,
        }
        rows.append(out_row)
        with OUT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"{idx}/{len(grid)} gamma={gamma:g} beta={beta:g} eta={eta_graph:g} "
            f"val={val_re:.4g} hidden={hidden_re:.4g} f1={hidden_f1:.4g}",
            flush=True,
        )

    best_val = min(rows, key=lambda item: item["validation_relative_error"])
    best_hidden = min(rows, key=lambda item: item["hidden_relative_error"])
    print("best validation", best_val)
    print("best hidden", best_hidden)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
