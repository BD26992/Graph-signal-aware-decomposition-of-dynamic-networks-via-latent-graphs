"""Focused USTemp DGD 10% probe over delta, gamma, beta, and mu."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from real_repeated_mask_experiments import load_adjacency_signals
from real_tuned_mask_experiments import evaluate, make_params, reconstruct_dgd, validation_split_mask
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--observed-percent", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--deltas", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0001, 0.001, 0.01])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.001, 0.01])
    parser.add_argument("--mus", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/csv/diagnostics/ustemp_dgd_mu_probe_obs10.csv",
    )
    args = parser.parse_args()

    adjacency, signals = load_adjacency_signals(DATA)
    observed_percent = args.observed_percent
    mask_seed = args.seed + 1000 * args.repeat + observed_percent
    validation_seed = args.seed + 1000 * args.repeat + 1000 + observed_percent
    mask = matlab_mask_type2(adjacency, observed_percent, seed=mask_seed)
    train_mask, validation_mask = validation_split_mask(
        mask,
        validation_fraction=args.validation_fraction,
        seed=validation_seed,
    )
    test_mask = hidden_mask(mask)

    rows: list[dict[str, object]] = []
    grid = [
        (delta, gamma, beta, mu)
        for delta in args.deltas
        for gamma in args.gammas
        for beta in args.betas
        for mu in args.mus
    ]
    best_validation = None
    best_hidden = None

    for idx, (delta, gamma, beta, mu) in enumerate(grid, start=1):
        params_row = {
            "alpha": 1.0,
            "gamma": gamma,
            "beta": beta,
            "delta": delta,
            "delta_a": delta,
            "mu": mu,
            "rho": 0.001,
            "eta_graph": 0.0,
        }
        params = make_params(params_row, rank=3, outer_iterations=args.outer_iterations)
        start = time.perf_counter()
        validation_reconstruction = reconstruct_dgd(adjacency, signals, train_mask, params)
        validation_elapsed = time.perf_counter() - start
        val_re, val_f1 = evaluate(adjacency, validation_reconstruction, validation_mask)

        start = time.perf_counter()
        hidden_reconstruction = reconstruct_dgd(adjacency, signals, mask, params)
        hidden_elapsed = time.perf_counter() - start
        hidden_re, hidden_f1 = evaluate(adjacency, hidden_reconstruction, test_mask)

        row = {
            "dataset": "USTemp",
            "observed_percent": observed_percent,
            "repeat": args.repeat,
            "mask_seed": mask_seed,
            "validation_seed": validation_seed,
            "rank": 3,
            "validation_fraction": args.validation_fraction,
            **params_row,
            "validation_relative_error": val_re,
            "validation_f1": val_f1,
            "hidden_relative_error": hidden_re,
            "hidden_f1": hidden_f1,
            "validation_elapsed_seconds": validation_elapsed,
            "hidden_elapsed_seconds": hidden_elapsed,
        }
        rows.append(row)
        if best_validation is None or val_re < best_validation["validation_relative_error"]:
            best_validation = row
        if best_hidden is None or hidden_re < best_hidden["hidden_relative_error"]:
            best_hidden = row
        write_csv(args.output, rows)
        print(
            f"{idx}/{len(grid)} delta={delta:g} gamma={gamma:g} beta={beta:g} mu={mu:g} "
            f"val={val_re:.4g} hidden={hidden_re:.4g} f1={hidden_f1:.4g}",
            flush=True,
        )

    print("best validation", best_validation)
    print("best hidden", best_hidden)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
