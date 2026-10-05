"""Export the saved MATLAB SeaSurf Figure-8 reference values to CSV.

The MATLAB experiment stores the SeaSurf mask-type-2 results in
``surface_results_mask2.mat`` and the SICA/ICA curve separately in
``ICA_surface_error.mat``. This script converts those arrays to a tidy CSV
with the paper method names.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio


DEFAULT_RESULTS = Path(
    "/Users/bishwadeep/Downloads/tensors_decomp-2/experiments/journal/exp4/"
    "surface_results_mask2.mat"
)
DEFAULT_ICA = Path(
    "/Users/bishwadeep/Downloads/tensors_decomp-2/experiments/journal/exp4/"
    "ICA_surface_error.mat"
)

MATLAB_METHODS = ["UNC", "CPD", "BTD", "SGL", "DGD", "NSDGD"]
PAPER_ORDER = ["DGD", "NSDGD", "SICA", "BTD", "CPD", "SGL", "UNC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--ica-results", type=Path, default=DEFAULT_ICA)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/csv/exp8_seasurf_matlab_reference_mask2.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    surface = sio.loadmat(args.surface_results)
    ica = sio.loadmat(args.ica_results)

    observed = np.asarray(surface["obs_d"]).reshape(-1).astype(int)
    hidden_re = np.asarray(surface["herrT"], dtype=float)
    hidden_f1 = np.asarray(surface["hfscT"], dtype=float)
    sica_re = np.asarray(ica["herrT"], dtype=float).reshape(-1)

    rows = []
    for obs_idx, observed_percent in enumerate(observed):
        by_method = {}
        for method_idx, method in enumerate(MATLAB_METHODS):
            by_method[method] = {
                "relative_error": float(hidden_re[obs_idx, method_idx]),
                "f1": float(hidden_f1[obs_idx, method_idx]),
            }
        by_method["SICA"] = {
            "relative_error": float(sica_re[obs_idx]),
            "f1": "",
        }
        for method in PAPER_ORDER:
            rows.append(
                {
                    "dataset": "SeaSurf",
                    "source": "matlab_reference",
                    "mask_type": 2,
                    "rank": 10,
                    "observed_percent": int(observed_percent),
                    "method": method,
                    "relative_error": by_method[method]["relative_error"],
                    "f1": by_method[method]["f1"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
