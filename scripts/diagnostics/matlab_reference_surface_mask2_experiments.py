"""Run the mislabeled MATLAB ``surface_results_mask2`` reference dataset.

The saved MATLAB file below is the source of the commonly quoted row

``DGD=0.119, NSDGD=0.135, SICA=0.137, CPD=0.181, BTD=0.203, SGL=0.682, UNC=0.873``.

Despite the filename, the saved arrays are not the real SeaSurf data:
``A=(40, 40, 50)`` and ``X=(40, 1000, 50)``. This wrapper deliberately labels
the run as a MATLAB reference/synthetic-style case while reusing the same
Python experiment machinery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import seasurf_paper_experiments as runner


DEFAULT_REFERENCE_DATA = Path(
    "/Users/bishwadeep/Downloads/tensors_decomp-2/experiments/journal/exp4/"
    "surface_results_mask2.mat"
)


def main() -> None:
    argv = sys.argv[1:]
    if "--data" not in argv:
        argv.extend(["--data", str(DEFAULT_REFERENCE_DATA)])
    if "--dataset-name" not in argv:
        argv.extend(["--dataset-name", "MatlabReferenceSurfaceMask2_40x40x50"])
    if "--output" not in argv:
        argv.extend(
            [
                "--output",
                "results/csv/matlab_reference_surface_mask2_python_detail.csv",
            ]
        )
    if "--summary-output" not in argv:
        argv.extend(
            [
                "--summary-output",
                "results/csv/matlab_reference_surface_mask2_python_summary.csv",
            ]
        )
    if "--rank" not in argv:
        argv.extend(["--rank", "3"])
    sys.argv = [sys.argv[0], *argv]
    runner.main()


if __name__ == "__main__":
    main()
