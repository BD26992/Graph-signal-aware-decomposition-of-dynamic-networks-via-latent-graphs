"""Compare Python and MATLAB outputs for one Contact SGL slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio


BASE = Path("results/matlab_compare")
PYTHON_PATH = BASE / "contact_gsp_slice_python_output.mat"
MATLAB_PATH = BASE / "contact_gsp_slice_matlab_output.mat"


def stats(name: str, graph: np.ndarray) -> None:
    graph = np.asarray(graph, dtype=float)
    positive = graph[graph > 0]
    print(
        f"{name}: shape={graph.shape} min={graph.min():.6g} "
        f"max={graph.max():.6g} mean={graph.mean():.6g} "
        f"density={(graph > 1e-5).mean():.6g} "
        f"positive_mean={(positive.mean() if positive.size else 0.0):.6g}"
    )


def main() -> None:
    if not PYTHON_PATH.exists():
        raise SystemExit(f"Missing Python output: {PYTHON_PATH}")
    if not MATLAB_PATH.exists():
        raise SystemExit(
            f"Missing MATLAB output: {MATLAB_PATH}\n"
            "Run compare_gsp_log_degrees_contact.m in MATLAB first."
        )

    py = sio.loadmat(PYTHON_PATH)
    ml = sio.loadmat(MATLAB_PATH)
    py_raw = np.asarray(py["A_hat"], dtype=float)
    ml_raw = np.asarray(ml["A_hat"], dtype=float)
    py_norm = np.asarray(py["A_hat_normalized"], dtype=float)
    ml_norm = np.asarray(ml["A_hat_normalized"], dtype=float)

    stats("python raw", py_raw)
    stats("matlab raw", ml_raw)
    stats("python normalized", py_norm)
    stats("matlab normalized", ml_norm)

    for label, left, right in [
        ("raw", py_raw, ml_raw),
        ("normalized", py_norm, ml_norm),
    ]:
        diff = left - right
        denom = max(np.linalg.norm(right, "fro"), 1e-12)
        print(
            f"{label} diff: max_abs={np.max(np.abs(diff)):.6g} "
            f"fro_rel={np.linalg.norm(diff, 'fro') / denom:.6g}"
        )


if __name__ == "__main__":
    main()
