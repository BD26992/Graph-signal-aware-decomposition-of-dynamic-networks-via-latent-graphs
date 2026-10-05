"""Build processed SeaSurf tensors from the raw SeaSurf signal matrix.

The MATLAB preprocessing uses the raw matrix ``R`` from
``seasurfacetemperature.mat``. For each temporal block, half of the samples are
used to build a binary smoothness graph and the alternating half is saved as the
signal tensor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


def smoothness_graph_from_signals(signals: np.ndarray, n_edges: int) -> np.ndarray:
    """Keep the ``n_edges`` node pairs with smallest squared signal distance."""

    n_nodes = signals.shape[0]
    gram = signals @ signals.T
    norms = np.sum(signals * signals, axis=1)
    distances = np.maximum(norms[:, None] + norms[None, :] - 2.0 * gram, 0.0)
    rows, cols = np.triu_indices(n_nodes, k=1)
    weights = distances[rows, cols]
    keep_count = min(n_edges, weights.size)
    keep = np.argpartition(weights, keep_count - 1)[:keep_count]
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    adjacency[rows[keep], cols[keep]] = 1.0
    adjacency[cols[keep], rows[keep]] = 1.0
    return adjacency


def compute_w_matrices(signals: np.ndarray) -> np.ndarray:
    """Compute pairwise squared signal distances for each time slice."""

    n_nodes, _, n_times = signals.shape
    weights = np.empty((n_nodes, n_nodes, n_times), dtype=float)
    for t in range(n_times):
        x_t = signals[:, :, t]
        gram = x_t @ x_t.T
        norms = np.sum(x_t * x_t, axis=1)
        weights[:, :, t] = np.maximum(norms[:, None] + norms[None, :] - 2.0 * gram, 0.0)
    return weights


def build_seasurf(
    raw_path: Path,
    *,
    n_times: int,
    f_split: int = 2,
    edges_per_node: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    mat = sio.loadmat(raw_path)
    raw = np.asarray(mat["R"], dtype=float)
    n_nodes, n_samples = raw.shape
    samples_per_half = n_samples // (f_split * n_times)
    if samples_per_half <= 0:
        raise ValueError(f"n_times={n_times} is too large for {n_samples} samples")

    total_samples = samples_per_half * f_split * n_times
    data = raw[:, n_samples - total_samples :]
    samples_per_block = f_split * samples_per_half
    n_edges = edges_per_node * n_nodes

    adjacency = np.zeros((n_nodes, n_nodes, n_times), dtype=float)
    signals = np.zeros((n_nodes, samples_per_half, n_times), dtype=float)
    for t in range(n_times):
        block = data[:, t * samples_per_block : (t + 1) * samples_per_block]
        train = block[:, 0::2]
        test = block[:, 1::2]
        adjacency[:, :, t] = smoothness_graph_from_signals(train, n_edges)
        signals[:, :, t] = test

    weights = compute_w_matrices(signals)
    metadata = {
        "n_nodes": n_nodes,
        "n_raw_samples": n_samples,
        "n_times": n_times,
        "samples_per_time": samples_per_half,
        "samples_per_block": samples_per_block,
        "total_samples_used": total_samples,
        "dropped_initial_samples": n_samples - total_samples,
        "n_edges_per_slice": n_edges,
    }
    return adjacency, signals, weights, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/surface/seasurfacetemperature.mat"),
    )
    parser.add_argument("--n-times", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/surface_8_data.mat"),
    )
    args = parser.parse_args()

    adjacency, signals, weights, metadata = build_seasurf(args.raw, n_times=args.n_times)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(args.output, {"A": adjacency, "X": signals, "W": weights})
    print(f"wrote {args.output}")
    print(metadata)


if __name__ == "__main__":
    main()
