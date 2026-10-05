"""Construct and run a higher-T SeaSurf dataset from the raw MATLAB signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

from convergence_observed_curves import REAL_DATASETS
from dgd_rank_sweep_observed_error import params_for_rank
from dynamic_graph_decomposition import DynamicGraphDecomposer
from seasurf_paper_experiments import matlab_mask_type2
from swdyn_paper_experiments import f1_score_hidden, hidden_mask, relative_error_squared


RAW_SURFACE = Path(
    "/Users/bishwadeep/Downloads/tensors_decomp-2/real_data/surface/"
    "seasurfacetemperature.mat"
)


def smoothness_graph_from_signals(signals: np.ndarray, n_edges: int) -> np.ndarray:
    """Keep edges with the smallest squared signal differences."""

    n_nodes = signals.shape[0]
    gram = signals @ signals.T
    norms = np.sum(signals * signals, axis=1)
    distances = norms[:, None] + norms[None, :] - 2.0 * gram
    distances = np.maximum(distances, 0.0)
    rows, cols = np.triu_indices(n_nodes, k=1)
    weights = distances[rows, cols]
    keep_count = min(n_edges, weights.size)
    keep = np.argpartition(weights, keep_count - 1)[:keep_count]
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    adjacency[rows[keep], cols[keep]] = 1.0
    adjacency[cols[keep], rows[keep]] = 1.0
    return adjacency


def construct_seasurface_dataset(
    *,
    n_times: int,
    source: Path = RAW_SURFACE,
    f_split: int = 2,
    edges_per_node: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    mat = sio.loadmat(source)
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

    metadata = {
        "n_nodes": n_nodes,
        "n_raw_samples": n_samples,
        "n_times": n_times,
        "samples_per_half": samples_per_half,
        "samples_per_block": samples_per_block,
        "total_samples_used": total_samples,
        "dropped_initial_samples": n_samples - total_samples,
        "n_edges_per_slice": n_edges,
    }
    return adjacency, signals, metadata


def run_dgd(
    adjacency: np.ndarray,
    signals: np.ndarray,
    *,
    observed_percent: int,
    outer_iterations: int,
    seed: int,
    mask_seed: int,
) -> dict[str, object]:
    spec = dict(REAL_DATASETS["seasurface"])
    spec["mask_seed"] = mask_seed
    mask = matlab_mask_type2(adjacency, observed_percent, seed=mask_seed)
    n_nodes, n_signals, _ = signals.shape
    scaled_signals = signals / np.sqrt(n_nodes * n_signals)
    params = params_for_rank(spec, rank=3, seed=seed)
    params.outer_iterations = outer_iterations
    solver = DynamicGraphDecomposer(params)
    result = solver.fit(adjacency, scaled_signals, mask=mask)
    reconstruction = solver.reconstruct(result.latent_graphs, result.temporal_signatures)
    hmask = hidden_mask(mask)
    metrics = {
        "observed_relative_error": relative_error_squared(adjacency, reconstruction, mask),
        "hidden_relative_error": relative_error_squared(adjacency, reconstruction, hmask),
        "observed_f1": f1_score_hidden(adjacency, reconstruction, mask),
        "hidden_f1": f1_score_hidden(adjacency, reconstruction, hmask),
        "latent_graphs": result.latent_graphs,
        "temporal_signatures": result.temporal_signatures,
        "reconstruction": reconstruction,
        "mask": mask,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-times", type=int, default=16)
    parser.add_argument("--observed-percent", type=int, default=90)
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-seed", type=int, default=91)
    parser.add_argument("--output-dir", type=Path, default=Path("results/data"))
    args = parser.parse_args()

    adjacency, signals, metadata = construct_seasurface_dataset(n_times=args.n_times)
    metrics = run_dgd(
        adjacency,
        signals,
        observed_percent=args.observed_percent,
        outer_iterations=args.outer_iterations,
        seed=args.seed,
        mask_seed=args.mask_seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = (
        args.output_dir
        / f"seasurface_highT_T{args.n_times}_R3_obs{args.observed_percent}"
        f"_outer{args.outer_iterations}_latents.npz"
    )
    np.savez(
        output,
        adjacency=adjacency,
        signals=signals,
        latent_graphs=metrics["latent_graphs"],
        temporal_signatures=metrics["temporal_signatures"],
        reconstruction=metrics["reconstruction"],
        mask=metrics["mask"],
        **metadata,
        observed_percent=args.observed_percent,
        outer_iterations=args.outer_iterations,
        seed=args.seed,
        mask_seed=args.mask_seed,
        observed_relative_error=metrics["observed_relative_error"],
        hidden_relative_error=metrics["hidden_relative_error"],
        observed_f1=metrics["observed_f1"],
        hidden_f1=metrics["hidden_f1"],
    )
    print(f"wrote {output}")
    print(metadata)
    print(
        "observed RE={observed_relative_error:.4f} hidden RE={hidden_relative_error:.4f} "
        "observed F1={observed_f1:.4f} hidden F1={hidden_f1:.4f}".format(**metrics)
    )
    print("temporal signatures:")
    print(np.array2string(metrics["temporal_signatures"], precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
