"""Baseline methods for dynamic graph decomposition experiments.

The MATLAB code in ``tensors_decomp-2`` compares the proposed method against
CPD, BTD/LL1, smoothness-based topology inference, and simpler factorization
baselines.  This file provides NumPy equivalents that use the same data shape
as ``dynamic_graph_decomposition.py``: adjacency tensors are ``(N, N, T)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dynamic_graph_decomposition import (
    DecompositionParams,
    DynamicGraphDecomposer,
    fit_sica_baseline,
    signal_distance_matrices,
)


Array = np.ndarray


@dataclass
class BaselineResult:
    name: str
    reconstruction: Array
    objective_history: list[float] = field(default_factory=list)
    factors: dict[str, Array] = field(default_factory=dict)


def masked_normalized_error(adjacency: Array, reconstruction: Array, mask: Array) -> float:
    denominator = np.linalg.norm(mask * adjacency)
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(mask * (adjacency - reconstruction)) / denominator)


def heldout_mask(mask: Array) -> Array:
    test_mask = 1.0 - np.asarray(mask, dtype=float)
    for t in range(test_mask.shape[2]):
        np.fill_diagonal(test_mask[:, :, t], 0.0)
    return test_mask


def vectorize_tensor(tensor: Array) -> Array:
    return np.stack([tensor[:, :, t].reshape(-1) for t in range(tensor.shape[2])], axis=1)


def unvectorize_tensor(matrix: Array, n_nodes: int, n_times: int) -> Array:
    return np.stack(
        [matrix[:, t].reshape(n_nodes, n_nodes) for t in range(n_times)], axis=2
    )


def masked_unconstrained_als(
    adjacency: Array,
    mask: Array,
    rank: int,
    *,
    n_iterations: int = 100,
    ridge: float = 1e-8,
    random_state: Optional[int] = 0,
) -> BaselineResult:
    """Fit the observed least-squares term with unconstrained factor matrices.

    This baseline solves

        min_{B,C} 0.5 || M o (A_vec - B C^T) ||_F^2

    by weighted alternating least squares.  It does not impose nonnegativity,
    symmetry, zero diagonal, smoothness, or graph-signal constraints.
    """

    adjacency = np.asarray(adjacency, dtype=float)
    mask = np.asarray(mask, dtype=float)
    n_nodes, _, n_times = adjacency.shape
    x = vectorize_tensor(adjacency)
    m = vectorize_tensor(mask)
    rng = np.random.default_rng(random_state)

    basis = rng.standard_normal((n_nodes * n_nodes, rank))
    temporal = rng.standard_normal((n_times, rank))
    history: list[float] = []
    eye = ridge * np.eye(rank)

    for _ in range(n_iterations):
        for row in range(x.shape[0]):
            weights = m[row]
            active = weights > 0
            if not np.any(active):
                basis[row] = 0.0
                continue
            c_active = temporal[active]
            lhs = c_active.T @ (weights[active, None] * c_active) + eye
            rhs = c_active.T @ (weights[active] * x[row, active])
            basis[row] = np.linalg.solve(lhs, rhs)

        for t in range(n_times):
            weights = m[:, t]
            active = weights > 0
            if not np.any(active):
                temporal[t] = 0.0
                continue
            b_active = basis[active]
            lhs = b_active.T @ (weights[active, None] * b_active) + eye
            rhs = b_active.T @ (weights[active] * x[active, t])
            temporal[t] = np.linalg.solve(lhs, rhs)

        residual = m * (x - basis @ temporal.T)
        history.append(float(0.5 * np.sum(residual * residual)))

    reconstruction = unvectorize_tensor(basis @ temporal.T, n_nodes, n_times)
    return BaselineResult(
        name="masked_unconstrained_als",
        reconstruction=reconstruction,
        objective_history=history,
        factors={"basis": basis, "temporal": temporal},
    )


def cpd_als(
    adjacency_observed: Array,
    rank: int,
    *,
    n_iterations: int = 100,
    ridge: float = 1e-8,
    random_state: Optional[int] = 0,
) -> BaselineResult:
    """Canonical polyadic decomposition via standard CP-ALS.

    This mirrors the MATLAB baseline that calls ``cp_als(tensor(A_O), Rcpd)``.
    Missing entries are already zeroed in ``adjacency_observed``.
    """

    x = np.asarray(adjacency_observed, dtype=float)
    n_i, n_j, n_t = x.shape
    rng = np.random.default_rng(random_state)
    u = rng.standard_normal((n_i, rank))
    v = rng.standard_normal((n_j, rank))
    c = rng.standard_normal((n_t, rank))
    history: list[float] = []
    eye = ridge * np.eye(rank)

    for _ in range(n_iterations):
        gram = (v.T @ v) * (c.T @ c) + eye
        khatri = khatri_rao(v, c)
        rhs = x.reshape(n_i, n_j * n_t) @ khatri
        u = rhs @ np.linalg.pinv(gram)

        gram = (u.T @ u) * (c.T @ c) + eye
        khatri = khatri_rao(u, c)
        rhs = np.transpose(x, (1, 0, 2)).reshape(n_j, n_i * n_t) @ khatri
        v = rhs @ np.linalg.pinv(gram)

        gram = (u.T @ u) * (v.T @ v) + eye
        khatri = khatri_rao(u, v)
        rhs = np.transpose(x, (2, 0, 1)).reshape(n_t, n_i * n_j) @ khatri
        c = rhs @ np.linalg.pinv(gram)

        reconstruction = reconstruct_cpd(u, v, c)
        history.append(float(0.5 * np.sum((x - reconstruction) ** 2)))

    return BaselineResult(
        name="cpd_als",
        reconstruction=reconstruct_cpd(u, v, c),
        objective_history=history,
        factors={"U": u, "V": v, "C": c},
    )


def reconstruct_cpd(u: Array, v: Array, c: Array) -> Array:
    return np.einsum("ir,jr,tr->ijt", u, v, c)


def khatri_rao(left: Array, right: Array) -> Array:
    """Columnwise Kronecker product with left index varying slowest."""

    return np.einsum("ir,jr->ijr", left, right).reshape(
        left.shape[0] * right.shape[0], left.shape[1]
    )


def btd_ll1_als(
    adjacency_observed: Array,
    n_blocks: int,
    block_rank: int,
    *,
    n_iterations: int = 100,
    ridge: float = 1e-8,
    random_state: Optional[int] = 0,
) -> BaselineResult:
    """Block-term LL1 decomposition baseline.

    The MATLAB BTD baseline calls ``bcdLL1_alsls`` and reconstructs

        X_t = sum_k (A_k B_k^T) C[t, k].

    This NumPy implementation alternates between rank-``L`` SVD updates for
    the matrix blocks and least-squares updates for the temporal weights.
    """

    x = np.asarray(adjacency_observed, dtype=float)
    n_nodes, _, n_times = x.shape
    rng = np.random.default_rng(random_state)
    blocks = rng.standard_normal((n_blocks, n_nodes, n_nodes))
    temporal = rng.standard_normal((n_times, n_blocks))
    history: list[float] = []

    for _ in range(n_iterations):
        full_reconstruction = reconstruct_btd_blocks(blocks, temporal)
        for k in range(n_blocks):
            residual = x - full_reconstruction + (
                temporal[:, k][None, None, :] * blocks[k][:, :, None]
            )
            denom = float(np.dot(temporal[:, k], temporal[:, k]) + ridge)
            target = np.tensordot(residual, temporal[:, k], axes=([2], [0])) / denom
            blocks[k] = truncated_svd_matrix(target, block_rank)
            full_reconstruction = reconstruct_btd_blocks(blocks, temporal)

        block_matrix = np.stack([blocks[k].reshape(-1) for k in range(n_blocks)], axis=1)
        lhs = block_matrix.T @ block_matrix + ridge * np.eye(n_blocks)
        for t in range(n_times):
            rhs = block_matrix.T @ x[:, :, t].reshape(-1)
            temporal[t] = np.linalg.solve(lhs, rhs)

        reconstruction = reconstruct_btd_blocks(blocks, temporal)
        history.append(float(0.5 * np.sum((x - reconstruction) ** 2)))

    return BaselineResult(
        name="btd_ll1_als",
        reconstruction=reconstruct_btd_blocks(blocks, temporal),
        objective_history=history,
        factors={"blocks": blocks, "temporal": temporal},
    )


def reconstruct_btd_blocks(blocks: Array, temporal: Array) -> Array:
    return np.einsum("kij,tk->ijt", blocks, temporal)


def spatial_ica_baseline(
    adjacency_observed: Array,
    rank: int,
    *,
    n_iterations: int = 500,
    random_state: Optional[int] = 0,
) -> BaselineResult:
    """Spatial ICA/SICA baseline matching the MATLAB ``ICA`` branch."""

    result = fit_sica_baseline(
        adjacency_observed,
        rank=rank,
        mask=np.ones_like(adjacency_observed, dtype=float),
        max_iterations=n_iterations,
        random_state=random_state,
    )
    return BaselineResult(
        name="spatial_ica_baseline",
        reconstruction=result.reconstruction,
        objective_history=result.objective_history,
        factors=result.factors,
    )


def truncated_svd_matrix(matrix: Array, rank: int) -> Array:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    keep = min(rank, len(s))
    return (u[:, :keep] * s[:keep]) @ vt[:keep, :]


def smoothness_topology_estimate(
    signals: Array,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    n_iterations: int = 500,
    step_size: float = 0.02,
    eps: float = 1e-6,
) -> BaselineResult:
    """Estimate each graph independently from signal smoothness.

    This is a NumPy analogue of MATLAB ``smooth_topoID``/
    ``gsp_learn_graph_log_degrees``.  For each time step it solves a projected
    gradient approximation to

        min_A <W_t, A> + beta ||A||_F^2 - alpha sum_i log(deg_i)

    subject to ``A >= 0``, symmetric, and zero diagonal.  Each estimated graph
    is normalized by its maximum entry, matching ``smooth_topoID.m``.
    """

    distances = signal_distance_matrices(signals)
    n_times, n_nodes, _ = distances.shape
    reconstruction = np.zeros((n_nodes, n_nodes, n_times), dtype=float)
    history: list[float] = []

    for t in range(n_times):
        w = distances[t]
        graph = np.exp(-w / max(np.mean(w[w > 0]) if np.any(w > 0) else 1.0, eps))
        np.fill_diagonal(graph, 0.0)

        for _ in range(n_iterations):
            degrees = np.sum(graph, axis=1) + eps
            gradient = w + 2.0 * beta * graph
            gradient -= alpha * (1.0 / degrees[:, None] + 1.0 / degrees[None, :])
            graph = graph - step_size * gradient
            graph = 0.5 * (graph + graph.T)
            graph = np.maximum(graph, 0.0)
            np.fill_diagonal(graph, 0.0)

        max_entry = np.max(graph)
        if max_entry > 0.0:
            graph = graph / max_entry
        reconstruction[:, :, t] = graph

        degrees = np.sum(graph, axis=1) + eps
        obj = np.sum(w * graph) + beta * np.sum(graph * graph) - alpha * np.sum(
            np.log(degrees)
        )
        history.append(float(obj))

    return BaselineResult(
        name="smoothness_topology_estimate",
        reconstruction=reconstruction,
        objective_history=history,
        factors={"signal_distances": distances},
    )


def proposed_without_signal(
    adjacency: Array,
    mask: Array,
    signals: Array,
    params: DecompositionParams,
) -> BaselineResult:
    """Run the proposed decomposition with ``delta = 0``."""

    no_signal_params = DecompositionParams(**{**params.__dict__, "delta": 0.0})
    solver = DynamicGraphDecomposer(no_signal_params)
    result = solver.fit(adjacency=adjacency, mask=mask, signals=signals)
    reconstruction = solver.reconstruct(result.latent_graphs, result.temporal_signatures)
    return BaselineResult(
        name="proposed_delta_zero",
        reconstruction=reconstruction,
        objective_history=result.objective_history,
        factors={
            "latent_graphs": result.latent_graphs,
            "temporal_signatures": result.temporal_signatures,
        },
    )


def matlab_parameter_matched_cpd_rank(n_nodes: int, n_times: int, proposed_rank: int) -> int:
    tgd_params = (n_nodes * n_nodes + n_times) * proposed_rank
    return max(1, int(round(tgd_params / (2 * n_nodes + n_times + 1))))


def matlab_parameter_matched_btd_blocks(n_nodes: int, n_times: int, proposed_rank: int) -> int:
    tgd_params = (n_nodes * n_nodes + n_times) * proposed_rank
    value = (-n_times + np.sqrt(n_times * n_times + 8 * n_nodes * tgd_params)) / (
        4 * n_nodes
    )
    return max(1, int(round(value)))
