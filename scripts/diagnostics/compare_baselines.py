"""Run proposed method and baselines on the current synthetic experiment."""

from __future__ import annotations

import time

import numpy as np

from baselines import (
    btd_ll1_als,
    cpd_als,
    heldout_mask,
    masked_normalized_error,
    masked_unconstrained_als,
    matlab_parameter_matched_btd_blocks,
    matlab_parameter_matched_cpd_rank,
    proposed_without_signal,
    smoothness_topology_estimate,
    spatial_ica_baseline,
)
from dynamic_graph_decomposition import DecompositionParams, DynamicGraphDecomposer
from example import make_synthetic_problem


def main() -> None:
    proposed_rank = 2
    adjacency, mask, signals, *_ = make_synthetic_problem()
    adjacency_observed = mask * adjacency
    test_mask = heldout_mask(mask)
    n_nodes, _, n_times = adjacency.shape

    params = DecompositionParams(
        rank=proposed_rank,
        gamma=1e-3,
        delta=1e-3,
        beta=1e-3,
        mu=1e-2,
        rho=1e-3,
        zeta=1e-4,
        outer_iterations=20,
        inner_iterations_A=5,
        inner_iterations_C=5,
        random_state=0,
    )

    cpd_rank = matlab_parameter_matched_cpd_rank(n_nodes, n_times, proposed_rank)
    btd_blocks = matlab_parameter_matched_btd_blocks(n_nodes, n_times, proposed_rank)
    btd_block_rank = btd_blocks

    methods = [
        (
            "Proposed",
            lambda: run_proposed(adjacency, mask, signals, params),
        ),
        (
            "Proposed delta=0",
            lambda: proposed_without_signal(adjacency, mask, signals, params),
        ),
        (
            "Unconstrained masked ALS",
            lambda: masked_unconstrained_als(
                adjacency,
                mask,
                proposed_rank,
                n_iterations=20,
                random_state=0,
            ),
        ),
        (
            f"CPD-ALS rank={cpd_rank}",
            lambda: cpd_als(
                adjacency_observed,
                cpd_rank,
                n_iterations=30,
                random_state=0,
            ),
        ),
        (
            f"BTD/LL1 blocks={btd_blocks}, L={btd_block_rank}",
            lambda: btd_ll1_als(
                adjacency_observed,
                btd_blocks,
                btd_block_rank,
                n_iterations=30,
                random_state=0,
            ),
        ),
        (
            "SICA",
            lambda: spatial_ica_baseline(
                adjacency_observed,
                proposed_rank,
                n_iterations=100,
                random_state=0,
            ),
        ),
        (
            "Smoothness topology",
            lambda: smoothness_topology_estimate(
                signals,
                alpha=1.0,
                beta=1.0,
                n_iterations=150,
                step_size=0.02,
            ),
        ),
    ]

    print("Synthetic setup", flush=True)
    print(f"N={n_nodes}, T={n_times}, proposed R={proposed_rank}", flush=True)
    print(f"observed entries={int(mask.sum() / 2)}", flush=True)
    print(f"held-out entries={int(test_mask.sum() / 2)}", flush=True)
    print(flush=True)
    print(
        f"{'method':34s} {'obs error':>12s} {'heldout error':>14s} "
        f"{'final obj':>12s} {'seconds':>9s}",
        flush=True,
    )
    print("-" * 88, flush=True)

    for label, factory in methods:
        start = time.perf_counter()
        result = factory()
        elapsed = time.perf_counter() - start
        observed_error = masked_normalized_error(
            adjacency, result.reconstruction, mask
        )
        heldout_error = masked_normalized_error(
            adjacency, result.reconstruction, test_mask
        )
        final_obj = result.objective_history[-1] if result.objective_history else np.nan
        print(
            f"{label:34s} {observed_error:12.6f} {heldout_error:14.6f} "
            f"{final_obj:12.6f} {elapsed:9.2f}",
            flush=True,
        )


def run_proposed(adjacency, mask, signals, params):
    solver = DynamicGraphDecomposer(params)
    result = solver.fit(adjacency=adjacency, mask=mask, signals=signals)
    reconstruction = solver.reconstruct(result.latent_graphs, result.temporal_signatures)
    from baselines import BaselineResult

    return BaselineResult(
        name="proposed",
        reconstruction=reconstruction,
        objective_history=result.objective_history,
        factors={
            "latent_graphs": result.latent_graphs,
            "temporal_signatures": result.temporal_signatures,
        },
    )


if __name__ == "__main__":
    main()
