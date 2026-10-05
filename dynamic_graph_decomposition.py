"""Mask-aware dynamic graph decomposition via latent graphs.

This module implements the corrected algorithm described in
``Graph_signal_aware_decomposition_of_dynamic_networks_via_latent_graphs``.
It factors a dynamic adjacency tensor A[:, :, t] into nonnegative temporal
weights C[t, r] and symmetric nonnegative latent graphs latent_graphs[r].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


Array = np.ndarray


@dataclass
class DecompositionParams:
    """Hyperparameters for the dynamic graph decomposition solver."""

    rank: int
    alpha: float = 1.0
    gamma: float = 1e-2
    delta: float = 1e-2
    delta_A: Optional[float] = None
    beta: float = 1e-2
    mu: float = 1e-2
    rho: float = 1e-2
    eta_graph: float = 0.0
    zeta: float = 1e-3
    lambda_A: float = 1.0
    lambda_C: float = 1.0
    eta_A: Optional[float] = None
    eta_C: Optional[float] = None
    outer_iterations: int = 50
    inner_iterations_A: int = 10
    inner_iterations_C: int = 10
    random_state: Optional[int] = 0
    record_objective: bool = True


@dataclass
class DecompositionResult:
    """Output of the decomposition solver."""

    latent_graphs: Array
    temporal_signatures: Array
    signal_distances: Array
    objective_history: list[float] = field(default_factory=list)


@dataclass
class BaselineResult:
    """Output returned by baseline estimators."""

    name: str
    reconstruction: Array
    n_parameters: int
    factors: dict[str, Array] = field(default_factory=dict)
    objective_history: list[float] = field(default_factory=list)


def project_latent_graph(matrix: Array) -> Array:
    """Project a matrix onto nonnegative symmetric zero-diagonal graphs."""

    projected = 0.5 * (matrix + matrix.T)
    projected = np.maximum(projected, 0.0)
    np.fill_diagonal(projected, 0.0)
    return projected


def project_temporal_signatures(matrix: Array) -> Array:
    """Project temporal signatures onto the nonnegative orthant."""

    return np.maximum(matrix, 0.0)


def signal_distance_matrices(signals: Array) -> Array:
    """Compute Z[t, i, j] = ||X[i, :, t] - X[j, :, t]||_2^2."""

    if signals.ndim != 3:
        raise ValueError("signals must have shape (N, Q, T)")

    n_nodes, _, n_times = signals.shape
    distances = np.empty((n_times, n_nodes, n_nodes), dtype=float)
    for t in range(n_times):
        x_t = signals[:, :, t]
        squared_norms = np.sum(x_t * x_t, axis=1)
        z_t = squared_norms[:, None] + squared_norms[None, :] - 2.0 * x_t @ x_t.T
        distances[t] = np.maximum(z_t, 0.0)
    return distances


def temporal_difference_matrix(n_times: int) -> Array:
    """First-order temporal difference matrix D."""

    if n_times <= 1:
        return np.zeros((0, n_times), dtype=float)

    diff = np.zeros((n_times - 1, n_times), dtype=float)
    rows = np.arange(n_times - 1)
    diff[rows, rows] = -1.0
    diff[rows, rows + 1] = 1.0
    return diff


def product_graph(adjacency: Array) -> Array:
    """Build the MATLAB ``gen_prod_graph`` space-time adjacency.

    The result has ``T`` diagonal spatial graph blocks and identity temporal
    links between each node and itself at adjacent time instants.
    """

    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 3:
        raise ValueError("adjacency must have shape (N, N, T)")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency matrices must be square")

    n_nodes, _, n_times = adjacency.shape
    graph = np.zeros((n_nodes * n_times, n_nodes * n_times), dtype=float)

    for t in range(n_times):
        start = t * n_nodes
        stop = start + n_nodes
        block = adjacency[:, :, t].copy()
        np.fill_diagonal(block, 0.0)
        graph[start:stop, start:stop] = block

    identity = np.eye(n_nodes)
    for t in range(n_times - 1):
        left = t * n_nodes
        right = (t + 1) * n_nodes
        graph[left:left + n_nodes, right:right + n_nodes] = identity
        graph[right:right + n_nodes, left:left + n_nodes] = identity

    return graph


def graph_laplacian(adjacency: Array) -> Array:
    """Return the combinatorial graph Laplacian."""

    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    return np.diag(np.sum(adjacency, axis=1)) - adjacency


def generate_graph_smooth_signals(
    laplacian: Array,
    n_signals: int,
    *,
    smoothness: float = 10.0,
    noise_power: float = 0.0,
    normalize_noise: bool = True,
    random_state: Optional[int] = None,
) -> Array:
    """Sample graph-smooth signals using the MATLAB generator model.

    This mirrors ``generate_graph_signals.m`` for ``sig_type='smooth'``:
    samples are drawn in the graph Fourier domain with covariance
    ``pinv(smoothness * Lambda + I)`` and transformed back to node space.
    """

    laplacian = np.asarray(laplacian, dtype=float)
    if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
        raise ValueError("laplacian must be a square matrix")
    if n_signals <= 0:
        raise ValueError("n_signals must be positive")

    rng = np.random.default_rng(random_state)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (laplacian + laplacian.T))
    eigvals = np.maximum(eigvals, 0.0)
    spectral_std = np.sqrt(1.0 / (smoothness * eigvals + 1.0))
    spectral_samples = spectral_std[:, None] * rng.standard_normal(
        (laplacian.shape[0], n_signals)
    )
    clean = eigvecs @ spectral_samples

    signal_power = np.linalg.norm(clean, "fro") ** 2 / n_signals
    sigma = float(noise_power)
    if normalize_noise:
        sigma = np.sqrt(sigma * signal_power / laplacian.shape[0])
    noise = sigma * rng.standard_normal(clean.shape)
    return clean + noise


def generate_product_graph_smooth_signals(
    adjacency: Array,
    n_signals: int,
    *,
    smoothness: float = 10.0,
    noise_power: float = 0.0,
    normalize_noise: bool = True,
    random_state: Optional[int] = None,
) -> Array:
    """Generate dynamic node signals with the MATLAB product-graph method.

    Returns an array with shape ``(N, M, T)``. This corresponds to
    ``gen_smooth_tensor_signals3.m``: build the product graph, sample smooth
    signals on that graph, then reshape the result back into node-time slices.
    """

    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 3:
        raise ValueError("adjacency must have shape (N, N, T)")

    n_nodes, _, n_times = adjacency.shape
    space_time_graph = product_graph(adjacency)
    laplacian = graph_laplacian(space_time_graph)
    stacked_signals = generate_graph_smooth_signals(
        laplacian,
        n_signals,
        smoothness=smoothness,
        noise_power=noise_power,
        normalize_noise=normalize_noise,
        random_state=random_state,
    )
    return stacked_signals.reshape(n_times, n_nodes, n_signals).transpose(1, 2, 0)


def fit_unconstrained_observation_als(
    adjacency: Array,
    rank: int,
    mask: Optional[Array] = None,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    ridge: float = 1e-8,
    random_state: Optional[int] = 0,
) -> BaselineResult:
    """Fit ``A C.T`` to observed entries without constraints.

    This is the unconstrained alternating least-squares baseline. Unlike the
    MATLAB zero-filled version, the least-squares updates use only observed
    entries when a mask is provided.
    """

    adjacency, mask = _validate_adjacency_and_mask(adjacency, mask)
    if rank <= 0:
        raise ValueError("rank must be positive")

    n_nodes, _, n_times = adjacency.shape
    x_matrix = _tensor_to_matrix(adjacency)
    mask_matrix = _tensor_to_matrix(mask)
    rng = np.random.default_rng(random_state)
    graph_factors = rng.standard_normal((n_nodes * n_nodes, rank))
    temporal_factors = rng.standard_normal((n_times, rank))
    objective_history: list[float] = []

    for _ in range(max_iterations):
        for edge_index in range(n_nodes * n_nodes):
            observed = mask_matrix[edge_index] > 0
            if not np.any(observed):
                graph_factors[edge_index] = 0.0
                continue
            design = temporal_factors[observed]
            target = x_matrix[edge_index, observed]
            graph_factors[edge_index] = _regularized_lstsq(design, target, ridge)

        for t in range(n_times):
            observed = mask_matrix[:, t] > 0
            if not np.any(observed):
                temporal_factors[t] = 0.0
                continue
            design = graph_factors[observed]
            target = x_matrix[observed, t]
            temporal_factors[t] = _regularized_lstsq(design, target, ridge)

        reconstruction_matrix = graph_factors @ temporal_factors.T
        objective = _masked_squared_loss(x_matrix, reconstruction_matrix, mask_matrix)
        objective_history.append(objective)
        if _has_converged(objective_history, tolerance):
            break

    reconstruction = _matrix_to_tensor(graph_factors @ temporal_factors.T, n_nodes, n_times)
    return BaselineResult(
        name="unconstrained_observation_als",
        reconstruction=reconstruction,
        n_parameters=(n_nodes * n_nodes + n_times) * rank,
        factors={"A": graph_factors, "C": temporal_factors},
        objective_history=objective_history,
    )


def fit_cpd_als(
    adjacency: Array,
    rank: int,
    mask: Optional[Array] = None,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
    ridge: float = 1e-8,
    random_state: Optional[int] = 0,
    backend: str = "auto",
) -> BaselineResult:
    """Fit a CP decomposition baseline with a NumPy ALS solver.

    The MATLAB baseline calls Tensor Toolbox ``cp_als`` on the observed,
    zero-filled tensor. ``backend='tensorly'`` follows that convention with
    TensorLy. ``backend='tensorly_masked'`` is available only for optional
    ablations that intentionally give the solver the observation mask. With
    ``backend='auto'``, TensorLy is used on the zero-filled observed tensor when
    installed; otherwise the code falls back to the local NumPy ALS
    implementation.
    """

    adjacency, mask = _validate_adjacency_and_mask(adjacency, mask)
    observed_tensor = adjacency * mask
    n_nodes, _, n_times = adjacency.shape
    backend_used = backend
    if backend not in {"auto", "numpy", "tensorly", "tensorly_masked"}:
        raise ValueError(
            "backend must be 'auto', 'numpy', 'tensorly', or 'tensorly_masked'"
        )
    use_masked_tensorly = backend == "tensorly_masked"
    if backend in {"auto", "tensorly", "tensorly_masked"}:
        try:
            reconstruction, factors, objective_history = _tensorly_cpd_solver(
                adjacency if use_masked_tensorly else observed_tensor,
                rank,
                max_iterations=max_iterations,
                tolerance=tolerance,
                random_state=random_state,
                mask=mask if use_masked_tensorly else None,
            )
            backend_used = "tensorly_masked" if use_masked_tensorly else "tensorly"
        except ImportError:
            if backend in {"tensorly", "tensorly_masked"}:
                raise
            weights, factors_list, objective_history = _cp_als_solver(
                observed_tensor,
                rank,
                max_iterations=max_iterations,
                tolerance=tolerance,
                ridge=ridge,
                random_state=random_state,
            )
            reconstruction = _cp_reconstruct(weights, factors_list)
            factors = {
                "weights": weights,
                "node_left": factors_list[0],
                "node_right": factors_list[1],
                "time": factors_list[2],
            }
            backend_used = "numpy"
    else:
        weights, factors_list, objective_history = _cp_als_solver(
            observed_tensor,
            rank,
            max_iterations=max_iterations,
            tolerance=tolerance,
            ridge=ridge,
            random_state=random_state,
        )
        reconstruction = _cp_reconstruct(weights, factors_list)
        factors = {
            "weights": weights,
            "node_left": factors_list[0],
            "node_right": factors_list[1],
            "time": factors_list[2],
        }
        backend_used = "numpy"
    return BaselineResult(
        name=f"cpd_{backend_used}",
        reconstruction=reconstruction,
        n_parameters=(2 * n_nodes + n_times + 1) * rank,
        factors=factors,
        objective_history=objective_history,
    )


def fit_btd_ll1_als(
    adjacency: Array,
    n_blocks: int,
    block_rank: int,
    mask: Optional[Array] = None,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    ridge: float = 1e-8,
    random_state: Optional[int] = 0,
) -> BaselineResult:
    """Fit a rank-``(L,L,1)`` block term decomposition baseline.

    This is a compact NumPy ALS counterpart to the MATLAB ``bcdLL1_alsls``
    baseline. It decomposes the observed, zero-filled tensor.
    """

    adjacency, mask = _validate_adjacency_and_mask(adjacency, mask)
    if n_blocks <= 0 or block_rank <= 0:
        raise ValueError("n_blocks and block_rank must be positive")

    observed_tensor = adjacency * mask
    left, right, time, objective_history = _btd_ll1_als_solver(
        observed_tensor,
        n_blocks=n_blocks,
        block_rank=block_rank,
        max_iterations=max_iterations,
        tolerance=tolerance,
        ridge=ridge,
        random_state=random_state,
    )
    reconstruction = _btd_ll1_reconstruct(left, right, time, block_rank)
    n_nodes, _, n_times = adjacency.shape
    return BaselineResult(
        name="btd_ll1_als",
        reconstruction=reconstruction,
        n_parameters=(2 * n_nodes * block_rank + n_times) * n_blocks,
        factors={"left": left, "right": right, "time": time},
        objective_history=objective_history,
    )


def fit_sica_baseline(
    adjacency: Array,
    rank: int,
    mask: Optional[Array] = None,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    random_state: Optional[int] = 0,
    backend: str = "auto",
) -> BaselineResult:
    """Fit the MATLAB ``ICA``/SICA baseline on the zero-filled tensor.

    The MATLAB code reshapes the observed tensor to ``(N*N, T)``, calls
    ``fastICA(Z, R, 'kurtosis', 0)``, and reconstructs the centered data from
    the recovered independent components. This implementation follows the same
    convention and uses only ``mask * adjacency`` when a mask is supplied. With
    ``backend='auto'`` it uses scikit-learn's ``FastICA`` when available,
    otherwise it falls back to a compact local FastICA implementation.
    """

    adjacency, mask = _validate_adjacency_and_mask(adjacency, mask)
    if rank <= 0:
        raise ValueError("rank must be positive")

    observed_matrix = _tensor_to_matrix(adjacency * mask)
    backend_used = backend
    if backend not in {"auto", "numpy", "sklearn"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'sklearn'")
    if backend in {"auto", "sklearn"}:
        try:
            reconstruction_matrix, factors, objective_history = _sklearn_fastica_reconstruct(
                observed_matrix,
                rank=rank,
                max_iterations=max_iterations,
                tolerance=tolerance,
                random_state=random_state,
            )
            backend_used = "sklearn"
        except ImportError:
            if backend == "sklearn":
                raise
            reconstruction_matrix, factors, objective_history = _fastica_reconstruct(
                observed_matrix,
                rank=rank,
                max_iterations=max_iterations,
                tolerance=tolerance,
                random_state=random_state,
            )
            backend_used = "numpy"
    else:
        reconstruction_matrix, factors, objective_history = _fastica_reconstruct(
            observed_matrix,
            rank=rank,
            max_iterations=max_iterations,
            tolerance=tolerance,
            random_state=random_state,
        )
        backend_used = "numpy"
    n_nodes, _, n_times = adjacency.shape
    rank_eff = factors["sources"].shape[0]
    return BaselineResult(
        name=f"sica_{backend_used}",
        reconstruction=_matrix_to_tensor(reconstruction_matrix, n_nodes, n_times),
        n_parameters=(n_nodes * n_nodes + n_times) * rank_eff,
        factors=factors,
        objective_history=objective_history,
    )


def fit_smooth_topology_baseline(
    signal_distances: Array,
    *,
    a: float = 1.0,
    b: float = 1.0,
    max_iterations: int = 1000,
    step_size: Optional[float] = None,
    tolerance: float = 1e-6,
) -> BaselineResult:
    """Estimate each topology from signal distances using smoothness only.

    This mirrors the MATLAB ``smooth_topoID`` baseline, which calls
    ``gsp_learn_graph_log_degrees`` independently for each time slice.
    """

    signal_distances = np.asarray(signal_distances, dtype=float)
    if signal_distances.ndim != 3:
        raise ValueError("signal_distances must have shape (T, N, N) or (N, N, T)")
    if signal_distances.shape[1] == signal_distances.shape[2]:
        distances_tnn = signal_distances
    elif signal_distances.shape[0] == signal_distances.shape[1]:
        distances_tnn = np.moveaxis(signal_distances, 2, 0)
    else:
        raise ValueError("signal_distances must have shape (T, N, N) or (N, N, T)")

    n_times, n_nodes, _ = distances_tnn.shape
    reconstruction = np.zeros((n_nodes, n_nodes, n_times), dtype=float)
    objective_history: list[float] = []
    for t in range(n_times):
        graph, history = _learn_graph_log_degrees(
            distances_tnn[t],
            a=a,
            b=b,
            max_iterations=max_iterations,
            step_size=step_size,
            tolerance=tolerance,
        )
        max_value = np.max(graph)
        if max_value > 0.0:
            graph = graph / max_value
        reconstruction[:, :, t] = graph
        objective_history.extend(history)

    return BaselineResult(
        name="smooth_topology",
        reconstruction=reconstruction,
        n_parameters=n_nodes * n_nodes * n_times,
        factors={},
        objective_history=objective_history,
    )


def fit_proposed_without_signals(
    adjacency: Array,
    mask: Optional[Array],
    params: DecompositionParams,
    signals: Optional[Array] = None,
) -> DecompositionResult:
    """Run the proposed decomposition with ``delta=0``.

    Signals are ignored by the objective in this setting. If omitted, a zero
    signal tensor with one feature is supplied only to satisfy shape checks.
    """

    adjacency, mask = _validate_adjacency_and_mask(adjacency, mask)
    no_signal_params = DecompositionParams(
        **{**params.__dict__, "delta": 0.0, "delta_A": None}
    )
    if signals is None:
        n_nodes, _, n_times = adjacency.shape
        signals = np.zeros((n_nodes, 1, n_times), dtype=float)
    solver = DynamicGraphDecomposer(no_signal_params)
    return solver.fit(adjacency=adjacency, signals=signals, mask=mask)


class DynamicGraphDecomposer:
    """Corrected mask-aware dynamic graph decomposition solver."""

    def __init__(self, params: DecompositionParams):
        if params.rank <= 0:
            raise ValueError("rank must be positive")
        self.params = params

    @staticmethod
    def reconstruct(latent_graphs: Array, temporal_signatures: Array) -> Array:
        """Return reconstructed dynamic adjacency with shape (N, N, T)."""

        reconstruction_tnn = np.tensordot(temporal_signatures, latent_graphs, axes=(1, 0))
        return np.moveaxis(reconstruction_tnn, 0, 2)

    def fit(
        self,
        adjacency: Array,
        signals: Array,
        mask: Optional[Array] = None,
        initial_latent_graphs: Optional[Array] = None,
        initial_temporal_signatures: Optional[Array] = None,
        signal_distances: Optional[Array] = None,
    ) -> DecompositionResult:
        """Fit latent graphs and temporal signatures.

        Parameters
        ----------
        adjacency:
            Dynamic adjacency tensor with shape (N, N, T).
        signals:
            Node signal tensor with shape (N, Q, T).
        mask:
            Binary observation mask with shape (N, N, T). If omitted, all
            entries are treated as observed.
        initial_latent_graphs:
            Optional initialization with shape (R, N, N).
        initial_temporal_signatures:
            Optional initialization with shape (T, R).
        """

        adjacency = np.asarray(adjacency, dtype=float)
        signals = np.asarray(signals, dtype=float)
        if mask is None:
            mask = np.ones_like(adjacency, dtype=float)
        else:
            mask = np.asarray(mask, dtype=float)

        self._validate_inputs(adjacency, signals, mask)

        n_nodes, _, n_times = adjacency.shape
        rank = self.params.rank
        rng = np.random.default_rng(self.params.random_state)

        latent_graphs = self._initialize_latent_graphs(
            rng, n_nodes, rank, initial_latent_graphs
        )
        temporal_signatures = self._initialize_temporal_signatures(
            rng, n_times, rank, initial_temporal_signatures
        )

        if signal_distances is None:
            distances = signal_distance_matrices(signals)
        else:
            distances = np.asarray(signal_distances, dtype=float)
            expected_shape = (n_times, n_nodes, n_nodes)
            if distances.shape != expected_shape:
                raise ValueError(f"signal_distances must have shape {expected_shape}")
        diff = temporal_difference_matrix(n_times)
        objective_history: list[float] = []

        for _ in range(self.params.outer_iterations):
            for r in range(rank):
                self._update_latent_graph(
                    adjacency=adjacency,
                    mask=mask,
                    distances=distances,
                    latent_graphs=latent_graphs,
                    temporal_signatures=temporal_signatures,
                    component=r,
                )

            self._update_temporal_signatures(
                adjacency=adjacency,
                mask=mask,
                distances=distances,
                diff=diff,
                latent_graphs=latent_graphs,
                temporal_signatures=temporal_signatures,
            )

            if self.params.record_objective:
                objective_history.append(
                    self.objective(
                        adjacency,
                        mask,
                        distances,
                        diff,
                        latent_graphs,
                        temporal_signatures,
                    )
                )

        return DecompositionResult(
            latent_graphs=latent_graphs,
            temporal_signatures=temporal_signatures,
            signal_distances=distances,
            objective_history=objective_history,
        )

    def objective(
        self,
        adjacency: Array,
        mask: Array,
        distances: Array,
        diff: Array,
        latent_graphs: Array,
        temporal_signatures: Array,
    ) -> float:
        """Evaluate the constrained objective without ADMM terms."""

        p = self.params
        rank, n_nodes, _ = latent_graphs.shape
        n_times = temporal_signatures.shape[0]

        reconstruction_loss = 0.0
        signal_penalty = 0.0
        for t in range(n_times):
            reconstruction = np.tensordot(temporal_signatures[t], latent_graphs, axes=(0, 0))
            residual = mask[:, :, t] * (adjacency[:, :, t] - reconstruction)
            reconstruction_loss += 0.5 * p.alpha * np.sum(residual * residual)
            for r in range(rank):
                signal_penalty += (
                    0.5
                    * p.delta
                    * temporal_signatures[t, r]
                    * np.sum(latent_graphs[r] * distances[t])
                )

        sparsity = p.gamma * np.sum(latent_graphs)
        overlap = 0.0
        for r in range(rank):
            for s in range(rank):
                if r != s:
                    overlap += p.beta * np.sum(latent_graphs[r] * latent_graphs[s])

        smoothness = p.mu * np.sum((diff @ temporal_signatures) ** 2)
        ridge = 0.5 * p.rho * np.sum(temporal_signatures * temporal_signatures)
        graph_ridge = 0.5 * p.eta_graph * np.sum(latent_graphs * latent_graphs)
        return float(
            reconstruction_loss
            + sparsity
            + signal_penalty
            + overlap
            + smoothness
            + ridge
            + graph_ridge
        )

    def _update_latent_graph(
        self,
        adjacency: Array,
        mask: Array,
        distances: Array,
        latent_graphs: Array,
        temporal_signatures: Array,
        component: int,
    ) -> None:
        p = self.params
        rank, n_nodes, _ = latent_graphs.shape
        n_times = temporal_signatures.shape[0]
        r = component

        phi_r = np.ones((n_nodes, 1)) @ temporal_signatures[:, r][None, :]
        gamma_r = -p.zeta * np.ones((n_nodes, n_times), dtype=float)
        for s in range(rank):
            if s == r:
                continue
            phi_s = np.ones((n_nodes, 1)) @ temporal_signatures[:, s][None, :]
            gamma_r += latent_graphs[s] @ phi_s

        graph = latent_graphs[r].copy()
        lagrange = np.zeros((n_nodes, n_times), dtype=float)
        slack = np.maximum(graph @ phi_r + gamma_r, 0.0)

        eta = p.eta_A
        if eta is None:
            weighted_mask = np.zeros((n_nodes, n_nodes), dtype=float)
            for t in range(n_times):
                weighted_mask += temporal_signatures[t, r] ** 2 * mask[:, :, t]
            curvature = (
                p.alpha * np.max(weighted_mask)
                + p.lambda_A * np.linalg.norm(phi_r, ord=2) ** 2
                + p.eta_graph
            )
            eta = 1.0 / (curvature + 1e-12)

        for _ in range(p.inner_iterations_A):
            reconstruction = np.tensordot(temporal_signatures, latent_graphs, axes=(1, 0))
            data_gradient = np.zeros((n_nodes, n_nodes), dtype=float)
            signal_gradient = np.zeros((n_nodes, n_nodes), dtype=float)
            for t in range(n_times):
                residual_t = reconstruction[t] - adjacency[:, :, t]
                data_gradient += p.alpha * (
                    temporal_signatures[t, r] * mask[:, :, t] * residual_t
                )
                signal_gradient += temporal_signatures[t, r] * distances[t]

            overlap_gradient = np.zeros((n_nodes, n_nodes), dtype=float)
            for s in range(rank):
                if s != r:
                    overlap_gradient += latent_graphs[s]

            gradient = (
                data_gradient
                + p.gamma * np.ones((n_nodes, n_nodes), dtype=float)
                + 0.5 * (p.delta_A if p.delta_A is not None else p.delta) * signal_gradient
                + 2.0 * p.beta * overlap_gradient
                + p.eta_graph * graph
            )
            residual_constraint = graph @ phi_r + gamma_r - slack
            augmented_gradient = (
                gradient
                + lagrange @ phi_r.T
                + p.lambda_A * residual_constraint @ phi_r.T
            )
            graph = project_latent_graph(graph - eta * augmented_gradient)

            slack = np.maximum(graph @ phi_r + gamma_r + lagrange / p.lambda_A, 0.0)
            lagrange = lagrange + p.lambda_A * (graph @ phi_r + gamma_r - slack)
            latent_graphs[r] = graph

    def _update_temporal_signatures(
        self,
        adjacency: Array,
        mask: Array,
        distances: Array,
        diff: Array,
        latent_graphs: Array,
        temporal_signatures: Array,
    ) -> None:
        p = self.params
        rank, n_nodes, _ = latent_graphs.shape
        n_times = temporal_signatures.shape[0]

        latent_matrix = latent_graphs.reshape(rank, n_nodes * n_nodes).T
        adjacency_matrix = np.stack(
            [adjacency[:, :, t].reshape(-1) for t in range(n_times)], axis=1
        )
        mask_matrix = np.stack([mask[:, :, t].reshape(-1) for t in range(n_times)], axis=1)
        distance_matrix = np.stack([distances[t].reshape(-1) for t in range(n_times)], axis=0)
        degrees = np.stack([latent_graphs[r] @ np.ones(n_nodes) for r in range(rank)], axis=1)

        lagrange = np.zeros((n_times, n_nodes), dtype=float)
        slack = np.maximum(temporal_signatures @ degrees.T - p.zeta, 0.0)

        eta = p.eta_C
        if eta is None:
            data_curvature = p.alpha * np.linalg.norm(latent_matrix, ord=2) ** 2
            smooth_curvature = 2.0 * p.mu * np.linalg.norm(diff, ord=2) ** 2 if diff.size else 0.0
            degree_curvature = p.lambda_C * np.linalg.norm(degrees, ord=2) ** 2
            curvature = data_curvature + smooth_curvature + p.rho + degree_curvature
            eta = 1.0 / (curvature + 1e-12)

        for _ in range(p.inner_iterations_C):
            residual = mask_matrix * (latent_matrix @ temporal_signatures.T - adjacency_matrix)
            gradient = (
                p.alpha * (residual.T @ latent_matrix)
                + 0.5 * p.delta * distance_matrix @ latent_matrix
                + 2.0 * p.mu * diff.T @ diff @ temporal_signatures
                + p.rho * temporal_signatures
            )
            degree_residual = temporal_signatures @ degrees.T - p.zeta - slack
            augmented_gradient = (
                gradient
                + lagrange @ degrees
                + p.lambda_C * degree_residual @ degrees
            )
            temporal_signatures[:] = project_temporal_signatures(
                temporal_signatures - eta * augmented_gradient
            )
            slack = np.maximum(
                temporal_signatures @ degrees.T - p.zeta + lagrange / p.lambda_C,
                0.0,
            )
            lagrange = lagrange + p.lambda_C * (
                temporal_signatures @ degrees.T - p.zeta - slack
            )

    def _initialize_latent_graphs(
        self,
        rng: np.random.Generator,
        n_nodes: int,
        rank: int,
        initial_latent_graphs: Optional[Array],
    ) -> Array:
        if initial_latent_graphs is not None:
            latent_graphs = np.asarray(initial_latent_graphs, dtype=float).copy()
            expected_shape = (rank, n_nodes, n_nodes)
            if latent_graphs.shape != expected_shape:
                raise ValueError(f"initial_latent_graphs must have shape {expected_shape}")
            for r in range(rank):
                latent_graphs[r] = project_latent_graph(latent_graphs[r])
            return latent_graphs

        latent_graphs = rng.random((rank, n_nodes, n_nodes))
        for r in range(rank):
            latent_graphs[r] = project_latent_graph(latent_graphs[r])
        return latent_graphs

    def _initialize_temporal_signatures(
        self,
        rng: np.random.Generator,
        n_times: int,
        rank: int,
        initial_temporal_signatures: Optional[Array],
    ) -> Array:
        if initial_temporal_signatures is not None:
            temporal_signatures = np.asarray(initial_temporal_signatures, dtype=float).copy()
            expected_shape = (n_times, rank)
            if temporal_signatures.shape != expected_shape:
                raise ValueError(
                    f"initial_temporal_signatures must have shape {expected_shape}"
                )
            return project_temporal_signatures(temporal_signatures)
        return rng.random((n_times, rank))

    def _validate_inputs(self, adjacency: Array, signals: Array, mask: Array) -> None:
        if adjacency.ndim != 3:
            raise ValueError("adjacency must have shape (N, N, T)")
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency matrices must be square")
        if signals.ndim != 3:
            raise ValueError("signals must have shape (N, Q, T)")
        if signals.shape[0] != adjacency.shape[0] or signals.shape[2] != adjacency.shape[2]:
            raise ValueError("signals shape must agree with adjacency on N and T")
        if mask.shape != adjacency.shape:
            raise ValueError("mask must have the same shape as adjacency")


def _validate_adjacency_and_mask(
    adjacency: Array,
    mask: Optional[Array],
) -> tuple[Array, Array]:
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 3:
        raise ValueError("adjacency must have shape (N, N, T)")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency matrices must be square")
    if mask is None:
        mask_array = np.ones_like(adjacency, dtype=float)
    else:
        mask_array = np.asarray(mask, dtype=float)
        if mask_array.shape != adjacency.shape:
            raise ValueError("mask must have the same shape as adjacency")
    return adjacency, mask_array


def _tensor_to_matrix(tensor: Array) -> Array:
    n_nodes, _, n_times = tensor.shape
    return np.stack([tensor[:, :, t].reshape(-1) for t in range(n_times)], axis=1)


def _matrix_to_tensor(matrix: Array, n_nodes: int, n_times: int) -> Array:
    tensor = np.empty((n_nodes, n_nodes, n_times), dtype=float)
    for t in range(n_times):
        tensor[:, :, t] = matrix[:, t].reshape(n_nodes, n_nodes)
    return tensor


def _regularized_lstsq(design: Array, target: Array, ridge: float) -> Array:
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    rhs = design.T @ target
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _masked_squared_loss(target: Array, reconstruction: Array, mask: Array) -> float:
    residual = mask * (target - reconstruction)
    return float(np.sum(residual * residual))


def _has_converged(history: list[float], tolerance: float) -> bool:
    if len(history) < 2:
        return False
    previous = history[-2]
    current = history[-1]
    scale = max(abs(previous), 1.0)
    return abs(previous - current) / scale < tolerance


def _khatri_rao(left: Array, right: Array) -> Array:
    if left.shape[1] != right.shape[1]:
        raise ValueError("factors must have the same number of columns")
    return np.einsum("ir,jr->ijr", left, right).reshape(
        left.shape[0] * right.shape[0], left.shape[1]
    )


def _unfold(tensor: Array, mode: int) -> Array:
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def _normalize_cp_factors(weights: Array, factors: list[Array]) -> tuple[Array, list[Array]]:
    weights = weights.copy()
    normalized = []
    for factor in factors:
        factor = factor.copy()
        norms = np.linalg.norm(factor, axis=0)
        norms = np.where(norms > 0.0, norms, 1.0)
        factor /= norms
        weights *= norms
        normalized.append(factor)
    return weights, normalized


def _cp_als_solver(
    tensor: Array,
    rank: int,
    *,
    max_iterations: int,
    tolerance: float,
    ridge: float,
    random_state: Optional[int],
) -> tuple[Array, list[Array], list[float]]:
    if rank <= 0:
        raise ValueError("rank must be positive")

    rng = np.random.default_rng(random_state)
    factors = [rng.random((dim, rank)) for dim in tensor.shape]
    weights = np.ones(rank, dtype=float)
    objective_history: list[float] = []

    for _ in range(max_iterations):
        weighted_factors = [factor.copy() for factor in factors]
        weighted_factors[0] = weighted_factors[0] * weights

        for mode in range(3):
            other_modes = [idx for idx in range(3) if idx != mode]
            kr = _khatri_rao(
                weighted_factors[other_modes[1]],
                weighted_factors[other_modes[0]],
            )
            unfolded = _unfold(tensor, mode)
            gram = kr.T @ kr + ridge * np.eye(rank)
            updated = np.linalg.solve(gram, kr.T @ unfolded.T).T
            weighted_factors[mode] = updated

        weights, factors = _normalize_cp_factors(
            np.ones(rank, dtype=float),
            weighted_factors,
        )
        reconstruction = _cp_reconstruct(weights, factors)
        objective_history.append(float(np.sum((tensor - reconstruction) ** 2)))
        if _has_converged(objective_history, tolerance):
            break

    return weights, factors, objective_history


def _cp_reconstruct(weights: Array, factors: list[Array]) -> Array:
    return np.einsum(
        "r,ir,jr,kr->ijk",
        weights,
        factors[0],
        factors[1],
        factors[2],
    )


def _tensorly_cpd_solver(
    tensor: Array,
    rank: int,
    *,
    max_iterations: int,
    tolerance: float,
    random_state: Optional[int],
    mask: Optional[Array] = None,
) -> tuple[Array, dict[str, Array], list[float]]:
    try:
        import tensorly as tl
        from tensorly.cp_tensor import cp_to_tensor
        from tensorly.decomposition import parafac
    except ImportError as exc:
        raise ImportError(
            "TensorLy is required for backend='tensorly'. Install tensorly."
        ) from exc

    tl.set_backend("numpy")
    cp_tensor = parafac(
        tensor,
        rank=rank,
        n_iter_max=max_iterations,
        init="random",
        tol=tolerance,
        random_state=random_state,
        normalize_factors=False,
        verbose=False,
        mask=mask,
    )
    reconstruction = np.asarray(cp_to_tensor(cp_tensor), dtype=float)
    weights, factors = cp_tensor
    objective = float(np.sum((tensor - reconstruction) ** 2))
    return reconstruction, {
        "weights": np.asarray(weights, dtype=float),
        "node_left": np.asarray(factors[0], dtype=float),
        "node_right": np.asarray(factors[1], dtype=float),
        "time": np.asarray(factors[2], dtype=float),
    }, [objective]


def _btd_ll1_als_solver(
    tensor: Array,
    *,
    n_blocks: int,
    block_rank: int,
    max_iterations: int,
    tolerance: float,
    ridge: float,
    random_state: Optional[int],
) -> tuple[Array, Array, Array, list[float]]:
    rng = np.random.default_rng(random_state)
    n_left, n_right, n_times = tensor.shape
    total_rank = n_blocks * block_rank
    left = rng.standard_normal((n_left, total_rank))
    right = rng.standard_normal((n_right, total_rank))
    time = rng.standard_normal((n_times, n_blocks))
    objective_history: list[float] = []

    for _ in range(max_iterations):
        basis = _btd_basis(left, right, block_rank)
        time = np.linalg.solve(
            basis.T @ basis + ridge * np.eye(n_blocks),
            basis.T @ _tensor_to_matrix(tensor),
        ).T

        for block in range(n_blocks):
            residual = tensor - _btd_ll1_reconstruct(left, right, time, block_rank)
            block_slice = slice(block * block_rank, (block + 1) * block_rank)
            residual += np.einsum(
                "il,jl,t->ijt",
                left[:, block_slice],
                right[:, block_slice],
                time[:, block],
            )

            weighted_left_rows = []
            targets = []
            for t in range(n_times):
                weighted_left_rows.append(time[t, block] * left[:, block_slice])
                targets.append(residual[:, :, t])
            design_right = np.vstack(weighted_left_rows)
            target_right = np.vstack(targets)
            right[:, block_slice] = np.linalg.solve(
                design_right.T @ design_right + ridge * np.eye(block_rank),
                design_right.T @ target_right,
            ).T

            weighted_right_rows = []
            targets = []
            for t in range(n_times):
                weighted_right_rows.append(time[t, block] * right[:, block_slice])
                targets.append(residual[:, :, t].T)
            design_left = np.vstack(weighted_right_rows)
            target_left = np.vstack(targets)
            left[:, block_slice] = np.linalg.solve(
                design_left.T @ design_left + ridge * np.eye(block_rank),
                design_left.T @ target_left,
            ).T

            for column in range(block_slice.start, block_slice.stop):
                scale = np.linalg.norm(left[:, column])
                if scale > 0.0:
                    left[:, column] /= scale
                    right[:, column] *= scale

        reconstruction = _btd_ll1_reconstruct(left, right, time, block_rank)
        objective_history.append(float(np.sum((tensor - reconstruction) ** 2)))
        if _has_converged(objective_history, tolerance):
            break

    return left, right, time, objective_history


def _btd_basis(left: Array, right: Array, block_rank: int) -> Array:
    n_blocks = left.shape[1] // block_rank
    basis = np.empty((left.shape[0] * right.shape[0], n_blocks), dtype=float)
    for block in range(n_blocks):
        block_slice = slice(block * block_rank, (block + 1) * block_rank)
        basis[:, block] = (left[:, block_slice] @ right[:, block_slice].T).reshape(-1)
    return basis


def _btd_ll1_reconstruct(
    left: Array,
    right: Array,
    time: Array,
    block_rank: int,
) -> Array:
    n_left = left.shape[0]
    n_right = right.shape[0]
    n_times, n_blocks = time.shape
    reconstruction = np.zeros((n_left, n_right, n_times), dtype=float)
    for block in range(n_blocks):
        block_slice = slice(block * block_rank, (block + 1) * block_rank)
        spatial = left[:, block_slice] @ right[:, block_slice].T
        reconstruction += spatial[:, :, None] * time[None, None, :, block]
    return reconstruction


def _fastica_reconstruct(
    matrix: Array,
    *,
    rank: int,
    max_iterations: int,
    tolerance: float,
    random_state: Optional[int],
) -> tuple[Array, dict[str, Array], list[float]]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    n_features, n_samples = matrix.shape
    mean = np.mean(matrix, axis=1, keepdims=True)
    centered = matrix - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank_eff = min(rank, int(np.sum(s > 1e-12)), n_samples)
    if rank_eff == 0:
        reconstruction = np.repeat(mean, n_samples, axis=1)
        return reconstruction, {
            "mixing": np.zeros((n_features, 0), dtype=float),
            "sources": np.zeros((0, n_samples), dtype=float),
            "mean": mean,
            "whitening": np.zeros((0, n_features), dtype=float),
            "unmixing": np.zeros((0, 0), dtype=float),
        }, [float(np.sum((matrix - reconstruction) ** 2))]

    u_r = u[:, :rank_eff]
    s_r = s[:rank_eff]
    vt_r = vt[:rank_eff]
    whitening = (np.sqrt(n_samples) / s_r)[:, None] * u_r.T
    whitened = np.sqrt(n_samples) * vt_r

    rng = np.random.default_rng(random_state)
    unmixing = rng.standard_normal((rank_eff, rank_eff))
    unmixing = _symmetric_decorrelation(unmixing)
    objective_history: list[float] = []

    for _ in range(max_iterations):
        projected = unmixing @ whitened
        updated = (projected**3 @ whitened.T) / n_samples - 3.0 * unmixing
        updated = _symmetric_decorrelation(updated)
        alignment = np.abs(np.diag(updated @ unmixing.T))
        objective_history.append(float(np.max(np.abs(1.0 - alignment))))
        unmixing = updated
        if objective_history[-1] < tolerance:
            break

    sources = unmixing @ whitened
    source_gram = sources @ sources.T
    mixing = centered @ sources.T @ np.linalg.pinv(source_gram)
    reconstruction = mixing @ sources + mean
    objective_history.append(float(np.sum((matrix - reconstruction) ** 2)))
    return reconstruction, {
        "mixing": mixing,
        "sources": sources,
        "mean": mean,
        "whitening": whitening,
        "unmixing": unmixing,
    }, objective_history


def _sklearn_fastica_reconstruct(
    matrix: Array,
    *,
    rank: int,
    max_iterations: int,
    tolerance: float,
    random_state: Optional[int],
) -> tuple[Array, dict[str, Array], list[float]]:
    try:
        from sklearn.decomposition import FastICA
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for backend='sklearn'. Install scikit-learn."
        ) from exc

    matrix = np.asarray(matrix, dtype=float)
    n_features, n_samples = matrix.shape
    rank_eff = min(rank, n_features, n_samples)
    if rank_eff <= 0:
        reconstruction = np.zeros_like(matrix)
        return reconstruction, {
            "mixing": np.zeros((n_features, 0), dtype=float),
            "sources": np.zeros((0, n_samples), dtype=float),
            "mean": np.zeros((n_features, 1), dtype=float),
        }, [float(np.sum((matrix - reconstruction) ** 2))]

    ica = FastICA(
        n_components=rank_eff,
        algorithm="parallel",
        whiten="unit-variance",
        fun="cube",
        max_iter=max_iterations,
        tol=tolerance,
        random_state=random_state,
    )
    samples_by_features = matrix.T
    sources_samples = ica.fit_transform(samples_by_features)
    reconstruction = np.asarray(ica.inverse_transform(sources_samples).T, dtype=float)
    sources = np.asarray(sources_samples.T, dtype=float)
    objective = float(np.sum((matrix - reconstruction) ** 2))
    return reconstruction, {
        "mixing": np.asarray(ica.mixing_, dtype=float),
        "sources": sources,
        "mean": np.asarray(ica.mean_, dtype=float).reshape(-1, 1),
        "components": np.asarray(ica.components_, dtype=float),
    }, [objective]


def _symmetric_decorrelation(matrix: Array) -> Array:
    gram = matrix @ matrix.T
    eigvals, eigvecs = np.linalg.eigh(0.5 * (gram + gram.T))
    eigvals = np.maximum(eigvals, 1e-12)
    return (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T) @ matrix


def _learn_graph_log_degrees(
    distances: Array,
    *,
    a: float,
    b: float,
    max_iterations: int,
    step_size: Optional[float],
    tolerance: float,
) -> tuple[Array, list[float]]:
    """GSPBox-style ``gsp_learn_graph_log_degrees`` solver.

    This follows the forward-backward-forward primal-dual updates used by the
    MATLAB GSPBox routine. The optimization variable is the upper-triangular
    edge-weight vector ``w`` and ``K`` maps edge weights to node degrees.
    """

    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix")
    n_nodes = distances.shape[0]
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, 0.0)
    rows, cols = np.triu_indices(n_nodes, k=1)
    z = distances[rows, cols].astype(float)
    w = np.zeros_like(z)
    dual = _edge_weights_to_degrees(w, rows, cols, n_nodes)
    norm_k = np.sqrt(2.0 * (n_nodes - 1))
    h_beta = 2.0 * b
    mu = h_beta + norm_k
    epsilon = 1.0 / (2.0 * (1.0 + mu))
    if step_size is None:
        step_size = 0.5
    gamma = epsilon + step_size * ((1.0 - epsilon) / mu - epsilon)
    gamma = float(np.clip(gamma, np.finfo(float).eps, np.inf))
    max_weight = np.inf
    objective_history: list[float] = []
    for _ in range(max_iterations):
        y_primal = w - gamma * (2.0 * b * w + _degree_dual_to_edges(dual, rows, cols))
        y_dual = dual + gamma * _edge_weights_to_degrees(w, rows, cols, n_nodes)

        p_primal = np.minimum(max_weight, np.maximum(0.0, y_primal - 2.0 * gamma * z))
        p_dual = y_dual - gamma * a * _prox_sum_log(
            y_dual / (gamma * a),
            1.0 / (gamma * a),
        )

        q_primal = p_primal - gamma * (
            2.0 * b * p_primal + _degree_dual_to_edges(p_dual, rows, cols)
        )
        q_dual = p_dual + gamma * _edge_weights_to_degrees(
            p_primal,
            rows,
            cols,
            n_nodes,
        )

        rel_primal = np.linalg.norm(-y_primal + q_primal) / max(
            np.linalg.norm(w),
            np.finfo(float).eps,
        )
        rel_dual = np.linalg.norm(-y_dual + q_dual) / max(
            np.linalg.norm(dual),
            np.finfo(float).eps,
        )

        w = w - y_primal + q_primal
        dual = dual - y_dual + q_dual
        graph = _edge_vector_to_symmetric_matrix(w, rows, cols, n_nodes)
        objective = _smooth_topology_objective(graph, distances, a, b)
        objective_history.append(objective)
        if rel_primal < tolerance and rel_dual < tolerance:
            break

    graph = _edge_vector_to_symmetric_matrix(w, rows, cols, n_nodes)
    return graph, objective_history


def _smooth_topology_objective(graph: Array, distances: Array, a: float, b: float) -> float:
    degrees = np.maximum(np.sum(graph, axis=1), 1e-12)
    return float(
        np.sum(graph * distances)
        - a * np.sum(np.log(degrees))
        + 0.5 * b * np.sum(graph * graph)
    )


def _prox_sum_log(values: Array, gamma: float) -> Array:
    return 0.5 * (values + np.sqrt(values * values + 4.0 * gamma))


def _edge_weights_to_degrees(
    weights: Array,
    rows: Array,
    cols: Array,
    n_nodes: int,
) -> Array:
    degrees = np.zeros(n_nodes, dtype=float)
    np.add.at(degrees, rows, weights)
    np.add.at(degrees, cols, weights)
    return degrees


def _degree_dual_to_edges(dual: Array, rows: Array, cols: Array) -> Array:
    return dual[rows] + dual[cols]


def _edge_vector_to_symmetric_matrix(
    weights: Array,
    rows: Array,
    cols: Array,
    n_nodes: int,
) -> Array:
    graph = np.zeros((n_nodes, n_nodes), dtype=float)
    graph[rows, cols] = weights
    graph[cols, rows] = weights
    return graph
