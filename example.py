"""Small runnable example for the dynamic graph decomposition solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dynamic_graph_decomposition import (
    DecompositionParams,
    DynamicGraphDecomposer,
    generate_product_graph_smooth_signals,
)


def make_synthetic_problem(
    n_nodes: int = 50,
    n_signals: int = 2,
    n_times: int = 50,
    observed_fraction: float = 0.8,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)

    graph_2, labels_2 = stochastic_block_model(
        rng=rng,
        n_nodes=n_nodes,
        n_communities=2,
        p_in=0.65,
        p_out=0.06,
    )
    graph_4, labels_4 = stochastic_block_model(
        rng=rng,
        n_nodes=n_nodes,
        n_communities=4,
        p_in=0.65,
        p_out=0.06,
        shuffle_labels=True,
    )
    true_graphs = np.stack([graph_2, graph_4], axis=0)
    true_labels = [labels_2, labels_4]

    true_c = np.column_stack(
        [
            np.linspace(1.0, 0.0, n_times),
            np.linspace(0.0, 1.0, n_times),
        ]
    )
    clean = DynamicGraphDecomposer.reconstruct(true_graphs, true_c)
    adjacency = (clean > 0.0).astype(float)

    mask = random_symmetric_mask(
        rng=rng,
        n_nodes=n_nodes,
        n_times=n_times,
        observed_fraction=observed_fraction,
    )
    signals = generate_product_graph_smooth_signals(
        adjacency,
        n_signals=n_signals,
        smoothness=10.0,
        random_state=seed,
    )
    return adjacency, mask, signals, true_graphs, true_c, true_labels


def stochastic_block_model(
    rng: np.random.Generator,
    n_nodes: int,
    n_communities: int,
    p_in: float,
    p_out: float,
    shuffle_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.arange(n_nodes) * n_communities // n_nodes
    if shuffle_labels:
        labels = labels.copy()
        rng.shuffle(labels)
    probabilities = np.where(labels[:, None] == labels[None, :], p_in, p_out)
    upper = rng.random((n_nodes, n_nodes)) < probabilities
    graph = np.triu(upper.astype(float), k=1)
    return graph + graph.T, labels


def random_symmetric_mask(
    rng: np.random.Generator,
    n_nodes: int,
    n_times: int,
    observed_fraction: float,
) -> np.ndarray:
    if not 0.0 < observed_fraction < 1.0:
        raise ValueError("observed_fraction must be between 0 and 1")

    mask = np.zeros((n_nodes, n_nodes, n_times), dtype=float)
    for t in range(n_times):
        upper = rng.random((n_nodes, n_nodes)) < observed_fraction
        upper = np.triu(upper.astype(float), k=1)
        mask[:, :, t] = upper + upper.T
    return mask


def main() -> None:
    rank = 2
    adjacency, mask, signals, true_graphs, true_c, true_labels = make_synthetic_problem()
    params = DecompositionParams(
        rank=rank,
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

    solver = DynamicGraphDecomposer(params)
    result = solver.fit(adjacency=adjacency, mask=mask, signals=signals)
    reconstruction = solver.reconstruct(result.latent_graphs, result.temporal_signatures)
    observed_rmse = np.sqrt(np.mean((mask * (adjacency - reconstruction)) ** 2))
    normalized_error = np.linalg.norm(mask * (adjacency - reconstruction)) / np.linalg.norm(
        mask * adjacency
    )
    test_mask = 1.0 - mask
    for t in range(test_mask.shape[2]):
        np.fill_diagonal(test_mask[:, :, t], 0.0)
    test_normalized_error = np.linalg.norm(
        test_mask * (adjacency - reconstruction)
    ) / np.linalg.norm(test_mask * adjacency)
    plot_path = Path("loss_over_iterations.svg")
    write_loss_plot(result.objective_history, plot_path)
    diagnostics_path = Path("sbm_recovery_diagnostics.svg")
    similarity, matched_indices = match_recovered_graphs(
        true_graphs, result.latent_graphs
    )
    block_summaries = write_sbm_diagnostics_plot(
        true_graphs=true_graphs,
        recovered_graphs=result.latent_graphs,
        true_labels=true_labels,
        matched_indices=matched_indices,
        path=diagnostics_path,
    )

    print("latent_graphs shape:", result.latent_graphs.shape)
    print("temporal_signatures shape:", result.temporal_signatures.shape)
    print("true_graphs shape:", true_graphs.shape)
    print("true C first row:", true_c[0])
    print("true C last row:", true_c[-1])
    print("observed entries:", int(mask.sum() / 2))
    print("held-out entries:", int(test_mask.sum() / 2))
    print("initial objective:", result.objective_history[0])
    print("final objective:", result.objective_history[-1])
    print("observed RMSE:", observed_rmse)
    print("observed normalized error:", normalized_error)
    print("held-out normalized error:", test_normalized_error)
    print("graph correlation matrix:")
    print(similarity)
    print("matched recovered graph indices:", matched_indices)
    for name, summary in block_summaries.items():
        print(f"{name} within mean:", summary["within_mean"])
        print(f"{name} between mean:", summary["between_mean"])
        print(f"{name} within/between ratio:", summary["within_between_ratio"])
    print("loss plot:", plot_path)
    print("SBM diagnostics plot:", diagnostics_path)


def write_loss_plot(losses: list[float], path: Path) -> None:
    if not losses:
        raise ValueError("losses must not be empty")

    width = 800
    height = 460
    margin_left = 78
    margin_right = 26
    margin_top = 34
    margin_bottom = 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    min_loss = min(losses)
    max_loss = max(losses)
    loss_range = max(max_loss - min_loss, 1e-12)

    def x_coord(index: int) -> float:
        if len(losses) == 1:
            return margin_left + plot_width / 2.0
        return margin_left + plot_width * index / (len(losses) - 1)

    def y_coord(loss: float) -> float:
        return margin_top + plot_height * (max_loss - loss) / loss_range

    points = " ".join(
        f"{x_coord(index):.2f},{y_coord(loss):.2f}" for index, loss in enumerate(losses)
    )

    y_ticks = [min_loss + loss_range * i / 4.0 for i in range(5)]
    x_ticks = [0, 4, 9, 14, 19]
    y_tick_markup = []
    for tick in y_ticks:
        y = y_coord(tick)
        y_tick_markup.append(
            f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{width - margin_right}" '
            f'y2="{y:.2f}" stroke="#e6e8eb" stroke-width="1"/>'
        )
        y_tick_markup.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#3c4043">{tick:.1f}</text>'
        )

    x_tick_markup = []
    for tick in x_ticks:
        x = x_coord(tick)
        x_tick_markup.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
            f'y2="{height - margin_bottom + 5}" stroke="#eef0f2" stroke-width="1"/>'
        )
        x_tick_markup.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle" '
            f'font-size="12" fill="#3c4043">{tick + 1}</text>'
        )

    circles = "\n".join(
        f'<circle cx="{x_coord(index):.2f}" cy="{y_coord(loss):.2f}" r="3" fill="#1f77b4"/>'
        for index, loss in enumerate(losses)
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="22" text-anchor="middle" font-size="18" font-family="Arial, sans-serif" fill="#202124">Objective Loss Over 20 Outer Iterations</text>
  {"".join(y_tick_markup)}
  {"".join(x_tick_markup)}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#202124" stroke-width="1.2"/>
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#202124" stroke-width="1.2"/>
  <polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  {circles}
  <text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#202124">Outer iteration</text>
  <text x="18" y="{height / 2}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#202124" transform="rotate(-90 18 {height / 2})">Objective</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def match_recovered_graphs(true_graphs: np.ndarray, recovered_graphs: np.ndarray):
    similarity = np.zeros((true_graphs.shape[0], recovered_graphs.shape[0]), dtype=float)
    for i, true_graph in enumerate(true_graphs):
        for j, recovered_graph in enumerate(recovered_graphs):
            similarity[i, j] = off_diagonal_correlation(true_graph, recovered_graph)

    if true_graphs.shape[0] != 2 or recovered_graphs.shape[0] != 2:
        matched_indices = list(np.argmax(similarity, axis=1))
    elif similarity[0, 0] + similarity[1, 1] >= similarity[0, 1] + similarity[1, 0]:
        matched_indices = [0, 1]
    else:
        matched_indices = [1, 0]
    return similarity, matched_indices


def off_diagonal_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = ~np.eye(left.shape[0], dtype=bool)
    x = left[mask]
    y = right[mask]
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0.0:
        return 0.0
    return float(x @ y / denom)


def write_sbm_diagnostics_plot(
    true_graphs: np.ndarray,
    recovered_graphs: np.ndarray,
    true_labels: list[np.ndarray],
    matched_indices: list[int],
    path: Path,
) -> dict[str, dict[str, float]]:
    panels = []
    summaries = {}
    names = ["2-community SBM", "4-community SBM"]
    for graph_index, name in enumerate(names):
        recovered_index = matched_indices[graph_index]
        labels = true_labels[graph_index]
        order = np.argsort(labels, kind="stable")
        true_sorted = true_graphs[graph_index][np.ix_(order, order)]
        recovered_sorted = recovered_graphs[recovered_index][np.ix_(order, order)]
        panels.append((f"True {name}", true_sorted))
        panels.append((f"Recovered match for {name}", recovered_sorted))
        summaries[f"recovered {name}"] = block_summary(
            recovered_graphs[recovered_index], labels
        )

    width = 980
    height = 620
    panel_size = 220
    gap_x = 34
    gap_y = 54
    start_x = 40
    start_y = 72

    panel_markup = []
    for index, (title, matrix) in enumerate(panels):
        row = index // 2
        col = index % 2
        x = start_x + col * (panel_size + gap_x)
        y = start_y + row * (panel_size + gap_y)
        panel_markup.append(
            f'<text x="{x + panel_size / 2}" y="{y - 16}" text-anchor="middle" '
            f'font-size="15" font-family="Arial, sans-serif" fill="#202124">{title}</text>'
        )
        panel_markup.append(matrix_svg(matrix, x, y, panel_size))

    summary_x = start_x + 2 * (panel_size + gap_x) + 20
    summary_lines = [
        "Recovered Block Statistics",
        "",
    ]
    for name, summary in summaries.items():
        summary_lines.extend(
            [
                name,
                f"within mean: {summary['within_mean']:.4f}",
                f"between mean: {summary['between_mean']:.4f}",
                f"ratio: {summary['within_between_ratio']:.2f}",
                "",
            ]
        )

    summary_markup = []
    for i, line in enumerate(summary_lines):
        weight = "700" if i == 0 or line.startswith("recovered") else "400"
        summary_markup.append(
            f'<text x="{summary_x}" y="{start_y + 10 + i * 24}" '
            f'font-size="14" font-family="Arial, sans-serif" font-weight="{weight}" '
            f'fill="#202124">{line}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="18" font-family="Arial, sans-serif" fill="#202124">True vs Recovered Latent Graphs Sorted by SBM Communities</text>
  {"".join(panel_markup)}
  {"".join(summary_markup)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return summaries


def matrix_svg(matrix: np.ndarray, x: int, y: int, size: int) -> str:
    n_nodes = matrix.shape[0]
    cell = size / n_nodes
    max_value = max(float(np.max(matrix)), 1e-12)
    rects = [
        f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="#f8f9fa" stroke="#202124" stroke-width="1"/>'
    ]
    for i in range(n_nodes):
        for j in range(n_nodes):
            value = matrix[i, j] / max_value
            shade = int(round(255 - 220 * value))
            color = f"rgb({shade},{shade},{255})"
            rects.append(
                f'<rect x="{x + j * cell:.2f}" y="{y + i * cell:.2f}" '
                f'width="{cell + 0.05:.2f}" height="{cell + 0.05:.2f}" fill="{color}"/>'
            )
    return "".join(rects)


def block_summary(graph: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    off_diagonal = ~np.eye(graph.shape[0], dtype=bool)
    same_block = labels[:, None] == labels[None, :]
    within_values = graph[same_block & off_diagonal]
    between_values = graph[(~same_block) & off_diagonal]
    within_mean = float(np.mean(within_values))
    between_mean = float(np.mean(between_values))
    ratio = within_mean / max(between_mean, 1e-12)
    return {
        "within_mean": within_mean,
        "between_mean": between_mean,
        "within_between_ratio": ratio,
    }


if __name__ == "__main__":
    main()
