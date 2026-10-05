"""DGD component-isolation experiment.

For R=3, this script evaluates each recovered component by itself after
least-squares scalar rescaling, and compares it with the usual combined DGD
reconstruction. Metrics are computed on observed and hidden entries.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dgd_rank_sweep_observed_error import SWDYN_DEFAULTS, params_for_rank
from convergence_observed_curves import REAL_DATASETS
from dynamic_graph_decomposition import (
    DynamicGraphDecomposer,
    signal_distance_matrices,
    temporal_difference_matrix,
)
from seasurf_paper_experiments import load_matlab_adjacency_signals, matlab_mask_type2
from swdyn_paper_experiments import (
    hidden_mask,
    make_swdyn,
    symmetric_observation_mask,
    two_means_binary,
)


DATASET_KEYS = ["swdyn", "ustemp", "seasurface", "contact"]


def masked_relative_error(
    truth: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray,
) -> float:
    denom = float(np.sum((mask * truth) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum((mask * (estimate - truth)) ** 2) / denom)


def masked_f1(truth: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    selected = mask > 0.0
    if not np.any(selected):
        return float("nan")
    truth_binary = two_means_binary(truth[selected])
    estimate_binary = two_means_binary(estimate[selected])
    true_pos = np.sum((truth_binary == 1) & (estimate_binary == 1))
    predicted_pos = np.sum(estimate_binary == 1)
    relevant = np.sum(truth_binary == 1)
    if predicted_pos == 0 or relevant == 0:
        return 0.0
    precision = true_pos / predicted_pos
    recall = true_pos / relevant
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def component_reconstruction(
    latent_graphs: np.ndarray,
    temporal_signatures: np.ndarray,
    component: int,
) -> np.ndarray:
    return latent_graphs[component, :, :, None] * temporal_signatures[None, None, :, component]


def best_scalar_on_mask(
    truth: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray,
) -> float:
    masked_estimate = mask * estimate
    denom = float(np.sum(masked_estimate * masked_estimate))
    if denom == 0.0:
        return 0.0
    return float(np.sum((mask * truth) * masked_estimate) / denom)


def load_dataset(dataset_key: str, observed_percent: int, mask_seed: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if dataset_key == "swdyn":
        data = make_swdyn()
        mask = symmetric_observation_mask(
            data.adjacency.shape,
            observed_percent / 100.0,
            seed=mask_seed,
        )
        return "SwDyn", data.adjacency, data.signals, mask, SWDYN_DEFAULTS

    spec = REAL_DATASETS[dataset_key]
    adjacency, signals = load_matlab_adjacency_signals(Path(spec["path"]))
    mask = matlab_mask_type2(adjacency, observed_percent, seed=mask_seed)
    return str(spec["label"]), adjacency, signals, mask, spec


def fit_dgd_components(
    dataset_label: str,
    adjacency: np.ndarray,
    signals: np.ndarray,
    mask: np.ndarray,
    spec: dict[str, object],
    *,
    outer_iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes, n_signals, _ = signals.shape
    scaled_signals = signals / np.sqrt(n_nodes * n_signals)
    distances = signal_distance_matrices(scaled_signals)
    diff = temporal_difference_matrix(adjacency.shape[2])
    params = params_for_rank(spec, rank=3, seed=seed)
    solver = DynamicGraphDecomposer(params)
    rng = np.random.default_rng(params.random_state)
    latent_graphs = solver._initialize_latent_graphs(rng, adjacency.shape[0], 3, None)
    temporal_signatures = solver._initialize_temporal_signatures(
        rng,
        adjacency.shape[2],
        3,
        None,
    )
    for iteration in range(1, outer_iterations + 1):
        for component in range(3):
            solver._update_latent_graph(
                adjacency=adjacency,
                mask=mask,
                distances=distances,
                latent_graphs=latent_graphs,
                temporal_signatures=temporal_signatures,
                component=component,
            )
        solver._update_temporal_signatures(
            adjacency=adjacency,
            mask=mask,
            distances=distances,
            diff=diff,
            latent_graphs=latent_graphs,
            temporal_signatures=temporal_signatures,
        )
        if iteration == 1 or iteration % 10 == 0 or iteration == outer_iterations:
            reconstruction = solver.reconstruct(latent_graphs, temporal_signatures)
            rel = masked_relative_error(adjacency, reconstruction, mask)
            print(
                f"{dataset_label} iter={iteration} observed_RE={rel:.4g}",
                flush=True,
            )
    return latent_graphs, temporal_signatures, solver.reconstruct(latent_graphs, temporal_signatures)


def evaluate_fit(
    adjacency: np.ndarray,
    mask: np.ndarray,
    latent_graphs: np.ndarray,
    temporal_signatures: np.ndarray,
    reconstruction: np.ndarray,
    *,
    dataset_label: str,
    dataset_key: str,
    observed_percent: int,
    mask_seed: int,
    seed: int,
) -> list[dict[str, object]]:
    hmask = hidden_mask(mask)
    rows: list[dict[str, object]] = []
    for component in range(3):
        raw_component = component_reconstruction(latent_graphs, temporal_signatures, component)
        scales = {
            "observed": best_scalar_on_mask(adjacency, raw_component, mask),
            "unobserved": best_scalar_on_mask(adjacency, raw_component, hmask),
        }
        estimates = {
            "observed": scales["observed"] * raw_component,
            "unobserved": scales["unobserved"] * raw_component,
        }
        rows.append(
            {
                "dataset": dataset_label,
                "dataset_key": dataset_key,
                "observed_percent": observed_percent,
                "rank": 3,
                "entry_type": "component",
                "component": component + 1,
                "mask_seed": mask_seed,
                "seed": seed,
                "scale_observed": scales["observed"],
                "scale_unobserved": scales["unobserved"],
                "observed_relative_error": masked_relative_error(adjacency, estimates["observed"], mask),
                "observed_f1": masked_f1(adjacency, estimates["observed"], mask),
                "unobserved_relative_error": masked_relative_error(adjacency, estimates["unobserved"], hmask),
                "unobserved_f1": masked_f1(adjacency, estimates["unobserved"], hmask),
            }
        )

    rows.append(
        {
            "dataset": dataset_label,
            "dataset_key": dataset_key,
            "observed_percent": observed_percent,
            "rank": 3,
            "entry_type": "combined",
            "component": "combined",
            "mask_seed": mask_seed,
            "seed": seed,
            "scale_observed": 1.0,
            "scale_unobserved": 1.0,
            "observed_relative_error": masked_relative_error(adjacency, reconstruction, mask),
            "observed_f1": masked_f1(adjacency, reconstruction, mask),
            "unobserved_relative_error": masked_relative_error(adjacency, reconstruction, hmask),
            "unobserved_f1": masked_f1(adjacency, reconstruction, hmask),
        }
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict[str, object]], metric: str, output: Path, ylabel: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        write_basic_plot_pdf(rows, metric, output, ylabel)
        return

    series = ["component 1", "component 2", "component 3", "combined"]
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    for label in series:
        if label == "combined":
            selected = [row for row in rows if row["entry_type"] == "combined"]
        else:
            component = int(label.split()[1])
            selected = [
                row
                for row in rows
                if row["entry_type"] == "component" and int(row["component"]) == component
            ]
        selected = sorted(selected, key=lambda row: int(row["observed_percent"]))
        ax.plot(
            [int(row["observed_percent"]) for row in selected],
            [float(row[metric]) for row in selected],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.set_xlabel("Observed data (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(10, 100, 10))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_basic_plot_pdf(rows: list[dict[str, object]], metric: str, output: Path, ylabel: str) -> None:
    width, height = 612, 396
    left, right, bottom, top = 62, 24, 54, 34
    plot_w = width - left - right
    plot_h = height - bottom - top
    x_values = list(range(10, 100, 10))
    values = [float(row[metric]) for row in rows if not np.isnan(float(row[metric]))]
    y_min = 0.0
    y_max = max(values) * 1.08 if values else 1.0
    if "f1" in metric:
        y_max = 1.0
    y_max = max(y_max, 1e-9)
    colors = {
        "component 1": (0.122, 0.467, 0.706),
        "component 2": (1.000, 0.498, 0.055),
        "component 3": (0.173, 0.627, 0.173),
        "combined": (0.839, 0.153, 0.157),
    }

    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def px(value: float) -> float:
        return left + (value - 10) / 80 * plot_w

    def py(value: float) -> float:
        return bottom + (value - y_min) / (y_max - y_min) * plot_h

    commands: list[str] = ["1 1 1 rg 0 0 612 396 re f", "0.92 0.92 0.92 RG 0.5 w"]
    for i in range(6):
        yy = bottom + i * plot_h / 5
        commands.append(f"{left:.2f} {yy:.2f} m {left + plot_w:.2f} {yy:.2f} l S")
    commands.append("0 0 0 RG 1 w")
    commands.append(f"{left:.2f} {bottom:.2f} m {left + plot_w:.2f} {bottom:.2f} l S")
    commands.append(f"{left:.2f} {bottom:.2f} m {left:.2f} {bottom + plot_h:.2f} l S")
    commands.append("0 0 0 rg /F1 10 Tf")
    for x in x_values:
        commands.append(f"BT {px(x) - 5:.2f} {bottom - 18:.2f} Td ({x}) Tj ET")
    for i in range(6):
        value = y_min + i * (y_max - y_min) / 5
        commands.append(f"BT {left - 48:.2f} {py(value) - 3:.2f} Td ({value:.3g}) Tj ET")
    commands.append(f"BT {left + plot_w / 2 - 40:.2f} 18 Td (Observed data (%)) Tj ET")
    commands.append(f"BT 16 202 Td 90 Tz ({esc(ylabel)}) Tj ET")
    commands.append(f"/F1 12 Tf BT {left + 70:.2f} {height - 22:.2f} Td ({esc(output.stem)}) Tj ET")

    for label in ["component 1", "component 2", "component 3", "combined"]:
        if label == "combined":
            selected = [row for row in rows if row["entry_type"] == "combined"]
        else:
            component = int(label.split()[1])
            selected = [
                row
                for row in rows
                if row["entry_type"] == "component" and int(row["component"]) == component
            ]
        selected = sorted(selected, key=lambda row: int(row["observed_percent"]))
        cr, cg, cb = colors[label]
        commands.append(f"{cr:.3f} {cg:.3f} {cb:.3f} RG {cr:.3f} {cg:.3f} {cb:.3f} rg 1.8 w")
        points = [(px(int(row["observed_percent"])), py(float(row[metric]))) for row in selected]
        if not points:
            continue
        path = [f"{points[0][0]:.2f} {points[0][1]:.2f} m"]
        path.extend(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
        commands.append(" ".join(path) + " S")
        for x, y in points:
            commands.append(f"{x:.2f} {y:.2f} 3 0 360 arc f")

    legend_x, legend_y = width - right - 120, height - top - 10
    commands.append("/F1 9 Tf")
    for idx, label in enumerate(["component 1", "component 2", "component 3", "combined"]):
        cr, cg, cb = colors[label]
        y = legend_y - idx * 15
        commands.append(f"{cr:.3f} {cg:.3f} {cb:.3f} rg {legend_x:.2f} {y - 3:.2f} 10 3 re f")
        commands.append(f"0 0 0 rg BT {legend_x + 15:.2f} {y - 5:.2f} Td ({esc(label)}) Tj ET")

    write_pdf(output, "\n".join(commands).encode("latin-1"))


def write_pdf(output: Path, content: bytes) -> None:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 396] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    output.write_bytes(pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_KEYS, default="swdyn")
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-seed-base", type=int, default=790)
    parser.add_argument("--observed-percents", type=int, nargs="+", default=list(range(10, 100, 10)))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/csv/swdyn_component_isolation_R3_obs10_90_outer50.csv"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results/plots/component_isolation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, object]] = []
    for observed_percent in args.observed_percents:
        mask_seed = args.mask_seed_base + observed_percent
        dataset_label, adjacency, signals, mask, spec = load_dataset(
            args.dataset,
            observed_percent,
            mask_seed,
        )
        latent_graphs, temporal_signatures, reconstruction = fit_dgd_components(
            dataset_label,
            adjacency,
            signals,
            mask,
            spec,
            outer_iterations=args.outer_iterations,
            seed=args.seed,
        )
        rows = evaluate_fit(
            adjacency,
            mask,
            latent_graphs,
            temporal_signatures,
            reconstruction,
            dataset_label=dataset_label,
            dataset_key=args.dataset,
            observed_percent=observed_percent,
            mask_seed=mask_seed,
            seed=args.seed,
        )
        all_rows.extend(rows)
        combined = rows[-1]
        print(
            f"{dataset_label} "
            f"obs={observed_percent} combined "
            f"obs_RE={combined['observed_relative_error']:.4g} "
            f"obs_F1={combined['observed_f1']:.4g} "
            f"hidden_RE={combined['unobserved_relative_error']:.4g} "
            f"hidden_F1={combined['unobserved_f1']:.4g}",
            flush=True,
        )

    write_csv(args.output, all_rows)
    prefix = args.dataset
    plot_specs = [
        ("observed_relative_error", f"{prefix}_component_isolation_observed_relative_error.pdf", "Observed relative error"),
        ("unobserved_relative_error", f"{prefix}_component_isolation_unobserved_relative_error.pdf", "Unobserved relative error"),
        ("observed_f1", f"{prefix}_component_isolation_observed_f1.pdf", "Observed F1"),
        ("unobserved_f1", f"{prefix}_component_isolation_unobserved_f1.pdf", "Unobserved F1"),
    ]
    for metric, filename, ylabel in plot_specs:
        plot_metric(all_rows, metric, args.plot_dir / filename, ylabel)
    print(f"wrote {args.output}")
    for _, filename, _ in plot_specs:
        print(f"wrote {args.plot_dir / filename}")


if __name__ == "__main__":
    main()
