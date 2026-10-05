"""DGD rank sweep with observed-entry per-iteration relative error."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from convergence_observed_curves import REAL_DATASETS
from dynamic_graph_decomposition import (
    DecompositionParams,
    DynamicGraphDecomposer,
    signal_distance_matrices,
    temporal_difference_matrix,
)
from seasurf_paper_experiments import load_matlab_adjacency_signals, matlab_mask_type2
from swdyn_paper_experiments import make_swdyn, symmetric_observation_mask


SWDYN_DEFAULTS = {
    "label": "SwDyn",
    "rank": 2,
    "mask_seed": 790,
    "alpha": 1.0,
    "gamma": 0.01,
    "beta": 0.001,
    "delta": 100.0,
    "mu": 0.1,
    "rho": 0.001,
    "eta_graph": 0.0,
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_dataset(name: str, observed_percent: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if name == "swdyn":
        data = make_swdyn()
        spec = SWDYN_DEFAULTS
        mask = symmetric_observation_mask(
            data.adjacency.shape,
            observed_percent / 100.0,
            seed=int(spec["mask_seed"]),
        )
        return str(spec["label"]), data.adjacency, data.signals, mask, spec

    spec = REAL_DATASETS[name]
    adjacency, signals = load_matlab_adjacency_signals(Path(spec["path"]))
    mask = matlab_mask_type2(adjacency, observed_percent, seed=int(spec["mask_seed"]))
    return str(spec["label"]), adjacency, signals, mask, spec


def params_for_rank(spec: dict[str, object], rank: int, seed: int) -> DecompositionParams:
    delta = float(spec["delta"])
    return DecompositionParams(
        rank=rank,
        alpha=float(spec["alpha"]),
        gamma=float(spec["gamma"]),
        beta=float(spec["beta"]),
        delta=delta,
        delta_A=delta,
        mu=float(spec["mu"]),
        rho=float(spec["rho"]),
        eta_graph=float(spec["eta_graph"]),
        zeta=1e-4,
        outer_iterations=1,
        inner_iterations_A=5,
        inner_iterations_C=5,
        random_state=seed,
    )


def observed_relative_error(
    adjacency: np.ndarray,
    reconstruction: np.ndarray,
    mask: np.ndarray,
) -> float:
    denom = float(np.sum((mask * adjacency) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum((mask * (reconstruction - adjacency)) ** 2) / denom)


def run_dataset(
    name: str,
    *,
    ranks: list[int],
    observed_percent: int,
    outer_iterations: int,
    seed: int,
) -> list[dict[str, object]]:
    label, adjacency, signals, mask, spec = load_dataset(name, observed_percent)
    n_nodes, n_signals, _ = signals.shape
    scaled_signals = signals / np.sqrt(n_nodes * n_signals)
    distances = signal_distance_matrices(scaled_signals)
    diff = temporal_difference_matrix(adjacency.shape[2])
    rows: list[dict[str, object]] = []
    for rank in ranks:
        params = params_for_rank(spec, rank, seed)
        solver = DynamicGraphDecomposer(params)
        rng = np.random.default_rng(params.random_state)
        latent_graphs = solver._initialize_latent_graphs(rng, adjacency.shape[0], rank, None)
        temporal_signatures = solver._initialize_temporal_signatures(
            rng,
            adjacency.shape[2],
            rank,
            None,
        )
        for iteration in range(1, outer_iterations + 1):
            for component in range(rank):
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
            reconstruction = solver.reconstruct(latent_graphs, temporal_signatures)
            rel_error = observed_relative_error(adjacency, reconstruction, mask)
            row = {
                "dataset": label,
                "dataset_key": name,
                "observed_percent": observed_percent,
                "rank": rank,
                "iteration": iteration,
                "mask_seed": spec["mask_seed"],
                "alpha": spec["alpha"],
                "gamma": spec["gamma"],
                "beta": spec["beta"],
                "delta": spec["delta"],
                "delta_A": spec["delta"],
                "mu": spec["mu"],
                "rho": spec["rho"],
                "eta_graph": spec["eta_graph"],
                "observed_relative_error": rel_error,
            }
            rows.append(row)
            print(
                f"{label} R={rank} iter={iteration} observed_RE={rel_error:.4g}",
                flush=True,
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["swdyn", "seasurface", "ustemp", "contact"],
        choices=["swdyn", "seasurface", "ustemp", "contact"],
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--observed-percent", type=int, default=90)
    parser.add_argument("--outer-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/csv/dgd_rank_sweep_observed_error_obs90_outer20.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/csv/dgd_rank_sweep_observed_error_summary_obs90_outer20.csv"),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("results/plots/observed_rank_sweep/dgd_observed_error_vs_rank_obs90_outer20.pdf"),
    )
    return parser.parse_args()


def summarize_final(rows: list[dict[str, object]], final_iteration: int) -> list[dict[str, object]]:
    return [
        row for row in rows if int(row["iteration"]) == final_iteration
    ]


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def plot_error_vs_rank(summary_rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        write_basic_multi_line_pdf(summary_rows, output)
        return

    by_dataset: dict[str, list[dict[str, object]]] = {}
    for row in summary_rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for dataset, rows in by_dataset.items():
        rows = sorted(rows, key=lambda row: int(row["rank"]))
        ax.plot(
            [int(row["rank"]) for row in rows],
            [float(row["observed_relative_error"]) for row in rows],
            marker="o",
            linewidth=2,
            label=dataset,
        )
    ax.set_yscale("log")
    ax.set_xlabel("R")
    ax.set_ylabel("Observed relative error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_basic_multi_line_pdf(summary_rows: list[dict[str, object]], output: Path) -> None:
    by_dataset: dict[str, list[dict[str, object]]] = {}
    for row in summary_rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    width, height = 460.0, 320.0
    left, right, bottom, top = 62.0, 104.0, 50.0, 34.0
    plot_w = width - left - right
    plot_h = height - bottom - top
    all_x = [int(row["rank"]) for row in summary_rows]
    all_y = [float(row["observed_relative_error"]) for row in summary_rows]
    x_min, x_max = min(all_x), max(all_x)
    y_values = [math.log10(max(v, 1e-300)) for v in all_y]
    y_min, y_max = min(y_values), max(y_values)
    y_pad = 0.08 * (y_max - y_min if y_max > y_min else 1.0)
    y_min -= y_pad
    y_max += y_pad

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1) * plot_w

    def sy(value: float) -> float:
        return bottom + (math.log10(max(value, 1e-300)) - y_min) / (y_max - y_min) * plot_h

    colors = [
        (0.12, 0.47, 0.71),
        (1.00, 0.50, 0.05),
        (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16),
    ]
    commands: list[str] = ["1 1 1 rg 0 0 460 320 re f"]
    commands.append("0.92 0.92 0.92 RG 0.5 w")
    for i in range(6):
        yy = bottom + i * plot_h / 5
        commands.append(f"{left:.2f} {yy:.2f} m {left + plot_w:.2f} {yy:.2f} l S")
    commands.append("0 0 0 RG 1 w")
    commands.append(f"{left:.2f} {bottom:.2f} m {left + plot_w:.2f} {bottom:.2f} l S")
    commands.append(f"{left:.2f} {bottom:.2f} m {left:.2f} {bottom + plot_h:.2f} l S")
    for idx, (dataset, rows) in enumerate(by_dataset.items()):
        rows = sorted(rows, key=lambda row: int(row["rank"]))
        r, g, b = colors[idx % len(colors)]
        commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG 2 w")
        points = [(sx(int(row["rank"])), sy(float(row["observed_relative_error"]))) for row in rows]
        path = [f"{points[0][0]:.2f} {points[0][1]:.2f} m"]
        path.extend(f"{px:.2f} {py:.2f} l" for px, py in points[1:])
        commands.append(" ".join(path) + " S")
        commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
        for px, py in points:
            commands.append(f"{px - 2.1:.2f} {py - 2.1:.2f} 4.2 4.2 re f")
        legend_y = height - top - 18.0 * idx
        commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg {width - right + 10:.2f} {legend_y - 2:.2f} 9 9 re f")
        commands.append(f"0 0 0 rg BT /F1 9 Tf {width - right + 24:.2f} {legend_y:.2f} Td ({_pdf_escape(dataset)}) Tj ET")
    commands.append("0 0 0 rg")
    commands.append("BT /F1 13 Tf 122 300 Td (DGD observed error vs R) Tj ET")
    commands.append("BT /F1 10 Tf 200 18 Td (R) Tj ET")
    commands.append("BT /F1 10 Tf 8 174 Td (Observed relative error) Tj ET")
    for value in range(x_min, x_max + 1):
        xx = sx(value)
        commands.append(f"BT /F1 8 Tf {xx - 3:.2f} 36 Td ({value}) Tj ET")
    for i in range(6):
        value = y_min + i * (y_max - y_min) / 5
        yy = bottom + i * plot_h / 5
        commands.append(f"BT /F1 8 Tf 18 {yy - 3:.2f} Td (1e{value:.1f}) Tj ET")

    stream = "\n".join(commands).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 460 320] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "ascii"
        )
    )
    output.write_bytes(pdf)


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        all_rows.extend(
            run_dataset(
                dataset,
                ranks=args.ranks,
                observed_percent=args.observed_percent,
                outer_iterations=args.outer_iterations,
                seed=args.seed,
            )
        )
    write_csv(args.output, all_rows)
    summary_rows = summarize_final(all_rows, args.outer_iterations)
    write_csv(args.summary_output, summary_rows)
    plot_error_vs_rank(summary_rows, args.plot_output)
    print(f"wrote {args.output}")
    print(f"wrote {args.summary_output}")
    print(f"wrote {args.plot_output}")


if __name__ == "__main__":
    main()
