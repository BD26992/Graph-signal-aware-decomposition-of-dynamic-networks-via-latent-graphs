"""Observed-entry convergence curves for DGD.

This reproduces the MATLAB exp1 convergence idea, but uses the corrected
observed-entry objective and observed-entry F1 requested for the Python setup.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from dynamic_graph_decomposition import (
    DecompositionParams,
    DynamicGraphDecomposer,
    signal_distance_matrices,
    temporal_difference_matrix,
)
from seasurf_paper_experiments import load_matlab_adjacency_signals, matlab_mask_type2
from swdyn_paper_experiments import f1_score_hidden, make_swdyn, symmetric_observation_mask


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

REAL_DATASETS = {
    "contact": {
        "label": "Contact",
        "path": DATA_DIR / "processed/contact_50_data.mat",
        "rank": 3,
        "mask_seed": 91,
        "alpha": 1.0,
        "gamma": 3.0,
        "beta": 0.01,
        "delta": 2.0,
        "mu": 0.1,
        "rho": 1.0,
        "eta_graph": 0.0,
    },
    "seasurface": {
        "label": "SeaSurface",
        "path": DATA_DIR / "processed/surface_8_data.mat",
        "rank": 3,
        "mask_seed": 91,
        "alpha": 1.0,
        "gamma": 0.001,
        "beta": 0.0001,
        "delta": 10.0,
        "mu": 0.1,
        "rho": 0.001,
        "eta_graph": 0.0,
    },
    "ustemp": {
        "label": "USTemp",
        "path": DATA_DIR / "processed/UStemp_15_data.mat",
        "rank": 3,
        "mask_seed": 91,
        "alpha": 1.0,
        "gamma": 0.001,
        "beta": 0.001,
        "delta": 10.0,
        "mu": 0.1,
        "rho": 0.001,
        "eta_graph": 0.0,
    },
}

PARSER_DEFAULTS = {
    "rank": 2,
    "mask_seed": 790,
    "alpha": 1.0,
    "gamma": 0.01,
    "beta": 0.001,
    "delta": 100.0,
    "mu": 0.1,
    "rho": 0.001,
    "eta_graph": 0.0,
    "csv_output": Path("results/csv/swdyn_dgd_observed_convergence_R2_obs90_outer30.csv"),
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def observed_objective_terms(
    adjacency: np.ndarray,
    reconstruction: np.ndarray,
    mask: np.ndarray,
    distances: np.ndarray,
    latent_graphs: np.ndarray,
    temporal_signatures: np.ndarray,
    params: DecompositionParams,
) -> tuple[float, float, float]:
    fit = params.alpha * float(np.sum((mask * (adjacency - reconstruction)) ** 2))
    signal = 0.0
    rank = latent_graphs.shape[0]
    for t in range(adjacency.shape[2]):
        for r in range(rank):
            signal += (
                params.delta
                * temporal_signatures[t, r]
                * float(np.sum(latent_graphs[r] * distances[t]))
            )
    return fit, signal, fit + signal


def make_params(args: argparse.Namespace) -> DecompositionParams:
    return DecompositionParams(
        rank=args.rank,
        alpha=args.alpha,
        gamma=args.gamma,
        beta=args.beta,
        delta=args.delta,
        delta_A=args.delta,
        mu=args.mu,
        rho=args.rho,
        eta_graph=args.eta_graph,
        zeta=1e-4,
        outer_iterations=1,
        inner_iterations_A=5,
        inner_iterations_C=5,
        random_state=args.seed,
    )


def run_dataset(
    *,
    label: str,
    adjacency: np.ndarray,
    signals: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    n_nodes, n_signals, _ = signals.shape
    scaled_signals = signals / np.sqrt(n_nodes * n_signals)
    distances = signal_distance_matrices(scaled_signals)

    params = make_params(args)
    solver = DynamicGraphDecomposer(params)
    latent_graphs = None
    temporal_signatures = None
    rows: list[dict[str, object]] = []
    normalizer = None
    diff = temporal_difference_matrix(adjacency.shape[2])

    for iteration in range(1, args.outer_iterations + 1):
        result = solver.fit(
            adjacency,
            scaled_signals,
            mask=mask,
            initial_latent_graphs=latent_graphs,
            initial_temporal_signatures=temporal_signatures,
        )
        latent_graphs = result.latent_graphs
        temporal_signatures = result.temporal_signatures
        reconstruction = solver.reconstruct(latent_graphs, temporal_signatures)
        fit, signal, objective = observed_objective_terms(
            adjacency,
            reconstruction,
            mask,
            distances,
            latent_graphs,
            temporal_signatures,
            params,
        )
        if normalizer is None:
            normalizer = objective if objective > 0.0 else 1.0
        rows.append(
            {
                "dataset": label,
                "iteration": iteration,
                "observed_percent": args.observed_percent,
                "rank": args.rank,
                "mask_seed": args.mask_seed,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "beta": args.beta,
                "delta": args.delta,
                "delta_A": args.delta,
                "mu": args.mu,
                "rho": args.rho,
                "eta_graph": args.eta_graph,
                "observed_fit_term": fit,
                "signal_smoothness_term": signal,
                "observed_objective": objective,
                "normalized_observed_objective": objective / normalizer,
                "observed_f1": f1_score_hidden(adjacency, reconstruction, mask),
                "solver_total_objective": solver.objective(
                    adjacency,
                    mask,
                    distances,
                    diff,
                    latent_graphs,
                    temporal_signatures,
                ),
            }
        )
        print(
            f"{label} "
            f"iter={iteration} norm_obj={rows[-1]['normalized_observed_objective']:.4g} "
            f"obs_f1={rows[-1]['observed_f1']:.4g}",
            flush=True,
        )
    return rows


def run_swdyn(args: argparse.Namespace) -> list[dict[str, object]]:
    data = make_swdyn()
    mask = symmetric_observation_mask(
        data.adjacency.shape,
        args.observed_percent / 100.0,
        seed=args.mask_seed,
    )
    return run_dataset(
        label="SwDyn",
        adjacency=data.adjacency,
        signals=data.signals,
        mask=mask,
        args=args,
    )


def run_real(args: argparse.Namespace) -> list[dict[str, object]]:
    spec = REAL_DATASETS[args.dataset]
    adjacency, signals = load_matlab_adjacency_signals(args.data or spec["path"])
    mask = matlab_mask_type2(adjacency, args.observed_percent, seed=args.mask_seed)
    return run_dataset(
        label=str(spec["label"]),
        adjacency=adjacency,
        signals=signals,
        mask=mask,
        args=args,
    )


def plot_curve(
    rows: list[dict[str, object]],
    *,
    key: str,
    ylabel: str,
    title: str,
    output: Path,
    log_y: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    x = [int(row["iteration"]) for row in rows]
    y = [float(row[key]) for row in rows]
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        write_basic_pdf_line_plot(
            x,
            y,
            output=output,
            xlabel="Outer iteration",
            ylabel=ylabel,
            title=title,
            log_y=log_y,
        )
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(x, y, marker="o", markersize=3.5, linewidth=2.0, color="#1f77b4")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_basic_pdf_line_plot(
    x: list[int],
    y: list[float],
    *,
    output: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    log_y: bool,
) -> None:
    width, height = 432.0, 300.0
    left, right, bottom, top = 58.0, 20.0, 48.0, 34.0
    plot_w = width - left - right
    plot_h = height - bottom - top
    x_min, x_max = min(x), max(x)
    y_values = [math.log10(v) if log_y and v > 0 else v for v in y]
    y_min, y_max = min(y_values), max(y_values)
    if math.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1) * plot_w

    def sy(value: float) -> float:
        return bottom + (value - y_min) / (y_max - y_min) * plot_h

    points = [(sx(xx), sy(yy)) for xx, yy in zip(x, y_values)]
    commands: list[str] = []
    commands.append("1 1 1 rg 0 0 432 300 re f")
    commands.append("0.92 0.92 0.92 RG 0.5 w")
    for i in range(6):
        yy = bottom + i * plot_h / 5
        commands.append(f"{left:.2f} {yy:.2f} m {left + plot_w:.2f} {yy:.2f} l S")
    for i in range(6):
        xx = left + i * plot_w / 5
        commands.append(f"{xx:.2f} {bottom:.2f} m {xx:.2f} {bottom + plot_h:.2f} l S")
    commands.append("0 0 0 RG 1 w")
    commands.append(f"{left:.2f} {bottom:.2f} m {left + plot_w:.2f} {bottom:.2f} l S")
    commands.append(f"{left:.2f} {bottom:.2f} m {left:.2f} {bottom + plot_h:.2f} l S")
    commands.append("0.12 0.47 0.71 RG 2 w")
    first_x, first_y = points[0]
    path = [f"{first_x:.2f} {first_y:.2f} m"]
    path.extend(f"{px:.2f} {py:.2f} l" for px, py in points[1:])
    commands.append(" ".join(path) + " S")
    commands.append("0.12 0.47 0.71 rg")
    for px, py in points:
        commands.append(f"{px - 2.2:.2f} {py - 2.2:.2f} 4.4 4.4 re f")
    commands.append("0 0 0 rg")
    commands.append(f"BT /F1 13 Tf 72 282 Td ({_pdf_escape(title)}) Tj ET")
    commands.append(f"BT /F1 10 Tf 174 18 Td ({_pdf_escape(xlabel)}) Tj ET")
    commands.append(f"BT /F1 10 Tf 8 164 Td ({_pdf_escape(ylabel)}) Tj ET")
    for i in range(6):
        value = x_min + i * (x_max - x_min) / 5
        xx = left + i * plot_w / 5
        commands.append(f"BT /F1 8 Tf {xx - 6:.2f} 34 Td ({value:.0f}) Tj ET")
    for i in range(6):
        value = y_min + i * (y_max - y_min) / 5
        label = f"1e{value:.1f}" if log_y else f"{value:.2f}"
        yy = bottom + i * plot_h / 5
        commands.append(f"BT /F1 8 Tf 18 {yy - 3:.2f} Td ({_pdf_escape(label)}) Tj ET")

    stream = "\n".join(commands).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 432 300] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["swdyn", "contact", "seasurface", "ustemp"],
        default="swdyn",
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--observed-percent", type=int, default=90)
    parser.add_argument("--outer-iterations", type=int, default=30)
    parser.add_argument("--rank", type=int, default=PARSER_DEFAULTS["rank"])
    parser.add_argument("--mask-seed", type=int, default=PARSER_DEFAULTS["mask_seed"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=PARSER_DEFAULTS["alpha"])
    parser.add_argument("--gamma", type=float, default=PARSER_DEFAULTS["gamma"])
    parser.add_argument("--beta", type=float, default=PARSER_DEFAULTS["beta"])
    parser.add_argument("--delta", type=float, default=PARSER_DEFAULTS["delta"])
    parser.add_argument("--mu", type=float, default=PARSER_DEFAULTS["mu"])
    parser.add_argument("--rho", type=float, default=PARSER_DEFAULTS["rho"])
    parser.add_argument("--eta-graph", type=float, default=PARSER_DEFAULTS["eta_graph"])
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PARSER_DEFAULTS["csv_output"],
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results/plots/observed_convergence"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset in REAL_DATASETS:
        spec = REAL_DATASETS[args.dataset]
        defaults = {
            "rank": "rank",
            "mask_seed": "mask_seed",
            "alpha": "alpha",
            "gamma": "gamma",
            "beta": "beta",
            "delta": "delta",
            "mu": "mu",
            "rho": "rho",
            "eta_graph": "eta_graph",
        }
        for arg_name, spec_name in defaults.items():
            if getattr(args, arg_name) == PARSER_DEFAULTS[arg_name]:
                setattr(args, arg_name, spec[spec_name])
        if args.csv_output == PARSER_DEFAULTS["csv_output"]:
            args.csv_output = Path(
                f"results/csv/{args.dataset}_dgd_observed_convergence_"
                f"R{args.rank}_obs{args.observed_percent}_outer{args.outer_iterations}.csv"
            )
    rows = run_real(args) if args.dataset in REAL_DATASETS else run_swdyn(args)
    write_csv(args.csv_output, rows)
    stem = (
        f"{args.dataset}_dgd_observed_"
        f"R{args.rank}_obs{args.observed_percent}_outer{args.outer_iterations}"
    )
    plot_curve(
        rows,
        key="normalized_observed_objective",
        ylabel="Normalized observed objective",
        title=f"{rows[0]['dataset']} DGD observed objective",
        output=args.plot_dir / f"{stem}_objective.pdf",
        log_y=True,
    )
    plot_curve(
        rows,
        key="observed_f1",
        ylabel="Observed F1 score",
        title=f"{rows[0]['dataset']} DGD observed F1",
        output=args.plot_dir / f"{stem}_f1.pdf",
        log_y=False,
    )
    print(f"wrote {args.csv_output}")
    print(f"wrote {args.plot_dir / f'{stem}_objective.pdf'}")
    print(f"wrote {args.plot_dir / f'{stem}_f1.pdf'}")


if __name__ == "__main__":
    main()
