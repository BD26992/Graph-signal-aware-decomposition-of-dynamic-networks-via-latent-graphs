"""Regenerate Exp. 1 DGD convergence figures with larger fonts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATASETS = [
    ("SwDyn", Path("results/csv/Exp 1/swdyn_dgd_observed_convergence_R2_obs90_outer100.csv")),
    ("Contact", Path("results/csv/Exp 1/contact_dgd_observed_convergence_R3_obs90_outer100.csv")),
    ("USTemp", Path("results/csv/Exp 1/ustemp_dgd_observed_convergence_R3_obs90_outer100.csv")),
    ("SeaSurf", Path("results/csv/Exp 1/seasurface_dgd_observed_convergence_R3_obs90_outer100.csv")),
]


def load_rows(max_iteration: int) -> pd.DataFrame:
    frames = []
    for label, path in DATASETS:
        frame = pd.read_csv(path)
        frame = frame[frame["iteration"] <= max_iteration].copy()
        frame["plot_dataset"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_normalized_objective(output: Path, *, max_iteration: int = 60) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = load_rows(max_iteration)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    markers = ["o", "s", "^", "D"]
    for idx, (label, _path) in enumerate(DATASETS):
        subset = data[data["plot_dataset"] == label].sort_values("iteration")
        ax.semilogy(
            subset["iteration"],
            subset["normalized_observed_objective"],
            marker=markers[idx],
            markersize=4.2,
            linewidth=1.8,
            markevery=1,
            label=label,
        )
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Normalized objective")
    ax.set_xlim(1, max_iteration)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_observed_f1(output: Path, *, max_iteration: int = 60) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = load_rows(max_iteration)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    markers = ["o", "s", "^", "D"]
    for idx, (label, _path) in enumerate(DATASETS):
        subset = data[data["plot_dataset"] == label].sort_values("iteration")
        ax.plot(
            subset["iteration"],
            subset["observed_f1"],
            marker=markers[idx],
            markersize=4.2,
            linewidth=1.8,
            markevery=1,
            label=label,
        )
    ax.set_xlabel("Iterations")
    ax.set_ylabel("F1 score")
    ax.set_xlim(1, max_iteration)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    plot_normalized_objective(
        Path("results/plots/Exp 1/dgd_normalized_objective_all_datasets_square_iter60_markers_large_fonts_no_title.pdf")
    )
    plot_observed_f1(
        Path("results/plots/Exp 1/dgd_observed_f1_all_datasets_square_iter60_markers_large_fonts_no_title.pdf")
    )


if __name__ == "__main__":
    main()
