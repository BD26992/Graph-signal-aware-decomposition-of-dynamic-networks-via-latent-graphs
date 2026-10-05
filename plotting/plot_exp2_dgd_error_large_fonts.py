"""Regenerate Exp. 2 DGD error figures with larger fonts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import pandas as pd


FONT = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
}

RANK_DATASETS = [
    ("SwDyn", "SwDyn"),
    ("Contact", "Contact"),
    ("USTemp", "USTemp"),
    ("SeaSurface", "SeaSurf"),
]
OBSERVATION_DATASETS = [
    ("Swdyn", "SwDyn"),
    ("Contact", "Contact"),
    ("USTemp", "USTemp"),
    ("SeaSurf", "SeaSurf"),
]
MARKERS = ["o", "s", "^", "D"]


def apply_style() -> None:
    plt.rcParams.update(FONT)


def plot_rank_sweep(output: Path, *, ylim: tuple[float, float] | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv("results/csv/Exp 2/dgd_rank_sweep_observed_error_summary_obs90_outer50.csv")
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for idx, (source_dataset, plot_label) in enumerate(RANK_DATASETS):
        subset = data[data["dataset"] == source_dataset].sort_values("rank")
        if subset.empty:
            continue
        ax.plot(
            subset["rank"],
            subset["observed_relative_error"],
            marker=MARKERS[idx],
            markersize=5.4,
            linewidth=1.9,
            label=plot_label,
        )
    ax.set_xlabel(r"$R$")
    ax.set_ylabel("Relative error")
    ax.set_xticks(range(1, 11))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_observation_sweep(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv("results/csv/Exp 2/all_datasets_unobserved_relative_error_all_methods_sgl_from_xlsx.csv")
    data = data[data["method"] == "DGD"].copy()
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for idx, (source_dataset, plot_label) in enumerate(OBSERVATION_DATASETS):
        subset = data[data["dataset"] == source_dataset].sort_values("observed_percent")
        if subset.empty:
            continue
        ax.plot(
            subset["observed_percent"],
            subset["unobserved_relative_error"],
            marker=MARKERS[idx],
            markersize=5.4,
            linewidth=1.9,
            label=plot_label,
        )
    ax.set_xlabel("Percentage of observed data")
    ax.set_ylabel("Relative error")
    tick_values = [10, 30, 50, 70, 90]
    ax.xaxis.set_major_locator(FixedLocator(tick_values))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(value) for value in tick_values])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    plot_rank_sweep(
        Path("results/plots/Exp 2/dgd_observed_error_rank_sweep_all_datasets_square_markers_large_fonts_no_title.pdf")
    )
    plot_rank_sweep(
        Path("results/plots/Exp 2/dgd_observed_error_rank_sweep_all_datasets_square_markers_large_fonts_no_title_full_y.pdf"),
        ylim=(0.0, 0.5),
    )
    plot_observation_sweep(
        Path("results/plots/Exp 2/dgd_relative_error_observation_sweep_all_datasets_square_markers_large_fonts_no_title.pdf")
    )
    plot_observation_sweep(
        Path("results/plots/Exp 2/dgd_relative_error_observation_sweep_all_datasets_square_markers_no_title.pdf")
    )
    plot_observation_sweep(
        Path("results/plots/Exp 2/dgd_observed_relative_error_observation_sweep_all_datasets_square_markers_no_title.pdf")
    )


if __name__ == "__main__":
    main()
