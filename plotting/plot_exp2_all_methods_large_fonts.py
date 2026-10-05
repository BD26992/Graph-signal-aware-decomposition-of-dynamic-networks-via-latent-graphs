"""Regenerate per-dataset all-method unobserved relative-error plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator


DATASETS = [
    (
        "SwDyn",
        Path("results/csv/Exp 2/swdyn_unobserved_relative_error_all_methods_sgl_from_xlsx.csv"),
        Path("results/plots/Exp 2/SwDyn_all_methods_unobserved_relative_error_font20_legend16.pdf"),
        Path("results/plots/Exp 2/Swdyn_all.pdf"),
    ),
    (
        "Contact",
        Path("results/csv/Exp 2/contact_unobserved_relative_error_all_methods_union_dgd_nsdgd_sgl_from_xlsx.csv"),
        Path("results/plots/Exp 2/Contact_all_methods_unobserved_relative_error_font20_legend16.pdf"),
        Path("results/plots/Exp 2/Contact_all.pdf"),
    ),
    (
        "USTemp",
        Path("results/csv/diagnostics/ustemp_all_methods_revised_dgd_nsdgd_sgl_from_xlsx.csv"),
        Path("results/plots/Exp 2/USTemp_all_methods_unobserved_relative_error_font20_legend16.pdf"),
        Path("results/plots/Exp 2/USTemp_all.pdf"),
    ),
    (
        "SeaSurf",
        Path("results/csv/Exp 2/seasurf_unobserved_relative_error_all_methods_retuned_nsdgd_sgl_from_xlsx.csv"),
        Path("results/plots/Exp 2/SeaSurf_all_methods_unobserved_relative_error_font20_legend16.pdf"),
        Path("results/plots/Exp 2/Seasurf_all.pdf"),
    ),
]

METHOD_ORDER = ["DGD", "NSDGD", "SICA", "BTD", "CPD", "SGL", "UNC"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 20,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 12,
        }
    )


def plot_dataset(label: str, csv_path: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(csv_path)
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for idx, method in enumerate(METHOD_ORDER):
        subset = data[data["plot_method"] == method].sort_values("observed_percent")
        if subset.empty:
            continue
        ax.plot(
            subset["observed_percent"],
            subset["unobserved_relative_error"],
            marker=MARKERS[idx],
            markersize=5.2,
            linewidth=1.8,
            label=method,
        )
    ticks = [10, 30, 50, 70, 90]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set_xlabel("Percentage of observed data")
    ax.set_ylabel("Unobserved relative error")
    ax.set_title(label)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best", fontsize=12)
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    for label, csv_path, explicit_output, paper_output in DATASETS:
        plot_dataset(label, csv_path, explicit_output)
        plot_dataset(label, csv_path, paper_output)


if __name__ == "__main__":
    main()
