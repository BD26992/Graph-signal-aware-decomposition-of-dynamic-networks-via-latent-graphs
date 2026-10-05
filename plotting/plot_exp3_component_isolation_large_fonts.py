"""Regenerate Exp. 3 component-isolation unobserved performance plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator


DATASETS = [
    ("swdyn", "SwDyn", Path("results/csv/Exp 3/swdyn_component_isolation_R3_obs10_90_outer50.csv")),
    ("ustemp", "USTemp", Path("results/csv/Exp 3/ustemp_component_isolation_R3_obs10_90_outer50.csv")),
    ("seasurface", "SeaSurf", Path("results/csv/Exp 3/seasurface_component_isolation_R3_obs10_90_outer50.csv")),
    ("contact", "Contact", Path("results/csv/Exp 3/contact_component_isolation_R3_obs10_90_outer50.csv")),
]

SERIES = [
    ("component", "1", r"$\mathbf{A}_1$"),
    ("component", "2", r"$\mathbf{A}_2$"),
    ("component", "3", r"$\mathbf{A}_3$"),
    ("combined", "combined", "Combined"),
]
MARKERS = ["o", "s", "^", "D"]


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
            "legend.fontsize": 16,
        }
    )


def select_series(data: pd.DataFrame, entry_type: str, component: str) -> pd.DataFrame:
    if entry_type == "combined":
        return data[data["entry_type"] == "combined"].copy()
    return data[
        (data["entry_type"] == "component")
        & (data["component"].astype(str) == component)
    ].copy()


def plot_metric(data: pd.DataFrame, dataset_label: str, metric: str, ylabel: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for idx, (entry_type, component, label) in enumerate(SERIES):
        subset = select_series(data, entry_type, component).sort_values("observed_percent")
        ax.plot(
            subset["observed_percent"],
            subset[metric],
            marker=MARKERS[idx],
            markersize=5.4,
            linewidth=1.9,
            label=label,
        )
    ticks = [10, 30, 50, 70, 90]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set_xlabel("Percentage of observed data")
    ax.set_ylabel(ylabel)
    ax.set_title(dataset_label)
    if metric == "unobserved_f1":
        ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    output_dir = Path("results/plots/Exp 3")
    for dataset_key, dataset_label, csv_path in DATASETS:
        data = pd.read_csv(csv_path)
        re_output = output_dir / f"{dataset_label}_isolated_RE_font20_xticks_10_30_50_70_90.pdf"
        f1_output = output_dir / f"{dataset_label}_isolated_F1_font20_xticks_10_30_50_70_90.pdf"
        plot_metric(data, dataset_label, "unobserved_relative_error", "Unobserved relative error", re_output)
        plot_metric(data, dataset_label, "unobserved_f1", "Unobserved F1", f1_output)
        # Overwrite the short paper-facing names too.
        plot_metric(data, dataset_label, "unobserved_relative_error", "Unobserved relative error", output_dir / f"{dataset_label}_isolated_RE.pdf")
        plot_metric(data, dataset_label, "unobserved_f1", "Unobserved F1", output_dir / f"{dataset_label}_isolated_F1.pdf")


if __name__ == "__main__":
    main()
