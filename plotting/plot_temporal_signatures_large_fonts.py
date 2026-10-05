"""Regenerate temporal-signature plots with uniform larger fonts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _set_temporal_style(font_size: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
        }
    )


def plot_swdyn_r2(font_size: int = 20) -> None:
    latents = np.load("results/data/swdyn_R2_beta_0p1_delta_100_temporal_signatures.npz")
    signatures = np.asarray(latents["temporal_signatures"], dtype=float)
    time = np.arange(1, signatures.shape[0] + 1)
    output_dir = Path("results/plots/swdyn_component_structure")
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_temporal_style(font_size)

    colors = ["#1f77b4", "#d95f02"]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for component in range(signatures.shape[1]):
        ax.plot(
            time,
            signatures[:, component],
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color=colors[component % len(colors)],
            label=rf"$\mathbf{{c}}_{component + 1}$",
        )
    ax.set_title("SwDyn")
    ax.set_xlabel(r"$t$")
    ax.set_xticks([0, 20, 40])
    ax.set_ylabel("")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best", fontsize=font_size)
    fig.tight_layout()
    pdf = output_dir / "swdyn_R2_temporal_signatures_SwDyn_square_font20.pdf"
    fig.savefig(pdf)
    fig.savefig(pdf.with_suffix(".png"), dpi=300)
    fig.savefig(output_dir / "swdyn_R2_temporal_signatures_SwDyn_square.pdf")
    fig.savefig(output_dir / "swdyn_R2_temporal_signatures_SwDyn_square.png", dpi=300)
    plt.close(fig)


def plot_ustemp(font_size: int = 20) -> None:
    latents = np.load("results/data/ustemp_report_R3_obs90_outer50_latents.npz")
    signatures = np.asarray(latents["temporal_signatures"], dtype=float)
    time = np.arange(1, signatures.shape[0] + 1)
    output_dir = Path("results/plots/temporal_signatures")
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_temporal_style(font_size)

    colors = ["#1f77b4", "#d95f02", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for component in range(signatures.shape[1]):
        ax.plot(
            time,
            signatures[:, component],
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color=colors[component % len(colors)],
            label=rf"$\mathbf{{c}}_{component + 1}$",
        )
    ax.set_title("USTemp")
    ax.set_xlabel(r"$t$")
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax.set_ylabel("")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    pdf = output_dir / "ustemp_temporal_signatures_large_font20.pdf"
    fig.savefig(pdf)
    fig.savefig(pdf.with_suffix(".png"), dpi=300)
    # Also overwrite the prior paper-style names for convenience.
    fig.savefig(output_dir / "ustemp_temporal_signatures.pdf")
    fig.savefig(output_dir / "ustemp_temporal_signatures.png", dpi=300)
    plt.close(fig)


def plot_seasurf(font_size: int = 20) -> None:
    latents = np.load("results/data/seasurface_report_R3_obs90_outer50_latents.npz")
    signatures = np.asarray(latents["temporal_signatures"], dtype=float)
    time = np.arange(1, signatures.shape[0] + 1)
    output_dir = Path("results/plots/atlas/seasurface_report_R3_obs90_outer50")
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_temporal_style(font_size)

    colors = ["#1f77b4", "#d95f02", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for component in range(signatures.shape[1]):
        ax.plot(
            time,
            signatures[:, component],
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color=colors[component % len(colors)],
            label=rf"$\mathbf{{c}}_{component + 1}$",
        )
    ax.set_title("SeaSurf")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    pdf = output_dir / "seasurface_temporal_signatures_SeaSurf_font20.pdf"
    fig.savefig(pdf)
    fig.savefig(pdf.with_suffix(".png"), dpi=300)
    fig.savefig(output_dir / "seasurface_temporal_signatures_SeaSurf_square.pdf")
    fig.savefig(output_dir / "seasurface_temporal_signatures_SeaSurf_square.png", dpi=300)
    plt.close(fig)


def plot_contact(font_size: int = 20) -> None:
    latents = np.load("results/data/contact_report_R3_obs90_outer50_latents.npz")
    signatures = np.asarray(latents["temporal_signatures"], dtype=float)
    time = np.arange(1, signatures.shape[0] + 1)
    output_dir = Path("results/plots/temporal_signatures")
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_temporal_style(font_size)

    colors = ["#1f77b4", "#d95f02", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for component in range(signatures.shape[1]):
        ax.plot(
            time,
            signatures[:, component],
            marker="o",
            markersize=5.0,
            linewidth=1.8,
            color=colors[component % len(colors)],
            label=rf"$\mathbf{{c}}_{component + 1}$",
        )
    ax.set_title("Contact")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    pdf = output_dir / "contact_temporal_signatures_font20.pdf"
    fig.savefig(pdf)
    fig.savefig(pdf.with_suffix(".png"), dpi=300)
    fig.savefig(output_dir / "contact_temporal_signatures.pdf")
    fig.savefig(output_dir / "contact_temporal_signatures.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_swdyn_r2()
    plot_ustemp()
    plot_seasurf()
    plot_contact()


if __name__ == "__main__":
    main()
