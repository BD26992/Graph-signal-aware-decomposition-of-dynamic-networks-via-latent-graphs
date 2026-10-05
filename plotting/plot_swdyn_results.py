"""Create PDF plots from SwDyn experiment CSV files."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


PAPER_WIDTH, PAPER_HEIGHT = letter
PLOT_LEFT = 72
PLOT_BOTTOM = 86
PLOT_WIDTH = 460
PLOT_HEIGHT = 310


PALETTE = [
    colors.HexColor("#1f77b4"),
    colors.HexColor("#ff7f0e"),
    colors.HexColor("#2ca02c"),
    colors.HexColor("#d62728"),
    colors.HexColor("#9467bd"),
    colors.HexColor("#8c564b"),
    colors.HexColor("#17becf"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_line_plot(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: dict[str, list[tuple[float, float]]],
    *,
    y_log: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(PAPER_WIDTH / 2, PAPER_HEIGHT - 54, title)

    all_x = [x for points in series.values() for x, _ in points]
    all_y = [y for points in series.values() for _, y in points]
    if y_log:
        all_y = [max(y, 1e-12) for y in all_y]
        y_values = [log10_safe(y) for y in all_y]
    else:
        y_values = all_y

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(y_values), max(y_values)
    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0
    y_pad = 0.08 * (max_y - min_y)
    min_y -= y_pad
    max_y += y_pad

    def sx(x: float) -> float:
        return PLOT_LEFT + PLOT_WIDTH * (x - min_x) / (max_x - min_x)

    def sy(y: float) -> float:
        value = log10_safe(max(y, 1e-12)) if y_log else y
        return PLOT_BOTTOM + PLOT_HEIGHT * (value - min_y) / (max_y - min_y)

    draw_axes(c, xlabel, ylabel)
    draw_ticks(c, min_x, max_x, min_y, max_y, y_log=y_log)

    for idx, (label, points) in enumerate(series.items()):
        points = sorted(points)
        color = PALETTE[idx % len(PALETTE)]
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(2.0)
        coords = [(sx(x), sy(y)) for x, y in points]
        for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
            c.line(x1, y1, x2, y2)
        for x, y in coords:
            c.circle(x, y, 3, stroke=0, fill=1)
        legend_x = PLOT_LEFT + PLOT_WIDTH + 28
        legend_y = PLOT_BOTTOM + PLOT_HEIGHT - 18 * idx
        c.rect(legend_x, legend_y - 5, 9, 9, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 9)
        c.drawString(legend_x + 14, legend_y - 3, label)

    c.save()


def draw_axes(c: canvas.Canvas, xlabel: str, ylabel: str) -> None:
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)
    c.line(PLOT_LEFT, PLOT_BOTTOM, PLOT_LEFT + PLOT_WIDTH, PLOT_BOTTOM)
    c.line(PLOT_LEFT, PLOT_BOTTOM, PLOT_LEFT, PLOT_BOTTOM + PLOT_HEIGHT)
    c.setFont("Helvetica", 11)
    c.drawCentredString(PLOT_LEFT + PLOT_WIDTH / 2, 42, xlabel)
    c.saveState()
    c.translate(24, PLOT_BOTTOM + PLOT_HEIGHT / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()


def draw_ticks(
    c: canvas.Canvas,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    *,
    y_log: bool,
) -> None:
    c.setFont("Helvetica", 8)
    c.setStrokeColor(colors.lightgrey)
    for i in range(5):
        x = min_x + (max_x - min_x) * i / 4
        px = PLOT_LEFT + PLOT_WIDTH * i / 4
        c.line(px, PLOT_BOTTOM, px, PLOT_BOTTOM + PLOT_HEIGHT)
        c.setFillColor(colors.black)
        c.drawCentredString(px, PLOT_BOTTOM - 15, format_tick(x))
    for i in range(5):
        value = min_y + (max_y - min_y) * i / 4
        py = PLOT_BOTTOM + PLOT_HEIGHT * i / 4
        c.setStrokeColor(colors.lightgrey)
        c.line(PLOT_LEFT, py, PLOT_LEFT + PLOT_WIDTH, py)
        label_value = 10 ** value if y_log else value
        c.setFillColor(colors.black)
        c.drawRightString(PLOT_LEFT - 8, py - 3, format_tick(label_value))


def format_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def log10_safe(value: float) -> float:
    import math

    return math.log10(max(value, 1e-12))


def plot_convergence(csv_dir: Path, plot_dir: Path) -> None:
    rows = read_csv(csv_dir / "exp1_swdyn_convergence.csv")
    series = {
        "DGD": [
            (float(row["iteration"]), float(row["normalized_objective"]))
            for row in rows
        ]
    }
    make_line_plot(
        plot_dir / "exp1_swdyn_objective_convergence.pdf",
        "SwDyn DGD Convergence",
        "Iteration",
        "Normalized objective",
        series,
        y_log=True,
    )


def plot_rank(csv_dir: Path, plot_dir: Path) -> None:
    rows = read_csv(csv_dir / "exp2_swdyn_rank_sweep.csv")
    series = {
        "DGD": [(float(row["rank"]), float(row["relative_error"])) for row in rows]
    }
    make_line_plot(
        plot_dir / "exp2_swdyn_re_vs_rank.pdf",
        "SwDyn Relative Error vs Rank",
        "Rank R",
        "Hidden RE",
        series,
    )


def plot_observed(csv_dir: Path, plot_dir: Path) -> None:
    rows = read_csv(csv_dir / "exp3_swdyn_observation_sweep.csv")
    series = {
        "DGD": [
            (float(row["observed_percent"]), float(row["relative_error"]))
            for row in rows
        ]
    }
    make_line_plot(
        plot_dir / "exp3_swdyn_re_vs_observed.pdf",
        "SwDyn Relative Error vs Observed Topology",
        "Observed topology (%)",
        "Hidden RE",
        series,
    )


def plot_baselines(csv_dir: Path, plot_dir: Path) -> None:
    rows = read_csv(csv_dir / "exp6_swdyn_baseline_check.csv")
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(
            (float(row["observed_percent"]), float(row["relative_error"]))
        )
    make_line_plot(
        plot_dir / "exp6_swdyn_baseline_comparison.pdf",
        "SwDyn Baseline Comparison",
        "Observed topology (%)",
        "Hidden RE",
        dict(sorted(grouped.items())),
        y_log=True,
    )


def main() -> None:
    csv_dir = Path("results/csv")
    plot_dir = Path("plots/swdyn")
    plot_convergence(csv_dir, plot_dir)
    plot_rank(csv_dir, plot_dir)
    plot_observed(csv_dir, plot_dir)
    plot_baselines(csv_dir, plot_dir)
    print(f"wrote plots to {plot_dir}")


if __name__ == "__main__":
    main()
