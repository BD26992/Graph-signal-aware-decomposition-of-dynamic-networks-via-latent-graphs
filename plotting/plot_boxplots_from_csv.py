"""Create shaded boxplots from repeat experiment CSV files.

The expected input is the detail CSV emitted by the experiment runner, with
columns including dataset, method, observed_percent, relative_error, and f1.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


PAGE_WIDTH, PAGE_HEIGHT = letter
LEFT = 64
BOTTOM = 82
PLOT_WIDTH = 455
PLOT_HEIGHT = 320

METHOD_ORDER = ["DGD_delta1", "DGD", "NSDGD_delta0", "NSDGD", "SICA", "BTD", "CPD", "SGL", "UNC"]
METHOD_LABELS = {
    "DGD_delta1": "DGD",
    "DGD": "DGD",
    "NSDGD_delta0": "NSDGD",
    "NSDGD": "NSDGD",
    "SICA": "SICA",
    "BTD": "BTD",
    "CPD": "CPD",
    "SGL": "SGL",
    "UNC": "UNC",
}
PALETTE = {
    "DGD_delta1": colors.HexColor("#1f77b4"),
    "DGD": colors.HexColor("#1f77b4"),
    "NSDGD_delta0": colors.HexColor("#ff7f0e"),
    "NSDGD": colors.HexColor("#ff7f0e"),
    "SICA": colors.HexColor("#17becf"),
    "BTD": colors.HexColor("#9467bd"),
    "CPD": colors.HexColor("#d62728"),
    "SGL": colors.HexColor("#8c564b"),
    "UNC": colors.HexColor("#2ca02c"),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def fmt_tick(value: float) -> str:
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def light_color(color: colors.Color, amount: float = 0.62) -> colors.Color:
    return colors.Color(
        color.red + (1.0 - color.red) * amount,
        color.green + (1.0 - color.green) * amount,
        color.blue + (1.0 - color.blue) * amount,
    )


def grouped_values(
    rows: list[dict[str, str]], metric: str
) -> dict[int, dict[str, list[float]]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = row["method"]
        if method not in METHOD_ORDER:
            continue
        value_text = row.get(metric, "")
        if not value_text:
            continue
        grouped[int(float(row["observed_percent"]))][method].append(float(value_text))
    return grouped


def make_boxplot(
    path: Path,
    rows: list[dict[str, str]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    y_log: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = grouped_values(rows, metric)
    observed = sorted(data)
    methods = [method for method in METHOD_ORDER if any(data[o].get(method) for o in observed)]
    raw_values = [
        value
        for obs in observed
        for method in methods
        for value in data[obs].get(method, [])
        if (not y_log or value > 0)
    ]
    if not raw_values:
        raise ValueError(f"No plottable values found for metric {metric}")

    if y_log:
        plotted_values = [math.log10(value) for value in raw_values]
    else:
        plotted_values = raw_values
    min_y = min(plotted_values)
    max_y = max(plotted_values)
    if min_y == max_y:
        min_y -= 0.5
        max_y += 0.5
    pad = 0.08 * (max_y - min_y)
    min_y -= pad
    max_y += pad

    def tx(index: int) -> float:
        if len(observed) == 1:
            return LEFT + PLOT_WIDTH / 2
        return LEFT + PLOT_WIDTH * index / (len(observed) - 1)

    def ty(value: float) -> float:
        plotted = math.log10(value) if y_log else value
        return BOTTOM + PLOT_HEIGHT * (plotted - min_y) / (max_y - min_y)

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 52, title)
    c.setFont("Helvetica", 10)
    c.drawCentredString(LEFT + PLOT_WIDTH / 2, 42, "Observed topology (%)")
    c.saveState()
    c.translate(22, BOTTOM + PLOT_HEIGHT / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()

    c.setStrokeColor(colors.black)
    c.line(LEFT, BOTTOM, LEFT + PLOT_WIDTH, BOTTOM)
    c.line(LEFT, BOTTOM, LEFT, BOTTOM + PLOT_HEIGHT)

    c.setFont("Helvetica", 8)
    for i in range(5):
        plotted = min_y + (max_y - min_y) * i / 4
        label_value = 10**plotted if y_log else plotted
        y = BOTTOM + PLOT_HEIGHT * i / 4
        c.setStrokeColor(colors.HexColor("#d9d9d9"))
        c.line(LEFT, y, LEFT + PLOT_WIDTH, y)
        c.setFillColor(colors.black)
        c.drawRightString(LEFT - 8, y - 3, fmt_tick(label_value))

    cluster_width = min(86, PLOT_WIDTH / max(len(observed), 1) * 0.76)
    box_width = cluster_width / max(len(methods), 1) * 0.72

    for obs_idx, obs in enumerate(observed):
        center = tx(obs_idx)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 9)
        c.drawCentredString(center, BOTTOM - 18, str(obs))
        first_x = center - cluster_width / 2
        step = cluster_width / len(methods)
        for method_idx, method in enumerate(methods):
            values = [v for v in data[obs].get(method, []) if (not y_log or v > 0)]
            if not values:
                continue
            q1 = percentile(values, 0.25)
            q2 = median(values)
            q3 = percentile(values, 0.75)
            low = min(values)
            high = max(values)
            x = first_x + step * (method_idx + 0.5)
            color = PALETTE[method]
            fill = light_color(color)

            c.setStrokeColor(color)
            c.setFillColor(fill)
            c.setLineWidth(1.0)
            c.rect(x - box_width / 2, ty(q1), box_width, ty(q3) - ty(q1), stroke=1, fill=1)
            c.setStrokeColor(color)
            c.line(x, ty(low), x, ty(q1))
            c.line(x, ty(q3), x, ty(high))
            c.line(x - box_width * 0.35, ty(low), x + box_width * 0.35, ty(low))
            c.line(x - box_width * 0.35, ty(high), x + box_width * 0.35, ty(high))
            c.setStrokeColor(colors.black)
            c.setLineWidth(1.4)
            c.line(x - box_width / 2, ty(q2), x + box_width / 2, ty(q2))

            c.setFillColor(color)
            for repeat_idx, value in enumerate(sorted(values)):
                jitter = (repeat_idx - (len(values) - 1) / 2) * min(2.0, box_width / 5)
                c.circle(x + jitter, ty(value), 1.5, stroke=0, fill=1)

    legend_x = LEFT + PLOT_WIDTH + 22
    legend_y = BOTTOM + PLOT_HEIGHT - 8
    c.setFont("Helvetica", 8)
    for idx, method in enumerate(methods):
        y = legend_y - 15 * idx
        color = PALETTE[method]
        c.setFillColor(light_color(color))
        c.setStrokeColor(color)
        c.rect(legend_x, y - 7, 10, 10, stroke=1, fill=1)
        c.setFillColor(colors.black)
        c.drawString(legend_x + 14, y - 5, METHOD_LABELS.get(method, method))

    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#555555"))
    scale_note = "log scale" if y_log else "linear scale"
    c.drawRightString(LEFT + PLOT_WIDTH, 24, f"Boxes: Q1-median-Q3; whiskers: min-max; {scale_note}")
    c.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/csv/exp6_seasurf_repeats_detail.csv"),
        help="Detail CSV containing per-repeat experiment values.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots/seasurf"),
        help="Output directory for PDF plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.csv)
    dataset = rows[0].get("dataset", args.csv.stem) if rows else args.csv.stem
    slug = dataset.lower().replace(" ", "_")
    make_boxplot(
        args.out_dir / f"exp6_{slug}_relative_error_boxplot.pdf",
        rows,
        metric="relative_error",
        title=f"{dataset} Hidden Relative Error by Method",
        ylabel="Hidden relative error",
        y_log=True,
    )
    make_boxplot(
        args.out_dir / f"exp6_{slug}_f1_boxplot.pdf",
        rows,
        metric="f1",
        title=f"{dataset} F1 by Method",
        ylabel="F1 score",
        y_log=False,
    )
    print(f"wrote boxplots to {args.out_dir}")


if __name__ == "__main__":
    main()
