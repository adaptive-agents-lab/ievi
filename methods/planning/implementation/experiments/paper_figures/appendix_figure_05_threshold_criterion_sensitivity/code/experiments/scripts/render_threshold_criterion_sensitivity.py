#!/usr/bin/env python3
"""Threshold-and-criterion sensitivity for the Cliffwalk Control panel.

Six small multiples in a 2x3 grid:

    rows    : convergence criterion (stable-hit, first-hit)
    columns : threshold (1e-2, 1e-3, 1e-4)

Each panel plots q^* vs lambda for Adaptive IE-VI and OS-VI, with the VI hit-time
shown as a horizontal baseline. The piecewise-linear lambda axis matches the
reference comparison (`control_stable_hit_query_vs_lambda.pdf`).

Compares algorithm rankings across threshold choices and first-hit versus
stable-hit criteria.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common.plot_style import apply_conference_matplotlib_style  # noqa: E402

VI_COLOR = "#9D9D9D"
AIEVI_COLOR = "#CA5800"
OSVI_COLOR = "#1696D2"
TEXT_COLOR = "#242424"
GRID_COLOR = "#DADADA"
RUN_BUDGET = 1000
DNF_PLOT_Y = 1680

FIG_SIZE = (5.50, 3.55)
LEGEND_BBOX = [0.325, 0.870, 0.350, 0.105]
LEGEND_LABEL_Y = 0.74
LEGEND_SWATCH_Y = 0.28
LEGEND_FONT_SIZE = 7.25

THRESHOLDS: list[float] = [1e-2, 1e-3, 1e-4]
THRESHOLD_LABELS = {
    1e-2: r"$\gamma=0.99,\ \tau=10^{-2}$",
    1e-3: r"$\gamma=0.99,\ \tau=10^{-3}$",
    1e-4: r"$\gamma=0.99,\ \tau=10^{-4}$",
}

DISPLAY_SEGMENTS: list[tuple[float, float, float]] = [
    (0.00, 0.60, 0.30),
    (0.60, 0.80, 0.45),
    (0.80, 1.00, 0.25),
]


def lambda_to_display(lam: float) -> float:
    cumulative = 0.0
    for lam_start, lam_end, width in DISPLAY_SEGMENTS:
        if lam <= lam_end + 1e-9:
            frac = (lam - lam_start) / (lam_end - lam_start)
            return cumulative + width * max(0.0, min(1.0, frac))
        cumulative += width
    return cumulative


def _legend_line(
    legend_ax: Any,
    x0: float,
    x1: float,
    y: float,
    *,
    color: str,
    linestyle: str | tuple[int, tuple[float, ...]],
    linewidth: float,
    marker: str | None = None,
    markerfacecolor: str | None = None,
    markeredgewidth: float = 0.55,
    markersize: float = 3.2,
) -> None:
    legend_ax.add_line(
        Line2D(
            [x0, x1],
            [y, y],
            transform=legend_ax.transAxes,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            markerfacecolor=markerfacecolor if markerfacecolor is not None else color,
            markeredgecolor=color if markerfacecolor == "none" else "white",
            markeredgewidth=markeredgewidth,
            solid_capstyle="round",
            zorder=8,
        )
    )


def _draw_legend_strip(fig: Any) -> None:
    legend_ax = fig.add_axes(LEGEND_BBOX, zorder=12)
    legend_ax.set_axis_off()
    entries = [
        (0.190, "Adaptive IE-VI", AIEVI_COLOR, "-", "o", AIEVI_COLOR, 0.45),
        (0.555, "OS-VI", OSVI_COLOR, (0, (3.4, 1.8)), "s", "none", 0.75),
    ]
    for x_center, label, color, linestyle, marker, markerface, markeredgewidth in entries:
        legend_ax.text(
            x_center,
            LEGEND_LABEL_Y,
            label,
            transform=legend_ax.transAxes,
            ha="center",
            va="center",
            fontsize=LEGEND_FONT_SIZE,
            color=TEXT_COLOR,
            zorder=8,
        )
        _legend_line(
            legend_ax,
            x_center - 0.052,
            x_center + 0.052,
            LEGEND_SWATCH_Y,
            color=color,
            linestyle=linestyle,
            linewidth=1.20,
            marker=marker,
            markerfacecolor=markerface,
            markeredgewidth=markeredgewidth,
            markersize=3.25,
        )

    no_hit_x = 0.840
    legend_ax.text(
        no_hit_x,
        LEGEND_LABEL_Y,
        "no hit",
        transform=legend_ax.transAxes,
        ha="center",
        va="center",
        fontsize=LEGEND_FONT_SIZE,
        color=TEXT_COLOR,
        zorder=8,
    )
    _legend_line(
        legend_ax,
        no_hit_x - 0.032,
        no_hit_x - 0.032,
        LEGEND_SWATCH_Y,
        color=AIEVI_COLOR,
        linestyle="None",
        linewidth=0.0,
        marker="^",
        markerfacecolor=AIEVI_COLOR,
        markeredgewidth=0.45,
        markersize=4.0,
    )
    _legend_line(
        legend_ax,
        no_hit_x + 0.032,
        no_hit_x + 0.032,
        LEGEND_SWATCH_Y,
        color=OSVI_COLOR,
        linestyle="None",
        linewidth=0.0,
        marker="^",
        markerfacecolor="none",
        markeredgewidth=0.75,
        markersize=4.0,
    )


def _stable_hit_query(queries: list[int], values: list[float], threshold: float) -> int | None:
    last_above = None
    for q, y in zip(queries, values, strict=True):
        if y >= threshold:
            last_above = q
    if last_above is None:
        return queries[0] if queries else None
    after = [q for q in queries if q > last_above]
    return after[0] if after else None


def _first_hit_query(queries: list[int], values: list[float], threshold: float) -> int | None:
    for q, y in zip(queries, values, strict=True):
        if y < threshold:
            return q
    return None


CRITERIA: list[tuple[str, str, Callable[[list[int], list[float], float], int | None]]] = [
    ("stable", "stable hit", _stable_hit_query),
    ("first", "first hit", _first_hit_query),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_file", type=Path, help="Path to dense control_traces.json")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path; PDF written alongside.")
    return parser.parse_args()


def _draw_panel(
    ax,
    rows: list[dict[str, Any]],
    threshold: float,
    criterion_fn: Callable[[list[int], list[float], float], int | None],
    *,
    show_y_label: bool,
    show_x_label: bool,
    title: str | None,
    show_legend: bool,
) -> dict[str, Any]:
    lambdas = [r["lambda"] for r in rows]

    aievi_hits: list[int | None] = []
    osvi_hits: list[int | None] = []
    vi_hits: list[int | None] = []
    for entry in rows:
        for alg_key, trace_name, sink in (
            ("aievi", "waterfall", aievi_hits),
            ("osvi", "osvi", osvi_hits),
            ("vi", "vi", vi_hits),
        ):
            tr = entry["traces"][trace_name]
            qs = list(tr["queries"])
            ys = list(tr["policy_value_errors"])
            sink.append(criterion_fn(qs, ys, threshold))

    # VI is monotone, so stable and first hit coincide; report the integer
    # at lambda=0 (VI is lambda-independent, so all entries agree).
    vi_value = vi_hits[0] if vi_hits and vi_hits[0] is not None else None

    ax.set_xlim(-0.025, 1.025)
    label_lambdas = [0.0, 0.6, 0.8, 1.0]
    ax.set_xticks([lambda_to_display(lam) for lam in label_lambdas])
    ax.set_xticklabels([f"${lam:g}$" for lam in label_lambdas])
    minor_lambdas = sorted(set(round(float(lam), 4) for lam in lambdas) - set(label_lambdas))
    ax.set_xticks([lambda_to_display(lam) for lam in minor_lambdas], minor=True)
    ax.tick_params(axis="x", which="minor", length=2.0, width=0.45, color="#9D9D9D")

    ax.set_yscale("log")
    ax.set_ylim(0.7, 2300)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels([r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$"])
    ax.tick_params(axis="y", which="both", labelleft=show_y_label)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="y", color=GRID_COLOR, alpha=0.55, linewidth=0.36)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(0.58)

    for boundary in (0.60, 0.80):
        ax.axvline(
            lambda_to_display(boundary),
            color="#9E9E9E",
            linestyle=(0, (1.4, 2.0)),
            linewidth=0.42,
            alpha=0.54,
            zorder=1,
        )

    if vi_value is not None and vi_value <= RUN_BUDGET:
        ax.axhline(
            vi_value,
            color=VI_COLOR,
            linestyle=(0, (1.0, 2.0)),
            linewidth=0.78,
            alpha=0.82,
            zorder=2,
        )
        ax.text(
            0.985,
            vi_value,
            f"VI: {vi_value}",
            color=VI_COLOR,
            fontsize=6.6,
            va="bottom",
            ha="right",
            transform=ax.get_yaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88),
            zorder=3,
        )

    ax.axhline(
        RUN_BUDGET,
        color="#9E9E9E",
        linestyle=(0, (1.0, 2.4)),
        linewidth=0.46,
        alpha=0.7,
        zorder=1,
    )

    aievi_lambdas = [lam for lam, h in zip(lambdas, aievi_hits, strict=True) if h is not None]
    aievi_values = [h for h in aievi_hits if h is not None]
    aievi_dnf_lambdas = [lam for lam, h in zip(lambdas, aievi_hits, strict=True) if h is None]
    osvi_lambdas = [lam for lam, h in zip(lambdas, osvi_hits, strict=True) if h is not None]
    osvi_values = [h for h in osvi_hits if h is not None]
    osvi_dnf_lambdas = [lam for lam, h in zip(lambdas, osvi_hits, strict=True) if h is None]

    ax.plot(
        [lambda_to_display(lam) for lam in aievi_lambdas],
        aievi_values,
        marker="o",
        color=AIEVI_COLOR,
        linestyle="-",
        linewidth=1.10,
        markersize=3.1,
        markerfacecolor=AIEVI_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.38,
        alpha=0.95,
        zorder=6,
    )
    ax.plot(
        [lambda_to_display(lam) for lam in osvi_lambdas],
        osvi_values,
        marker="s",
        color=OSVI_COLOR,
        linestyle=(0, (3.4, 1.8)),
        linewidth=1.00,
        markersize=3.8,
        markerfacecolor="none",
        markeredgecolor=OSVI_COLOR,
        markeredgewidth=0.75,
        alpha=0.92,
        zorder=5,
    )

    for dnf_lambdas, color, y_value, x_offset in (
        (osvi_dnf_lambdas, OSVI_COLOR, DNF_PLOT_Y, 0.007),
        (aievi_dnf_lambdas, AIEVI_COLOR, int(DNF_PLOT_Y * 0.75), -0.007),
    ):
        if dnf_lambdas:
            ax.plot(
                [lambda_to_display(lam) + x_offset for lam in dnf_lambdas],
                [y_value] * len(dnf_lambdas),
                marker="^",
                color=color,
                linestyle="None",
                markersize=4.2,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=0.75,
                alpha=0.95,
                zorder=5,
            )

    if title is not None:
        ax.set_title(title, fontsize=7.7, pad=3.0, color=TEXT_COLOR)
    if show_x_label:
        ax.set_xlabel(r"$\lambda$", fontsize=7.6)
    ax.tick_params(axis="both", which="major", labelsize=6.8, length=2.5, width=0.56)

    # Per-panel legend handled at figure level.

    return {
        "lambdas": lambdas,
        "vi_hit": vi_value,
        "aievi_hits": aievi_hits,
        "osvi_hits": osvi_hits,
    }


def render(trace_path: Path, output: Path) -> dict[str, Any]:
    trace_path = trace_path.resolve()
    with open(trace_path) as f:
        data = json.load(f)
    data.sort(key=lambda e: e["lambda"])

    apply_conference_matplotlib_style(plt, font_scale=0.94)
    plt.rcParams.update({
        "figure.constrained_layout.use": False,
        "figure.autolayout": False,
        "savefig.bbox": None,
    })
    fig, axes = plt.subplots(
        2,
        len(THRESHOLDS),
        figsize=FIG_SIZE,
        dpi=260,
        sharex=True,
        sharey=True,
        gridspec_kw={
            "left": 0.125,
            "right": 0.985,
            "top": 0.830,
            "bottom": 0.128,
            "wspace": 0.115,
            "hspace": 0.320,
        },
    )
    fig.set_layout_engine(None)

    rows: list[list[dict[str, Any]]] = [[] for _ in CRITERIA]
    for row_idx, (criterion_key, criterion_label, criterion_fn) in enumerate(CRITERIA):
        for col_idx, threshold in enumerate(THRESHOLDS):
            ax = axes[row_idx][col_idx]
            is_top = row_idx == 0
            is_bottom = row_idx == len(CRITERIA) - 1
            is_left = col_idx == 0
            is_rightmost = col_idx == len(THRESHOLDS) - 1
            title = THRESHOLD_LABELS[threshold] if is_top else None
            payload = _draw_panel(
                ax,
                data,
                threshold,
                criterion_fn,
                show_y_label=is_left,
                show_x_label=is_bottom,
                title=title,
                show_legend=(is_top and is_rightmost),
            )
            payload["criterion"] = criterion_key
            payload["threshold"] = threshold
            rows[row_idx].append(payload)

    for row_idx, (_, criterion_label, _) in enumerate(CRITERIA):
        axes[row_idx][0].annotate(
            f"{criterion_label}\nqueries",
            xy=(-0.31, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=8.3,
            rotation=90,
            color=TEXT_COLOR,
            fontweight="bold",
        )

    _draw_legend_strip(fig)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches=None, pad_inches=0)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    metadata = {
        "figure": "threshold_criterion_sensitivity",
        "thresholds": THRESHOLDS,
        "criteria": [c[0] for c in CRITERIA],
        "run_budget": RUN_BUDGET,
        "panels": rows,
        "output_png": str(output),
        "output_pdf": str(pdf_output),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = _parse_args()
    metadata = render(args.trace_file, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
