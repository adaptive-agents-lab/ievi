#!/usr/bin/env python3
"""Render a two-panel Four Rooms PE stable-hit comparison across discounts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[3]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common.plot_style import (  # noqa: E402
    apply_conference_matplotlib_style,
    palette_accessibility_report,
)

VI_COLOR = "#9D9D9D"
OSVI_COLOR = "#1696D2"
AIEVI_COLOR = "#CA5800"
TEXT_COLOR = "#242424"
GRID_COLOR = "#DADADA"

FIG_SIZE = (5.50, 2.30)
AX_LEFT = 0.102
AX_RIGHT = 0.985
AX_BOTTOM = 0.220
AX_TOP = 0.715
LEGEND_BBOX = [0.325, 0.755, 0.350, 0.210]
LEGEND_LABEL_Y = 0.76
LEGEND_SWATCH_Y = 0.44
LEGEND_FONT_SIZE = 7.35

THRESHOLD = 1e-3
DISPLAY_SEGMENTS: tuple[tuple[float, float, float], ...] = (
    (0.00, 0.60, 0.30),
    (0.60, 0.80, 0.45),
    (0.80, 1.00, 0.25),
)


def lambda_to_display(lam: float) -> float:
    cumulative = 0.0
    for lam_start, lam_end, width in DISPLAY_SEGMENTS:
        if lam <= lam_end + 1e-9:
            frac = (lam - lam_start) / (lam_end - lam_start)
            return cumulative + width * max(0.0, min(1.0, frac))
        cumulative += width
    return cumulative


def first_hit_query(queries: list[int], values: list[float], threshold: float) -> int | None:
    for query, value in zip(queries, values, strict=True):
        if math.isfinite(value) and value < threshold:
            return query
    return None


def stable_hit_query(queries: list[int], values: list[float], threshold: float) -> int | None:
    last_above = None
    for query, value in zip(queries, values, strict=True):
        if not math.isfinite(value) or value >= threshold:
            last_above = query
    if last_above is None:
        return queries[0] if queries else None
    later_queries = [query for query in queries if query > last_above]
    return later_queries[0] if later_queries else None


def load_rows(trace_path: Path, threshold: float) -> list[dict[str, Any]]:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected list trace data in {trace_path}")
    data.sort(key=lambda entry: float(entry["lambda"]))

    rows: list[dict[str, Any]] = []
    for entry in data:
        row: dict[str, Any] = {"lambda": float(entry["lambda"]), "seed": entry.get("seed")}
        for alg_key, trace_name in (
            ("vi", "vi"),
            ("osvi", "osvi"),
            ("aievi", "waterfall_besp"),
        ):
            trace = entry["traces"][trace_name]
            queries = [int(query) for query in trace["queries"]]
            errors = [float(value) for value in trace["errors"]]
            row[f"{alg_key}_first_hit"] = first_hit_query(queries, errors, threshold)
            row[f"{alg_key}_stable_hit"] = stable_hit_query(queries, errors, threshold)
            row[f"{alg_key}_last_query"] = queries[-1] if queries else None
            row[f"{alg_key}_last_error"] = errors[-1] if errors else None
            row[f"{alg_key}_max_error"] = max(errors) if errors else None
        rows.append(row)
    return rows


def contiguous_runs(
    lambdas: list[float],
    values: list[int | None],
) -> tuple[list[list[tuple[float, int]]], list[float]]:
    runs: list[list[tuple[float, int]]] = []
    current: list[tuple[float, int]] = []
    dnf_lambdas: list[float] = []
    for lam, value in zip(lambdas, values, strict=True):
        if value is None:
            if current:
                runs.append(current)
                current = []
            dnf_lambdas.append(lam)
        else:
            current.append((lam, int(value)))
    if current:
        runs.append(current)
    return runs, dnf_lambdas


def plot_series(
    ax: Any,
    lambdas: list[float],
    values: list[int | None],
    *,
    color: str,
    marker: str,
    linestyle: str | tuple[int, tuple[float, ...]],
    markerfacecolor: str,
    dnf_y: int,
    dnf_x_offset: float,
    zorder: int,
) -> list[float]:
    runs, dnf_lambdas = contiguous_runs(lambdas, values)
    for run in runs:
        ax.plot(
            [lambda_to_display(lam) for lam, _ in run],
            [value for _, value in run],
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=1.20,
            markersize=3.7,
            markerfacecolor=markerfacecolor,
            markeredgecolor=color if markerfacecolor == "none" else "white",
            markeredgewidth=0.75 if markerfacecolor == "none" else 0.45,
            alpha=0.96,
            zorder=zorder,
        )
    if dnf_lambdas:
        ax.plot(
            [lambda_to_display(lam) + dnf_x_offset for lam in dnf_lambdas],
            [dnf_y] * len(dnf_lambdas),
            marker="^",
            color=color,
            linestyle="None",
            markersize=4.4,
            markerfacecolor=markerfacecolor,
            markeredgecolor=color,
            markeredgewidth=0.75,
            alpha=0.96,
            zorder=zorder,
        )
    return dnf_lambdas


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


def setup_axis(ax: Any, lambdas: list[float], *, run_budget: int, title: str) -> None:
    label_lambdas = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    minor_lambdas = sorted(set(round(float(lam), 4) for lam in lambdas) - set(label_lambdas))
    total_width = sum(width for _, _, width in DISPLAY_SEGMENTS)

    ax.set_xlim(-0.025 * total_width, total_width + 0.025 * total_width)
    ax.set_xticks([lambda_to_display(lam) for lam in label_lambdas])
    ax.set_xticklabels([f"${lam:g}$" for lam in label_lambdas])
    ax.set_xticks([lambda_to_display(lam) for lam in minor_lambdas], minor=True)
    ax.tick_params(axis="x", which="minor", length=2.2, width=0.50, color="#9D9D9D")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 5200)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels([r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$"])
    ax.grid(True, which="major", axis="y", color=GRID_COLOR, alpha=0.55, linewidth=0.38)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(0.62)
    for _, boundary, _ in DISPLAY_SEGMENTS[:-1]:
        ax.axvline(
            lambda_to_display(boundary),
            color="#9E9E9E",
            linestyle=(0, (1.2, 2.2)),
            linewidth=0.46,
            alpha=0.55,
            zorder=1,
        )
    ax.axhline(
        run_budget,
        color="#8C8C8C",
        linestyle=(0, (1.0, 2.4)),
        linewidth=0.50,
        alpha=0.72,
        zorder=1,
    )
    ax.set_title(title, fontsize=8.4, pad=2.6, color=TEXT_COLOR)


def render(gamma09_trace: Path, gamma099_trace: Path, output: Path) -> dict[str, Any]:
    datasets = [
        {
            "key": "gamma09",
            "gamma": 0.9,
            "label": r"$\gamma=0.9$",
            "trace_file": gamma09_trace,
            "rows": load_rows(gamma09_trace, THRESHOLD),
        },
        {
            "key": "gamma099",
            "gamma": 0.99,
            "label": r"$\gamma=0.99$",
            "trace_file": gamma099_trace,
            "rows": load_rows(gamma099_trace, THRESHOLD),
        },
    ]
    run_budget = max(
        int(row["vi_last_query"])
        for dataset in datasets
        for row in dataset["rows"]
        if row["vi_last_query"] is not None
    )
    dnf_y = int(max(2600, run_budget * 2.60))

    palette_report = palette_accessibility_report([VI_COLOR, AIEVI_COLOR, OSVI_COLOR])
    apply_conference_matplotlib_style(plt, font_scale=0.95)
    plt.rcParams.update({
        "figure.constrained_layout.use": False,
        "figure.autolayout": False,
        "savefig.bbox": None,
    })
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=260, sharey=True)
    fig.set_layout_engine(None)

    for ax, dataset in zip(axes, datasets, strict=True):
        rows = dataset["rows"]
        lambdas = [float(row["lambda"]) for row in rows]
        setup_axis(ax, lambdas, run_budget=run_budget, title=str(dataset["label"]))

        vi_hit = rows[0]["vi_stable_hit"]
        ax.axhline(
            vi_hit,
            color=VI_COLOR,
            linestyle=(0, (1.0, 2.2)),
            linewidth=0.82,
            alpha=0.82,
            zorder=2,
        )
        ax.text(
            0.985,
            max(0.9, vi_hit * 0.92),
            f"VI: {vi_hit}",
            color=VI_COLOR,
            fontsize=6.6,
            va="top",
            ha="right",
            transform=ax.get_yaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88),
            zorder=3,
        )

        aievi_stable = [row["aievi_stable_hit"] for row in rows]
        osvi_stable = [row["osvi_stable_hit"] for row in rows]
        plot_series(
            ax,
            lambdas,
            aievi_stable,
            color=AIEVI_COLOR,
            marker="o",
            linestyle="-",
            markerfacecolor=AIEVI_COLOR,
            dnf_y=int(dnf_y * 0.62),
            dnf_x_offset=-0.007,
            zorder=6,
        )
        plot_series(
            ax,
            lambdas,
            osvi_stable,
            color=OSVI_COLOR,
            marker="s",
            linestyle=(0, (3.4, 1.8)),
            markerfacecolor="none",
            dnf_y=dnf_y,
            dnf_x_offset=0.007,
            zorder=5,
        )

    _draw_legend_strip(fig)

    axes[0].set_ylabel("stable-hit query\n$q_{\\mathrm{SH}}(10^{-3})$", fontsize=7.6, labelpad=1.4)
    for ax in axes:
        ax.set_xlabel(r"$\lambda$", fontsize=7.6)
        ax.tick_params(axis="both", which="major", labelsize=7.1, length=2.6, width=0.58)

    fig.subplots_adjust(left=AX_LEFT, right=AX_RIGHT, bottom=AX_BOTTOM, top=AX_TOP, wspace=0.145)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches=None, pad_inches=0)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    metadata = {
        "figure": "figure_app_four_rooms_pe_gamma_comparison",
        "threshold": THRESHOLD,
        "run_budget": run_budget,
        "stable_hit_definition": "Smallest recorded query after which normalized policy-evaluation error remains below threshold.",
        "environment": "four_rooms",
        "noise_model": "support_uniform",
        "display_segments": [
            {"lambda_start": start, "lambda_end": end, "width": width}
            for start, end, width in DISPLAY_SEGMENTS
        ],
        "datasets": [
            {
                **{key: value for key, value in dataset.items() if key != "rows"},
                "trace_file": str(Path(dataset["trace_file"]).resolve()),
                "rows": dataset["rows"],
            }
            for dataset in datasets
        ],
        "palette_accessibility_report": palette_report,
        "output_png": str(output.resolve()),
        "output_pdf": str(pdf_output.resolve()),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma09-trace", type=Path, required=True)
    parser.add_argument("--gamma099-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = render(args.gamma09_trace, args.gamma099_trace, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
