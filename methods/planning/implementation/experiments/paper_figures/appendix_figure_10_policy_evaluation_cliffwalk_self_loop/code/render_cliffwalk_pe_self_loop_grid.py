#!/usr/bin/env python3
"""Render the four-panel Cliffwalk Policy Evaluation self-loop figure."""

from __future__ import annotations

import argparse
import csv
import json
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
AIEVI_COLOR = "#CA5800"
OSVI_COLOR = "#1696D2"
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


def parse_hit(raw_value: str, *, run_budget: int) -> int | None:
    if raw_value == "":
        return None
    value = int(float(raw_value))
    if value > run_budget:
        return None
    return value


def load_rows(csv_path: Path, *, run_budget: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "lambda": float(row["lambda"]),
                    "seed": int(float(row["seed"])),
                    "vi_stable_hit": parse_hit(row["vi"], run_budget=run_budget),
                    "osvi_stable_hit": parse_hit(row["osvi"], run_budget=run_budget),
                    "aievi_stable_hit": parse_hit(row["waterfall_besp"], run_budget=run_budget),
                }
            )
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
    dnf_y: float,
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
            markersize=3.9,
            markerfacecolor=markerfacecolor,
            markeredgecolor=color if markerfacecolor == "none" else "white",
            markeredgewidth=0.9 if markerfacecolor == "none" else 0.48,
            alpha=0.96,
            zorder=zorder,
        )
    if dnf_lambdas:
        ax.plot(
            [lambda_to_display(lam) for lam in dnf_lambdas],
            [dnf_y] * len(dnf_lambdas),
            marker="^",
            color=color,
            linestyle="None",
            markersize=4.8,
            markerfacecolor=markerfacecolor,
            markeredgecolor=color,
            markeredgewidth=0.9,
            alpha=0.96,
            zorder=zorder,
        )
    return dnf_lambdas


def setup_axis(ax: Any, *, run_budget: int, title: str) -> None:
    total_width = sum(width for _, _, width in DISPLAY_SEGMENTS)
    label_lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xlim(-0.025 * total_width, total_width + 0.025 * total_width)
    ax.set_xticks([lambda_to_display(lam) for lam in label_lambdas])
    ax.set_xticklabels([f"${lam:g}$" for lam in label_lambdas])
    ax.set_yscale("log")
    y_top = 5600 if run_budget == 1000 else 56000
    ax.set_ylim(0.78, y_top)
    y_ticks = [tick for tick in (1, 10, 100, 1000, 10000) if tick < y_top]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([rf"$10^{index}$" for index, _ in enumerate(y_ticks)])
    ax.grid(True, which="major", axis="y", color="#B8B8B8", alpha=0.24, linewidth=0.48)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(0.72)
    for _, boundary, _ in DISPLAY_SEGMENTS[:-1]:
        ax.axvline(
            lambda_to_display(boundary),
            color="#9E9E9E",
            linestyle=(0, (1.2, 2.2)),
            linewidth=0.50,
            alpha=0.62,
            zorder=1,
        )
    ax.axhline(
        run_budget,
        color="#8C8C8C",
        linestyle=(0, (1.0, 2.4)),
        linewidth=0.56,
        alpha=0.72,
        zorder=1,
    )
    ax.set_title(title, fontsize=7.7, pad=3.0)


def add_no_hit_label(ax: Any, dnf_lambdas: list[float], *, run_budget: int) -> None:
    return


def plot_panel(ax: Any, panel: dict[str, Any]) -> None:
    rows = panel["rows"]
    lambdas = [row["lambda"] for row in rows]
    run_budget = int(panel["run_budget"])
    setup_axis(ax, run_budget=run_budget, title=str(panel["title"]))

    vi_hits = [row["vi_stable_hit"] for row in rows]
    finite_vi = [hit for hit in vi_hits if hit is not None]
    if finite_vi and len(set(finite_vi)) == 1 and len(finite_vi) == len(vi_hits):
        vi_value = finite_vi[0]
        ax.axhline(
            vi_value,
            color=VI_COLOR,
            linestyle=(0, (1.0, 2.2)),
            linewidth=0.96,
            alpha=0.86,
            zorder=2,
        )
        ax.text(
            0.985,
            max(0.9, vi_value * 0.92),
            f"VI: {vi_value}",
            color=VI_COLOR,
            fontsize=6.6,
            va="top",
            ha="right",
            transform=ax.get_yaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=1.0),
            zorder=10,
        )
    elif finite_vi:
        plot_series(
            ax,
            lambdas,
            vi_hits,
            color=VI_COLOR,
            marker="^",
            linestyle=(0, (1.0, 2.2)),
            markerfacecolor="none",
            dnf_y=run_budget * 1.18,
            zorder=3,
        )
    else:
        ax.axhline(
            run_budget,
            color=VI_COLOR,
            linestyle=(0, (1.0, 2.2)),
            linewidth=0.96,
            alpha=0.86,
            zorder=2,
        )
        ax.text(
            0.985,
            max(0.9, run_budget * 0.92),
            rf"VI $>$ {run_budget}",
            color=VI_COLOR,
            fontsize=6.6,
            va="top",
            ha="right",
            transform=ax.get_yaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=1.0),
            zorder=10,
        )

    plot_series(
        ax,
        lambdas,
        [row["aievi_stable_hit"] for row in rows],
        color=AIEVI_COLOR,
        marker="o",
        linestyle="-",
        markerfacecolor=AIEVI_COLOR,
        dnf_y=run_budget * 1.55,
        zorder=6,
    )
    plot_series(
        ax,
        lambdas,
        [row["osvi_stable_hit"] for row in rows],
        color=OSVI_COLOR,
        marker="s",
        linestyle=(0, (3.4, 1.8)),
        markerfacecolor="none",
        dnf_y=run_budget * 2.65,
        zorder=5,
    )


def render(
    gamma099_budget1000: Path,
    gamma099_budget10000: Path,
    gamma0999_budget1000: Path,
    gamma0999_budget10000: Path,
    output: Path,
) -> dict[str, Any]:
    panels = [
        {
            "key": "gamma099_budget1000",
            "title": r"(a) $\gamma=0.99$, budget 1000",
            "run_budget": 1000,
            "csv_file": gamma099_budget1000,
        },
        {
            "key": "gamma099_budget10000",
            "title": r"(b) $\gamma=0.99$, budget 10000",
            "run_budget": 10000,
            "csv_file": gamma099_budget10000,
        },
        {
            "key": "gamma0999_budget1000",
            "title": r"(c) $\gamma=0.999$, budget 1000",
            "run_budget": 1000,
            "csv_file": gamma0999_budget1000,
        },
        {
            "key": "gamma0999_budget10000",
            "title": r"(d) $\gamma=0.999$, budget 10000",
            "run_budget": 10000,
            "csv_file": gamma0999_budget10000,
        },
    ]
    for panel in panels:
        panel["rows"] = load_rows(Path(panel["csv_file"]), run_budget=int(panel["run_budget"]))

    palette_report = palette_accessibility_report([VI_COLOR, AIEVI_COLOR, OSVI_COLOR])
    apply_conference_matplotlib_style(plt, font_scale=0.92)
    plt.rcParams.update({"figure.constrained_layout.use": False})
    fig, axes = plt.subplots(2, 2, figsize=(5.50, 4.20), dpi=240, sharex=True)

    for ax, panel in zip(axes.ravel(), panels, strict=True):
        plot_panel(ax, panel)
        ax.tick_params(axis="both", which="major", labelsize=6.8)

    for ax in axes[:, 0]:
        ax.set_ylabel("stable-hit query\n$q_{\mathrm{SH}}(10^{-3})$", fontsize=7.8)
    for ax in axes[1, :]:
        ax.set_xlabel(r"$\lambda$", fontsize=7.8)

    from matplotlib.legend_handler import HandlerTuple

    aievi_line = Line2D(
        [0], [0], color=AIEVI_COLOR, marker="o", linestyle="-",
        markersize=4.0, markerfacecolor=AIEVI_COLOR,
        markeredgecolor="white", linewidth=1.2, label="Adaptive IE-VI",
    )
    osvi_line = Line2D(
        [0], [0], color=OSVI_COLOR, marker="s", linestyle=(0, (3.4, 1.8)),
        markersize=4.4, markerfacecolor="none",
        markeredgecolor=OSVI_COLOR, linewidth=1.15, label="OS-VI",
    )
    vi_line = Line2D(
        [0], [0], color=VI_COLOR, linestyle=(0, (1.0, 2.2)),
        linewidth=0.96, label="VI",
    )
    aievi_dnf = Line2D(
        [0], [0], color=AIEVI_COLOR, marker="^", linestyle="None",
        markersize=4.8, markerfacecolor=AIEVI_COLOR,
        markeredgecolor=AIEVI_COLOR, markeredgewidth=0.55,
    )
    osvi_dnf = Line2D(
        [0], [0], color=OSVI_COLOR, marker="^", linestyle="None",
        markersize=4.8, markerfacecolor="none",
        markeredgecolor=OSVI_COLOR, markeredgewidth=0.95,
    )
    primary_handles = [aievi_line, osvi_line, vi_line]
    primary_labels = ["Adaptive IE-VI", "OS-VI", "VI"]
    primary_legend = fig.legend(
        handles=primary_handles,
        labels=primary_labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015, 0.86, 0.10),
        mode="expand",
        ncol=len(primary_handles),
        fontsize=6.5,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.2,
        columnspacing=1.0,
        frameon=False,
    )
    fig.add_artist(primary_legend)
    fig.legend(
        handles=[(aievi_dnf, osvi_dnf)],
        labels=["no hit"],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015, 0.0, 0.10),
        fontsize=6.9,
        handlelength=2.0,
        handletextpad=0.45,
        borderpad=0.2,
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.7)},
    )
    fig.subplots_adjust(left=0.105, right=0.990, bottom=0.115, top=0.865, wspace=0.170, hspace=0.330)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output)
    plt.close(fig)

    metadata = {
        "figure": "figure_app_cliffwalk_pe_self_loop_grid",
        "threshold": THRESHOLD,
        "environment": "cliffwalk",
        "regime": "policy_evaluation",
        "noise_model": "self_loop",
        "stable_hit_definition": "CSV values above the recording budget denote a no-hit setting.",
        "display_segments": [
            {"lambda_start": start, "lambda_end": end, "width": width}
            for start, end, width in DISPLAY_SEGMENTS
        ],
        "panels": [
            {
                "key": panel["key"],
                "title": panel["title"],
                "run_budget": panel["run_budget"],
                "csv_file": str(Path(panel["csv_file"]).resolve()),
                "rows": panel["rows"],
            }
            for panel in panels
        ],
        "palette_accessibility_report": palette_report,
        "output_png": str(output.resolve()),
        "output_pdf": str(pdf_output.resolve()),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma099-budget1000", type=Path, required=True)
    parser.add_argument("--gamma099-budget10000", type=Path, required=True)
    parser.add_argument("--gamma0999-budget1000", type=Path, required=True)
    parser.add_argument("--gamma0999-budget10000", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = render(
        args.gamma099_budget1000,
        args.gamma099_budget10000,
        args.gamma0999_budget1000,
        args.gamma0999_budget10000,
        args.output,
    )
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
