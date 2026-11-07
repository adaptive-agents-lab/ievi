#!/usr/bin/env python3
"""Render stable-hit query versus lambda for Control maze AISP.

The plotted quantity is computed from ``control_traces.json``. For each
algorithm and lambda, the stable-hit query is the smallest recorded query after
which the policy-value error remains below the target threshold. Non-finite
values are treated as above threshold. No smoothing or interpolation is
applied.
"""

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
AIEVI_COLOR = "#CA5800"
OSVI_COLOR = "#1696D2"
THRESHOLD = 1e-3
RUN_BUDGET = 10000
AIEVI_DNF_Y = 11500
OSVI_DNF_Y = 19500

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


def stable_hit_query(queries: list[int], values: list[float], threshold: float) -> int | None:
    last_bad = None
    for query, value in zip(queries, values, strict=True):
        if (not math.isfinite(value)) or value >= threshold:
            last_bad = query
    if last_bad is None:
        return queries[0] if queries else None
    later_queries = [query for query in queries if query > last_bad]
    return later_queries[0] if later_queries else None


def first_hit_query(queries: list[int], values: list[float], threshold: float) -> int | None:
    for query, value in zip(queries, values, strict=True):
        if math.isfinite(value) and value < threshold:
            return query
    return None


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_file", type=Path, help="Path to control_traces.json")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path; PDF is written beside it.")
    return parser.parse_args()


def trace_rows(trace_path: Path) -> list[dict[str, Any]]:
    with trace_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    data.sort(key=lambda entry: entry["lambda"])

    rows: list[dict[str, Any]] = []
    for entry in data:
        row: dict[str, Any] = {"lambda": float(entry["lambda"])}
        for alg_key, trace_name in [("vi", "vi"), ("aievi", "aisp"), ("osvi", "osvi")]:
            trace = entry["traces"][trace_name]
            queries = [int(query) for query in trace["queries"]]
            values = [float(value) for value in trace["policy_value_errors"]]
            row[f"{alg_key}_first_hit"] = first_hit_query(queries, values, THRESHOLD)
            row[f"{alg_key}_stable_hit"] = stable_hit_query(queries, values, THRESHOLD)
            row[f"{alg_key}_recorded_queries"] = len(queries)
            row[f"{alg_key}_last_query"] = queries[-1] if queries else None
            row[f"{alg_key}_last_error"] = finite_or_none(values[-1] if values else None)
            row[f"{alg_key}_nonfinite_policy_value_errors"] = sum(
                1 for value in values if not math.isfinite(value)
            )
        rows.append(row)
    return rows


def plot_series_with_dnf(
    ax: Any,
    lambdas: list[float],
    values: list[int | None],
    *,
    color: str,
    marker: str,
    linestyle: str | tuple[int, tuple[float, ...]],
    dnf_y: int,
    markerfacecolor: str,
    label_y_scale: float,
) -> tuple[list[float], list[float]]:
    dnf_lambdas = [lam for lam, value in zip(lambdas, values, strict=True) if value is None]

    segment_lams: list[float] = []
    segment_values: list[float] = []

    def flush_segment() -> None:
        if not segment_lams:
            return
        ax.plot(
            [lambda_to_display(lam) for lam in segment_lams],
            segment_values,
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=1.35,
            markersize=4.8,
            markerfacecolor=markerfacecolor,
            markeredgecolor=color if markerfacecolor == "none" else "white",
            markeredgewidth=1.0 if markerfacecolor == "none" else 0.55,
            alpha=0.96,
            zorder=5,
        )
        segment_lams.clear()
        segment_values.clear()

    for lam, value in zip(lambdas, values, strict=True):
        if value is None:
            flush_segment()
            continue
        segment_lams.append(lam)
        segment_values.append(float(value))
    flush_segment()

    if dnf_lambdas:
        ax.plot(
            [lambda_to_display(lam) for lam in dnf_lambdas],
            [dnf_y] * len(dnf_lambdas),
            marker="^",
            color=color,
            linestyle="None",
            markersize=5.6,
            markerfacecolor=markerfacecolor,
            markeredgecolor=color,
            markeredgewidth=1.0,
            alpha=0.96,
            zorder=5,
        )
    return (
        [lambda_to_display(lam) for lam in dnf_lambdas],
        [dnf_y * label_y_scale for _ in dnf_lambdas],
    )


def render(trace_path: Path, output: Path) -> dict[str, Any]:
    trace_path = trace_path.resolve()
    rows = trace_rows(trace_path)
    lambdas = [row["lambda"] for row in rows]
    vi_hit = rows[0]["vi_stable_hit"]
    aievi_stable = [row["aievi_stable_hit"] for row in rows]
    osvi_stable = [row["osvi_stable_hit"] for row in rows]
    has_dnf = any(value is None for value in aievi_stable + osvi_stable)

    palette = [VI_COLOR, AIEVI_COLOR, OSVI_COLOR]
    palette_report = palette_accessibility_report(palette)

    apply_conference_matplotlib_style(plt, font_scale=0.92)
    plt.rcParams.update({"figure.constrained_layout.use": False})
    fig = plt.figure(figsize=(5.50, 2.45), dpi=240)
    ax = fig.add_axes([0.108, 0.185, 0.875, 0.650])

    ax.set_xlim(-0.025, 1.025)
    label_lambdas = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_xticks([lambda_to_display(lam) for lam in label_lambdas])
    ax.set_xticklabels([f"${lam:g}$" for lam in label_lambdas])
    minor_lambdas = sorted(set(round(float(lam), 4) for lam in lambdas) - set(label_lambdas))
    ax.set_xticks([lambda_to_display(lam) for lam in minor_lambdas], minor=True)
    ax.tick_params(axis="x", which="minor", length=2.4, width=0.55, color="#9D9D9D")

    ax.set_yscale("log")
    if has_dnf:
        ax.set_ylim(0.7, 36000)
        ax.set_yticks([1, 10, 100, 1000, 10000])
        ax.set_yticklabels([r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"])
    else:
        ax.set_ylim(0.8, 80)
        ax.set_yticks([1, 3, 10, 30])
        ax.set_yticklabels(["$1$", "$3$", "$10$", "$30$"])
    ax.grid(True, which="major", axis="y", color="#B8B8B8", alpha=0.24, linewidth=0.50)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(0.78)

    for boundary in (0.60, 0.80):
        ax.axvline(
            lambda_to_display(boundary),
            color="#9E9E9E",
            linestyle=(0, (1.2, 2.2)),
            linewidth=0.55,
            alpha=0.62,
            zorder=1,
        )

    ax.axhline(
        vi_hit,
        color=VI_COLOR,
        linestyle=(0, (1.0, 2.2)),
        linewidth=1.05,
        alpha=0.86,
        zorder=2,
    )
    ax.text(
        0.985,
        vi_hit,
        f"VI: {vi_hit}",
        color=VI_COLOR,
        fontsize=6.6,
        va="bottom",
        ha="right",
        transform=ax.get_yaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88),
        zorder=3,
    )

    if has_dnf:
        ax.axhline(
            RUN_BUDGET,
            color="#8C8C8C",
            linestyle=(0, (1.0, 2.4)),
            linewidth=0.60,
            alpha=0.72,
            zorder=1,
        )

    plot_series_with_dnf(
        ax,
        lambdas,
        aievi_stable,
        color=AIEVI_COLOR,
        marker="o",
        linestyle="-",
        dnf_y=AIEVI_DNF_Y,
        markerfacecolor=AIEVI_COLOR,
        label_y_scale=1.0,
    )
    plot_series_with_dnf(
        ax,
        lambdas,
        osvi_stable,
        color=OSVI_COLOR,
        marker="s",
        linestyle=(0, (3.4, 1.8)),
        dnf_y=OSVI_DNF_Y,
        markerfacecolor="none",
        label_y_scale=1.0,
    )

    ax.set_title(r"$\gamma=0.99$", fontsize=9.0, pad=3.2)

    ax.set_xlabel(r"$\lambda$", fontsize=9.0)
    ax.set_ylabel("stable-hit query\n$q_{\mathrm{SH}}(10^{-3})$", fontsize=9.0)
    ax.tick_params(axis="both", which="major", labelsize=8.3)

    from matplotlib.legend_handler import HandlerTuple

    aievi_line = Line2D(
        [0], [0], color=AIEVI_COLOR, marker="o", linestyle="-",
        markersize=4.6, markerfacecolor=AIEVI_COLOR,
        markeredgecolor="white", linewidth=1.35, label="Adaptive IE-VI",
    )
    osvi_line = Line2D(
        [0], [0], color=OSVI_COLOR, marker="s", linestyle=(0, (3.4, 1.8)),
        markersize=5.0, markerfacecolor="none",
        markeredgecolor=OSVI_COLOR, linewidth=1.25, label="OS-VI",
    )
    vi_line = Line2D(
        [0], [0], color=VI_COLOR, linestyle=(0, (1.0, 2.2)),
        linewidth=1.05, label="VI",
    )
    aievi_dnf = Line2D(
        [0], [0], color=AIEVI_COLOR, marker="^", linestyle="None",
        markersize=5.4, markerfacecolor=AIEVI_COLOR,
        markeredgecolor=AIEVI_COLOR, markeredgewidth=0.55,
    )
    osvi_dnf = Line2D(
        [0], [0], color=OSVI_COLOR, marker="^", linestyle="None",
        markersize=5.4, markerfacecolor="none",
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

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output)
    plt.close(fig)

    metadata = {
        "figure": "figure_app_maze_control_self_loop_gamma099",
        "environment": "maze",
        "error_mode": "self_loop",
        "gamma": 0.99,
        "trace_file": str(trace_path),
        "threshold": THRESHOLD,
        "run_budget": RUN_BUDGET,
        "stable_hit_definition": "Smallest recorded query after which finite policy_value_error remains below threshold; non-finite values count as failures.",
        "vi_stable_hit_query": vi_hit,
        "palette_accessibility_report": palette_report,
        "rows": rows,
        "output_png": str(output.resolve()),
        "output_pdf": str(pdf_output.resolve()),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = render(args.trace_file, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
