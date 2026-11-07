#!/usr/bin/env python3
"""Render stable-hit query versus lambda for deterministic-control AISP.

The plotted quantity is computed from ``control_traces.json``. For each
algorithm and lambda, the stable-hit query is the smallest recorded query after
which the policy-value error remains below the target threshold. No smoothing
or interpolation is applied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common.plot_style import (  # noqa: E402
    apply_conference_matplotlib_style,
    palette_accessibility_report,
)

VI_COLOR = "#5A5A5A"
AIEVI_COLOR = "#D55E00"
OSVI_COLOR = "#0072B2"
THRESHOLD = 1e-3
RUN_BUDGET = 1000
AIEVI_DNF_Y = 1320
OSVI_DNF_Y = 1680

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
    last_above = None
    for query, value in zip(queries, values, strict=True):
        if value >= threshold:
            last_above = query
    if last_above is None:
        return queries[0] if queries else None
    later_queries = [query for query in queries if query > last_above]
    return later_queries[0] if later_queries else None


def first_hit_query(queries: list[int], values: list[float], threshold: float) -> int | None:
    for query, value in zip(queries, values, strict=True):
        if value < threshold:
            return query
    return None


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
            row[f"{alg_key}_last_error"] = values[-1] if values else None
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
    finite_points = [(lam, value) for lam, value in zip(lambdas, values, strict=True) if value is not None]
    dnf_lambdas = [lam for lam, value in zip(lambdas, values, strict=True) if value is None]
    if finite_points:
        finite_lams = [lam for lam, _ in finite_points]
        finite_values = [float(value) for _, value in finite_points]
        ax.plot(
            [lambda_to_display(lam) for lam in finite_lams],
            finite_values,
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

    palette = [VI_COLOR, AIEVI_COLOR, OSVI_COLOR]
    palette_report = palette_accessibility_report(palette)

    apply_conference_matplotlib_style(plt, font_scale=0.92)
    fig = plt.figure(figsize=(5.50, 2.55), dpi=240)
    ax = fig.add_axes([0.108, 0.180, 0.875, 0.74])

    ax.set_xlim(-0.025, 1.025)
    label_lambdas = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_xticks([lambda_to_display(lam) for lam in label_lambdas])
    ax.set_xticklabels([f"${lam:g}$" for lam in label_lambdas])
    minor_lambdas = sorted(set(round(float(lam), 4) for lam in lambdas) - set(label_lambdas))
    ax.set_xticks([lambda_to_display(lam) for lam in minor_lambdas], minor=True)
    ax.tick_params(axis="x", which="minor", length=2.4, width=0.55, color="#5A5A5A")

    ax.set_yscale("log")
    ax.set_ylim(0.7, 2300)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["$1$", "$10$", "$100$", "$1000$"])
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
        lambda_to_display(0.0) + 0.012,
        vi_hit * 1.18,
        f"VI ($q^\\star={vi_hit}$)",
        color=VI_COLOR,
        fontsize=8.2,
        va="bottom",
        ha="left",
    )

    ax.axhline(
        RUN_BUDGET,
        color="#8C8C8C",
        linestyle=(0, (1.0, 2.4)),
        linewidth=0.60,
        alpha=0.72,
        zorder=1,
    )
    ax.text(
        lambda_to_display(0.03),
        RUN_BUDGET * 1.05,
        "budget",
        color="#6A6A6A",
        fontsize=7.2,
        va="bottom",
        ha="left",
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

    ax.text(
        lambda_to_display(0.705),
        235,
        "Adaptive IE-VI\n(target $10^{-3}$)",
        color=AIEVI_COLOR,
        fontsize=8.1,
        va="bottom",
        ha="left",
        linespacing=0.92,
    )
    ax.text(
        lambda_to_display(0.665),
        44,
        "OS-VI",
        color=OSVI_COLOR,
        fontsize=8.6,
        va="bottom",
        ha="right",
        fontweight="bold",
    )
    ax.text(
        lambda_to_display(0.91),
        OSVI_DNF_Y * 1.12,
        "no stable hit by $q=1000$",
        color="#4D4D4D",
        fontsize=7.4,
        va="bottom",
        ha="center",
    )

    ax.set_xlabel(r"surrogate-error scaling $\lambda$ (piecewise scale)", fontsize=9.0)
    ax.set_ylabel("queries to stable\nthreshold crossing", fontsize=9.0)
    ax.tick_params(axis="both", which="major", labelsize=8.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output)
    plt.close(fig)

    metadata = {
        "figure": "planning_stable_hit_query_vs_lambda",
        "trace_file": str(trace_path),
        "threshold": THRESHOLD,
        "run_budget": RUN_BUDGET,
        "stable_hit_definition": "Smallest recorded query after which policy_value_error remains below threshold.",
        "vi_stable_hit_query": vi_hit,
        "palette_accessibility_report": palette_report,
        "rows": rows,
        "output_png": str(output.resolve()),
        "output_pdf": str(pdf_output.resolve()),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    metadata = render(args.trace_file, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
