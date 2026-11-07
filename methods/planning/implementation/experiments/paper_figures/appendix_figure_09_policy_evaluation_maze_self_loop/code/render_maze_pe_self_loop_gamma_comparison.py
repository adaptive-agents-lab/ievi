#!/usr/bin/env python3
"""Render a two-panel Maze PE self-loop stable-hit comparison across discounts."""

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
            linewidth=1.35,
            markersize=4.7,
            markerfacecolor=markerfacecolor,
            markeredgecolor=color if markerfacecolor == "none" else "white",
            markeredgewidth=1.0 if markerfacecolor == "none" else 0.55,
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
            markersize=5.8,
            markerfacecolor=markerfacecolor,
            markeredgecolor=color,
            markeredgewidth=1.0,
            alpha=0.96,
            zorder=zorder,
        )
    return dnf_lambdas


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
    y_top = max(2300, run_budget * 2.25)
    ax.set_ylim(0.7, y_top)
    y_ticks = [tick for tick in [1, 10, 100, 1000, 10000] if tick <= y_top / 1.3]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([rf"$10^{index}$" for index, _ in enumerate(y_ticks)])
    ax.grid(True, which="major", axis="y", color="#B8B8B8", alpha=0.24, linewidth=0.50)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(0.75)
    for _, boundary, _ in DISPLAY_SEGMENTS[:-1]:
        ax.axvline(
            lambda_to_display(boundary),
            color="#9E9E9E",
            linestyle=(0, (1.2, 2.2)),
            linewidth=0.52,
            alpha=0.62,
            zorder=1,
        )
    ax.axhline(
        run_budget,
        color="#8C8C8C",
        linestyle=(0, (1.0, 2.4)),
        linewidth=0.58,
        alpha=0.72,
        zorder=1,
    )
    ax.set_title(title, fontsize=9.0, pad=3.2)


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
    dnf_y = int(max(1680, run_budget * 1.68))

    palette_report = palette_accessibility_report([VI_COLOR, AIEVI_COLOR, OSVI_COLOR])
    apply_conference_matplotlib_style(plt, font_scale=0.95)
    plt.rcParams.update({"figure.constrained_layout.use": False})
    fig, axes = plt.subplots(1, 2, figsize=(5.43, 2.575), dpi=240, sharey=True)

    for ax, dataset in zip(axes, datasets, strict=True):
        rows = dataset["rows"]
        lambdas = [float(row["lambda"]) for row in rows]
        setup_axis(ax, lambdas, run_budget=run_budget, title=str(dataset["label"]))

        vi_hit = rows[0]["vi_stable_hit"]
        ax.axhline(
            vi_hit,
            color=VI_COLOR,
            linestyle=(0, (1.0, 2.2)),
            linewidth=1.03,
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

        aievi_stable = [row["aievi_stable_hit"] for row in rows]
        osvi_stable = [row["osvi_stable_hit"] for row in rows]
        aievi_dnf = plot_series(
            ax,
            lambdas,
            aievi_stable,
            color=AIEVI_COLOR,
            marker="o",
            linestyle="-",
            markerfacecolor=AIEVI_COLOR,
            dnf_y=dnf_y - 260,
            zorder=6,
        )
        osvi_dnf = plot_series(
            ax,
            lambdas,
            osvi_stable,
            color=OSVI_COLOR,
            marker="s",
            linestyle=(0, (3.4, 1.8)),
            markerfacecolor="none",
            dnf_y=dnf_y,
            zorder=5,
        )

    axes[0].set_ylabel("stable-hit query\n$q_{\mathrm{SH}}(10^{-3})$", fontsize=8.6)
    for ax in axes:
        ax.set_xlabel(r"$\lambda$", fontsize=8.6)
        ax.tick_params(axis="both", which="major", labelsize=7.6)

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
    fig.subplots_adjust(left=0.102, right=0.990, bottom=0.228, top=0.800, wspace=0.145)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output)
    plt.close(fig)

    metadata = {
        "figure": "figure_app_maze_pe_self_loop_gamma_comparison",
        "threshold": THRESHOLD,
        "run_budget": run_budget,
        "stable_hit_definition": "Smallest recorded query after which normalized policy-evaluation error remains below threshold.",
        "environment": "maze",
        "noise_model": "self_loop",
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
