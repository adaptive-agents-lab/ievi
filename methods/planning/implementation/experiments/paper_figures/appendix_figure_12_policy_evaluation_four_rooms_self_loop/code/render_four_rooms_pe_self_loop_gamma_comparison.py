#!/usr/bin/env python3
"""Render a two-panel Four Rooms Policy Evaluation self-loop comparison."""

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

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[3]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common.plot_style import palette_accessibility_report  # noqa: E402
from stable_hit_style import (  # noqa: E402
    AIEVI_COLOR,
    DISPLAY_SEGMENTS,
    FIG_SIZE,
    GRID_COLOR,
    OSVI_COLOR,
    TEXT_COLOR,
    VI_COLOR,
    add_legend,
    annotate_vi_baseline,
    apply_style,
    display_segments_metadata,
    finalize,
    format_axis,
    lambda_to_display as stable_lambda_to_display,
    plot_series as plot_styled_series,
)

THRESHOLD = 1e-3


def lambda_to_display(lam: float) -> float:
    return stable_lambda_to_display(lam, DISPLAY_SEGMENTS)


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
    return plot_styled_series(
        ax,
        lambdas,
        values,
        color=color,
        marker=marker,
        linestyle=linestyle,
        markerfacecolor=markerfacecolor,
        dnf_y=dnf_y,
        dnf_x_offset=dnf_x_offset,
        display_segments=DISPLAY_SEGMENTS,
        zorder=zorder,
    )


def _draw_legend_strip(fig: Any) -> None:
    add_legend(fig)


def setup_axis(ax: Any, lambdas: list[float], *, run_budget: int, title: str) -> None:
    format_axis(ax, lambdas, run_budget=run_budget, title=title, display_segments=DISPLAY_SEGMENTS)


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
    apply_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=260, sharey=True)
    fig.set_layout_engine(None)

    for ax, dataset in zip(axes, datasets, strict=True):
        rows = dataset["rows"]
        lambdas = [float(row["lambda"]) for row in rows]
        setup_axis(ax, lambdas, run_budget=run_budget, title=str(dataset["label"]))

        vi_hit = rows[0]["vi_stable_hit"]
        annotate_vi_baseline(ax, vi_hit, display_segments=DISPLAY_SEGMENTS)

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

    finalize(fig, list(axes))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches=None, pad_inches=0)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    metadata = {
        "figure": "figure_app_four_rooms_pe_self_loop_gamma_comparison",
        "threshold": THRESHOLD,
        "run_budget": run_budget,
        "stable_hit_definition": "Smallest recorded query after which normalized policy-evaluation error remains below threshold.",
        "environment": "four_rooms",
        "noise_model": "self_loop",
        "display_segments": display_segments_metadata(DISPLAY_SEGMENTS),
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
