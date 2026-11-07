#!/usr/bin/env python3
"""Render deterministic-control stable-hit query curves for two discounts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common.plot_style import palette_accessibility_report  # noqa: E402
from common.stable_hit_style import (  # noqa: E402
    AIEVI_COLOR,
    DISPLAY_SEGMENTS as STABLE_DISPLAY_SEGMENTS,
    FIG_SIZE,
    GRID_COLOR,
    OSVI_COLOR,
    TEXT_COLOR,
    VI_COLOR,
    add_legend,
    annotate_vi_baseline,
    apply_style,
    finalize,
    format_axis,
    lambda_to_display as stable_lambda_to_display,
    plot_series as plot_styled_series,
)

DEFAULT_DISPLAY_SEGMENTS = list(STABLE_DISPLAY_SEGMENTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="Path to the figure spec JSON.")
    parser.add_argument("--output", type=Path, help="Optional PNG output override.")
    return parser.parse_args()


def coerce_display_segments(raw_segments: Any | None) -> list[tuple[float, float, float]]:
    if raw_segments is None:
        return DEFAULT_DISPLAY_SEGMENTS
    segments: list[tuple[float, float, float]] = []
    for segment in raw_segments:
        if isinstance(segment, dict):
            segments.append(
                (
                    float(segment["lambda_start"]),
                    float(segment["lambda_end"]),
                    float(segment["width"]),
                )
            )
        else:
            lam_start, lam_end, width = segment
            segments.append((float(lam_start), float(lam_end), float(width)))
    return segments


def lambda_to_display(lam: float, display_segments: list[tuple[float, float, float]]) -> float:
    return stable_lambda_to_display(lam, display_segments)


def first_hit_query(queries: list[int], values: list[float], threshold: float) -> int | None:
    for query, value in zip(queries, values, strict=True):
        if value < threshold:
            return query
    return None


def stable_hit_query(queries: list[int], values: list[float], threshold: float) -> int | None:
    last_above = None
    for query, value in zip(queries, values, strict=True):
        if value >= threshold:
            last_above = query
    if last_above is None:
        return queries[0] if queries else None
    later_queries = [query for query in queries if query > last_above]
    return later_queries[0] if later_queries else None


def load_rows(trace_path: Path, threshold: float) -> list[dict[str, Any]]:
    with trace_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"expected list trace data in {trace_path}")
    data.sort(key=lambda entry: float(entry["lambda"]))

    rows: list[dict[str, Any]] = []
    for entry in data:
        row: dict[str, Any] = {"lambda": float(entry["lambda"]), "seed": entry.get("seed")}
        for alg_key, trace_name in [("vi", "vi"), ("aievi", "waterfall"), ("osvi", "osvi")]:
            try:
                trace = entry["traces"][trace_name]
            except KeyError as exc:
                raise KeyError(
                    f"algorithm {trace_name!r} missing for lambda={entry.get('lambda')}"
                ) from exc
            queries = [int(query) for query in trace["queries"]]
            values = [float(value) for value in trace["policy_value_errors"]]
            row[f"{alg_key}_first_hit"] = first_hit_query(queries, values, threshold)
            row[f"{alg_key}_stable_hit"] = stable_hit_query(queries, values, threshold)
            row[f"{alg_key}_last_query"] = queries[-1] if queries else None
            row[f"{alg_key}_last_error"] = values[-1] if values else None
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
    display_segments: list[tuple[float, float, float]],
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
        display_segments=display_segments,
    )


def _draw_legend_strip(fig: Any) -> None:
    add_legend(fig)


def setup_axis(
    ax: Any,
    lambdas: list[float],
    *,
    run_budget: int,
    title: str,
    display_segments: list[tuple[float, float, float]],
) -> None:
    format_axis(ax, lambdas, run_budget=run_budget, title=title, display_segments=display_segments)


def render(spec_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    exp_dir = spec_path.parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    threshold = float(spec.get("threshold", 1e-3))
    run_budget = int(spec.get("run_budget", 1000))
    dnf_y = int(max(int(spec.get("dnf_y", 1680)), run_budget * 2.60, 2600))
    output = output_override or exp_dir / spec["output_path"]
    if not output.is_absolute():
        output = exp_dir / output

    datasets = []
    for dataset in spec["datasets"]:
        trace_path = Path(dataset["trace_file"])
        if not trace_path.is_absolute():
            trace_path = exp_dir / trace_path
        rows = load_rows(trace_path, threshold)
        display_segments = coerce_display_segments(dataset.get("display_segments"))
        datasets.append(
            {**dataset, "trace_file": str(trace_path), "display_segments": display_segments, "rows": rows}
        )

    palette = [VI_COLOR, AIEVI_COLOR, OSVI_COLOR]
    palette_report = palette_accessibility_report(palette)

    apply_style(plt)
    fig, axes = plt.subplots(1, len(datasets), figsize=FIG_SIZE, dpi=260, sharey=True)
    fig.set_layout_engine(None)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets, strict=True):
        rows = dataset["rows"]
        lambdas = [float(row["lambda"]) for row in rows]
        display_segments = dataset["display_segments"]
        setup_axis(
            ax,
            lambdas,
            run_budget=run_budget,
            title=str(dataset["label"]),
            display_segments=display_segments,
        )

        vi_hit = rows[0]["vi_stable_hit"]
        annotate_vi_baseline(ax, vi_hit, display_segments=display_segments)

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
            display_segments=display_segments,
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
            dnf_x_offset=0.007,
            display_segments=display_segments,
        )
    _draw_legend_strip(fig)

    finalize(fig, list(axes))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches=None, pad_inches=0)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    metadata = {
        "figure": spec.get("figure", "control_gamma_comparison_hit_query_vs_lambda"),
        "threshold": threshold,
        "run_budget": run_budget,
        "stable_hit_definition": "Smallest recorded query after which policy_value_error remains below threshold.",
        "datasets": datasets,
        "palette_accessibility_report": palette_report,
        "output_png": str(output.resolve()),
        "output_pdf": str(pdf_output.resolve()),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    metadata = render(args.spec, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
