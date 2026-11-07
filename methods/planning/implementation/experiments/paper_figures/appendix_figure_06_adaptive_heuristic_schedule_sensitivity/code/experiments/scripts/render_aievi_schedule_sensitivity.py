#!/usr/bin/env python3
"""Render schedule sensitivity figures for Figure 2 deterministic control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common.plot_style import apply_conference_matplotlib_style  # noqa: E402

THRESHOLD = 1e-3
RUN_BUDGET = 1000
DNF_PLOT_Y = 1680
LEGACY_VARIANT_PREFIX = "water" "fall"
DEFAULT_ALGORITHM = f"{LEGACY_VARIANT_PREFIX}_d0p10_s3"
VI_COLOR = "#9D9D9D"
AIEVI_COLOR = "#CA5800"
OSVI_COLOR = "#1696D2"
TEXT_COLOR = "#242424"
GRID_COLOR = "#DADADA"
BEST_COLOR = "#5F5F5F"
EDGE_LOW_COLOR = "#777777"
EDGE_HIGH_COLOR = "#242424"
HEATMAP_CMAP = colors.LinearSegmentedColormap.from_list(
    "ie_query_map",
    ["#F7F7F7", "#D9E8F0", "#8FBED8", "#1696D2", "#0A3D5C"],
)
DECAY_FACTORS = [0.03, 0.05, 0.1, 0.2, 0.5]
STALL_PATIENCE_VALUES = [1, 2, 3, 5, 8]
DISPLAY_SEGMENTS: list[tuple[float, float, float]] = [
    (0.00, 0.60, 0.30),
    (0.60, 0.80, 0.45),
    (0.80, 1.00, 0.25),
]

VARIANT_RE = re.compile(rf"^{LEGACY_VARIANT_PREFIX}_d(?P<decay>\d+p\d+)_s(?P<stall>\d+)$")


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
    for q, y in zip(queries, values, strict=True):
        if y >= threshold:
            last_above = q
    if last_above is None:
        return queries[0] if queries else None
    after = [q for q in queries if q > last_above]
    return after[0] if after else None


def first_hit_query(queries: list[int], values: list[float], threshold: float) -> int | None:
    for q, y in zip(queries, values, strict=True):
        if y < threshold:
            return q
    return None


def variant_params(name: str) -> tuple[float, int] | None:
    match = VARIANT_RE.match(name)
    if match is None:
        return None
    decay = float(match.group("decay").replace("p", "."))
    stall = int(match.group("stall"))
    return decay, stall


def variant_name(decay: float, stall: int) -> str:
    token = f"{decay:.2f}".replace(".", "p")
    return f"{LEGACY_VARIANT_PREFIX}_d{token}_s{stall}"


def public_variant_name(decay: float, stall: int) -> str:
    token = f"{decay:.2f}".replace(".", "p")
    return f"aievi_beta{token}_S{stall}"


def public_algorithm_name(name: str) -> str:
    params = variant_params(name)
    if params is None:
        return name
    return public_variant_name(*params)


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    exported = dict(row)
    if "algorithm" in exported:
        exported["algorithm"] = public_algorithm_name(str(exported["algorithm"]))
    return exported


def load_trace_rows(trace_file: Path) -> list[dict[str, Any]]:
    rows = json.loads(trace_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Trace file did not contain a row list: {trace_file}")
    rows.sort(key=lambda row: float(row["lambda"]))
    return rows


def compute_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = sorted(
        (
            (name, params[0], params[1])
            for name in rows[0]["traces"]
            if (params := variant_params(name)) is not None
        ),
        key=lambda item: (item[1], item[2]),
    )
    by_lambda: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []

    for name, decay, stall in variants:
        hits: list[int | None] = []
        first_hits: list[int | None] = []
        final_errors: list[float] = []
        for row in rows:
            trace = row["traces"][name]
            queries = [int(value) for value in trace["queries"]]
            errors = [float(value) for value in trace["policy_value_errors"]]
            stable = stable_hit_query(queries, errors, THRESHOLD)
            first = first_hit_query(queries, errors, THRESHOLD)
            hits.append(stable)
            first_hits.append(first)
            final_errors.append(errors[-1] if errors else math.inf)
            by_lambda.append(
                {
                    "algorithm": name,
                    "decay_factor": decay,
                    "stall_patience": stall,
                    "lambda": float(row["lambda"]),
                    "stable_hit_query": RUN_BUDGET + 1 if stable is None else stable,
                    "stable_hit_dnf": stable is None,
                    "first_hit_query": RUN_BUDGET + 1 if first is None else first,
                    "first_hit_dnf": first is None,
                    "final_policy_value_error": final_errors[-1],
                }
            )

        penalized = [RUN_BUDGET + 1 if hit is None else int(hit) for hit in hits]
        dnf_count = sum(hit is None for hit in hits)
        aggregate.append(
            {
                "algorithm": name,
                "decay_factor": decay,
                "stall_patience": stall,
                "mean_stable_hit_query_penalized": sum(penalized) / len(penalized),
                "median_stable_hit_query_penalized": float(sorted(penalized)[len(penalized) // 2]),
                "mean_log10_one_plus_stable_hit_query_penalized": sum(
                    math.log10(1.0 + value) for value in penalized
                )
                / len(penalized),
                "max_stable_hit_query_penalized": max(penalized),
                "dnf_count": dnf_count,
                "final_error_max": max(final_errors),
            }
        )
    return aggregate, by_lambda


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _grid_from_aggregate(
    aggregate: list[dict[str, Any]],
    key: str,
) -> list[list[float]]:
    lookup = {
        (float(row["decay_factor"]), int(row["stall_patience"])): float(row[key])
        for row in aggregate
    }
    return [[lookup[(decay, stall)] for decay in DECAY_FACTORS] for stall in STALL_PATIENCE_VALUES]


def _algorithm_summary(
    aggregate: list[dict[str, Any]],
    decay: float,
    stall: int,
) -> dict[str, Any]:
    for row in aggregate:
        if math.isclose(float(row["decay_factor"]), decay) and int(row["stall_patience"]) == stall:
            return row
    raise KeyError((decay, stall))


def _select_profile_algorithms(aggregate: list[dict[str, Any]]) -> list[str]:
    best = min(
        aggregate,
        key=lambda row: (
            int(row["dnf_count"]),
            float(row["mean_stable_hit_query_penalized"]),
            float(row["max_stable_hit_query_penalized"]),
        ),
    )
    candidates = [
        DEFAULT_ALGORITHM,
        str(best["algorithm"]),
        variant_name(0.03, 1),
        variant_name(0.5, 8),
    ]
    selected: list[str] = []
    for name in candidates:
        if name not in selected:
            selected.append(name)
    return selected


def _profile_hits(
    rows: list[dict[str, Any]],
    algorithm: str,
) -> tuple[list[float], list[int | None]]:
    lambdas = [float(row["lambda"]) for row in rows]
    hits: list[int | None] = []
    for row in rows:
        trace = row["traces"][algorithm]
        queries = [int(value) for value in trace["queries"]]
        errors = [float(value) for value in trace["policy_value_errors"]]
        hits.append(stable_hit_query(queries, errors, THRESHOLD))
    return lambdas, hits


def _draw_heatmap(
    ax,
    values: list[list[float]],
    *,
    title: str,
    cmap: str | colors.Colormap,
    cbar_label: str,
    fmt: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="upper")
    ax.set_title(title, fontsize=8.4, pad=4, color=TEXT_COLOR)
    ax.set_xticks(range(len(DECAY_FACTORS)))
    ax.set_xticklabels([f"{decay:g}" for decay in DECAY_FACTORS], rotation=35, ha="right")
    ax.set_yticks(range(len(STALL_PATIENCE_VALUES)))
    ax.set_yticklabels([str(value) for value in STALL_PATIENCE_VALUES])
    ax.set_xlabel(r"decay $\beta$", fontsize=7.4, color=TEXT_COLOR)
    ax.set_ylabel(r"stall patience $S$", fontsize=7.4, color=TEXT_COLOR)
    ax.tick_params(axis="both", length=0, labelsize=7.0, colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(DECAY_FACTORS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(STALL_PATIENCE_VALUES))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.72, alpha=0.62)
    ax.tick_params(which="minor", bottom=False, left=False)

    norm = image.norm
    cmap_obj = image.cmap
    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            rgba = cmap_obj(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "black" if luminance > 0.58 else "white"
            ax.text(
                col_index,
                row_index,
                format(value, fmt),
                ha="center",
                va="center",
                color=text_color,
                fontsize=6.9,
            )

    default_col = DECAY_FACTORS.index(0.1)
    default_row = STALL_PATIENCE_VALUES.index(3)
    for edgecolor, linewidth in (("white", 2.6), (AIEVI_COLOR, 1.45)):
        ax.add_patch(
            Rectangle(
                (default_col - 0.5, default_row - 0.5),
                1,
                1,
                fill=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
        )
    cbar = plt.colorbar(image, ax=ax, fraction=0.044, pad=0.024)
    cbar.ax.tick_params(labelsize=6.8, length=2, colors=TEXT_COLOR)
    cbar.set_label(cbar_label, fontsize=7.1, color=TEXT_COLOR)
    cbar.outline.set_visible(False)


def _plot_profile_line(
    ax,
    rows: list[dict[str, Any]],
    algorithm: str,
    *,
    label: str,
    color: str,
    marker: str,
    linestyle: str | tuple[int, tuple[float, ...]],
    linewidth: float,
    zorder: int,
    dnf_y: float = DNF_PLOT_Y,
) -> None:
    lambdas, hits = _profile_hits(rows, algorithm)
    finite_lambdas = [lam for lam, hit in zip(lambdas, hits, strict=True) if hit is not None]
    finite_hits = [int(hit) for hit in hits if hit is not None]
    dnf_lambdas = [lam for lam, hit in zip(lambdas, hits, strict=True) if hit is None]
    if finite_hits:
        ax.plot(
            [lambda_to_display(lam) for lam in finite_lambdas],
            finite_hits,
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=3.7,
            markerfacecolor=color if marker != "s" else "white",
            markeredgecolor=color,
            markeredgewidth=0.75 if marker == "s" else 0.55,
            label=label,
            zorder=zorder,
        )
    if dnf_lambdas:
        ax.plot(
            [lambda_to_display(lam) for lam in dnf_lambdas],
            [dnf_y] * len(dnf_lambdas),
            marker="^",
            color=color,
            linestyle="None",
            markersize=4.4,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=0.75,
            label="_nolegend_",
            zorder=zorder,
        )


def _draw_profile_panel(
    ax,
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
) -> None:
    ax.set_xlim(-0.025, 1.025)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 2300)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels([r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$"])
    label_lambdas = [0.0, 0.4, 0.6, 0.7, 0.8, 1.0]
    ax.set_xticks([lambda_to_display(lam) for lam in label_lambdas])
    ax.set_xticklabels([f"{lam:g}" for lam in label_lambdas])
    minor_lambdas = sorted(
        set(round(float(row["lambda"]), 4) for row in rows) - set(label_lambdas)
    )
    ax.set_xticks([lambda_to_display(lam) for lam in minor_lambdas], minor=True)
    ax.tick_params(axis="x", which="minor", length=2.0, width=0.45, color="#9D9D9D")
    ax.grid(True, which="major", axis="y", color=GRID_COLOR, alpha=0.55, linewidth=0.38)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(0.62)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("stable-hit query\n" + r"$q_{\mathrm{SH}}(10^{-3})$")
    for boundary in (0.60, 0.80):
        ax.axvline(
            lambda_to_display(boundary),
            color="#9E9E9E",
            linestyle=(0, (1.4, 2.0)),
            linewidth=0.46,
            alpha=0.55,
            zorder=1,
        )
    ax.axhline(
        RUN_BUDGET,
        color="#9E9E9E",
        linestyle=(0, (1.0, 2.4)),
        linewidth=0.50,
        alpha=0.7,
        zorder=1,
    )

    vi_trace = rows[0]["traces"]["vi"]
    vi_hit = first_hit_query(
        [int(value) for value in vi_trace["queries"]],
        [float(value) for value in vi_trace["policy_value_errors"]],
        THRESHOLD,
    )
    if vi_hit is not None:
        ax.axhline(
            vi_hit,
            color=VI_COLOR,
            linestyle=(0, (1.0, 2.0)),
            linewidth=0.82,
            alpha=0.82,
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

    styles = {
        DEFAULT_ALGORITHM: (r"Adaptive IE-VI $(0.1,3)$", AIEVI_COLOR, "o", "-", 1.22, 7, 1290),
        variant_name(0.03, 1): (r"$\beta=0.03,S=1$", EDGE_LOW_COLOR, "^", (0, (2.4, 1.6)), 0.98, 5, 1500),
        variant_name(0.5, 8): (r"$\beta=0.5,S=8$", EDGE_HIGH_COLOR, "D", (0, (1.0, 1.6)), 0.98, 5, 1880),
    }
    best_name = _select_profile_algorithms(aggregate)[1] if len(_select_profile_algorithms(aggregate)) > 1 else DEFAULT_ALGORITHM
    if best_name not in styles:
        best_row = next(row for row in aggregate if row["algorithm"] == best_name)
        styles[best_name] = (
            "min mean",
            BEST_COLOR,
            "s",
            (0, (3.4, 1.8)),
            1.00,
            6,
            1390,
        )

    for algorithm in _select_profile_algorithms(aggregate):
        if algorithm in styles:
            label, color, marker, linestyle, linewidth, zorder, dnf_y = styles[algorithm]
            _plot_profile_line(
                ax,
                rows,
                algorithm,
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=zorder,
                dnf_y=dnf_y,
            )

    _plot_profile_line(
        ax,
        rows,
        "osvi",
        label="OS-VI",
        color=OSVI_COLOR,
        marker="s",
        linestyle=(0, (3.4, 1.8)),
        linewidth=1.00,
        zorder=4,
        dnf_y=DNF_PLOT_Y,
    )
    from matplotlib.legend_handler import HandlerTuple

    existing_handles, existing_labels = ax.get_legend_handles_labels()
    primary_legend = ax.legend(
        existing_handles,
        existing_labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015, 0.86, 0.10),
        mode="expand",
        ncol=len(existing_handles),
        fontsize=6.5,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.2,
        columnspacing=1.0,
        frameon=False,
    )
    ax.add_artist(primary_legend)

    aievi_dnf = Line2D(
        [0], [0], color=AIEVI_COLOR, marker="^", linestyle="None",
        markersize=4.6, markerfacecolor="none", markeredgecolor=AIEVI_COLOR,
        markeredgewidth=0.75,
    )
    osvi_dnf = Line2D(
        [0], [0], color=OSVI_COLOR, marker="^", linestyle="None",
        markersize=4.6, markerfacecolor="none", markeredgecolor=OSVI_COLOR,
        markeredgewidth=0.75,
    )
    ax.legend(
        [(aievi_dnf, osvi_dnf)],
        ["no hit"],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015, 0.0, 0.10),
        fontsize=6.9,
        handlelength=2.0,
        handletextpad=0.45,
        borderpad=0.2,
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.7)},
    )


def render_main_figure(
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    output: Path,
) -> None:
    apply_conference_matplotlib_style(plt, font_scale=0.88)
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams["figure.autolayout"] = False
    fig = plt.figure(figsize=(5.50, 4.58), dpi=240, layout=None)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.55],
        left=0.096,
        right=0.982,
        bottom=0.096,
        top=0.930,
        hspace=0.450,
        wspace=0.380,
    )
    ax_mean = fig.add_subplot(gs[0, 0])
    ax_worst = fig.add_subplot(gs[0, 1])
    ax_profile = fig.add_subplot(gs[1, :])

    mean_grid = _grid_from_aggregate(aggregate, "mean_stable_hit_query_penalized")
    max_grid = _grid_from_aggregate(aggregate, "max_stable_hit_query_penalized")
    _draw_heatmap(
        ax_mean,
        mean_grid,
        title=r"Mean $q_{\mathrm{SH}}$ ($\gamma=0.99$)",
        cmap=HEATMAP_CMAP,
        cbar_label="queries",
        fmt=".0f",
    )
    _draw_heatmap(
        ax_worst,
        max_grid,
        title=r"Max $q_{\mathrm{SH}}$ ($\gamma=0.99$)",
        cmap=HEATMAP_CMAP,
        cbar_label="queries",
        fmt=".0f",
    )
    _draw_profile_panel(ax_profile, rows, aggregate)

    for label, ax in (("(a)", ax_mean), ("(b)", ax_worst), ("(c)", ax_profile)):
        ax.text(
            -0.115 if ax is not ax_profile else -0.055,
            1.10 if ax is not ax_profile else 1.09,
            label,
            transform=ax.transAxes,
            fontsize=9.0,
            fontweight="bold",
            ha="left",
            va="top",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)




def render(trace_file: Path, output: Path, derived_dir: Path) -> dict[str, Any]:
    rows = load_trace_rows(trace_file)
    aggregate, by_lambda = compute_tables(rows)
    derived_dir.mkdir(parents=True, exist_ok=True)
    public_aggregate = [public_row(row) for row in aggregate]
    public_by_lambda = [public_row(row) for row in by_lambda]
    write_csv(derived_dir / "aievi_schedule_summary.csv", public_aggregate)
    write_csv(derived_dir / "aievi_schedule_by_lambda.csv", public_by_lambda)
    (derived_dir / "aievi_schedule_summary.json").write_text(
        json.dumps(public_aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    (derived_dir / "aievi_schedule_by_lambda.json").write_text(
        json.dumps(public_by_lambda, indent=2) + "\n",
        encoding="utf-8",
    )

    render_main_figure(rows, aggregate, output)

    best_by_mean = min(
        aggregate,
        key=lambda row: (
            int(row["dnf_count"]),
            float(row["mean_stable_hit_query_penalized"]),
            float(row["max_stable_hit_query_penalized"]),
        ),
    )
    best_by_log = min(
        aggregate,
        key=lambda row: (
            int(row["dnf_count"]),
            float(row["mean_log10_one_plus_stable_hit_query_penalized"]),
            float(row["mean_stable_hit_query_penalized"]),
        ),
    )
    default = next(row for row in aggregate if row["algorithm"] == DEFAULT_ALGORITHM)
    metadata = {
        "trace_file": "data/raw/control_traces.json",
        "threshold": THRESHOLD,
        "run_budget": RUN_BUDGET,
        "criterion": "stable_hit",
        "default_algorithm": public_algorithm_name(DEFAULT_ALGORITHM),
        "best_algorithm_by_dnf_then_mean": public_row(best_by_mean),
        "best_algorithm_by_dnf_then_mean_log": public_row(best_by_log),
        "default_summary": public_row(default),
        "lambda_count": len(rows),
        "variant_count": len(aggregate),
        "output_png": str(output),
        "output_pdf": str(output.with_suffix(".pdf")),
        "summary_csv": str(derived_dir / "aievi_schedule_summary.csv"),
        "by_lambda_csv": str(derived_dir / "aievi_schedule_by_lambda.csv"),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = render(args.trace_file.resolve(), args.output, args.derived_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
