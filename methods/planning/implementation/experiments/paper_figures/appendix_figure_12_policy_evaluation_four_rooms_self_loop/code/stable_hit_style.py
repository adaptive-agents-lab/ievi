"""Standard layout contract for stable-hit query figures."""

from __future__ import annotations

from typing import Any

from common.plot_style import apply_conference_matplotlib_style


VI_COLOR = "#9D9D9D"
AIEVI_COLOR = "#CA5800"
OSVI_COLOR = "#1696D2"
TEXT_COLOR = "#242424"
GRID_COLOR = "#DADADA"

FIG_SIZE = (5.50, 2.18)
LAYOUT = {
    "left": 0.104,
    "right": 0.985,
    "bottom": 0.238,
    "top": 0.755,
    "wspace": 0.145,
}
DISPLAY_SEGMENTS = (
    (0.00, 0.60, 0.30),
    (0.60, 0.80, 0.45),
    (0.80, 1.00, 0.25),
)
LABEL_LAMBDAS = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)


def lambda_to_display(
    lam: float,
    display_segments: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]] = DISPLAY_SEGMENTS,
) -> float:
    cumulative = 0.0
    for lam_start, lam_end, width in display_segments:
        if lam <= lam_end + 1e-9:
            frac = (lam - lam_start) / (lam_end - lam_start)
            return cumulative + width * max(0.0, min(1.0, frac))
        cumulative += width
    return cumulative


def display_segments_metadata(
    display_segments: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]] = DISPLAY_SEGMENTS,
) -> list[dict[str, float]]:
    return [
        {"lambda_start": start, "lambda_end": end, "width": width}
        for start, end, width in display_segments
    ]


def apply_style(plt: Any, *, font_scale: float = 0.95) -> None:
    apply_conference_matplotlib_style(plt, font_scale=font_scale)
    plt.rcParams.update(
        {
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
            "savefig.bbox": None,
        }
    )


def add_legend(fig: Any, *, aievi_label: str = "Adaptive IE-VI") -> Any:
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    aievi_line = Line2D(
        [0],
        [0],
        color=AIEVI_COLOR,
        marker="o",
        linestyle="-",
        linewidth=1.20,
        markersize=3.7,
        markerfacecolor=AIEVI_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.45,
        solid_capstyle="round",
    )
    osvi_line = Line2D(
        [0],
        [0],
        color=OSVI_COLOR,
        marker="s",
        linestyle=(0, (3.4, 1.8)),
        linewidth=1.20,
        markersize=3.7,
        markerfacecolor="none",
        markeredgecolor=OSVI_COLOR,
        markeredgewidth=0.75,
    )
    aievi_dnf = Line2D(
        [0],
        [0],
        color=AIEVI_COLOR,
        marker="^",
        linestyle="None",
        markersize=4.3,
        markerfacecolor=AIEVI_COLOR,
        markeredgecolor=AIEVI_COLOR,
        markeredgewidth=0.45,
    )
    osvi_dnf = Line2D(
        [0],
        [0],
        color=OSVI_COLOR,
        marker="^",
        linestyle="None",
        markersize=4.3,
        markerfacecolor="none",
        markeredgecolor=OSVI_COLOR,
        markeredgewidth=0.75,
    )
    primary_handles = [aievi_line, osvi_line]
    primary_labels = [aievi_label, "OS-VI"]
    primary_legend = fig.legend(
        primary_handles,
        primary_labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015, 0.86, 0.10),
        mode="expand",
        ncol=len(primary_handles),
        frameon=False,
        fontsize=6.5,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.2,
        columnspacing=1.0,
    )
    for text in primary_legend.get_texts():
        text.set_color(TEXT_COLOR)
    fig.add_artist(primary_legend)
    secondary_legend = fig.legend(
        [(aievi_dnf, osvi_dnf)],
        ["no hit"],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015, 0.0, 0.10),
        frameon=False,
        fontsize=6.9,
        handlelength=2.0,
        handletextpad=0.45,
        borderpad=0.2,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.7)},
    )
    for text in secondary_legend.get_texts():
        text.set_color(TEXT_COLOR)
    return primary_legend


def format_axis(
    ax: Any,
    lambdas: list[float],
    *,
    run_budget: int,
    title: str,
    display_segments: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]] = DISPLAY_SEGMENTS,
) -> None:
    total_width = sum(width for _, _, width in display_segments)
    minor_lambdas = sorted(set(round(float(lam), 4) for lam in lambdas) - set(LABEL_LAMBDAS))
    ax.set_xlim(-0.025 * total_width, total_width + 0.025 * total_width)
    ax.set_xticks([lambda_to_display(lam, display_segments) for lam in LABEL_LAMBDAS])
    ax.set_xticklabels([f"${lam:g}$" for lam in LABEL_LAMBDAS])
    ax.set_xticks([lambda_to_display(lam, display_segments) for lam in minor_lambdas], minor=True)
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
    for _, boundary, _ in display_segments[:-1]:
        ax.axvline(
            lambda_to_display(boundary, display_segments),
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
    ax.set_title(title, fontsize=8.4, pad=2.7, color=TEXT_COLOR)


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
    display_segments: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]] = DISPLAY_SEGMENTS,
    zorder: int = 5,
) -> list[float]:
    runs, dnf_lambdas = contiguous_runs(lambdas, values)
    for run in runs:
        ax.plot(
            [lambda_to_display(lam, display_segments) for lam, _ in run],
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
            [lambda_to_display(lam, display_segments) + dnf_x_offset for lam in dnf_lambdas],
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


def annotate_vi_baseline(
    ax: Any,
    vi_hit: int,
    *,
    display_segments: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]] = DISPLAY_SEGMENTS,
) -> None:
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


def finalize(
    fig: Any,
    axes: list[Any],
    *,
    ylabel: str = "stable-hit query\n$q_{\\mathrm{SH}}(10^{-3})$",
    xlabel: str = r"$\lambda$",
) -> None:
    axes[0].set_ylabel(ylabel, fontsize=7.6, labelpad=1.8)
    for ax in axes:
        ax.set_xlabel(xlabel, fontsize=7.6, labelpad=2.0)
        ax.tick_params(axis="both", which="major", labelsize=7.1, length=2.6, width=0.58)
    try:
        fig.align_ylabels(axes)
    except Exception:
        pass
    fig.subplots_adjust(**LAYOUT)
