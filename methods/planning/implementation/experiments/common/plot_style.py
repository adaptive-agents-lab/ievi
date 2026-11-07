"""Shared plotting style for paper-side experiment figures."""

from __future__ import annotations

import math
from typing import Any


Y_FLOOR = 1e-4
TOP_CONFERENCE_PALETTE = (
    "#CA5800",
    "#1696D2",
    "#009E73",
    "#9D9D9D",
    "#242424",
    "#B58B00",
    "#6B4E9B",
    "#006D77",
)
LAMBDA_COLORS = {
    0.0: "#DADADA",
    0.1: "#C5DAE6",
    0.2: "#A8CDE0",
    0.3: "#7EB8D6",
    0.4: "#4EA4CE",
    0.5: "#1696D2",
    0.6: "#0F789E",
    0.7: "#0A5E7A",
    0.8: "#5F5F5F",
    1.0: "#242424",
}
Q_LEARNING_COLOR = "#9D9D9D"
OS_DYNA_COLOR = "#1696D2"
STANDARD_DYNA_COLOR = "#5F5F5F"
IE_DYNA_COLOR = "#CA5800"

ALGORITHM_LEGEND = {
    "Q-Learning": "Q-Learning",
    "Standard-Dyna": "Dyna",
    "OS-Dyna": "OS-Dyna",
    "NR1-Staircase-Waterfall-RM-ISP-Dyna": "IE-Dyna",
}
ALGORITHM_COLOR = {
    "Q-Learning": Q_LEARNING_COLOR,
    "Standard-Dyna": STANDARD_DYNA_COLOR,
    "OS-Dyna": OS_DYNA_COLOR,
    "NR1-Staircase-Waterfall-RM-ISP-Dyna": IE_DYNA_COLOR,
}


def lambda_color(lambda_value: float) -> str:
    for known_lambda, color in LAMBDA_COLORS.items():
        if math.isclose(lambda_value, known_lambda, rel_tol=0.0, abs_tol=1e-12):
            return color
    return "#5F5F5F"


def apply_matplotlib_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    )


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    raw = color.strip().lstrip("#")
    if len(raw) != 6:
        raise ValueError(f"expected #rrggbb color, got {color!r}")
    return tuple(int(raw[index : index + 2], 16) for index in (0, 2, 4))


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    channels = []
    for value in rgb:
        srgb = value / 255.0
        channels.append(srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def color_contrast_ratio(left: str, right: str = "#ffffff") -> float:
    left_luma = _relative_luminance(_hex_to_rgb(left))
    right_luma = _relative_luminance(_hex_to_rgb(right))
    lighter, darker = max(left_luma, right_luma), min(left_luma, right_luma)
    return (lighter + 0.05) / (darker + 0.05)


def _rgb_distance(left: str, right: str) -> float:
    left_rgb = _hex_to_rgb(left)
    right_rgb = _hex_to_rgb(right)
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left_rgb, right_rgb)))


def palette_accessibility_report(
    palette: tuple[str, ...] | list[str],
    *,
    min_contrast_vs_white: float = 2.4,
    min_rgb_distance: float = 40.0,
) -> dict[str, Any]:
    low_contrast = [
        {"color": color, "contrast_vs_white": color_contrast_ratio(color, "#ffffff")}
        for color in palette
        if color_contrast_ratio(color, "#ffffff") < min_contrast_vs_white
    ]
    near_duplicates = []
    for index, left in enumerate(palette):
        for right in palette[index + 1 :]:
            distance = _rgb_distance(left, right)
            if distance < min_rgb_distance:
                near_duplicates.append({"left": left, "right": right, "rgb_distance": distance})
    return {
        "palette": list(palette),
        "min_contrast_vs_white": min_contrast_vs_white,
        "min_rgb_distance": min_rgb_distance,
        "low_contrast_vs_white": low_contrast,
        "near_duplicate_pairs": near_duplicates,
    }


def apply_conference_matplotlib_style(
    plt: Any,
    *,
    font_scale: float = 1.0,
    palette: tuple[str, ...] | list[str] = TOP_CONFERENCE_PALETTE,
) -> None:
    """Apply conservative NeurIPS/ICML-style defaults for paper figures."""

    style: dict[str, Any] = {}
    try:
        from tueplots import bundles  # type: ignore

        style.update(bundles.neurips2023())
    except Exception:
        pass
    try:
        from cycler import cycler

        style["axes.prop_cycle"] = cycler(color=list(palette))
    except Exception:
        pass
    style.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.labelsize": 9.0 * font_scale,
            "axes.titlesize": 9.5 * font_scale,
            "legend.fontsize": 7.8 * font_scale,
            "xtick.labelsize": 8.0 * font_scale,
            "ytick.labelsize": 8.0 * font_scale,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.28,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    plt.rcParams.update(style)


def apply_stochastic_matplotlib_style(plt: Any) -> None:
    """Apply the larger typography used by stochastic-control panels."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "lines.solid_capstyle": "round",
        }
    )


def x_ticks(x_max: float) -> list[int]:
    if x_max <= 25:
        step = 5
    elif x_max <= 60:
        step = 10
    else:
        step = 20
    return list(range(0, int(x_max) + 1, step))


def format_control_axis(ax: Any, *, x_max: float, y_upper: float) -> None:
    ax.set_yscale("log")
    ax.set_xlim(-0.04 * x_max, x_max + 0.02 * x_max)
    ax.set_ylim(Y_FLOOR, y_upper)
    ax.set_xticks(x_ticks(x_max))
    ax.set_xlabel(r"Iterations $(k)$")
    ax.set_ylabel(r"Normalized $||V^{\pi_k}-V^*||_1$")
    ax.grid(True, which="major", color="#DADADA", alpha=0.55, linewidth=0.38)
    ax.tick_params(axis="both", which="major", labelsize=8.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.62)
        spine.set_color("#242424")


def format_stochastic_axis(ax: Any, *, x_max: int, y_min: float = -90, y_max: float = 16) -> None:
    """Format stochastic-control axes with physical-sample x ticks."""

    ax.set_xlim(-0.02 * x_max, x_max * 1.02)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Environment Samples $(t)$")
    ax.set_ylabel(r"$V^{\pi_t}(0)$")
    if x_max == 40_000:
        ticks = [0, 10_000, 20_000, 30_000, 40_000]
        labels = ["0k", "10k", "20k", "30k", "40k"]
    else:
        step = max(1, int(x_max / 4))
        ticks = list(range(0, x_max + 1, step))
        labels = [f"{int(value / 1000)}k" if value >= 1000 else str(value) for value in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.grid(True, color="#DADADA", alpha=0.55, linewidth=0.38)
    for spine in ax.spines.values():
        spine.set_linewidth(0.62)
        spine.set_color("#242424")


def derive_stochastic_y_bounds(
    curves: list[dict[str, Any]],
    *,
    pad_rel: float = 0.06,
    target_value: float | None = None,
) -> tuple[float, float]:
    """Pick y-axis bounds from selected-curve data, padded for the target line."""

    means: list[float] = []
    for record in curves:
        trace = record.get("trace", {})
        means.extend(float(value) for value in trace.get("mean", []))
        stderr = trace.get("stderr") or []
        means.extend(
            float(m) - float(s) for m, s in zip(trace.get("mean", []), stderr)
        )
        means.extend(
            float(m) + float(s) for m, s in zip(trace.get("mean", []), stderr)
        )
    if target_value is not None:
        means.append(float(target_value))
    if not means:
        return -90.0, 16.0
    lo, hi = min(means), max(means)
    pad = max(1.0, abs(hi - lo)) * float(pad_rel)
    return lo - pad, hi + pad
