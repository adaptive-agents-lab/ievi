"""Maintained plotting helpers for stochastic-control baseline bundles."""
# pyright: reportMissingImports=false

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import (
    JAX_RL_DEFAULT_SUSTAINED_WINDOW,
    JAX_RL_FIRST_HIT_MEAN_METRIC_NAME,
    JAX_RL_SUSTAINED_HIT_MEAN_METRIC_NAME,
    jax_rl_hit_rate_by_budget_curve,
    jax_rl_seedwise_hit_metrics,
)


CANONICAL_VALUE_HIT_FIGURE_SET_ID = "stochastic_control_value_hits"
CANONICAL_FIRST_HIT_RATE_FIGURE_SET_ID = "stochastic_control_first_hit_rates_by_budget"
CANONICAL_SUSTAINED_HIT_RATE_FIGURE_SET_ID = "stochastic_control_sustained_hit_rates_by_budget"


@dataclass(frozen=True)
class BaselineValueTrace:
    """One algorithm trace for a fixed lambda panel."""

    algorithm_name: str
    x_axis: tuple[int, ...]
    mean: tuple[float, ...]


@dataclass(frozen=True)
class BaselineHitMarker:
    """One named hit marker on a value trace."""

    sample_count: float
    value: float


@dataclass(frozen=True)
class BaselineValuePanel:
    """One rendered value-curve panel for a fixed lambda."""

    baseline_id: str
    env_name: str
    gamma: float
    lambda_value: float
    lambda_token: str
    target_value: float
    target_band_radius: float
    traces: tuple[BaselineValueTrace, ...]
    markers: dict[str, dict[str, BaselineHitMarker | None]]


@dataclass(frozen=True)
class BaselineRateTrace:
    """One algorithm rate curve for a fixed lambda panel."""

    algorithm_name: str
    x_axis: tuple[int, ...]
    rates: tuple[float, ...]


@dataclass(frozen=True)
class BaselineHitRatePanel:
    """One rendered hit-rate panel for a fixed lambda and hit variant."""

    baseline_id: str
    env_name: str
    gamma: float
    lambda_value: float
    lambda_token: str
    hit_kind: str
    traces: tuple[BaselineRateTrace, ...]


_ALGORITHM_COLORS = {
    "Q-Learning": "#1f77b4",
    "Dyna": "#2ca02c",
    "OS-Dyna": "#d62728",
}

_ALGORITHM_MARKERS = {
    "Q-Learning": "o",
    "Dyna": "s",
    "OS-Dyna": "D",
}

_MARKER_STYLES = {
    "first_hit": {
        "fill": "hollow",
        "label": "First hit",
    },
    "sustained_hit": {
        "fill": "filled",
        "label": "Sustained hit",
    },
}

_TARGET_BAND_STYLE = {
    "fill_alpha": 0.18,
    "glow_alpha": 0.18,
    "glow_linewidth": 10.0,
    "glow_color": "#f6d365",
    "boundary_color": "#d4a017",
    "boundary_alpha": 0.9,
    "min_y_span_fraction": 0.01,
}

_HIT_RATE_STYLE = {
    "fill_alpha": 0.07,
    "line_width": 2.4,
    "line_alpha": 0.95,
    "final_marker_size": 44,
    "final_marker_edge_color": "#111111",
    "final_marker_edge_width": 0.8,
    "grid_alpha": 0.25,
    "x_padding_fraction": 0.02,
}

_HIT_RATE_SOURCE_BY_KIND = {
    "first_hit": "first_hit_seedwise_samples",
    "sustained_hit": "sustained_hit_seedwise_samples",
}


def _humanize_env_name(env_name: str) -> str:
    if env_name == "cliffwalk":
        return "Cliffwalk"
    if env_name == "maze":
        return "Maze"
    return env_name.replace("_", " ").title()


def _lambda_token(lambda_value: float) -> str:
    return format(lambda_value, "g").replace("-", "neg").replace(".", "p")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hit_kind_title(hit_kind: str) -> str:
    if hit_kind == "first_hit":
        return "First-Hit Rate by Budget"
    if hit_kind == "sustained_hit":
        return "Sustained-Hit Rate by Budget"
    raise ValueError(f"Unsupported hit kind '{hit_kind}'.")


def _figure_set_id_for_hit_kind(hit_kind: str) -> str:
    if hit_kind == "first_hit":
        return CANONICAL_FIRST_HIT_RATE_FIGURE_SET_ID
    if hit_kind == "sustained_hit":
        return CANONICAL_SUSTAINED_HIT_RATE_FIGURE_SET_ID
    raise ValueError(f"Unsupported hit kind '{hit_kind}'.")


def _figure_dir_name_for_set_id(figure_set_id: str) -> str:
    if figure_set_id == CANONICAL_VALUE_HIT_FIGURE_SET_ID:
        return "value_curves"
    if figure_set_id == CANONICAL_FIRST_HIT_RATE_FIGURE_SET_ID:
        return "first_hit_rates"
    if figure_set_id == CANONICAL_SUSTAINED_HIT_RATE_FIGURE_SET_ID:
        return "sustained_hit_rates"
    raise ValueError(f"Unsupported figure set id '{figure_set_id}'.")


def _extract_single_trace_payload(payload: dict[str, Any], *, family_name: str) -> dict[str, Any]:
    family = payload["results"][family_name]
    if len(family) != 1:
        raise ValueError(f"Expected exactly one trace for {family_name}, found {len(family)}")
    return next(iter(family.values()))


def _extract_single_trace(payload: dict[str, Any], *, family_name: str) -> BaselineValueTrace:
    trace_payload = _extract_single_trace_payload(payload, family_name=family_name)
    x_axis = tuple(int(value) for value in trace_payload["x_axis"])
    mean = tuple(float(value) for value in trace_payload["mean"])
    if len(x_axis) != len(mean):
        raise ValueError(f"Mismatched x-axis/mean lengths for {family_name}")
    return BaselineValueTrace(
        algorithm_name=family_name,
        x_axis=x_axis,
        mean=mean,
    )


def _extract_rate_trace(
    payload: dict[str, Any],
    *,
    family_name: str,
    hit_kind: str,
    sustained_window: int,
) -> BaselineRateTrace:
    trace_payload = _extract_single_trace_payload(payload, family_name=family_name)
    x_axis = tuple(int(value) for value in trace_payload["x_axis"])
    mean = tuple(float(value) for value in trace_payload["mean"])
    hit_metrics = jax_rl_seedwise_hit_metrics(
        x_axis=x_axis,
        mean_trace=mean,
        seed_payload=trace_payload["seeds"],
        target_value=float(payload["target_value"]),
        band_radius=float(payload["target_band_radius"]),
        sustained_window=sustained_window,
        max_samples=max(x_axis),
    )
    hit_samples = (
        hit_metrics.first_hit_seedwise_samples
        if hit_kind == "first_hit"
        else hit_metrics.sustained_hit_seedwise_samples
    )
    return BaselineRateTrace(
        algorithm_name=family_name,
        x_axis=x_axis,
        rates=jax_rl_hit_rate_by_budget_curve(
            x_axis=x_axis,
            hit_samples=hit_samples,
        ),
    )


def _lookup_marker(
    trace: BaselineValueTrace,
    *,
    sample_count: float,
) -> BaselineHitMarker | None:
    if sample_count > max(trace.x_axis):
        return None
    if sample_count < min(trace.x_axis):
        raise ValueError(
            f"Sample count {sample_count} precedes the first checkpoint for {trace.algorithm_name}"
        )
    if sample_count in trace.x_axis:
        trace_points = dict(zip(trace.x_axis, trace.mean, strict=True))
        return BaselineHitMarker(sample_count=sample_count, value=trace_points[int(sample_count)])

    insertion_idx = bisect_left(trace.x_axis, sample_count)
    if insertion_idx == 0 or insertion_idx >= len(trace.x_axis):
        raise ValueError(
            f"Could not interpolate sample count {sample_count} for {trace.algorithm_name}"
        )
    left_x = float(trace.x_axis[insertion_idx - 1])
    right_x = float(trace.x_axis[insertion_idx])
    left_y = float(trace.mean[insertion_idx - 1])
    right_y = float(trace.mean[insertion_idx])
    weight = (float(sample_count) - left_x) / (right_x - left_x)
    return BaselineHitMarker(
        sample_count=float(sample_count),
        value=left_y + weight * (right_y - left_y),
    )


def _row_metric(row: dict[str, Any], algorithm_prefix: str, metric_name: str, legacy_name: str) -> float:
    qualified_key = f"{algorithm_prefix}_{metric_name}"
    if qualified_key in row:
        return float(row[qualified_key])
    return float(row[legacy_name])


def _extract_panel_markers(
    row: dict[str, Any],
    *,
    q_learning_trace: BaselineValueTrace,
    dyna_trace: BaselineValueTrace,
    os_dyna_trace: BaselineValueTrace,
) -> dict[str, dict[str, BaselineHitMarker | None]]:
    return {
        "Q-Learning": {
            "first_hit": _lookup_marker(
                q_learning_trace,
                sample_count=_row_metric(
                    row,
                    "q_learning",
                    JAX_RL_FIRST_HIT_MEAN_METRIC_NAME,
                    "q_learning_first_hit_samples",
                ),
            ),
            "sustained_hit": _lookup_marker(
                q_learning_trace,
                sample_count=_row_metric(
                    row,
                    "q_learning",
                    JAX_RL_SUSTAINED_HIT_MEAN_METRIC_NAME,
                    "q_learning_sustained_hit_samples",
                ),
            ),
        },
        "Dyna": {
            "first_hit": _lookup_marker(
                dyna_trace,
                sample_count=_row_metric(
                    row,
                    "dyna",
                    JAX_RL_FIRST_HIT_MEAN_METRIC_NAME,
                    "dyna_first_hit_samples",
                ),
            ),
            "sustained_hit": _lookup_marker(
                dyna_trace,
                sample_count=_row_metric(
                    row,
                    "dyna",
                    JAX_RL_SUSTAINED_HIT_MEAN_METRIC_NAME,
                    "dyna_sustained_hit_samples",
                ),
            ),
        },
        "OS-Dyna": {
            "first_hit": _lookup_marker(
                os_dyna_trace,
                sample_count=_row_metric(
                    row,
                    "os_dyna",
                    JAX_RL_FIRST_HIT_MEAN_METRIC_NAME,
                    "os_dyna_first_hit_samples",
                ),
            ),
            "sustained_hit": _lookup_marker(
                os_dyna_trace,
                sample_count=_row_metric(
                    row,
                    "os_dyna",
                    JAX_RL_SUSTAINED_HIT_MEAN_METRIC_NAME,
                    "os_dyna_sustained_hit_samples",
                ),
            ),
        },
    }


def load_jax_rl_baseline_value_panels(baseline_dir: str | Path) -> list[BaselineValuePanel]:
    """Load one panel per lambda from a finalized JAX-RL baseline bundle."""

    baseline_root = Path(baseline_dir)
    manifest = _load_json(baseline_root / "baseline_manifest.json")
    raw_traces = _load_json(baseline_root / "raw" / "final_traces.json")
    score_rows = _load_json(baseline_root / "baseline_scores.json")["rows"]

    baseline_id = str(manifest["baseline_id"])
    env_name = str(manifest["env_name"])
    gamma = float(manifest["env_params"]["gamma"])

    q_learning_payload = raw_traces["q_learning"]
    q_learning_trace = _extract_single_trace(q_learning_payload, family_name="Q-Learning")
    q_learning_target = float(q_learning_payload["target_value"])
    q_learning_target_band = float(q_learning_payload["target_band_radius"])

    panels: list[BaselineValuePanel] = []
    for row in score_rows:
        lambda_value = float(row["lambda"])
        lambda_token = _lambda_token(lambda_value)

        dyna_payload = raw_traces["dyna"][lambda_token]
        os_dyna_payload = raw_traces["os_dyna"][lambda_token]
        dyna_trace = _extract_single_trace(dyna_payload, family_name="Dyna")
        os_dyna_trace = _extract_single_trace(os_dyna_payload, family_name="OS-Dyna")

        target_candidates = {
            q_learning_target,
            float(dyna_payload["target_value"]),
            float(os_dyna_payload["target_value"]),
        }
        if len(target_candidates) != 1:
            raise ValueError(
                f"Inconsistent target values for lambda={lambda_value}: {sorted(target_candidates)}"
            )
        band_candidates = {
            q_learning_target_band,
            float(dyna_payload["target_band_radius"]),
            float(os_dyna_payload["target_band_radius"]),
        }
        if len(band_candidates) != 1:
            raise ValueError(
                f"Inconsistent target band radii for lambda={lambda_value}: {sorted(band_candidates)}"
            )

        panels.append(
            BaselineValuePanel(
                baseline_id=baseline_id,
                env_name=env_name,
                gamma=gamma,
                lambda_value=lambda_value,
                lambda_token=lambda_token,
                target_value=q_learning_target,
                target_band_radius=q_learning_target_band,
                traces=(q_learning_trace, dyna_trace, os_dyna_trace),
                markers=_extract_panel_markers(
                    row,
                    q_learning_trace=q_learning_trace,
                    dyna_trace=dyna_trace,
                    os_dyna_trace=os_dyna_trace,
                ),
            )
        )
    return panels


def load_jax_rl_baseline_hit_rate_panels(
    baseline_dir: str | Path,
    *,
    hit_kind: str,
) -> list[BaselineHitRatePanel]:
    """Load one hit-rate panel per lambda from a finalized JAX-RL baseline bundle."""

    baseline_root = Path(baseline_dir)
    manifest = _load_json(baseline_root / "baseline_manifest.json")
    raw_traces = _load_json(baseline_root / "raw" / "final_traces.json")
    score_rows = _load_json(baseline_root / "baseline_scores.json")["rows"]

    baseline_id = str(manifest["baseline_id"])
    env_name = str(manifest["env_name"])
    gamma = float(manifest["env_params"]["gamma"])
    sustained_window = int(
        dict(manifest.get("metrics", {})).get("sustained_window", JAX_RL_DEFAULT_SUSTAINED_WINDOW)
    )

    q_learning_trace = _extract_rate_trace(
        raw_traces["q_learning"],
        family_name="Q-Learning",
        hit_kind=hit_kind,
        sustained_window=sustained_window,
    )

    panels: list[BaselineHitRatePanel] = []
    for row in score_rows:
        lambda_value = float(row["lambda"])
        lambda_token = _lambda_token(lambda_value)
        panels.append(
            BaselineHitRatePanel(
                baseline_id=baseline_id,
                env_name=env_name,
                gamma=gamma,
                lambda_value=lambda_value,
                lambda_token=lambda_token,
                hit_kind=hit_kind,
                traces=(
                    q_learning_trace,
                    _extract_rate_trace(
                        raw_traces["dyna"][lambda_token],
                        family_name="Dyna",
                        hit_kind=hit_kind,
                        sustained_window=sustained_window,
                    ),
                    _extract_rate_trace(
                        raw_traces["os_dyna"][lambda_token],
                        family_name="OS-Dyna",
                        hit_kind=hit_kind,
                        sustained_window=sustained_window,
                    ),
                ),
            )
        )
    return panels


def render_jax_rl_baseline_value_figures(
    panels: list[BaselineValuePanel],
    output_dir: str | Path,
    *,
    dpi: int = 200,
) -> list[Path]:
    """Render one line-plot value figure per lambda panel."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError("matplotlib is required to render JAX RL baseline figures") from exc

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    rendered_paths: list[Path] = []
    for panel in panels:
        figure_path = destination / f"value_curve_lambda_{panel.lambda_token}.png"
        fig, ax = plt.subplots(figsize=(9, 5.5))
        band_lower = panel.target_value - panel.target_band_radius
        band_upper = panel.target_value + panel.target_band_radius
        trace_values = [value for trace in panel.traces for value in trace.mean]
        y_min = min([band_lower, *trace_values])
        y_max = max([band_upper, *trace_values])
        visual_band_radius = max(
            panel.target_band_radius,
            (y_max - y_min) * _TARGET_BAND_STYLE["min_y_span_fraction"],
        )
        ax.axhspan(
            panel.target_value - visual_band_radius,
            panel.target_value + visual_band_radius,
            color=_TARGET_BAND_STYLE["glow_color"],
            alpha=0.10,
            zorder=0.6,
        )
        ax.axhline(
            panel.target_value,
            color=_TARGET_BAND_STYLE["glow_color"],
            linewidth=_TARGET_BAND_STYLE["glow_linewidth"],
            alpha=_TARGET_BAND_STYLE["glow_alpha"],
            zorder=0.8,
        )
        ax.axhspan(
            band_lower,
            band_upper,
            color=_TARGET_BAND_STYLE["glow_color"],
            alpha=_TARGET_BAND_STYLE["fill_alpha"],
            label="Hit band",
        )
        ax.axhline(
            band_lower,
            color=_TARGET_BAND_STYLE["boundary_color"],
            linestyle=":",
            linewidth=1.0,
            alpha=_TARGET_BAND_STYLE["boundary_alpha"],
        )
        ax.axhline(
            band_upper,
            color=_TARGET_BAND_STYLE["boundary_color"],
            linestyle=":",
            linewidth=1.0,
            alpha=_TARGET_BAND_STYLE["boundary_alpha"],
        )
        for trace in panel.traces:
            ax.plot(
                trace.x_axis,
                trace.mean,
                linewidth=2.2,
                color=_ALGORITHM_COLORS[trace.algorithm_name],
                label=trace.algorithm_name,
            )
        ax.axhline(
            panel.target_value,
            color="#111111",
            linestyle="--",
            linewidth=1.5,
            label=r"$V^*(s_0)$",
        )
        for trace in panel.traces:
            algorithm_markers = panel.markers[trace.algorithm_name]
            for marker_name, marker in algorithm_markers.items():
                if marker is None:
                    continue
                marker_style = _MARKER_STYLES[marker_name]
                algorithm_color = _ALGORITHM_COLORS[trace.algorithm_name]
                facecolor = algorithm_color if marker_style["fill"] == "filled" else "white"
                edgecolor = "#111111" if marker_style["fill"] == "filled" else algorithm_color
                ax.scatter(
                    [marker.sample_count],
                    [marker.value],
                    s=70,
                    marker=_ALGORITHM_MARKERS[trace.algorithm_name],
                    facecolors=facecolor,
                    edgecolors=edgecolor,
                    linewidths=2.0,
                    zorder=5,
                )
        ax.set_title(
            f"{_humanize_env_name(panel.env_name)} | gamma={panel.gamma:g} | lambda={panel.lambda_value:g}"
        )
        ax.set_xlabel("Environment Samples")
        ax.set_ylabel(r"$V^{\pi_k}(s_0)$")
        ax.grid(True, alpha=0.25, linewidth=0.8)
        line_legend = ax.legend(
            loc="lower left",
            bbox_to_anchor=(1.02, 0.02),
            borderaxespad=0.0,
        )
        marker_handles = [
            Line2D(
                [],
                [],
                linestyle="None",
                marker="o",
                markersize=8,
                markerfacecolor="white",
                markeredgecolor="#111111",
                markeredgewidth=2.0,
                label="First hit",
            ),
            Line2D(
                [],
                [],
                linestyle="None",
                marker="o",
                markersize=8,
                markerfacecolor="#111111",
                markeredgecolor="#111111",
                markeredgewidth=1.0,
                label="Sustained hit",
            ),
        ]
        marker_legend = ax.legend(
            handles=marker_handles,
            title="Hit markers",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        ax.add_artist(line_legend)
        ax.add_artist(marker_legend)
        fig.tight_layout()
        fig.savefig(
            figure_path,
            dpi=dpi,
            bbox_inches="tight",
            bbox_extra_artists=(line_legend, marker_legend),
        )
        plt.close(fig)
        rendered_paths.append(figure_path)
    return rendered_paths


def render_jax_rl_baseline_hit_rate_figures(
    panels: list[BaselineHitRatePanel],
    output_dir: str | Path,
    *,
    dpi: int = 200,
) -> list[Path]:
    """Render one hit-rate-by-budget figure per lambda panel."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError("matplotlib is required to render JAX RL baseline figures") from exc

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    rendered_paths: list[Path] = []
    for panel in panels:
        figure_path = destination / f"{panel.hit_kind}_rate_lambda_{panel.lambda_token}.png"
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for trace in panel.traces:
            color = _ALGORITHM_COLORS[trace.algorithm_name]
            ax.fill_between(
                trace.x_axis,
                trace.rates,
                step="post",
                color=color,
                alpha=_HIT_RATE_STYLE["fill_alpha"],
                zorder=1,
            )
            ax.step(
                trace.x_axis,
                trace.rates,
                where="post",
                linewidth=_HIT_RATE_STYLE["line_width"],
                color=color,
                alpha=_HIT_RATE_STYLE["line_alpha"],
                label=trace.algorithm_name,
                zorder=2,
            )
            ax.scatter(
                [trace.x_axis[-1]],
                [trace.rates[-1]],
                s=_HIT_RATE_STYLE["final_marker_size"],
                marker=_ALGORITHM_MARKERS[trace.algorithm_name],
                color=color,
                edgecolors=_HIT_RATE_STYLE["final_marker_edge_color"],
                linewidths=_HIT_RATE_STYLE["final_marker_edge_width"],
                zorder=3,
            )
        max_budget = max(max(trace.x_axis) for trace in panel.traces)
        ax.set_xlim(0, max_budget * (1.0 + _HIT_RATE_STYLE["x_padding_fraction"]))
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(
            f"{_humanize_env_name(panel.env_name)} | gamma={panel.gamma:g} | lambda={panel.lambda_value:g}\n"
            f"{_hit_kind_title(panel.hit_kind)}"
        )
        ax.set_xlabel("Environment Samples")
        ax.set_ylabel("Rate")
        ax.grid(True, alpha=_HIT_RATE_STYLE["grid_alpha"], linewidth=0.8)
        legend = ax.legend(
            loc="lower left",
            bbox_to_anchor=(1.02, 0.02),
            borderaxespad=0.0,
        )
        fig.tight_layout()
        fig.savefig(
            figure_path,
            dpi=dpi,
            bbox_inches="tight",
            bbox_extra_artists=(legend,),
        )
        plt.close(fig)
        rendered_paths.append(figure_path)
    return rendered_paths


def _marker_payload(marker: BaselineHitMarker | None) -> dict[str, float] | None:
    if marker is None:
        return None
    return {
        "x": marker.sample_count,
        "y": marker.value,
    }


def build_jax_rl_baseline_value_figure_manifest(
    baseline_dir: str | Path,
    *,
    output_dir: str | Path,
    panels: list[BaselineValuePanel],
    rendered_paths: list[Path],
) -> dict[str, Any]:
    """Build the canonical value-figure manifest for one finalized baseline bundle."""

    return {
        "baseline_dir": str(Path(baseline_dir)),
        "output_dir": str(Path(output_dir)),
        "target_band_visualization": {
            "mode": "data_band_plus_glow",
            "color": _TARGET_BAND_STYLE["glow_color"],
            "glow_linewidth": _TARGET_BAND_STYLE["glow_linewidth"],
            "glow_alpha": _TARGET_BAND_STYLE["glow_alpha"],
            "min_y_span_fraction": _TARGET_BAND_STYLE["min_y_span_fraction"],
        },
        "marker_style_legend": {
            "metric_source": {
                "first_hit": JAX_RL_FIRST_HIT_MEAN_METRIC_NAME,
                "sustained_hit": JAX_RL_SUSTAINED_HIT_MEAN_METRIC_NAME,
            },
            "algorithms": {
                algorithm_name: {
                    "marker": marker,
                    "color": _ALGORITHM_COLORS[algorithm_name],
                }
                for algorithm_name, marker in _ALGORITHM_MARKERS.items()
            },
            "hit_variants": {
                marker_name: {"fill": str(marker_style["fill"])}
                for marker_name, marker_style in _MARKER_STYLES.items()
            },
        },
        "figures": [
            {
                "lambda": panel.lambda_value,
                "lambda_token": panel.lambda_token,
                "path": str(path),
                "target_value": panel.target_value,
                "target_band_radius": panel.target_band_radius,
                "target_band_lower": panel.target_value - panel.target_band_radius,
                "target_band_upper": panel.target_value + panel.target_band_radius,
                "markers": {
                    algorithm_name: {
                        marker_name: _marker_payload(marker)
                        for marker_name, marker in marker_group.items()
                    }
                    for algorithm_name, marker_group in panel.markers.items()
                },
            }
            for panel, path in zip(panels, rendered_paths, strict=True)
        ],
    }


def build_jax_rl_baseline_hit_rate_figure_manifest(
    baseline_dir: str | Path,
    *,
    output_dir: str | Path,
    panels: list[BaselineHitRatePanel],
    rendered_paths: list[Path],
) -> dict[str, Any]:
    """Build the canonical hit-rate figure manifest for one finalized baseline bundle."""

    if not panels:
        raise ValueError("Cannot build a hit-rate figure manifest without panels.")
    hit_kind = panels[0].hit_kind
    return {
        "baseline_dir": str(Path(baseline_dir)),
        "output_dir": str(Path(output_dir)),
        "hit_kind": hit_kind,
        "curve_style": {
            "mode": "cumulative_rate_by_budget_step",
            "fill_alpha": _HIT_RATE_STYLE["fill_alpha"],
            "line_width": _HIT_RATE_STYLE["line_width"],
            "final_marker_size": _HIT_RATE_STYLE["final_marker_size"],
            "x_padding_fraction": _HIT_RATE_STYLE["x_padding_fraction"],
        },
        "rate_source": {
            "curve_quantity": "rate_by_budget",
            "hit_sample_source": _HIT_RATE_SOURCE_BY_KIND[hit_kind],
            "y_range": [0.0, 1.0],
            "budgets": "evaluation_checkpoints",
        },
        "algorithms": {
            algorithm_name: {
                "color": _ALGORITHM_COLORS[algorithm_name],
                "marker": _ALGORITHM_MARKERS[algorithm_name],
            }
            for algorithm_name in _ALGORITHM_COLORS
        },
        "figures": [
            {
                "lambda": panel.lambda_value,
                "lambda_token": panel.lambda_token,
                "path": str(path),
                "max_budget": max(max(trace.x_axis) for trace in panel.traces),
                "final_rates": {
                    trace.algorithm_name: trace.rates[-1] for trace in panel.traces
                },
            }
            for panel, path in zip(panels, rendered_paths, strict=True)
        ],
    }


def render_canonical_jax_rl_baseline_bundle_figures(
    baseline_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    dpi: int = 200,
) -> dict[str, Any]:
    """Render and write the canonical stochastic-control value-hit figure set for a bundle."""

    baseline_root = Path(baseline_dir)
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else baseline_root / "figures" / _figure_dir_name_for_set_id(CANONICAL_VALUE_HIT_FIGURE_SET_ID)
    )
    panels = load_jax_rl_baseline_value_panels(baseline_root)
    rendered_paths = render_jax_rl_baseline_value_figures(
        panels,
        resolved_output_dir,
        dpi=dpi,
    )
    manifest = build_jax_rl_baseline_value_figure_manifest(
        baseline_root,
        output_dir=resolved_output_dir,
        panels=panels,
        rendered_paths=rendered_paths,
    )
    (resolved_output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def render_canonical_jax_rl_baseline_hit_rate_figures(
    baseline_dir: str | Path,
    *,
    hit_kind: str,
    output_dir: str | Path | None = None,
    dpi: int = 200,
) -> dict[str, Any]:
    """Render and write one canonical stochastic-control hit-rate figure set for a bundle."""

    baseline_root = Path(baseline_dir)
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else baseline_root / "figures" / _figure_dir_name_for_set_id(_figure_set_id_for_hit_kind(hit_kind))
    )
    panels = load_jax_rl_baseline_hit_rate_panels(
        baseline_root,
        hit_kind=hit_kind,
    )
    rendered_paths = render_jax_rl_baseline_hit_rate_figures(
        panels,
        resolved_output_dir,
        dpi=dpi,
    )
    manifest = build_jax_rl_baseline_hit_rate_figure_manifest(
        baseline_root,
        output_dir=resolved_output_dir,
        panels=panels,
        rendered_paths=rendered_paths,
    )
    (resolved_output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def render_canonical_jax_rl_baseline_bundle_figure_sets(
    baseline_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    dpi: int = 200,
) -> dict[str, dict[str, Any]]:
    """Render and write every maintained stochastic-control figure set for a bundle."""

    baseline_root = Path(baseline_dir)
    resolved_output_root = Path(output_root) if output_root is not None else baseline_root / "figures"

    value_dir = resolved_output_root / _figure_dir_name_for_set_id(CANONICAL_VALUE_HIT_FIGURE_SET_ID)
    first_dir = resolved_output_root / _figure_dir_name_for_set_id(CANONICAL_FIRST_HIT_RATE_FIGURE_SET_ID)
    sustained_dir = resolved_output_root / _figure_dir_name_for_set_id(CANONICAL_SUSTAINED_HIT_RATE_FIGURE_SET_ID)

    value_manifest = render_canonical_jax_rl_baseline_bundle_figures(
        baseline_root,
        output_dir=value_dir,
        dpi=dpi,
    )
    first_manifest = render_canonical_jax_rl_baseline_hit_rate_figures(
        baseline_root,
        hit_kind="first_hit",
        output_dir=first_dir,
        dpi=dpi,
    )
    sustained_manifest = render_canonical_jax_rl_baseline_hit_rate_figures(
        baseline_root,
        hit_kind="sustained_hit",
        output_dir=sustained_dir,
        dpi=dpi,
    )
    return {
        CANONICAL_VALUE_HIT_FIGURE_SET_ID: {
            "figure_manifest_path": str(value_dir / "figure_manifest.json"),
            "output_dir": str(value_dir),
            "manifest": value_manifest,
        },
        CANONICAL_FIRST_HIT_RATE_FIGURE_SET_ID: {
            "figure_manifest_path": str(first_dir / "figure_manifest.json"),
            "output_dir": str(first_dir),
            "manifest": first_manifest,
        },
        CANONICAL_SUSTAINED_HIT_RATE_FIGURE_SET_ID: {
            "figure_manifest_path": str(sustained_dir / "figure_manifest.json"),
            "output_dir": str(sustained_dir),
            "manifest": sustained_manifest,
        },
    }


__all__ = [
    "CANONICAL_FIRST_HIT_RATE_FIGURE_SET_ID",
    "CANONICAL_SUSTAINED_HIT_RATE_FIGURE_SET_ID",
    "CANONICAL_VALUE_HIT_FIGURE_SET_ID",
    "BaselineHitMarker",
    "BaselineHitRatePanel",
    "BaselineRateTrace",
    "BaselineValuePanel",
    "BaselineValueTrace",
    "build_jax_rl_baseline_hit_rate_figure_manifest",
    "build_jax_rl_baseline_value_figure_manifest",
    "load_jax_rl_baseline_hit_rate_panels",
    "load_jax_rl_baseline_value_panels",
    "render_canonical_jax_rl_baseline_bundle_figures",
    "render_canonical_jax_rl_baseline_bundle_figure_sets",
    "render_canonical_jax_rl_baseline_hit_rate_figures",
    "render_jax_rl_baseline_hit_rate_figures",
    "render_jax_rl_baseline_value_figures",
]
