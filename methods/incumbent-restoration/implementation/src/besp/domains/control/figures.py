"""Reusable deterministic-control figure helpers built on maintained control artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ...core import load_json_payload

DEFAULT_CONTROL_DISPLAY_ITERATIONS = 100


@dataclass(frozen=True)
class ControlPaperPanel:
    label: str
    env_name: str
    env_params: Mapping[str, Any]
    lambda_values: tuple[float, ...]
    trace_rows: tuple[Mapping[str, Any], ...]
    run_dir: Path


def _humanize_env_name(env_name: str) -> str:
    if env_name == "cliffwalk":
        return "Cliffwalk"
    if env_name == "maze":
        return "Maze"
    if env_name in {"garnet", "garnet_frozen"}:
        return "Garnet"
    return env_name.replace("_", " ").title()


def _row_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _row_group_id(row: Mapping[str, Any]) -> str:
    metadata = _row_metadata(row)
    for key in ("variant_id", "instance_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return f"seed_{int(row['seed'])}"


def _curve_mean_and_stderr(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(samples, dtype=float)
    mean = arr.mean(axis=0)
    stderr = arr.std(axis=0) / max(np.sqrt(arr.shape[0]), 1.0)
    return mean, stderr


def _sanitize_curve(values: Sequence[float], *, display_iterations: int | None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if display_iterations is not None:
        arr = arr[: int(display_iterations)]
    arr = np.clip(arr, np.finfo(float).tiny, 1e4)
    return arr


def load_control_paper_panel(run_dir: str | Path, *, label: str | None = None) -> ControlPaperPanel:
    run_path = Path(run_dir)
    config_path = run_path / "config.yaml"
    trace_path = run_path / "raw" / "control_traces.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config: {config_path}")
    if not trace_path.exists():
        raise FileNotFoundError(f"missing control trace artifact: {trace_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    rows = load_json_payload(trace_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{trace_path} does not contain a non-empty control trace list")
    lambda_values = tuple(sorted({float(row["lambda"]) for row in rows}))

    return ControlPaperPanel(
        label=label or _humanize_env_name(str(config.get("env", "control"))),
        env_name=str(config["env"]),
        env_params=dict(config.get("env_params", {})),
        lambda_values=lambda_values,
        trace_rows=tuple(rows),
        run_dir=run_path,
    )


def render_control_paper_style_figure(
    panels: Sequence[ControlPaperPanel],
    output_path: str | Path,
    *,
    comparison_algorithm: str = "osvi",
    comparison_label: str = "OS-VI",
    show_titles: bool = False,
    display_iterations: int | None = DEFAULT_CONTROL_DISPLAY_ITERATIONS,
) -> Path:
    if not panels:
        raise ValueError("at least one panel is required")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError("matplotlib is required to render control figures") from exc

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, len(panels), figsize=(8 * len(panels), 7), squeeze=False)

    for panel_index, panel in enumerate(panels):
        ax = axs[0, panel_index]
        grouped: dict[float, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        for row in panel.trace_rows:
            lam = float(row["lambda"])
            for algorithm in ("vi", comparison_algorithm):
                trace = row["traces"][algorithm]
                grouped[lam][algorithm].append(
                    _sanitize_curve(trace["policy_value_errors"], display_iterations=display_iterations)
                )

        for algorithm, color, linestyle, label in (
            ("vi", "tab:green", "-", "VI"),
            (comparison_algorithm, "tab:blue", "-", comparison_label),
        ):
            for lam in panel.lambda_values:
                samples = grouped[lam][algorithm]
                stacked = np.vstack(samples)
                x_axis = np.arange(1, stacked.shape[1] + 1, dtype=int)
                if stacked.shape[0] == 1:
                    ax.plot(
                        x_axis,
                        stacked[0],
                        color=color,
                        linestyle=linestyle,
                        alpha=0.9,
                        label=f"{label} λ={lam:g}",
                    )
                else:
                    mean, stderr = _curve_mean_and_stderr(stacked)
                    ax.plot(
                        x_axis,
                        mean,
                        color=color,
                        linestyle=linestyle,
                        alpha=0.9,
                        label=f"{label} λ={lam:g}",
                    )
                    ax.fill_between(x_axis, mean - stderr, mean + stderr, color=color, alpha=0.15)

        if show_titles:
            ax.set_title(_humanize_env_name(panel.env_name))
        ax.set_xlabel("Outer Iteration")
        ax.set_ylabel("Normalized Policy Value Error")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


__all__ = [
    "ControlPaperPanel",
    "DEFAULT_CONTROL_DISPLAY_ITERATIONS",
    "load_control_paper_panel",
    "render_control_paper_style_figure",
]
