"""Reusable policy-evaluation figure helpers built on maintained PE artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
import yaml

from ...core import load_json_payload
from ...environments.jax_native import create_environment_jax_native
from ...model_error_runtime import apply_model_error_for_task
from . import build_policy_matrices, solve_value, value_iteration_control

DEFAULT_PAPER_DISPLAY_ITERATIONS = 100


@dataclass(frozen=True)
class PEPaperPanel:
    """One paper-style PE panel backed by a maintained run directory."""

    label: str
    env_name: str
    env_params: Mapping[str, Any]
    error_mode: str
    value_norm: str
    value_normalized: bool
    lambda_values: tuple[float, ...]
    trace_rows: tuple[Mapping[str, Any], ...]
    run_dir: Path


def _value_norm(vec: jnp.ndarray, *, norm_name: str) -> float:
    norm_key = norm_name.upper()
    if norm_key == "L1":
        return float(jnp.sum(jnp.abs(vec)))
    if norm_key == "L2":
        return float(jnp.linalg.norm(vec))
    if norm_key in {"LINF", "INF", "INFINITY"}:
        return float(jnp.max(jnp.abs(vec)))
    raise ValueError(f"Unsupported value norm {norm_name!r}")


def _compute_value_error(
    value: jnp.ndarray,
    target: jnp.ndarray,
    *,
    norm_name: str,
    normalized: bool,
    eps: float = 1e-12,
) -> float:
    diff = jnp.asarray(value, dtype=jnp.float64) - jnp.asarray(target, dtype=jnp.float64)
    numer = _value_norm(diff, norm_name=norm_name)
    if not normalized:
        return numer
    denom = _value_norm(jnp.asarray(target, dtype=jnp.float64), norm_name=norm_name) + eps
    return numer / denom


def _sanitize_log_series(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(arr)
    arr[finite_mask] = np.clip(arr[finite_mask], np.finfo(float).tiny, 1e2)
    return arr


def _curve_mean_and_stderr(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D samples array, got shape {arr.shape}")
    finite = np.isfinite(arr)
    counts = finite.sum(axis=0)
    safe = np.where(finite, arr, 0.0)
    mean = np.divide(
        safe.sum(axis=0),
        counts,
        out=np.full(arr.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )
    centered = np.where(finite, arr - mean[None, :], 0.0)
    variance = np.divide(
        (centered**2).sum(axis=0),
        counts,
        out=np.full(arr.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )
    stderr = np.divide(
        np.sqrt(variance),
        np.sqrt(counts),
        out=np.full(arr.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )
    return mean, stderr


def _scalar_mean_and_stderr(samples: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(samples, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return math.nan, math.nan
    mean = float(finite.mean())
    stderr = float(finite.std() / math.sqrt(finite.size))
    return mean, stderr


def _humanize_env_name(env_name: str) -> str:
    if env_name == "cliffwalk":
        return "Cliffwalk"
    if env_name == "maze":
        return "Maze"
    if env_name == "garnet":
        return "Garnet"
    if env_name == "garnet_frozen":
        return "Garnet"
    return env_name.replace("_", " ").title()


def load_pe_paper_panel(run_dir: str | Path, *, label: str | None = None) -> PEPaperPanel:
    """Load one maintained PE run directory into a reusable panel spec."""

    run_path = Path(run_dir)
    config_path = run_path / "config.yaml"
    trace_path = run_path / "raw" / "pe_traces.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config: {config_path}")
    if not trace_path.exists():
        raise FileNotFoundError(f"missing PE trace artifact: {trace_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    rows = load_json_payload(trace_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{trace_path} does not contain a non-empty PE trace list")

    lambda_values = tuple(sorted({float(row["lambda"]) for row in rows}))
    panel_label = label or str(config.get("name") or _humanize_env_name(str(config.get("env", "pe"))))
    env_params = dict(config.get("env_params", {}))

    return PEPaperPanel(
        label=panel_label,
        env_name=str(config["env"]),
        env_params=env_params,
        error_mode=str(config.get("error_mode", "support_uniform")),
        value_norm=str(config.get("value_norm", "L1")),
        value_normalized=bool(config.get("value_normalized", True)),
        lambda_values=lambda_values,
        trace_rows=tuple(rows),
        run_dir=run_path,
    )


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


def _rows_by_instance(panel: PEPaperPanel) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in panel.trace_rows:
        grouped[_row_group_id(row)].append(row)
    return dict(grouped)


def _vi_seed_traces(panel: PEPaperPanel) -> tuple[np.ndarray, np.ndarray]:
    per_seed_queries: list[np.ndarray] = []
    per_seed_errors: list[np.ndarray] = []
    for rows in _rows_by_instance(panel).values():
        trace = rows[0]["traces"]["vi"]
        per_seed_queries.append(np.asarray(trace["queries"], dtype=int))
        per_seed_errors.append(_sanitize_log_series(trace["errors"]))
    if not per_seed_queries:
        raise ValueError(f"panel {panel.label!r} has no VI traces")
    query_grid = per_seed_queries[0]
    for other in per_seed_queries[1:]:
        if not np.array_equal(query_grid, other):
            raise ValueError(f"VI query grids differ across seeds in panel {panel.label!r}")
    return query_grid, np.vstack(per_seed_errors)


def _comparison_lambda_traces(
    panel: PEPaperPanel,
    *,
    algorithm_name: str,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[float, list[np.ndarray]] = {lam: [] for lam in panel.lambda_values}
    query_grids: dict[float, np.ndarray] = {}
    for row in panel.trace_rows:
        lam = float(row["lambda"])
        trace = row["traces"][algorithm_name]
        queries = np.asarray(trace["queries"], dtype=int)
        errors = _sanitize_log_series(trace["errors"])
        existing = query_grids.get(lam)
        if existing is None:
            query_grids[lam] = queries
        elif not np.array_equal(existing, queries):
            raise ValueError(
                f"{algorithm_name} query grids differ for lambda={lam} in panel {panel.label!r}"
            )
        grouped[lam].append(errors)
    return {lam: (query_grids[lam], np.vstack(grouped[lam])) for lam in panel.lambda_values}


def _model_reference_samples(panel: PEPaperPanel) -> dict[float, np.ndarray]:
    grouped: dict[float, list[float]] = {lam: [] for lam in panel.lambda_values}
    for rows in _rows_by_instance(panel).values():
        representative = rows[0]
        metadata = _row_metadata(representative)
        env_name = str(metadata.get("env_name", panel.env_name))
        env_kwargs = dict(panel.env_params)
        metadata_env_params = metadata.get("env_params")
        if isinstance(metadata_env_params, Mapping):
            env_kwargs.update(dict(metadata_env_params))
        if "source_seed" in metadata and "seed" not in env_kwargs:
            env_kwargs["seed"] = int(metadata["source_seed"])
        elif "seed" not in env_kwargs:
            env_kwargs["seed"] = int(representative["seed"])

        mdp = create_environment_jax_native(env_name, **env_kwargs)
        target_manifest = metadata.get("deterministic_target_manifest")
        if isinstance(target_manifest, str) and target_manifest:
            from ...runners.deterministic_runtime import load_deterministic_target_bundle

            target_bundle = load_deterministic_target_bundle(target_manifest, mdp=mdp)
            pi_star = jnp.asarray(target_bundle.pi_star, dtype=jnp.int32)
            r_pi = jnp.asarray(target_bundle.r_pi, dtype=jnp.float64)
            v_true = jnp.asarray(target_bundle.v_pi_star, dtype=jnp.float64)
        else:
            _, pi_star = value_iteration_control(mdp)
            p_pi, r_pi = build_policy_matrices(mdp, pi_star, mdp.P)
            v_true = solve_value(p_pi, r_pi, mdp.gamma)
        for lam in panel.lambda_values:
            p_hat = apply_model_error_for_task(
                mdp.P,
                lam,
                panel.error_mode,
                seed=int(metadata.get("source_seed", representative["seed"])),
                env_name=env_name,
                env_params=env_kwargs,
            )
            p_hat_pi, _ = build_policy_matrices(mdp, pi_star, p_hat)
            model_value = solve_value(p_hat_pi, r_pi, mdp.gamma)
            grouped[lam].append(
                _compute_value_error(
                    model_value,
                    v_true,
                    norm_name=panel.value_norm,
                    normalized=panel.value_normalized,
                )
            )
    return {lam: np.asarray(samples, dtype=float) for lam, samples in grouped.items()}


def compute_model_reference_errors(panel: PEPaperPanel) -> dict[float, float]:
    """Return mean model-reference error per lambda for one paper panel."""

    return {lam: float(np.mean(samples)) for lam, samples in _model_reference_samples(panel).items()}


def _configure_paper_style(matplotlib: Any) -> None:
    params = {
        "font.size": 26,
        "axes.labelsize": 26,
        "axes.titlesize": 22,
        "legend.fontsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "lines.linewidth": 3,
        "axes.linewidth": 2,
    }
    matplotlib.rcParams.update(params)


def _panel_title(panel: PEPaperPanel, *, show_titles: bool) -> str | None:
    if not show_titles:
        return None
    return _humanize_env_name(panel.env_name)


def _truncate_curve(
    queries: np.ndarray,
    samples: np.ndarray,
    *,
    display_iterations: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if display_iterations is None:
        return queries, samples
    if display_iterations <= 0:
        raise ValueError("display_iterations must be positive when provided")
    limit = min(int(display_iterations), int(queries.shape[0]))
    return queries[:limit], samples[:, :limit]


def render_pe_paper_style_figure(
    panels: Sequence[PEPaperPanel],
    output_path: str | Path,
    *,
    comparison_algorithm: str = "osvi",
    comparison_label: str = "OS-VI",
    legend_mode: str = "auto",
    show_titles: bool = False,
    display_iterations: int | None = DEFAULT_PAPER_DISPLAY_ITERATIONS,
) -> Path:
    """Render one or more paper-style PE panels from maintained traces."""

    if not panels:
        raise ValueError("at least one panel is required")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ModuleNotFoundError(
            "render_osvi_paper_style_figure requires the optional dependency 'matplotlib'."
        ) from exc

    _configure_paper_style(plt)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel_count = len(panels)
    fig, axs = plt.subplots(1, panel_count, sharey=True)
    axes = [axs] if panel_count == 1 else list(axs)
    fig.set_figwidth(10 * panel_count if panel_count > 1 else 11)
    fig.set_figheight(8)
    fig.set_dpi(300)

    cmap = plt.get_cmap("tab10")
    can_share_legend = len({panel.lambda_values for panel in panels}) == 1
    resolved_legend_mode = "shared" if legend_mode == "auto" and can_share_legend else legend_mode
    if resolved_legend_mode == "auto":
        resolved_legend_mode = "panel"

    for index, (ax, panel) in enumerate(zip(axes, panels, strict=True)):
        vi_queries, vi_errors = _vi_seed_traces(panel)
        vi_queries, vi_errors = _truncate_curve(
            vi_queries,
            vi_errors,
            display_iterations=display_iterations,
        )
        vi_mean, vi_stderr = _curve_mean_and_stderr(vi_errors)
        ax.plot(vi_queries, vi_mean, label="VI", linestyle="dotted", color=cmap(0))
        if vi_errors.shape[0] > 1:
            ax.fill_between(vi_queries, np.maximum(vi_mean - vi_stderr, np.finfo(float).tiny), vi_mean + vi_stderr, alpha=0.2, color=cmap(0))

        comparison_traces = _comparison_lambda_traces(
            panel,
            algorithm_name=comparison_algorithm,
        )
        model_samples = _model_reference_samples(panel)
        for color_index, lam in enumerate(panel.lambda_values, start=1):
            queries, errors = comparison_traces[lam]
            queries, errors = _truncate_curve(
                queries,
                errors,
                display_iterations=display_iterations,
            )
            mean, stderr = _curve_mean_and_stderr(errors)
            color = cmap(color_index)
            ax.plot(queries, mean, label=rf"{comparison_label} ($\lambda$ = {lam:g})", color=color)
            if errors.shape[0] > 1:
                ax.fill_between(
                    queries,
                    np.maximum(mean - stderr, np.finfo(float).tiny),
                    mean + stderr,
                    alpha=0.2,
                    color=color,
                )

            model_mean, model_stderr = _scalar_mean_and_stderr(model_samples[lam])
            ax.axhline(
                y=max(model_mean, np.finfo(float).tiny),
                label=rf"Model ($\lambda$ = {lam:g})",
                linestyle="dashed",
                color=color,
            )
            if model_samples[lam].shape[0] > 1:
                ax.fill_between(
                    queries,
                    np.maximum(model_mean - model_stderr, np.finfo(float).tiny),
                    model_mean + model_stderr,
                    alpha=0.2,
                    color=color,
                )

        ax.set_xlabel("Iteration (k)")
        if index == 0:
            ax.set_ylabel(r"Normalized $||V_k - V^\pi||_1$")
        title = _panel_title(panel, show_titles=show_titles)
        if title:
            ax.set_title(title)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-4, top=1e1)
        ax.set_xlim(left=0, right=int(vi_queries[-1]))
        ax.grid()
        if resolved_legend_mode == "panel":
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=14)

    if resolved_legend_mode == "shared":
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=16)

    plt.subplots_adjust(left=0.1, bottom=0.22 if resolved_legend_mode == "panel" else 0.12, right=0.97, top=0.92, wspace=0.12, hspace=0.1)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_osvi_paper_style_figure(
    panels: Sequence[PEPaperPanel],
    output_path: str | Path,
    *,
    legend_mode: str = "auto",
    show_titles: bool = False,
    display_iterations: int | None = DEFAULT_PAPER_DISPLAY_ITERATIONS,
) -> Path:
    """Backward-compatible OSVI wrapper over the generic paper-style renderer."""

    return render_pe_paper_style_figure(
        panels,
        output_path,
        comparison_algorithm="osvi",
        comparison_label="OS-VI",
        legend_mode=legend_mode,
        show_titles=show_titles,
        display_iterations=display_iterations,
    )


__all__ = [
    "DEFAULT_PAPER_DISPLAY_ITERATIONS",
    "PEPaperPanel",
    "compute_model_reference_errors",
    "load_pe_paper_panel",
    "render_pe_paper_style_figure",
    "render_osvi_paper_style_figure",
]
