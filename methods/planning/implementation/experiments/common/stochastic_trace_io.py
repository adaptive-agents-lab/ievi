"""Trace loading and HP-selection helpers for stochastic-control figures."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


Q_LEARNING_ALGORITHM = "Q-Learning"
STANDARD_DYNA_ALGORITHM = "Standard-Dyna"
OS_DYNA_ALGORITHM = "OS-Dyna"
WATERFALL_ALGORITHM = "NR1-Staircase-Waterfall-RM-ISP-Dyna"

LAMBDA_TOKEN_PATTERN = re.compile(r"_lambda_[^_]+_")


def hp_slot_id(label: str) -> str:
    """Canonical HP-slot id, stable across lambda values.

    The upstream label encodes the lambda token after `_lambda_`, so substituting
    the token with a placeholder yields the same id for the same HP across the
    lambda sweep.
    """
    return LAMBDA_TOKEN_PATTERN.sub("_lambda_X_", label)


def iqm(values: list[float]) -> float:
    """Interquartile mean: mean of values inside the trimmed middle 50%.

    Returns NaN for empty inputs. Matches scipy.stats.trim_mean(values, 0.25)
    (which is what rliable's IQM reduces to per task in the no-stratification case).
    """
    if not values:
        return float("nan")
    sorted_values = sorted(float(v) for v in values)
    n = len(sorted_values)
    cut = int(0.25 * n)
    trimmed = sorted_values[cut : n - cut] if n - cut > cut else sorted_values
    if not trimmed:
        return float("nan")
    return sum(trimmed) / len(trimmed)


def bootstrap_iqm_ci(
    values: list[float],
    *,
    n_samples: int = 2000,
    ci_low: float = 0.025,
    ci_high: float = 0.975,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for IQM.

    Resamples `values` with replacement `n_samples` times, computes IQM on each
    resample, returns the (ci_low, ci_high) quantiles. Returns (NaN, NaN) if
    the input has fewer than 2 values.
    """
    if len(values) < 2:
        return float("nan"), float("nan")
    import random

    rng = random.Random(rng_seed)
    n = len(values)
    floats = [float(v) for v in values]
    iqms: list[float] = []
    for _ in range(n_samples):
        resample = [floats[rng.randrange(n)] for _ in range(n)]
        iqms.append(iqm(resample))
    iqms.sort()
    lo_index = max(0, min(n_samples - 1, int(round(ci_low * (n_samples - 1)))))
    hi_index = max(0, min(n_samples - 1, int(round(ci_high * (n_samples - 1)))))
    return iqms[lo_index], iqms[hi_index]


def lambda_token(lambda_value: float) -> str:
    if math.isclose(lambda_value, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "0"
    return f"{lambda_value:g}".replace(".", "p")


def lambda_label(lambda_value: float) -> str:
    if math.isclose(lambda_value, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "0"
    return f"{lambda_value:g}"


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_path(exp_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else exp_dir / path


def iter_result_payloads(raw_root: Path, *, lambda_value: float) -> list[tuple[Path, dict[str, Any]]]:
    study_dir = raw_root / f"lambda_{lambda_token(lambda_value)}"
    if not study_dir.exists():
        raise FileNotFoundError(f"missing raw lambda directory: {study_dir}")
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for results_path in sorted(study_dir.glob("*/results.json")):
        payloads.append((results_path, load_json(results_path)))
    if not payloads:
        raise FileNotFoundError(f"no results.json files under {study_dir}")
    return payloads


def trace_from_payload(payload: dict[str, Any], *, algorithm: str, label: str) -> dict[str, Any]:
    trace = payload["results"][algorithm][label]
    x_axis = [int(value) for value in trace["x_axis"]]
    mean = [float(value) for value in trace["mean"]]
    std = [float(value) for value in trace["std"]]
    seeds = trace.get("seeds", [])
    if not (len(x_axis) == len(mean) == len(std)):
        raise ValueError(f"mismatched trace lengths for {label}")
    n_seeds = len(seeds[0]) if seeds and isinstance(seeds[0], list) else 0
    stderr = [value / math.sqrt(n_seeds) if n_seeds else 0.0 for value in std]
    return {
        "x_axis": x_axis,
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "n_seeds": n_seeds,
    }


def metric_row_for_label(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    for row in payload.get("metric_rows", []):
        if row.get("label") == label:
            return row
    raise KeyError(f"could not find metric row for {label}")


def assert_physical_sample_axis(row: dict[str, Any], trace: dict[str, Any], *, label: str) -> None:
    axis_kind = row.get("sample_axis_kind")
    if axis_kind != "physical_samples":
        raise ValueError(f"{label} is not on physical sample axis: {axis_kind!r}")
    sample_stride = int(row.get("sample_stride", 1) or 1)
    physical_budget = int(row.get("physical_samples_budget", max(trace["x_axis"])))
    if trace["x_axis"][-1] > physical_budget:
        raise ValueError(
            f"{label} exceeds physical budget: x_max={trace['x_axis'][-1]} budget={physical_budget}"
        )
    if sample_stride < 1:
        raise ValueError(f"{label} has invalid sample_stride={sample_stride}")


def target_band_radius(target_value: float, tolerance_rel: float) -> float:
    return float(tolerance_rel) * max(1.0, abs(float(target_value)))


def first_hit_of_mean_curve(
    trace: dict[str, Any],
    *,
    target_value: float,
    tolerance_rel: float,
    max_samples: int,
) -> int:
    band_radius = target_band_radius(target_value, tolerance_rel)
    for sample_count, value in zip(trace["x_axis"], trace["mean"], strict=True):
        if abs(float(value) - float(target_value)) <= band_radius:
            return int(sample_count)
    return int(max_samples) + 1


def final_target_gap(trace: dict[str, Any], *, target_value: float) -> float:
    return abs(float(trace["mean"][-1]) - float(target_value))


def candidate_phase_matches(metric_row: dict[str, Any], candidate_phase: str) -> bool:
    if candidate_phase == "all":
        return True
    phase = metric_row.get("phase") or metric_row.get("stage_name") or ""
    return str(phase) == candidate_phase


def candidate_record(
    *,
    results_path: Path,
    payload: dict[str, Any],
    algorithm: str,
    lambda_value: float,
    label: str,
) -> dict[str, Any]:
    trace = trace_from_payload(payload, algorithm=algorithm, label=label)
    metric_row = metric_row_for_label(payload, label=label)
    assert_physical_sample_axis(metric_row, trace, label=label)
    return {
        "algorithm": algorithm,
        "lambda": lambda_value,
        "label": label,
        "results_path": results_path,
        "trace": trace,
        "metric_row": metric_row,
    }


def all_candidate_records(
    *,
    raw_root: Path,
    algorithm: str,
    lambda_value: float,
    tolerance_rel: float,
    candidate_phase: str,
    physical_samples_budget_max: int | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for results_path, payload in iter_result_payloads(raw_root, lambda_value=lambda_value):
        labels = sorted(dict(payload.get("results", {})).get(algorithm, {}))
        for label in labels:
            record = candidate_record(
                results_path=results_path,
                payload=payload,
                algorithm=algorithm,
                lambda_value=lambda_value,
                label=label,
            )
            metric_row = record["metric_row"]
            if not candidate_phase_matches(metric_row, candidate_phase):
                continue
            candidate_budget = int(
                metric_row.get("physical_samples_budget", max(record["trace"]["x_axis"]))
            )
            if (
                physical_samples_budget_max is not None
                and candidate_budget > int(physical_samples_budget_max)
            ):
                continue
            max_samples = candidate_budget
            target_value = float(metric_row.get("target_value", payload.get("target_value")))
            mean_curve_hit = first_hit_of_mean_curve(
                record["trace"],
                target_value=target_value,
                tolerance_rel=tolerance_rel,
                max_samples=max_samples,
            )
            record["selection_info"] = {
                "mode": "mean-curve-first-hit",
                "tolerance_rel": float(tolerance_rel),
                "target_value": target_value,
                "target_band_radius": target_band_radius(target_value, tolerance_rel),
                "mean_curve_first_hit_samples": mean_curve_hit,
                "mean_curve_final_target_gap": final_target_gap(record["trace"], target_value=target_value),
                "candidate_phase": candidate_phase,
            }
            candidates.append(record)
    return candidates


def select_mean_curve_first_hit_record(
    *,
    raw_root: Path,
    algorithm: str,
    lambda_value: float,
    tolerance_rel: float,
    candidate_phase: str,
    physical_samples_budget_max: int | None = None,
) -> dict[str, Any]:
    candidates = all_candidate_records(
        raw_root=raw_root,
        algorithm=algorithm,
        lambda_value=lambda_value,
        tolerance_rel=tolerance_rel,
        candidate_phase=candidate_phase,
        physical_samples_budget_max=physical_samples_budget_max,
    )
    if not candidates:
        raise KeyError(f"no candidates for algorithm={algorithm!r}, lambda={lambda_value:g}")
    candidates.sort(
        key=lambda record: (
            int(record["selection_info"]["mean_curve_first_hit_samples"]),
            float(record["selection_info"]["mean_curve_final_target_gap"]),
            str(record["label"]),
        )
    )
    selected = candidates[0]
    selected["selection_info"]["candidate_count"] = len(candidates)
    return selected


def spec_algorithm_roots(spec: dict[str, Any], exp_dir: Path) -> dict[str, dict[str, Any]]:
    roots: dict[str, dict[str, Any]] = {}
    for key, item in spec["raw_result_roots"].items():
        roots[key] = {
            "algorithm": item["algorithm"],
            "path": resolve_path(exp_dir, item["path"]),
        }
    return roots


def selected_curve_records(spec: dict[str, Any], exp_dir: Path) -> list[dict[str, Any]]:
    roots = spec_algorithm_roots(spec, exp_dir)
    lambdas = [float(value) for value in spec["lambdas"]]
    tolerance_rel = float(spec["selection"]["tolerance_rel"])
    candidate_phase = str(spec["selection"].get("candidate_phase", "main"))
    budget_cap_raw = spec["selection"].get("physical_samples_budget_max")
    budget_cap = int(budget_cap_raw) if budget_cap_raw is not None else None

    records = [
        select_mean_curve_first_hit_record(
            raw_root=roots["q_learning"]["path"],
            algorithm=roots["q_learning"]["algorithm"],
            lambda_value=0.0,
            tolerance_rel=tolerance_rel,
            candidate_phase=candidate_phase,
            physical_samples_budget_max=budget_cap,
        )
    ]
    for lambda_value in lambdas:
        records.append(
            select_mean_curve_first_hit_record(
                raw_root=roots["standard_dyna"]["path"],
                algorithm=roots["standard_dyna"]["algorithm"],
                lambda_value=lambda_value,
                tolerance_rel=tolerance_rel,
                candidate_phase=candidate_phase,
                physical_samples_budget_max=budget_cap,
            )
        )
    if "os_dyna" in roots:
        for lambda_value in lambdas:
            records.append(
                select_mean_curve_first_hit_record(
                    raw_root=roots["os_dyna"]["path"],
                    algorithm=roots["os_dyna"]["algorithm"],
                    lambda_value=lambda_value,
                    tolerance_rel=tolerance_rel,
                    candidate_phase=candidate_phase,
                    physical_samples_budget_max=budget_cap,
                )
            )
    for lambda_value in lambdas:
        records.append(
            select_mean_curve_first_hit_record(
                raw_root=roots["waterfall_isp_dyna"]["path"],
                algorithm=roots["waterfall_isp_dyna"]["algorithm"],
                lambda_value=lambda_value,
                tolerance_rel=tolerance_rel,
                candidate_phase=candidate_phase,
                physical_samples_budget_max=budget_cap,
            )
        )
    return records


def repo_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)
