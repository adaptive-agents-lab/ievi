"""Residual-stall diagnostics for adaptive IE-VI traces."""

from __future__ import annotations

import math
from typing import Any


TraceEntry = dict[str, Any]


def infer_waterfall_h_events(
    entry: TraceEntry,
    *,
    x_max: float,
    h_init: float = 1e6,
    h_min: float = 1.0,
    stall_patience: int = 3,
    decay_factor: float = 0.1,
) -> dict[str, list[dict[str, float]]]:
    """Infer residual stalls and h decays from persisted residual traces.

    The raw traces persist Bellman residuals but not h. This mirrors the
    experiment heuristic: h decays after ``stall_patience`` consecutive
    non-improvements in the recorded residual.
    """

    trace = entry["traces"]["waterfall"]
    queries = [float(value) for value in trace["queries"]]
    residuals = [float(value) for value in trace["residuals"]]
    policy_errors = [float(value) for value in trace["policy_value_errors"]]

    best_residual = math.inf
    stall_count = 0
    h_value = float(h_init)
    stalls: list[dict[str, float]] = []
    decays: list[dict[str, float]] = []

    for query, residual, policy_error in zip(queries, residuals, policy_errors, strict=True):
        if query > x_max:
            break
        if residual < best_residual:
            best_residual = residual
            stall_count = 0
            continue

        stall_count += 1
        stalls.append(
            {
                "query": query,
                "residual": residual,
                "previous_best_residual": best_residual,
                "policy_value_error": policy_error,
                "stall_count": float(stall_count),
                "h_before": h_value,
            }
        )
        if stall_count >= stall_patience:
            next_h = max(float(h_min), h_value * float(decay_factor))
            decays.append(
                {
                    "query": query,
                    "residual": residual,
                    "previous_best_residual": best_residual,
                    "policy_value_error": policy_error,
                    "h_before": h_value,
                    "h_after": next_h,
                }
            )
            h_value = next_h
            stall_count = 0

    return {"stalls": stalls, "decays": decays}


def residual_rows(entry: TraceEntry, lambda_value: float) -> list[dict[str, float | int | str]]:
    """Return one CSV-friendly row per persisted adaptive IE-VI checkpoint."""

    trace = entry["traces"]["waterfall"]
    value_errors = trace.get("value_errors", [])
    rows: list[dict[str, float | int | str]] = []
    for idx, (query, residual, policy_error) in enumerate(
        zip(trace["queries"], trace["residuals"], trace["policy_value_errors"], strict=True)
    ):
        row: dict[str, float | int | str] = {
            "lambda": float(lambda_value),
            "query": int(query),
            "residual": float(residual),
            "policy_value_error": float(policy_error),
        }
        if idx < len(value_errors):
            row["value_error"] = float(value_errors[idx])
        rows.append(row)
    return rows
