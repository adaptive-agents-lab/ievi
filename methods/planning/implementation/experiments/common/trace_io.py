"""Trace loading and display-alignment helpers for control experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


TraceEntry = dict[str, Any]
Point = tuple[float, float]


def load_control_traces(path: Path) -> list[TraceEntry]:
    """Load a ``control_traces.json`` file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected a list of trace entries in {path}")
    return data


def lambda_label(lambda_value: float) -> str:
    """Return a stable compact label for lambda values."""

    if math.isclose(lambda_value, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "0"
    return f"{lambda_value:g}"


def merge_entries(trace_groups: Iterable[Iterable[TraceEntry]]) -> list[TraceEntry]:
    """Merge trace entries by lambda, keeping the first occurrence."""

    by_label: dict[str, TraceEntry] = {}
    for group in trace_groups:
        for entry in group:
            label = lambda_label(float(entry["lambda"]))
            by_label.setdefault(label, entry)
    return [by_label[key] for key in sorted(by_label, key=float)]


def find_lambda_entry(entries: Iterable[TraceEntry], lambda_value: float) -> TraceEntry:
    """Find one trace entry by lambda value."""

    for entry in entries:
        if math.isclose(float(entry["lambda"]), lambda_value, rel_tol=0.0, abs_tol=1e-12):
            return entry
    raise KeyError(f"lambda={lambda_value:g} not found")


def trace_points(entry: TraceEntry, algorithm: str, y_key: str = "policy_value_errors") -> list[Point]:
    """Return raw ``(query, y)`` points for one algorithm trace."""

    try:
        trace = entry["traces"][algorithm]
    except KeyError as exc:
        raise KeyError(f"algorithm {algorithm!r} not present for lambda={entry.get('lambda')}") from exc
    xs = [float(value) for value in trace["queries"]]
    ys = [max(float(value), 1e-14) for value in trace[y_key]]
    return list(zip(xs[: len(ys)], ys[: len(xs)], strict=True))


def vi_display_curve(entry: TraceEntry, *, x_max: float) -> list[Point]:
    """Display VI with raw query 1 shifted to iteration 0."""

    points = ((x - 1.0, y) for x, y in trace_points(entry, "vi"))
    return [(x, y) for x, y in points if x <= x_max]


def waterfall_display_curve(
    entry: TraceEntry,
    *,
    initial_error: float,
    x_max: float,
    algorithm: str = "waterfall",
    truncate_at_hit: bool = False,
) -> list[Point]:
    """Display adaptive-schedule post-update traces with a common initial point."""

    hit_query = float(entry.get("results", {}).get(algorithm, math.inf))
    limit = min(float(x_max), hit_query) if truncate_at_hit else float(x_max)
    points: list[Point] = [(0.0, float(initial_error))]
    points.extend((x, y) for x, y in trace_points(entry, algorithm) if x <= limit)
    return points


def first_y(points: list[Point]) -> float:
    if not points:
        raise ValueError("cannot read first y-value from an empty curve")
    return float(points[0][1])


def hit_query(entry: TraceEntry, algorithm: str) -> int | None:
    value = entry.get("results", {}).get(algorithm)
    return None if value is None else int(value)

