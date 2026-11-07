"""Load and assemble Control traces for q_SH comparison."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .metrics import first_hit_query, stable_hit_query

PIECEWISE_19_LAMBDAS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.625,
    0.65,
    0.675,
    0.7,
    0.725,
    0.75,
    0.775,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
)
ALG_TRACE_KEYS = (("vi", "vi"), ("waterfall", "waterfall"), ("osvi", "osvi"))
AIEVI_TRACE_NAMES = ("waterfall", "waterfall_besp", "aisp")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_control_traces(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, str):
        raise ValueError(f"trace file {path} is a path stub, not a JSON list")
    if not isinstance(data, list):
        raise ValueError(f"expected a list of trace rows in {path}")
    data.sort(key=lambda entry: float(entry["lambda"]))
    return data


def resolve_aievi_trace_name(traces: Mapping[str, Any]) -> str:
    for name in AIEVI_TRACE_NAMES:
        if name in traces:
            return name
    raise KeyError(f"no AIEVI trace among {AIEVI_TRACE_NAMES}; have {sorted(traces)}")


def _trace_errors(trace: Mapping[str, Any]) -> tuple[list[int], list[float]]:
    queries = [int(query) for query in trace["queries"]]
    if "policy_value_errors" in trace:
        values = [float(value) for value in trace["policy_value_errors"]]
    else:
        values = [float(value) for value in trace["errors"]]
    return queries, values


def hit_row(
    entry: Mapping[str, Any],
    *,
    threshold: float = 1e-3,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "lambda": float(entry["lambda"]),
        "seed": entry.get("seed"),
    }
    traces = entry["traces"]
    for out_key, trace_name in ALG_TRACE_KEYS:
        trace = traces[trace_name]
        queries, values = _trace_errors(trace)
        row[f"{out_key}_first_hit"] = first_hit_query(queries, values, threshold)
        row[f"{out_key}_stable_hit"] = stable_hit_query(queries, values, threshold)
        row[f"{out_key}_last_query"] = queries[-1] if queries else None
        row[f"{out_key}_last_error"] = values[-1] if values else None
    return row


def hit_table(
    entries: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 1e-3,
    lambdas: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    wanted = None if lambdas is None else [float(value) for value in lambdas]
    rows = [hit_row(entry, threshold=threshold) for entry in entries]
    if wanted is None:
        return rows
    by_lambda = {row["lambda"]: row for row in rows}
    selected: list[dict[str, Any]] = []
    for lam in wanted:
        match = next((row for key, row in by_lambda.items() if abs(key - lam) < 1e-12), None)
        if match is None:
            raise KeyError(f"lambda {lam} missing from traces")
        selected.append(match)
    return selected


def planning_hit_row(
    entry: Mapping[str, Any],
    *,
    threshold: float = 1e-3,
) -> dict[str, Any]:
    """Score VI / OS-VI / AIEVI traces. AIEVI may be waterfall, waterfall_besp, or aisp."""

    traces = entry["traces"]
    aievi_name = resolve_aievi_trace_name(traces)
    row: dict[str, Any] = {
        "lambda": float(entry["lambda"]),
        "seed": entry.get("seed"),
        "aievi_trace_name": aievi_name,
    }
    for out_key, trace_name in (("vi", "vi"), ("osvi", "osvi"), ("aievi", aievi_name)):
        queries, values = _trace_errors(traces[trace_name])
        row[f"{out_key}_first_hit"] = first_hit_query(queries, values, threshold)
        row[f"{out_key}_stable_hit"] = stable_hit_query(queries, values, threshold)
        row[f"{out_key}_last_query"] = queries[-1] if queries else None
        row[f"{out_key}_last_error"] = values[-1] if values else None
    return row


def planning_hit_table(
    entries: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 1e-3,
    lambdas: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    wanted = None if lambdas is None else [float(value) for value in lambdas]
    rows = [planning_hit_row(entry, threshold=threshold) for entry in entries]
    if wanted is None:
        return rows
    return select_lambdas(rows, wanted)


def assemble_by_lambda(tables: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[float, dict[str, Any]] = {}
    for table in tables:
        for row in table:
            merged[float(row["lambda"])] = dict(row)
    return [merged[key] for key in sorted(merged)]


def select_lambdas(
    rows: Sequence[Mapping[str, Any]],
    lambdas: Iterable[float],
) -> list[dict[str, Any]]:
    by_lambda = {float(row["lambda"]): dict(row) for row in rows}
    selected: list[dict[str, Any]] = []
    for lam in lambdas:
        match = next((row for key, row in by_lambda.items() if abs(key - float(lam)) < 1e-12), None)
        if match is None:
            raise KeyError(f"lambda {lam} missing from rows")
        selected.append(match)
    return selected
