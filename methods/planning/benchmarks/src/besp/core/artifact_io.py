"""Canonical JSON artifact I/O for the maintained surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    as_mapping = getattr(value, "as_mapping", None)
    if callable(as_mapping):
        return to_jsonable(as_mapping())
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            scalar = value.item()
        except (TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return to_jsonable(scalar)
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    return value


def save_json_payload(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.write_text(json.dumps(to_jsonable(value), indent=2), encoding="utf-8")


def save_json_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    save_json_payload(path, list(rows))


def load_json_payload(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
