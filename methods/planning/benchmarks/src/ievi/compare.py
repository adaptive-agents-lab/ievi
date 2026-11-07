"""Compare assembled stable-hit query counts with the reference rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def compare_qsh_rows(
    actual: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str] = ("vi_stable_hit", "waterfall_stable_hit", "osvi_stable_hit"),
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    expected_by_lambda = {float(row["lambda"]): row for row in expected}
    for row in actual:
        lam = float(row["lambda"])
        other = expected_by_lambda.get(lam)
        if other is None:
            mismatches.append({"lambda": lam, "reason": "missing_expected"})
            continue
        diffs = {
            key: {"actual": row.get(key), "expected": other.get(key)}
            for key in keys
            if row.get(key) != other.get(key)
        }
        if diffs:
            mismatches.append({"lambda": lam, "diffs": diffs})
    return mismatches
