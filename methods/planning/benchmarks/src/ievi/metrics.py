"""Stable-hit rules for trace comparisons.

``q_SH`` is last-crossing-then-stay-below with the closed band ``error <= tau``.
``q_FH`` is the first recorded query in that same closed band.

Query-charge note: Control/PE traces already store the charged true-model
query index. The two-state closed form charges ``m`` true-model refreshes
per surrogate outer step and one query per exact-IE / VI step.
"""

from __future__ import annotations

from collections.abc import Sequence


def first_hit_query(
    queries: Sequence[int],
    values: Sequence[float],
    threshold: float,
) -> int | None:
    """Smallest recorded query with ``value <= threshold`` (``q_FH``)."""

    if len(queries) != len(values):
        raise ValueError("queries and values must have the same length.")
    for query, value in zip(queries, values):
        if value <= threshold:
            return int(query)
    return None


def stable_hit_query(
    queries: Sequence[int],
    values: Sequence[float],
    threshold: float,
) -> int | None:
    """Last crossing, then stay at or below ``threshold`` (paper ``q_SH``).

    The last query with ``value > threshold`` is the last exit from the band.
    ``q_SH`` is the next recorded query, after which every later error satisfies
    ``<= threshold``. If the series never exceeds the band, return the first
    query. If it is still above at the last sample, return ``None``.
    """

    if len(queries) != len(values):
        raise ValueError("queries and values must have the same length.")
    if not queries:
        return None
    last_above = None
    for query, value in zip(queries, values):
        if value > threshold:
            last_above = int(query)
    if last_above is None:
        return int(queries[0])
    later = [int(query) for query in queries if int(query) > last_above]
    return later[0] if later else None
