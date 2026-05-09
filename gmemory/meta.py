"""Meta-loop tag query API — read-only access for the recursion loop.

Tag schema (5 axes): domain, pattern, pitfall, swarm, meta.

The Phase 3 meta_loop reads these to ask questions like:
  "Among last 200 tasks, did MAR persona critics improve quality on
  logical-reasoning topics?"
  "Which swarm config had the best $/quality ratio on crypto?"

All queries: pure SQL, zero LLM cost, microsecond latency.
"""

from __future__ import annotations

import sqlite3
import statistics
import time
from typing import Iterable

TAG_AXES = ("domain", "pattern", "pitfall", "swarm", "meta")


def _validate_tag(tag: str) -> tuple[str, str]:
    if ":" not in tag:
        raise ValueError(f"tag must be axis:value, got {tag!r}")
    axis, value = tag.split(":", 1)
    if axis not in TAG_AXES:
        raise ValueError(f"unknown tag axis {axis!r}; valid: {TAG_AXES}")
    return axis, value


def search_tasks(
    db: sqlite3.Connection,
    *,
    must_have: Iterable[str] = (),
    none_of: Iterable[str] = (),
    status: str | None = None,
    since_days: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """Tag-and-status filter over Tier 2."""
    must = [_validate_tag(t) for t in must_have]
    none = [_validate_tag(t) for t in none_of]

    where_clauses: list[str] = []
    params: list = []
    for axis, value in must:
        where_clauses.append(
            "id IN (SELECT query_id FROM query_tags WHERE axis = ? AND value = ?)"
        )
        params.extend([axis, value])
    for axis, value in none:
        where_clauses.append(
            "id NOT IN (SELECT query_id FROM query_tags WHERE axis = ? AND value = ?)"
        )
        params.extend([axis, value])
    if status is not None:
        where_clauses.append("status = ?")
        params.append(status)
    if since_days is not None:
        where_clauses.append("created_at >= ?")
        params.append(time.time() - since_days * 86400)

    where = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    rows = db.execute(
        f"""
        SELECT id, query_text, status, quality_score, cost_usd, created_at
        FROM queries{where}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return [
        {
            "id": r[0], "query_text": r[1], "status": r[2],
            "quality_score": r[3], "cost_usd": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]


def search_insights(
    db: sqlite3.Connection,
    *,
    must_have: Iterable[str] = (),
    none_of: Iterable[str] = (),
    min_confidence: float = 0.0,
    limit: int = 50,
) -> list[dict]:
    must = [_validate_tag(t) for t in must_have]
    none = [_validate_tag(t) for t in none_of]
    where_clauses: list[str] = ["confidence >= ?"]
    params: list = [min_confidence]
    for axis, value in must:
        where_clauses.append(
            "id IN (SELECT insight_id FROM insight_tags WHERE axis = ? AND value = ?)"
        )
        params.extend([axis, value])
    for axis, value in none:
        where_clauses.append(
            "id NOT IN (SELECT insight_id FROM insight_tags WHERE axis = ? AND value = ?)"
        )
        params.extend([axis, value])
    where = " WHERE " + " AND ".join(where_clauses)
    rows = db.execute(
        f"""
        SELECT id, content, confidence, last_validated_at, supporting_queries
        FROM insights{where}
        ORDER BY confidence DESC, last_validated_at DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    import json
    out = []
    for r in rows:
        try:
            supporting = json.loads(r[4]) if r[4] else []
        except json.JSONDecodeError:
            supporting = []
        out.append({
            "id": r[0], "content": r[1],
            "confidence": float(r[2] or 0),
            "last_validated_at": float(r[3] or 0),
            "supporting_queries": supporting,
        })
    return out


def aggregate_by_tag(
    db: sqlite3.Connection,
    *,
    group_by: str,
    metric: str = "quality_score",
    must_have: Iterable[str] = (),
    since_days: int | None = None,
) -> dict[str, dict]:
    """Group tasks by tag-value within an axis; aggregate `metric`."""
    if group_by not in TAG_AXES:
        raise ValueError(f"unknown axis {group_by!r}")
    if metric not in ("quality_score", "cost_usd", "wall_clock_s"):
        raise ValueError(f"unknown metric {metric!r}")
    must = [_validate_tag(t) for t in must_have]

    where_clauses: list[str] = [f"q.{metric} IS NOT NULL"]
    params: list = []
    for axis, value in must:
        where_clauses.append(
            "q.id IN (SELECT query_id FROM query_tags WHERE axis = ? AND value = ?)"
        )
        params.extend([axis, value])
    if since_days is not None:
        where_clauses.append("q.created_at >= ?")
        params.append(time.time() - since_days * 86400)
    where = " WHERE " + " AND ".join(where_clauses)

    rows = db.execute(
        f"""
        SELECT qt.value, q.{metric}
        FROM query_tags qt
        JOIN queries q ON qt.query_id = q.id
        {where} AND qt.axis = ?
        """,
        (*params, group_by),
    ).fetchall()
    grouped: dict[str, list[float]] = {}
    for value, m in rows:
        grouped.setdefault(value, []).append(float(m))
    out: dict[str, dict] = {}
    for value, vals in grouped.items():
        out[value] = {
            "n": len(vals),
            "mean": float(statistics.mean(vals)) if vals else 0.0,
            "median": float(statistics.median(vals)) if vals else 0.0,
            "stddev": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
        }
    return out


def correlate_pattern_with_outcome(
    db: sqlite3.Connection,
    pattern_tag: str,
    *,
    outcome_metric: str = "quality_score",
    control_tags: Iterable[str] = (),
) -> dict:
    """Compute Δ = mean(metric | with pattern) − mean(metric | without)."""
    axis, value = _validate_tag(pattern_tag)
    if outcome_metric not in ("quality_score", "cost_usd", "wall_clock_s"):
        raise ValueError(f"unknown metric {outcome_metric!r}")

    controls = [_validate_tag(t) for t in control_tags]
    base_filters = [f"q.{outcome_metric} IS NOT NULL"]
    base_params: list = []
    for c_axis, c_value in controls:
        base_filters.append(
            "q.id IN (SELECT query_id FROM query_tags WHERE axis = ? AND value = ?)"
        )
        base_params.extend([c_axis, c_value])

    with_clause = (
        f"q.id IN (SELECT query_id FROM query_tags WHERE axis = ? AND value = ?)"
    )
    without_clause = with_clause.replace(" IN ", " NOT IN ")

    where_with = " AND ".join(base_filters + [with_clause])
    where_without = " AND ".join(base_filters + [without_clause])

    with_vals = [
        float(r[0]) for r in db.execute(
            f"SELECT q.{outcome_metric} FROM queries q WHERE {where_with}",
            (*base_params, axis, value),
        ).fetchall()
    ]
    without_vals = [
        float(r[0]) for r in db.execute(
            f"SELECT q.{outcome_metric} FROM queries q WHERE {where_without}",
            (*base_params, axis, value),
        ).fetchall()
    ]

    mean_w = statistics.mean(with_vals) if with_vals else 0.0
    mean_wo = statistics.mean(without_vals) if without_vals else 0.0

    return {
        "n_with": len(with_vals),
        "n_without": len(without_vals),
        "mean_with": mean_w,
        "mean_without": mean_wo,
        "delta": mean_w - mean_wo,
    }
