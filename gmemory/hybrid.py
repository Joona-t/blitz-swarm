"""Reciprocal Rank Fusion of vector and BM25 retrieval.

Weighted RRF as in Cormack et al.; the recommended k=60 default
flattens above 40 so small-K perturbations don't change ranking.
Vector dominates on paraphrase; BM25 catches rare-token jargon.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Iterable

RRF_K = 60
VEC_WEIGHT = 0.6
BM25_WEIGHT = 0.4


def rrf_fuse(
    vector_hits: list[tuple[str, float]],
    bm25_hits: list[tuple[str, float]],
    *,
    k: int = RRF_K,
    vec_w: float = VEC_WEIGHT,
    bm25_w: float = BM25_WEIGHT,
) -> list[tuple[str, float]]:
    """Combine two ranked lists with weighted RRF.

    Each input list is `[(id, score), ...]` sorted by descending score
    (or returned in rank order — only the rank position matters for RRF).
    Returns `[(id, fused_score), ...]` sorted descending.
    """
    score: dict[str, float] = defaultdict(float)
    for rank, (qid, _) in enumerate(vector_hits):
        score[qid] += vec_w * (1.0 / (k + rank + 1))
    for rank, (qid, _) in enumerate(bm25_hits):
        score[qid] += bm25_w * (1.0 / (k + rank + 1))
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


def bm25_search(
    db: sqlite3.Connection,
    query: str,
    *,
    table: str = "queries_fts",
    join_table: str = "queries",
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Return `[(id, bm25_score)]` from FTS5 in descending rank.

    Returns empty list if FTS5 unavailable or query has no match.
    `bm25()` is negative (lower = better in SQLite's API); we flip the
    sign so callers can interpret a higher score as a stronger match.
    """
    try:
        rows = db.execute(
            f"""
            SELECT t.id, bm25({table}) AS score
            FROM {table}
            JOIN {join_table} t ON t.rowid = {table}.rowid
            WHERE {table} MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [(qid, -float(score)) for (qid, score) in rows]
    except sqlite3.OperationalError:
        # FTS5 not compiled in, or table doesn't exist
        return []


def fts5_available(db: sqlite3.Connection) -> bool:
    """True if SQLite was compiled with FTS5 support."""
    try:
        rows = db.execute(
            "SELECT 1 FROM pragma_compile_options() WHERE compile_options = 'ENABLE_FTS5'"
        ).fetchall()
        return bool(rows)
    except sqlite3.OperationalError:
        # Some builds don't expose pragma_compile_options
        try:
            db.execute("CREATE VIRTUAL TABLE _probe_fts USING fts5(content)")
            db.execute("DROP TABLE _probe_fts")
            return True
        except sqlite3.OperationalError:
            return False
