"""Reciprocal Rank Fusion of vector and BM25 retrieval.

Weighted RRF as in Cormack et al.; the recommended k=60 default
flattens above 40 so small-K perturbations don't change ranking.
Vector dominates on paraphrase; BM25 catches rare-token jargon.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from typing import Iterable

RRF_K = 60
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
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


def fts_match_query(text: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    FTS5 has its own query grammar: `?`, `-`, `(`, `)`, `:`, `*`, `^`, bare
    apostrophes and the words AND/OR/NOT all mean something. Every token is
    therefore double-quoted (a literal phrase; embedded quotes doubled) and
    tokens are OR-ed so a long topic still matches partial overlap -- bm25()
    does the ranking. Returns "" when the text has no word tokens (callers
    must then skip the MATCH entirely).
    """
    tokens = _TOKEN_RE.findall(text or "")
    if not tokens:
        return ""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def bm25_search(
    db: sqlite3.Connection,
    query: str,
    *,
    table: str = "queries_fts",
    join_table: str = "queries",
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Return `[(id, bm25_score)]` from FTS5 in descending rank.

    `query` is free text; it is escaped with `fts_match_query` here, so
    callers never hand raw user text to MATCH. Returns an empty list if
    FTS5 is unavailable, the text has no tokens, or nothing matches.
    `bm25()` is negative (lower = better in SQLite's API); we flip the
    sign so callers can interpret a higher score as a stronger match.
    """
    match = fts_match_query(query)
    if not match:
        return []
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
            (match, limit),
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
