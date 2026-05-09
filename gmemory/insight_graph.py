"""Tier 3 — Insight Graph.

Owns the lifecycle of candidate insights and promoted insight nodes.
Distillation is a Haiku subprocess in production; this module exposes
a pluggable `distill_fn` so tests can run without LLM calls.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from gmemory.promotion import (
    _decode_blob,
    _encode_blob,
    cosine,
    evaluate_candidates,
)


DISTILL_MAX_INSIGHTS = 3
DISTILL_MIN_TRACE_LEN = 3


@dataclass(slots=True)
class InsightCandidate:
    content: str
    tags: list[str]
    confidence: float = 0.5


@dataclass(slots=True)
class InsightRow:
    id: str
    content: str
    supporting_queries: list[str]
    tags: list[str]
    confidence: float
    created_at: float
    last_validated_at: float


# Pluggable LLM hook
DistillFn = Callable[[str, str, list[dict], str], list[InsightCandidate]]


def heuristic_distill(
    topic: str,
    status: str,
    utterances: list[dict],
    cost_str: str = "",
) -> list[InsightCandidate]:
    """Heuristic distillation — no LLM call.

    Extracts up to DISTILL_MAX_INSIGHTS one-line lessons from key_points,
    deduplicating within the call (a real distillation LLM produces
    distinct insights, never repeats). Real implementation calls Haiku
    with a structured JSON prompt.
    """
    insights: list[InsightCandidate] = []
    seen: set[str] = set()
    for u in utterances:
        kps = u.get("key_points") or []
        for kp in kps:
            if not isinstance(kp, str):
                continue
            kp = kp.strip()
            if not kp or kp in seen:
                continue
            seen.add(kp)
            insights.append(InsightCandidate(
                content=kp[:180],
                tags=[f"domain:{u.get('domain', 'general')}"],
                confidence=float(u.get("confidence", 0.6) or 0.6),
            ))
            if len(insights) >= DISTILL_MAX_INSIGHTS:
                return insights
    if not insights and utterances:
        # Fallback: short summary line
        first_findings = utterances[0].get("findings", "")
        if first_findings:
            insights.append(InsightCandidate(
                content=f"On '{topic}': " + first_findings[:120],
                tags=[],
                confidence=0.5,
            ))
    return insights


class InsightGraph:
    """Tier 3 — distilled cross-task wisdom."""

    def __init__(
        self,
        db: sqlite3.Connection,
        *,
        embedder=None,
        distill_fn: DistillFn | None = None,
    ):
        self.db = db
        self.embedder = embedder
        self.distill_fn = distill_fn or heuristic_distill

    def _embed(self, text: str) -> tuple[float, ...]:
        if self.embedder is not None:
            try:
                return tuple(float(x) for x in self.embedder.encode(text))
            except Exception:
                pass
        # Reuse QueryGraph's deterministic stub for consistency
        from gmemory.query_graph import QueryGraph
        return QueryGraph._fallback_embed(text)

    # ------------------------------------------------------------------
    # Distillation
    # ------------------------------------------------------------------

    def extract_insights(
        self,
        query_id: str,
        topic: str,
        utterances: list[dict],
        status: str,
    ) -> list[str]:
        """Run distillation and write candidates. Returns list of candidate IDs.

        Idempotent on (query_id) — re-runs that produce duplicate candidates
        are deduplicated at the (query_id, content) level.
        """
        if len(utterances) < DISTILL_MIN_TRACE_LEN:
            return []

        candidates = self.distill_fn(topic, status, utterances, "")
        if not candidates:
            return []

        existing_contents: set[str] = set()
        rows = self.db.execute(
            "SELECT content FROM insight_candidates WHERE query_id = ?",
            (query_id,),
        ).fetchall()
        for r in rows:
            existing_contents.add(r[0])

        new_ids: list[str] = []
        now = time.time()
        with self.db:
            for cand in candidates:
                if cand.content in existing_contents:
                    continue
                cand_id = str(uuid.uuid4())
                emb = self._embed(cand.content)
                self.db.execute(
                    """
                    INSERT INTO insight_candidates
                    (id, content, query_id, embedding, confidence, tags_json, created_at)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        cand_id, cand.content, query_id,
                        _encode_blob(emb), cand.confidence,
                        json.dumps(cand.tags), now,
                    ),
                )
                new_ids.append(cand_id)
        return new_ids

    # ------------------------------------------------------------------
    # Promotion (delegates to gmemory.promotion)
    # ------------------------------------------------------------------

    def promote_to_insight(
        self,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[str]:
        return evaluate_candidates(self.db)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_insights_for_query(
        self,
        query_text: str,
        related_query_ids: Iterable[str],
        *,
        limit: int = 5,
    ) -> list[dict]:
        related = list(related_query_ids)
        if not related:
            return self._top_insights_by_text(query_text, limit=limit)

        placeholders = ",".join("?" for _ in related)
        rows = self.db.execute(
            f"""
            SELECT i.id, i.content, i.confidence, i.last_validated_at,
                   i.supporting_queries,
                   COUNT(DISTINCT iqi.query_id) AS overlap
            FROM insights i
            JOIN insight_query_index iqi ON i.id = iqi.insight_id
            WHERE iqi.query_id IN ({placeholders})
            GROUP BY i.id
            ORDER BY overlap DESC, i.confidence DESC, i.last_validated_at DESC
            LIMIT ?
            """,
            (*related, limit),
        ).fetchall()
        out: list[dict] = []
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
                "supporting_count": int(r[5] or 0),
            })
        return out

    def _top_insights_by_text(self, query_text: str, *, limit: int) -> list[dict]:
        """Fallback when there are no related queries: rank by recency + confidence."""
        rows = self.db.execute(
            """
            SELECT id, content, confidence, last_validated_at, supporting_queries
            FROM insights
            ORDER BY last_validated_at DESC, confidence DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        out: list[dict] = []
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
                "supporting_count": 0,
            })
        return out

    def aggregate_overnight(self, *, since_hours: int = 24) -> dict:
        """Run a promotion sweep + dedup-merge of near-duplicate insights."""
        promoted = evaluate_candidates(self.db)
        # Skip merge for now — needs full embedding comparison loop
        return {
            "promoted": len(promoted),
            "merged": 0,
            "candidates_seen": int(self.db.execute(
                "SELECT COUNT(*) FROM insight_candidates "
                "WHERE created_at > ?",
                (time.time() - since_hours * 3600,),
            ).fetchone()[0]),
        }
