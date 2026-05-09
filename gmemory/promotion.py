"""Promotion gate — N-distinct-query support across single-link clusters.

Adapts GAM (arXiv 2604.12285) for sessionless swarm runs: G-Memory
session boundaries do not exist for Blitz-Swarm, so the LLM-discrimination
signal GAM uses isn't available. We replace it with a structural rule:
a candidate insight is only promoted when N=3+ distinct query traces
contribute semantically clustered candidates. Trades recall for
precision; the holding table preserves recall.

Algorithm:
    1. Pull unpromoted candidates with embeddings.
    2. Single-link agglomerative cluster on cosine, threshold 0.78.
    3. For each cluster:
        a. support_size = |distinct(query_ids)|
        b. If support_size < N: tag last_evaluated_at, skip.
        c. Else: promote (insert into insights, mark candidates promoted).
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Sequence


PROMOTION_N = 3
CLUSTER_COSINE_THRESH = 0.78
MAX_PROMOTIONS_PER_RUN = 25
CANDIDATE_TTL_DAYS = 14


@dataclass(slots=True)
class CandidateRow:
    id: str
    content: str
    query_id: str
    embedding: tuple[float, ...]
    confidence: float
    tags: list[str]
    created_at: float


@dataclass(slots=True)
class Cluster:
    candidate_ids: list[str]
    embeddings: list[tuple[float, ...]]
    query_ids: set[str]
    confidences: list[float]
    tags: set[str]
    contents: list[str]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _decode_blob(b: bytes) -> tuple[float, ...]:
    """Decode a raw f32 little-endian BLOB into a tuple of floats."""
    import struct
    if not b:
        return ()
    n = len(b) // 4
    return struct.unpack(f"<{n}f", b)


def _encode_blob(vec: Sequence[float]) -> bytes:
    import struct
    return struct.pack(f"<{len(vec)}f", *vec)


def _fetch_candidates(db: sqlite3.Connection) -> list[CandidateRow]:
    rows = db.execute(
        """
        SELECT id, content, query_id, embedding, confidence, tags_json, created_at
        FROM insight_candidates
        WHERE promoted_at IS NULL
        ORDER BY created_at ASC
        """
    ).fetchall()
    out: list[CandidateRow] = []
    for r in rows:
        emb = _decode_blob(r[3])
        try:
            tags = json.loads(r[5]) if r[5] else []
        except json.JSONDecodeError:
            tags = []
        out.append(CandidateRow(
            id=r[0], content=r[1], query_id=r[2], embedding=emb,
            confidence=float(r[4] or 0.5), tags=tags, created_at=float(r[6]),
        ))
    return out


def _cluster_candidates(
    candidates: list[CandidateRow], threshold: float,
) -> list[Cluster]:
    """Single-link agglomerative clustering on cosine."""
    if not candidates:
        return []

    parent: dict[str, str] = {c.id: c.id for c in candidates}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            sim = cosine(candidates[i].embedding, candidates[j].embedding)
            if sim >= threshold:
                union(candidates[i].id, candidates[j].id)

    clusters_by_root: dict[str, list[CandidateRow]] = {}
    for c in candidates:
        clusters_by_root.setdefault(find(c.id), []).append(c)

    out: list[Cluster] = []
    for members in clusters_by_root.values():
        out.append(Cluster(
            candidate_ids=[m.id for m in members],
            embeddings=[m.embedding for m in members],
            query_ids={m.query_id for m in members},
            confidences=[m.confidence for m in members],
            tags={t for m in members for t in m.tags},
            contents=[m.content for m in members],
        ))
    return out


def _centroid(embeddings: list[tuple[float, ...]]) -> tuple[float, ...]:
    if not embeddings:
        return ()
    dim = len(embeddings[0])
    out = [0.0] * dim
    for emb in embeddings:
        for i in range(min(dim, len(emb))):
            out[i] += emb[i]
    n = len(embeddings)
    for i in range(dim):
        out[i] /= n
    norm = math.sqrt(sum(x * x for x in out))
    if norm > 0:
        out = [x / norm for x in out]
    return tuple(out)


def _canonical_member(cluster: Cluster) -> int:
    """Index of the candidate closest to the centroid; confidence as tiebreak."""
    centroid = _centroid(cluster.embeddings)
    best_idx = 0
    best = (-1.0, -1.0)
    for i, emb in enumerate(cluster.embeddings):
        sim = cosine(emb, centroid)
        score = (sim, cluster.confidences[i])
        if score > best:
            best = score
            best_idx = i
    return best_idx


def _promote_cluster(
    db: sqlite3.Connection,
    cluster: Cluster,
) -> str | None:
    """Insert one promoted insight; mark candidates promoted."""
    if not cluster.candidate_ids:
        return None
    canonical_idx = _canonical_member(cluster)
    canonical_content = cluster.contents[canonical_idx]
    centroid_emb = _centroid(cluster.embeddings)
    avg_conf = sum(cluster.confidences) / len(cluster.confidences)

    insight_id = str(uuid.uuid4())
    now = time.time()

    with db:
        db.execute(
            """
            INSERT INTO insights
            (id, content, supporting_queries, tags_json, created_at,
             last_validated_at, access_count, confidence, embedding)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                insight_id, canonical_content,
                json.dumps(sorted(cluster.query_ids)),
                json.dumps(sorted(cluster.tags)),
                now, now, 0, avg_conf,
                _encode_blob(centroid_emb),
            ),
        )
        for cand_id in cluster.candidate_ids:
            db.execute(
                """
                UPDATE insight_candidates
                SET promoted_to = ?, promoted_at = ?
                WHERE id = ?
                """,
                (insight_id, now, cand_id),
            )
        for qid in cluster.query_ids:
            db.execute(
                "INSERT OR IGNORE INTO insight_query_index (insight_id, query_id) "
                "VALUES (?, ?)",
                (insight_id, qid),
            )
        for tag in cluster.tags:
            if ":" in tag:
                axis, value = tag.split(":", 1)
                db.execute(
                    "INSERT OR IGNORE INTO insight_tags (insight_id, axis, value) "
                    "VALUES (?, ?, ?)",
                    (insight_id, axis, value),
                )
    return insight_id


def _connect_to_existing_insights(
    db: sqlite3.Connection, new_insight_id: str,
) -> int:
    """Form insight_edges to insights sharing >=2 query_ids."""
    rows = db.execute(
        """
        SELECT iqi1.insight_id, COUNT(*) AS shared, MAX(iqi1.query_id) AS last_qid
        FROM insight_query_index iqi1
        JOIN insight_query_index iqi2 ON iqi1.query_id = iqi2.query_id
        WHERE iqi1.insight_id != ? AND iqi2.insight_id = ?
        GROUP BY iqi1.insight_id
        HAVING shared >= 2
        """,
        (new_insight_id, new_insight_id),
    ).fetchall()
    now = time.time()
    edge_count = 0
    with db:
        for (other_insight, _shared, via_qid) in rows:
            db.execute(
                """
                INSERT OR IGNORE INTO insight_edges
                (source_insight_id, target_insight_id, via_query_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (new_insight_id, other_insight, via_qid, now),
            )
            edge_count += 1
    return edge_count


def evaluate_candidates(
    db: sqlite3.Connection,
    *,
    n_required: int = PROMOTION_N,
    cluster_thresh: float = CLUSTER_COSINE_THRESH,
    max_promotions: int = MAX_PROMOTIONS_PER_RUN,
) -> list[str]:
    """Run the promotion gate. Returns IDs of newly promoted insights."""
    candidates = _fetch_candidates(db)
    if len(candidates) < n_required:
        return []

    clusters = _cluster_candidates(candidates, cluster_thresh)
    promoted: list[str] = []
    now = time.time()
    for cluster in clusters:
        if len(promoted) >= max_promotions:
            break
        if len(cluster.query_ids) < n_required:
            with db:
                db.executemany(
                    "UPDATE insight_candidates SET last_evaluated_at = ? WHERE id = ?",
                    [(now, cid) for cid in cluster.candidate_ids],
                )
            continue
        new_id = _promote_cluster(db, cluster)
        if new_id:
            _connect_to_existing_insights(db, new_id)
            promoted.append(new_id)
    return promoted


def prune_stale_candidates(
    db: sqlite3.Connection, ttl_days: int = CANDIDATE_TTL_DAYS,
) -> int:
    """Delete unpromoted candidates older than TTL. Returns count pruned."""
    cutoff = time.time() - ttl_days * 86400
    with db:
        cur = db.execute(
            "DELETE FROM insight_candidates WHERE promoted_at IS NULL AND created_at < ?",
            (cutoff,),
        )
        return cur.rowcount or 0
