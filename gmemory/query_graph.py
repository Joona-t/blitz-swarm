"""Tier 2 — Query Graph operations.

Owns the lifecycle of QueryNode rows in SQLite. The vector side is
optional: with an embedder, query vectors are stored in `queries.embedding`
and kNN is a brute-force cosine scan (fine to ~10K nodes; no LanceDB --
that path had no caller and read L2 distance as a similarity, so it was
deleted rather than kept). Without an embedder `nearest_neighbors` returns
nothing and FTS5/BM25 (gmemory.hybrid) is the retrieval signal. No
pseudo-vectors are ever fabricated -- a missing embedder is a missing
vector, flagged as such, never a hash dressed up as semantics.

Pure data-layer module: zero LLM calls, zero subprocess.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from gmemory.hybrid import bm25_search
from gmemory.promotion import _decode_blob, _encode_blob, cosine

KNN_K = 5
COSINE_LINK_THRESHOLD = 0.70
ONE_HOP_FANOUT_CAP = 30
EMBED_DIM = 384


@dataclass(slots=True)
class QueryRecord:
    id: str
    query_text: str
    status: str
    created_at: float
    n_agents: int
    n_rounds: int
    cost_usd: float | None      # None = provider reported no cost (never 0)
    wall_clock_s: float
    quality_score: float | None
    tags: list[str]


def _materialize_tags(db: sqlite3.Connection, query_id: str, tags: Sequence[str]) -> None:
    """Denormalize tag JSON into query_tags table for meta-loop fast filter."""
    db.execute("DELETE FROM query_tags WHERE query_id = ?", (query_id,))
    for tag in tags:
        if ":" in tag:
            axis, value = tag.split(":", 1)
            db.execute(
                "INSERT OR IGNORE INTO query_tags (query_id, axis, value) VALUES (?, ?, ?)",
                (query_id, axis, value),
            )


class QueryGraph:
    """Tier 2 — task-level memory with kNN edge formation."""

    def __init__(
        self,
        db: sqlite3.Connection,
        *,
        embedder=None,
        link_threshold: float = COSINE_LINK_THRESHOLD,
    ):
        self.db = db
        self.embedder = embedder
        self.link_threshold = link_threshold
        # In-process cache of (qid, vector) mirroring queries.embedding.
        self._vectors: list[tuple[str, tuple[float, ...]]] = []
        self._reload_vectors()

    def _reload_vectors(self) -> None:
        """Load every stored query vector (rows written with an embedder)."""
        try:
            rows = self.db.execute(
                "SELECT id, embedding FROM queries WHERE embedding IS NOT NULL"
            ).fetchall()
        except sqlite3.OperationalError:   # schema not applied yet
            rows = []
        self._vectors = [(qid, _decode_blob(blob)) for qid, blob in rows if blob]

    def _embed(self, text: str) -> tuple[float, ...] | None:
        """Model embedding, or None when no embedder is configured / it fails."""
        if self.embedder is None:
            return None
        try:
            vec = self.embedder.encode(text)
            return tuple(float(x) for x in vec)
        except Exception:
            return None

    @staticmethod
    def lexical_similarity(a: str, b: str) -> float:
        """Jaccard token overlap in [0, 1] -- the edge weight used when no
        vector is available. Identical texts score 1.0."""
        aw = set(a.lower().split())
        bw = set(b.lower().split())
        if not aw or not bw:
            return 0.0
        return len(aw & bw) / len(aw | bw)

    def lexical_neighbors(
        self,
        query_text: str,
        *,
        k: int = KNN_K,
        threshold: float = COSINE_LINK_THRESHOLD,
        exclude_ids: Iterable[str] = (),
    ) -> list[tuple[str, float]]:
        """BM25 candidates re-scored with Jaccard so the edge weight is a
        bounded similarity. Pure SQLite + FTS5; no optional dependency."""
        excl = set(exclude_ids)
        hits = bm25_search(self.db, query_text, limit=k * 4)
        if not hits:
            return []
        texts = {r.id: r.query_text for r in self.get_records([qid for qid, _ in hits])}
        out: list[tuple[str, float]] = []
        for qid, _ in hits:
            if qid in excl or qid not in texts:
                continue
            sim = self.lexical_similarity(query_text, texts[qid])
            if sim >= threshold:
                out.append((qid, sim))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:k]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add_task(
        self,
        query_text: str,
        status: str,
        *,
        n_agents: int = 0,
        n_rounds: int = 0,
        cost_usd: float | None = None,
        wall_clock_s: float = 0.0,
        quality_score: float | None = None,
        tags: Sequence[str] = (),
        query_id: str | None = None,
    ) -> str:
        qid = query_id or str(uuid.uuid4())
        now = time.time()
        embedding = self._embed(query_text)

        if embedding is not None:
            neighbors = self.nearest_neighbors(
                query_text, k=KNN_K, threshold=self.link_threshold,
                exclude_ids=(qid,), embedding=embedding,
            )
        else:
            neighbors = self.lexical_neighbors(
                query_text, k=KNN_K, threshold=self.link_threshold,
                exclude_ids=(qid,),
            )
        with self.db:
            self.db.execute(
                """
                INSERT OR IGNORE INTO queries
                (id, query_text, status, created_at, access_count, last_accessed,
                 n_agents, n_rounds, cost_usd, wall_clock_s, quality_score, tags_json,
                 embedding)
                VALUES (?,?,?,?,0,?,?,?,?,?,?,?,?)
                """,
                (
                    qid, query_text, status, now, now,
                    n_agents, n_rounds, cost_usd, wall_clock_s,
                    quality_score, json.dumps(list(tags)),
                    _encode_blob(embedding) if embedding is not None else None,
                ),
            )
            for nid, sim in neighbors:
                self.db.execute(
                    "INSERT OR IGNORE INTO query_edges (source_id, target_id, similarity, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (qid, nid, sim, now),
                )
            _materialize_tags(self.db, qid, tags)

        if embedding is not None and not any(v[0] == qid for v in self._vectors):
            self._vectors.append((qid, embedding))
        return qid

    def link(self, source_id: str, target_id: str, similarity: float) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO query_edges (source_id, target_id, similarity, created_at) "
                "VALUES (?, ?, ?, ?)",
                (source_id, target_id, similarity, time.time()),
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def query_count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM queries").fetchone()[0])

    def nearest_neighbors(
        self,
        query_text: str,
        *,
        k: int = KNN_K,
        threshold: float = COSINE_LINK_THRESHOLD,
        exclude_ids: Iterable[str] = (),
        embedding: Sequence[float] | None = None,
    ) -> list[tuple[str, float]]:
        if embedding is None:
            embedding = self._embed(query_text)
        if embedding is None:
            # No vector side available: callers fall back to BM25/lexical.
            return []
        excl = set(exclude_ids)

        # Brute-force cosine over the stored vectors
        scored: list[tuple[str, float]] = []
        for qid, emb in self._vectors:
            if qid in excl:
                continue
            sim = cosine(embedding, emb)
            if sim >= threshold:
                scored.append((qid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def one_hop(self, seed_ids: Iterable[str]) -> set[str]:
        seeds = list(seed_ids)
        if not seeds:
            return set()
        placeholders = ",".join("?" for _ in seeds)
        rows = self.db.execute(
            f"""
            SELECT source_id, target_id FROM query_edges
            WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
            """,
            (*seeds, *seeds),
        ).fetchall()
        # Dedup, cap fanout per seed
        per_seed_count: dict[str, int] = {s: 0 for s in seeds}
        out: set[str] = set()
        for src, tgt in rows:
            for seed, neighbor in ((src, tgt), (tgt, src)):
                if seed not in per_seed_count:
                    continue
                if per_seed_count[seed] >= ONE_HOP_FANOUT_CAP:
                    continue
                if neighbor in seeds:
                    continue
                out.add(neighbor)
                per_seed_count[seed] += 1
        return out

    def get_records(self, query_ids: Iterable[str]) -> list[QueryRecord]:
        ids = list(query_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.db.execute(
            f"""
            SELECT id, query_text, status, created_at, n_agents, n_rounds,
                   cost_usd, wall_clock_s, quality_score, tags_json
            FROM queries
            WHERE id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        out: list[QueryRecord] = []
        for r in rows:
            try:
                tags = json.loads(r[9]) if r[9] else []
            except json.JSONDecodeError:
                tags = []
            out.append(QueryRecord(
                id=r[0], query_text=r[1], status=r[2],
                created_at=float(r[3]), n_agents=int(r[4] or 0),
                n_rounds=int(r[5] or 0), cost_usd=float(r[6]) if r[6] is not None else None,
                wall_clock_s=float(r[7] or 0),
                quality_score=float(r[8]) if r[8] is not None else None,
                tags=tags,
            ))
        return out

    def retrieve_context(
        self,
        query_text: str,
        *,
        k: int = KNN_K,
        threshold: float = COSINE_LINK_THRESHOLD,
    ) -> dict:
        seeds = [qid for (qid, _) in self.nearest_neighbors(
            query_text, k=k, threshold=threshold, exclude_ids=(),
        )] or [qid for (qid, _) in self.lexical_neighbors(
            query_text, k=k, threshold=threshold,
        )]
        expanded = self.one_hop(seeds) | set(seeds)
        return {
            "seeds": seeds,
            "expanded": sorted(expanded),
            "records": self.get_records(expanded),
        }

    def touch(self, query_ids: Iterable[str]) -> None:
        """Bump access_count / last_accessed for retrieved queries (eviction
        keys off these)."""
        ids = list(query_ids)
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self.db:
            self.db.execute(
                f"UPDATE queries SET access_count = access_count + 1, last_accessed = ? "
                f"WHERE id IN ({placeholders})",
                (time.time(), *ids),
            )

    def stats(self) -> dict:
        n_queries = int(self.db.execute("SELECT COUNT(*) FROM queries").fetchone()[0])
        n_edges = int(self.db.execute("SELECT COUNT(*) FROM query_edges").fetchone()[0])
        return {
            "n_queries": n_queries,
            "n_edges": n_edges,
            "mean_degree": (n_edges * 2 / n_queries) if n_queries else 0,
        }


def initialize_schema(db: sqlite3.Connection, schema_path: Path | None = None) -> None:
    """Apply gmemory/schema.sql to the connected database."""
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text()
    db.executescript(sql)
