"""Tier 2 — Query Graph operations.

Owns the lifecycle of QueryNode rows in SQLite. LanceDB is the
preferred ANN backend; when missing, we fall back to brute-force cosine
in Python (fine up to ~10K nodes, which is well past the typical
solo-dev usage horizon).

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
    cost_usd: float
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
        lance_table=None,
    ):
        self.db = db
        self.embedder = embedder
        self.lance_table = lance_table
        # Pure-Python fallback vector store: list of (qid, embedding)
        self._vectors: list[tuple[str, tuple[float, ...]]] = []
        if self.lance_table is None:
            self._reload_vectors()

    def _reload_vectors(self) -> None:
        """Reload embeddings from a stash table when no LanceDB."""
        # We don't store embeddings on the queries table by default;
        # this fallback uses an in-memory cache. New embeddings get
        # appended in add_task().
        self._vectors = []

    def _embed(self, text: str) -> tuple[float, ...]:
        if self.embedder is None:
            # Deterministic stub so tests can run without sentence-transformers
            return self._fallback_embed(text)
        try:
            vec = self.embedder.encode(text)
            return tuple(float(x) for x in vec)
        except Exception:
            return self._fallback_embed(text)

    @staticmethod
    def _fallback_embed(text: str, dim: int = EMBED_DIM) -> tuple[float, ...]:
        """Hash-based pseudo-embedding for tests when embedder is unavailable.

        Maps text -> normalized vector deterministically. Two identical
        strings get identical vectors; semantic similarity is replaced
        by hash collision (so vector kNN is meaningless under fallback —
        BM25 is the only useful retrieval signal in this mode).
        """
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # Tile the digest to fill `dim` floats in [-1, 1]
        out = [0.0] * dim
        for i in range(dim):
            byte = h[i % len(h)]
            out[i] = (byte / 127.5) - 1.0
        norm = math.sqrt(sum(x * x for x in out))
        if norm > 0:
            out = [x / norm for x in out]
        return tuple(out)

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
        cost_usd: float = 0.0,
        wall_clock_s: float = 0.0,
        quality_score: float | None = None,
        tags: Sequence[str] = (),
        query_id: str | None = None,
    ) -> str:
        qid = query_id or str(uuid.uuid4())
        now = time.time()
        embedding = self._embed(query_text)

        neighbors = self.nearest_neighbors(
            query_text, k=KNN_K, threshold=COSINE_LINK_THRESHOLD,
            exclude_ids=(qid,), embedding=embedding,
        )
        with self.db:
            self.db.execute(
                """
                INSERT OR IGNORE INTO queries
                (id, query_text, status, created_at, access_count, last_accessed,
                 n_agents, n_rounds, cost_usd, wall_clock_s, quality_score, tags_json)
                VALUES (?,?,?,?,0,?,?,?,?,?,?,?)
                """,
                (
                    qid, query_text, status, now, now,
                    n_agents, n_rounds, cost_usd, wall_clock_s,
                    quality_score, json.dumps(list(tags)),
                ),
            )
            for nid, sim in neighbors:
                self.db.execute(
                    "INSERT OR IGNORE INTO query_edges (source_id, target_id, similarity, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (qid, nid, sim, now),
                )
            _materialize_tags(self.db, qid, tags)

        if self.lance_table is not None:
            try:
                self.lance_table.add([{
                    "query_id": qid, "text": query_text, "vector": list(embedding),
                    "status": status, "created_at": now,
                }])
            except Exception:
                pass
        else:
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
        excl = set(exclude_ids)

        if self.lance_table is not None:
            try:
                df = (self.lance_table
                      .search(list(embedding))
                      .distance_type("cosine")
                      .limit(k * 2)
                      .to_pandas())
                out: list[tuple[str, float]] = []
                for _, row in df.iterrows():
                    qid = row["query_id"]
                    if qid in excl:
                        continue
                    sim = 1.0 - float(row["_distance"])
                    if sim >= threshold:
                        out.append((qid, sim))
                    if len(out) >= k:
                        break
                return out
            except Exception:
                pass

        # Brute-force fallback over self._vectors
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
                n_rounds=int(r[5] or 0), cost_usd=float(r[6] or 0),
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
        )]
        expanded = self.one_hop(seeds) | set(seeds)
        return {
            "seeds": seeds,
            "expanded": sorted(expanded),
            "records": self.get_records(expanded),
        }

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
