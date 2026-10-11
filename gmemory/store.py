"""MemoryStore — the one memory surface the orchestrators talk to.

    store = MemoryStore.open("memory.db", cfg=load_config())
    context, result = store.context_for(topic)        # fenced "## Relevant prior findings"
    report = store.persist_task(topic, status, round_outputs)   # Tier 1 + 2 + 3, gate, eviction
    store.close()

One SQLite connection, canonical gmemory schema (migrated on open), FTS5
retrieval that needs nothing beyond the stdlib. The vector side (an
embedder; vectors live in SQLite) and the LLM hooks (`memory_ops` tier)
are optional extras that switch on when present -- their absence never
empties a result or aborts a write.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from gmemory.insight_graph import InsightGraph, heuristic_distill
from gmemory.migrate import is_legacy, migrate_db
from gmemory.promotion import evaluate_candidates, prune_stale_candidates
from gmemory.query_graph import QueryGraph
from gmemory.retrieval import LLM_OPS_THRESHOLD, RetrievalResult, format_context, retrieve

REPO_ROOT = Path(__file__).resolve().parent.parent
UTTERANCE_CONTENT_CHARS = 500


@dataclass
class MemorySettings:
    """The `[memory]` / `[eviction]` / `[storage]` knobs the store reads."""

    top_k: int = 2
    hops: int = 1                     # [memory] hop_expansion; 0 disables graph expansion
    link_threshold: float = 0.7
    dedup_threshold: float = 0.85     # promotion-gate cluster threshold
    llm_ops_threshold: int = LLM_OPS_THRESHOLD
    gmemory_tier: int = 3             # < 3 skips Tier 3 (insight candidates)
    max_context_chars: int = 8000
    archive_days: int = 90
    purge_days: int = 30
    max_archive: int = 50
    cycle_frequency: int = 10

    @classmethod
    def from_config(cls, cfg) -> "MemorySettings":
        m, e = cfg.memory, cfg.eviction
        return cls(
            top_k=int(m.top_k_retrieval),
            hops=int(m.hop_expansion),
            link_threshold=float(m.query_link_threshold),
            dedup_threshold=float(m.insight_dedup_threshold),
            llm_ops_threshold=int(m.llm_ops_threshold),
            gmemory_tier=int(m.gmemory_tier),
            max_context_chars=int(m.max_context_tokens) * 4,
            archive_days=int(e.query_archive_days),
            purge_days=int(e.interaction_purge_days),
            max_archive=int(e.max_archive_per_cycle),
            cycle_frequency=int(e.cycle_frequency),
        )


@dataclass
class PersistReport:
    query_id: str
    status: str
    n_utterances: int
    n_utterance_edges: int
    candidate_ids: list[str] = field(default_factory=list)
    promoted_ids: list[str] = field(default_factory=list)
    vector_source: str = "none"        # 'model' when an embedder produced vectors
    distilled_by: str = "heuristic"    # 'llm' when the memory_ops tier ran
    eviction: dict | None = None       # the cycle's counts when it ran


def build_trace(round_outputs: Sequence[Sequence[dict]]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Tier 1 rows from per-round agent outputs: one utterance per output,
    edges from every output in round n-1 to every output in round n."""
    utterances: list[dict] = []
    edges: list[tuple[str, str]] = []
    prev_ids: list[str] = []
    now = time.time()
    for epoch, outputs in enumerate(round_outputs):
        current: list[str] = []
        for o in outputs:
            uid = str(uuid.uuid4())
            utterances.append({
                "id": uid,
                "agent_id": str(o.get("agent_id") or o.get("role") or ""),
                "content": str(o.get("findings") or o.get("content") or "")[:UTTERANCE_CONTENT_CHARS],
                "epoch": epoch,
                "timestamp": now,
            })
            edges.extend((p, uid) for p in prev_ids)
            current.append(uid)
        prev_ids = current
    return utterances, edges


class MemoryStore:
    def __init__(
        self,
        db: sqlite3.Connection,
        *,
        embedder=None,
        settings: MemorySettings | None = None,
        llm="auto",
        cfg=None,
    ):
        self.db = db
        self.settings = settings or MemorySettings()
        self.embedder = embedder
        self.cfg = cfg
        self._llm = llm              # "auto" -> built lazily once the store is warm
        self.qg = QueryGraph(db, embedder=embedder, link_threshold=self.settings.link_threshold)
        self.ig = InsightGraph(db, embedder=embedder)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path: Path | str, *, cfg=None, embedder="auto", llm="auto") -> "MemoryStore":
        """Open (creating / migrating as needed) the SQLite file at `path`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(path), check_same_thread=False)
        if path.exists() and is_legacy(db):
            # The migration drops the legacy tables in place; keep a one-time
            # copy next to the file so a bad migration is never data loss.
            backup = path.with_name(path.name + ".pre-gmemory.bak")
            if not backup.exists():
                db.close()
                shutil.copy2(path, backup)
                db = sqlite3.connect(str(path), check_same_thread=False)
        migrate_db(db)
        db.execute("PRAGMA foreign_keys = ON")
        if embedder == "auto":
            from embedder import load_embedder  # noqa: PLC0415
            embedder = load_embedder()
        settings = MemorySettings.from_config(cfg) if cfg is not None else MemorySettings()
        return cls(db, embedder=embedder, settings=settings, llm=llm, cfg=cfg)

    @classmethod
    def open_default(cls, cfg=None, **kw) -> "MemoryStore":
        """The store at `[storage] sqlite_path` (relative paths resolve to the repo root)."""
        if cfg is None:
            from config import load_config  # noqa: PLC0415
            cfg = load_config()
        p = Path(cfg.storage.sqlite_path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return cls.open(p, cfg=cfg, **kw)

    # ------------------------------------------------------------------
    # LLM hooks (memory_ops tier) -- only once the store is warm
    # ------------------------------------------------------------------

    @property
    def warm(self) -> bool:
        return self.qg.query_count() >= self.settings.llm_ops_threshold

    def llm(self):
        """The resolved memory_ops helper, or None (heuristics)."""
        if not self.warm:
            return None
        if self._llm == "auto":
            from gmemory.llm_ops import build_memory_llm  # noqa: PLC0415
            self._llm = build_memory_llm(self.cfg)
        return self._llm

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def retrieve(self, topic: str) -> RetrievalResult:
        llm = self.llm()
        return retrieve(
            topic, self.qg, self.ig,
            top_k=self.settings.top_k,
            hops=self.settings.hops,
            use_llm=llm is not None,
            relevance_fn=llm.relevance if llm else None,
            sparsify_fn=llm.sparsify if llm else None,
            model=llm.model if llm else "",
            llm_ops_threshold=self.settings.llm_ops_threshold,
        )

    def context_for(self, topic: str) -> tuple[str, RetrievalResult]:
        """Fenced `## Relevant prior findings` block ("" when nothing) + the raw result."""
        result = self.retrieve(topic)
        return format_context(result, max_chars=self.settings.max_context_chars, fenced=True), result

    def stats(self) -> dict:
        counts = {}
        for table in ("queries", "query_edges", "utterances", "insight_candidates", "insights"):
            counts[table] = int(self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        counts["vector_side"] = "model" if self.embedder is not None else "none"
        return counts

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def persist_task(
        self,
        topic: str,
        status: str,
        round_outputs: Sequence[Sequence[dict]],
        *,
        n_agents: int = 0,
        cost_usd: float | None = None,
        wall_clock_s: float = 0.0,
        quality_score: float | None = None,
        tags: Sequence[str] = (),
        query_id: str | None = None,
    ) -> PersistReport:
        """Store one completed task across all three tiers.

        Tier 2 query node + similarity edges, Tier 1 utterances + edges,
        Tier 3 insight candidates (LLM-distilled on the memory_ops tier when
        warm, heuristic otherwise), then the N=3 promotion gate, stale-
        candidate pruning and -- every `cycle_frequency` tasks -- eviction.
        A missing embedder stores rows without vectors and flags them; it
        never rolls the task back.
        """
        if status not in ("resolved", "failed", "partial"):
            status = "failed"
        rounds = [list(r) for r in round_outputs]
        qid = self.qg.add_task(
            topic, status, n_agents=n_agents, n_rounds=len(rounds), cost_usd=cost_usd,
            wall_clock_s=wall_clock_s, quality_score=quality_score, tags=tags, query_id=query_id,
        )

        utterances, edges = build_trace(rounds)
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO utterances (id, query_id, agent_id, content, epoch, timestamp) "
                "VALUES (?,?,?,?,?,?)",
                [(u["id"], qid, u["agent_id"], u["content"], u["epoch"], u["timestamp"]) for u in utterances],
            )
            self.db.executemany(
                "INSERT OR IGNORE INTO utterance_edges (source_id, target_id, query_id) VALUES (?,?,?)",
                [(s, t, qid) for (s, t) in edges],
            )

        report = PersistReport(
            query_id=qid, status=status,
            n_utterances=len(utterances), n_utterance_edges=len(edges),
            vector_source="model" if self.embedder is not None else "none",
        )

        if self.settings.gmemory_tier >= 3:
            llm = self.llm()
            self.ig.distill_fn = llm.distill if llm else heuristic_distill
            report.distilled_by = "llm" if llm else "heuristic"
            flat = [o for rnd in rounds for o in rnd if isinstance(o, dict)]
            report.candidate_ids = self.ig.extract_insights(qid, topic, flat, status, min_trace_len=1)
            report.promoted_ids = evaluate_candidates(
                self.db, cluster_thresh=self.settings.dedup_threshold,
            )
            prune_stale_candidates(self.db)

        report.eviction = self.maybe_run_eviction()
        return report

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def run_eviction_cycle(self) -> dict:
        """Archive cold queries, purge old Tier 1 detail. Insights never evict.

        Cascades (schema ON DELETE CASCADE) remove the archived queries'
        utterances, edges, tags and candidate insights with them.
        """
        s = self.settings
        now = time.time()
        archive_cutoff = now - s.archive_days * 86400
        purge_cutoff = now - s.purge_days * 86400
        with self.db:
            cold = [r[0] for r in self.db.execute(
                "SELECT id FROM queries WHERE last_accessed < ? AND access_count < 3 "
                "ORDER BY last_accessed ASC LIMIT ?",
                (archive_cutoff, s.max_archive),
            ).fetchall()]
            for qid in cold:
                self.db.execute(
                    "INSERT OR IGNORE INTO queries_archive "
                    "(id, query_text, status, created_at, access_count, last_accessed, archived_at) "
                    "SELECT id, query_text, status, created_at, access_count, last_accessed, ? "
                    "FROM queries WHERE id = ?",
                    (now, qid),
                )
                self.db.execute("DELETE FROM queries WHERE id = ?", (qid,))
            purged = self.db.execute(
                "DELETE FROM utterances WHERE query_id IN ("
                "  SELECT id FROM queries WHERE created_at < ?"
                ") AND query_id NOT IN ("
                "  SELECT id FROM queries ORDER BY access_count DESC LIMIT 100)",
                (purge_cutoff,),
            ).rowcount or 0
        counts = {"archived": len(cold), "purged_utterances": int(purged)}
        if cold or purged:
            print(f"[memory] eviction: archived {len(cold)} queries, purged {purged} utterances")
        return counts

    def maybe_run_eviction(self) -> dict | None:
        """Run the cycle every `cycle_frequency` stored tasks (counted in the
        DB, so one-task-per-process runs still trigger it)."""
        n = self.settings.cycle_frequency
        if n <= 0:
            return None
        total = self.qg.query_count()
        if total > 0 and total % n == 0:
            return self.run_eviction_cycle()
        return None

    def close(self) -> None:
        try:
            self.db.close()
        except sqlite3.Error:
            pass
