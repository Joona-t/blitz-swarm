"""MemoryWriter — single-writer daemon for all memory mutations.

All memory writes funnel through this single process, eliminating SQLite
write contention at any agent count. Consumes from Redis Stream
`memory:writes` and persists to SQLite (G-Memory tiers) + LanceDB (vectors).
"""

import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path

from embedder import cosine_similarity, get_embedder
from memory.models import TaskStatus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = Path(__file__).parent.parent / "memory.db"
DEFAULT_LANCE_PATH = Path(__file__).parent.parent / "memory_vectors"
INSIGHT_DEDUP_THRESHOLD = 0.85
QUERY_LINK_THRESHOLD = 0.7


class MemoryWriter:
    """Single-writer daemon for all memory mutations.

    Consumes write events and persists to:
    - SQLite: G-Memory three tiers (interaction, query, insight graphs)
    - LanceDB: vector embeddings for query similarity search
    """

    def __init__(
        self,
        redis_client=None,
        db_path: Path = DEFAULT_DB_PATH,
        lance_path: Path = DEFAULT_LANCE_PATH,
    ):
        self.redis = redis_client
        self.db_path = db_path
        self.lance_path = lance_path
        self._db = None
        self._lance_db = None
        self._query_table = None
        self._embedder = None
        self._running = False
        self._last_id = "0-0"
        self._write_count = 0

    def initialize(self):
        """Initialize storage backends."""
        self._init_sqlite()
        self._init_lancedb()
        self._embedder = get_embedder()

    def _init_sqlite(self):
        """Create SQLite database and apply schema."""
        self._db = sqlite3.connect(str(self.db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.execute("PRAGMA synchronous=NORMAL")

        schema_sql = SCHEMA_PATH.read_text()
        # Execute each statement separately (executescript handles this)
        self._db.executescript(schema_sql)

    def _init_lancedb(self):
        """Initialize LanceDB for vector storage."""
        try:
            import lancedb
            self._lance_db = lancedb.connect(str(self.lance_path))

            # Create or open the queries table
            try:
                self._query_table = self._lance_db.open_table("queries")
            except Exception:
                # Table doesn't exist yet — will be created on first insert
                self._query_table = None

        except ImportError:
            print("[MemoryWriter] LanceDB not available — vector search disabled")
            self._lance_db = None

    # ------------------------------------------------------------------
    # Redis Stream consumer
    # ------------------------------------------------------------------

    async def start(self):
        """Start consuming memory writes from the Redis Stream."""
        if self.redis is None:
            return

        self._running = True
        print("[MemoryWriter] Started — consuming memory:writes stream")

        while self._running:
            try:
                entries = self.redis.xread(
                    {"memory:writes": self._last_id},
                    count=10,
                    block=500,
                )

                if not entries:
                    continue

                for stream_name, messages in entries:
                    for msg_id, fields in messages:
                        self._last_id = msg_id
                        payload = json.loads(fields.get("payload", "{}"))
                        self.process_write(payload)

            except Exception as e:
                print(f"[MemoryWriter] Error: {e}")
                await asyncio.sleep(1)

    def stop(self):
        """Stop the writer loop and close connections."""
        self._running = False
        if self._db:
            self._db.close()
        print(f"[MemoryWriter] Stopped — processed {self._write_count} writes")

    # ------------------------------------------------------------------
    # Write processing
    # ------------------------------------------------------------------

    def process_write(self, payload: dict):
        """Process a memory write event. Dispatches by type."""
        write_type = payload.get("type", "unknown")
        self._write_count += 1

        if write_type == "task_complete":
            self._process_task_complete(payload)
        else:
            print(f"[MemoryWriter] Unknown write type: {write_type}")

    def _process_task_complete(self, payload: dict):
        """Process a completed task — persist all three memory tiers."""
        query_id = payload.get("query_id", str(uuid.uuid4()))
        topic = payload.get("topic", "")
        status = "resolved" if payload.get("consensus", False) else "failed"
        utterances = payload.get("utterances", [])
        utterance_edges = payload.get("utterance_edges", [])
        insight_text = payload.get("insight", "")

        now = time.time()

        with self._db:
            # --- Tier 1: Store interaction graph ---
            for utt in utterances:
                self._db.execute(
                    "INSERT OR IGNORE INTO utterances VALUES (?,?,?,?,?,?)",
                    (
                        utt.get("id", str(uuid.uuid4())),
                        query_id,
                        utt.get("agent_id", ""),
                        utt.get("content", ""),
                        utt.get("epoch", 0),
                        utt.get("timestamp", now),
                    ),
                )

            for src, tgt in utterance_edges:
                self._db.execute(
                    "INSERT OR IGNORE INTO utterance_edges VALUES (?,?,?)",
                    (src, tgt, query_id),
                )

            # --- Tier 2: Create query node + edges ---
            self._db.execute(
                "INSERT OR IGNORE INTO queries VALUES (?,?,?,?,0,?)",
                (query_id, topic, status, now, now),
            )

            # Link to semantically similar historical queries
            related_ids = self._find_related_queries(topic, query_id)
            for related_id in related_ids:
                self._db.execute(
                    "INSERT OR IGNORE INTO query_edges VALUES (?,?,?)",
                    (related_id, query_id, now),
                )

            # --- Tier 3: Generate and store insight ---
            if insight_text:
                self._store_insight(insight_text, query_id, now)

        # --- Update LanceDB vector index ---
        self._store_query_vector(query_id, topic, status, now)

        print(
            f"[MemoryWriter] Stored task: {topic[:40]} | "
            f"status={status} | utterances={len(utterances)} | "
            f"related_queries={len(related_ids)}"
        )

    # ------------------------------------------------------------------
    # Query similarity
    # ------------------------------------------------------------------

    def _find_related_queries(self, query_text: str, exclude_id: str) -> list[str]:
        """Find queries semantically similar to the new query.

        Returns IDs of queries with embedding similarity > QUERY_LINK_THRESHOLD.
        """
        if self._query_table is None or self._embedder is None:
            return []

        try:
            results = (
                self._query_table.search(query_text)
                .limit(5)
                .to_pandas()
            )

            related = []
            for _, row in results.iterrows():
                qid = row.get("query_id", "")
                dist = row.get("_distance", 1.0)
                # LanceDB returns L2 distance by default; convert to similarity
                similarity = max(0, 1 - dist)
                if qid != exclude_id and similarity > QUERY_LINK_THRESHOLD:
                    related.append(qid)

            return related

        except Exception:
            return []

    def _store_query_vector(self, query_id: str, text: str, status: str, created_at: float):
        """Store a query embedding in LanceDB."""
        if self._lance_db is None or self._embedder is None:
            return

        try:
            embedding = self._embedder.encode(text)
            data = [{
                "query_id": query_id,
                "text": text,
                "vector": embedding,
                "status": status,
                "created_at": created_at,
            }]

            if self._query_table is None:
                self._query_table = self._lance_db.create_table("queries", data)
            else:
                self._query_table.add(data)

        except Exception as e:
            print(f"[MemoryWriter] LanceDB write error: {e}")

    # ------------------------------------------------------------------
    # Insight management
    # ------------------------------------------------------------------

    def _store_insight(self, insight_text: str, query_id: str, now: float):
        """Store a new insight, deduplicating against existing insights.

        If an existing insight has embedding similarity > INSIGHT_DEDUP_THRESHOLD,
        merge by adding this query to its supporting_queries set.
        Otherwise, create a new insight node.
        """
        if self._embedder is None:
            # Can't deduplicate without embeddings — just insert
            self._insert_new_insight(insight_text, query_id, now)
            return

        new_embedding = self._embedder.encode(insight_text)

        # Check for duplicate insights
        existing = self._db.execute(
            "SELECT id, content FROM insights"
        ).fetchall()

        for existing_id, existing_content in existing:
            existing_embedding = self._embedder.encode(existing_content)
            similarity = cosine_similarity(new_embedding, existing_embedding)

            if similarity > INSIGHT_DEDUP_THRESHOLD:
                # Merge: add query to existing insight's supporting set
                self._merge_insight(existing_id, query_id)
                return

        # No duplicate found — create new insight
        self._insert_new_insight(insight_text, query_id, now)

    def _insert_new_insight(self, content: str, query_id: str, now: float):
        """Insert a new insight node."""
        insight_id = str(uuid.uuid4())
        supporting = json.dumps([query_id])
        self._db.execute(
            "INSERT INTO insights VALUES (?,?,?,?,0)",
            (insight_id, content, supporting, now),
        )

    def _merge_insight(self, insight_id: str, query_id: str):
        """Merge a query into an existing insight's supporting set."""
        row = self._db.execute(
            "SELECT supporting_queries FROM insights WHERE id=?",
            (insight_id,),
        ).fetchone()

        if row:
            supporting = json.loads(row[0])
            if query_id not in supporting:
                supporting.append(query_id)
                self._db.execute(
                    "UPDATE insights SET supporting_queries=?, access_count=access_count+1 WHERE id=?",
                    (json.dumps(supporting), insight_id),
                )

    # ------------------------------------------------------------------
    # LLM-powered insight generation
    # ------------------------------------------------------------------

    def generate_insight_llm(self, topic: str, utterances: list[dict], status: str) -> str:
        """Generate an insight from a completed task using an LLM.

        This is the G-Memory J(interaction_graph, status) function.
        Returns a one-sentence generalizable lesson, or empty string on failure.
        """
        import subprocess

        # Build a compact trace
        trace_lines = []
        for u in utterances[:15]:
            agent = u.get("agent_id", "?")
            content = u.get("content", "")[:200]
            trace_lines.append(f"[{agent}]: {content}")
        trace = "\n".join(trace_lines)

        prompt = (
            f"A multi-agent swarm just completed a research task.\n\n"
            f"Topic: {topic}\n"
            f"Outcome: {status}\n\n"
            f"Agent interaction trace:\n{trace}\n\n"
            f"Generate ONE concise, generalizable insight or lesson learned from "
            f"this task. The insight should be useful for future tasks on similar "
            f"topics. Focus on what worked, what didn't, or a key finding."
        )

        try:
            result = subprocess.run(
                [
                    "claude", "-p", prompt,
                    "--system-prompt",
                    "Return JSON with a single 'insight' field containing one sentence.",
                    "--output-format", "json",
                    "--model", "haiku",
                    "--dangerously-skip-permissions",
                ],
                capture_output=True, text=True, timeout=20,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                return data.get("insight", "")
        except Exception:
            pass

        return ""

    # ------------------------------------------------------------------
    # Direct write API (for use without Redis)
    # ------------------------------------------------------------------

    def store_task(
        self,
        topic: str,
        status: str,
        utterances: list[dict],
        utterance_edges: list[tuple[str, str]],
        insight: str = "",
    ) -> str:
        """Directly store a completed task. Returns the query ID.

        Use this when Redis is not available (--no-redis mode).
        """
        query_id = str(uuid.uuid4())
        self.process_write({
            "type": "task_complete",
            "query_id": query_id,
            "topic": topic,
            "consensus": status == "resolved",
            "utterances": utterances,
            "utterance_edges": utterance_edges,
            "insight": insight,
        })
        return query_id

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def run_eviction_cycle(
        self,
        archive_days: int = 90,
        purge_days: int = 30,
        max_archive: int = 50,
    ):
        """Run a memory eviction cycle.

        - Archive queries not accessed in archive_days with low access count
        - Purge old interaction details older than purge_days
        - Never evict insights (highest-value, most-compressed tier)
        """
        if self._db is None:
            return

        now = time.time()
        archive_cutoff = now - (archive_days * 86400)
        purge_cutoff = now - (purge_days * 86400)

        archived = 0
        purged_utterances = 0

        with self._db:
            # Archive old, unaccessed queries
            old_queries = self._db.execute(
                """
                SELECT id FROM queries
                WHERE last_accessed < ? AND access_count < 3
                ORDER BY last_accessed ASC LIMIT ?
                """,
                (archive_cutoff, max_archive),
            ).fetchall()

            for (qid,) in old_queries:
                # Copy to archive
                self._db.execute(
                    """
                    INSERT OR IGNORE INTO queries_archive
                    SELECT *, ? FROM queries WHERE id = ?
                    """,
                    (now, qid),
                )
                self._db.execute("DELETE FROM queries WHERE id = ?", (qid,))
                archived += 1

            # Purge old interaction details (keep top-100 most accessed)
            result = self._db.execute(
                """
                DELETE FROM utterances
                WHERE query_id IN (
                    SELECT id FROM queries WHERE created_at < ?
                ) AND query_id NOT IN (
                    SELECT id FROM queries ORDER BY access_count DESC LIMIT 100
                )
                """,
                (purge_cutoff,),
            )
            purged_utterances = result.rowcount

            # Clean orphaned utterance edges
            self._db.execute(
                """
                DELETE FROM utterance_edges
                WHERE query_id NOT IN (SELECT DISTINCT query_id FROM utterances)
                """
            )

            # Clean orphaned query edges
            self._db.execute(
                """
                DELETE FROM query_edges
                WHERE source_id NOT IN (SELECT id FROM queries)
                   OR target_id NOT IN (SELECT id FROM queries)
                """
            )

        if archived > 0 or purged_utterances > 0:
            print(
                f"[MemoryWriter] Eviction: archived {archived} queries, "
                f"purged {purged_utterances} utterances"
            )

    def maybe_run_eviction(self, cycle_frequency: int = 10):
        """Run eviction if we've processed enough writes."""
        if self._write_count > 0 and self._write_count % cycle_frequency == 0:
            self.run_eviction_cycle()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        counts = {}
        if self._db:
            for table in ("queries", "utterances", "insights"):
                row = self._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = row[0] if row else 0
        return {
            "write_count": self._write_count,
            "running": self._running,
            "tables": counts,
        }
