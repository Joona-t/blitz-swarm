"""MemoryReader — retrieval pipeline for G-Memory hierarchical memory.

Implements the four-step retrieval from G-Memory (NeurIPS 2025):
1. Embedding similarity search (top-k=2)
2. 1-hop graph expansion
3. Upward traversal to insights
4. Downward traversal to interactions

LLM relevance scoring (R_LLM) and sparsification (S_LLM) added in Phase 4.
"""

import json
import sqlite3
from pathlib import Path

from embedder import Embedder, get_embedder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = Path(__file__).parent.parent / "memory.db"
DEFAULT_LANCE_PATH = Path(__file__).parent.parent / "memory_vectors"
TOP_K = 2  # k=2 per research (k=5 degrades by 7.71%)
MAX_CONTEXT_TOKENS = 2000
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * 4


class MemoryReader:
    """Reads from G-Memory storage to provide historical context for agents."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        lance_path: Path = DEFAULT_LANCE_PATH,
    ):
        self.db_path = db_path
        self.lance_path = lance_path
        self._db = None
        self._lance_db = None
        self._query_table = None
        self._embedder = None

    def initialize(self):
        """Open read-only connections to storage backends."""
        self._embedder = get_embedder()

        if self.db_path.exists():
            self._db = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.row_factory = sqlite3.Row

        try:
            import lancedb
            self._lance_db = lancedb.connect(str(self.lance_path))
            try:
                self._query_table = self._lance_db.open_table("queries")
            except Exception:
                self._query_table = None
        except ImportError:
            pass

    def is_available(self) -> bool:
        """Check if memory storage exists and has data."""
        if self._db is None:
            return False
        row = self._db.execute("SELECT COUNT(*) FROM queries").fetchone()
        return row[0] > 0 if row else False

    def query_count(self) -> int:
        """Return the number of stored queries."""
        if self._db is None:
            return 0
        row = self._db.execute("SELECT COUNT(*) FROM queries").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Main retrieval pipeline
    # ------------------------------------------------------------------

    def retrieve_memory(self, query_text: str) -> dict:
        """Execute the full G-Memory retrieval pipeline.

        Returns:
            {
                "insights": [{"id": ..., "content": ..., "access_count": ...}, ...],
                "interactions": [{"query_id": ..., "utterances": [...], ...}, ...],
                "related_queries": [{"id": ..., "query_text": ..., "status": ...}, ...],
            }

        Returns empty dict structure if no memory is available.
        """
        empty = {"insights": [], "interactions": [], "related_queries": []}

        if not self.is_available():
            return empty

        # Step 1: Embedding similarity search — top-k
        seed_ids = self._embedding_search(query_text, top_k=TOP_K)
        if not seed_ids:
            return empty

        # Step 2: 1-hop expansion in query graph
        expanded_ids = self._one_hop_expand(seed_ids)

        # Step 3: Upward traversal — find relevant insights
        insights = self._get_insights_for_queries(expanded_ids)

        # Step 4: Downward traversal — get interaction fragments
        interactions = self._get_interactions_for_queries(expanded_ids)

        # Get query metadata for context
        related_queries = self._get_query_details(expanded_ids)

        # Update access counts
        self._update_access_counts(expanded_ids)

        return {
            "insights": insights,
            "interactions": interactions,
            "related_queries": related_queries,
        }

    # ------------------------------------------------------------------
    # Step 1: Embedding search
    # ------------------------------------------------------------------

    def _embedding_search(self, query_text: str, top_k: int = TOP_K) -> set[str]:
        """Find the top-k most similar historical queries by embedding."""
        if self._query_table is None or self._embedder is None:
            return set()

        try:
            results = (
                self._query_table.search(query_text)
                .limit(top_k)
                .to_pandas()
            )
            return {row["query_id"] for _, row in results.iterrows()}

        except Exception:
            return set()

    # ------------------------------------------------------------------
    # Step 2: 1-hop expansion
    # ------------------------------------------------------------------

    def _one_hop_expand(self, seed_ids: set[str]) -> set[str]:
        """Expand seed query set by 1 hop in the query graph.

        1-hop is optimal per research — 2+ hops degrade performance.
        """
        if not seed_ids or self._db is None:
            return seed_ids

        expanded = set(seed_ids)
        placeholders = ",".join("?" * len(seed_ids))
        seed_list = list(seed_ids)

        # Get both directions of edges
        rows = self._db.execute(
            f"""
            SELECT DISTINCT neighbor_id FROM (
                SELECT target_id AS neighbor_id FROM query_edges
                WHERE source_id IN ({placeholders})
                UNION
                SELECT source_id AS neighbor_id FROM query_edges
                WHERE target_id IN ({placeholders})
            )
            """,
            seed_list + seed_list,
        ).fetchall()

        for row in rows:
            expanded.add(row[0])

        return expanded

    # ------------------------------------------------------------------
    # Step 3: Upward — insights
    # ------------------------------------------------------------------

    def _get_insights_for_queries(self, query_ids: set[str]) -> list[dict]:
        """Find insights whose supporting_queries overlap with query_ids."""
        if not query_ids or self._db is None:
            return []

        rows = self._db.execute(
            "SELECT id, content, supporting_queries, access_count FROM insights"
        ).fetchall()

        relevant = []
        for row in rows:
            supporting = set(json.loads(row["supporting_queries"]))
            if supporting & query_ids:
                relevant.append({
                    "id": row["id"],
                    "content": row["content"],
                    "access_count": row["access_count"],
                    "supporting_count": len(supporting & query_ids),
                })

        # Sort by overlap count (most supported first)
        relevant.sort(key=lambda x: x["supporting_count"], reverse=True)
        return relevant

    # ------------------------------------------------------------------
    # Step 4: Downward — interactions
    # ------------------------------------------------------------------

    def _get_interactions_for_queries(self, query_ids: set[str]) -> list[dict]:
        """Get interaction graph fragments for the expanded query set."""
        if not query_ids or self._db is None:
            return []

        interactions = []
        for qid in query_ids:
            utterances = self._db.execute(
                "SELECT agent_id, content, epoch FROM utterances "
                "WHERE query_id = ? ORDER BY epoch, timestamp",
                (qid,),
            ).fetchall()

            if utterances:
                interactions.append({
                    "query_id": qid,
                    "utterances": [
                        {
                            "agent_id": u["agent_id"],
                            "content": u["content"][:300],
                            "epoch": u["epoch"],
                        }
                        for u in utterances[:10]  # cap at 10 per query
                    ],
                })

        return interactions

    # ------------------------------------------------------------------
    # Query details
    # ------------------------------------------------------------------

    def _get_query_details(self, query_ids: set[str]) -> list[dict]:
        """Get metadata for a set of query IDs."""
        if not query_ids or self._db is None:
            return []

        placeholders = ",".join("?" * len(query_ids))
        rows = self._db.execute(
            f"SELECT id, query_text, status FROM queries WHERE id IN ({placeholders})",
            list(query_ids),
        ).fetchall()

        return [
            {"id": r["id"], "query_text": r["query_text"], "status": r["status"]}
            for r in rows
        ]

    def _update_access_counts(self, query_ids: set[str]):
        """Increment access counts for retrieved queries and their insights."""
        if not query_ids or self._db is None:
            return

        import time
        now = time.time()
        placeholders = ",".join("?" * len(query_ids))
        qid_list = list(query_ids)

        with self._db:
            self._db.execute(
                f"UPDATE queries SET access_count = access_count + 1, "
                f"last_accessed = ? WHERE id IN ({placeholders})",
                [now] + qid_list,
            )

    # ------------------------------------------------------------------
    # LLM-powered memory operations (activate after >10 queries)
    # ------------------------------------------------------------------

    def score_relevance(self, new_query: str, candidate_text: str) -> float:
        """Use LLM to score relevance of a candidate query to the new query.

        Returns 0.0-1.0 relevance score. Only used when memory has >10 queries.
        Falls back to 0.5 on failure.
        """
        import subprocess

        prompt = (
            f"Rate how relevant this historical task is to the new query.\n\n"
            f"New query: {new_query}\n"
            f"Historical task: {candidate_text}\n\n"
            f"Return a JSON object with a single 'relevance' field (0.0-1.0)."
        )

        try:
            result = subprocess.run(
                [
                    "claude", "-p", prompt,
                    "--system-prompt", "Return JSON only. Score relevance 0.0-1.0.",
                    "--output-format", "json",
                    "--model", "haiku",
                    "--dangerously-skip-permissions",
                ],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout.strip())
                return float(data.get("relevance", 0.5))
        except Exception:
            pass

        return 0.5

    def sparsify_interaction(self, utterances: list[dict], query: str) -> list[dict]:
        """Use LLM to extract essential steps from an interaction trace.

        Returns a compressed list of the most important utterances.
        Only used when memory has >10 queries.
        Falls back to returning first 5 utterances.
        """
        if len(utterances) <= 5:
            return utterances

        import subprocess
        import json

        trace = "\n".join(
            f"[{u['agent_id']}]: {u['content'][:200]}"
            for u in utterances
        )

        prompt = (
            f"Given this agent interaction trace and a new query, extract only the "
            f"essential steps that would be relevant.\n\n"
            f"New query: {query}\n\n"
            f"Trace:\n{trace}\n\n"
            f"Return JSON with a 'relevant_indices' array of 0-based indices of "
            f"the most relevant utterances (max 5)."
        )

        try:
            result = subprocess.run(
                [
                    "claude", "-p", prompt,
                    "--system-prompt", "Return JSON only.",
                    "--output-format", "json",
                    "--model", "haiku",
                    "--dangerously-skip-permissions",
                ],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                indices = data.get("relevant_indices", [])
                return [utterances[i] for i in indices if 0 <= i < len(utterances)]
        except Exception:
            pass

        return utterances[:5]

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def build_memory_context(
        self,
        memory: dict,
        for_role: str = "",
        max_chars: int = MAX_CONTEXT_CHARS,
    ) -> str:
        """Format retrieved memory as a context string for agent prompts.

        Returns empty string if no relevant memory was found.
        """
        insights = memory.get("insights", [])
        interactions = memory.get("interactions", [])
        related = memory.get("related_queries", [])

        if not insights and not interactions:
            return ""

        sections = ["## Historical Memory (from prior tasks)"]

        if insights:
            sections.append("\n### Relevant Insights")
            for ins in insights[:5]:
                sections.append(f"- {ins['content']}")

        if related:
            sections.append("\n### Related Past Queries")
            for q in related[:3]:
                sections.append(f"- [{q['status']}] {q['query_text']}")

        if interactions:
            sections.append("\n### Relevant Past Interactions")
            for inter in interactions[:3]:
                sections.append(f"\n**Query:** (related task)")
                for u in inter["utterances"][:5]:
                    sections.append(f"  [{u['agent_id']}]: {u['content'][:150]}")

        context = "\n".join(sections)

        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[Memory context truncated]"

        return context

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close database connections."""
        if self._db:
            self._db.close()
