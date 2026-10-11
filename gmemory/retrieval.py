"""End-to-end memory retrieval — six-step pipeline.

    1. Embed (optional) + BM25/FTS5 -> RRF fuse -> seeds
    2. 1-hop SQL expansion
    3. Insight overlap via inverted index
    4. LLM relevance scoring (memory_ops tier)   [skipped during cold start]
    5. Tier 1 utterances + sparsification (memory_ops tier) [heuristic during cold start]
    6. Format `## Relevant prior findings` block, fenced as untrusted data

The vector side is optional: with no embedder / LanceDB the BM25 path is the
seed source and the pipeline still returns results (never empty because an
optional dependency is missing). LLM hooks are pluggable so tests run
dependency-free and model-free.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Iterable

from gmemory.hybrid import bm25_search, rrf_fuse
from gmemory.insight_graph import InsightGraph
from gmemory.query_graph import QueryGraph
from context import MEMORY_FENCE_CLOSE as FENCE_CLOSE, MEMORY_FENCE_OPEN as FENCE_OPEN, fence_memory

TOP_K_RETRIEVAL = 2
TOP_M_INTERACTIONS = 3
MAX_LLM_RELEVANCE_CALLS = 6
LLM_OPS_THRESHOLD = 10
RELEVANCE_MIN_SCORE = 0.4
MAX_UTTERANCES_PER_QUERY = 10
UTTERANCE_SNIPPET_CHARS = 300

# Prompt-injection fence around anything read back from memory. The agent
# output fence in orchestrator.build_context shares the same shape; both
# live in context.py (one helper, two variants).


# Pluggable LLM hooks. The second element of each result is the call's cost:
# 0.0 for the free heuristics, None when a backend did not report one.
RelevanceFn = Callable[[str, str, str], tuple[float, float | None]]
SparsifyFn = Callable[[list[dict], str, str], tuple[list[dict], float | None]]


def _add_cost(total: float | None, cost: float | None) -> float | None:
    """Sum known costs; once any call's cost is unknown the total is unknown."""
    if total is None or cost is None:
        return None
    return total + cost


def heuristic_relevance(new_query: str, candidate_text: str, model: str = "") -> tuple[float, float]:
    """Heuristic relevance — Jaccard token overlap. Returns (score 0..1, cost 0)."""
    a_words = set(new_query.lower().split())
    b_words = set(candidate_text.lower().split())
    if not a_words or not b_words:
        return 0.0, 0.0
    jaccard = len(a_words & b_words) / len(a_words | b_words)
    return jaccard, 0.0


def heuristic_sparsify(
    utterances: list[dict], new_query: str, model: str = "",
) -> tuple[list[dict], float]:
    """Heuristic sparsification — keep top 5 by relevance to new_query. Free."""
    scored: list[tuple[float, dict]] = []
    for u in utterances:
        text = u.get("findings", "") or u.get("content", "")
        rel, _ = heuristic_relevance(new_query, text)
        scored.append((rel, u))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [u for _, u in scored[:5]], 0.0


@dataclass(slots=True)
class RetrievalResult:
    insights: list[dict]
    interactions: list[dict]
    related_queries: list[dict]
    used_llm: bool
    cost_usd: float | None          # 0.0 = heuristics only; None = an LLM call reported no cost
    seeds: list[str] = field(default_factory=list)


def retrieve(
    new_query: str,
    qg: QueryGraph,
    ig: InsightGraph,
    *,
    top_k: int = TOP_K_RETRIEVAL,
    top_m: int = TOP_M_INTERACTIONS,
    hops: int = 1,
    use_llm: bool = True,
    relevance_fn: RelevanceFn | None = None,
    sparsify_fn: SparsifyFn | None = None,
    model: str = "",
    llm_ops_threshold: int = LLM_OPS_THRESHOLD,
    touch: bool = True,
) -> RetrievalResult:
    """Run the 6-step retrieval pipeline.

    `model` is forwarded to the LLM hooks; the orchestrator resolves it from
    the `memory_ops` role tier. Empty = heuristic hooks (no model needed).
    `touch` bumps access counters on the queries that were surfaced (the
    eviction cycle keys off them). `hops` is `[memory] hop_expansion`: 0
    disables graph expansion; anything >= 1 is one hop (G-Memory's
    measured optimum; deeper walks degrade precision).
    """
    relevance_fn = relevance_fn or heuristic_relevance
    sparsify_fn = sparsify_fn or heuristic_sparsify
    cost: float | None = 0.0

    if qg.query_count() < llm_ops_threshold:
        use_llm = False

    # Step 1 — hybrid seed retrieval (vector when available + BM25/FTS5)
    vec_hits = qg.nearest_neighbors(new_query, k=top_k, threshold=0.0)
    bm25_hits = bm25_search(qg.db, new_query, limit=top_k)
    fused = rrf_fuse(vec_hits, bm25_hits)[:top_k]
    seed_ids = [qid for qid, _ in fused]

    if not seed_ids:
        return RetrievalResult(
            insights=[], interactions=[], related_queries=[],
            used_llm=False, cost_usd=0.0, seeds=[],
        )

    # Step 2 — 1-hop expansion
    expanded = (qg.one_hop(seed_ids) if hops >= 1 else set()) | set(seed_ids)

    # Step 3 — insight overlap via inverted index
    insights = ig.retrieve_insights_for_query(
        new_query, related_query_ids=expanded, limit=5,
    )

    # Step 4 — LLM relevance scoring
    candidates = qg.get_records(expanded)
    if use_llm and candidates:
        scored: list[tuple[float, object]] = []
        for c in candidates[:MAX_LLM_RELEVANCE_CALLS]:
            rel, c_cost = relevance_fn(new_query, c.query_text, model)
            scored.append((rel, c))
            cost = _add_cost(cost, c_cost)
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for (rel, c) in scored[:top_m] if rel >= RELEVANCE_MIN_SCORE]
    else:
        top = list(candidates[:top_m])

    # Step 5 — Tier 1 utterances for the surviving queries, sparsified
    interactions: list[dict] = []
    for c in top:
        utts = _load_utterances(qg.db, c.id)
        if utts and use_llm:
            utts, s_cost = sparsify_fn(utts, new_query, model)
            cost = _add_cost(cost, s_cost)
        elif utts:
            utts, _ = heuristic_sparsify(utts, new_query)
        interactions.append({
            "query_id": c.id,
            "query_text": c.query_text,
            "status": c.status,
            "utterances": utts,
        })

    if touch:
        qg.touch(expanded)

    return RetrievalResult(
        insights=insights,
        interactions=interactions,
        related_queries=[
            {"id": c.id, "query_text": c.query_text, "status": c.status}
            for c in candidates
        ],
        used_llm=use_llm,
        cost_usd=cost,
        seeds=seed_ids,
    )


def _load_utterances(db: sqlite3.Connection, query_id: str) -> list[dict]:
    """Tier 1 rows for one query (empty when the table has none)."""
    try:
        rows = db.execute(
            "SELECT agent_id, content, epoch FROM utterances "
            "WHERE query_id = ? ORDER BY epoch, timestamp LIMIT ?",
            (query_id, MAX_UTTERANCES_PER_QUERY),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {"agent_id": r[0], "content": (r[1] or "")[:UTTERANCE_SNIPPET_CHARS], "epoch": int(r[2] or 0)}
        for r in rows
    ]


def format_context(result: RetrievalResult, *, max_chars: int = 8000, fenced: bool = True) -> str:
    """Render the `## Relevant prior findings` markdown block.

    With `fenced=True` (the default used by every prompt injection) the block
    is wrapped in FENCE_OPEN / FENCE_CLOSE so the receiving agent is told it
    is data from earlier runs, not instructions.
    """
    if not result.insights and not result.interactions:
        return ""

    lines = ["## Relevant prior findings", ""]
    if result.insights:
        lines.append("### Insights from prior tasks")
        for ins in result.insights[:5]:
            confidence = ins.get("confidence", 0)
            lines.append(f"- {ins['content']} *(confidence: {confidence:.0%})*")
        lines.append("")
    if result.interactions:
        lines.append("### Related prior queries")
        for inter in result.interactions[:5]:
            lines.append(f"- **{inter['query_text']}** ({inter['status']})")
            for u in (inter.get("utterances") or [])[:3]:
                snippet = " ".join((u.get("content") or "").split())[:150]
                if snippet:
                    lines.append(f"  - [{u.get('agent_id', '?')}] {snippet}")
        lines.append("")
    text = "\n".join(lines)
    budget = max_chars - (len(FENCE_OPEN) + len(FENCE_CLOSE) + 2 if fenced else 0)
    if len(text) > budget:
        text = text[: budget - 100] + "\n\n[truncated to fit token budget]"
    if fenced:
        text = fence_memory(text)
    return text
