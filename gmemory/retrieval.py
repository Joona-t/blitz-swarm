"""End-to-end memory retrieval — six-step pipeline.

    1. Embed + BM25 -> RRF fuse -> seeds
    2. 1-hop SQL expansion
    3. Insight overlap via inverted index
    4. LLM relevance scoring (Haiku)   [skipped during cold start]
    5. LLM sparsification (Haiku)      [skipped during cold start]
    6. Format `## Relevant prior findings` block

LLM hooks are pluggable so tests run scipy-free.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Iterable

from gmemory.hybrid import bm25_search, rrf_fuse
from gmemory.insight_graph import InsightGraph
from gmemory.query_graph import QueryGraph

TOP_K_RETRIEVAL = 2
TOP_M_INTERACTIONS = 3
MAX_LLM_RELEVANCE_CALLS = 6
LLM_OPS_THRESHOLD = 10
RELEVANCE_MIN_SCORE = 0.4


# Pluggable LLM hooks
RelevanceFn = Callable[[str, str, str], tuple[float, float]]
SparsifyFn = Callable[[list[dict], str, str], tuple[list[dict], float]]


def heuristic_relevance(new_query: str, candidate_text: str, model: str = "haiku") -> tuple[float, float]:
    """Heuristic relevance — Jaccard token overlap. Returns (score 0..1, cost 0)."""
    a_words = set(new_query.lower().split())
    b_words = set(candidate_text.lower().split())
    if not a_words or not b_words:
        return 0.0, 0.0
    jaccard = len(a_words & b_words) / len(a_words | b_words)
    return jaccard, 0.0


def heuristic_sparsify(
    utterances: list[dict], new_query: str, model: str = "haiku",
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
    cost_usd: float
    seeds: list[str] = field(default_factory=list)


def retrieve(
    new_query: str,
    qg: QueryGraph,
    ig: InsightGraph,
    *,
    top_k: int = TOP_K_RETRIEVAL,
    top_m: int = TOP_M_INTERACTIONS,
    use_llm: bool = True,
    relevance_fn: RelevanceFn | None = None,
    sparsify_fn: SparsifyFn | None = None,
) -> RetrievalResult:
    """Run the 6-step retrieval pipeline."""
    relevance_fn = relevance_fn or heuristic_relevance
    sparsify_fn = sparsify_fn or heuristic_sparsify
    cost = 0.0

    if qg.query_count() < LLM_OPS_THRESHOLD:
        use_llm = False

    # Step 1 — hybrid seed retrieval (vector + BM25)
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
    expanded = qg.one_hop(seed_ids) | set(seed_ids)

    # Step 3 — insight overlap via inverted index
    insights = ig.retrieve_insights_for_query(
        new_query, related_query_ids=expanded, limit=5,
    )

    # Step 4 — LLM relevance scoring
    candidates = qg.get_records(expanded)
    if use_llm and candidates:
        scored: list[tuple[float, object]] = []
        for c in candidates[:MAX_LLM_RELEVANCE_CALLS]:
            rel, c_cost = relevance_fn(new_query, c.query_text, "haiku")
            scored.append((rel, c))
            cost += c_cost
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for (rel, c) in scored[:top_m] if rel >= RELEVANCE_MIN_SCORE]
    else:
        top = list(candidates[:top_m])

    # Step 5 — sparsification (defer to caller; legacy memory/reader.py
    # supplies utterances if available)
    interactions: list[dict] = []
    for c in top:
        interactions.append({
            "query_id": c.id,
            "query_text": c.query_text,
            "status": c.status,
            "utterances": [],          # populated by orchestrator integration
        })

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


def format_context(result: RetrievalResult, *, max_chars: int = 8000) -> str:
    """Render the `## Relevant prior findings` markdown block."""
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
        lines.append("")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 100] + "\n\n[truncated to fit token budget]"
    return text
