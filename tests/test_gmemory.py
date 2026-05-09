"""Tests for gmemory/* — Tier 2/3 build-out."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gmemory.hybrid import RRF_K, fts5_available, rrf_fuse
from gmemory.insight_graph import InsightCandidate, InsightGraph, heuristic_distill
from gmemory.meta import (
    aggregate_by_tag,
    correlate_pattern_with_outcome,
    search_insights,
    search_tasks,
)
from gmemory.promotion import (
    CLUSTER_COSINE_THRESH,
    PROMOTION_N,
    cosine,
    evaluate_candidates,
    prune_stale_candidates,
    _encode_blob,
)
from gmemory.query_graph import QueryGraph, initialize_schema
from gmemory.retrieval import (
    LLM_OPS_THRESHOLD,
    RetrievalResult,
    format_context,
    heuristic_relevance,
    retrieve,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """In-memory SQLite with full Tier 2/3 schema applied."""
    conn = sqlite3.connect(":memory:")
    initialize_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def qg(db):
    return QueryGraph(db)


@pytest.fixture
def ig(db):
    return InsightGraph(db)


# ---------------------------------------------------------------------------
# hybrid.py
# ---------------------------------------------------------------------------


def test_rrf_fuse_combines_ranks():
    vec_hits = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    bm25_hits = [("c", 5.0), ("a", 3.0), ("d", 1.0)]
    fused = rrf_fuse(vec_hits, bm25_hits)
    fused_ids = [qid for qid, _ in fused]
    # 'a' is rank 0 in vector, rank 1 in bm25 — should top
    assert fused_ids[0] == "a"
    # 'd' only appears in bm25 at rank 2 — last
    assert fused_ids[-1] == "d"


def test_rrf_fuse_empty_inputs():
    assert rrf_fuse([], []) == []
    assert len(rrf_fuse([("a", 1.0)], [])) == 1


def test_fts5_available_in_memory(db):
    """Default sqlite3 ships with FTS5 in modern Python."""
    assert fts5_available(db) in (True, False)  # not raising is the assertion


# ---------------------------------------------------------------------------
# query_graph.py
# ---------------------------------------------------------------------------


def test_initialize_schema_idempotent(db):
    """Running schema twice must not error."""
    initialize_schema(db)


def test_add_task_creates_row(qg, db):
    qid = qg.add_task("Test topic", "resolved", n_agents=5, quality_score=8.0)
    row = db.execute("SELECT id, status FROM queries WHERE id = ?", (qid,)).fetchone()
    assert row[0] == qid
    assert row[1] == "resolved"


def test_add_task_idempotent_on_query_id(qg, db):
    qg.add_task("topic", "resolved", query_id="fixed-id")
    qg.add_task("topic", "resolved", query_id="fixed-id")
    n = db.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
    assert n == 1


def test_add_task_materializes_tags(qg, db):
    qg.add_task("topic", "resolved", tags=["domain:crypto", "pattern:tsmom"])
    rows = db.execute("SELECT axis, value FROM query_tags").fetchall()
    assert ("domain", "crypto") in rows
    assert ("pattern", "tsmom") in rows


def test_add_task_with_neighbors_creates_edges(qg, db):
    qg.add_task("Bitcoin SegWit BIP141 throughput", "resolved", query_id="q1")
    qg.add_task("Bitcoin SegWit BIP141 throughput", "resolved", query_id="q2")
    edges = db.execute(
        "SELECT source_id, target_id FROM query_edges"
    ).fetchall()
    # Identical text → cosine similarity 1.0 → edge formed
    assert any(("q1", "q2") == e or ("q2", "q1") == e for e in edges)


def test_query_count(qg, db):
    qg.add_task("a", "resolved")
    qg.add_task("b", "resolved")
    assert qg.query_count() == 2


def test_one_hop_returns_neighbors(qg, db):
    qg.add_task("a", "resolved", query_id="a")
    qg.add_task("b", "resolved", query_id="b")
    qg.link("a", "b", 0.8)
    expanded = qg.one_hop(["a"])
    assert "b" in expanded


def test_one_hop_excludes_seeds_themselves(qg, db):
    qg.add_task("a", "resolved", query_id="a")
    qg.add_task("b", "resolved", query_id="b")
    qg.link("a", "b", 0.8)
    expanded = qg.one_hop(["a"])
    assert "a" not in expanded


def test_get_records_returns_query_records(qg):
    qg.add_task("topic", "resolved", query_id="q1", n_agents=5,
                quality_score=8.0, tags=["domain:crypto"])
    records = qg.get_records(["q1"])
    assert len(records) == 1
    assert records[0].id == "q1"
    assert records[0].quality_score == 8.0
    assert "domain:crypto" in records[0].tags


def test_stats(qg):
    qg.add_task("a", "resolved", query_id="a")
    qg.add_task("b", "resolved", query_id="b")
    qg.link("a", "b", 0.8)
    stats = qg.stats()
    assert stats["n_queries"] == 2
    assert stats["n_edges"] == 1


# ---------------------------------------------------------------------------
# promotion.py
# ---------------------------------------------------------------------------


def test_promotion_below_threshold_no_promote(db, qg, ig):
    qg.add_task("a", "resolved", query_id="qa")
    qg.add_task("b", "resolved", query_id="qb")
    # Two candidates from two queries — needs N=3
    ig.extract_insights("qa", "topic", [
        {"key_points": ["walk-forward CV beats fixed split"]},
        {"key_points": ["walk-forward CV beats fixed split"]},
        {"key_points": ["walk-forward CV beats fixed split"]},
    ], "resolved")
    ig.extract_insights("qb", "topic", [
        {"key_points": ["walk-forward CV beats fixed split"]},
        {"key_points": ["walk-forward CV beats fixed split"]},
        {"key_points": ["walk-forward CV beats fixed split"]},
    ], "resolved")
    promoted = evaluate_candidates(db)
    assert promoted == []


def test_promotion_at_threshold_promotes(db, qg, ig):
    """3 distinct queries, all with the same insight → promotes."""
    insight_text = "walk-forward CV beats fixed split"
    for qid in ["qa", "qb", "qc"]:
        qg.add_task(qid, "resolved", query_id=qid)
        ig.extract_insights(qid, "topic", [
            {"key_points": [insight_text]},
            {"key_points": [insight_text]},
            {"key_points": [insight_text]},
        ], "resolved")
    promoted = evaluate_candidates(db)
    assert len(promoted) == 1
    rows = db.execute(
        "SELECT supporting_queries FROM insights WHERE id = ?", promoted,
    ).fetchall()
    assert len(rows) == 1
    supporting = json.loads(rows[0][0])
    assert set(supporting) == {"qa", "qb", "qc"}


def test_promotion_three_in_one_query_does_not_promote(db, qg, ig):
    """Three insights all from same query_id → only 1 distinct support."""
    qg.add_task("only", "resolved", query_id="only")
    # Multiple distinct candidates from one query
    ig.extract_insights("only", "topic", [
        {"key_points": ["lesson A"]},
        {"key_points": ["lesson B"]},
        {"key_points": ["lesson C"]},
    ], "resolved")
    promoted = evaluate_candidates(db)
    assert promoted == []


def test_prune_stale_candidates_removes_old(db, qg, ig):
    qg.add_task("q", "resolved", query_id="q")
    ig.extract_insights("q", "topic", [
        {"key_points": ["lesson"]},
        {"key_points": ["lesson"]},
        {"key_points": ["lesson"]},
    ], "resolved")
    # Backdate created_at to 30 days ago
    db.execute(
        "UPDATE insight_candidates SET created_at = ?",
        (time.time() - 30 * 86400,),
    )
    db.commit()
    pruned = prune_stale_candidates(db, ttl_days=14)
    assert pruned >= 1


# ---------------------------------------------------------------------------
# insight_graph.py
# ---------------------------------------------------------------------------


def test_extract_insights_writes_candidates(db, qg, ig):
    qg.add_task("q1", "resolved", query_id="q1")
    cand_ids = ig.extract_insights("q1", "topic", [
        {"key_points": ["lesson 1", "lesson 2"]},
        {"key_points": ["lesson 3"]},
        {"key_points": []},
    ], "resolved")
    assert len(cand_ids) >= 1


def test_extract_insights_idempotent(db, qg, ig):
    qg.add_task("q1", "resolved", query_id="q1")
    utterances = [
        {"key_points": ["unique lesson"]},
        {"key_points": ["unique lesson"]},
        {"key_points": ["unique lesson"]},
    ]
    ig.extract_insights("q1", "topic", utterances, "resolved")
    ig.extract_insights("q1", "topic", utterances, "resolved")
    n = db.execute(
        "SELECT COUNT(*) FROM insight_candidates WHERE query_id = ?", ("q1",),
    ).fetchone()[0]
    # Dedup at content level — second call adds 0 new candidates
    assert n == 1


def test_retrieve_insights_for_query_overlaps(db, qg, ig):
    """Promoted insight with Ω=[q1, q2, q3] retrievable by any of them."""
    insight_text = "lesson"
    for qid in ["q1", "q2", "q3"]:
        qg.add_task(qid, "resolved", query_id=qid)
        ig.extract_insights(qid, "t", [
            {"key_points": [insight_text]},
            {"key_points": [insight_text]},
            {"key_points": [insight_text]},
        ], "resolved")
    evaluate_candidates(db)
    out = ig.retrieve_insights_for_query("anything", related_query_ids=["q1"])
    assert len(out) == 1


def test_aggregate_overnight_returns_counts(db, qg, ig):
    qg.add_task("q1", "resolved", query_id="q1")
    ig.extract_insights("q1", "t", [
        {"key_points": ["a"]},
    ] * 3, "resolved")
    result = ig.aggregate_overnight(since_hours=24)
    assert "promoted" in result
    assert "candidates_seen" in result


# ---------------------------------------------------------------------------
# retrieval.py
# ---------------------------------------------------------------------------


def test_retrieve_cold_start_skips_llm(db, qg, ig):
    """With < LLM_OPS_THRESHOLD queries stored, use_llm degrades to False."""
    qg.add_task("q", "resolved", query_id="q")
    out = retrieve("anything", qg, ig, use_llm=True)
    assert out.used_llm is False
    assert out.cost_usd == 0.0


def test_retrieve_warm_uses_llm(db, qg, ig):
    """With >= LLM_OPS_THRESHOLD queries, use_llm stays True."""
    for i in range(LLM_OPS_THRESHOLD + 1):
        qg.add_task(f"topic {i}", "resolved", query_id=f"q{i}")
    out = retrieve("topic 0", qg, ig, use_llm=True)
    assert out.used_llm is True


def test_format_context_empty_returns_empty(qg, ig):
    out = RetrievalResult(insights=[], interactions=[], related_queries=[],
                          used_llm=False, cost_usd=0.0)
    assert format_context(out) == ""


def test_format_context_includes_insights():
    out = RetrievalResult(
        insights=[{"content": "lesson X", "confidence": 0.8}],
        interactions=[], related_queries=[],
        used_llm=False, cost_usd=0.0,
    )
    text = format_context(out)
    assert "lesson X" in text
    assert "Insights" in text


def test_heuristic_relevance_jaccard():
    rel, cost = heuristic_relevance("crypto trading momentum",
                                     "momentum trading in crypto markets")
    assert 0 < rel < 1
    assert cost == 0.0


# ---------------------------------------------------------------------------
# meta.py
# ---------------------------------------------------------------------------


def test_search_tasks_by_tag(db, qg):
    qg.add_task("crypto query", "resolved", query_id="q1",
                tags=["domain:crypto"])
    qg.add_task("biology query", "resolved", query_id="q2",
                tags=["domain:biology"])
    out = search_tasks(db, must_have=["domain:crypto"])
    assert len(out) == 1
    assert out[0]["id"] == "q1"


def test_search_tasks_invalid_tag_raises(db):
    with pytest.raises(ValueError):
        search_tasks(db, must_have=["malformed_no_colon"])


def test_search_tasks_unknown_axis_raises(db):
    with pytest.raises(ValueError):
        search_tasks(db, must_have=["unknown_axis:value"])


def test_aggregate_by_tag_groups(db, qg):
    qg.add_task("a1", "resolved", query_id="a1",
                quality_score=8.0, tags=["domain:crypto"])
    qg.add_task("a2", "resolved", query_id="a2",
                quality_score=7.5, tags=["domain:crypto"])
    qg.add_task("b1", "resolved", query_id="b1",
                quality_score=6.0, tags=["domain:biology"])
    out = aggregate_by_tag(db, group_by="domain", metric="quality_score")
    assert "crypto" in out
    assert "biology" in out
    assert out["crypto"]["n"] == 2
    assert out["biology"]["n"] == 1


def test_correlate_pattern_with_outcome(db, qg):
    qg.add_task("with", "resolved", query_id="w1",
                quality_score=9.0, tags=["pattern:judge_ensemble"])
    qg.add_task("with", "resolved", query_id="w2",
                quality_score=8.5, tags=["pattern:judge_ensemble"])
    qg.add_task("without", "resolved", query_id="x1", quality_score=7.0)
    qg.add_task("without", "resolved", query_id="x2", quality_score=6.5)
    res = correlate_pattern_with_outcome(db, "pattern:judge_ensemble")
    assert res["n_with"] == 2
    assert res["n_without"] == 2
    assert res["delta"] > 0
