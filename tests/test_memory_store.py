"""WP2 — memory actually runs: MemoryStore round-trip on stdlib sqlite3 + FTS5.

No optional dependency (no sentence-transformers, no lancedb), no model
call: the memory_ops path is exercised with a fake backend injected at the
`make_provider_backend` boundary. Every test runs on a temp database.
"""

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

from gmemory.hybrid import bm25_search, fts_match_query
from gmemory.migrate import is_legacy, migrate_db, migrate_path
from gmemory.promotion import PROMOTION_N
from gmemory.retrieval import FENCE_CLOSE, FENCE_OPEN, format_context
from gmemory.store import MemorySettings, MemoryStore, build_trace

LEGACY_SCHEMA = """
CREATE TABLE insights (id TEXT PRIMARY KEY, content TEXT NOT NULL,
    supporting_queries TEXT NOT NULL DEFAULT '[]', created_at REAL NOT NULL, access_count INTEGER DEFAULT 0);
CREATE TABLE insight_edges (source_insight_id TEXT NOT NULL REFERENCES insights(id),
    target_insight_id TEXT NOT NULL REFERENCES insights(id), via_query_id TEXT NOT NULL,
    created_at REAL NOT NULL, PRIMARY KEY (source_insight_id, target_insight_id, via_query_id));
CREATE TABLE queries (id TEXT PRIMARY KEY, query_text TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('resolved', 'failed')), created_at REAL NOT NULL,
    access_count INTEGER DEFAULT 0, last_accessed REAL NOT NULL);
CREATE TABLE query_edges (source_id TEXT NOT NULL REFERENCES queries(id),
    target_id TEXT NOT NULL REFERENCES queries(id), created_at REAL NOT NULL, PRIMARY KEY (source_id, target_id));
CREATE TABLE utterances (id TEXT PRIMARY KEY, query_id TEXT NOT NULL REFERENCES queries(id),
    agent_id TEXT NOT NULL, content TEXT NOT NULL, epoch INTEGER NOT NULL, timestamp REAL NOT NULL);
CREATE TABLE utterance_edges (source_id TEXT NOT NULL REFERENCES utterances(id),
    target_id TEXT NOT NULL REFERENCES utterances(id), query_id TEXT NOT NULL REFERENCES queries(id),
    PRIMARY KEY (source_id, target_id));
CREATE TABLE queries_archive (id TEXT PRIMARY KEY, query_text TEXT NOT NULL, status TEXT NOT NULL,
    created_at REAL NOT NULL, access_count INTEGER DEFAULT 0, last_accessed REAL NOT NULL, archived_at REAL NOT NULL);
CREATE INDEX idx_utterances_query ON utterances(query_id);
CREATE INDEX idx_queries_status ON queries(status);
"""


def _rounds(topic: str, lessons: list[str]) -> list[list[dict]]:
    """Two researcher outputs + one synthesizer, like a real consensus run."""
    return [
        [
            {"agent_id": "researcher_1", "role": "researcher",
             "findings": f"Findings on {topic}: first pass.", "key_points": lessons[:1], "confidence": 0.8},
            {"agent_id": "researcher_2", "role": "researcher",
             "findings": f"Findings on {topic}: second pass.", "key_points": lessons[1:2], "confidence": 0.7},
        ],
        [
            {"agent_id": "synthesizer", "role": "synthesizer",
             "findings": f"Synthesis on {topic}.", "key_points": lessons, "confidence": 0.9},
        ],
    ]


@pytest.fixture
def store(tmp_path):
    s = MemoryStore.open(tmp_path / "mem.db", embedder=None, llm=None)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# FTS5 escaping (memory-08)
# ---------------------------------------------------------------------------


def test_fts_match_query_quotes_every_token():
    q = fts_match_query("what's the (fastest) way? -- SQLite \"WAL\" mode: NOT AND")
    assert q.startswith('"what" OR "s" OR "the"')
    assert '"fastest"' in q and '"WAL"' in q and '"NOT"' in q and '"AND"' in q
    assert "(" not in q and "?" not in q and "-" not in q and ":" not in q
    assert fts_match_query("?? -- ()") == ""
    assert fts_match_query("") == ""


@pytest.mark.parametrize("text", [
    "what's new in SQLite?", "foo(bar) -baz", "a:b c*d", '"quoted" NOT AND OR',
    "trailing -", "😀 emoji only", "", "   ",
])
def test_bm25_search_never_raises_on_hostile_input(store, text):
    store.persist_task("SQLite WAL mode internals", "resolved", _rounds("wal", ["a lesson"]))
    assert isinstance(bm25_search(store.db, text), list)


def test_bm25_search_finds_stored_query_with_apostrophe(store):
    store.persist_task("what's the deal with SQLite's WAL mode?", "resolved", _rounds("wal", ["l"]))
    hits = bm25_search(store.db, "SQLite's WAL mode?")
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# Retrieval round-trip without optional deps (memory-01, memory-02)
# ---------------------------------------------------------------------------


def test_persist_then_retrieve_round_trip_without_embedder(store):
    report = store.persist_task(
        "SQLite WAL mode internals", "resolved", _rounds("wal", ["checkpoint starvation is real"]),
        n_agents=3,
    )
    assert report.vector_source == "none"
    assert report.n_utterances == 3 and report.n_utterance_edges == 2
    assert report.distilled_by == "heuristic"
    assert len(report.candidate_ids) >= 1

    context, result = store.context_for("How does WAL mode work in SQLite?")
    assert result.seeds == [report.query_id]
    assert result.related_queries[0]["query_text"] == "SQLite WAL mode internals"
    assert result.interactions and result.interactions[0]["utterances"]
    assert result.used_llm is False and result.cost_usd == 0.0
    assert context.startswith(FENCE_OPEN) and context.rstrip().endswith(FENCE_CLOSE)
    assert "## Relevant prior findings" in context
    assert "SQLite WAL mode internals" in context
    assert "[researcher_1]" in context


def test_retrieve_bumps_access_counters(store):
    report = store.persist_task("topic alpha", "resolved", _rounds("alpha", ["x"]))
    before = store.db.execute("SELECT access_count FROM queries WHERE id=?", (report.query_id,)).fetchone()[0]
    store.retrieve("topic alpha")
    after = store.db.execute("SELECT access_count FROM queries WHERE id=?", (report.query_id,)).fetchone()[0]
    assert after == before + 1


def test_unrelated_query_returns_empty_context_not_error(store):
    store.persist_task("quantum error correction codes", "resolved", _rounds("qec", ["x"]))
    context, result = store.context_for("sourdough hydration ratios")
    assert context == "" and result.seeds == []


def test_lexical_edges_form_without_vectors(store):
    a = store.persist_task("Bitcoin SegWit BIP141 throughput", "resolved", _rounds("a", ["x"]))
    b = store.persist_task("Bitcoin SegWit BIP141 throughput", "resolved", _rounds("b", ["x"]))
    edges = store.db.execute("SELECT source_id, target_id, similarity FROM query_edges").fetchall()
    assert (b.query_id, a.query_id, 1.0) in edges


def test_candidate_without_vector_is_flagged_not_rolled_back(store):
    report = store.persist_task("topic", "resolved", _rounds("t", ["no embedder here"]))
    rows = store.db.execute(
        "SELECT embedding, vector_source FROM insight_candidates WHERE query_id=?", (report.query_id,)
    ).fetchall()
    assert rows and all(r == (b"", "none") for r in rows)
    assert store.db.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 1


def test_format_context_fence_is_optional_and_truncates():
    _, result = _fake_result()
    plain = format_context(result, fenced=False)
    assert FENCE_OPEN not in plain and plain.startswith("## Relevant prior findings")
    short = format_context(result, max_chars=len(FENCE_OPEN) + len(FENCE_CLOSE) + 160)
    assert short.startswith(FENCE_OPEN) and short.rstrip().endswith(FENCE_CLOSE)
    assert "[truncated" in short


def _fake_result():
    from gmemory.retrieval import RetrievalResult
    r = RetrievalResult(
        insights=[{"content": "lesson " * 30, "confidence": 0.7}],
        interactions=[{"query_id": "q", "query_text": "t " * 40, "status": "resolved",
                       "utterances": [{"agent_id": "r", "content": "c " * 100, "epoch": 0}]}],
        related_queries=[], used_llm=False, cost_usd=0.0,
    )
    return None, r


# ---------------------------------------------------------------------------
# Promotion gate N=3 on the live writer (memory-05)
# ---------------------------------------------------------------------------


def test_promotion_gate_requires_three_distinct_queries(store):
    lesson = "Walk-forward validation beats a fixed split"
    r1 = store.persist_task("task one", "resolved", _rounds("one", [lesson]))
    r2 = store.persist_task("task two", "resolved", _rounds("two", [lesson]))
    assert r1.promoted_ids == [] and r2.promoted_ids == []
    assert store.db.execute("SELECT COUNT(*) FROM insights").fetchone()[0] == 0
    # Re-running the same query does not count as new support.
    store.persist_task("task two again", "resolved", _rounds("two", [lesson]), query_id=r2.query_id)
    assert store.db.execute("SELECT COUNT(*) FROM insights").fetchone()[0] == 0

    r3 = store.persist_task("task three", "resolved", _rounds("three", [lesson]))
    assert len(r3.promoted_ids) == 1
    row = store.db.execute("SELECT content, supporting_queries, vector_source FROM insights").fetchone()
    assert row[0] == lesson
    assert set(json.loads(row[1])) == {r1.query_id, r2.query_id, r3.query_id}
    assert row[2] == "none"
    assert PROMOTION_N == 3

    # The promoted insight is now retrievable through the inverted index.
    context, result = store.context_for("task one")
    assert result.insights and result.insights[0]["content"] == lesson
    assert lesson in context


# ---------------------------------------------------------------------------
# Eviction cycle wired every cycle_frequency tasks (memory-06)
# ---------------------------------------------------------------------------


def test_eviction_runs_every_cycle_frequency_tasks(tmp_path, monkeypatch):
    store = MemoryStore.open(tmp_path / "mem.db", embedder=None, llm=None)
    store.settings.cycle_frequency = 3
    store.settings.archive_days = 1
    store.settings.purge_days = 1
    calls: list[dict] = []
    real = store.run_eviction_cycle
    monkeypatch.setattr(store, "run_eviction_cycle", lambda: calls.append(real()) or calls[-1])

    reports = [store.persist_task(f"task {i}", "resolved", _rounds(str(i), ["x"])) for i in range(1, 7)]
    assert [bool(r.eviction) for r in reports] == [False, False, True, False, False, True]
    assert len(calls) == 2


def test_eviction_archives_cold_queries_and_purges_old_traces(store):
    store.settings.archive_days = 1
    store.settings.purge_days = 1
    cold = store.persist_task("cold query", "resolved", _rounds("cold", ["x"]))
    hot = store.persist_task("hot query", "resolved", _rounds("hot", ["y"]))
    old = time.time() - 10 * 86400
    store.db.execute("UPDATE queries SET last_accessed=?, created_at=? WHERE id=?", (old, old, cold.query_id))
    store.db.execute("UPDATE queries SET access_count=5 WHERE id=?", (hot.query_id,))
    store.db.commit()

    counts = store.run_eviction_cycle()
    assert counts["archived"] == 1
    assert store.db.execute("SELECT id FROM queries_archive").fetchone()[0] == cold.query_id
    assert store.db.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 1
    # Cascades took the cold query's traces and candidates with it; insights never evict.
    assert store.db.execute("SELECT COUNT(*) FROM utterances WHERE query_id=?", (cold.query_id,)).fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM insight_candidates WHERE query_id=?", (cold.query_id,)).fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM utterances WHERE query_id=?", (hot.query_id,)).fetchone()[0] == 3


# ---------------------------------------------------------------------------
# Migration of a legacy memory/-schema database (idempotent)
# ---------------------------------------------------------------------------


def _make_legacy_db(path: Path) -> None:
    db = sqlite3.connect(str(path))
    db.executescript(LEGACY_SCHEMA)
    now = time.time()
    db.executemany("INSERT INTO queries VALUES (?,?,?,?,?,?)", [
        ("q1", "SQLite WAL mode internals", "failed", now - 100, 0, now - 100),
        ("q2", "SQLite WAL mode internals", "failed", now - 50, 2, now - 50),
        ("q3", "Consensus mechanisms in distributed systems", "failed", now, 0, now),
    ])
    db.execute("INSERT INTO query_edges VALUES ('q1','q2',?)", (now,))
    db.execute("INSERT INTO insights VALUES ('i1','Checkpointing stalls readers', '[\"q1\",\"q2\"]', ?, 1)", (now,))
    db.executemany("INSERT INTO utterances VALUES (?,?,?,?,?,?)", [
        ("u1", "q1", "researcher_1", "WAL appends frames", 0, now),
        ("u2", "q1", "synthesizer", "summary", 1, now),
        ("u3", "q9-orphan", "ghost", "no parent query", 0, now),
    ])
    db.execute("INSERT INTO utterance_edges VALUES ('u1','u2','q1')")
    db.commit()
    db.close()


def test_migrate_legacy_db_is_idempotent_and_keeps_rows(tmp_path):
    path = tmp_path / "legacy.db"
    _make_legacy_db(path)
    db = sqlite3.connect(str(path))
    assert is_legacy(db)
    db.close()

    first = migrate_path(path)
    assert first["legacy"] is True
    assert (first["queries"], first["query_edges"], first["insights"], first["utterances"]) == (3, 1, 1, 3)

    second = migrate_path(path)       # idempotent: nothing to rebuild the second time
    assert second["legacy"] is False and second["queries"] == 0

    db = sqlite3.connect(str(path))
    assert not is_legacy(db)
    assert db.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 3
    assert db.execute("SELECT similarity FROM query_edges").fetchone()[0] == pytest.approx(0.7)
    assert db.execute("SELECT embedding, vector_source, last_validated_at > 0 FROM insights").fetchone() == (b"", "none", 1)
    assert {r[0] for r in db.execute("SELECT query_id FROM insight_query_index WHERE insight_id='i1'")} == {"q1", "q2"}
    assert db.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 2     # orphan dropped
    assert db.execute("SELECT COUNT(*) FROM utterance_edges").fetchone()[0] == 1
    # FTS mirror is populated for the legacy rows.
    assert len(bm25_search(db, "WAL mode")) == 2
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    db.close()

    # The migrated file is a fully working store.
    store = MemoryStore.open(path, embedder=None, llm=None)
    context, result = store.context_for("SQLite WAL mode")
    assert result.insights[0]["content"] == "Checkpointing stalls readers"
    assert "Checkpointing stalls readers" in context
    report = store.persist_task("SQLite WAL checkpoints", "resolved", _rounds("ckpt", ["z"]))
    assert report.query_id
    store.close()


def test_migrate_script_cli_on_a_copy(tmp_path):
    import subprocess
    path = tmp_path / "copy.db"
    _make_legacy_db(path)
    script = ROOT / "scripts" / "migrate_memory_db.py"
    check = subprocess.run([sys.executable, str(script), "--check", str(path)], capture_output=True, text=True)
    assert check.returncode == 1 and "LEGACY" in check.stdout
    run = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "migrated legacy schema" in run.stdout
    assert (tmp_path / "copy.db.pre-gmemory.bak").exists()
    again = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)
    assert again.returncode == 0 and "already canonical" in again.stdout


def test_migrate_adds_columns_introduced_after_first_gmemory_revision(tmp_path):
    db = sqlite3.connect(str(tmp_path / "old-gmemory.db"))
    migrate_db(db)
    db.execute("ALTER TABLE insights DROP COLUMN vector_source")
    db.execute("ALTER TABLE insight_candidates DROP COLUMN vector_source")
    db.execute("ALTER TABLE queries DROP COLUMN embedding")
    db.commit()
    report = migrate_db(db)
    assert sorted(report["columns_added"]) == [
        "insight_candidates.vector_source", "insights.vector_source", "queries.embedding",
    ]
    db.close()


def test_hop_expansion_zero_disables_graph_walk(store):
    a = store.persist_task("graph walk alpha", "resolved", _rounds("a", ["x"]))
    b = store.persist_task("graph walk alpha", "resolved", _rounds("b", ["x"]))
    store.qg.link(a.query_id, b.query_id, 0.9)
    store.settings.top_k = 1
    with_hop = store.retrieve("graph walk alpha")
    assert {q["id"] for q in with_hop.related_queries} == {a.query_id, b.query_id}
    store.settings.hops = 0
    no_hop = store.retrieve("graph walk alpha")
    assert len(no_hop.related_queries) == 1


# ---------------------------------------------------------------------------
# The orchestrator's own persistence function, against a temp store
# ---------------------------------------------------------------------------


def test_orchestrator_persist_to_memory_writes_all_tiers(tmp_path, capsys):
    import orchestrator

    store = MemoryStore.open(tmp_path / "mem.db", embedder=None, llm=None)
    rounds = _rounds("consensus", ["lesson one", "lesson two"])
    agents = [object(), object(), object()]
    orchestrator._persist_to_memory(store, None, "Consensus mechanisms", True, False, rounds, agents)
    out = capsys.readouterr().out
    assert "Memory: stored query" in out and "3 utterances" in out

    check = MemoryStore.open(tmp_path / "mem.db", embedder=None, llm=None)   # the function closed `store`
    row = check.db.execute("SELECT status, n_agents, n_rounds, tags_json FROM queries").fetchone()
    assert row[0] == "resolved" and row[1] == 3 and row[2] == 2
    assert "swarm:consensus" in json.loads(row[3])
    assert check.db.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 3
    assert check.db.execute("SELECT COUNT(*) FROM insight_candidates").fetchone()[0] >= 1
    context, _ = check.context_for("Consensus mechanisms in distributed systems")
    assert "Consensus mechanisms" in context and context.startswith(FENCE_OPEN)
    check.close()


# ---------------------------------------------------------------------------
# memory_ops routing: resolve_model + make_provider_backend, no model call
# ---------------------------------------------------------------------------


class _FakeBackend:
    backend_id = "fake"

    def __init__(self):
        self.calls = []

    def call(self, call):
        from backends.core import AgentResult
        self.calls.append(call)
        if call.role == "memory_distill":
            parsed = {"insights": [{"content": "LLM lesson", "confidence": 0.9, "tags": ["domain:db"]}]}
        elif call.role == "memory_relevance":
            parsed = {"relevance": 0.9}
        else:
            parsed = {"relevant_indices": [0]}
        return AgentResult(backend_id="fake", model=call.model, text="{}", raw_stdout="{}",
                           parsed=parsed, elapsed_s=0.01, cost_usd=None)


def test_memory_llm_routes_through_memory_ops_tier_with_fake_backend(monkeypatch):
    import gmemory.llm_ops as llm_ops
    from config import load_config

    for var in ("BLITZ_MODEL", "BLITZ_MODEL_MEMORY_OPS", "BLITZ_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config()
    fake = _FakeBackend()
    import backends
    monkeypatch.setattr(backends, "make_provider_backend", lambda *a, **k: fake)

    llm = llm_ops.build_memory_llm(cfg, backend_id="claude")
    assert llm is not None
    expected = backends.resolve_model("memory_ops", "claude", cfg)
    assert llm.model == expected.model and llm.backend_id == "claude"

    cands = llm.distill("t", "resolved", [{"agent_id": "r", "findings": "f"}])
    assert [c.content for c in cands] == ["LLM lesson"] and cands[0].tags == ["domain:db"]
    assert fake.calls[-1].model_role == "memory_ops" and fake.calls[-1].model == expected.model
    assert llm.relevance("a", "b") == (0.9, None)
    kept, cost = llm.sparsify([{"content": str(i)} for i in range(7)], "q")
    assert kept == [{"content": "0"}] and cost is None


def test_memory_llm_is_none_when_role_tier_unresolvable(monkeypatch):
    import gmemory.llm_ops as llm_ops
    from config import load_config

    monkeypatch.delenv("BLITZ_MODEL", raising=False)
    cfg = load_config()
    assert llm_ops.build_memory_llm(cfg, backend_id="no-such-provider") is None


def test_store_uses_llm_hooks_only_when_warm(tmp_path, monkeypatch):
    fake = _FakeBackend()
    store = MemoryStore.open(tmp_path / "mem.db", embedder=None, llm=None)
    store.settings.llm_ops_threshold = 3
    for i in range(2):
        r = store.persist_task(f"task {i}", "resolved", _rounds(str(i), ["x"]))
        assert r.distilled_by == "heuristic"
    assert fake.calls == []

    from gmemory.llm_ops import MemoryLLM
    store._llm = MemoryLLM(backend=fake, backend_id="fake", model="tier-model", effort=None,
                           sandbox="read-only", approval_policy="never", ephemeral=True)
    r = store.persist_task("task 2", "resolved", _rounds("2", ["x"]))      # third task -> warm
    assert r.distilled_by == "llm"
    assert any(c.role == "memory_distill" for c in fake.calls)
    result = store.retrieve("task 2")
    assert result.used_llm is True and result.cost_usd is None       # unknown cost stays unknown
    store.close()


# ---------------------------------------------------------------------------
# build_trace + settings
# ---------------------------------------------------------------------------


def test_build_trace_links_consecutive_rounds():
    utts, edges = build_trace([[{"agent_id": "a", "findings": "1"}, {"agent_id": "b", "findings": "2"}],
                               [{"agent_id": "s", "findings": "3"}]])
    assert [u["epoch"] for u in utts] == [0, 0, 1]
    assert len(edges) == 2 and all(t == utts[2]["id"] for _, t in edges)


def test_settings_come_from_blitz_config():
    from config import load_config
    cfg = load_config()
    s = MemorySettings.from_config(cfg)
    assert s.top_k == cfg.memory.top_k_retrieval
    assert s.cycle_frequency == cfg.eviction.cycle_frequency
    assert s.max_context_chars == cfg.memory.max_context_tokens * 4


def test_legacy_memory_package_is_a_shim():
    import memory
    assert memory.MemoryStore is MemoryStore
    with pytest.raises(ImportError, match="gmemory"):
        import memory.writer  # noqa: F401


def test_open_backs_up_legacy_db_once_before_migrating(tmp_path):
    from gmemory.store import MemoryStore
    path = tmp_path / "legacy.db"
    _make_legacy_db(path)
    backup = path.with_name(path.name + ".pre-gmemory.bak")
    assert not backup.exists()
    store = MemoryStore.open(path, embedder=None, llm=None)
    assert backup.exists() and backup.stat().st_size > 0
    assert not is_legacy(store.db)
    assert is_legacy(sqlite3.connect(str(backup)))
    mtime = backup.stat().st_mtime_ns
    store.db.close()
    MemoryStore.open(path, embedder=None, llm=None).db.close()   # second open: no re-copy
    assert backup.stat().st_mtime_ns == mtime
