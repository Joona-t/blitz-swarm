"""Idempotent migration of a memory.db into the canonical gmemory schema.

Two shapes exist in the wild:

  * legacy `memory/schema.sql` (v0.1-v0.2): `queries` has six columns, no
    FTS mirror, `query_edges` has no similarity, `insights` has no vector.
  * canonical `gmemory/schema.sql`: everything above plus Tier 1 traces,
    FTS5 mirrors, insight candidates, tags, `vector_source` flags.

`migrate_db` detects a legacy database by the absence of `queries.n_agents`,
copies every legacy row out, drops the legacy tables, applies the canonical
schema and re-inserts the rows with honest defaults (edge similarity = the
link threshold they were known to exceed; insights get an empty vector and
`vector_source='none'`). Running it on a canonical database only applies
`CREATE ... IF NOT EXISTS` statements and adds any column a newer schema
revision introduced, so it is safe to call on every open.

No optional dependency: stdlib sqlite3 only.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Legacy query_edges were only written when embedding similarity exceeded the
# link threshold (memory/writer.py QUERY_LINK_THRESHOLD = 0.7); the value
# itself was never stored, so the threshold is the honest lower bound.
LEGACY_EDGE_SIMILARITY = 0.7

LEGACY_TABLES = (
    "queries", "query_edges", "insights", "insight_edges",
    "utterances", "utterance_edges", "queries_archive",
)

# Columns added after the first gmemory schema revision: (table, column, DDL).
_ADDED_COLUMNS = (
    ("queries", "embedding", "BLOB"),
    ("insight_candidates", "vector_source", "TEXT NOT NULL DEFAULT 'model'"),
    ("insights", "vector_source", "TEXT NOT NULL DEFAULT 'model'"),
)


def _columns(db: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in db.execute(f"PRAGMA table_info({table})").fetchall()]


def _table_exists(db: sqlite3.Connection, table: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?", (table,)
    ).fetchone()
    return row is not None


def is_legacy(db: sqlite3.Connection) -> bool:
    """True when `queries` exists in the memory/-era shape."""
    cols = _columns(db, "queries")
    return bool(cols) and "n_agents" not in cols


def apply_schema(db: sqlite3.Connection) -> None:
    db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def _add_missing_columns(db: sqlite3.Connection) -> list[str]:
    added: list[str] = []
    for table, column, ddl in _ADDED_COLUMNS:
        if _table_exists(db, table) and column not in _columns(db, table):
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            added.append(f"{table}.{column}")
    return added


def _fetch(db: sqlite3.Connection, table: str, cols: str) -> list[tuple]:
    if not _table_exists(db, table):
        return []
    return db.execute(f"SELECT {cols} FROM {table}").fetchall()


def migrate_db(db: sqlite3.Connection) -> dict:
    """Bring `db` to the canonical schema. Idempotent. Returns a report."""
    report: dict = {
        "legacy": False, "queries": 0, "query_edges": 0, "insights": 0,
        "insight_edges": 0, "utterances": 0, "utterance_edges": 0,
        "queries_archive": 0, "columns_added": [],
    }
    if not is_legacy(db):
        apply_schema(db)
        report["columns_added"] = _add_missing_columns(db)
        db.commit()
        return report

    report["legacy"] = True
    # Pull every legacy row out before touching the tables.
    queries = _fetch(db, "queries", "id, query_text, status, created_at, access_count, last_accessed")
    edges = _fetch(db, "query_edges", "source_id, target_id, created_at")
    insights = _fetch(db, "insights", "id, content, supporting_queries, created_at, access_count")
    insight_edges = _fetch(db, "insight_edges",
                           "source_insight_id, target_insight_id, via_query_id, created_at")
    utterances = _fetch(db, "utterances", "id, query_id, agent_id, content, epoch, timestamp")
    utt_edges = _fetch(db, "utterance_edges", "source_id, target_id, query_id")
    archive = _fetch(db, "queries_archive",
                     "id, query_text, status, created_at, access_count, last_accessed, archived_at")

    # Legacy tables carry FK clauses without ON DELETE CASCADE; they must be
    # rebuilt, not kept, or eviction would trip the constraint later.
    db.execute("PRAGMA foreign_keys = OFF")
    with db:
        for table in LEGACY_TABLES:
            db.execute(f"DROP TABLE IF EXISTS {table}")
    apply_schema(db)            # also turns foreign_keys back ON
    db.execute("PRAGMA foreign_keys = OFF")   # orphans in legacy data must not abort the copy

    known_qids = {r[0] for r in queries}
    with db:
        db.executemany(
            "INSERT OR IGNORE INTO queries (id, query_text, status, created_at, access_count, "
            "last_accessed, n_agents, n_rounds, cost_usd, wall_clock_s, quality_score, tags_json) "
            "VALUES (?,?,?,?,?,?,0,0,NULL,0.0,NULL,'[]')",
            [(r[0], r[1], r[2] if r[2] in ("resolved", "failed", "partial") else "failed",
              float(r[3]), int(r[4] or 0), float(r[5] or r[3])) for r in queries],
        )
        db.executemany(
            "INSERT OR IGNORE INTO query_edges (source_id, target_id, similarity, created_at) "
            "VALUES (?,?,?,?)",
            [(s, t, LEGACY_EDGE_SIMILARITY, float(c)) for (s, t, c) in edges
             if s != t and s in known_qids and t in known_qids],
        )
        for (iid, content, supporting_json, created_at, access_count) in insights:
            try:
                supporting = json.loads(supporting_json) if supporting_json else []
            except json.JSONDecodeError:
                supporting = []
            db.execute(
                "INSERT OR IGNORE INTO insights (id, content, supporting_queries, tags_json, "
                "created_at, last_validated_at, access_count, confidence, embedding, vector_source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (iid, content, json.dumps(sorted(supporting)), "[]", float(created_at),
                 float(created_at), int(access_count or 0), 0.6, b"", "none"),
            )
            db.executemany(
                "INSERT OR IGNORE INTO insight_query_index (insight_id, query_id) VALUES (?,?)",
                [(iid, q) for q in supporting if q in known_qids],
            )
        db.executemany(
            "INSERT OR IGNORE INTO insight_edges (source_insight_id, target_insight_id, "
            "via_query_id, created_at) VALUES (?,?,?,?)",
            [(s, t, v, float(c)) for (s, t, v, c) in insight_edges],
        )
        db.executemany(
            "INSERT OR IGNORE INTO utterances (id, query_id, agent_id, content, epoch, timestamp) "
            "VALUES (?,?,?,?,?,?)",
            [(i, q, a or "", c or "", int(e or 0), float(t)) for (i, q, a, c, e, t) in utterances
             if q in known_qids],
        )
        db.executemany(
            "INSERT OR IGNORE INTO utterance_edges (source_id, target_id, query_id) VALUES (?,?,?)",
            utt_edges,
        )
        db.executemany(
            "INSERT OR IGNORE INTO queries_archive (id, query_text, status, created_at, "
            "access_count, last_accessed, archived_at) VALUES (?,?,?,?,?,?,?)",
            archive,
        )
        # Rows inserted above fired the FTS triggers; a rebuild is cheap insurance.
        db.execute("INSERT INTO queries_fts(queries_fts) VALUES('rebuild')")
        db.execute("INSERT INTO insights_fts(insights_fts) VALUES('rebuild')")
    db.execute("PRAGMA foreign_keys = ON")

    report.update({
        "queries": len(queries), "query_edges": len(edges), "insights": len(insights),
        "insight_edges": len(insight_edges), "utterances": len(utterances),
        "utterance_edges": len(utt_edges), "queries_archive": len(archive),
    })
    return report


def migrate_path(path: Path | str) -> dict:
    """Open `path`, migrate it, close it. The report says what happened."""
    db = sqlite3.connect(str(path))
    try:
        return migrate_db(db)
    finally:
        db.close()
