-- G-Memory Tier 2/3 schema additions
-- Idempotent migration on top of memory/schema.sql; existing rows survive.

PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 30000;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- Tier 2 — Query graph (extends existing `queries`, additive columns guarded
-- by their absence in the legacy schema; if they already exist, ALTER TABLE
-- is skipped by the migrate.py runner).
CREATE TABLE IF NOT EXISTS queries (
    id              TEXT PRIMARY KEY,
    query_text      TEXT NOT NULL,
    status          TEXT NOT NULL CHECK(status IN ('resolved','failed','partial')),
    created_at      REAL NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed   REAL NOT NULL,
    n_agents        INTEGER NOT NULL DEFAULT 0,
    n_rounds        INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL    NOT NULL DEFAULT 0.0,
    wall_clock_s    REAL    NOT NULL DEFAULT 0.0,
    quality_score   REAL,
    tags_json       TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS query_edges (
    source_id   TEXT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    target_id   TEXT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    similarity  REAL NOT NULL CHECK(similarity BETWEEN 0 AND 1),
    created_at  REAL NOT NULL,
    PRIMARY KEY (source_id, target_id),
    CHECK (source_id <> target_id)
);

-- FTS5 mirror for hybrid retrieval. Triggers keep it in sync with `queries`.
CREATE VIRTUAL TABLE IF NOT EXISTS queries_fts USING fts5(
    query_text,
    content='queries',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS queries_ai AFTER INSERT ON queries BEGIN
    INSERT INTO queries_fts(rowid, query_text) VALUES (new.rowid, new.query_text);
END;
CREATE TRIGGER IF NOT EXISTS queries_ad AFTER DELETE ON queries BEGIN
    INSERT INTO queries_fts(queries_fts, rowid, query_text)
        VALUES('delete', old.rowid, old.query_text);
END;
CREATE TRIGGER IF NOT EXISTS queries_au AFTER UPDATE ON queries BEGIN
    INSERT INTO queries_fts(queries_fts, rowid, query_text)
        VALUES('delete', old.rowid, old.query_text);
    INSERT INTO queries_fts(rowid, query_text) VALUES (new.rowid, new.query_text);
END;

-- Tier 3 — Insight graph
CREATE TABLE IF NOT EXISTS insight_candidates (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    query_id        TEXT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    embedding       BLOB NOT NULL,           -- raw f32 little-endian, 384*4 B
    confidence      REAL NOT NULL DEFAULT 0.5,
    tags_json       TEXT NOT NULL DEFAULT '[]',
    created_at      REAL NOT NULL,
    last_evaluated_at REAL,
    promoted_to     TEXT,
    promoted_at     REAL
);

CREATE TABLE IF NOT EXISTS insights (
    id                  TEXT PRIMARY KEY,
    content             TEXT NOT NULL,
    supporting_queries  TEXT NOT NULL DEFAULT '[]',
    tags_json           TEXT NOT NULL DEFAULT '[]',
    created_at          REAL NOT NULL,
    last_validated_at   REAL NOT NULL,
    access_count        INTEGER NOT NULL DEFAULT 0,
    confidence          REAL NOT NULL DEFAULT 0.6,
    embedding           BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS insight_edges (
    source_insight_id   TEXT NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
    target_insight_id   TEXT NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
    via_query_id        TEXT NOT NULL REFERENCES queries(id)  ON DELETE CASCADE,
    created_at          REAL NOT NULL,
    PRIMARY KEY (source_insight_id, target_insight_id, via_query_id)
);

-- Inverted index — query_id -> insight_id, for fast meta-loop queries.
-- Maintained by application code (insight_graph.py) on insert/update.
CREATE TABLE IF NOT EXISTS insight_query_index (
    insight_id  TEXT NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
    query_id    TEXT NOT NULL REFERENCES queries(id)  ON DELETE CASCADE,
    PRIMARY KEY (insight_id, query_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS insights_fts USING fts5(
    content, content='insights', content_rowid='rowid', tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS insights_ai AFTER INSERT ON insights BEGIN
    INSERT INTO insights_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS insights_ad AFTER DELETE ON insights BEGIN
    INSERT INTO insights_fts(insights_fts, rowid, content)
        VALUES('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS insights_au AFTER UPDATE ON insights BEGIN
    INSERT INTO insights_fts(insights_fts, rowid, content)
        VALUES('delete', old.rowid, old.content);
    INSERT INTO insights_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- Meta-loop tag denormalization
CREATE TABLE IF NOT EXISTS query_tags (
    query_id    TEXT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    axis        TEXT NOT NULL,              -- 'domain'|'pattern'|'pitfall'|'swarm'|'meta'
    value       TEXT NOT NULL,
    PRIMARY KEY (query_id, axis, value)
);
CREATE INDEX IF NOT EXISTS idx_qtags_axis_val ON query_tags(axis, value);
CREATE INDEX IF NOT EXISTS idx_qtags_query   ON query_tags(query_id);

CREATE TABLE IF NOT EXISTS insight_tags (
    insight_id  TEXT NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
    axis        TEXT NOT NULL,
    value       TEXT NOT NULL,
    PRIMARY KEY (insight_id, axis, value)
);
CREATE INDEX IF NOT EXISTS idx_itags_axis_val ON insight_tags(axis, value);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_queries_status        ON queries(status);
CREATE INDEX IF NOT EXISTS idx_queries_created       ON queries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_queries_last_accessed ON queries(last_accessed);
CREATE INDEX IF NOT EXISTS idx_query_edges_target    ON query_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_query_edges_sim       ON query_edges(similarity DESC);
CREATE INDEX IF NOT EXISTS idx_cand_unpromoted       ON insight_candidates(promoted_at)
    WHERE promoted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_cand_query            ON insight_candidates(query_id);
CREATE INDEX IF NOT EXISTS idx_iqi_query             ON insight_query_index(query_id);
CREATE INDEX IF NOT EXISTS idx_insights_validated    ON insights(last_validated_at DESC);
CREATE INDEX IF NOT EXISTS idx_insights_conf         ON insights(confidence DESC);
