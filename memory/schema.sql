-- G-Memory three-tier hierarchical memory schema for Blitz-Swarm
-- Based on G-Memory (NeurIPS 2025, arXiv 2506.07398)

PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 30000;
PRAGMA synchronous = NORMAL;

-- =========================================================================
-- Tier 3: Insight Graph (highest abstraction — distilled wisdom)
-- =========================================================================

CREATE TABLE IF NOT EXISTS insights (
    id                  TEXT PRIMARY KEY,
    content             TEXT NOT NULL,
    supporting_queries  TEXT NOT NULL DEFAULT '[]',  -- JSON array of query IDs
    created_at          REAL NOT NULL,
    access_count        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS insight_edges (
    source_insight_id   TEXT NOT NULL REFERENCES insights(id),
    target_insight_id   TEXT NOT NULL REFERENCES insights(id),
    via_query_id        TEXT NOT NULL,
    created_at          REAL NOT NULL,
    PRIMARY KEY (source_insight_id, target_insight_id, via_query_id)
);

-- =========================================================================
-- Tier 2: Query Graph (task-level associations)
-- =========================================================================

CREATE TABLE IF NOT EXISTS queries (
    id                  TEXT PRIMARY KEY,
    query_text          TEXT NOT NULL,
    status              TEXT NOT NULL CHECK(status IN ('resolved', 'failed')),
    created_at          REAL NOT NULL,
    access_count        INTEGER DEFAULT 0,
    last_accessed       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS query_edges (
    source_id           TEXT NOT NULL REFERENCES queries(id),
    target_id           TEXT NOT NULL REFERENCES queries(id),
    created_at          REAL NOT NULL,
    PRIMARY KEY (source_id, target_id)
);

-- =========================================================================
-- Tier 1: Interaction Graph (raw agent communication traces)
-- =========================================================================

CREATE TABLE IF NOT EXISTS utterances (
    id                  TEXT PRIMARY KEY,
    query_id            TEXT NOT NULL REFERENCES queries(id),
    agent_id            TEXT NOT NULL,
    content             TEXT NOT NULL,
    epoch               INTEGER NOT NULL,
    timestamp           REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS utterance_edges (
    source_id           TEXT NOT NULL REFERENCES utterances(id),
    target_id           TEXT NOT NULL REFERENCES utterances(id),
    query_id            TEXT NOT NULL REFERENCES queries(id),
    PRIMARY KEY (source_id, target_id)
);

-- =========================================================================
-- Archive table for evicted queries (Phase 5)
-- =========================================================================

CREATE TABLE IF NOT EXISTS queries_archive (
    id                  TEXT PRIMARY KEY,
    query_text          TEXT NOT NULL,
    status              TEXT NOT NULL,
    created_at          REAL NOT NULL,
    access_count        INTEGER DEFAULT 0,
    last_accessed       REAL NOT NULL,
    archived_at         REAL NOT NULL
);

-- =========================================================================
-- Indexes for retrieval performance
-- =========================================================================

CREATE INDEX IF NOT EXISTS idx_utterances_query ON utterances(query_id);
CREATE INDEX IF NOT EXISTS idx_queries_status ON queries(status);
CREATE INDEX IF NOT EXISTS idx_queries_last_accessed ON queries(last_accessed);
CREATE INDEX IF NOT EXISTS idx_insights_access ON insights(access_count DESC);
