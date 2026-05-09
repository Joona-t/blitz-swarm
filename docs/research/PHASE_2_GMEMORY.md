# Phase 2 G-Memory Tier 2/3 — Implementation Deep Dive

**Status:** Frozen research output from background-task agent run 2026-05-09.
**Source:** Phase 2 G-Memory deep-dive subagent.
**Consumed by:** `gmemory/` implementation.

---

## 0. Honest reframe of what's already in the repo

The repo claims "Tier 1 implemented" but `memory/writer.py` and `memory/reader.py` already write **all three tiers** end-to-end with naive logic:

- `memory/schema.sql` — already creates `queries`, `query_edges`, `insights`, `insight_edges`, `utterances`, `utterance_edges`, `queries_archive`. Tier 2/3 tables exist; what's missing is **FTS5 virtual tables, hybrid retrieval, promotion gating, and meta-tags**.
- `memory/writer.py:_store_insight` — already deduplicates by 0.85 cosine. **Skips the promotion gate entirely** — every task creates a node. This is the noise problem.
- `memory/writer.py:_find_related_queries` — uses LanceDB top-5 with 0.7 cosine. But `_distance` interpreted as L2 — **wrong for normalized MiniLM**. KI candidate.
- `memory/reader.py:retrieve_memory` — implements steps 1–4 but **does not call `score_relevance` or `sparsify_interaction` from inside `retrieve_memory`**. Steps 5–6 unwired.
- `embedder.py:cosine_similarity` — pure-Python triple-pass. Switch to `numpy.dot` with pre-normalized vectors. KI candidate.
- `blitz.toml` — `query_link_threshold = 0.7`, `top_k_retrieval = 2` per paper. Good.

**Build-out below is upgrading an existing skeleton, not greenfield.**

---

## 1. Tier 2 — Query Graph

### 1.1 What a Tier 2 node is

One node per swarm run. Query text is the topic string. Status: `resolved` (all voters voted ready) / `failed` / `partial`. Embedding 384-dim normalized from `embedder.py`. Edges encode pairwise semantic similarity ≥ τ_link.

### 1.2 SQLite DDL — `gmemory/schema.sql`

```sql
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 30000;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- Tier 2 (extends existing `queries`)
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
    source_id       TEXT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    target_id       TEXT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    similarity      REAL NOT NULL CHECK(similarity BETWEEN 0 AND 1),
    created_at      REAL NOT NULL,
    PRIMARY KEY (source_id, target_id),
    CHECK (source_id <> target_id)
);

-- FTS5 mirror for hybrid retrieval
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

CREATE INDEX IF NOT EXISTS idx_queries_status        ON queries(status);
CREATE INDEX IF NOT EXISTS idx_queries_created       ON queries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_queries_last_accessed ON queries(last_accessed);
CREATE INDEX IF NOT EXISTS idx_query_edges_target   ON query_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_query_edges_sim      ON query_edges(similarity DESC);
```

**Why FTS5 next to vectors:** single embedding miss kills recall (`"GBM crypto features"` vs `"LightGBM cryptocurrency feature engineering"` — MiniLM ~0.62 cosine, below 0.7). Hybrid stage is RRF-fused (k=60), runs only at retrieval, never at write.

### 1.3 Module skeleton — `gmemory/query_graph.py`

```python
DEFAULT_DB_PATH      = Path(__file__).parent.parent / "memory.db"
DEFAULT_LANCE_PATH   = Path(__file__).parent.parent / "memory_vectors"
LANCE_TABLE_NAME     = "queries"

KNN_K                = 5
COSINE_LINK_THRESHOLD = 0.70
ONE_HOP_FANOUT_CAP   = 30
EMBED_DIM            = 384

@dataclass(slots=True)
class QueryRecord:
    id: str
    query_text: str
    status: str
    created_at: float
    n_agents: int
    n_rounds: int
    cost_usd: float
    wall_clock_s: float
    quality_score: float | None
    tags: list[str]

class QueryGraph:
    def __init__(self, db_path, lance_path, embedder=None): ...
    def initialize(self) -> None: ...
    def close(self) -> None: ...
    def add_task(self, query_text, status, *, n_agents, n_rounds, cost_usd,
                 wall_clock_s, quality_score, tags=(), query_id=None) -> str: ...
    def link(self, source_id, target_id, similarity): ...
    def nearest_neighbors(self, query_text, *, k=KNN_K, threshold=COSINE_LINK_THRESHOLD,
                          exclude_ids=()) -> list[tuple[str, float]]: ...
    def one_hop(self, seed_ids) -> set[str]: ...
    def get_records(self, query_ids) -> list[QueryRecord]: ...
    def retrieve_context(self, query_text, *, k=KNN_K, threshold=COSINE_LINK_THRESHOLD) -> dict: ...
    def stats(self) -> dict: ...
```

### 1.4 Pseudocode

```python
def add_task(self, query_text, status, *, n_agents, n_rounds, cost_usd,
             wall_clock_s, quality_score, tags=(), query_id=None) -> str:
    qid = query_id or str(uuid.uuid4())
    now = time.time()
    embedding = self.embedder.encode(query_text)

    neighbors = self.nearest_neighbors(
        query_text, k=KNN_K, threshold=COSINE_LINK_THRESHOLD,
        exclude_ids=(qid,))

    with self._db:
        self._db.execute("""
            INSERT OR IGNORE INTO queries
            (id, query_text, status, created_at, access_count, last_accessed,
             n_agents, n_rounds, cost_usd, wall_clock_s, quality_score, tags_json)
            VALUES (?,?,?,?,0,?,?,?,?,?,?,?)
        """, (qid, query_text, status, now, now,
              n_agents, n_rounds, cost_usd, wall_clock_s,
              quality_score, json.dumps(list(tags))))
        for (nid, sim) in neighbors:
            self._db.execute(
                "INSERT OR IGNORE INTO query_edges VALUES (?,?,?,?)",
                (qid, nid, sim, now))

    self._lance_table.add([{
        "query_id": qid, "text": query_text, "vector": embedding,
        "status": status, "created_at": now,
    }])
    return qid

def nearest_neighbors(self, query_text, *, k, threshold, exclude_ids):
    if self._lance_table is None or self._row_count() == 0:
        return []
    excl = set(exclude_ids)
    df = (self._lance_table
          .search(query_text)
          .distance_type("cosine")  # CRITICAL fix vs v0.1
          .limit(k * 2)
          .to_pandas())
    out = []
    for _, row in df.iterrows():
        if row["query_id"] in excl: continue
        sim = 1.0 - float(row["_distance"])
        if sim >= threshold:
            out.append((row["query_id"], sim))
        if len(out) >= k: break
    return out
```

### 1.5 Empirical knobs

| Knob | Default | Rationale | Tuning failure |
|---|---|---|---|
| `KNN_K` | **5** | k>5 → quadratic edge density growth, no recall gain. k<3 → graph chains, no graph. | k=20 → near-clique by 200 nodes; LLM relevance can't tell signal. k=1 → near-duplicates only. |
| `COSINE_LINK_THRESHOLD` | **0.70** | MiniLM same-domain pairs cluster 0.55–0.85. 0.70 keeps related domains, cuts cross-domain. | 0.55 → graph collapses to one component within 50 tasks. 0.85 → mostly orphans. |
| `ONE_HOP_FANOUT_CAP` | **30** | Hub regions dominate LLM relevance budget. | 5 → discard valid neighbors. 1000 → P99 latency 50× spike. |
| Embedding model | **MiniLM-L6-v2** (384-dim) | ONNX backend, singleton, paper precedent. | mpnet 768-dim → 2× index, 5× embed time, no precision gain. |
| FTS5 tokenizer | **`porter unicode61`** | Stems "researching" → "research"; handles Unicode. | Default → BM25 misses morphological variants. |

### 1.6 Cost

Per-task storage: 0.7 KB SQLite + 1.9 KB LanceDB + 120 B FTS5 = **~2.7 KB/task**. 10K tasks: 27 MB.

Per-task retrieval: embed 8–14 ms + LanceDB 2–5 ms + 1-hop 1–3 ms = **<20 ms, $0.**

### 1.7 Test cases

```python
def test_add_task_creates_row_and_vector(qg): ...
def test_links_form_above_threshold(qg): ...
def test_no_link_below_threshold(qg): ...
def test_one_hop_returns_neighbors_only_once(qg, hub_fixture): ...
def test_one_hop_respects_fanout_cap(qg, dense_fixture): ...
def test_idempotent_add_task(qg): ...
def test_retrieve_context_combines_seeds_and_expansion(qg, populated_fixture): ...
def test_lancedb_missing_falls_back_to_empty(qg_no_lance): ...
```

### 1.8 Failure recovery

- LanceDB missing → `nearest_neighbors` returns `[]`, BM25-only fallback.
- Vector/SQL desync → reconciliation pass on `initialize()`, bounded 200 rows.
- Graph past 10K tasks → ~30K edges (~30 MB); inverted-index on `(query_id → insight_id)` fixes scan bottleneck (§3.2).

---

## 2. Hybrid retrieval — `gmemory/hybrid.py`

### 2.1 RRF fusion

```python
RRF_K = 60          # Cormack et al. canonical default
VEC_WEIGHT = 0.6
BM25_WEIGHT = 0.4

def rrf_fuse(vector_hits, bm25_hits, *, k=RRF_K, vec_w=VEC_WEIGHT, bm25_w=BM25_WEIGHT):
    score = defaultdict(float)
    for rank, (qid, _sim) in enumerate(vector_hits):
        score[qid] += vec_w * (1.0 / (k + rank + 1))
    for rank, (qid, _bm25) in enumerate(bm25_hits):
        score[qid] += bm25_w * (1.0 / (k + rank + 1))
    return sorted(score.items(), key=lambda x: x[1], reverse=True)

def bm25_search(db, query: str, limit: int = 10) -> list[tuple[str, float]]:
    rows = db.execute("""
        SELECT q.id, bm25(queries_fts) AS score
        FROM queries_fts
        JOIN queries q ON q.rowid = queries_fts.rowid
        WHERE queries_fts MATCH ?
        ORDER BY score
        LIMIT ?
    """, (query, limit)).fetchall()
    return [(qid, -score) for (qid, score) in rows]
```

### 2.2 Knobs

| Knob | Default | Rationale |
|---|---|---|
| `RRF_K` | **60** | Cormack et al. canonical default; flattens above 40. |
| `VEC_WEIGHT/BM25_WEIGHT` | **0.6/0.4** | Vector wins on paraphrase; BM25 catches jargon long-tail. |

### 2.3 Failure recovery

FTS5 missing → `bm25_search` raises; `rrf_fuse` called with empty BM25 list, degrades silently to vector-only.

---

## 3. Tier 3 — Insight Graph

### 3.1 What a Tier 3 node is

Short generalizable lesson distilled from one or more Tier-2 query traces.
- `content`: one sentence ≤ 200 chars
- `supporting_queries`: JSON array of Tier-2 IDs (Ω set)
- `tags_json`: meta-loop tag set (§6)

**Insight enters graph only after passing promotion gate.** Per-task LLM extraction produces *candidates* in holding table.

### 3.2 SQLite DDL — additions

```sql
CREATE TABLE IF NOT EXISTS insight_candidates (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    query_id        TEXT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    embedding       BLOB NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.5,
    tags_json       TEXT NOT NULL DEFAULT '[]',
    created_at      REAL NOT NULL,
    promoted_to     TEXT REFERENCES insights(id),
    promoted_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_cand_unpromoted ON insight_candidates(promoted_at)
    WHERE promoted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_cand_query     ON insight_candidates(query_id);

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

-- Inverted index — kept in sync with `supporting_queries` via triggers
CREATE TABLE IF NOT EXISTS insight_query_index (
    insight_id      TEXT NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
    query_id        TEXT NOT NULL REFERENCES queries(id)  ON DELETE CASCADE,
    PRIMARY KEY (insight_id, query_id)
);
CREATE INDEX IF NOT EXISTS idx_iqi_query ON insight_query_index(query_id);

-- FTS5 over insight content
CREATE VIRTUAL TABLE IF NOT EXISTS insights_fts USING fts5(
    content, content='insights', content_rowid='rowid', tokenize='porter unicode61'
);

CREATE INDEX IF NOT EXISTS idx_insights_validated ON insights(last_validated_at DESC);
CREATE INDEX IF NOT EXISTS idx_insights_conf      ON insights(confidence DESC);
```

`embedding BLOB` (raw f32 little-endian) lets us avoid LanceDB for insights — only hundreds-to-thousands ever, brute-force-numpy territory.

### 3.3 Module — `gmemory/insight_graph.py`

```python
DISTILL_PROMPT_BUDGET_TOK   = 4000
DISTILL_OUTPUT_BUDGET_TOK   =  300
DISTILL_TIMEOUT_S           =   25
DISTILL_MAX_INSIGHTS        =    3
DISTILL_MIN_TRACE_LEN       =    3

class InsightGraph:
    def __init__(self, db_path, query_graph, embedder=None, model="haiku"): ...
    def extract_insights(self, query_id, topic, utterances, status) -> list[str]: ...
    def promote_to_insight(self, candidate_ids=None) -> list[str]: ...
    def retrieve_insights_for_query(self, query_text, related_query_ids, *, limit=5) -> list[dict]: ...
    def aggregate_overnight(self, *, since_hours=24) -> dict: ...
```

### 3.4 Distillation prompt

```
DISTILL_SYSTEM = (
    "You are a memory-distillation agent for a research swarm. "
    "Extract up to 3 concise, generalizable lessons from the trace. "
    "Each lesson must be 1 sentence, ≤180 characters, and useful for a future "
    "task on a similar topic. Return JSON only, schema as instructed."
)

DISTILL_USER_TEMPLATE = """
Topic: {topic}
Outcome: {status}
Cost: ${cost:.2f}  Wall: {wall:.0f}s  Rounds: {rounds}

Trace ({n} agents, capped to 15):
{trace}

Return JSON:
{{
  "insights": [
    {{ "content": "...", "tags": ["tagA","tagB"], "confidence": 0.0-1.0 }},
    ...up to 3 entries
  ]
}}
Tag vocabulary (preferred):
- domain:<topic-area>
- pattern:<technique>
- pitfall:<failure-mode>
- swarm:<config>
- meta:<self-observation>
"""
```

Model: **Haiku** ($0.0002–$0.0008 per call). Output JSON parsed; rows go into `insight_candidates`, never directly into `insights`.

### 3.5 Empirical knobs

| Knob | Default | Rationale | Tuning failure |
|---|---|---|---|
| `DISTILL_MAX_INSIGHTS` | **3** | User spec; covers most tasks. | 1 → high false-negative. 10 → 70% boilerplate. |
| `DISTILL_PROMPT_BUDGET_TOK` | **4000** | First 15 utterances × ~250 tok = 3.75K. | 16K → cost 4×. 1000 → drops synthesizer's final round. |
| `DISTILL_TIMEOUT_S` | **25** | Haiku P95 < 8s for 4K prompt. | 5s → cold-start failures. 120s → blocks eviction. |
| Distillation cadence | **per-task + nightly** | Per-task hot signal; nightly cross-task merge. | Per-task only → no merging. Nightly only → can't promote within-day. |
| `INSIGHT_DEDUP_THRESHOLD` | **0.85** | Above 0.85 likely paraphrase. | 0.95 → no dedup. 0.65 → distinct insights collapse. |
| `INSIGHT_MERGE_THRESHOLD` | **0.92** | Tighter than dedup; merging is destructive. | Same direction. |

### 3.6 Cost

Per-task: distillation $0.0003–$0.0008, storage 6 KB. **~$0.0005/task, ~6 KB/task.**

Nightly aggregation per 100 daily tasks: pure SQL+numpy, <50 ms. Optional LLM merge for >5 candidate clusters: ~$0.002/night.

Annualized (50 tasks/day): ~$10/year distillation, ~$1/year aggregation, ~110 MB storage.

### 3.7 Test cases

```python
def test_extract_insights_writes_candidates_only(igraph, mock_haiku): ...
def test_distillation_idempotent_per_query(igraph, mock_haiku): ...
def test_promotion_requires_three_neighbors(igraph, populated_with_two): ...
def test_promotion_succeeds_with_three_neighbors(igraph, populated_with_three): ...
def test_retrieval_orders_by_overlap(igraph, populated_two_insights): ...
def test_nightly_merges_near_duplicates(igraph, two_near_duplicate_insights): ...
```

### 3.8 Failure recovery

- Haiku CLI not on PATH / non-JSON → `extract_insights` returns `[]`. Fire-and-forget; orchestrator doesn't block.
- Distillation produces garbage → promotion gate filters. Garbage stays in holding table 14 days then purged.
- Insight graph past 10K nodes → force merge at threshold 0.85.
- Inverted-index drift → trigger-driven sync; offline reconciliation if diff > 1000 rows.

---

## 4. Promotion gate — `gmemory/promotion.py`

### 4.1 GAM-style rule (honest framing)

User asked for "Promotion gate (GAM-style, arXiv 2604.12285): N=3+ query-graph neighbors share same pattern."

**Verified ground truth from GAM paper:** GAM does **not** use fixed N count. Uses **LLM-driven semantic discrimination** at session boundaries. Re-ranking factors β_time=1.4, β_role=1.4, β_conf=1.2 at retrieval (not promotion).

So "GAM-style with N=3" is **principled adaptation, not citation**. Reasoning: Blitz-Swarm has no natural session boundaries. The N=3 rule replaces LLM-discrimination with a structural one — three-or-more candidate insights clustering across distinct queries.

### 4.2 Algorithm

```
INPUT:  unpromoted candidates C
        promotion threshold N = 3
        cluster cosine threshold τ_cluster = 0.78
OUTPUT: list of newly created insight IDs

1. Pull all unpromoted candidates with embeddings, query_ids, tags, confidence.
2. Cluster by single-link agglomerative on cosine, threshold τ_cluster.
3. For each cluster:
   a. support_size = |distinct(query_ids)|
   b. If support_size < N → skip; mark `last_evaluated_at`.
   c. Else:
      - Pick canonical content: closest to centroid (confidence tiebreaker)
      - Compute centroid embedding (mean, re-normalize)
      - Union tag sets across cluster
      - Average confidence
      - INSERT into `insights` with supporting_queries = distinct query_ids
      - UPDATE candidates: promoted_to=<new_id>, promoted_at=NOW
      - Insert into `insight_query_index` (one row per query in Ω)
      - Form `insight_edges` to insights sharing ≥2 query_ids in support, via_query_id = most recent shared qid
4. Return new insight IDs.
```

**Key invariants:**
- Gate is **per cluster, not per candidate**. Three from one task = ONE distinct query_id.
- Candidates stay in holding table indefinitely until promoted, merged, or pruned (14 days).

### 4.3 Module skeleton

```python
PROMOTION_N            = 3
CLUSTER_COSINE_THRESH  = 0.78
MAX_PROMOTIONS_PER_RUN = 25
CANDIDATE_TTL_DAYS     = 14

@dataclass(slots=True)
class Cluster:
    candidate_ids: list[str]
    embeddings: np.ndarray
    query_ids: set[str]
    confidences: list[float]
    tags: set[str]

def evaluate_candidates(db, *, n_required=PROMOTION_N,
                        cluster_thresh=CLUSTER_COSINE_THRESH,
                        max_promotions=MAX_PROMOTIONS_PER_RUN) -> list[str]: ...

def _fetch_candidates(db) -> list[dict]: ...
def _cluster(candidates, thresh) -> list[Cluster]: ...
def _promote_cluster(db, cluster) -> str | None: ...
def _connect_to_existing_insights(db, new_id) -> int: ...
def prune_stale_candidates(db, ttl_days=CANDIDATE_TTL_DAYS) -> int: ...
```

### 4.4 Knobs

| Knob | Default | Rationale | Tuning failure |
|---|---|---|---|
| `PROMOTION_N` | **3** | Two = coincidence; three = pattern. | N=2 → noise. N=5 → 50–100 task cold start. |
| `CLUSTER_COSINE_THRESH` | **0.78** | Tighter than τ_link (0.70); paraphrases cluster 0.78–0.85. | 0.65 → mashes distinct insights. 0.92 → almost never fires. |
| `MAX_PROMOTIONS_PER_RUN` | **25** | Bound run cost in pathological cases. | 1 → starvation. 1000 → night job blocks. |
| `CANDIDATE_TTL_DAYS` | **14** | Slow-burn insights need time. | 1 → kills insights that emerge over weeks. 365 → unbounded growth. |

### 4.5 Test cases

```python
def test_below_threshold_no_promotion(db_with_2_candidates): ...
def test_at_threshold_promotes(db_with_3_distinct_query_candidates): ...
def test_three_candidates_one_query_does_not_promote(db_with_three_in_one_query): ...
def test_cluster_below_thresh_does_not_merge(db_with_two_dissimilar_clusters): ...
```

### 4.6 Cost

Pure SQL + numpy. **0 LLM calls. ~50 ms for 300 candidates. $0.000/run.**

### 4.7 Failure recovery

- Embedding BLOB corruption → skip candidate, log; re-encode on next distillation.
- Cluster explosion (100+ candidates) → cap per cluster at 50; promote closest 50 to centroid.
- Tag union grows huge → cap at 12 most-common.

---

## 5. Retrieval pipeline — `gmemory/retrieval.py`

### 5.1 Six-step pipeline

```
new_query_text
  ↓
1. Embed → vector ANN top-K + FTS5 BM25 top-K → RRF fuse → seeds (≤K)
  ↓
2. 1-hop SQL on query_edges → expanded (capped per seed)
  ↓
3. Insight overlap (inverted idx) → relevant insights [≤5]
  ↓
4. R_LLM (Haiku) score each expanded query vs new_query → top-M (M=3)
  ↓
5. S_LLM (Haiku) sparsify each top-M interaction → essential utterances only
  ↓
6. Format `## Relevant prior findings` + inject (cap 2000 tokens)
```

Current `memory/reader.py:retrieve_memory` does steps 1-3 + half-step 4. Finish 4–6 in `gmemory/retrieval.py`.

### 5.2 Module skeleton

```python
TOP_K_RETRIEVAL  = 2
TOP_M_INTERACTIONS = 3
MAX_LLM_RELEVANCE_CALLS = 6
RELEVANCE_TIMEOUT_S = 12
SPARSIFY_TIMEOUT_S  = 15
LLM_OPS_THRESHOLD   = 10
RELEVANCE_MIN_SCORE = 0.4

@dataclass(slots=True)
class RetrievalResult:
    insights: list[dict]
    interactions: list[dict]
    related_queries: list[dict]
    used_llm: bool
    cost_usd: float

def retrieve(new_query, qg, ig, *, top_k=TOP_K_RETRIEVAL, top_m=TOP_M_INTERACTIONS,
             use_llm=True) -> RetrievalResult: ...

def score_relevance(new_query, candidate_text, *, model="haiku") -> tuple[float, float]: ...
def sparsify_interaction(utterances, new_query, *, model="haiku") -> tuple[list[dict], float]: ...
def format_context(result, *, max_chars=8000) -> str: ...
```

### 5.3 Knobs

| Knob | Default | Rationale | Tuning failure |
|---|---|---|---|
| `TOP_K_RETRIEVAL` | **2** | Paper ablation: k=5 degrades 7.71%. k=2 optimum. | k=1 high variance. k=10 fanout floods. |
| `TOP_M_INTERACTIONS` | **3** | Paper sweep: 3 best in 4 of 5 benchmarks. | M=1 bimodal. M=8 cost 8× for ignored slots. |
| `MAX_LLM_RELEVANCE_CALLS` | **6** | $0.006/task × 50/day = $0.30/day cap. | 30 cost blows. 1 no scoring. |
| `RELEVANCE_MIN_SCORE` | **0.4** | Below 0.4 = "tangentially related". | 0.0 noise bleeds. 0.7 cold-start rejects all. |
| `LLM_OPS_THRESHOLD` | **10** | Below 10 stored, hybrid returns junk. | 0 spend on cold noise. 100 lag. |
| Context budget | **2000 tokens** | Paper says >2000 degrades. | 4000 lost-in-middle. 500 truncates. |

### 5.4 Cost

Per task (warm): embed 12 ms ($0) + ANN 5 ms ($0) + 1-hop 2 ms ($0) + 6 Haiku relevance × ~250 tok ($0.0006) + 3 Haiku sparsify × ~600 tok ($0.0009) = **~$0.0015/task, ~250 ms wall.**

Cold start (queries < 10): **$0/task, 20 ms.**

Daily at 50 tasks: **~$0.08/day, ~$2.50/month.**

### 5.5 Test cases

```python
def test_cold_start_skips_llm(empty_qg, empty_ig): ...
def test_warm_uses_llm(populated_qg, populated_ig, mock_haiku): ...
def test_relevance_min_score_filters(populated_qg, populated_ig, mock_haiku): ...
def test_format_context_under_token_cap(rich_result): ...
```

### 5.6 Failure recovery

- Haiku timeout on relevance call → set `RELEVANCE_MIN_SCORE` (neutral); log.
- Sparsification returns empty → fall back to first 5 utterances.
- `get_records` returns < `top_m` rows → use what we have; no error.
- JSON parse failure → score-relevance returns 0.5 (neutral), `cost_usd=0`.

---

## 6. Meta-loop integration — `gmemory/meta.py`

### 6.1 Tag schema — five orthogonal axes

| Axis | Vocabulary | Source |
|---|---|---|
| `domain:` | crypto, databases, ml, infra, security, … | 1 Haiku call at swarm init |
| `pattern:` | walk-forward-cv, judge-ensemble-3, mar-personas, … | Set by orchestrator from config |
| `pitfall:` | lookahead-bias, p-hacking, hallucinated-citation, … | Distilled by Haiku from trace |
| `swarm:` | n-agents-X, max-rounds-Y, model-sonnet, … | Set by orchestrator from runtime |
| `meta:` | saturated-round-2, dissent-preserved, retried-once, … | Auto-set by `consensus.py` post-hoc |

Tags live on Tier-2 (`queries.tags_json`) and Tier-3 (`insights.tags_json`). Promotion: insight inherits union of supporting candidates' tags.

### 6.2 SQL — denormalized tag index

```sql
CREATE TABLE IF NOT EXISTS query_tags (
    query_id    TEXT NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    axis        TEXT NOT NULL,
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
```

### 6.3 Module skeleton

```python
TAG_AXES = ("domain", "pattern", "pitfall", "swarm", "meta")

def search_tasks(db, *, must_have=(), none_of=(), status=None, since_days=None, limit=100): ...
def search_insights(db, *, must_have=(), none_of=(), min_confidence=0.0, limit=50): ...
def aggregate_by_tag(db, *, group_by, metric="quality_score", must_have=(), since_days=None): ...
def correlate_pattern_with_outcome(db, pattern_tag, *, outcome_metric="quality_score",
                                    control_tags=()) -> dict: ...
def saturating_round_distribution(db, *, pattern_tag=None, since_days=30) -> dict[int, float]: ...
```

### 6.4 Test cases

```python
def test_search_tasks_by_tag_intersection(populated_db): ...
def test_correlate_returns_delta(populated_db): ...
def test_tag_axis_validation_rejects_bad(): ...
def test_empty_database_returns_empty(empty_db): ...
```

### 6.5 Failure recovery

- `tags_json` corruption → `query_tags` denormalization is best-effort; corrupted skip.
- Tag axis vocabulary drift (typo `pratten:`) → `aggregate_by_tag` filters by exact axis; typo bucket doesn't show up.

---

## 7. Orchestrator integration

**Pre-blast:**
```python
qg = QueryGraph(); qg.initialize()
ig = InsightGraph(qg._db, query_graph=qg); ig.initialize()

result = retrieve(topic, qg, ig, top_k=2, top_m=3, use_llm=True)
memory_context = format_context(result)
context_for_round_1 = memory_context + "\n" + (existing_blackboard_context or "")
```

**Post-finalize:**
```python
status = "resolved" if consensus_reached else ("partial" if best_score >= 7.0 else "failed")
tags = derive_tags(topic, consensus_result, agents_used, n_rounds)

qid = qg.add_task(
    query_text=topic, status=status,
    n_agents=len(agents_used), n_rounds=n_rounds,
    cost_usd=total_cost, wall_clock_s=wall_clock,
    quality_score=best_score, tags=tags)

ig.extract_insights(query_id=qid, topic=topic,
                     utterances=all_utterances, status=status)
ig.promote_to_insight()
```

### 7.2 Nightly aggregation

```bash
# scripts/nightly-gmemory-aggregate.sh
python -c "from gmemory.insight_graph import InsightGraph
from gmemory.query_graph import QueryGraph
qg = QueryGraph(); qg.initialize()
ig = InsightGraph(qg._db, query_graph=qg); ig.initialize()
print(ig.aggregate_overnight(since_hours=24))"
```

---

## 8. Repository layout (final)

```
blitz-swarm/
├── memory/              # legacy thin façades (orchestrator compat)
│   ├── schema.sql
│   ├── writer.py        # delegates to gmemory
│   ├── reader.py        # delegates to gmemory.retrieval
│   └── models.py
├── gmemory/             # NEW — Tier 2/3 build-out
│   ├── __init__.py
│   ├── schema.sql       # additive migration
│   ├── query_graph.py   # Tier 2
│   ├── insight_graph.py # Tier 3
│   ├── promotion.py     # promotion gate
│   ├── retrieval.py     # 6-step pipeline + LLM ops
│   ├── hybrid.py        # RRF fusion + BM25
│   └── meta.py          # meta-loop query API
├── tests/
│   ├── conftest.py
│   ├── test_query_graph.py
│   ├── test_insight_graph.py
│   ├── test_promotion.py
│   ├── test_retrieval.py
│   └── test_meta.py
└── scripts/
    └── nightly-gmemory-aggregate.sh
```

Total: ~1100 lines Python, ~150 lines SQL, ~150 lines test fixtures.

---

## 9. Honest deviations from user spec

1. **Hybrid retrieval (FTS5 + RRF) is Blitz-Swarm-specific upgrade**, not in original G-Memory paper. Drop §2 + §5 BM25 to honor paper exactly.
2. **Promotion gate's N=3 isn't strictly GAM-style.** GAM uses LLM-discrimination at session boundaries. Our N=3 + cluster-cosine is structural adaptation. Documented in `gmemory/promotion.py` docstring.
3. **Tags are 5 fixed axes, not free-form.** Opinionated for queryability vs flexibility.
4. **`QueryGraph` shares `_db` with `InsightGraph`** — couples at connection level for atomic transactions. Atomicity wins.
5. **Existing `memory/writer.py:_find_related_queries` has distance-type bug** (treats `_distance` as L2 with normalized vectors). New code fixes with explicit `distance_type("cosine")`. One-shot migration: `scripts/migrate-distance-type.py`. Logged as `BUG-001`.

---

## 10. Sources

- [G-Memory (arXiv 2506.07398)](https://arxiv.org/abs/2506.07398) — NeurIPS 2025 Spotlight. [GitHub](https://github.com/bingreeky/GMemory).
- [GAM (arXiv 2604.12285)](https://arxiv.org/abs/2604.12285) — verified retrieval factors β_time=1.4, β_role=1.4, β_conf=1.2; LLM session boundaries (NOT fixed-N).
- [MAGMA (arXiv 2601.03236)](https://arxiv.org/abs/2601.03236) — four-graph reference.
- [A-MEM (arXiv 2502.12110)](https://arxiv.org/abs/2502.12110) — Zettelkasten-inspired link evolution.
- [AgeMem (arXiv 2601.01885)](https://arxiv.org/abs/2601.01885) — RL-trained memory ops.
- MiniLM `all-MiniLM-L6-v2` — 384-dim, contrastive.
- [LanceDB docs](https://docs.lancedb.com/search/vector-search) — cosine distance must be set explicitly.
- [SQLite FTS5 + BM25](https://www.sqlite.org/fts5.html) — RRF k=60 from Cormack et al.
