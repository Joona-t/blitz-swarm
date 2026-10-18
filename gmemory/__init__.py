"""G-Memory — the hierarchical memory of the swarm (the only memory package).

  Tier 1 — interaction traces -> raw per-round agent outputs + edges
  Tier 2 — query graph        -> task-level neighbor expansion (vector or BM25)
  Tier 3 — insight graph      -> distilled cross-task generalizations, N=3 gate

`gmemory.store.MemoryStore` is the orchestrator-facing surface (open /
retrieve / persist / evict). Runs on stdlib sqlite3 + FTS5; the embedder,
LanceDB and the `memory_ops` LLM hooks are optional extras. The legacy
memory/ package (v0.1-v0.2) is gone; `gmemory.migrate` upgrades its DBs.

Anchors:
  - G-Memory: Zhang et al. arXiv 2506.07398 (NeurIPS 2025 Spotlight)
  - GAM:      arXiv 2604.12285 (Apr 2026, hierarchical promotion)
  - A-MEM:    Xu et al. arXiv 2502.12110 (NeurIPS 2025)
  - MAGMA:    arXiv 2601.03236 (multi-graph reference for v0.3)

Modules:
  store.py         MemoryStore — open/migrate, retrieve, persist, eviction cycle
  hybrid.py        RRF fusion of vector + BM25 retrieval; FTS5 MATCH escaping
  promotion.py     GAM-style structural promotion gate (N=3 distinct queries)
  query_graph.py   Tier 2 — task nodes + kNN / lexical edges
  insight_graph.py Tier 3 — candidates + promoted insights with hyperedges to Ω
  retrieval.py     6-step pipeline (seeds -> 1-hop -> insights -> rank -> traces -> fence)
  llm_ops.py       memory_ops-tier LLM hooks (distill / relevance / sparsify)
  migrate.py       legacy memory/ schema -> canonical schema (idempotent)
  meta.py          Meta-loop tag API (queries the recursion loop calls)
  schema.sql       Canonical SQLite schema

LLM hooks live behind injection — every callable that would normally
shell to Claude is overrideable so the test suite can run scipy-free
and LLM-free.
"""
