"""G-Memory Tier 2/3 build-out — Phase 2 of the v0.2 frontier upgrade.

Hierarchical memory for the parallel multi-agent swarm. Builds on top
of the legacy memory/ package (Tier 1: interaction traces) by adding:

  Tier 2 — query graph     -> task-level semantic neighbor expansion
  Tier 3 — insight graph   -> LLM-distilled cross-task generalizations

Anchors:
  - G-Memory: Zhang et al. arXiv 2506.07398 (NeurIPS 2025 Spotlight)
  - GAM:      arXiv 2604.12285 (Apr 2026, hierarchical promotion)
  - A-MEM:    Xu et al. arXiv 2502.12110 (NeurIPS 2025)
  - MAGMA:    arXiv 2601.03236 (multi-graph reference for v0.3)

Modules:
  hybrid.py        RRF fusion of vector + BM25 retrieval
  promotion.py     GAM-style structural promotion gate (N=3 distinct queries)
  query_graph.py   Tier 2 — task nodes + kNN edges
  insight_graph.py Tier 3 — promoted insights with hyperedges to Ω
  retrieval.py     6-step pipeline (embed -> kNN -> 1-hop -> rank -> sparsify)
  meta.py          Meta-loop tag API (queries the recursion loop calls)
  schema.sql       Additive SQLite migration

LLM hooks live behind injection — every callable that would normally
shell to Claude is overrideable so the test suite can run scipy-free
and LLM-free.
"""
