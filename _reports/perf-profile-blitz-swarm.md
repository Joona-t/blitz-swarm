# perf-profiler Report: Blitz Swarm

_Agent: perf-profiler | Reporting to: Carmack (Performance Lead) | Date: 2026-03-12_

## Summary
Dominant cost is subprocess overhead (claude CLI). Three meaningful scaling risks identified.

## Critical
- **CRIT-01**: O(N²) insight dedup — re-encodes ALL stored insights on every new task. Pure-Python cosine similarity (384-dim). Will break at scale.
- **CRIT-02**: ThreadPoolExecutor ceiling (min(32, cpu+4)) matches max_agents=12 on 8-core; 4-core hosts serialize agents.

## Warnings
- **WARN-01**: `accumulated_outputs` rebuilt from scratch each round — O(R²*A) copies
- **WARN-02**: `build_context` called per-agent with identical results for same-role agents
- **WARN-03**: `import uuid` inside nested loop
- **WARN-05**: L2→cosine conversion incorrect (`1-dist` should be `1-dist²/2`)
- **WARN-06**: `SELECT * FROM insights` unbounded full-table scan on every retrieval
- **WARN-07**: MemoryReader issues writes on conceptually read-only connection
- **WARN-08**: Two sequential subprocess calls at startup (planning) — 60s worst case

## Recommendations
1. Cache insight embeddings in DB; use numpy for cosine
2. Fix L2-to-cosine formula
3. Use `extend()` instead of full rebuild
4. Deduplicate build_context calls for same-role agents
5. Move `import uuid` to file top
