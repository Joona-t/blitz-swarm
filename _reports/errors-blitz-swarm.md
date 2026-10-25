# error-scanner Report: Blitz Swarm

_Agent: error-scanner | Reporting to: Hamilton (Reliability Lead) | Date: 2026-03-12_

## Summary
27 findings: 7 critical, 12 warnings, 8 info. Highest risk: memory/writer.py (unhandled exceptions in transactions) and Redis blackboard (zero error handling on methods after init).

## Critical
- **CRIT-01**: save_output has no error handling — disk error after full swarm run loses everything
- **CRIT-02**: _init_sqlite has no error handling — missing schema.sql crashes, leaks LanceDB handles
- **CRIT-03**: SQLite transaction swallows inner exceptions — embedder crash rolls back all 3 tiers
- **CRIT-04**: scan_iter in cleanup not atomic; Redis restart mid-scan leaves stale data
- **CRIT-05**: write_agent_output has no error handling — Redis drop mid-round crashes the loop
- **CRIT-06**: Malformed agent output (findings=42) causes TypeError in build_context
- **CRIT-07**: _merge_insight silently drops if insight_id is stale

## Warnings
- WARN-01-03: Silent exception swallowing in _find_related_queries, _store_query_vector, _embedding_search
- WARN-04: TOML parse failure silently swallowed in config.py
- WARN-05: Redis consumer loop retries forever with no backoff
- WARN-06: _split_subtopics_llm doesn't validate list type — string sliced into chars
- WARN-08: _update_access_counts unguarded — SQLite lock crashes entire swarm before round 1
- WARN-11: Broad except catches KeyboardInterrupt — Ctrl+C ineffective during blast

## Crash Scenarios Documented
A: Redis drops mid-round → unhandled ConnectionError → output lost
B: Embedder crash during persist → full rollback + crash → output lost
C: Malformed agent output → TypeError in build_context → crash mid-round
D: SQLite locked during access count update → crash before round 1
