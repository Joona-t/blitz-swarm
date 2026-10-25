# pattern-miner Report: Blitz Swarm

_Agent: pattern-miner | Reporting to: Karpathy (Intelligence Lead) | Date: 2026-03-12_

## Summary
Architecture is coherent and well-reasoned. Four significant structural problems.

## Consistent Patterns (Good)
- Dataclass-first data modeling throughout
- Fail-soft optional subsystems (Redis, LanceDB, sentence-transformers)
- Consistent subprocess invocation convention
- Snake_case naming throughout
- INSERT OR IGNORE for idempotency
- WAL mode + busy_timeout on all SQLite connections
- Singleton pattern for expensive resources (embedder)

## Inconsistent Patterns (Problems)
1. **Context building duplicated** — orchestrator.py:build_context() and blackboard.py:_format_context() are near-identical; blackboard version is dead code
2. **Constants defined 3-5 times** — MAX_CONTEXT_TOKENS in 3 files + blitz.toml + config.py
3. **Config loaded but never consumed** — config.py is fully implemented but disconnected
4. **--verbose parsed but never wired**
5. **aiosqlite declared but never used**
6. **Inconsistent import placement** — uuid imported inside loop
7. **BlitzAgent.max_iterations never read**
8. **Round numbering off-by-one risk** between orchestrator and blackboard
9. **Dead Redis memory write path** — queue_memory_write enqueues but start() daemon never launched
10. **Synthesizer outputs stored as Tier 1 utterances** — may inflate interaction graph

## Config Gap Table
Every config value in blitz.toml is hardcoded separately in the consuming files. Zero values read from config at runtime.
