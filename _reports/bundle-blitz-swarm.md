# bundle-analyzer Report: Blitz Swarm

_Agent: bundle-analyzer | Reporting to: Carmack (Performance Lead) | Date: 2026-03-12_

## Summary
9 Python files, ~2,273 lines. Lean codebase. Primary weight concern is transitive deps (sentence-transformers pulls PyTorch ~2GB).

## File Inventory
| File | Lines |
|------|-------|
| orchestrator.py | 734 |
| memory/writer.py | 502 |
| memory/reader.py | 420 |
| agents.py | 364 |
| blackboard.py | 259 |
| memory/models.py | 123 |
| config.py | 102 |
| consensus.py | 93 |
| embedder.py | 75 |

## Key Findings
- `aiosqlite` declared but never imported — dead dependency
- `config.py` entirely unimported — dead module
- `import sys` in orchestrator.py unused
- `AGENT_OUTPUT_SCHEMA_JSON` imported but unused in orchestrator
- `TaskStatus` imported but unused in writer.py
- `Embedder` class imported but unused in reader.py (only `get_embedder` used)
- `PLANNING_SCHEMA` defined but never referenced in agents.py
- Context-building duplicated between orchestrator.py and blackboard.py
- Most memory/models.py dataclasses are never instantiated
- sentence-transformers + lancedb already soft-imported — should be optional deps
