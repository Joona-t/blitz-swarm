# dead-code-hunter Report: Blitz Swarm

_Agent: dead-code-hunter | Reporting to: Matz (Code Quality Lead) | Date: 2026-03-12_

## Summary
No orphaned files. Significant dead code in 5 categories. Clean — no commented-out code blocks.

## Critical Dead Code
- **config.py** — entire module never imported. load_config() never called. blitz.toml has zero runtime effect.

## Dead Functions (never called)
1. `Embedder.encode_batch()` — embedder.py:47
2. `Blackboard.build_context_string()` — blackboard.py:161
3. `Blackboard.read_all_outputs()` — blackboard.py:130
4. `Blackboard.get_agent_status()` — blackboard.py:153
5. `Blackboard.consume_memory_writes()` — blackboard.py:185
6. `MemoryReader.score_relevance()` — reader.py:283
7. `MemoryReader.sparsify_interaction()` — reader.py:318
8. `MemoryReader.query_count()` — reader.py:73
9. `MemoryReader.close()` — reader.py:416
10. `MemoryWriter.generate_insight_llm()` — writer.py:321
11. `MemoryWriter.run_eviction_cycle()` — writer.py:399
12. `MemoryWriter.maybe_run_eviction()` — writer.py:481
13. `MemoryWriter.stats` — writer.py:490
14. `_split_subtopics_llm()` — agents.py:156

## Unused Imports
- `AGENT_OUTPUT_SCHEMA_JSON` in orchestrator.py
- `TaskStatus` in memory/writer.py
- `Embedder` class in memory/reader.py
- `import sys` in orchestrator.py

## Unused Constants
- `PLANNING_SCHEMA` in agents.py:217
- `BlitzAgent.max_iterations` field — never read

## Dead Parameters
- `--verbose` flag — parsed but never checked
