# Karpathy Domain Verdict: Blitz Swarm Intelligence Features

_Lead: Karpathy (Smart Features & Intelligence) | Date: 2026-03-12_
_Sources: pattern-miner report, docs-generator report, full source review of all .py files_

---

## 1. Domain Assessment

The intelligent subsystems in Blitz Swarm -- G-Memory, consensus logic, agent planning, and the embedding pipeline -- are **architecturally sound and well-researched**. The G-Memory implementation faithfully follows the NeurIPS 2025 paper's three-tier hierarchy (interaction graph, query graph, insight graph) with correct retrieval semantics (top-k=2, 1-hop expansion, upward/downward traversal). The consensus system is clean and minimal. The embedding pipeline makes smart choices (MiniLM-L6-v2 for speed, ONNX fallback, singleton pattern, normalize_embeddings=True for cosine search).

However, there is a systemic problem: **the intelligence features are 70% built and 30% disconnected**. Three LLM-powered memory operations exist as fully-implemented methods but are never called. The config system is complete but no runtime code reads from it. This creates a deceptive surface -- the code looks more capable than it actually is at runtime.

Overall intelligence/ML health: **B+**. The design decisions are excellent. The execution stopped one integration pass short of completion.

---

## 2. Confirmed Issues

I verify the following findings from both specialists after reading every source file:

### From pattern-miner:

1. **Context building duplication -- CONFIRMED.** `orchestrator.py:build_context()` (lines 211-270) and `blackboard.py:_format_context()` (lines 209-258) are near-identical implementations. The blackboard version is dead code -- `Blackboard.build_context_string()` is never called during any run. The orchestrator version is the sole active path.

2. **MAX_CONTEXT_TOKENS defined in 3 files + blitz.toml + config.py -- CONFIRMED.** Identical `MAX_CONTEXT_TOKENS = 2000` and `MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * 4` appear in `orchestrator.py:38-39`, `blackboard.py:24-25`, and `memory/reader.py:25-26`. The same value exists in `blitz.toml` and `config.py:MemoryConfig`. Zero of these read from config at runtime.

3. **config.py fully implemented but disconnected -- CONFIRMED.** `load_config()` is never called anywhere. Every constant it would provide is hardcoded at the point of use. This is the single highest-impact structural deficiency: wiring config would fix issues #2, #4, and several others simultaneously.

4. **--verbose parsed but never wired -- CONFIRMED.** `args.verbose` is captured by argparse (line 712) but never passed to `run_swarm()` or read anywhere.

5. **aiosqlite declared but never used -- CONFIRMED.** Listed in `pyproject.toml` (line 11) and README install command. All SQLite access uses synchronous `sqlite3`. No import of `aiosqlite` exists in any `.py` file.

6. **uuid imported inside loop -- CONFIRMED.** `orchestrator.py:382` has `import uuid` inside the `_persist_to_memory` function's nested loop. This works (Python caches module imports), but it is a code smell and a performance micro-pessimization on every iteration.

7. **BlitzAgent.max_iterations never read -- CONFIRMED.** The field exists on the dataclass (line 64 of `agents.py`) but no code ever inspects it. Global `max_rounds` controls iteration for all agents uniformly.

8. **Dead Redis memory write path -- CONFIRMED.** `blackboard.py:queue_memory_write()` enqueues to the `memory:writes` stream. `MemoryWriter.start()` would consume it. But `start()` is never launched as a daemon anywhere in the codebase. The orchestrator calls `mem_writer.store_task()` directly (the bypass path), making the entire Redis stream consumer dead code.

9. **Synthesizer outputs stored as Tier 1 utterances -- CONFIRMED.** In `_persist_to_memory()`, the synthesis round is appended to `all_round_outputs` (line 626) and all its outputs become utterances. The synthesizer's consolidated output is thus stored alongside raw researcher utterances in Tier 1, which inflates the interaction graph with derived content. This can distort retrieval: future 1-hop expansion may surface the synthesizer's summary instead of the primary research.

### From docs-generator:

10. **7-step lifecycle in spec vs 6-step in code -- CONFIRMED.** BLITZ-SWARM.md describes CONSOLIDATE as a separate step; in code, the synthesizer handles both consolidation and writing in one pass within FINALIZE.

11. **Round 1 is two-phase, not truly parallel -- CONFIRMED.** `orchestrator.py:511-534` shows round 1 blasts researchers first (Phase A), then evaluators second (Phase B) with researcher context. This is undocumented and differs from the spec's implication that all agents blast simultaneously.

12. **LLM memory ops (R_LLM, S_LLM) defined but never called -- CONFIRMED.** `MemoryReader.score_relevance()` (line 283) and `MemoryReader.sparsify_interaction()` (line 318) are fully implemented but have zero callers. Similarly, `MemoryWriter.generate_insight_llm()` (line 321) is never called -- the orchestrator uses key_points concatenation as a heuristic insight instead (lines 404-407).

---

## 3. Disputed Issues

### 3a. "Round numbering off-by-one risk" (pattern-miner #8) -- DISPUTED, NOT A BUG

The orchestrator uses `range(1, max_rounds + 1)` for round numbering and `Blackboard.advance_round()` uses `HINCRBY 1` starting from 0. Both start at round 1. The blackboard `advance_round()` is called at the top of each loop iteration before the blast. The orchestrator's `round_n` and the blackboard's `current_round` will be in sync. I traced the full flow: no off-by-one exists. The pattern-miner flagged a risk that is not realized in practice.

### 3b. "Minimum 5 agents achievable, not 4 as spec states" (docs-generator #5) -- PARTIALLY DISPUTED

The heuristic fallback creates: 2 researchers + 1 critic + 1 fact_checker + 1 quality_judge + 1 synthesizer = 6 agents minimum, not 5. The docs-generator's count of 5 is wrong. The spec's "minimum 4" is also wrong. The true minimum with the default plan is 6. With LLM planning, the minimum is still 2 researchers + 1 critic + 0 fact_checkers + 1 quality_judge + 1 synthesizer = 5 (when `needs_fact_checker=false`). So the minimum is 5 with LLM planning, 6 without.

---

## 4. Priority Fix List (ordered by impact)

### P1. Wire config.py into the runtime

**Impact: Fixes 5+ issues simultaneously.** The config system is complete and well-structured. Adding `config = load_config()` at the top of `orchestrator.py` and passing values to `MemoryWriter`, `MemoryReader`, `Blackboard`, and agent planning would eliminate every hardcoded constant duplication. This is a 30-minute integration task that resolves issues #2, #3, partially #4, and makes the system actually configurable.

**Specific wiring needed:**
- `orchestrator.py`: `MAX_CONTEXT_TOKENS`, `DEFAULT_MAX_ROUNDS`, `AGENT_TIMEOUT_SECONDS` from `config.swarm`
- `memory/writer.py`: `INSIGHT_DEDUP_THRESHOLD`, `QUERY_LINK_THRESHOLD` from `config.memory`
- `memory/reader.py`: `TOP_K`, `MAX_CONTEXT_TOKENS` from `config.memory`
- `blackboard.py`: Redis connection params from `config.redis`

### P2. Connect LLM memory operations or remove them

Three fully-implemented LLM methods sit uncalled: `score_relevance()`, `sparsify_interaction()`, `generate_insight_llm()`. The `llm_ops_threshold = 10` config value implies these should activate after 10 queries. Either wire them into `retrieve_memory()` and `_process_task_complete()` with the threshold check, or move them to a `_future/` directory. Dead intelligence code that looks active is worse than missing code -- it misleads anyone reading the system.

### P3. Eliminate context-building duplication

Delete `blackboard.py:_format_context()` and `Blackboard.build_context_string()`. The orchestrator's `build_context()` is the active path and has identical logic. If blackboard-based context building is needed later, it should call the orchestrator's function, not maintain a parallel copy.

### P4. Fix synthesizer utterance storage pollution

The synthesizer's output should not be stored as Tier 1 utterances alongside raw agent interactions. The synthesizer is a derived view, not a primary interaction. Either: (a) tag synthesizer utterances with a `synthetic=True` flag and filter them during retrieval, or (b) exclude the synthesis round from the utterance list passed to `store_task()`. Option (b) is simpler and more correct -- the synthesizer output is already captured as the insight.

### P5. Remove dead dependencies and dead fields

- Remove `aiosqlite` from `pyproject.toml` and README
- Either implement `--verbose` (add a `verbose` parameter to `run_swarm()` that controls print granularity) or remove the flag
- Either use `BlitzAgent.max_iterations` for per-agent iteration caps or remove the field
- Move `import uuid` in `orchestrator.py:382` to the file-level imports

---

## 5. Verdict

### PASS WITH CONDITIONS

The intelligence architecture is well-designed. The G-Memory implementation follows the paper correctly. The consensus system is clean and minimal. The embedding pipeline makes pragmatic choices. The data models are well-structured.

**Conditions for unconditional PASS:**

1. Wire `config.py` into runtime (P1) -- eliminates the systemic constant-duplication problem
2. Either connect or clearly mark as future the three dead LLM memory methods (P2) -- the current state is misleading
3. Fix synthesizer pollution of Tier 1 memory (P4) -- this will silently corrupt retrieval quality as memory accumulates

P3 and P5 are cleanup items that should be done but are not blocking.

The core intelligence design is strong. The gap is purely in the last-mile integration -- the pieces are all built, they just need to be plugged together.
