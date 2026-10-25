# docs-generator Report: Blitz Swarm

**Agent:** docs-generator
**Reports to:** Karpathy (Smart Features & Intelligence Lead)
**Scan date:** 2026-03-12
**Files read:** README.md, BLITZ-SWARM.md, orchestrator.py, agents.py, consensus.py, blackboard.py, embedder.py, config.py, memory/writer.py, memory/reader.py, memory/models.py, memory/schema.sql, blitz.toml, pyproject.toml

---

## Summary

The Blitz Swarm codebase is well-structured and internally consistent. Module-level docstrings are present on every file, and key public functions are documented. The primary documentation issues fall into three categories:

1. **README inaccuracies** — The README describes a simpler, single-phase blast model. The actual orchestrator runs a two-phase Round 1 (researchers first, then evaluators) followed by true simultaneous blasts in subsequent rounds. The lifecycle diagram and "Agent Roles" counts are partially wrong.
2. **Spec drift in BLITZ-SWARM.md** — The spec's agent count table (Researcher: 2–4, Critic: 1–2) conflicts with code constraints (Researcher: 2–6, Critic: 1–3). The lifecycle diagram is also missing a step present in both docs and code (CONSOLIDATE is absent from the README, and the code's two-phase round behavior is undocumented in both).
3. **Missing documentation** — Several areas critical for contributors are completely undocumented: the two-phase round-1 behavior, the `build_context` / `_format_context` duplication, the `--verbose` flag (declared but unimplemented), and config file integration (neither the README nor BLITZ-SWARM.md mentions that `blitz.toml` values are not actually wired into the runtime).

Overall docstring quality is high. The codebase reads cleanly and most docstrings are accurate.

---

## README Accuracy Audit

### CLI Flags — ACCURATE
All five flags documented in the README (`--max-rounds`, `--no-redis`, `--no-llm-plan`, `--dry-run`, `--verbose`) are present in `orchestrator.py`'s `argparse` setup. Default for `--max-rounds` is `5` — matches documentation.

**Exception — `--verbose` is a no-op.** The flag is registered in argparse but `args.verbose` is never read or passed to `run_swarm()`. The README documents it as "Detailed logging" but it has no effect.

### Quickstart Commands — ACCURATE
All four example commands are syntactically correct and will dispatch to real code paths. Output path pattern `output/{topic}_{timestamp}.md` is accurate (see `save_output()` in orchestrator.py).

### Orchestrator Lifecycle Diagram — INACCURATE (missing step, missing phase split)
README documents 6 steps: SPAWN, BLAST, WRITE, CHECK, ITERATE, FINALIZE.
BLITZ-SWARM.md documents 7 steps: adds CONSOLIDATE (step 4) between WRITE and CHECK.
The actual code in `run_swarm()` does: SPAWN → Phase A blast (researchers only in round 1) → Phase B blast (evaluators only in round 1) → simultaneous blast in rounds 2+ → WRITE → CHECK → ITERATE → FINALIZE (synthesizer).

Neither document describes the **two-phase Round 1 split**. This is a meaningful architectural behavior: Round 1 is explicitly sequential (researchers then evaluators), while all subsequent rounds are fully simultaneous. The README's claim "All agents invoked simultaneously via asyncio" is only true from round 2 onward.

### Agent Roles Table — INACCURATE (counts)
README table:

| Role | Count |
|------|-------|
| Researcher | 2-6 |
| Critic | 1-3 |
| Fact-Checker | 0-1 |
| Quality Judge | 1 |
| Synthesizer | 1 |

This matches the code constraints in `agents.py` (`PLANNING_SCHEMA` min/max values). However, the text above the table says "Agent count is determined dynamically by an LLM analyzing the topic's complexity and breadth" — accurate. No inaccuracy here on the README side; the discrepancy is in BLITZ-SWARM.md (see below).

### Memory (G-Memory) Section — ACCURATE
The three-tier description (Interaction Graph, Query Graph, Insight Graph) matches the implementation in `memory/models.py` and `memory/schema.sql`. The arXiv reference (2506.07398) is present in `schema.sql` and reader.py docstring, confirming the research attribution is intentional.

Storage stack description (SQLite + LanceDB + Redis) matches `memory/writer.py` and `memory/reader.py`. The statement "Loaded once in orchestrator, never in agents" is accurate — `get_embedder()` is a singleton and agents are stateless subprocess invocations.

### Architecture File Tree — ACCURATE
All listed files exist. No files are listed that don't exist. The `memory/` subdirectory listing is correct.

### Configuration Section — INCOMPLETE
The README `blitz.toml` example is accurate and matches the actual `blitz.toml`. However, a critical omission: **`blitz.toml` is loaded by `config.py` but the resulting `BlitzConfig` object is never imported or used by `orchestrator.py`, `agents.py`, `blackboard.py`, or any other runtime module.** The configuration system is fully implemented but fully disconnected. Runtime behavior always uses hardcoded constants (e.g., `DEFAULT_MAX_ROUNDS = 5` in orchestrator.py, `AGENT_TIMEOUT_SECONDS = 180`, `MAX_CONTEXT_TOKENS = 2000`). The README implies editing `blitz.toml` changes runtime behavior — this is currently false.

### Requirements — PARTIALLY INACCURATE
README lists: `pip install redis aiosqlite lancedb sentence-transformers`

`pyproject.toml` lists: `redis>=5.0`, `aiosqlite>=0.20`, `lancedb>=0.6`, `sentence-transformers>=3.0`

The `aiosqlite` dependency is in `pyproject.toml` but the code uses synchronous `sqlite3` throughout — `aiosqlite` is never imported. The `pip install` command in the README omits version pins but is otherwise consistent.

`tomllib` (used in config.py) is stdlib in Python 3.11+, so no install needed. Not a concern.

---

## Spec Drift (BLITZ-SWARM.md vs Code)

### Agent Count Ranges — DIVERGED
BLITZ-SWARM.md spec table:

| Role | Count |
|------|-------|
| Researcher | 2–4 |
| Critic | 1–2 |

Code (`agents.py`, `PLANNING_SCHEMA`):

| Role | min | max |
|------|-----|-----|
| Researcher | 2 | 6 |
| Critic | 1 | 3 |

The spec was not updated when the code was written with wider ranges. The typical spawn count claim ("6–8 agents for a standard technical topic") and "up to 12 agents" for broad topics is consistent with the code's schema bounds (2–6 researchers + 1–3 critics + 0–1 fact_checker + 1 judge + 1 synthesizer = up to 12). The max-12 claim checks out. The per-role ranges in the spec table do not.

### Lifecycle Diagram — DIVERGED (BLITZ-SWARM.md has 7 steps, code has 8 distinct phases)
BLITZ-SWARM.md step 4 "CONSOLIDATE — MemoryWriter commits to SQLite + LanceDB, updates G-Memory tiers" does not match the code. In the current implementation, memory consolidation happens at FINALIZE (after the synthesizer), not between WRITE and CHECK. The spec implies per-round consolidation; the code does end-of-task consolidation.

### Invocation Pattern — DIVERGED
The BLITZ-SWARM.md pseudocode shows `invoke_agent` using `subprocess.run(["claude", "-p", prompt])` with a single combined prompt. The actual implementation uses separate `--system-prompt` and `-p` (user prompt) flags, plus `--output-format json` and `--model` and `--dangerously-skip-permissions`. The spec pseudocode is simplified to the point of being misleading for someone trying to understand the actual invocation contract.

### Consensus Algorithm — ACCURATE
The pseudocode in BLITZ-SWARM.md matches `check_consensus()` in `consensus.py` exactly (voters filtered by `quality_vote is not None`, unanimous "ready" required). The holdout override condition (one holdout after 3+ rounds) also matches `should_override_holdout()`.

### Retrieval Pipeline Step Count — DIVERGED (minor)
BLITZ-SWARM.md documents a 6-step retrieval pipeline. `memory/reader.py` docstring documents 4 steps (embedding search, 1-hop expand, upward traversal, downward traversal). The spec's steps 4–6 (LLM relevance scoring, LLM sparsification, inject context) are implemented as separate methods (`score_relevance`, `sparsify_interaction`, `build_memory_context`) but are NOT called from `retrieve_memory()`. They exist but are currently unused by the main pipeline.

### `BlitzAgent.max_iterations` Field — PRESENT IN SPEC, UNUSED IN CODE
BLITZ-SWARM.md documents `max_iterations: int = 3` in the `BlitzAgent` schema. The dataclass field exists in `agents.py`. However, it is never read or enforced anywhere in the orchestrator. The global `max_rounds` controls iteration, not per-agent `max_iterations`.

### `config.py` and `blitz.toml` — NOT MENTIONED IN BLITZ-SWARM.md
The spec has no section for configuration. The `config.py` module and `blitz.toml` file are undocumented in the spec. Given the config-is-disconnected issue noted above, this may be intentional (spec describes intended behavior only), but it means the spec is silent on an entire implemented subsystem.

---

## Docstring Coverage

### File-level module docstrings — COMPLETE (8/8 files)

| File | Module docstring | Quality |
|------|-----------------|---------|
| orchestrator.py | Yes | Good — includes usage examples |
| agents.py | Yes | Adequate |
| consensus.py | Yes | Adequate |
| blackboard.py | Yes | Good — includes Redis key schema |
| embedder.py | Yes | Good — explains model choice rationale |
| config.py | Yes | Adequate |
| memory/writer.py | Yes | Good — explains serialization design |
| memory/reader.py | Yes | Good — lists retrieval pipeline steps |
| memory/models.py | Yes | Adequate |

### Public function/method docstrings — MOSTLY COMPLETE

**Fully documented (docstring present and accurate):**
- `invoke_agent()` — accurate
- `_parse_agent_output()` — accurate, describes fallback chain
- `blast_agents()` — accurate
- `build_context()` — accurate, describes role-filtering behavior
- `run_swarm()` — accurate, lifecycle steps listed
- `plan_agents()` — accurate
- `check_consensus()` — accurate, edge case documented
- `extract_dissent()` — accurate
- `should_override_holdout()` — accurate
- `format_dissent_section()` — accurate
- `Blackboard.initialize()` — accurate
- `Blackboard.write_agent_output()` — accurate
- `Blackboard.build_context_string()` — accurate
- `Embedder.encode()` — accurate
- `Embedder.encode_batch()` — accurate
- `cosine_similarity()` — accurate
- `MemoryWriter.initialize()` — accurate
- `MemoryWriter.store_task()` — accurate
- `MemoryWriter.run_eviction_cycle()` — accurate, explains the three operations
- `MemoryWriter._store_insight()` — accurate, explains dedup logic
- `MemoryReader.retrieve_memory()` — accurate, documents return shape
- `MemoryReader.score_relevance()` — accurate
- `MemoryReader.sparsify_interaction()` — accurate
- `MemoryReader.build_memory_context()` — accurate
- `InteractionGraph.add_utterance()` — accurate

**Missing docstrings (functions with no docstring):**
- `_build_user_prompt()` — no docstring (behavior is clear from code but not documented)
- `_error_output()` — no docstring
- `format_final_output()` — no docstring (non-trivial: handles synthesizer-absent fallback)
- `save_output()` — no docstring
- `_try_connect_blackboard()` — no docstring
- `_write_to_blackboard()` — no docstring
- `_try_init_memory()` — no docstring
- `_persist_to_memory()` — no docstring (longest private function in orchestrator; complex enough to warrant documentation)
- `_split_subtopics_heuristic()` — has docstring ✓
- `_split_subtopics_llm()` — has docstring ✓
- `_llm_plan()` — has docstring ✓
- `Blackboard._verify_connection()` — has docstring ✓
- `Blackboard.get_current_round()` — no docstring
- `Blackboard.get_agent_status()` — no docstring
- `Blackboard.set_agent_status()` — has docstring ✓
- `Blackboard.cleanup()` — has docstring ✓
- `Blackboard.finalize()` — has docstring ✓
- `Blackboard.consume_memory_writes()` — has docstring ✓
- `Embedder._load_model()` — has docstring ✓
- `MemoryWriter._init_sqlite()` — has docstring ✓
- `MemoryWriter._init_lancedb()` — has docstring ✓
- `MemoryWriter.start()` — has docstring ✓
- `MemoryWriter.stop()` — has docstring ✓
- `MemoryWriter.process_write()` — has docstring ✓
- `MemoryWriter._process_task_complete()` — has docstring ✓
- `MemoryWriter._find_related_queries()` — has docstring ✓
- `MemoryWriter._store_query_vector()` — has docstring ✓
- `MemoryWriter._merge_insight()` — has docstring ✓
- `MemoryWriter._insert_new_insight()` — has docstring ✓
- `MemoryWriter.generate_insight_llm()` — has docstring ✓
- `MemoryWriter.maybe_run_eviction()` — no docstring
- `MemoryReader.initialize()` — has docstring ✓
- `MemoryReader.is_available()` — has docstring ✓
- `MemoryReader.query_count()` — has docstring ✓
- `MemoryReader._embedding_search()` — has docstring ✓
- `MemoryReader._one_hop_expand()` — has docstring ✓
- `MemoryReader._get_insights_for_queries()` — has docstring ✓
- `MemoryReader._get_interactions_for_queries()` — no docstring
- `MemoryReader._get_query_details()` — no docstring
- `MemoryReader._update_access_counts()` — no docstring
- `MemoryReader.close()` — no docstring
- `load_config()` — has docstring ✓
- `_dry_run()` — no docstring
- `main()` — no docstring

**Class-level docstrings:**
- `BlitzAgent` — no class docstring (fields are self-explanatory but the class purpose is unstated)
- `Blackboard` — has docstring ✓
- `Embedder` — has docstring ✓
- `MemoryWriter` — has docstring ✓
- `MemoryReader` — has docstring ✓
- `SwarmConfig`, `MemoryConfig`, `RedisConfig`, `StorageConfig`, `EvictionConfig`, `BlitzConfig` — no docstrings (dataclasses with obvious field names; acceptable)
- `Utterance`, `InteractionGraph`, `QueryNode`, `InsightNode`, `InsightEdge`, `AgentOutput` — all have docstrings ✓

**Docstring coverage estimate:** ~75% of public functions and ~90% of non-trivial functions are documented. Private helper functions in orchestrator.py are the main gap.

---

## Missing Documentation

### 1. Two-phase Round 1 architecture is undocumented everywhere
The code makes a deliberate design decision: Round 1 runs researchers first (Phase A), then evaluators (Phase B) — sequentially. Rounds 2+ are fully simultaneous. This is not mentioned in the README, BLITZ-SWARM.md, or any code comment. It is the correct design (evaluators need findings before they can evaluate), but a contributor reading the docs would not know the "simultaneous blast" claim has an important caveat.

### 2. `blitz.toml` is disconnected from runtime — not documented anywhere
`config.py` and `BlitzConfig` are fully implemented but zero runtime modules import or use `load_config()`. The README implies editing `blitz.toml` changes behavior. There is no comment, docstring, or TODO anywhere noting this disconnection. A user editing `blitz.toml` will see no effect and receive no error or warning.

### 3. `--verbose` flag is a no-op — not documented
`args.verbose` is parsed but never used. No log verbosity changes on `--verbose`. This should be either implemented or removed, and in either case the current state should be noted.

### 4. `build_context()` and `_format_context()` duplication is undocumented
`orchestrator.py` defines `build_context()` and `blackboard.py` defines `_format_context()`. They implement nearly identical logic (same role-filtering, same character cap, same section structure). There is no comment explaining why both exist or which should be called when. In practice, `build_context()` in orchestrator.py is always used; `Blackboard.build_context_string()` (which calls `_format_context`) is never called during a normal run.

### 5. `BlitzAgent.max_iterations` is a dead field
The field is in the dataclass, it is in the spec, but it is never read. No comment on the field or in the planning code explains this. A developer would reasonably assume this field controls something.

### 6. LLM memory ops (`score_relevance`, `sparsify_interaction`) are implemented but never called
Both methods exist in `MemoryReader` with docstrings noting they activate "after >10 queries." The threshold constant (`LLM_OPS_THRESHOLD = 10`) exists in `blitz.toml` and `config.py`. However, neither method is called from `retrieve_memory()`. This is either a planned Phase 4 feature or an oversight, but it is not documented as such anywhere. The BLITZ-SWARM.md spec's retrieval pipeline lists them as active steps.

### 7. `aiosqlite` dependency is unused
`pyproject.toml` declares `aiosqlite>=0.20`. All database access uses synchronous `sqlite3`. `aiosqlite` is never imported in any file. This phantom dependency should be removed or its intended use explained.

### 8. `generate_insight_llm()` is implemented but never called from main flow
`MemoryWriter.generate_insight_llm()` generates insights using an LLM call. The actual `_persist_to_memory()` in orchestrator.py builds an insight from synthesizer `key_points` directly (no LLM call). `generate_insight_llm()` is a public method with no callers. There is no comment noting it is available but not wired in.

### 9. No CONTRIBUTING.md or development setup guide
The project has `pyproject.toml` with dev dependencies (`pytest`, `pytest-asyncio`) but no test files exist. There is no guide covering: how to run tests, how to set up a local Redis for development, what the expected development workflow is, or what "Phase 4/5" features (referenced in reader.py, schema.sql comments) refer to.

### 10. Redis key schema is documented only in blackboard.py module docstring
The Redis key layout is clearly documented at the top of `blackboard.py`. It is not mentioned in BLITZ-SWARM.md's storage stack section. For a project that calls itself "infrastructure for the AI field," the wire format documentation lives only in a single module docstring.

---

## Recommendations

Listed by priority (highest impact first).

**P1 — Fix misleading documentation that causes user-visible failures:**

1. **Add a warning to the README Configuration section** that `blitz.toml` is not yet wired to the runtime. Until `load_config()` is connected, document the actual constants to edit (orchestrator.py lines 35–39, blackboard.py lines 24–25). Alternatively, wire in the config — but that is an implementation task, not a docs task.

2. **Remove or implement `--verbose`** and update the README CLI options table accordingly. A flag that is documented but does nothing erodes trust in the documentation.

**P2 — Correct spec-level inaccuracies:**

3. **Update BLITZ-SWARM.md agent count table** to match code: Researcher 2–6, Critic 1–3. The current spec table (2–4, 1–2) is stale.

4. **Update BLITZ-SWARM.md lifecycle diagram** to accurately reflect: (a) the two-phase Round 1 structure, and (b) that CONSOLIDATE/memory persistence happens at FINALIZE, not between rounds.

5. **Add a note to BLITZ-SWARM.md** clarifying which spec features are implemented vs. planned: LLM memory ops (`score_relevance`, `sparsify_interaction`) are unconnected; `BlitzAgent.max_iterations` is unused.

**P3 — Fill docstring gaps:**

6. **Add a docstring to `_persist_to_memory()`** — it is the most complex private function in the orchestrator (30+ lines, builds utterance graphs, manages multiple backends) and has no documentation.

7. **Add a docstring to `format_final_output()`** — documents the synthesizer-absent fallback behavior which is non-obvious.

8. **Add docstrings to `_build_user_prompt()` and `_error_output()`** — minor but these are called on every agent invocation.

9. **Add a `BlitzAgent` class docstring** naming the five valid role values and pointing to `ROLE_PROMPTS` for the prompt content.

**P4 — Structural documentation gaps:**

10. **Document the Round 1 two-phase split** with an inline comment block in `run_swarm()` or a new README section. The existing "No waves. All agents fire simultaneously" tagline is accurate from round 2 onward but actively misleading for round 1.

11. **Document the `build_context` / `_format_context` duplication** with a TODO comment in `blackboard.py` explaining that `build_context()` in orchestrator.py is the active path and `Blackboard.build_context_string()` is unused.

12. **Remove `aiosqlite` from `pyproject.toml`** (or add a comment that it is reserved for a future async migration) and update the README `pip install` command to match.

13. **Add a `# TODO: wire into runtime` comment** to `load_config()` in `config.py`, and on each hardcoded constant in orchestrator.py that has a corresponding `blitz.toml` key, add a reference comment pointing to the config field.
