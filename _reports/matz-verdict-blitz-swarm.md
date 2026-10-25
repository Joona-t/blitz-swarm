# Matz Verdict: Blitz Swarm Code Quality & API Design

_Agent: Matz (Code Quality & API Design Lead) | Date: 2026-03-12_
_Inputs: dead-code-hunter report, manifest-validator report, full source verification_

---

## 1. Domain Assessment

Blitz Swarm is a well-structured v0.1.0 codebase with clear separation of concerns: orchestration, agent definitions, consensus logic, memory read/write, blackboard coordination, and embedding. The code reads well. Function and variable names say what they mean. The dataclass-based agent model is clean. The orchestrator's main loop (SPAWN -> BLAST -> WRITE -> CHECK -> ITERATE -> FINALIZE) is easy to follow and correctly implements an iterative consensus protocol.

The principal design flaw is a pattern I would call **"premature infrastructure"** -- several subsystems were built out before they were wired in. The config module, the LLM memory operations (`score_relevance`, `sparsify_interaction`, `generate_insight_llm`), the eviction system, and `encode_batch` all exist as complete implementations with no callers. This is not technical debt from decay; it is forward-looking code that arrived before the call sites were ready. The risk is that this dead code silently drifts out of sync with the live code paths it was designed to serve.

The dependency manifest has real correctness issues: a phantom dependency (`aiosqlite`) and a missing one (`pandas`). These will bite a fresh `pip install` user immediately.

Overall code quality: **solid for a v0.1.0 prototype**. The live code paths are clean. The dead code is the primary concern.

---

## 2. Confirmed Issues

Every finding from both specialist reports was independently verified against source. All confirmed:

### From dead-code-hunter:

| Finding | Verified | Notes |
|---------|----------|-------|
| `config.py` never imported | YES | Zero imports of `config`, `load_config`, or `BlitzConfig` anywhere outside `config.py` itself. Constants are hardcoded in `orchestrator.py` (lines 35-39) and `blackboard.py` (lines 24-25). |
| `AGENT_OUTPUT_SCHEMA_JSON` unused in orchestrator | YES | Imported at line 19, never referenced in any function body. The schema is instead inlined as prose in `_build_user_prompt()`. |
| `import sys` unused in orchestrator | YES | Imported at line 12, `sys.` never appears in the file. |
| `Embedder` class unused in reader.py | YES | Line 16 imports both `Embedder` and `get_embedder`; only `get_embedder` is called. |
| `TaskStatus` unused in writer.py | YES | Imported at line 16, never referenced. Writer uses raw string `"resolved"/"failed"` throughout. |
| `PLANNING_SCHEMA` never referenced | YES | Defined at line 217 of agents.py, never passed to any subprocess call or used elsewhere. `_llm_plan()` does not use it. |
| `BlitzAgent.max_iterations` never read | YES | Field defined at line 64, no code reads `.max_iterations` on any agent. Global `max_rounds` controls iteration. |
| `--verbose` parsed but never checked | YES | `args.verbose` is parsed (line 716) but never passed to `run_swarm()` or checked anywhere. |
| 14 dead functions | YES | All 14 verified as having zero callers in the codebase. Notably: `encode_batch`, `build_context_string`, `read_all_outputs`, `get_agent_status`, `consume_memory_writes`, `score_relevance`, `sparsify_interaction`, `query_count`, `close`, `generate_insight_llm`, `run_eviction_cycle`, `maybe_run_eviction`, `stats`, `_split_subtopics_llm`. |

### From manifest-validator:

| Finding | Verified | Notes |
|---------|----------|-------|
| `aiosqlite` declared, never imported | YES | In pyproject.toml deps, zero imports across all `.py` files. All SQLite access uses synchronous `sqlite3`. |
| `pandas` used but undeclared | YES | `.to_pandas()` called in `reader.py:140` and `writer.py:217`. `pandas` is not in pyproject.toml. (LanceDB may bundle it as a transitive dep, but it should be declared explicitly.) |
| No `[build-system]` table | YES | pyproject.toml has no build-system table. |
| No `[project.scripts]` entry point | YES | No CLI entry point defined. |
| Config system bypassed | YES | Duplicate of dead-code-hunter finding. Thoroughly confirmed. |
| `.gitignore` missing `.pytest_cache/`, `.env`, `_reports/` | YES | Verified. |

---

## 3. Disputed Issues

### 3a. `_split_subtopics_llm` flagged as dead -- PARTIALLY DISPUTED

The dead-code-hunter correctly notes this function has zero callers. However, examining the code, `plan_agents()` calls `_llm_plan()` which returns a plan that may already include subtopics. If `_llm_plan()` returns subtopics, those are used directly. If not, `_split_subtopics_heuristic()` is called. The function `_split_subtopics_llm()` was clearly intended as a standalone subtopic splitter but was superseded by the integrated `_llm_plan()` approach. It is genuinely dead, but the report does not distinguish it from the other dead functions as a **superseded design** rather than an **unfinished feature**. Severity is the same (remove it), but the narrative matters for understanding the codebase's evolution.

### 3b. `pandas` missing from deps -- SEVERITY NUANCE

LanceDB's `.to_pandas()` method requires pandas, but LanceDB itself pulls pandas as a transitive dependency (`lancedb>=0.6` depends on `pyarrow` which typically co-installs with pandas, and LanceDB's own API surface assumes pandas). In practice, a `pip install lancedb` will bring pandas along. The finding is still correct -- explicit is better than implicit -- but it is unlikely to cause a real installation failure. This is a **should fix**, not a **blocks users**.

### 3c. No other disputes

Both reports were conservative and accurate. No findings were overstated.

---

## 4. Priority Fix List

Ordered by user-facing impact:

### P1. Wire config.py into the runtime OR remove it
**Impact: HIGH.** A user editing `blitz.toml` gets zero effect and zero feedback. This is the definition of surprising behavior. Either `orchestrator.py` should call `load_config()` and use the returned `BlitzConfig` for its constants, or `config.py` + `blitz.toml` should be removed with a note that configuration is planned for a future version. The former is the better path -- the config system is already built.

### P2. Fix dependency manifest (remove aiosqlite, declare pandas)
**Impact: HIGH.** A phantom dependency is confusing; a missing dependency can break installs on environments where transitive deps don't cover it. Two-line fix in pyproject.toml.

### P3. Delete or annotate the 14 dead functions
**Impact: MEDIUM.** 14 dead functions across 4 files is significant surface area for a ~750-line-of-logic codebase. The LLM memory ops (`score_relevance`, `sparsify_interaction`, `generate_insight_llm`) and eviction system (`run_eviction_cycle`, `maybe_run_eviction`) are clearly Phase 2+ features. Either delete them and re-add when call sites exist, or add explicit `# Phase 2: not yet wired` comments. Dead code without annotation is a maintenance trap.

### P4. Clean unused imports and constants
**Impact: LOW.** Remove `AGENT_OUTPUT_SCHEMA_JSON` import in orchestrator (or actually use it instead of the prose schema in `_build_user_prompt`), remove `import sys`, remove `Embedder` name import in reader.py, remove `TaskStatus` import in writer.py, remove `PLANNING_SCHEMA` (or wire it into `_llm_plan`). Minor but these create noise for anyone reading the code.

### P5. Implement --verbose or remove it
**Impact: LOW.** A documented CLI flag that silently does nothing violates the principle of least surprise. Either thread it through to `run_swarm()` for debug output, or remove the flag and its documentation.

---

## 5. Verdict

**PASS WITH CONDITIONS**

The live code paths are well-designed, readable, and architecturally sound. The orchestrator loop, consensus protocol, memory retrieval pipeline, and agent planning all demonstrate clear thinking. The API surface (function signatures, data flow, error handling) follows the principle of least surprise where it matters -- in the code that actually runs.

The conditions for a clean PASS:

1. **P1 and P2 must be addressed before any public release or README-directed onboarding.** A config file that does nothing and a dependency list that is wrong are the two things most likely to erode a new user's trust in the project.
2. **P3 should be addressed before the codebase grows further.** At current scale, the dead code is manageable. At 2x the file count, it becomes a real navigation burden.

The codebase does not FAIL because: the live execution paths are correct, the architecture is sound, error handling is present and reasonable, and none of the dead code introduces runtime bugs -- it is simply unreachable. This is a v0.1.0 with some forward-looking scaffolding that arrived early. That is a quality problem, not a correctness problem.
