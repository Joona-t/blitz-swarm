# Linus Verdict: Blitz Swarm v0.1.0

_Lead: Linus (Code Review & Security) | Date: 2026-03-12 | Wave 2+3 Combined_

---

## 1. Domain Assessment

The security posture is **acceptable for a local dev tool at v0.1.0**, with one design choice that needs documentation rather than code changes, and one dead dependency that needs removal. The codebase is well-structured, fail-safe by default (Redis optional, LanceDB optional, memory optional), and has no external network exposure.

The security-scanner report is solid but slightly overweights the `--dangerously-skip-permissions` issue for a tool that is *designed* to be an autonomous agent orchestrator. The dependency-auditor report is accurate and actionable.

---

## 2. Confirmed Issues

### CONFIRMED: aiosqlite is a dead dependency
Verified. `aiosqlite` appears in `pyproject.toml` line 11 and `README.md` install instructions. It is **never imported anywhere** in the codebase. All SQLite access uses synchronous `sqlite3` stdlib. Remove it.

### CONFIRMED: --dangerously-skip-permissions on all 6 subprocess call sites
Verified. Found at:
- `orchestrator.py:59` — main agent invocation
- `agents.py:191` — subtopic splitting LLM call
- `agents.py:347` — swarm planning LLM call
- `memory/writer.py:355` — insight generation LLM call
- `memory/reader.py:305` — relevance scoring LLM call
- `memory/reader.py:352` — interaction sparsification LLM call

This is architecturally intentional — autonomous agents need to run without interactive prompts. However, the memory-tier LLM calls (writer.py, reader.py) should use `--allowedTools` to restrict to text-only output since they only need JSON responses, not tool use.

### CONFIRMED: f-string SQL table name in writer.py:stats
Line 495: `f"SELECT COUNT(*) FROM {table}"` — the `table` variable iterates over a hardcoded tuple `("queries", "utterances", "insights")`. Currently safe. But the pattern is a footgun. If anyone adds a config-driven table name later, it becomes SQL injection. Acceptable for v0.1.0 with a code comment.

### CONFIRMED: No lockfile
No `requirements.lock`, `uv.lock`, or `poetry.lock` exists. For a v0.1.0 dev tool this is fine. Should be added before any production deployment.

### CONFIRMED: Redis has no auth configured
`blackboard.py:35-39` — connects to `localhost:6379` with no password. For a local dev tool binding to localhost, this is expected. The Docker instructions in the error message (`redis:7-alpine`) also specify no auth. Acceptable for v0.1.0, document in README.

### CONFIRMED: No upper bounds on dependency versions
`pyproject.toml` uses `>=` for all deps with no upper bounds. Standard Python practice for libraries, slightly risky for applications. Low priority.

---

## 3. Disputed Issues

### DISPUTED: CRIT-01 severity overstated — "injection-to-execution chain"
The security-scanner calls the `--dangerously-skip-permissions` + topic input combo a CRITICAL injection-to-execution chain. Let me be precise about why this is overstated:

1. The `topic` string comes from `sys.argv` — the person running the CLI typed it. This is a **local tool**, not a web service. The user who provides the topic already has shell access. They can already run arbitrary code.
2. Prompt injection via topic string would require the user to inject into their own prompts, which is an attack against yourself.
3. The *real* risk vector is stored prompt injection (Attack Path 2 in the report) — a poisoned insight retrieved from the DB could influence future agent behavior. But this requires a prior compromised run, and the agents are read-only research tools without persistent side effects.

**Reclassified: MEDIUM for documentation purposes, not CRITICAL.**

### DISPUTED: WARN-01 — "agent.model not validated against allowlist"
The security-scanner claims agent model names could come from LLM planning. I verified the code: `_llm_plan()` in `agents.py` returns `researcher_count`, `critic_count`, `needs_fact_checker`, and `subtopics`. It does **not** return model names. Model assignment is hardcoded in `plan_agents()` — either from `ROLE_MODEL_OVERRIDES` dict (lines 212-215) or the default `"sonnet"` on the `BlitzAgent` dataclass (line 64). **No user or LLM input reaches `agent.model`. This is a false positive.**

### DISPUTED: WARN-07 — "IN-clause placeholder expansion unbounded"
The security-scanner flags `IN ({placeholders})` in `reader.py` as potentially exceeding `SQLITE_LIMIT_VARIABLE_NUMBER`. The actual data flow: `seed_ids` comes from `_embedding_search` with `top_k=2`, then `_one_hop_expand` adds neighbors. In practice this is bounded at ~10-15 IDs maximum. The SQLite default variable limit is 999. This is a theoretical concern that cannot trigger in practice. **Not a real issue.**

### DISPUTED: CVE-2025-49844 (CVSS 10.0) on Redis
The dependency-auditor cites `CVE-2025-49844`. I cannot verify this CVE exists or its details. The report says to pin the Redis Docker image. Since Redis is optional and the user controls their own Docker setup, this is outside the scope of Blitz Swarm's codebase. Add a note in the README at most.

### DISPUTED: "sentence-transformers 2-4GB transitive"
The report says 2-4GB install. The actual footprint for `all-MiniLM-L6-v2` (22M params, as specified in `embedder.py:19`) with PyTorch CPU-only is ~800MB-1.2GB. Still heavy, but not 2-4GB. The code already tries ONNX backend first (`embedder.py:32`) which avoids the full PyTorch install. Overstated.

---

## 4. Priority Fix List

| # | Severity | File | Fix |
|---|----------|------|-----|
| 1 | **HIGH** | `pyproject.toml` | Remove `aiosqlite>=0.20` — dead dep, confuses users, increases attack surface for zero benefit |
| 2 | **MEDIUM** | `memory/writer.py`, `memory/reader.py` | Add `--allowedTools` whitelist to the 3 memory-tier LLM subprocess calls to restrict them to text-only output (no Bash, no file tools) |
| 3 | **LOW** | `memory/writer.py:495` | Add `# SAFETY: table names are hardcoded — never use user input here` comment above the f-string SQL |
| 4 | **LOW** | `pyproject.toml` | Move `lancedb` and `sentence-transformers` to `[project.optional-dependencies]` since both are soft-imported with try/except fallbacks |
| 5 | **LOW** | root | Generate a `uv.lock` or `requirements.lock` for reproducible installs |

---

## 5. GO / NO-GO Decision

**GO — conditional on Fix #1 before tagging v0.1.0.**

Rationale:
- Fix #1 (remove aiosqlite) is a one-line edit in `pyproject.toml` and one line in `README.md`. Do it now.
- Fix #2 (allowedTools on memory LLM calls) is a good hardening measure but not blocking for v0.1.0 of a local dev tool. Track it for v0.1.1.
- Fixes #3-5 are polish. Do them when convenient.

The codebase is clean, well-architected, and has appropriate fail-safe patterns throughout (graceful Redis degradation, optional LanceDB, try/except on all external integrations). The `--dangerously-skip-permissions` flag is an inherent requirement of autonomous agent orchestration — you cannot build this tool without it. The attack surface is limited to the local machine of the person who intentionally runs the tool.

Ship it.

---

_Linus out._
