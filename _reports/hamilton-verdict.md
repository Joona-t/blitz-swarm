# Hamilton Verdict: Blitz Swarm Reliability & Data Safety

_Domain Lead: Hamilton (Reliability & Data Safety) | Date: 2026-03-12_

---

## 1. Domain Assessment

Blitz Swarm is architecturally sound for a v1 system. The core loop (SPAWN -> BLAST -> WRITE -> CHECK -> ITERATE -> FINALIZE) is well-structured with graceful degradation: Redis is optional, memory is optional, LLM planning falls back to heuristics. The agent invocation layer in `orchestrator.py` is the strongest component -- it handles timeouts, exit codes, malformed output, and even raw text fallback with four-level JSON parsing. The privacy posture is clean.

The primary risk area is the **persistence boundary** -- the seam between the ephemeral swarm loop and durable storage (SQLite/LanceDB/Redis). Failures at this boundary can silently lose completed work. The secondary risk is the lack of structured error propagation from the memory subsystem back to the caller.

**Overall health: Moderate.** The happy path is solid. The failure paths need hardening, but none of the issues are architectural -- they are all fixable with targeted error handling additions.

---

## 2. Confirmed Issues

### From error-scanner:

**CRIT-01: `save_output` has no error handling.** CONFIRMED. Lines 315-322 of `orchestrator.py`: `OUTPUT_DIR.mkdir()` and `filepath.write_text()` are both bare -- a disk full or permission error after a multi-minute swarm run loses the entire output with an unhandled exception. This is the highest-impact single bug because it occurs at the very last step.

**CRIT-02: `_init_sqlite` has no error handling.** CONFIRMED. Lines 61-69 of `writer.py`: `SCHEMA_PATH.read_text()` will raise `FileNotFoundError` if `schema.sql` is missing, and `executescript()` will raise on malformed SQL. Since `_init_sqlite` is called from `initialize()`, which is called inside a try/except in `_try_init_memory()` (orchestrator.py line 357), the failure IS caught at the orchestrator level and the system falls back to no-memory mode. However, if `_init_lancedb()` was called first and succeeded, the LanceDB handle is leaked. Severity: **medium** (not critical as reported, because the orchestrator-level catch prevents a crash).

**CRIT-03: SQLite transaction swallows inner exceptions.** PARTIALLY CONFIRMED. The `with self._db:` context manager on line 153 of `writer.py` does auto-rollback on exception, but the exception then propagates upward. It is NOT swallowed -- the `process_write` method does not catch it, so it would bubble up to the `start()` loop's broad `except Exception` on line 117, which prints the error and continues. The rollback behavior is correct. The real issue is that a failure inside `_store_insight` (e.g., embedder crash) rolls back the entire transaction including the already-written utterances and query node. This is an atomicity granularity problem, not a swallowed exception.

**CRIT-04: `scan_iter` in cleanup not atomic.** CONFIRMED. Lines 78-82 of `blackboard.py`: `scan_iter` + pipeline delete is not atomic. If Redis restarts mid-scan, some keys survive. Severity is low in practice because `cleanup()` is only called at the start of `initialize()` -- stale keys from a prior run simply get overwritten by the new run's data. The next `initialize()` call will clean them.

**CRIT-05: `write_agent_output` has no error handling.** CONFIRMED. Lines 105-118 of `blackboard.py`: Both `redis.set()` and `redis.xadd()` are bare calls. A Redis drop here means `_write_to_blackboard` in orchestrator.py line 348 crashes with `ConnectionError`. However, the in-memory `all_round_outputs` list in the orchestrator loop (line 564) already captured the outputs, so the research data is NOT lost -- only the Redis copy fails. The swarm loop itself would crash, but the outputs exist in-memory up to that point.

**CRIT-06: Malformed agent output causes TypeError in `build_context`.** NOT CONFIRMED. I examined `build_context` (orchestrator.py lines 211-270) carefully. Every field access uses `.get()` with defaults: `o.get("role", "unknown")`, `o.get("findings", "")[:800]`, `o.get("key_points", [])`, `o.get("confidence", 0)`. If `findings` were an integer (42), `[:800]` would fail on an int -- but this scenario requires the JSON schema to be violated AND all four parsing fallbacks in `_parse_agent_output` to produce a non-string `findings`. The `_error_output` fallback (line 163) always produces a string findings. This is theoretically possible only if the LLM returns valid JSON with `findings: 42`, which is guarded by the schema instruction. Severity: **low** (not critical).

**CRIT-07: `_merge_insight` silently drops if `insight_id` is stale.** CONFIRMED but low severity. Lines 301-315 of `writer.py`: if the `fetchone()` returns `None` (row deleted between check and merge), the function silently does nothing. This is actually correct behavior -- the insight was evicted, so merging into it is pointless. No data is lost.

**WARN-04: TOML parse failure silently swallowed in config.py.** CONFIRMED. Line 72 of `config.py`: the bare `except Exception: return config` means a malformed `blitz.toml` silently falls back to defaults with no warning printed. The user gets no indication their config file has a syntax error.

**WARN-05: Redis consumer loop retries forever with no backoff.** PARTIALLY CONFIRMED. Line 117-119 of `writer.py`: the `except` block does `await asyncio.sleep(1)` -- there IS a 1-second delay, but it's fixed, not exponential. Under sustained Redis failure, this logs once per second indefinitely. Not ideal but not dangerous.

**WARN-08: `_update_access_counts` unguarded.** CONFIRMED. Lines 262-278 of `reader.py`: this writes to SQLite from the reader, which could contend with the writer. However, WAL mode + `busy_timeout=30000` on both connections makes this extremely unlikely to actually crash. SQLite WAL allows concurrent readers and a single writer, and the 30-second busy timeout provides substantial margin. Severity: **low**.

**WARN-11: Broad `except` catches `KeyboardInterrupt`.** NOT CONFIRMED in the main orchestrator loop. The `blast_agents` function uses `asyncio.gather` which propagates `KeyboardInterrupt` correctly. The writer's `start()` loop (line 117) catches bare `Exception`, which does NOT catch `KeyboardInterrupt` in Python 3 (it inherits from `BaseException`). This warning appears to be a false positive.

### From privacy-guardian:

**PASS verdict: CONFIRMED.** I verified all four source files plus `config.py`, `agents.py`, and `embedder.py`. There are:
- Zero network calls outside `localhost:6379` (Redis) and local `claude` CLI subprocess
- Zero `os.environ` / `os.getenv` calls
- Zero hardcoded credentials
- Zero telemetry or analytics endpoints
- LanceDB runs in embedded mode (local filesystem only)
- SQLite is local filesystem only

**REC-01 (sentence-transformers download): CONFIRMED.** The `embedder.py` `SentenceTransformer()` constructor will download `all-MiniLM-L6-v2` from HuggingFace on first run. This is an implicit network call that should be documented.

**REC-02 (`--dangerously-skip-permissions`): CONFIRMED.** Used in `orchestrator.py` line 59, `agents.py` lines 190/345, `writer.py` line 354, `reader.py` lines 302/348. This flag bypasses Claude CLI permission checks. The risk is that agent-generated prompts could instruct Claude to execute arbitrary shell commands. Since the prompts are constructed from user-supplied topics and LLM-generated content, this is a real attack surface if the system is exposed to untrusted input.

**REC-03 (remote Redis warning): CONFIRMED.** `config.py` allows overriding `redis.host` via `blitz.toml`. No validation prevents pointing to a remote host, which would send all research data over the network unencrypted.

---

## 3. Disputed Issues

**CRIT-02 severity overstated.** The error-scanner classified `_init_sqlite` as critical with "crashes, leaks LanceDB handles." In reality, the orchestrator's `_try_init_memory()` wraps initialization in try/except and falls back gracefully. The LanceDB handle leak is real but minor (it's a file handle that gets cleaned up on process exit).

**CRIT-03 framing incorrect.** The report says "swallows inner exceptions." The `with self._db:` context manager rolls back and re-raises. The exception is caught by the consumer loop, logged, and the loop continues. This is not swallowing -- it's explicit error handling. The real issue (atomicity granularity) is valid but was misidentified.

**CRIT-06 severity overstated.** The four-level JSON parsing fallback in `_parse_agent_output` makes this scenario extremely unlikely. The last-resort fallback (line 163) always produces correctly-typed fields. A malformed `findings` field would require the LLM to return valid JSON that passes parsing but violates the schema -- possible but rare.

**CRIT-07 is not a bug.** Silent no-op on a missing insight is correct defensive behavior, not an error.

**WARN-11 is a false positive.** Python 3's `Exception` does not catch `KeyboardInterrupt`. Ctrl+C works correctly.

---

## 4. Priority Fix List (ordered by impact)

### P1: Wrap `save_output` in try/except with fallback
**File:** `orchestrator.py`, lines 315-322
**Impact:** Complete output loss after potentially minutes of LLM invocations.
**Fix:** Catch `OSError`, print the output to stdout as emergency fallback, attempt writing to `/tmp` as secondary location.

### P2: Add error handling around `_write_to_blackboard` in the main loop
**File:** `orchestrator.py`, line 563
**Impact:** Redis failure mid-round crashes the entire swarm loop even though all outputs are already in memory.
**Fix:** Wrap the `_write_to_blackboard` call in try/except. Log the failure, set `bb = None` to disable further blackboard writes, and continue the loop with in-memory data.

### P3: Split the SQLite transaction in `_process_task_complete` into two phases
**File:** `memory/writer.py`, lines 153-193
**Impact:** An embedder failure in `_store_insight` rolls back the already-written utterances and query node, losing all tier-1 and tier-2 data.
**Fix:** Commit utterances + query node first, then attempt insight storage in a separate transaction. If insight fails, the interaction graph is still persisted.

### P4: Print a warning when `blitz.toml` parse fails
**File:** `config.py`, line 72
**Impact:** Users with malformed config files get silent fallback to defaults with no indication.
**Fix:** Add `print(f"Warning: Failed to parse {path}: {e}. Using defaults.")` in the except block.

### P5: Validate Redis host is localhost or warn on remote
**File:** `config.py` and `blackboard.py`
**Impact:** Unencrypted research data could be sent to a remote Redis host without user awareness.
**Fix:** In `Blackboard.__init__`, if host is not `localhost`/`127.0.0.1`/`::1`, print a warning about unencrypted transport.

---

## 5. Verdict

**PASS WITH CONDITIONS**

The system is architecturally sound with good graceful degradation patterns (optional Redis, optional memory, four-level JSON parsing, timeout handling). The privacy posture is clean -- no telemetry, no data exfiltration, all storage local.

The conditions for unconditional PASS:
1. **P1 must be fixed before any production use.** Losing output after a multi-minute, multi-dollar swarm run due to a simple disk error is unacceptable.
2. **P2 must be fixed.** Redis is explicitly optional, but a Redis failure mid-run still crashes the system instead of falling back to in-memory mode.
3. **P3 should be fixed.** The all-or-nothing transaction boundary means an embedder hiccup can destroy an entire task's memory record.

P4 and P5 are quality-of-life improvements that can ship in a follow-up.

The error-scanner report was thorough but overstated severity on 4 of 7 critical findings. The privacy-guardian report was accurate and well-calibrated. Both specialists did useful work.
