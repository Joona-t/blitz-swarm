# privacy-guardian Report: Blitz Swarm

**Agent:** privacy-guardian
**Reporting to:** Hamilton (Reliability & Data Safety Lead)
**Date:** 2026-03-12
**Scope:** Full read of orchestrator.py, agents.py, consensus.py, blackboard.py, embedder.py, config.py, memory/writer.py, memory/reader.py, memory/models.py, memory/schema.sql, blitz.toml, pyproject.toml

---

## Summary

Blitz Swarm is a local-first, privacy-respecting architecture in its overall design intent. There is no telemetry, no analytics, no remote data collection, and no third-party network calls. All persistent storage is on-disk (SQLite + LanceDB) and in-process (Redis on localhost).

However, several concrete privacy and data safety risks exist that require attention before this system handles sensitive research topics:

1. **All six `claude` subprocess invocations use `--dangerously-skip-permissions`** — this flag bypasses the Claude CLI's tool-use safety prompts, meaning agents can silently invoke file system, shell, or network tools without user confirmation.
2. **Full research content (the user's topic and agent findings) is sent to Claude's API** on every subprocess call. There is no scrubbing, redaction, or consent mechanism before data leaves the process.
3. **Redis blackboard keys have no TTL** — all round data, findings previews, and the memory write queue persist indefinitely unless `cleanup()` is explicitly called at session start.
4. **Output files and the SQLite database are created with world-readable permissions (0755/0644 by default)** — no `chmod` hardening is applied.
5. **Utterance content (up to 500 chars of agent findings) is stored verbatim in SQLite** and retrieved across future sessions, creating a persistent data lake of prior research content with no user visibility or export/delete tooling.
6. **`sentence-transformers` first use silently downloads model weights from Hugging Face** — this is an outbound call that is undocumented in the README and not user-confirmed.

---

## Data Flow Analysis

### What the user's topic touches

The user's topic string propagates through the entire system without sanitization or truncation:

- `orchestrator.py:456` — printed to stdout verbatim
- `orchestrator.py:509` — embedded in `task_prompt` sent to every agent subprocess
- `orchestrator.py:318` — slugified and used in the output filename
- `blackboard.py:66` — stored as `topic` in the Redis `blackboard:meta` hash
- `memory/writer.py:146` — stored verbatim in SQLite `queries.query_text`
- `memory/writer.py:241` — embedded into LanceDB alongside its vector embedding
- `memory/writer.py:333–343` — passed to an LLM subprocess for insight generation, including up to 15 utterances of agent content (each up to 200 chars)

**Risk:** If the topic contains PII (e.g. "research on John Smith's medical condition"), that string is stored persistently in SQLite and LanceDB, sent to the Claude API, and printed to the terminal. There is no sanitization or warning.

### What agent findings touch

Agent findings (the LLM's full research output) propagate to:

- `blackboard.py:117` — stored in Redis as full JSON (unbounded, except by LLM output limits)
- `blackboard.py:118` — first 200 chars previewed in the `blackboard:findings` Redis Stream
- `memory/writer.py:158–163` — first 500 chars stored per utterance in SQLite `utterances.content`
- `orchestrator.py:406–408` — first 3 key points from synthesizer stored as the insight text in SQLite `insights.content`
- Output markdown file — full findings written to `output/<slug>_<timestamp>.md`

**Risk:** Findings that inadvertently contain sensitive information (e.g. an agent that searches and surfaces PII during research) are persisted across sessions with no cleanup path.

### LLM-powered memory operations send historical data back to Claude

`memory/reader.py:score_relevance()` (line 283) and `memory/reader.py:sparsify_interaction()` (line 318) both invoke `claude -p` subprocesses with content retrieved from SQLite — including `query_text` from prior sessions and utterance traces up to 200 chars each. This means historical stored content is sent to the Claude API on future runs, not just during the originating session.

`memory/writer.py:generate_insight_llm()` (line 321) similarly sends up to 15 utterances of content from a completed task back to Claude for insight generation.

---

## Network Boundaries

### Confirmed outbound network calls

| Call site | Destination | What is sent | Controlled by user? |
|-----------|------------|--------------|---------------------|
| `orchestrator.py:66` | Claude API (via `claude` CLI) | topic, system prompt, full agent context (up to 8000 chars) | Yes — CLI invocation |
| `agents.py:185` | Claude API (via `claude` CLI) | topic string, subtopic planning prompt | Yes — at startup |
| `agents.py:341` | Claude API (via `claude` CLI) | topic string, LLM planning prompt | Yes — at startup |
| `memory/writer.py:348` | Claude API (via `claude` CLI) | topic + up to 15 utterance snippets (200 chars each) | No — automatic, post-task |
| `memory/reader.py:299` | Claude API (via `claude` CLI) | new query + historical query text from SQLite | No — automatic, on retrieval |
| `memory/reader.py:346` | Claude API (via `claude` CLI) | new query + full utterance trace from SQLite | No — automatic, on retrieval |
| `embedder.py:32–34` | Hugging Face (model download) | none (weight fetch only) | No — silent on first use |
| `blackboard.py:35` | Redis (localhost:6379) | all agent data | Local only — safe |

### Notable finding: `--dangerously-skip-permissions` on all CLI calls

Every `subprocess.run(["claude", ...])` invocation across all files passes `--dangerously-skip-permissions`. This flag tells the Claude CLI to skip interactive approval of tool uses. If any agent is instructed (or jailbroken via a malicious topic string) to use tools like `Bash`, `Read`, or network tools, those calls will execute without user confirmation.

Files affected:
- `orchestrator.py:59`
- `agents.py:191`
- `agents.py:347`
- `memory/writer.py:355`
- `memory/reader.py:305`
- `memory/reader.py:352`

**Risk level: HIGH.** A maliciously crafted topic string could potentially inject instructions into the user prompt that cause an agent to exfiltrate data, read local files, or make network calls — all silently approved.

### Memory LLM operations are not gated by user consent

The `score_relevance()` and `sparsify_interaction()` calls in `memory/reader.py` activate automatically once the query count exceeds `llm_ops_threshold` (default: 10, configurable in `blitz.toml`). The user is not notified when this threshold is crossed and historical data begins flowing to the Claude API on every retrieval.

---

## Storage Safety

### SQLite (`memory.db`)

- **Location:** Project root, hardcoded as `memory.db` relative to the script.
- **Permissions:** Created with default `umask`-derived permissions (typically `0644` on macOS — world-readable to other users on a shared system).
- **No TTL on utterances or queries** at the database level; eviction is only triggered by `maybe_run_eviction()` after every 10 writes, and only archives queries older than 90 days.
- **No encryption at rest.**
- **No user-facing delete or export API.** There is no CLI command, helper script, or documented method for a user to purge their stored research history.
- **Utterance content stored verbatim** up to 500 chars per agent turn; on a 5-agent, 5-round run that is up to 12,500 chars of raw findings persisted indefinitely.
- **The `queries_archive` table** retains evicted queries permanently — eviction does not delete, it moves.

### LanceDB (`memory_vectors/`)

- **Location:** Project root as `./memory_vectors`.
- **Permissions:** Directory will be created with default permissions (world-readable).
- **Contains embeddings** of every query topic ever run. While embeddings are not directly reversible, the `queries` LanceDB table also stores the raw `text` field (the topic string) alongside the vector.
- **No user-facing delete API.**

### Redis (localhost:6379)

- **Connection:** `host=localhost`, `port=6379`, `db=0` — local only, not exposed externally.
- **No authentication configured** in `config.py` or `blitz.toml` (Redis default: no password). If Redis is reachable on the network interface rather than strictly loopback, data is unauthenticated.
- **No TTL set on any key.** `blackboard:round:*` keys, `blackboard:agent:*` keys, `blackboard:findings` stream, and `memory:writes` stream all persist until `cleanup()` is called. `cleanup()` is only called at the start of a new swarm run (`initialize()` calls `cleanup()`), meaning data from the last run persists in Redis indefinitely between sessions.
- **`blackboard:findings` stream is never deleted** after `finalize()` — only `blackboard:*` keys are deleted by `cleanup()`; the `memory:writes` stream is also deleted by `cleanup()`, but `blackboard:findings` is not explicitly covered (the `scan_iter("blackboard:*")` pattern would catch it, but this is not guaranteed if the stream key name were changed).

### Output files (`output/`)

- **Location:** `output/` directory in project root.
- **Permissions:** Directory is `0755` (world-readable). Files are written with `filepath.write_text()` which inherits the process `umask` — typically `0644`.
- **No encryption.** Full research output is in plaintext markdown.
- **No automatic cleanup.** Files accumulate indefinitely with timestamp suffixes.
- **Filename leaks topic:** The output filename slug is derived from the topic string (`re.sub(r"[^a-z0-9]+", "_", topic.lower())`). A topic like "Joona's health condition" becomes `joona_s_health_condition_20260312_120000.md` — visible in directory listings.

---

## Recommendations

These are observations only — no code has been changed.

### Critical (address before any sensitive-topic use)

1. **Remove or gate `--dangerously-skip-permissions`.** Consider using `--allowedTools` to restrict agent tool access to a safe whitelist (e.g. no `Bash`, no `Write`), or removing the flag entirely and accepting the interactive prompt overhead for security.

2. **Add a topic/content sensitivity warning at startup.** Before any data is sent to the Claude API, print a clear notice that the topic and generated findings will be transmitted to Anthropic's API. Give the user an opt-out.

3. **Apply TTLs to all Redis keys.** Set an expiry (e.g. 24 hours) on `blackboard:round:*`, `blackboard:agent:*`, and the findings stream after `finalize()` is called. The current model relies entirely on the next session's `cleanup()` call to remove prior data.

### High priority

4. **Harden file permissions on output and database files.** After creating `memory.db` and output `.md` files, apply `os.chmod(path, 0o600)` to restrict to owner-only access. Apply `os.chmod(OUTPUT_DIR, 0o700)` to the output directory.

5. **Notify the user when LLM memory operations activate.** When `_write_count` crosses `llm_ops_threshold`, print a clear message that historical data will now be sent to the Claude API on retrievals. This threshold crossing is currently silent.

6. **Document the Hugging Face model download.** The first call to `get_embedder()` silently downloads `all-MiniLM-L6-v2` from `huggingface.co`. Add a log message before the download and document this in the README.

### Medium priority

7. **Provide a `blitz clear-memory` CLI subcommand.** Users need a documented, safe way to delete `memory.db`, `memory_vectors/`, and clear Redis keys. Without this, accumulated research history has no lifecycle management from the user's perspective.

8. **Consider encrypting `memory.db` and `memory_vectors/`.** SQLite with SQLCipher or a simple key-derivation wrapper would protect research history on shared or cloud-synced machines.

9. **Cap the `queries_archive` table.** Currently, eviction moves rows to `queries_archive` but never deletes from it. Over time this table grows unboundedly with old research topics.

10. **Sanitize the output filename.** The topic slug in the output filename can leak sensitive subject matter in directory listings. Consider hashing or using only a timestamp as the filename, with topic stored only inside the file.

11. **Add Redis password support to `config.py`.** The `RedisConfig` dataclass has no `password` field. On any system where Redis listens on a non-loopback interface, all blackboard data is unauthenticated. Add a `password: str = ""` field and pass it to `redis.Redis(password=...)`.
