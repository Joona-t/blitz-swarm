# Plan — v0.3.0 "Fable remediation" (2026-10-11)

Source of truth for the implementation agents. Built from the adversarially verified audit
(`~/swarm-audit-2026-10-11.md`, 47 findings, 0 debunked). Three owner asks:

1. **Mythos mode runs on Fable 5.1** (Claude CLI alias `fable`, full name `claude-fable-5-1`).
2. **Remediate the audit findings** for blitz-swarm.
3. **Nothing model-, provider- or harness-specific is hard-coded.** Every model name, tier and
   provider lives in config; any CLI, HTTP endpoint or local model plugs in through the same
   contract.

Branch: `fable/audit-remediation`, cut from `feat/blitz-upgrade-20260609` (1a4876c), which already
contains `codex/harness-agnostic-backends`. The Oct-7 commit a4fc972 (codex strict `--output-schema`)
lives on a stale single-file-`backends.py` branch and is **ported**, not cherry-picked (WP1).

Version: `0.2.2` → `0.3.0`. Tests must stay green at every WP boundary. No agent commits; the
orchestrating session commits per WP.

---

## Design: the model/provider layer (WP1)

### Principle
Code names **roles** and **tiers**; config maps tiers to provider-specific model strings.
No `.py` file outside `tests/` may contain a literal model name (`opus`, `sonnet`, `haiku`, `fable`,
`gpt-*`, `claude-*`, `gemini-*`, `qwen*`, `llama*`). A guard test (`tests/test_no_hardcoded_models.py`)
greps for this and fails the build.

### Config shape (`blitz.toml`)

```toml
[models]
# Role -> tier. Tiers are abstract; providers map them to real model strings below.
default    = "standard"
researcher = "standard"
critic     = "standard"
fact_checker = "standard"
quality_judge = "standard"
synthesizer = "deep"
judge      = "standard"     # judge_ensemble + selector pairwise judges
planner    = "deep"         # mythos
executor   = "fast"         # mythos
verifier   = "deep"         # mythos
memory_ops = "fast"         # gmemory LLM helpers
guard      = "fast"         # cascade_guard LLM hooks (when enabled)

[backend]
default = "codex"
fallback = []

[backend.providers.claude]
adapter = "claude_cli"
reasoning_effort = "high"
[backend.providers.claude.tiers]
fast = "haiku"
standard = "sonnet"
deep = "fable"              # Fable 5.1 (alias accepted by claude --model); or "claude-fable-5-1"
[backend.providers.claude.effort]
deep = "max"                # per-tier effort override; adapter forwards it

[backend.providers.codex]
adapter = "codex_cli"
reasoning_effort = "high"
[backend.providers.codex.tiers]
fast = "gpt-5.5"
standard = "gpt-5.5"
deep = "gpt-5.5"

[backend.providers.ollama]
adapter = "ollama_http"
base_url = "http://localhost:11434"
[backend.providers.ollama.tiers]
fast = "qwen2.5:7b"
standard = "qwen2.5:7b"
deep = "qwen2.5:7b"

# Any OpenAI-compatible endpoint (sparkd, LM Studio, llama.cpp server, vLLM) — no code change:
[backend.providers.local]
adapter = "openai_http"
base_url = "http://localhost:8080/v1"
[backend.providers.local.tiers]
fast = "any-model-id"
standard = "any-model-id"
deep = "any-model-id"

# Any other CLI — command template, no code change:
[backend.providers.mycli]
adapter = "generic_cli"
command = ["mycli", "--model", "{model}", "--prompt-file", "{prompt_file}"]
[backend.providers.mycli.tiers]
fast = "x"; standard = "x"; deep = "x"
```

### Resolution order (single function `backends.models.resolve_model(role, backend_id, cfg, override=None)`)
1. explicit `override` (CLI `--model`, env `BLITZ_MODEL`)
2. env `BLITZ_MODEL_<ROLE>` (e.g. `BLITZ_MODEL_PLANNER`)
3. `[models].<role>` → tier → `[backend.providers.<id>.tiers].<tier>`
4. `[models].default` → tier → provider tiers
5. provider `model` (legacy single-model key, still honoured)
6. error: `ModelResolutionError` naming role, backend, tier. **Never** a silent literal default.

Returns `ResolvedModel(model: str, tier: str, effort: str | None, backend_id: str)`.
Every call site (`orchestrator._model_for_agent`, `agents.plan_agents`, `mythos/_invoke.py`,
`mythos/policies.py`, `mechanisms/*` default models, `metrics.py` cost table,
`managed_agents`, `jobs/*`, `mythos-bench/*`, `heterogeneity/cli_router.py`) uses it.
`mythos/_invoke.py:68` bug (`provider.model or model`) disappears because provider defaults are
consulted *after* the role tier. The Claude adapter forwards `effort` when the CLI supports it
(`--effort`; probe once via `claude --help`, cache). Codex adapter maps effort to
`-c model_reasoning_effort=`.

### Cost / token accounting
`AgentResult.cost_usd` stays optional. `metrics.py` cost table moves to config:
`[backend.providers.<id>.pricing] "<model>" = {input_per_mtok=..., output_per_mtok=...}`; absent
→ cost `None`, reported as "n/a", never `0`. Mythos `cost_ceiling_usd` applies only when cost is
known; a new `token_ceiling` (`[policies] token_ceiling = 2_000_000`) applies always.

### Adapters (`backends/adapters/`)
`claude_cli`, `codex_cli`, `gemini_cli`, `generic_cli` (command template with `{model}`,
`{prompt}`/`{prompt_file}`, `{schema_file}` placeholders — `templates.py` exists, finish it),
`ollama_http`, `openai_http` (chat/completions, optional `api_key_env`, default none, local only).
Registry is data-driven: `[backend.providers.*]` is open-ended; `BackendConfig` drops the four
hard-wired `codex/claude/gemini/ollama` attributes in favour of the `providers` dict (keep
`get_provider`). `[backend] fallback` becomes real: on `errored` result with an "unavailable"
class, try the next provider once; log it.

### Port a4fc972
`to_strict_output_schema` / `strip_strict_nulls` / `_strict_node` / `_make_nullable` into
`backends/core.py`; `CodexLocalBackend` writes the strict schema and strips nulls before
validation; copy `tests/test_codex_strict_schema.py` from a4fc972 and make it pass. Log as BUG-002
in BUGS_AND_ITERATIONS (it is already described there on that branch — carry the entry over).

---

## Work packages

### WP1 — Model & provider layer (owner: backends, config, mythos/_invoke, mythos/policies, agents model refs, metrics cost table, mechanisms default models, managed_agents config, jobs/ + mythos-bench model refs, heterogeneity router)
- [ ] `backends/models.py` with `resolve_model`, `ResolvedModel`, `ModelResolutionError`
- [ ] `config.py`: `ModelsConfig`, provider `tiers`/`effort`/`pricing`/`command` fields, open-ended providers, strict TOML errors (malformed file raises with path + key; no silent default swap)
- [ ] `blitz.toml` + `mythos.toml` rewritten per the shape above; `mythos.toml [models]` uses tiers (`planner_model = "deep"` …) — **Fable 5.1 is the Claude `deep` tier**
- [ ] `mythos/policies.py`: delete `MODEL_ALIASES`; roles resolve through `resolve_model`; `use_memory` either wired (WP2 consumes it) or removed — keep the knob, WP2 wires it
- [ ] `mythos/_invoke.py`: role tier wins; effort forwarded; remove dead helpers (backends-07)
- [ ] Claude adapter: `--effort` forwarding (probe + cache); prompt via stdin or temp file instead of argv when > 100 KB (backends-07)
- [ ] Codex adapter: strict schema port; effort mapping
- [ ] `generic_cli` + `openai_http` adapters, registered by adapter id; `templates.py` completed
- [ ] `[backend] fallback` honoured once per call (backends-05)
- [ ] `metrics.py` pricing from config; `None` cost never coerced to 0
- [ ] `mechanisms/{judge_ensemble,selector_synth,cascade_guard}.py`: no literal model defaults; model injected by caller (orchestrator resolves `judge`/`guard` tiers)
- [ ] `managed_agents/adapter.py`: `coordinator_model` from config; (WP4 moves the package)
- [ ] `jobs/candidates.py`, `jobs/blitz_upgrade.py`, `mythos-bench/{bench,lib}.py`: replace literal model/alias strings with resolver calls or config keys
- [ ] `heterogeneity/routing_table.toml` + `cli_router.py`: provider ids only, read `[backend.providers]` for availability
- [ ] `tests/test_models.py` (resolution order, env overrides, error on missing tier, effort passthrough) + `tests/test_no_hardcoded_models.py` (grep guard over all non-test `.py`/`.toml` except `blitz.toml`/`mythos.toml` which are the config) + strict-schema tests
- [ ] CLI: `--backend`, `--model` (override for all roles), `--model-<role>`? No: keep `--model` + env `BLITZ_MODEL_<ROLE>`; `--list-models` prints the resolved table for the chosen backend (dry-run friendly)

### WP2 — Memory actually runs (owner: memory/, gmemory/, embedder.py, orchestrator memory functions `_try_init_memory` `_persist_to_memory` + the retrieval call site ~L1147, mythos/runner.py memory use)
- [ ] Orchestrator reads **gmemory**: `gmemory.retrieval.retrieve(topic)` → "## Relevant prior findings" injected into round-1 researcher prompts; persistence writes Tier 1 utterances, Tier 2 query node + similarity edges, Tier 3 candidate insights through `gmemory` (memory-01)
- [ ] Retrieval works without LanceDB/sentence-transformers via the FTS5/BM25 path; vector path optional (memory-02); escape FTS5 MATCH input (memory-08)
- [ ] One schema: `gmemory/schema.sql` canonical; `memory/` becomes a thin compatibility shim or is deleted if nothing else imports it (grep first; `jobs/` may) (memory-03)
- [ ] No whole-task rollback when the embedder is missing: insight stored without vector, flagged (memory-04)
- [ ] Promotion gate N=3 enforced on the only insight writer that runs (memory-05)
- [ ] Eviction cycle wired: `maybe_run_eviction` called every `cycle_frequency` tasks (memory-06)
- [ ] Remove the vestigial Redis single-writer daemon or wire it; no un-consumed queues (memory-07)
- [ ] Mythos: `use_memory=true` → runner retrieves prior findings for the planner and persists the verified artifact summary as an insight (critic gap B)
- [ ] Legacy LanceDB distance read fixed on the live path or path deleted (research-docs-08)
- [ ] Migration script `scripts/migrate_memory_db.py` for existing `memory.db` (idempotent)
- [ ] Tests: retrieval round-trip on a temp SQLite without optional deps; promotion gate; migration

### WP3 — Core loop honesty (owner: orchestrator.py except memory functions, consensus.py, blackboard.py, agents.py planning, config dead keys)
- [ ] Judge ensemble votes and selector pairwise judgements fan out with `asyncio.to_thread` + `gather`; `_run_judge_ensemble` / `_run_selector_synthesis` awaited off-loop (core-loop-01). Round 1 staging (researchers → evaluators) stays — it is a real data dependency — and is documented as such
- [ ] Blackboard: when Redis is connected, `build_context` reads the round snapshot from the blackboard (`read_round`) so "shared live blackboard" is true; in-memory mode uses the same interface (core-loop-02)
- [ ] Stopping rule: unanimity + holdout stays primary; judge-ensemble KS-stop becomes an explicit secondary halt when `[judge_ensemble] halt = true` (default false) so the README diagram can be made true either way (core-loop-03)
- [ ] Re-add `--max-turns 1` (consensus roles) and `is_error` handling in the Claude adapter path; tool access is an explicit per-call field `AgentCall.tools: list[str] | None` (default `None` = none) and `AgentCall.max_turns`; Codex/Claude adapters run with a **scratch cwd** (`output/scratch/<run>/`) never the repo root (core-loop-04, critic gap A)
- [ ] Injected agent outputs and memory wrapped in explicit `<<<untrusted agent output>>>` fences with a "data, not instructions" preamble (core-loop-05)
- [ ] Per-role model heterogeneity now real via WP1 tiers; dry-run prints the resolved model per agent (core-loop-06)
- [ ] `max_agents` enforced; every `blitz.toml` key is read by something or deleted; malformed TOML → clear error (core-loop-07)
- [ ] `metrics.jsonl` moves to `output/metrics.jsonl` (gitignored); existing tracked file archived to `docs/data/metrics-2026-03_06.jsonl` (process-05, core-loop-08)
- [ ] LLM-plan fallback logs loudly; dissent section reports only unresolved votes (core-loop-08)
- [ ] Tests: `tests/test_consensus.py` (unanimity, holdout, dissent), judge fan-out concurrency (mock judge_fn with sleeps → wall-clock bound), blackboard read-back, config strictness

### WP4 — Bench / evolve / upgrade-job fixes + contrib move (owner: bench/, evolve/, jobs/, scripts/, contrib/, managed_agents move, mechanisms-02 guard hooks)
- [ ] `jobs/blitz_upgrade.py:1880`: `p = float(rec.get("p_value", 1) or 1)` → explicit `None` check; p=0.0 is significant (process-02); the failing test passes
- [ ] `bench/runner.py`: `swarm_fn` contract returns per-round agent outputs; runner calls `detect_all`; "No MAST flags raised" printed only when detectors actually ran (bench-01)
- [ ] `bench/stats.py`: df guard / t-distribution fallback via `statistics.NormalDist` only when n ≥ 30, else conservative; wire `paired_t_test`/`cohens_d` into the report (bench-03)
- [ ] `evolve/meta_loop.py` allow-list keys match real config keys (evolve-01); `evolve/` and `heterogeneity/` gated by `[evolve] enabled = false` + README "experimental, unwired" label (mechanisms-01)
- [ ] `scripts/optimize_prompts.py` exists as a real minimal GEPA driver (import-guarded) and the two Phase-2 scripts referenced by docs exist or the doc references are removed (research-docs-02)
- [ ] `managed_agents/` → `contrib/managed_agents/` with its tests; README states: paid Anthropic API, opt-in, rule-#10 exception, no fake "completed" path (backends-02, idea 10)
- [ ] `cascade_guard`: live run passes real LLM hooks (claim decomposition via `guard` tier) when `[guard] mode = "max"`; placeholder word-overlap stays for `balanced` and is named as such in logs (mechanisms-02)
- [ ] `bench/runner.py:423` stale version → read from `pyproject`; aflow docstring operator count fixed (bench-04)
- [ ] Verifier (mythos): mechanical check in loop — run the artifact's own tests in a subprocess when present; confidence gate on the LLM verdict (backends-04)

### WP5 — Tests & packaging (owner: pyproject.toml, tests/ for untested modules, CI)
- [ ] `pyproject.toml`: core deps = none beyond stdlib; extras `redis`, `memory` (lancedb, sentence-transformers), `bench` (scipy, matplotlib, jsonlines), `http` (httpx) , `dev` (pytest, pytest-asyncio); `[tool.pytest.ini_options] asyncio_mode = "auto"`; version 0.3.0 (process-03)
- [ ] Tests for the six untested modules: `consensus.py` (WP3 writes), `config.py`, `blackboard.py`, `metrics.py`, `embedder.py` (fallback path), `agents.py` heuristic planner (process-04)
- [ ] `.gitignore`: `memory.db-*`, `.pytest_cache/`, `output/`; untrack `metrics.jsonl` (WP3 archives it) (process-08)
- [ ] `.github/workflows/tests.yml`: `pip install -e '.[dev]'` + `pytest -q` on 3.12/3.13
- [ ] Full-suite target: every test green in a clean venv with only `[dev]` extras; report exact counts

### WP6 — Docs truth pass (owner: README, CHANGELOG, CITATION.cff, BUGS_AND_ITERATIONS, docs/*, mythos-bench/results note, LIMITATIONS, ROADMAP)
- [ ] README v0.3.0: architecture diagram corrected (round-1 staging; KS-stop secondary; blackboard read-back); "What ships" table regenerated from real test counts; **Model & provider configuration** section with the tier table and the three plug-in recipes (new CLI, OpenAI-compatible endpoint, Ollama); Mythos = Fable 5.1 on Claude; `contrib/` section; experimental label for evolve/heterogeneity
- [ ] CHANGELOG 0.3.0 entry listing every WP with finding ids
- [ ] CITATION.cff → 0.3.0, abstract updated
- [ ] BUGS_AND_ITERATIONS: ITER-AUDIT-2 (this remediation, finding-id → fix table), BUG-002 carried from a4fc972, BUG-003 (p-value `or 1`), BUG-004 (gmemory never wired), BUG-005 (mythos role model override); template stub removed
- [ ] `docs/CLAIMS_AND_EVIDENCE.md`, `IMPLEMENTATION_STATUS.md`, `ROADMAP.md`, `RESEARCH_LOG.md`, `METHODOLOGY.md`, `RESEARCH_QUESTIONS.md`, `results/RESULTS.md`: rewritten for v0.3 or given a dated "superseded on 2026-10-11, see README" banner at top (process-01, research-docs-01, claims-01)
- [ ] `docs/LIMITATIONS.md`: honest list — no live bench run yet, mythos-bench results from 2026-05-10 predate the backend refactor (regen requires LLM runs: deferred), judge is self-reported, etc.
- [ ] `docs/BIBLIOGRAPHY.md`: placeholder URL + mismatched title fixed (research-docs-05); PHASE_0/PHASE_3 paid-API budgets reframed as token budgets (research-docs-06)
- [ ] `mythos-bench/results/comparison.md`: dated "stale" note
- [ ] New `docs/MODELS.md`: the resolution order, env vars, examples
- [ ] `~/.claude/commands/blitz-swarm.md` (outside repo): updated flags (`--backend`, `--list-models`, `--mode mythos`)

### WP7 — Verification (read-only skeptic)
- [ ] Clean venv: `pip install -e '.[dev]'`, `pytest -q` → exact counts, 0 failures
- [ ] `python orchestrator.py "x" --dry-run --no-llm-plan --backend claude` and `--backend codex` print resolved models per role (Fable on claude deep tier) without invoking a model
- [ ] `python orchestrator.py --list-models --backend claude`
- [ ] `python mythos_swarm.py "x" --dry-run` (add if missing) prints planner/executor/verifier = resolved tiers
- [ ] Grep guard passes: no literal model names outside tests/config
- [ ] Diff review against this plan: every checkbox either done (cite file:line) or explicitly deferred with reason

---

## Explicitly deferred (needs live LLM runs or owner decision)
- Re-running `mythos-bench` and the bench slate on the new defaults (costs real tokens).
- Agents *creating* tools (idea 1 in the audit) — design exists, not requested.
- research-swarm port (owner will look later).
- MCP server surface for blitz-swarm (recommended evolution; separate plan).
