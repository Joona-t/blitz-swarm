# Changelog

## [0.3.0] — 2026-10-11 "Fable remediation"

Driven by the adversarial audit of 2026-10-11 (`~/swarm-audit-2026-10-11.md`, Part 1; 47 findings,
0 debunked) and the plan of record `docs/plans/2026-10-11-fable-remediation.md`. Finding ids in
parentheses. Three owner asks: Mythos on Fable 5.1, remediate the findings, nothing model-, provider-
or harness-specific hard-coded. Trail: `BUGS_AND_ITERATIONS.md` ITER-AUDIT-2.

### WP1 — Model & provider layer (backends-01, backends-05, backends-07, core-loop-06)

- `backends/models.py`: `resolve_model(role, backend, cfg)` with a fixed order — `--model`/`BLITZ_MODEL`
  > `BLITZ_MODEL_<ROLE>` > explicit tier > `[models].<role>` tier > `[models].default` tier > provider
  legacy `model` > `ModelResolutionError`. Code names roles and tiers; `blitz.toml` maps tiers to models.
- `config.py`: open-ended `[backend.providers.*]` with `tiers`/`effort`/`pricing`/`command`; strict loader
  (unknown key, bad type, undefined provider raise `ConfigError` naming path + key).
- Adapters: `claude_cli` (`--effort` forwarding, stdin prompt > 100 KB), `codex_cli` (strict
  `--output-schema` port of a4fc972 = BUG-002, effort mapping), `gemini_cli`, `ollama_http`,
  `generic_cli` (argv template), `openai_http` (any local OpenAI-compatible server). Registry keyed by
  adapter id; `[backend] fallback` honoured once per call on unavailable-class errors.
- Mythos: planner/verifier = `deep` tier = `fable` (Fable 5.1) with effort `max` on claude; executors =
  `fast`. `mythos/_invoke.py` provider-model-overrides-role bug fixed (BUG-005). `token_ceiling` applies
  always; cost ceiling only when cost is reported; unknown cost is `None`, printed `n/a`.
- `metrics.py` pricing from `[backend.providers.<id>.pricing]`; `mechanisms/*` take the model from the
  caller; `jobs/*`, `mythos-bench/*`, `heterogeneity/` resolve through the layer.
- `tests/test_no_hardcoded_models.py` greps the tree and fails on any literal model name outside
  `tests/`, `blitz.toml`, `mythos.toml`. CLI: `--backend`, `--model <model|tier>`, `--list-models`.

### WP2 — Memory runs (memory-01..08, research-docs-08, critic gap B)

- `gmemory/store.py` `MemoryStore` is the only memory implementation: opens/migrates (one-time
  `*.pre-gmemory.bak` of a legacy file), retrieves on stdlib `sqlite3` + FTS5 with no optional deps,
  persists all three tiers, enforces the N=3 promotion gate, runs eviction every `[eviction]
  cycle_frequency` tasks (BUG-004 — gmemory had zero production callers).
- Wired into `orchestrator.run_swarm` (retrieval before round 1, persistence after synthesis) and Mythos
  (`use_memory`). FTS5 MATCH input escaped. Missing embedder never aborts a write. `memory/` is a
  32-line re-export shim; the Redis single-writer daemon and the LanceDB L2-as-cosine read are gone.
- `scripts/migrate_memory_db.py` migrates old databases idempotently (`--check` reports only).

### WP3 — Core loop honesty (core-loop-01..08, critic gap A)

- Judge-ensemble votes and selector pairwise judgements fan out via `asyncio.to_thread` + `gather`.
- Round-1 staging (researchers → evaluators → judges) is a data dependency and is printed as one;
  round 2+ is a single blast.
- The blackboard (Redis or in-memory, same interface) is the only round store and is read back for every
  context. KS-stop is an explicit secondary halt behind `[judge_ensemble] halt` (default false).
- `AgentCall.tools`/`max_turns`: consensus roles run tool-less (`--max-turns 1`) in
  `output/scratch/<run_id>/`, never the repo root. All injected agent output and prior findings are fenced
  as untrusted data (`context.py`). `max_agents` enforced; dead config keys removed; malformed TOML raises.
- LLM-plan fallback logs loudly; the dissent section reports unresolved votes only (`consensus.final_votes`).
- Tracked `metrics.jsonl` archived to `docs/data/metrics-2026-03_06.jsonl`; runs write
  `output/metrics.jsonl` (gitignored).

### WP4 — Bench / evolve / jobs / contrib (process-02, bench-01, bench-03, bench-04, evolve-01, mechanisms-01, mechanisms-02, backends-02, backends-04)

- BUG-003: `jobs/blitz_upgrade.py` p-value `or 1` coerced p = 0.0 to 1.0 and dropped significant
  techniques; now an explicit `None` check.
- `bench/runner.py` runs `detect_all` on per-round outputs and says when it did not; "No MAST flags"
  only when detectors ran. `bench/stats.py` uses the exact Student-t for df < 30 (normal approximation only
  at df ≥ 30); paired comparison wired into the report. Runner reads the version from `pyproject.toml`.
- `[evolve] enabled = false` gates `evolve/`, `heterogeneity/` and `scripts/optimize_prompts.py` (now a
  real import-guarded GEPA driver); `meta_loop` allow-list names real config keys; all labelled
  EXPERIMENTAL, UNWIRED.
- `managed_agents/` → `contrib/managed_agents/` (paid API, opt-in, rule-#10 exception); the fake
  `completed` path is removed — `run_task` raises `NotImplementedError` with a client attached.
- Cascade guard: `[guard] mode = "max"` runs real LLM claim decomposition on the `guard` tier;
  `balanced` is logged as the heuristic screen it is.
- Mythos verifier demotes low-confidence passes; running the artifact's own tests in a subprocess is opt-in (`verifier_run_tests = false` by default — it executes model-written code; see BUG-006).

### WP5 — Tests & packaging (process-03, process-04, process-08)

- `pyproject.toml` version 0.3.0, extras, `.gitignore`, CI `.github/workflows/tests.yml`; tests for the
  previously untested modules — `tests/test_config.py` (7 → 26), new `test_agents_planner.py` (19),
  `test_embedder.py` (12), `test_metrics.py` (28), on top of WP3's `test_consensus.py` (9),
  `test_blackboard.py` (14) and WP1's `test_models.py` (38).

### WP6 — Docs truth pass (claims-01, process-01, research-docs-01, research-docs-02, research-docs-05, research-docs-06)

- README rewritten for v0.3.0: corrected architecture diagram, "What ships" from real per-file test
  counts, model/provider section with the role → tier table and three plug-in recipes, Mythos section,
  experimental + contrib sections, limitations.
- New `docs/MODELS.md` (resolution order, env vars, adapters, real `--list-models` / `--dry-run` output).
- `docs/CLAIMS_AND_EVIDENCE.md`, `IMPLEMENTATION_STATUS.md`, `ROADMAP.md`, `LIMITATIONS.md` rewritten;
  `RESEARCH_LOG.md`, `METHODOLOGY.md`, `RESEARCH_QUESTIONS.md`, `EVALUATION.md`, `ABLATIONS.md`,
  `ARTIFACTS.md`, `results/RESULTS.md` and `docs/research/PHASE_*.md` carry a dated "superseded" banner
  and every reference to a file that does not exist is marked.
- `docs/BIBLIOGRAPHY.md`: placeholder URL removed, titles replaced with the arXiv-verified ones,
  `managed_agents` path updated. PHASE_0/PHASE_3 paid-API dollar budgets reframed as token budgets.
- `BUGS_AND_ITERATIONS.md`: BUG-003/004/005 renumbered per plan, ITER-AUDIT-2 finding → fix table,
  template stub removed. `CITATION.cff` → 0.3.0. `mythos-bench/results/comparison.md` dated stale note.
  Founding design docs (`BLITZ-SWARM.md`, `MYTHOS-SWARM*.md`) get a historical banner.

### Tests

745 passed, 14 conditional skips, 0 failed in a clean venv with only `pytest` + `pytest-asyncio`
(759 collected; 667 / 14 at the WP1–4 commit 36767cd before WP5's tests). No test invokes a model.

### Known stale / deferred

- No live bench run (any version). `mythos-bench/results/` (4 rows, 2026-05-10) predates this release and
  was not regenerated — needs live LLM runs.
- The 10M-token upgrade job stalled at phase 1/5 on 2026-06-10; harness merged, run not completed.
- `evolve/` + `heterogeneity/` have never run outside tests.

## [0.2.2] — 2026-05-21

### Harness-agnostic backend package

- Replaced the single `backends.py` module with a `backends/` package containing the core call/result contract, registry, CLI adapters, Ollama HTTP adapter, and harness adapter template.
- Added `RuntimePolicy` and `BackendCapabilities` while keeping the existing `AgentCall` and `AgentResult` fields compatible.
- Added provider registry config under `[backend.providers.<id>]` with compatibility loading for legacy `[backend.codex]`, `[backend.claude]`, and `[backend.gemini]` sections.
- Removed backend-specific Claude branches from orchestration, planning, memory helpers, and Mythos invocation; adapter selection and model mapping now live in the backend layer.
- Made `--backend` accept any configured provider id, not only the built-in CLI names.

### Tests

264 passing, 14 conditional skips:

- Added registry tests for custom adapter registration and built-in adapter resolution.
- Added provider config migration tests covering registry-style and legacy backend config.
- Added Ollama fake-HTTP tests for one-call local harness behavior and normalized telemetry.

## [0.2.1] — 2026-05-20

### Codex-first orchestration

- Added `backends.py` with `AgentCall`, `AgentResult`, and local adapters for Codex, Claude, and Gemini.
- Made Codex CLI the default backend in `blitz.toml`, using read-only sandboxing, ephemeral sessions, JSONL output, final-message files, temp schema files, and local schema validation.
- Routed consensus agents, Mythos invocations, planning helpers, memory LLM helpers, judge ensemble hooks, and selector hooks through the backend layer.
- Added CLI/config controls: `--backend`, `--quality-profile`, `--sandbox`, `--no-selector`, `--no-judge-ensemble`, and `--no-cascade-guard`.
- Wired `cascade_guard`, `judge_ensemble`, selector synthesis, and conservative AgentDropout-style contribution pruning into the main run loop.
- Added pytest collection hygiene so generated `mythos-bench/artifacts/**/test_*.py` files are not collected by default.

### Tests

260 passing, 14 conditional skips:

- Added backend tests for Codex command construction, JSONL/final-message parsing, schema validation, timeout handling, missing CLI handling, and malformed intermediate output precedence.
- Added orchestration wiring tests for backend routing, cascade filtering, judge ensemble output, selector synthesis, AgentDropout floors, and no direct active-path Claude subprocess calls.

## [0.2.0] — 2026-05-09

### Frontier methodology + mechanism + memory + recursion upgrade

Lands the full v0.2 plan documented in `research.md`, `plan.md`, and the
four implementation deep dives in `docs/research/PHASE_*.md`. Every
mechanism is keyed to a verified arXiv paper from 2025-2026; full
citation backbone in `docs/BIBLIOGRAPHY.md`.

### Phase 0 — Methodology backbone

- 30-prompt research-synthesis bench slate (`bench/slate_v1.toml`):
  15 easy / 10 medium / 5 hard, spanning technical / open-ended /
  adversarial / multi-domain / compositional. Five adversarial,
  six multi-domain, five compositional. Slate hash pinned for
  reproducibility.
- `bench/runner.py` — dependency-injected `swarm_fn` so tests run
  without API tokens. Per-prompt JSONL log with fsync. Resumable
  via `_resumable_ids`. Per-run + total-budget cost gates.
- `bench/detectors.py` — 9 of 14 MAST failure-mode detectors
  (Cemri 2503.13657, NeurIPS 2025), rule-based, no LLM cost.
- `bench/mast_regression.py` — 14 synthetic scenarios producing
  `bench/mast_scoreboard.md`. v0.1.1 baseline: 9/14 detected, 5
  flagged for orchestrator integration in alpha.2.
- `bench/stats.py` — paired t-test, Cohen's d_z, bootstrap CI,
  Welch's t, Mann-Whitney U. Scipy-optional with normal-CDF fallback.

### Phase 1 — Mechanism upgrades

- `mechanisms/cascade_guard.py` — genealogy-graph fault containment
  (Xie 2603.04474, Mar 2026). LLM-free skeleton with pluggable
  decompose/screen hooks. Modes: off / speed / balanced / strict.
  BFS over `parent_claim_ids` reverse-edges propagates taint to all
  descendants of an errored agent.
- `mechanisms/judge_ensemble.py` — N=3 judges with empirical-CDF
  KS-test halting (Hu 2510.12697, NeurIPS 2025). Halt reasons:
  ks_stable / unanimous_saturation / max_rounds / all_judges_errored.
  Aggregators: mean / median / trimmed_mean.
- `mechanisms/selector_synth.py` — Bradley-Terry MLE via
  Minorization-Maximization (Hunter 2004) for pairwise span selection
  (Maryanskyy 2603.20324, Mar 2026). H2-aware section splitting,
  position-bias rotation, homogeneous-collapse warning.
- `prompts/general/critic_{factual,logical,counterfactual,steelman}.md`
  — MAR persona-typed critic prompts (arXiv 2512.20845, Dec 2025).
- `prompts/loader.py` — domain preset registry with sha256 trace
  hashing and cached load. `assign_personas()` returns the right
  persona mix for each round / dissent state.
- Crypto prompts moved to `prompts/crypto/` verbatim. v0.1.x
  behavior preserved with `domain = "crypto"`.

### Phase 2 — G-Memory Tier 2/3

- `gmemory/schema.sql` — additive SQLite migration (FTS5 over
  queries + insights, insight_candidates holding table,
  insight_query_index, query_tags, insight_tags).
- `gmemory/query_graph.py` — Tier 2 task-level memory with
  kNN edge formation. LanceDB-optional with brute-force-Python
  fallback. Hash-based deterministic embedding stub when
  sentence-transformers is missing.
- `gmemory/insight_graph.py` — Tier 3 promoted insights with
  hyperedges to validating Ω query sets. Pluggable `distill_fn`
  with content dedup.
- `gmemory/promotion.py` — N=3 distinct-query support gate via
  single-link agglomerative clustering on cosine (threshold 0.78).
  Honest GAM-adaptation documented in module docstring.
- `gmemory/retrieval.py` — six-step pipeline: embed → hybrid
  RRF → 1-hop → insight overlap → LLM relevance (cold-start gated)
  → sparsification → format. All LLM hooks pluggable.
- `gmemory/hybrid.py` — RRF fusion (k=60) of vector + BM25.
  FTS5 availability probe.
- `gmemory/meta.py` — read-only meta-loop tag API across five
  axes (domain / pattern / pitfall / swarm / meta) with schema
  validation.
- BUG-001 fix: LanceDB `distance_type("cosine")` set explicitly
  for normalized MiniLM embeddings.

### Phase 3 — Recursive self-improvement

- `evolve/aflow_search.py` — MCTS over `SwarmGraph` with 6
  operators (AddResearcher, RemoveCritic, AddDebateRound,
  SwapSynthForSelector, AddJudgeEnsemble, IncreaseRounds).
  UCB1 + canonical fingerprint dedup. Liu 2410.10762, ICLR 2025.
- `evolve/meta_loop.py` — `ConfigPatch` proposer with
  `ALLOWED_KEY_PATHS` allow-list. Auto-merge gates: Cohen's d_z,
  per-dim regression, cost ratio, holdout floor. High-risk paths
  escalate to L3. `RecursionLevelExceeded` enforces L3 ceiling.
- `evolve/gepa_adapter.py` — adapter for the gepa-ai/gepa library
  (lazy-imported). GEPA arXiv 2507.19457 (ICLR 2026 oral). Per-role
  reflective feedback drawn from quality_judge breakdown.
- `heterogeneity/cli_router.py` — claude/codex/gemini routing.
  TOML-driven, fallback to claude on missing CLI. Rule-#10
  compliant (existing subscriptions, no paid API keys).
- `managed_agents/adapter.py` — opt-in Anthropic Managed Agents
  adapter (May 7 2026 beta, header `managed-agents-2026-04-01`).
  Coordinator depth=1, max specialists=20, per-run + per-day
  spend caps with fail-closed.

### Phase 4 — Methodology + docs

- `docs/BIBLIOGRAPHY.md` — full citation backbone keyed to v0.2
  modules. Anchors that drive implementation in **bold**.
- `README.md` rewritten — replaces n=2 disclaimer with current
  v0.2 status; honest about what's tested vs needs orchestrator
  integration.
- `BUGS_AND_ITERATIONS.md` — BUG-001 (LanceDB distance type),
  ITER-001 (test-package shadowing), ITER-002 (zero-variance
  paired-stats edge case).

### Tests

238 passing, 14 conditional skips. Across 9 test files:

- `tests/bench/test_slate.py` (13)
- `tests/bench/test_detectors.py` (16)
- `tests/bench/test_runner.py` (15)
- `tests/bench/test_stats.py` (11)
- `tests/bench/test_mast_regression.py` (20 + 14 cond skips)
- `tests/test_prompt_loader.py` (24)
- `tests/test_cascade_guard.py` (23)
- `tests/test_judge_ensemble.py` (23)
- `tests/test_selector_synth.py` (21)
- `tests/test_gmemory.py` (31)
- `tests/test_phase3.py` (41)

### Honest deferred work

The v0.2.0 release ships the modules + tests. The orchestrator
integration of Phases 1-3 + a real n≥20 baseline run land in
alpha.2 — gated on actual API spend.

## [0.1.1] — 2026-05-09

### Reliability and observability

- Agent retry loop on malformed JSON output (1 retry, then graceful error)
- Partial-output recovery on subprocess timeout (parses any JSON the agent flushed before SIGKILL)
- Cost-and-token extraction from the Claude CLI envelope (per-agent, per-round)
- Trace IDs (UUID4) attached to every agent invocation for cross-log correlation
- New `metrics.py` module: `MetricsCollector`, `RunMetrics`, `RoundMetrics` dataclasses; per-run JSONL log at `metrics.jsonl`
- `--max-turns 3` cap on every agent subprocess to prevent runaway tool loops
- Quality-judge schema extended with explicit `coverage_score` / `accuracy_score` / `clarity_score` / `depth_score` (0-10 each)

### Configuration

- `blitz.toml` tuned for production-grade research runs: `max_rounds = 4`, `timeout_seconds = 300`, `max_agents = 10`
- Configuration now loaded centrally via `config.py` instead of hard-coded constants in `orchestrator.py`

### Domain specialization (interim)

- Role prompts specialized for crypto/quant trading research as the v0.1.1 default. v0.2 generalizes this and moves crypto into a preset registry.

## [0.1.0] — 2026-03-12

### Initial Release

- Parallel multi-agent orchestration via asyncio + subprocess
- 5 agent roles: researcher, critic, fact-checker, quality judge, synthesizer
- Dynamic agent planning (LLM-based with heuristic fallback)
- Consensus convergence with holdout override
- Dissent preservation in final output
- Redis-backed blackboard with no-Redis fallback
- G-Memory Tier 1 (interaction traces)
- Sentence-transformer embeddings (MiniLM-L6-v2)
- TOML-based configuration
- Per-run metrics logging

### Research Documentation (same day)

- Added memory architecture literature review
- Added G-Memory blueprint documentation
