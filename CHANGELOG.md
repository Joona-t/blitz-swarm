# Changelog

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
