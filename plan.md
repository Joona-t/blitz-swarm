# Plan — Blitz-Swarm v0.2 "Frontier"

**Source document:** `research.md` (read first).
**Status:** Operational. Derived from research.md.
**Methodology:** Test-driven where feasible (each mechanism has pytest cases written first). Atomic commits per logical step. Tag at each phase milestone.

---

## Conventions

- **Commit cadence:** every logical green step (passing tests + working unit) gets its own commit. Push after each commit.
- **Commit messages:** imperative voice, one-line subject ≤ 72 chars, body explains *why*. No `Co-Authored-By` (CLAUDE.md rule #6).
- **Tags:** `v0.2.0-alpha.N` per phase milestone; `v0.2.0` only after Phase 4 success criteria met.
- **Branch:** `main`. No feature branches unless a phase grows past 2 weeks of work.
- **TDD:** for `mechanisms/`, `gmemory/`, `evolve/`, `bench/` — pytest cases written before implementation. For `prompts/` and docs, no tests; review-only.
- **BUGS_AND_ITERATIONS.md:** every bug discovered + fixed during implementation gets a dated entry.
- **Verification before completion:** every phase milestone runs `pytest tests/` + the bench harness before claiming done. Never claim "passing" without seeing green output.

---

## Pre-flight checklist (one-time, before Phase 0)

- [x] Commit v0.1.1 WIP (crypto specialization + retries + metrics) — done in commit `d2826d8`.
- [x] Tag `v0.1.1` and push — done.
- [x] Scaffold v0.2 directory tree (`prompts/`, `mechanisms/`, `gmemory/`, `bench/`, `evolve/`, `heterogeneity/`, `managed_agents/`, `scripts/`).
- [x] Write `research.md` and `plan.md`.
- [ ] Move crypto prompts from `agents.py` inline strings to `prompts/crypto/*.md`.
- [ ] Write initial generalized prompts to `prompts/general/*.md` (will be refined by GEPA in Phase 3).
- [ ] Refactor `agents.py` to load prompts from `prompts/<domain>/<role>.md` based on `blitz.toml`'s `domain` flag.
- [ ] Add `pytest`, `pytest-asyncio`, `scipy`, `matplotlib`, `jsonlines`, `lancedb` to `pyproject.toml`.
- [ ] Initialize `tests/` directory with `tests/conftest.py` shared fixtures.
- [ ] Tag `v0.2.0-alpha.0` on the first scaffold commit.

---

## Phase 0 — Bench harness (the foundation)

**Estimated effort:** 1.5–2 weeks. **Blocks:** Phases 1, 2, 3 (any A/B testing).

**Goal:** A reproducible, statistically sound benchmark that any future swarm config can be scored against. Without this, every later phase produces vibes, not evidence.

### Tasks

#### P0.1 — Bench slate
- [ ] Define 30-prompt slate in `bench/slate_v1.toml` covering: easy/medium/hard tiers, 5 adversarial prompts, 5 multi-domain, 5 compositional. Each with `expected_coverage_areas`, `difficulty`, `domain`, `parallelizable: bool`.
- [ ] Pytest: `tests/bench/test_slate.py` validates schema, uniqueness, distribution.

#### P0.2 — Runner skeleton
- [ ] `bench/runner.py` with `BenchConfig` dataclass and `run_slate(config: BlitzConfig, slate_path: Path) -> RunSummary`.
- [ ] Per-prompt JSONL log to `bench/runs/<run_id>/<prompt_id>.json` (atomic writes for resume).
- [ ] Per-run summary to `bench/runs/<run_id>/summary.json` with reproducibility fields (seed, model versions, prompt-file hash, blitz.toml hash).
- [ ] Pytest: `tests/bench/test_runner.py` with mocked agent invocations validates flow + reproducibility.

#### P0.3 — Cost & resume controls
- [ ] Per-run budget cap (USD), per-prompt timeout, auto-skip if remaining budget < estimate.
- [ ] Resume capability if interrupted (atomic append to `bench/runs/<run_id>/state.json`).
- [ ] Pytest: `tests/bench/test_resume.py` simulates SIGINT mid-run, asserts resume continues from last completed prompt.

#### P0.4 — Statistical analysis
- [ ] `bench/stats.py` with paired t-test, Cohen's d, bootstrap 95% CI (1000 resamples).
- [ ] Outputs per-dim quality means, stddevs, paired comparisons against a baseline run.
- [ ] Generates `bench/runs/<run_id>/stats.md` markdown report.
- [ ] Pytest: `tests/bench/test_stats.py` against synthetic data validates t-test, Cohen's d, bootstrap.

#### P0.5 — Visualization
- [ ] `bench/charts.py` with matplotlib: per-run quality-by-dim bar chart, cross-run trend chart, parallel-vs-sequential paired chart.
- [ ] Saves PNGs to `bench/runs/<run_id>/charts/`.

#### P0.6 — Parallel-vs-sequential
- [ ] `bench/parallel_vs_sequential.py` per Shen 2603.29632 methodology.
- [ ] Same total-token budget; runs blitz-swarm parallel + a sequential pipeline (single researcher → critic → fact-checker → synthesizer) on each slate prompt.
- [ ] Outputs paired wins/losses/ties + task-class breakdown (parallelizable vs sequential per Google scaling paper).
- [ ] Pytest: `tests/bench/test_pvs.py` validates pairing logic + budget enforcement.

#### P0.7 — MAST regression suite (14 modes)
- [ ] `bench/mast_regression.py` with 14 named pytest cases:
  1. Step Repetition
  2. Loss of Conversation History
  3. Premature Termination
  4. Information Withholding
  5. Wrong Agent Acting
  6. Disregard for User Input
  7. No or Inadequate Response Verification
  8. Failure to Ask for Clarification
  9. Incorrect Task Specification
  10. Ignore Other Agents' Input
  11. Reasoning-Action Mismatch
  12. Tool Misuse
  13. Unintended Output / Hallucinated Tool Calls
  14. Conversation Reset
- [ ] Each test injects the failure (e.g., kill subprocess for #3 Premature Termination) and asserts orchestrator either contains the failure or surfaces it cleanly. v0.1.1 baseline expected to fail many; v0.2 with `cascade_guard` expected to pass ≥ 12.

#### P0.8 — Baseline run
- [ ] Run full slate on v0.1.1 config (`git checkout v0.1.1 && bench/runner.py`).
- [ ] Save as `bench/runs/baseline_v0.1.1/`.
- [ ] Run MAST regression on v0.1.1; record passing modes (expected: low — that's what motivates Phase 1's cascade_guard).

#### P0.9 — CI integration
- [ ] GitHub Actions workflow at `.github/workflows/bench.yml`: runs MAST regression + 5-prompt smoke slate on every PR; gates merge on regression-suite pass.
- [ ] Full 30-slate runs are manual (`make bench`).

### Phase 0 success criteria
- [ ] All pytest cases green on `pytest tests/bench/`.
- [ ] Baseline run completes on v0.1.1 with summary.json and stats.md generated.
- [ ] MAST regression scoreboard published for v0.1.1 baseline.
- [ ] Tag `v0.2.0-alpha.1` on Phase 0 completion.

---

## Phase 1 — Mechanism upgrades

**Estimated effort:** 2 weeks. **Depends on:** Phase 0.

**Goal:** Land four frontier mechanisms + generalized prompts. Each mechanism A/B-tested against the v0.1.1 baseline using Phase 0 bench.

### Tasks

#### P1.1 — Move crypto prompts to preset registry
- [ ] Extract `ROLE_PROMPTS` from `agents.py` into `prompts/crypto/researcher.md`, `critic.md`, `fact_checker.md`, `quality_judge.md`, `synthesizer.md` verbatim.
- [ ] Refactor `agents.py` to load prompts from `prompts/<domain>/<role>.md` based on `blitz.toml`'s `domain` flag (default `general`).
- [ ] Pytest: `tests/test_prompt_loader.py` validates fallback to `general` if domain missing, errors on missing prompt files.

#### P1.2 — Generalized role prompts (initial)
- [ ] Write `prompts/general/researcher.md`, `fact_checker.md`, `quality_judge.md`, `synthesizer.md` — domain-agnostic versions of the v0.1.1 prompts. (GEPA in Phase 3 will refine these against bench.)
- [ ] No automated tests; review-only. Run smoke bench on v0.1.1 + general prompts; aggregate quality should be ≥ 80% of crypto prompts on crypto-flavored slate prompts.

#### P1.3 — Persona critic templates (MAR)
- [ ] Write `prompts/general/critic_factual.md`, `critic_logical.md`, `critic_counterfactual.md` — 3 personas per arXiv 2512.20845.
- [ ] Update `agents.py` planner: when `critic_count > 1`, rotate through personas. Single critic gets `critic_factual` (matches v0.1.1 default behavior).
- [ ] A/B: HumanEval-style coding prompts + multi-hop reasoning prompts on the slate. Target: +3 pts on aggregate quality_judge over single-critic baseline.

#### P1.4 — `cascade_guard` (Xie 2603.04474)
- [ ] `mechanisms/cascade_guard.py` with:
  - `tag_message(content: str, parent_ids: list[str], agent_state: AgentState) -> TaggedMessage`
  - `mark_descendants_tainted(taint_root: str, blackboard: Blackboard) -> int`
  - `filter_tainted(messages: list[TaggedMessage]) -> list[TaggedMessage]`
- [ ] Integrate into `orchestrator.py::blast_round`: every blackboard write goes through `tag_message`; on agent error, descendants tainted before next round's context build.
- [ ] Pytest: 6+ cases including: single-error containment, multi-round cascade, taint persistence across rounds, overlap with judge ensemble.
- [ ] MAST regression: passing modes on v0.1.1 baseline + cascade_guard should rise from <8 to ≥ 12.

#### P1.5 — `judge_ensemble` (Hu 2510.12697 + Autorubric)
- [ ] `mechanisms/judge_ensemble.py` with:
  - `JudgeEnsemble` class: N=3 (configurable) judges with different prompt seeds, optional different model assignments per judge.
  - `BetaBinomialMixture` tracker for consensus stability.
  - `ks_test_stop(scores: list[float], window=2, threshold=0.05) -> bool`.
  - `Autorubric`-grounded scoring: each dim split into binary criteria summed to 0–10.
- [ ] Replace single-judge call in `orchestrator.py` with ensemble call when `blitz.toml`'s `judge_ensemble.n > 1`.
- [ ] Pytest: 8+ cases including: KS-stop firing, dissent preservation across judges, rubric aggregation, single-judge fallback.
- [ ] A/B: target ≥ 20% token reduction with no aggregate quality regression on bench slate.

#### P1.6 — `selector_synth` (Maryanskyy 2603.20324 + token-RR fallback)
- [ ] `mechanisms/selector_synth.py` with:
  - `SelectorSynthesizer` class: extracts spans from researcher outputs, judge selects best per coverage area, assembles final document.
  - Paragraph-level round-robin fallback when researchers split 2-vs-1 (per Liu 2604.17139 adaptation).
  - Backwards-compatible interface: existing `synthesizer` role can be replaced by setting `synth.mode = "selector"` in blitz.toml.
- [ ] Pytest: 8+ cases including: span extraction, judge selection, RR fallback trigger, output coherence smoke test.
- [ ] A/B: target Maryanskyy-style 0.81 win rate vs MoA baseline on bench slate.

#### P1.7 — `domain` preset registry
- [ ] `agents.py::get_domain_prompts(domain: str) -> dict[role, prompt]` reads `prompts/<domain>/`.
- [ ] `blitz.toml` adds `[swarm] domain = "general"` (default), valid: `general`, `crypto`, custom (any directory under `prompts/`).
- [ ] CLI flag `--domain` overrides config.
- [ ] Pytest: domain switching, fallback to general, error on bogus domain.

### Phase 1 success criteria
- [ ] All Phase 1 pytest cases green.
- [ ] cascade_guard MAST score ≥ 12/14.
- [ ] judge_ensemble: ≥ 20% token reduction with no quality regression.
- [ ] selector_synth: Cohen's d ≥ 0.3 vs v0.1.1 baseline on bench slate.
- [ ] Persona critics: +3 pts aggregate quality on multi-hop subset.
- [ ] Domain preset registry working; `--domain crypto` reproduces v0.1.1 prompts exactly.
- [ ] Tag `v0.2.0-alpha.2` on Phase 1 completion.

---

## Phase 2 — G-Memory Tier 2/3

**Estimated effort:** 1.5–2 weeks. **Depends on:** Phase 0 (bench), Phase 1 (judge ensemble for distillation cron).

**Goal:** Hierarchical memory — query graph with semantic neighbor expansion, insight graph with promotion-gated cross-task generalizations.

### Tasks

#### P2.1 — SQLite schema + LanceDB collection
- [ ] `gmemory/schema.sql` with tables: `query_node`, `query_edge`, `insight_node`, `insight_edge`, `support_link`. FTS5 virtual tables on query_node descriptions and insight texts. Indexes on embeddings_id and timestamps.
- [ ] LanceDB collection `query_embeddings` with 384-dim vectors (MiniLM compat).
- [ ] Migration script: `gmemory/migrate.py` runs schema + creates LanceDB collection idempotently.

#### P2.2 — Query graph (Tier 2)
- [ ] `gmemory/query_graph.py` with `add_task(query, embedding, swarm_output)`, `nearest_neighbors(query_emb, k=5, threshold=0.7)`, `one_hop_expand(node_ids)`.
- [ ] Pytest: 5+ cases — kNN correctness, threshold cutoff, 1-hop bounds, deduplication.

#### P2.3 — Insight graph (Tier 3)
- [ ] `gmemory/insight_graph.py` with `extract_insights_for_task(task_id)`, `find_overlapping_insights(query_ids)`, `traverse_upward(query_ids) -> insights`.
- [ ] Insight extraction shells out to `claude -p --model haiku` with a structured prompt; output validated via JSON schema.
- [ ] Pytest: 5+ cases — extraction format, traversal bounds, hyperedge correctness.

#### P2.4 — GAM-style promotion gate
- [ ] `gmemory/promotion.py::should_promote(insight_candidate, query_neighbors) -> bool`.
- [ ] Rule: promote if ≥ 3 query-graph neighbors share the candidate's semantic pattern (cosine ≥ 0.65 to insight embedding).
- [ ] Pytest: 4+ cases — N=2 rejected, N=3 promoted, contradicting evidence rejected, edge cases.

#### P2.5 — Retrieval pipeline
- [ ] `gmemory/retrieval.py::build_context(query, k=3) -> str` runs the 6-step pipeline: embed → kNN → 1-hop → upward traverse → LLM relevance score (Haiku) → LLM sparsify (Haiku) → format as `## Relevant prior findings`.
- [ ] Pytest: end-to-end on a fixture set of 50 prior tasks; assert top-3 relevance, sparse output ≤ 1500 tokens.

#### P2.6 — Distillation cron
- [ ] `scripts/distill_insights.py` runs nightly. Iterates over recent (last 24h) tasks, attempts insight extraction + promotion. Logs to `gmemory/distillation.log`.
- [ ] Add as a `cron` entry in `~/.claude/data/cron-jobs.json` per memory `reference_cron_jobs.md`.

#### P2.7 — Meta-tag schema for recursion loop
- [ ] Insights tagged `meta:` are queried by Phase 3 meta-loop.
- [ ] `gmemory/meta_query.py::query_meta_insights(predicate: str) -> list[Insight]` for the meta-loop interface.
- [ ] Pytest: tag filtering, predicate evaluation.

#### P2.8 — Orchestrator integration
- [ ] `orchestrator.py` calls `gmemory/retrieval.py::build_context(query)` before each round if `gmemory.tier ≥ 2`.
- [ ] Researchers see `## Relevant prior findings` section in their prompt.
- [ ] Pytest: integration test with a mocked retrieval returning canned content.

### Phase 2 success criteria
- [ ] All Phase 2 pytest cases green.
- [ ] Tier 2 vs Tier 1 quality lift ≥ 5% on cross-task slate (measured by re-running the slate with shared memory accumulated across runs).
- [ ] Tier 3 promotion gate prevents > 90% of low-quality candidate insights (sampled human review on first 50 promotions).
- [ ] Distillation cron runs successfully overnight; produces ≥ 5 promoted insights per 24h after 1 week of slate runs.
- [ ] Tag `v0.2.0-alpha.3` on Phase 2 completion.

---

## Phase 3 — Recursive self-improvement

**Estimated effort:** 1.5–2 weeks. **Depends on:** Phase 0, 1, 2.

**Goal:** The swarm proposes upgrades to itself; the bench validates; auto-merge guards prevent regression.

### Tasks

#### P3.1 — Cross-CLI heterogeneity
- [ ] `heterogeneity/cli_router.py::route_role(role, domain) -> tuple[cli, model]` with default routing table per Maryanskyy + Diversity for the Win.
- [ ] CLI executors: `claude`, `codex`, `gemini` — all subprocess invocations, all rule-#10 compliant.
- [ ] Graceful fallback to `claude` if a CLI is missing.
- [ ] Pytest: 5+ cases — routing correctness, fallback, env-var overrides.
- [ ] First measurement: bench with all-claude vs heterogeneous; report cohen's d in `bench/runs/<run_id>/heterogeneity.md`.

#### P3.2 — GEPA prompt evolution
- [ ] `scripts/optimize_prompts.py` wraps `gepa-ai/gepa` standalone library against the bench.
- [ ] Loads prompt file (e.g., `prompts/general/researcher.md`), evolves K iterations (default 50), outputs Pareto front to `evolve/prompt_candidates/<role>_<run_id>.jsonl`.
- [ ] Joona reviews + manually merges via `git checkout -b evolve/researcher-gen-N && cp candidate prompts/general/researcher.md`.
- [ ] Pytest: 4+ cases — fitness function correctness, Pareto frontier maintenance, candidate JSON schema.

#### P3.3 — Aflow architectural search
- [ ] `evolve/aflow_search.py` encodes swarm as graph (roles=nodes, data flow=edges).
- [ ] Mutations: add researcher, remove critic, add debate round, swap synthesizer, change judge_ensemble.n.
- [ ] MCTS explores mutations; each candidate evaluated against bench slate (with budget cap).
- [ ] Generation log: `evolve/aflow_runs/<run_id>/log.jsonl`.
- [ ] Pytest: 4+ cases — mutation legality, MCTS expansion bounds, budget enforcement.

#### P3.4 — Meta-loop
- [ ] `evolve/meta_loop.py::run_meta_generation()`:
  1. Query Tier 3 meta-insights (`gmemory/meta_query.py`).
  2. Propose specific config edits (e.g., "set judge_ensemble.n=5 for logical-reasoning topics").
  3. Run bench on candidate config.
  4. Apply auto-merge guards (Cohen's d ≥ 0.3, no dim regression > 5%, MAST ≥ 12/14, cost ≤ 1.5×, p < 0.05).
  5. If pass: write new `blitz.toml`, commit auto-generated change, tag `evolve-gen-N`.
  6. If fail: log to `evolve/candidates.jsonl` for human review.
- [ ] Pytest: 6+ cases — guard logic, regression detection, cost-cap honor.

#### P3.5 — Recursion bounds (safety)
- [ ] `evolve/recursion_guard.py::current_level() -> int` reads `blitz.toml`'s `evolve.level` (default 1; max 2 without `--allow-l3` flag).
- [ ] Unit test: attempting Level 3 without flag raises `RecursionLevelExceeded`.

#### P3.6 — Optional Managed Agents adapter
- [ ] `managed_agents/adapter.py::ManagedAgentsBackend` implements the same interface as `orchestrator.py`'s subprocess invoker.
- [ ] Uses Anthropic Managed Agents API with `managed-agents-2026-04-01` beta header.
- [ ] Documented as opt-in: `blitz.toml`'s `backend = "managed_agents"`.
- [ ] Pytest: mocked Managed Agents responses; 4+ cases.
- [ ] Documented cost trade-off in README v0.2.

#### P3.7 — Per-role model knobs
- [ ] `blitz.toml`'s `[models] researcher = "sonnet" critic = "sonnet" synthesizer = "opus-4-7" judge = "opus-4-7"` — each role's model configurable.
- [ ] `agents.py::plan_agents` reads role→model mapping.
- [ ] Pytest: per-role model assignment, fallback to default.

### Phase 3 success criteria
- [ ] All Phase 3 pytest cases green.
- [ ] GEPA produces a prompt variant for `prompts/general/researcher.md` that beats the original by Cohen's d ≥ 0.2 on bench slate.
- [ ] Aflow finds at least one architectural mutation that improves aggregate quality by Cohen's d ≥ 0.2.
- [ ] Meta-loop runs end-to-end on a fixture; auto-merge guards correctly accept good candidates and reject bad ones.
- [ ] Cross-CLI heterogeneity measured and reported.
- [ ] Managed Agents adapter passes integration smoke test.
- [ ] Tag `v0.2.0-alpha.4` on Phase 3 completion.

---

## Phase 4 — Methodology + docs

**Estimated effort:** 0.5–1 week. **Depends on:** Phases 0–3.

**Goal:** Replace the n=2 disclaimer with paper-grade results and a refreshed citation backbone.

### Tasks

#### P4.1 — n ≥ 20 results run
- [ ] Run full 30-prompt slate on final v0.2 config; aggregate stats to `bench/runs/v0.2.0_final/stats.md`.
- [ ] Generate all charts.
- [ ] If aggregate quality_judge score < v0.1.1 baseline + Cohen's d 0.5: investigate, fix, re-run. Don't ship a regression.

#### P4.2 — `docs/BIBLIOGRAPHY.md`
- [ ] All 13 new + 9 existing citations from `research.md` §6 with full arXiv links and one-line annotations.
- [ ] Cross-referenced from `BLITZ-SWARM.md` and `README.md`.

#### P4.3 — README v0.2 rewrite
- [ ] Replace "n=2 anecdote" disclaimer with n ≥ 20 results table.
- [ ] Add parallel-vs-sequential bench chart (PNG).
- [ ] Add MAST regression scoreboard.
- [ ] Update architecture diagram to reflect built (not proposed) Tier 2/3 + recursion loop.

#### P4.4 — `BLITZ-SWARM.md` refresh
- [ ] Update implementation-status table: Tier 2/3, mechanisms, evolve/, heterogeneity/, managed_agents/ all marked Implemented.
- [ ] Refresh citations to match BIBLIOGRAPHY.md.

#### P4.5 — `docs/ROADMAP.md`
- [ ] Mark Phase 0–3 deliverables as complete (or honestly "partially complete" where applicable).
- [ ] New roadmap entries for v0.3+: NEO Neural Theorizer feasibility study, full research-synthesis benchmark publication, Level-3 recursion safety study, multi-machine distributed swarm.

#### P4.6 — Optional short technical report
- [ ] If results warrant, generate `docs/TECHNICAL_REPORT_v0.2.pdf` via the `make-pdf` skill — 3–5 pages summarizing v0.2 changes, bench results, lessons learned. Archival quality.

#### P4.7 — Tag and ship
- [ ] Final pytest run: `pytest tests/ -v` all green.
- [ ] Final bench run on slate: produces `summary.json`, `stats.md`, charts.
- [ ] Tag `v0.2.0` on the final commit. Push.
- [ ] Update CHANGELOG.md with full v0.2.0 entry.

### Phase 4 success criteria
- [ ] README v0.2 rewritten and committed.
- [ ] BIBLIOGRAPHY.md committed.
- [ ] n ≥ 20 results published.
- [ ] Tag `v0.2.0` pushed.
- [ ] BUGS_AND_ITERATIONS.md updated with all session-level entries.

---

## Cross-cutting concerns

### Backwards compatibility

- v0.1.0 git tag preserved; v0.1.1 git tag preserved.
- v0.1.x configs run on v0.2.x because every new feature is gated by a flag with v0.1-equivalent default:
  - `domain = "general"` (default) — but `domain = "crypto"` gives v0.1.1 prompts verbatim.
  - `cascade_guard = true` (on by default — safe and unambiguous improvement).
  - `judge_ensemble.n = 1` preserves v0.1 single-judge behavior.
  - `gmemory.tier = 1` preserves v0.1 memory behavior.
  - `synth.mode = "blended"` preserves v0.1 synthesis (default `"selector"` opts into the new mechanism).
  - `evolve.enabled = false` (default) — recursion only runs when explicitly enabled.

### Dependency philosophy

- No new framework deps. GEPA standalone. Aflow ported. Vanilla asyncio + subprocess.
- New optional deps: `lancedb`, `scipy`, `matplotlib`, `jsonlines`, `pytest-asyncio`. All in `pyproject.toml [project.optional-dependencies]` so a minimal install (just `claude` CLI + `redis`) still runs the core swarm.

### BUGS_AND_ITERATIONS log discipline

Every bug discovered during implementation gets an entry. Format from CLAUDE.md rule #11:

```
## YYYY-MM-DD: BUG-NNN — Short Title

**Problem:** What hurt
**Root cause:** Why it happened
**Fix:** What changed
**Commit:** `<sha>` — message
```

### Verification before completion

Before marking any task done:
- Pytest cases green: `pytest tests/<module>` exit 0.
- Bench smoke run successful: `make bench-smoke` (5-prompt subset) completes.
- For Phase 4, full slate: `make bench` completes with stats.md generated.
- Never claim "passing" without seeing green output. Quote the actual test runner output in the commit message.

### Commit cadence checkpoints

- v0.2.0-alpha.0 — scaffold (this commit batch).
- v0.2.0-alpha.1 — Phase 0 complete.
- v0.2.0-alpha.2 — Phase 1 complete.
- v0.2.0-alpha.3 — Phase 2 complete.
- v0.2.0-alpha.4 — Phase 3 complete.
- v0.2.0 — Phase 4 complete (final ship).

---

## Open items requiring Joona's input

These are decisions where I'd benefit from a steer; *not* blockers — I'll make a best-judgment call autonomously per the Karpathy + autonomy memories unless flagged here:

1. **Bench slate publication** — should the 30-prompt slate be MIT-released as a separate `blitz-swarm-bench` repo for community contribution? (Default: yes, after v0.2.0 ships.)
2. **Optional technical-report PDF** — write one for archival? (Default: yes, short 3–5 page version after v0.2.0.)
3. **Level-3 recursion** — leave gated behind `--allow-l3`, ship empty/disabled? Or skip the L3 hooks entirely from v0.2 and revisit in v0.3? (Default: ship gated, empty.)
4. **Managed Agents adapter** — opt-in toggle in v0.2 (current plan), or punt to v0.3 entirely? (Default: opt-in v0.2.)

---

## Final note

This plan is a contract. Deviations from it land as BUGS_AND_ITERATIONS entries with a `**Plan deviation:**` line.

The recursion loop's whole point is that this plan should evolve too — once Phase 4 ships, the meta-loop's first job is to read this plan, the v0.2 bench results, and propose `plan.md` edits for v0.3. The future is plans that update themselves.
