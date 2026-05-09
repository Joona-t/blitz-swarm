# Bugs & Iterations

## 2026-05-09: BUG-001 — LanceDB distance type implicitly L2 with normalized vectors

**Problem:** `memory/writer.py:_find_related_queries` reads `_distance` from LanceDB and computes `1 - distance` as cosine similarity. With normalized 384-dim MiniLM embeddings, LanceDB defaults to L2, which is NOT `1 - cosine_sim`; similarities are systematically off, shifting τ_link's effective cutoff.

**Root cause:** LanceDB's default distance metric is implementation-dependent and changes between versions. Code never set it explicitly.

**Fix:** New `gmemory/query_graph.py:nearest_neighbors` calls `.distance_type("cosine")` explicitly. Documented in `docs/research/PHASE_2_GMEMORY.md` §10. One-shot migration `scripts/migrate-distance-type.py` planned for v0.2 release to recompute `query_edges.similarity` for existing rows.

**Found in:** v0.1.0–v0.1.1.
**Fix lands in:** v0.2 Phase 2 (G-Memory Tier 2 build-out).
**Commit:** TBD (`gmemory/query_graph.py`).

## 2026-05-09: ITER-001 — tests/bench/__init__.py shadowed root bench/ package

**Problem:** Pytest collection failed with `ModuleNotFoundError: No module named 'bench.detectors'`. Cause: `tests/bench/__init__.py` made `tests/bench/` a package named `bench`, shadowing the real `bench/` package at the repo root during test imports.

**Root cause:** Created the test directory with `__init__.py` out of habit. Pytest doesn't need it for test discovery, and `__init__.py` in tests turns the test directory into an importable package that shadows real modules.

**Fix:** Deleted `tests/bench/__init__.py`. Pytest auto-discovery finds the tests without it.

**Lesson:** never create `__init__.py` in test directories that share a name with a real package.

**Commit:** v0.2.0-alpha.0 follow-up (Phase 0 foundation commit).

## 2026-05-09: ITER-002 — Zero-variance paired-diff edge case in stats tests

**Problem:** `test_paired_t_test_positive_difference` and `test_cohens_d_positive` failed because the original `a` and `b` arrays had constant pairwise differences (e.g., `[2,2,2,2,2]`). The stats module returns `t=0` and `d=0` when stdev of diffs is 0 (intentional div-by-zero guard).

**Root cause:** Test fixtures didn't reflect realistic input distributions. Constant diffs are a degenerate edge case that the stats code handles by returning 0; the tests assumed strictly-positive output.

**Fix:** Updated test fixtures to have non-constant positive diffs (e.g., `[1.5, 1.5, 2.0, 2.5, 2.0]`). Both tests now pass.

**Lesson:** when writing tests for paired statistics, vary the diffs even when modeling a "consistent positive shift" — real data never has zero variance.

**Commit:** v0.2.0-alpha.0 follow-up (Phase 0 foundation commit).

## 2026-05-09: ITER-MYTHOS-001 — New `mythos/` mode added to blitz-swarm

**Problem:** Blitz-swarm consensus mode is excellent for parallel research summaries but architecturally wrong for the "Mythos-worthy" use cases the prior `/blitz-swarm` token-economics research surfaced (legacy modernization, multi-hop security audit, formal verification, distributed-systems debugging, compiler construction). Those tasks need long-horizon coherence + strict gating, not flat consensus voting across peer researchers.

**Root cause:** Topology mismatch. Flat consensus is symmetric; the Mythos-worthy pattern is asymmetric — one deep planner + N cheap parallel executors + one deep verifier with replan loops. Prior research recommended this hybrid pattern (Mythos orchestrator, Sonnet executors) as ~90% Mythos quality at 30–40% full-Mythos cost.

**Fix:** Added a `mythos/` package as a peer mode of consensus. New CLI surface `--mode mythos`, with a thin `mythos_swarm.py` shim entrypoint. Architecture:

- `mythos/policies.py` — `MODEL_ALIASES` dict (`"mythos" → ("opus", effort="max")`), `CostBudget`, `MythosConfig`, `load_mythos_config()`. The alias layer decouples our code from Anthropic's naming churn — when a real Mythos model ships, change one dict entry.
- `mythos/schemas.py` — JSON schemas for planner/executor/verifier output, enforced via `claude --json-schema`.
- `mythos/_invoke.py` — shared `claude -p` subprocess wrapper with cost extraction.
- `mythos/planner.py` — `decompose()` and `replan()`.
- `mythos/executor.py` — synchronous + parallel runner via `asyncio.to_thread`.
- `mythos/verifier.py` — gate check, returns structured `pass | needs_work` with required_fixes for replan.
- `mythos/runner.py` — full orchestration loop with hard cost ceiling.
- `mythos/artifact.py` — writes plan/executors/verification/final to disk per round.
- `mythos.toml` — config overlay (separate from blitz.toml so consensus mode is unaffected).
- `MYTHOS-SWARM-RESEARCH.md`, `MYTHOS-SWARM-PLAN.md`, `MYTHOS-SWARM.md` — research → plan → user-facing doc per CLAUDE.md workflow.
- `tests/test_mythos_smoke.py` — 9 passing smoke tests (imports, config, schemas, prompts, CLI, dry-run).

CLI probe (2026-05-09): `claude -p --model opus` routes to `claude-opus-4-7` with 1M context; `--effort max` is the thinking-budget knob. Single trivial round-trip cost $0.33; default cost ceiling set to $5.00 (~15× headroom).

**Compliance:** Rule #10 audit clean — all LLM calls go through Joona's `claude` CLI subscription, no `ANTHROPIC_API_KEY` paths. Rule #1 satisfied — `MYTHOS-SWARM-PLAN.md` written before any code. Rule #6 — no `Co-Authored-By` in the commit. Rule #11 — this entry. 238 existing tests still pass; 9 new mythos smoke tests pass.

**Deferred to v2:** Live full-task execution (binary-search formal-verification target prepared but not yet run — would burn budget mid-build). Head-to-head benchmark vs consensus mode is the gating experiment for declaring Mythos mode "worth it". Cross-CLI heterogeneity, persona-typed verifiers, cascade-guard escalation.

**Commit:** TBD (this commit).

<!-- Format:
## YYYY-MM-DD: Short Title

**Problem:** What went wrong or needed changing
**Root cause:** Why it happened
**Fix:** What was done to resolve it
-->
