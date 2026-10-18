# Bugs & Iterations

## 2026-06-10: ITER-AUDIT-1 — Fable audit wave on the upgrade harness (10 findings fixed)

**What hurt:** a model-upgrade audit of the Opus-built 10M-token upgrade harness
found the prior adversarial passes had verified each phase in isolation (dry-run)
while the seams between phases were hollow: research output was never parsed
(mythos got the literal string `'technique_01'`), ablation arms ran byte-identical
configs (the `arm` label toggled nothing), the default parameterization projected
~25–40M tokens against a 10M ceiling, errored cells were marked done forever and
poisoned samples with 0.0 (a codex smoke wrote floor=0.0, which everything
"clears"), and the gate compared verify's 0–1 composite against a 0–10 floor —
vacuously false for any technique.

**Root cause (meta):** unit tests + per-phase red-teams can't see cross-phase
data-flow breaks, live cost arithmetic, or metric incentive inversions. Those
require tracing data across boundaries and doing arithmetic against measured cost
(31,150 tokens / ~9.5 min per 1-round swarm run).

**Fix (commit `cdc89e0`):** candidates.json contract research→implement→verify
(new `jobs/candidates.py`); real arm toggling via `BLITZ_FEATURE_OVERRIDES` →
`orchestrator._feature_enabled` (live-proven flip); pre-flight cost calibrator
that prints the projection math and refuses infeasible jobs (rc=2); retry→
exclude failure semantics + usage-limit pause/resume-same-cell; timeouts fit to
measured reality (AgentCall 600s, bench 3600s, timeouts = missing not 0.0); true
K-single-agent floor on the unified 0–1 composite scale (`--floor-k`); per-
(seed,prompt) significance pairing (n=8 vs 3) with keep-ALL fallback removed +
evidence grading; composite no longer rewards citing nothing; normalized metrics
matching. Defaults re-parameterized feasible (rounds 2, techniques 3, seeds 2,
4-prompt verify mini-slate). Full suite 444 passed (+86 new tests).

**Prevention rule:** every multi-phase harness gets (1) a cross-phase contract
test (does phase N's consumer actually read phase N-1's producer?), (2) a cost
calibration gate before any live run, and (3) scale assertions wherever two
scores are compared.

## 2026-05-09: BUG-001 — LanceDB distance type implicitly L2 with normalized vectors

**Problem:** `memory/writer.py:_find_related_queries` reads `_distance` from LanceDB and computes `1 - distance` as cosine similarity. With normalized 384-dim MiniLM embeddings, LanceDB defaults to L2, which is NOT `1 - cosine_sim`; similarities are systematically off, shifting τ_link's effective cutoff.

**Root cause:** LanceDB's default distance metric is implementation-dependent and changes between versions. Code never set it explicitly.

**Fix:** New `gmemory/query_graph.py:nearest_neighbors` calls `.distance_type("cosine")` explicitly. Documented in `docs/research/PHASE_2_GMEMORY.md` §10. One-shot migration `scripts/migrate-distance-type.py` planned for v0.2 release to recompute `query_edges.similarity` for existing rows `[never created — LanceDB path removed in v0.3.0; see scripts/migrate_memory_db.py]`.

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

**Commit:** 54631d7.

## 2026-05-09: BUG-MYTHOS-001 — claude --json-schema returns parsed object in `structured_output`, not `result`

**Problem:** First live Mythos run (`mythos_swarm.py "Implement and verify binary_search ..."`) reported `planner_failed` in 77s with $0 cost. Run dir contained an empty plan with `error: "planner exit 1: "` and no parsed sub-specs. Same test in dry-run had passed; smoke tests passed; the failure only surfaced when actually invoking `claude -p`.

**Root cause:** Two distinct issues in `mythos/_invoke.py`:
1. **Parsing:** When `claude -p --output-format json --json-schema <s>` is used, the model emits the structured object via a tool call. The CLI surfaces it in envelope key `structured_output`, leaving `result` as an empty string. My `_parse_inner_result` only inspected `result`, so it always saw nothing and returned `parsed=None`.
2. **Turn budget:** `--max-turns 1` is insufficient with `--json-schema` because the structured-output tool call itself consumes a turn — the model needs at least 2 turns (call + completion). The default of 1 caused the CLI to terminate with `error_max_turns` (and in some flag combos, exit code 1 with empty stderr).

**Fix:** In `mythos/_invoke.py`:
- `_parse_inner_result` now checks `envelope["structured_output"]` first, then falls back to `result` (dict or string with embedded JSON).
- `invoke()` default `max_turns` bumped from 1 to 4 — gives headroom for the tool-call round-trip plus any internal retries.
- Added `is_error` envelope check so envelope-level errors that come back with shell exit 0 (e.g. `error_max_turns`, `error_during_execution`) are surfaced in `InvokeResult.error` rather than silently parsed as success.

**Validation:** Re-ran the same task. Result: status `passed`, 1 round, $1.07 / $5.00 budget, 203s wall. Extracted both deliverables to `/tmp/mythos_smoke_test/` and ran `pytest -q test_binary_search.py`: **9/9 passed in 0.10s**. Verifier verdict matched reality — not just hand-waving.

**Lesson:** when integrating with a CLI surface, the `--help` text rarely tells you which envelope field carries the parsed output. Probe with a real schema call before assuming, and always check `is_error` even when `returncode == 0`.

**Commit:** b995365.

## 2026-05-10: ITER-MYTHOS-002 — Head-to-head: Mythos beats consensus on artifact-producing tasks

**Observation:** Ran three live tasks to compare swarm topologies:
- (A) Consensus mode (`--max-rounds 2`) on binary search w/ tests: **$1.43, 8:01 wall, 11 invocations, no consensus reached, judge avg 6.2/10. Critic flagged twice that researchers truncated code blocks; synthesizer backfilled the deliverable. Pytest after extraction: 31/31 pass.**
- (Mythos orig) on the same binary-search task: **$1.07, 3:23 wall, 4 invocations, 1 round, verifier pass 0.95. Pytest: 9/9 pass.**
- (B) Mythos on a harder task — stable merge sort with Pre/Post conditions, termination by strong induction, stability proof via `<=` tie-breaking, 7 property-based tests: **$1.56, 4:54 wall, 4 invocations, 1 round, verifier pass 0.95. Pytest: 7/7 pass. `grep -E '(Termination|Stability|Pre/Post)' merge_sort.py` confirms the labeled proof blocks are present and substantive.**

**Verdict:** On the same binary-search task, Mythos was 25% cheaper, 58% faster, used 64% fewer agent calls, and emitted an artifact-shaped output (plan / executor outputs / verification trace / final_artifact.md) rather than a research-summary-shaped one. On the harder merge-sort task it still one-shot the work at similar cost.

**Structural reason:** consensus mode's researchers default to summarizing; the topology can't force them to emit complete artifacts (only the synthesizer's final pass backfilled the code). Mythos's planner-executor-verifier hierarchy is the right shape when the *deliverable IS the work*. Confirms the prior `/blitz-swarm` research thesis empirically — at least for this task class.

**Caveats not to forget:**
1. Consensus mode wasn't designed for code production — this isn't a fair benchmark *of consensus mode*; it's a sanity check that Mythos is the right tool for code-as-deliverable tasks.
2. n=3 runs. Strong directional signal, not a benchmark.
3. Verifier-blesses-broken-code is the failure mode to watch. We got pass-and-actually-correct twice; need to find a case where the verifier is overconfident before trusting it on higher-stakes work.

**Operational note:** First attempt at running consensus mode in the harness's background-task system died with exit 144 (signal 16 / SIGURG, empty stdout). Retried in foreground — ran cleanly. There's a sandbox interaction worth investigating before relying on background swarm runs.

**Run dirs:**
- `output/implement_and_verify_a_python_binary_search_arr_ta_20260510_081107.md` (A)
- `output/mythos/implement_and_verify_a_python_binary_search_arr_ta_20260509_185053/` (Mythos orig)
- `output/mythos/implement_and_verify_a_stable_merge_sort_in_python_20260510_075734/` (B)

**Next:** run on a task where one mode actually has a structural disadvantage we can prove (e.g., a debug task with a non-obvious root cause to test verifier trust; or a research-style question to confirm consensus mode wins on its home turf). Until then, default code-production tasks to Mythos.

**Commit:** 44f2961.

## 2026-05-10: ITER-MYTHOS-003 — mythos-bench harness + first verifier-trust probe

**What landed:** A real bench harness at `mythos-bench/`. Task specs are TOML; the runner shells out to `orchestrator.py`, extracts deliverables (mythos: per-executor `.md`; consensus: synthesized `.md`), runs pytest, optionally runs hidden adversarial tests, and writes a structured row to `results/runs.jsonl`. `bench.py compare` emits a markdown comparison table.

**Why now:** doing head-to-head runs by hand was already painful at n=3. The Karpathy move is to build the instrument, not run more comparisons by eye.

**First verifier-trust probe — `monetary_decimal`:** task asks for a Money class using `decimal.Decimal` with banker's rounding, no-float enforcement, currency-mismatch checks. The visible spec doesn't enumerate every adversarial edge case. Hidden adversarial tests at `mythos-bench/adversarial/monetary_decimal__adversarial.py` check 6 probe categories: float forbidden on construction, float forbidden on multiplication, currency-mismatch raises ValueError, banker's-rounding edge cases (0.005 → 0.00, 0.025 → 0.02, NOT half-up), large-number precision (Decimal must not route through float anywhere), and repr format.

**Result:** Mythos passed all 12 swarm-written tests AND all 17 hidden adversarial tests. Verifier verdict was `pass`; adversarial tests confirmed it. **No verifier-trust failure caught.** Cost: $1.50, 4:16 wall, 3 invocations, 1 round.

**What this tells us — and what it doesn't:**
- Mythos planner produced an aggressive spec that pre-empted most pitfalls (banker's rounding was already mandated in the planner output, so executors implemented it).
- The verifier was well-calibrated — no false-pass on this probe.
- This is one probe. We have not refuted the verifier-trust failure mode; we just didn't trigger it. Designing harder probes is the next priority — candidate: tasks where the natural Sonnet implementation has an emergent stress-test bug (e.g. concurrent state, hash collisions, IEEE-754 boundary behavior).

**Bench results so far** (4 runs):

| Task | Mode | Verdict | Cost | Wall | Inv | Primary | Adversarial |
|---|---|---|---|---|---|---|---|
| binary_search | mythos | passed | $1.07 | 203s | 4 | 9p/0f | — |
| binary_search | consensus | passed | $1.43 | 481s | 11 | 31p/0f | — |
| merge_sort | mythos | passed | $1.56 | 294s | 4 | 7p/0f | — |
| monetary_decimal | mythos | passed | $1.50 | 256s | 3 | 12p/0f | **17p/0f** |

**Files:**
- `mythos-bench/bench.py` — CLI
- `mythos-bench/lib.py` — task spec, runner, extractor, pytest runner, result io
- `mythos-bench/BENCH.md` — user doc
- `mythos-bench/tasks/{binary_search,merge_sort,monetary_decimal}.toml`
- `mythos-bench/adversarial/monetary_decimal__adversarial.py`
- `mythos-bench/results/{runs.jsonl,comparison.md}`

**Operational notes:**
- The backfill command's consensus-mode `.md` detection picks the first match in the run dir — if the dir contains many old outputs, it picks wrong. Workaround: copy the specific output file to an isolated dir and backfill from there. Fix to land in v2 — accept explicit `--output-path`.
- The `[verification.adversarial]` table is honor-system hidden — the swarm doesn't see the file. If Mythos ever gains shell/file access, this contract needs hardening.

**Next probes to design:**
1. A task with a non-local invariant (e.g. a small state machine where wrong handling of one transition breaks an invariant in a different transition).
2. A task where Sonnet's natural implementation has an IEEE-754 stress bug only visible at boundaries.
3. A debug-style task (give it broken code + a failure trace, ask to fix without breaking other invariants).

**Commit:** TBD.

## 2026-10-07: BUG-002 — Codex backend rejected every structured call (non-strict --output-schema)

**Problem:** With the default `[backend] default = "codex"`, every Blitz agent call, judge vote and pairwise verdict came back errored (`codex exit 1`). The API response was `Invalid schema for response_format 'codex_output_schema': ... 'additionalProperties' is required to be supplied and to be false`.

**Root cause:** `codex exec --output-schema` sends the schema as an OpenAI **strict** json_schema. Strict mode requires `additionalProperties: false` on every object and every property listed in `required`. It also can't express free-form objects. None of our schemas (AGENT_OUTPUT_SCHEMA, JUDGE_VOTE_SCHEMA with `rubric_scores: {type: object}`, PAIRWISE_VERDICT_SCHEMA, mythos/memory schemas) met that. The unit tests mocked `subprocess.run` and only asserted that `--output-schema` was passed, so the contract was never checked against the real CLI.

**Fix:** `backends.core.to_strict_output_schema()` normalises any schema before `CodexLocalBackend` (`backends/adapters/cli.py`) writes `schema.json`. It adds `additionalProperties: false`, lists every key in `required`, makes originally-optional keys nullable, and turns free-form objects into empty strict objects. `backends.core.strip_strict_nulls()` drops those nulls from the parsed result before `validate_json_schema`, so callers still see the loose contract. The judge call uses `orchestrator._judge_vote_schema(rubric_dims)`, which names each rubric dimension so the per-dimension scores survive strict mode. The `openai_http` adapter reuses the same conversion for `response_format` json_schema. Source schemas are unchanged; the Claude/Gemini backends are untouched.

**Regression test:** `tests/test_codex_strict_schema.py` runs a strict-mode checker over every real schema after conversion, checks the null round-trip, and checks that the backend writes a strict-valid `schema.json`. Live check (needs codex CLI, uses your own subscription): `python -c "import agents; from pathlib import Path; from backends import *; print(CodexLocalBackend(model='<your codex model>', reasoning_effort='low').call(AgentCall(role='t', prompt='Is 2+2=4? Fill the schema minimally.', schema=agents.AGENT_OUTPUT_SCHEMA, cwd=Path('.'))).errored)"` should print `False`. Verified 2026-10-07 with codex-cli 0.160.0 for the AGENT, JUDGE and PAIRWISE schemas.

**Found in:** PR #1 (`codex/codex-first-backend`), found by skeptic review. Originally landed as a4fc972 on the single-file `backends.py`; **ported** to the `backends/` package on `fable/audit-remediation` (v0.3.0 WP1) because the branch was not cherry-pickable.
**Commit:** a4fc972 (original), v0.3.0 WP1 (port).

## 2026-10-11: BUG-005 — Mythos ran every role on the provider-wide model; 36 hard-coded model literals swept (v0.3.0 WP1)

**Problem:** `mythos/_invoke.py` built the backend with `backend_model = provider.model or model`, so a provider-wide `model` key silently overrode the planner/executor/verifier alias — every Mythos role ran on the same model, and `mythos/policies.MODEL_ALIASES` passed the alias name ("deep"/"fast" after the config rewrite) straight through as the model string (live mode broken). Outside Mythos, 36 literal model names / tiers lived in 12 `.py` files (`agents.py` per-agent `model="sonnet"`, `mechanisms/*` judge/guard defaults, `metrics.py` cost table, `jobs/*`, `gmemory/retrieval.py`, `managed_agents/adapter.py`, `evolve/aflow_search.py`), and the judge/selector call sites in `orchestrator.py` still read the now-empty `cfg.swarm.default_model`, so with the v0.3.0 config they would have sent an empty model (NO_MODEL_ERROR) live.

**Root cause:** Model selection was scattered across call sites instead of going through one resolver; the alias table and the provider default both competed with the role tier, and nothing enforced "no literal model outside config".

**Fix:** Every call site resolves through `backends.models.resolve_model(role, backend_id, cfg)` and builds backends with `backends.make_provider_backend` (fallback chain included): `orchestrator.py` (invoke_agent forwards the tier's `effort`; `judge` tier for judge-ensemble + selector pairwise judges; `guard` tier injected into `CascadeGuardConfig`; banner + dry-run print the resolved model/tier/effort per agent), `agents.py` (`planner` tier; `BlitzAgent.model` is a blank hint, `ROLE_MODEL_OVERRIDES` deleted), `memory/{reader,writer}.py` (`memory_ops`), `mythos/_invoke.py` (`resolve_role_model`: `--model`/`BLITZ_MODEL` > `BLITZ_MODEL_<ROLE>` > mythos.toml tier > blitz.toml `[models]`; provider `model` never shadows the role tier; dead envelope helpers deleted), `mythos/policies.py` (`MODEL_ALIASES`/`resolve_model` deleted; `token_ceiling` read; `CostBudget` charges tokens and ignores unknown cost), `mythos/runner.py` (token ceiling enforced, `describe_models` for prints), `mechanisms/*` (empty defaults, model injected by caller), `metrics.py` (cost from `[backend.providers.<id>.pricing]` via `pricing_for`; unknown cost stays `None` and prints `n/a`; `AgentResult.cost_usd` is `float | None`), `jobs/candidates.py` (`candidate_extractor` role), `jobs/blitz_upgrade.py` (`researcher` tier on the job's `--backend`), `managed_agents/adapter.py`, `evolve/aflow_search.py` + `heterogeneity/cli_router.py` (provider ids only; router accepts `cfg` to restrict to `[backend.providers]`). `SwarmConfig.default_model` deleted. `tests/test_no_hardcoded_models.py` guard now runs for real (xfail removed, 0 hits).

**Regression test:** `tests/test_no_hardcoded_models.py::test_no_hardcoded_model_literals_outside_config` (full-repo grep guard), `tests/test_mythos_smoke.py::test_role_tier_resolution_order` (provider `model` cannot shadow the role tier; env overrides beat the mythos.toml tier), `::test_cost_budget_token_ceiling`. Verified: `orchestrator.py --list-models --backend claude`, `--dry-run --no-llm-plan --backend claude`, `--mode mythos --dry-run` all print resolved role -> tier -> model (planner/verifier = claude deep tier, effort max) without a model call. pytest: 523 passed / 14 skipped / 1 failed (pre-existing G5 p-value test, WP4).
**Commit:** v0.3.0 WP1 sweep (orchestrating session commits).

## 2026-10-11: BUG-003 — Upgrade job coerced p = 0.0 to 1.0 and dropped significant techniques (v0.3.0 WP4)

**Problem:** `jobs/blitz_upgrade.py` gate read the paired-test p-value as `float(rec.get("p_value", 1) or 1)`. A genuine p = 0.0 (the strongest possible evidence, which the scipy-free `bench/stats.py` path can return for large effects) is falsy, so `or 1` turned it into 1.0 and the KEEP decision flipped to DROP. One test failed on the branch (445 pass / 1 fail). Audit finding process-02.

**Root cause:** truthiness used as a null check on a numeric field whose legitimate range includes 0.

**Fix:** `jobs/blitz_upgrade.py:1886-1891` — `p_raw = rec.get("p_value"); p = 1.0 if p_raw is None else float(p_raw)`. Only a missing value means "no p-value". The pre-existing failing test passes.

**Regression test:** `tests/test_wp4_bench_evolve_guard_verifier.py::test_decision_rule_treats_p_zero_as_significant_and_none_as_not`.

**Found in:** `feat/blitz-upgrade-20260609` (2026-06). **Fix lands in:** v0.3.0 WP4 (commit 36767cd).

## 2026-10-11: BUG-004 — gmemory had zero production callers; "compounding intelligence" never ran (v0.3.0 WP2)

**Problem:** The orchestrator imported only the legacy `memory/` reader/writer. Tier 2 edges, Tier 3 promotion, hybrid BM25/RRF retrieval and the README's "## Relevant prior findings" injection never executed; the legacy retrieval path returned nothing without LanceDB; the live `memory.db` had 0 query edges, 0 insights and `access_count = 0` on every row after three runs of the same topic. Mythos `use_memory = true` was read by nothing. Audit findings memory-01..08, critic gap B.

**Root cause:** Phase 2 shipped `gmemory/` as a tested library and never swapped the orchestrator's imports; two incompatible `schema.sql` files coexisted; the legacy writer rolled back the whole task write when the embedder was missing.

**Fix:** `gmemory/store.py` `MemoryStore` is the one memory surface (open → migrate with a `*.pre-gmemory.bak` backup → FTS5 retrieval on stdlib `sqlite3` → `persist_task` for all three tiers → N=3 promotion gate → eviction every `cycle_frequency` tasks). Wired at `orchestrator.py:749-790` and `:1268-1279` and in `mythos/runner.py:83-150`. `memory/` is a 32-line re-export shim. FTS5 MATCH input is escaped. `scripts/migrate_memory_db.py` migrates old files idempotently.

**Regression tests:** `tests/test_memory_store.py` (31: round-trip on a temp SQLite with no optional deps, promotion gate, migration, backup) and `tests/test_gmemory.py` (35).
**Commit:** 36767cd (v0.3.0 WP2).

## 2026-10-11: BUG-006 — Mythos verifier executed model-written tests on the host by default (v0.3.0, security review)

**What hurt:** WP4 added the mechanical check (`mythos/verifier.py:run_artifact_tests`): when a deliverable ships a test file, the verifier runs it with the host Python. It landed with `verifier_run_tests = true`. A background security review of the pushed commit flagged it: the subprocess has a temp cwd, scrubbed env and timeout but no sandbox, so model-written code runs with the user's full filesystem and network access — the same class of risk the 2026-10-11 audit rated P1 in research-swarm (`--dangerously-skip-permissions`).

**Root cause:** the mechanical-check design (backends-04) was written for correctness, not for the threat model; "run the artifact's tests" is "run the artifact".

**Fix:** opt-in. `MythosConfig.verifier_run_tests` defaults to `False` (`mythos/policies.py`), `mythos.toml` ships `verifier_run_tests = false` with the risk stated in the comment, `mythos/verifier.py` reads the flag with a `False` default, and the tests that exercise the subprocess path opt in explicitly. README / LIMITATIONS / CLAIMS / CHANGELOG describe it as opt-in. The confidence gate stays on.

**Why this fix:** the check is valuable for artifacts you would run by hand anyway, and worthless as a default that silently executes untrusted code. A real sandbox (container, `--sandbox` of the backend CLI) is the v_next; until then the flag is the gate.

---

## 2026-10-11: ITER-AUDIT-2 — Fable remediation (v0.3.0): 47 audit findings → fixes

**What hurt:** the 2026-10-11 adversarial audit (`~/swarm-audit-2026-10-11.md`, Part 1) found the repo's documentation asserting things the code did not do: two of three memory tiers never ran (BUG-004), Mythos never ran its deep model (BUG-005), the README stopping-rule diagram was false, model names were hard-coded in 12 files, the bench runner could never raise a MAST flag, and seven `docs/*.md` were frozen at 2026-03-14. 47 findings, 0 debunked on re-verification.

**Root cause (meta):** each phase shipped as a tested library plus a doc, without a gate that checked the doc against the wiring. Unit tests passed while the orchestrator imported none of it.

**Fix:** six work packages on branch `fable/audit-remediation`, plan of record `docs/plans/2026-10-11-fable-remediation.md`. Gate at 36767cd (WP1–4): 667 passed / 14 skipped / 0 failed; with WP5's tests in the tree: 745 passed / 14 skipped / 0 failed (759 collected), only `pytest` + `pytest-asyncio`.

| Finding | Severity | Fix (file:line at 36767cd) | WP |
|---|---|---|---|
| memory-01 gmemory zero callers | P1 | `orchestrator.py:749-790`, `:1268-1279`; `gmemory/store.py` | WP2 / BUG-004 |
| memory-02 no retrieval without LanceDB | P1 | `gmemory/store.py:1-14` (FTS5 path, stdlib only); `gmemory/hybrid.py` | WP2 |
| backends-01 Mythos never on deep model | P1 | `mythos/_invoke.py:55-70` (`resolve_role_model`), `mythos.toml [models]`; `backends/models.py` | WP1 / BUG-005 |
| process-02 p-value `or 1` | P1 | `jobs/blitz_upgrade.py:1886-1891` | WP4 / BUG-003 |
| core-loop-01 serial judge/selector loops | P2 | `orchestrator.py:900-913`, `:1039-1052` (`to_thread` + `gather`); staging banner `:82-85` | WP3 |
| core-loop-02 blackboard write-only | P2 | `orchestrator.py:576-582`, `:1327`, `:1390`, `:1403` (`read_round_outputs`) | WP3 |
| core-loop-03 KS-stop not a halt | P2 | `orchestrator.py:731-745` `_judge_ks_halt`; `blitz.toml [judge_ensemble] halt` | WP3 |
| core-loop-04 tools/max-turns/cwd | P3 | `orchestrator.py:80` `CONSENSUS_MAX_TURNS = 1`, `:88-91` `scratch_cwd`; `AgentCall.tools` | WP3 |
| core-loop-05 unfenced injection | P3 | `context.py:24-32` fences; `orchestrator.build_context` | WP3 |
| core-loop-06 dead heterogeneity | P3 | `backends/models.py` tiers per role; dry-run prints model per agent (`orchestrator.py:1590-1648`) | WP1 |
| core-loop-07 dead config keys, `max_agents` | P2 | `config.py:214-260` strict loader; `agents.py:303-320` | WP3 |
| core-loop-08 metrics.jsonl / dissent / plan fallback | P2 | `metrics.py:25` (`output/metrics.jsonl`); `consensus.final_votes`; `docs/data/` archive | WP3 |
| memory-03 two schemas | P2 | `gmemory/schema.sql` canonical; `memory/__init__.py` shim; `gmemory/migrate.py` | WP2 |
| memory-04 rollback without embedder | P2 | `gmemory/store.py` (vector side optional, never aborts) | WP2 |
| memory-05 promotion gate unenforced | P2 | `gmemory/store.py:211-270` `persist_task` → `evaluate_candidates` | WP2 |
| memory-06 eviction unwired | P2 | `gmemory/store.py:277-322` `run_eviction_cycle` / `maybe_run_eviction` | WP2 |
| memory-07 vestigial Redis writer daemon | P2 | removed with `memory/writer.py` | WP2 |
| memory-08 FTS5 MATCH injection | P2 | `gmemory/retrieval.py` escaping | WP2 |
| research-docs-08 LanceDB L2 read on live path | P2 | path deleted (legacy writer gone) | WP2 |
| mechanisms-01 evolve/heterogeneity unwired, no GEPA script | P2 | `evolve/__init__.py:1-47` gate; `heterogeneity/cli_router.py:174-176`; `scripts/optimize_prompts.py` | WP4 |
| mechanisms-02 judge = N identical, guard heuristic | P3 | `blitz.toml [guard] mode = "max"` LLM hooks on `guard` tier; `mechanisms/cascade_guard.py:13-23`, `:333` labels | WP4 |
| bench-01 `flags = []` never `detect_all` | P2 | `bench/runner.py:424-432`, `:668` | WP4 |
| bench-03 normal-CDF at small n | P3 | `bench/stats.py:144-162` exact t for df < 30 | WP4 |
| bench-04 stale version literal | P3 | `bench/runner.py:162-178` reads `pyproject.toml` | WP4 |
| evolve-01 allow-list names dead keys | P3 | `evolve/meta_loop.py` `ALLOWED_KEY_PATHS` | WP4 |
| backends-02 managed_agents paid path with fake completion | P2 | `contrib/managed_agents/adapter.py` + README (opt-in, `NotImplementedError`) | WP4 |
| backends-04 verifier verdict unverified | P3 | `mythos/verifier.py:109-195` subprocess tests + confidence gate | WP4 |
| backends-05 dead fallback/registry surface | P3 | `backends/registry.py:128-180` `FallbackBackend` | WP1 |
| backends-06 upgrade job stalled | P2 | harness fixes merged (1a4876c); run **not** completed — documented in README/LIMITATIONS | WP6 (doc) |
| backends-07 effort dropped, argv prompt limit | P3 | `backends/adapters/cli.py` (`--effort`, stdin > 100 KB) | WP1 |
| process-01 / claims-01 / research-docs-01 docs frozen at 2026-03-14 | P2 | README, `docs/CLAIMS_AND_EVIDENCE.md`, `IMPLEMENTATION_STATUS.md`, `ROADMAP.md`, `LIMITATIONS.md` rewritten; the rest bannered | WP6 |
| process-03 packaging | P2 | `pyproject.toml` version 0.3.0 + extras | WP5 |
| process-04 six untested modules | P2 | `tests/test_consensus.py` (9), `test_blackboard.py` (14), `test_config.py` (26), `test_metrics.py` (28), `test_embedder.py` (12), `test_agents_planner.py` (19) | WP3/WP5 |
| process-05 tracked metrics.jsonl | P2 | archived to `docs/data/metrics-2026-03_06.jsonl` | WP3 |
| process-08 gitignore | P3 | `.gitignore` (WP5) | WP5 |
| research-docs-02 fictional scripts | P2 | `scripts/{migrate_memory_db.py, nightly-gmemory-aggregate.sh, optimize_prompts.py}` exist; dangling refs marked in `docs/research/PHASE_*.md` | WP4 / WP6 |
| research-docs-05 bibliography placeholder URL / title | P3 | `docs/BIBLIOGRAPHY.md` (titles verified against arXiv 2026-10-11) | WP6 |
| research-docs-06 paid-API dollar budgets | P3 | `docs/research/PHASE_0_BENCH.md` §7, `PHASE_3_RECURSION.md` budget tables reframed as token budgets | WP6 |
| critic: CITATION.cff at 0.1.0; stub in this file | P3 | `CITATION.cff` 0.3.0; stub removed | WP6 |
| critic gap A (Codex sandbox cwd) | P3 | `orchestrator.py:88-91` scratch cwd | WP3 |
| critic gap B (Mythos no memory) | P2 | `mythos/runner.py:83-150` | WP2 |

Not fixed, by decision: live bench slate run and mythos-bench regeneration (need real tokens); agents-make-tools; research-swarm port; MCP surface — all in `docs/ROADMAP.md`.

**Prevention rule:** every claim in README/docs must point at a `file:line` or say "not measured" (`docs/CLAIMS_AND_EVIDENCE.md` is the register). A library that ships without an orchestrator call site is labelled EXPERIMENTAL, UNWIRED in its `__init__` docstring and in the README until it is wired. Model names live in `blitz.toml` only; the grep guard is the gate.

**Commits:** 88cd49f (WP1), 36767cd (WP2-4); WP5/WP6 committed by the orchestrating session.
