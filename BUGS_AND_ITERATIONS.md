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

<!-- Format:
## YYYY-MM-DD: Short Title

**Problem:** What went wrong or needed changing
**Root cause:** Why it happened
**Fix:** What was done to resolve it
-->
