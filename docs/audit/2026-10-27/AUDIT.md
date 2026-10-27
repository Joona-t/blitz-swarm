# Blitz-Swarm audit — 2026-10-27

Scope: `main` @ `c1c0bec` (v0.2.0), plus the open v0.3.0 PR #2 (`fable/audit-remediation` @ `86ada20`) and the other unmerged branches. Read-only on everything except this branch. No LLM calls were made; findings come from reading code and running it against mocks or a fake `claude` binary.

**Verification labels.** **[ran]**: reproduced by executing code during this audit. **[read]**: confirmed by reading code, not executed. **[inferred]**: likely from the code, but couldn't be run here (the host disk was full, so `lancedb` wouldn't install).

---

## 1. The one-paragraph verdict

The repo describes a v0.2 "frontier" architecture, but none of it runs on `main`. The orchestrator imports none of `mechanisms/`, `gmemory/`, `evolve/`, `heterogeneity/` or `managed_agents/` **[ran: grep]**. What actually runs is the v0.1 loop: parallel Sonnet researchers with no tools, a unanimous-"ready" vote, and a synthesizer. That loop has bugs that make its output worse than its design suggests. Six real runs exist (`metrics.jsonl`, March 2026, n=1 per topic). One reached consensus. All quality numbers come from the swarm grading itself with the same model. PR #2 wires the mechanisms in for real and is much better tested (745 tests). It has still never been run end-to-end, and some of the wiring it adds is defective. **The repo has more mechanism than evidence.** The fix is mostly measurement and grounding, not more mechanism.

---

## 2. Findings on `main`, ranked

### Critical / high

| # | Finding | Where | Evidence |
|---|---|---|---|
| H1 | **The v0.2 mechanisms are not wired in.** The README's architecture diagram (cascade_guard → judge_ensemble → selector_synth → gmemory) describes code that only unit tests call. `blitz.toml` doesn't even contain the `[guard]`, `[judge_ensemble]`, `[selector]` or `[evolve]` sections the README shows, and `config.py` has no dataclasses for them. | `orchestrator.py:19-32`, `config.py`, `README.md:28-60` | [ran] |
| H2 | **The synthesizer can't see the latest round or any critique.** `build_context` concatenates all rounds oldest-first, caps each researcher at 800 chars, then truncates the whole string at 8,000 chars. With 4 researchers × 3 rounds, two of the four round-3 researchers and *all* critic and judge text were cut. Round-2+ researchers get their context the same way, so "address the critic feedback" is often given no critic feedback. | `orchestrator.py:325-384` | [ran] |
| H3 | **LLM swarm planning never takes effect.** `_llm_plan` and `_split_subtopics_llm` run `json.loads` on the `--output-format json` *envelope* and look for planning keys there. Those keys are never present, so every run spends a haiku call and falls back to the heuristic plan. `PLANNING_SCHEMA` is defined but never passed. Mythos already got this fix (commit `b995365`). | `agents.py:240-287, 499-534` | [read] + [ran, with a fake CLI] |
| H4 | **Agents have unrestricted tool access.** Every `claude` call passes `--dangerously-skip-permissions` with no `--tools`/`--allowedTools` restriction. "Do NOT use any tools" exists only as prompt text. Prompts include other agents' output and memory from earlier runs without delimiting, so a prompt injection in that content could run shell commands in the working directory. Related: `mythos-bench` runs the swarm with `cwd=REPO_ROOT`, so the "hidden" adversarial tests are readable. `run_pytest` also executes LLM-written code unsandboxed. | `orchestrator.py:83`, `agents.py:275,520`, `memory/reader.py:305,352`, `memory/writer.py:355`, `mythos/_invoke.py:69`, `mythos-bench/lib.py:112,269` | [read] |
| H5 | **The output schema is never enforced.** `AGENT_OUTPUT_SCHEMA_JSON` is imported but never passed (`--json-schema`). Parsing falls back to regex brace-matching and a raw-text wrapper that silently assigns `confidence=0.3` / `needs_work`. That is one plausible reason consensus almost never happens. | `orchestrator.py:20, 198-287` | [read] |
| H6 | **Research is parametric-only, but the prompts demand citations.** The researcher prompt requires "specific papers with authors and years" and measured Sharpe/R² numbers, while the output-format block forbids tools. The literature (see RESEARCH.md §4) is consistent that this produces fabricated or garbled references at high rates. Nothing in the system can detect them. | `agents.py` ROLE_PROMPTS, `prompts/*/researcher.md`, `orchestrator.py:171-183` | [read] |
| H7 | **CascadeGuard does nothing with its defaults.** A claim can only turn green if a prior-round green claim exists, and nothing creates the first one. Across 4 strict-mode rounds, including a direct contradiction, it recorded 0 green, 0 red and 0 blocked. Error taint can't propagate because errored outputs return before claims are created, and nothing fills `parent_claim_ids`. Tests get around this by planting green claims by hand. | `mechanisms/cascade_guard.py:153, 238-246, 385-394` | [ran] |
| H8 | **Memory retrieval probably always returns empty.** `table.search(<str>)` runs on a LanceDB table that has no embedding function, and the exception is swallowed into `set()`. Similarity is computed as `1 - _distance`, which is wrong because LanceDB's default distance is squared L2. There is no relevance threshold, so unrelated "memories" would be injected if search did work. | `memory/reader.py:137-141`, `memory/writer.py:215-250` | [inferred] |

### Medium

| # | Finding | Where | Evidence |
|---|---|---|---|
| M1 | **The KS "stability test" can't work at N=3.** With 3 judges, the KS distance takes only 0, ⅓, ⅔ or 1, so `threshold=0.05` really means "identical buckets twice". Read as a real test, the smallest possible two-sample KS p-value at n=m=3 is 2/C(6,3) = 0.10. In simulation it hit `max_rounds` 96% of the time (15 judge calls). Votes are never fed back as debate history, so there is no debate. `bb_mixture` and `seed_strategy` are silently ignored. | `mechanisms/judge_ensemble.py:309-338` | [ran] + arithmetic |
| M2 | **SelectorSynth turns into concatenation.** The Bradley-Terry MLE is correct (it matches scipy BFGS to 4 dp). But spans are clustered by exact `##` heading text, so "Risks" and "Risk Analysis" never compete: 0 judge calls. `hash(uuid)` ordering is non-reproducible. The mock judge prefers the longer span, so the tests only check length. | `mechanisms/selector_synth.py:393, 418-437` | [ran] |
| M3 | **gmemory schema migration fails** on an existing `memory.db` (`no such column: similarity`). The `migrate.py` it refers to doesn't exist. | `gmemory/schema.sql:9-11` | [ran] |
| M4 | **BM25 fails silently on ordinary text.** Raw queries go to FTS5 `MATCH`, and syntax errors are swallowed. `"HTTP/3's QUIC transport"`, `"QUIC: what changes?"` and `"Bitcoin SegWit (BIP141)"` all return `[]`. | `gmemory/hybrid.py:52-66` | [ran] |
| M5 | **The bench runner can't do what its docs say.** MAST detectors are never invoked (`flags=[]`). The budget deducts the planned amount, not the actual spend, and timeouts refund in full. A 30-prompt slate at the observed $0.47–3.70/run would overrun the $8 cap by roughly 10×. There is no resume. Errors are averaged in as quality 0. `_git_sha` uses an unquoted `os.popen` and returns `"nogit"` on paths with spaces. | `bench/runner.py:268` and budget logic | [ran] |
| M6 | **The statistics are mostly right, but the methodology doc overclaims.** Paired t, d_z and Welch match scipy to about 1e-15. However: the bootstrap is percentile, not the BCa that `PHASE_0_BENCH.md:240` claims. The no-scipy fallback uses a normal approximation (n=5, t=2.5 gives p=0.012 against a true 0.067). Wilcoxon, ANOVA/Tukey, Fisher and BH are documented but absent. The power claim is wrong: d_z=0.5 at 80% power needs n≈34, and n=30 detects d_z≈0.53, not 0.35. | `bench/stats.py`, `docs/research/PHASE_0_BENCH.md:240-248` | [ran] |
| M7 | **meta_loop can auto-merge on almost no evidence.** Its "d_z" is a difference in means divided by a baseline SD that defaults to 1.0, with no pairing, no n and no test. Missing holdout, per-dimension or cost checks pass silently: `{"aggregate": 7.0}` → `7.4` **merged**. `slate_v1_held.toml` and `scripts/optimize_prompts.py` don't exist. | `evolve/meta_loop.py:130-193` | [ran] |
| M8 | **The MAST scoreboard is circular.** "9/14 detected" counts keyword-heuristic detectors firing on synthetic scenarios written to trip them. FM-1.2 fires on 11/14 scenarios, including a legitimate critique. FM-2.1 fires on any rise-then-fall vote sequence. No false-positive rate has been measured on real traces. | `bench/detectors.py`, `bench/mast_scoreboard.md` | [ran] |
| M9 | **The unanimity rule and holdout override reward conformity.** Consensus requires every voter to say "ready". A single holdout is overridden from round 3. Errored agents vote "needs_work" and so block consensus. Nothing records whether the answer changed for better or worse between rounds. | `consensus.py` | [read]; see RESEARCH.md §2 |
| M10 | **Blackboard keys aren't namespaced per run,** and `cleanup()` deletes all `blackboard:*` keys. Two concurrent runs would overwrite each other. Redis is unauthenticated. | `blackboard.py` | [read] |
| M11 | **The Mythos verifier is LLM-only.** It never executes code, it is shown the executor's own `acceptance_check`, and it accepts a `"pass"` verdict even when its own `spec_results` fail. | `mythos/verifier.py:110`, `mythos-bench/lib.py:224-232` | [read] |

### Low / docs truth

- `metrics.py`: `avg_quality` excludes zero dimensions, which inflates it. `rounds_to_consensus` records rounds even when consensus wasn't reached. `input_tokens` of 3–15 per round means cache tokens aren't counted, so the token metrics are meaningless.
- README badge says 238 tests; 247 pass on `main`. `pyproject.toml` says `0.1.0`; README says `v0.2.0`.
- `docs/RESEARCH_LOG.md` contradicts `metrics.jsonl` and the code. It gives run-1 dimensions as 7/7/7/7 (the data says 6/9/8/5), describes a "2/3 majority" (the code requires unanimity), and names an "Opus judge" (the code uses Sonnet). Its run-2 diagnosis is speculation: rounds 2–3 errored in about 3s, which looks like a CLI or rate-limit failure.
- `docs/LIMITATIONS.md` #10 says `memory/` doesn't exist; it does. `CLAIMS_AND_EVIDENCE.md` still says "2 runs"; `metrics.jsonl` has 6.
- `docs/BIBLIOGRAPHY.md` / README attribute AFlow (2410.10762) to "Liu". The first author is Jiayi Zhang.
- Maryanskyy 2603.20324 is cited as support for selector *synthesis*. That paper found judge-based **selection** beat synthesis, and that synthesis was never preferred over the baseline. The citation supports selection, not synthesis.

### Tests

247 pass and 14 are skipped (all MAST scenarios). Tests for mechanisms, gmemory and evolve use mocks and check direction only (`t > 0`, `skill[a] > skill[b]`); none checks a value against a reference implementation. No test covers integration with the orchestrator, so H1–H8 all coexist with a green suite.

---

## 3. Evidence inventory: what has actually been measured

| Source | What it holds | What it supports |
|---|---|---|
| `metrics.jsonl` (main) | 6 real swarm runs, 2026-03-14 → 03-25. 6 distinct topics, n=1 each. Consensus in 1/6. Self-graded `avg_quality` from 0.0 to 7.2. Cost $0.47–3.70/run. | That the v0.1 loop runs and what it costs. Nothing about quality: the grading is same-model self-evaluation. |
| `results/RESULTS.md` | The first 2 of those runs. | Same. |
| `bench/runs/` | Doesn't exist. The 30-prompt slate has never been run. | — |
| `bench/mast_scoreboard.md` | Detectors against synthetic scenarios. | Detector code paths execute; nothing about real failure rates. |
| `mythos-bench/BENCH.md` | 4 rows (3 with backfilled cost and time). One head-to-head pair (binary_search: both passed, Mythos 25% cheaper). One adversarial probe, 17/17 passed. Primary tests were written by the swarm itself. | The claim in commit `44f2961`, "Mythos beats consensus on artifact tasks", rests on n=1. |
| `feat/blitz-upgrade-20260609` → `metrics.jsonl` | 28 runs from the 28 June upgrade job. 0/28 reached consensus. 16/28 recorded zero tokens (silent backend outage). The other 12 self-judged 5.9–8.4, at 31k–128k tokens and 13–94 min each. The job stalled at phase 1/5 after 829k tokens. | That the pipeline was unreliable at scale in June, and that consensus essentially never fires. |

**No v0.2 or v0.3 mechanism has ever run on real model output.**

---

## 4. PR #2 (`fable/audit-remediation`, v0.3.0)

**What's real.** `run_swarm` now calls cascade_guard on every write and read, judge_ensemble every round, selector_synth in place of the synthesizer, and gmemory for round-1 retrieval and end-of-run persistence. The blackboard is read back. Consensus agents run with `--tools ""` and `--max-turns 1` in a scratch cwd, which fixes most of H4 for the consensus path. Config-driven models and providers. 745 tests pass and 14 skip; CI is green on 3.12 and 3.13. It merges into `main` without conflicts (`git merge-tree` exit 0). The two `codex/*` branches are fully superseded by it.

**What's still wrong or new.** Spot-checked items are marked [read].
1. **The judge ensemble re-scores the same text in a loop.** A new `JudgeEnsemble` is built each swarm round, and `while not ensemble.is_stable()` re-judges an *unchanged* candidate. The KS test then measures judge self-consistency, not convergence, at 6–15 judge calls per round. If `halt=true` were enabled, it would stop the swarm after round 1 almost every time. `orchestrator.py:900-940` [read]
2. **Selector synthesis drops critique and dissent.** It assembles only researcher spans. The "preserve dissent / quality warning" task text is computed and then not used on that path.
3. **Synthesizer output still goes into Tier-1 memory.** The swarm's own summary feeds back in as interaction evidence; this was flagged in March and is still open.
4. **Packaging works only as an editable install.** `prompts/` isn't in `packages`, the `.toml` configs aren't shipped, and top-level module names like `agents`, `config` and `metrics` go straight into site-packages. CI only tests `-e`. `pyproject.toml [tool.setuptools]` [read]
5. **`--dry-run` calls the LLM planner** unless `--no-llm-plan` is passed.
6. **New hard-coded absolute path:** `/Users/darkfire/Claude x LoveSpark/research-swarm/...`. `jobs/blitz_upgrade.py:644-648` [read]
7. **Defaults are maximal** (`quality_profile="max"`, with guard, judge and selector all on, and codex/gpt-5.5 as the default backend). That is the most expensive configuration, and it has never been measured.
8. The diff is about 19.6k lines, including a 2,282-line `jobs/blitz_upgrade.py`, so review cost is high.

**Other branches.** `feat/blitz-upgrade-20260609` = PR #2's first 7 commits plus a WIP snapshot holding `metrics.jsonl` (the useful June data) and binary `memory.db-shm`/`-wal` files containing personal research topics. Don't merge it; keep the metrics as data. `wip/snapshot-2026-10-25` forks from March and adds 22 `_reports/*.md` files reviewing v0.1.0 (verdicts: PASS WITH CONDITIONS; Karpathy: "70% built / 30% disconnected"). It also adds a README section and a `lab` extra for a `neural_quest_lab/` app that isn't in the tree, and it conflicts with `main` in README.md.

---

## 5. What this audit did not do

- No live runs. Every behavioral claim about quality, cost or convergence above comes from existing logs.
- `lancedb` couldn't be installed (disk full), so H8 is inferred.
- PR #2 was audited to the depth above, not line by line.
- Findings H7, M1–M8 and the PR #2 judge-loop behavior were reproduced by a sub-audit using scratch scripts and a fake `claude` binary that prints a canned envelope. I re-ran H1, H2 and H3 myself and read the PR #2 judge loop, packaging and hard-coded path directly.
