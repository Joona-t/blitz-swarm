# Research Log

> **Corrections (2026-11-04, from `docs/audit/2026-10-27/AUDIT.md`).** The entries below are
> kept as written in March 2026. Three statements in them do not match the code or the data:
> (1) the consensus rule was **unanimous** "ready", not a 2/3 majority (`consensus.py`);
> (2) the judge ran on **Sonnet**, not Opus (`agents.py` role models at the time);
> (3) run 1's dimension scores in `metrics.jsonl` are coverage 6 / accuracy 9 / clarity 8 /
> depth 5, not 7/7/7/7. The run-2 diagnosis ("self-referential topic") was speculation:
> rounds 2–3 errored within ~3 s, which is a CLI or rate-limit failure signature.


> **Superseded 2026-10-11 (v0.3.0).** This file describes the v0.1 system as of 2026-03-14 and is kept as history. For what the code does today see `README.md`, `docs/MODELS.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/IMPLEMENTATION_STATUS.md` and `docs/LIMITATIONS.md`. `metrics.jsonl` referenced below was archived to `docs/data/metrics-2026-03_06.jsonl`; live runs write `output/metrics.jsonl`.

A lab-notebook-style record of Blitz-Swarm development and runs.

## 2026-03-12: Initial Architecture (v0.1.0)

**Question:** Can parallel multi-agent consensus with dissent preservation produce higher-quality research output than sequential pipelines?

**Design decisions:**
- 5 agents in parallel (not sequential) to test whether simultaneous analysis reduces groupthink
- Blackboard architecture for shared state (agents write to a common surface, read from each other)
- 3-round consensus with 2/3 majority threshold
- Dissent preservation: minority positions recorded even when consensus is reached
- G-Memory with 3 tiers (Short-term, Long-term, Meta) for cross-run learning

**Implementation:** Built core system: orchestrator.py, agents.py, consensus.py, blackboard.py, embedder.py (MiniLM for semantic similarity), config.py. Memory pipeline scaffolded but only Tier 1 (short-term) functional.

**No runs yet.** System built but not tested.

## 2026-03-14 20:16: Run 1 — SQLite WAL Mode Internals

**Topic:** SQLite WAL mode internals
**Config:** 5 agents, 3 rounds, Opus judge
**Wall clock:** 925.8 seconds (15.4 minutes)
**Cost:** $2.255

**Results:**
- Quality: 7.0/10 (highest single-run score in the LoveSpark research ecosystem)
- Coverage: 7.0, Accuracy: 7.0, Clarity: 7.0, Depth: 7.0
- Consensus reached: NO (override_applied=false)
- Agents never reached 2/3 majority across 3 rounds
- Despite no consensus, the synthesized output was rated high quality by the judge

**Observation:** High confidence (0.91 average) but low consensus suggests the threshold may be too strict, or the agents genuinely disagreed on significant points. The synthesis was good regardless — the disagreements were productive, not destructive.

**Question raised:** Is consensus the right goal, or is structured disagreement the better output format?

## 2026-03-14 20:32: Run 2 — Consensus Mechanisms (FAILED)

**Topic:** Consensus mechanisms in multi-agent systems
**Config:** 5 agents, 3 rounds, Opus judge
**Wall clock:** 267.9 seconds (4.5 minutes — premature termination)
**Cost:** $0.473

**Results:**
- Quality: 0.0/10 (total failure)
- Round 1: 2/5 agents errored
- Rounds 2-3: Cascading failure — remaining agents could not recover from missing peer outputs
- No synthesis produced

**Root cause analysis:**
- The topic (consensus mechanisms) caused agents to meta-reason about their own process rather than researching external literature
- Self-referential topics may trigger a feedback loop where agents analyze their own behavior instead of the research domain
- Cascading failure confirms LIMITATIONS.md concern about no graceful degradation when agents error

**Lesson:** Need per-agent error isolation. One failed agent should not poison the blackboard for subsequent rounds. Also: avoid meta-topics until the system is more robust.

## Current Status (2026-03-14)

- **Total runs:** 2 (1 success, 1 failure)
- **Sample size:** Insufficient for any statistical claims
- **Memory pipeline:** Tier 1 only, not validated across runs
- **Consensus mechanism:** Tested but never achieved (1 run disagreed productively, 1 run cascaded into failure)
- **Key open question:** RQ1 (does parallel outperform sequential?) remains untested — would need 10+ paired runs against Research Swarm on the same topics

**Next steps:**
1. Implement agent error isolation (catch per-agent failures, continue with remaining agents)
2. Run 5-10 more topics to get minimum viable data
3. Pair 3 topics with Research Swarm runs for direct comparison

## 2026-05-09 → 05-10: v0.2.0 modules, Mythos mode, mythos-bench

Phases 0–3 shipped as tested libraries (`bench/`, `mechanisms/`, `gmemory/`, `evolve/`); Mythos mode added
(`BUGS_AND_ITERATIONS.md` ITER-MYTHOS-001..003). Three live Mythos-vs-consensus runs on the Claude path:
Mythos 25 % cheaper, 58 % faster on `binary_search`; `monetary_decimal` passed 17 hidden adversarial tests.
n=3–4, directional only. These are the only live numbers the repo has ever produced after 2026-03-14.

## 2026-05-20 → 05-21: v0.2.1/0.2.2 Codex-first backend package

Backend contract (`AgentCall`/`AgentResult`), Codex default, mechanisms wired into the run loop. 264 tests.
No bench run.

## 2026-06-09 → 06-18: upgrade harness

`jobs/blitz_upgrade.py` 10M-token self-upgrade loop built and audited (ITER-AUDIT-1, 444 tests). Live run
2026-06-10 stalled at phase 1/5 after 829k tokens (claude outage burst → zero-signature cells; fixes in
1a4876c). Not completed. A 2026-06-18 consensus run had 4/5 agents errored every round (quality 0) —
in `docs/data/metrics-2026-03_06.jsonl`.

## 2026-10-11: audit + v0.3.0 remediation

Adversarial audit: 47 findings, 0 debunked (`docs/plans/2026-10-11-fable-remediation.md`). Headline: gmemory
had zero callers (BUG-004), Mythos never ran its deep model (BUG-005), p-value `or 1` (BUG-003), README
diagram false on staging/KS-stop/blackboard, 36 model literals in 12 files. Remediation WP1–WP6 → 745
passed / 14 skipped (667 / 14 at 36767cd before WP5). Mythos defaults to Fable 5.1 on claude. **Still no live bench run.** Next:
`docs/ROADMAP.md` item 1.

## 2026-11-04 — first live cross-provider runs on `next/v0.4` (n=2, one topic each)

Both runs: researchers on Codex (`gpt-6.1-sol`), critic / fact-checker / judges pinned to Claude
(`sonnet`) through `[models] <role>_provider`, no Redis, heuristic planner, `BLITZ_CONFIG` copy of
`blitz.toml`. Numbers are from `output/metrics.jsonl` and the run logs; quality is the swarm's own
judge (same caveat as every earlier entry: not an independent measurement).

**Run A** `20261104_224048` — "How does SQLite WAL mode change the write path, and what are its
failure modes?" — `cheap` profile (guard/judge-ensemble/selector off), 1 round + synthesis.
Round 1: 5/5 agents ok, `providers_by_role` = researcher→codex, critic/fact_checker/quality_judge→
claude; consensus 0/3 ready; cost `>= $0.48` (Claude reported, Codex unknown); 166,492 input /
29,339 output tokens; wall 549 s. Synthesis: Codex synthesizer **timed out at 300 s**; the document
fell back to an empty findings section. Judge dims 6/8/7/6 (avg 6.8).
→ fixes: `[swarm] synthesizer_timeout_seconds`, findings fallback to the last round with
researchers, timeouts counted as timeouts (commit `938ea77`).

**Run B** `20261104_225646` — "What does HTTP/3 change about head-of-line blocking compared with
HTTP/2, and where does it not help?" — `balanced` profile (guard + judge ensemble + selector on),
2 rounds + selector synthesis. One of two Codex researchers timed out at 300 s in **each** round
(effort `high`; the other took 85 s / 122 s). Judge ensemble: one scoring per round (15.3 s, 10.6 s)
— the persistence fix holds. Cascade guard promoted 2 / 2 / 9 Yellow claims to Green by
corroboration across the phases; 0 blocked. Consensus 2/3 then 1/3 ready. Selector synthesis
completed in 15.3 s and the document carries a "Critique and dissent" section. Cost `>= $0.47`;
151,999 tokens; wall 696 s; judge dims 6.7/9.2/8.7/7.7 (avg 8.0).
→ findings: the judge-ensemble and pairwise calls had no `model_role`, so they **bypassed the judge
pin** (ran on Codex) and were invisible to metrics (synthesis `by_role={}`, cost None) — fixed in
`653747e`; Codex provider effort lowered to `medium` (`623c834`).

What these two runs do and do not show: the provider layer routes roles as configured and records
it; the mechanisms run end-to-end on real output; a Codex researcher at `high` effort does not fit a
300 s budget about half the time. They say nothing about quality relative to a single agent
(proposal Phase 1 is still unrun).

**Run C** `20261104_231125` — "How do CRDTs guarantee convergence without coordination, and what can
they not express?" — `balanced` profile, 2 rounds + selector synthesis, after `653747e` (hook
routing/telemetry) and `623c834` (Codex effort `medium`). No timeouts: researchers 80–104 s,
evaluators 23–37 s, judge ensemble 7.3 s per round. `providers_by_role` now shows
`quality_judge → claude` in both rounds and `synthesizer → claude` (the selector's pairwise judges)
with cost **reported** ($0.204); round costs $0.288 / $0.284; total `>= $0.78` (Codex still
uncosted, no pricing table); 278,926 tokens; wall **258 s** (vs 696 s for run B). Guard promoted
4 / 4 / 7 claims, 0 blocked. Consensus 1/3 then 2/3; judge dims 7.6/9.0/8.5/8.0 (avg 8.3). The
320-line document has a "Critique and dissent" section. Same caveat: self-judged quality.
