# Research Log

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
