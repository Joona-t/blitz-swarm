# Research Log

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
