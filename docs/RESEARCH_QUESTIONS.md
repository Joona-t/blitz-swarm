# Research Questions

> **Superseded 2026-10-11 (v0.3.0).** This file describes the v0.1 system as of 2026-03-14 and is kept as history. For what the code does today see `README.md`, `docs/MODELS.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/IMPLEMENTATION_STATUS.md` and `docs/LIMITATIONS.md`. `metrics.jsonl` referenced below was archived to `docs/data/metrics-2026-03_06.jsonl`; live runs write `output/metrics.jsonl`.

> **Status update 2026-10-11:** RQ1 untested (no paired run; see `docs/ROADMAP.md` 1, 3). RQ2 mechanism implemented (`consensus.final_votes`, unresolved votes only); value unevaluated. RQ3 insufficient data (no live bench). RQ4 now *testable* — Tiers 2–3 run (`gmemory/store.py`) — but **not tested**: no multi-run series exists. RQ5 untested. Nothing below has changed status from "not tested" by measurement; only RQ4's blocker moved from "not implemented" to "not run".

## RQ1: Does simultaneous agent execution produce qualitatively different output than sequential phasing?

**Why it matters:** Sequential pipelines (like Research Swarm) impose an ordering that may be artificial for some tasks. If parallel execution discovers more diverse findings by avoiding anchoring effects, it could be preferable for exploratory research.

**Status:** Not tested. No direct comparison between Blitz-Swarm and Research Swarm on identical topics has been conducted.

## RQ2: Does explicit dissent preservation improve output completeness?

**Why it matters:** Most multi-agent systems force consensus or take majority votes, potentially losing valuable minority insights. If preserving dissent adds information that consensus suppresses, it changes how we should design convergence mechanisms.

**Status:** Partially implemented. The dissent preservation mechanism works (consensus.py extracts and formats minority views), but no evaluation of whether dissent sections add value to the final output has been conducted.

## RQ3: How many consensus rounds are needed for convergence?

**Why it matters:** More rounds mean higher cost and latency. If most topics converge in 2-3 rounds, the max_rounds parameter can be set tightly. If convergence is rare, the holdout override mechanism becomes the primary termination condition.

**Status:** Insufficient data. 1 successful run used 3 rounds without reaching consensus. The holdout override was not triggered. No conclusions possible from n=1.

## RQ4: Does hierarchical memory improve performance on related topics over time?

**Why it matters:** G-Memory's three-tier architecture (interaction traces → semantic graphs → distilled insights) promises compounding intelligence across runs. If validated, this would justify the implementation complexity.

**Status:** Not testable yet. Memory Tiers 2-3 are proposed but not implemented. Only 2 runs exist, which is insufficient to observe cross-run learning even if the pipeline were complete.

## RQ5: What is the optimal agent count for parallel consensus?

**Why it matters:** More agents provide broader coverage but increase cost and make consensus harder to achieve. The dynamic planning system spawns 4-12 agents based on topic complexity, but the mapping from complexity to optimal count is uncalibrated.

**Status:** Not tested. No ablation study on agent count has been conducted.
