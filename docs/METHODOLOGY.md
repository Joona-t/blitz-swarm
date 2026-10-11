# Methodology

> **Superseded 2026-10-11 (v0.3.0).** This file describes the v0.1 system as of 2026-03-14 and is kept as history. For what the code does today see `README.md`, `docs/MODELS.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/IMPLEMENTATION_STATUS.md` and `docs/LIMITATIONS.md`. `metrics.jsonl` referenced below was archived to `docs/data/metrics-2026-03_06.jsonl`; live runs write `output/metrics.jsonl`.

> **What changed since:** config is `blitz.toml [swarm]` (`max_rounds = 4`, `timeout_seconds = 300`, `max_agents = 10`), models are tiers resolved per role (`docs/MODELS.md`), round 1 is staged, the blackboard is read back, and the bench experiment design that replaces this ad-hoc per-run structure is `bench/runner.py` + `bench/stats.py` (paired t, exact Student-t for df < 30, Cohen's d_z, bootstrap CI) — none of which has been run live. Cost is `n/a` on providers without `[pricing]`.

## Experiment Structure

Each run constitutes a single experiment:

1. **Input:** A research topic string
2. **Configuration:** Max rounds, model, timeout, agent count bounds (blitz.toml)
3. **Execution:** Dynamic agent planning → parallel blast rounds → consensus check → synthesis
4. **Output:** A markdown document with synthesized findings, dissent section, and convergence report
5. **Measurement:** Per-run metrics logged to metrics.jsonl

## Variables

### Independent Variables
- Research topic
- Max consensus rounds (default: 5)
- Redis vs. in-memory blackboard
- LLM-planned vs. heuristic agent composition

### Controlled Variables
- Default model (Sonnet)
- Agent timeout (180s)
- Max agents (12)
- Consensus mechanism (unanimous voting)

### Dependent Variables
- Quality scores (coverage, accuracy, clarity, depth — 0-10)
- Rounds to convergence
- Consensus reached (boolean)
- Override applied (boolean)
- Total cost (USD)
- Wall clock time
- Agent success/error/timeout counts per round

## What Counts as Success

A run is "successful" when:
1. At least one round produces parseable agent output
2. The synthesizer produces a final document
3. The convergence report is generated

Note: consensus is not required for success — the system can produce output even when agents disagree.

## Evaluation Caveats

1. **n=2.** Two runs provide no statistical basis for any conclusion. All observations are anecdotal.
2. **Single evaluator.** Quality scores come from the judge agent (Claude Sonnet), not human evaluation.
3. **No baseline.** No comparison against sequential pipeline, single-agent, or manual research.
4. **Cost tracking is rough.** USD costs are computed from Claude CLI output, not from billing API.
