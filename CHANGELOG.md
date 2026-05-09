# Changelog

## [0.1.1] — 2026-05-09

### Reliability and observability

- Agent retry loop on malformed JSON output (1 retry, then graceful error)
- Partial-output recovery on subprocess timeout (parses any JSON the agent flushed before SIGKILL)
- Cost-and-token extraction from the Claude CLI envelope (per-agent, per-round)
- Trace IDs (UUID4) attached to every agent invocation for cross-log correlation
- New `metrics.py` module: `MetricsCollector`, `RunMetrics`, `RoundMetrics` dataclasses; per-run JSONL log at `metrics.jsonl`
- `--max-turns 3` cap on every agent subprocess to prevent runaway tool loops
- Quality-judge schema extended with explicit `coverage_score` / `accuracy_score` / `clarity_score` / `depth_score` (0-10 each)

### Configuration

- `blitz.toml` tuned for production-grade research runs: `max_rounds = 4`, `timeout_seconds = 300`, `max_agents = 10`
- Configuration now loaded centrally via `config.py` instead of hard-coded constants in `orchestrator.py`

### Domain specialization (interim)

- Role prompts specialized for crypto/quant trading research as the v0.1.1 default. v0.2 generalizes this and moves crypto into a preset registry.

## [0.1.0] — 2026-03-12

### Initial Release

- Parallel multi-agent orchestration via asyncio + subprocess
- 5 agent roles: researcher, critic, fact-checker, quality judge, synthesizer
- Dynamic agent planning (LLM-based with heuristic fallback)
- Consensus convergence with holdout override
- Dissent preservation in final output
- Redis-backed blackboard with no-Redis fallback
- G-Memory Tier 1 (interaction traces)
- Sentence-transformer embeddings (MiniLM-L6-v2)
- TOML-based configuration
- Per-run metrics logging

### Research Documentation (same day)

- Added memory architecture literature review
- Added G-Memory blueprint documentation
