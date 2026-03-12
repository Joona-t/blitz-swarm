# Blitz-Swarm

> A parallel multi-agent swarm architecture for collective AI intelligence.

All agents live. All agents share memory. All agents iterate together until consensus.

## Quickstart

```bash
# Basic run (no Redis required)
python orchestrator.py "SQLite WAL mode internals" --no-redis

# With Redis blackboard
docker run -d --name blitz-redis -p 6379:6379 redis:7-alpine
python orchestrator.py "multi-agent memory architectures"

# Preview the agent plan without executing
python orchestrator.py "CRDT conflict resolution" --dry-run

# Limit consensus rounds
python orchestrator.py "WebAssembly component model" --max-rounds 3
```

Output lands in `output/{topic}_{timestamp}.md`.

## How It Works

Given a topic string, Blitz-Swarm deploys a parallel agent swarm that collectively researches, cross-validates, critiques, and synthesizes a high-quality technical summary.

No single agent produces the output. The output *emerges* from the swarm's collective iteration over shared memory.

### Agent Roles

| Role | Count | Job |
|------|-------|-----|
| **Researcher** | 2-6 | Deep-dives an assigned subtopic |
| **Critic** | 1-3 | Flags gaps, contradictions, weak claims |
| **Fact-Checker** | 0-1 | Cross-validates specific claims |
| **Quality Judge** | 1 | Scores output on coverage/accuracy/clarity/depth |
| **Synthesizer** | 1 | Integrates everything into a coherent summary |

Agent count is determined dynamically by an LLM analyzing the topic's complexity and breadth.

### Orchestrator Lifecycle

```
1. SPAWN       — Analyze topic, plan agents, initialize blackboard
2. BLAST       — All agents invoked simultaneously via asyncio
3. WRITE       — Agent outputs written to blackboard
4. CHECK       — Evaluate consensus across quality votes
5. ITERATE     — If no consensus: re-invoke with updated context
6. FINALIZE    — Synthesizer produces final document
```

### Convergence

The swarm stops when all voting agents reach unanimous "ready" consensus. If one holdout remains after 3 rounds with all others ready, the holdout is overridden and their dissent is preserved in the final output. Max 5 rounds by default.

### Memory (G-Memory)

Blitz-Swarm implements G-Memory's three-tier hierarchical memory (NeurIPS 2025):

- **Tier 1 — Interaction Graph:** Raw agent communication traces per task
- **Tier 2 — Query Graph:** Task-level nodes connected by semantic similarity
- **Tier 3 — Insight Graph:** LLM-distilled generalizations that compound across tasks

Each completed task makes the swarm smarter on future tasks. Memory is stored in SQLite + LanceDB, with Redis for live coordination.

## Architecture

```
orchestrator.py          Main entrypoint + lifecycle
agents.py                Agent definitions + role prompts + planning
consensus.py             Convergence logic + dissent preservation
blackboard.py            Redis interface for live coordination
embedder.py              Sentence-transformers wrapper (MiniLM)
config.py                Configuration loading (blitz.toml)
memory/
  schema.sql             SQLite DDL for G-Memory tiers
  models.py              Dataclasses for all memory entities
  writer.py              Single-writer daemon for persistence
  reader.py              Retrieval pipeline (embed → expand → traverse)
```

## Configuration

Edit `blitz.toml` to customize:

```toml
[swarm]
max_rounds = 5
default_model = "sonnet"
timeout_seconds = 180

[memory]
max_context_tokens = 2000
top_k_retrieval = 2
insight_dedup_threshold = 0.85

[eviction]
query_archive_days = 90
interaction_purge_days = 30
```

## CLI Options

```
python orchestrator.py TOPIC [OPTIONS]

Options:
  --max-rounds N      Maximum consensus rounds (default: 5)
  --no-redis          Run without Redis (in-memory mode)
  --no-llm-plan       Use heuristic agent planning
  --dry-run           Show agent plan without executing
  --verbose           Detailed logging
```

## Requirements

- Python 3.12+
- Claude CLI (`claude`)
- Redis (optional, for blackboard — `docker run redis`)
- sentence-transformers (for memory embeddings)
- LanceDB (for vector search)

```bash
pip install redis aiosqlite lancedb sentence-transformers
```

## Design Principles

- **No waves.** All agents fire simultaneously, not sequentially.
- **No frameworks.** Vanilla Python + asyncio + subprocess. No LangGraph, no CrewAI.
- **Model-agnostic.** Any CLI-invocable model works as an agent backend.
- **Privacy-first.** All memory is local. No telemetry. No external APIs beyond the model.
- **Compounding intelligence.** Each task makes the swarm smarter.

## License

MIT. Build on it. Break it. Make it better.
