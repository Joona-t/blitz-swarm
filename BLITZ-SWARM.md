# BLITZ-SWARM

> A parallel multi-agent swarm architecture for collective AI intelligence.
> All agents live. All agents share memory. All agents iterate together until consensus.

---

## What This Is

Blitz-Swarm is an open-source, parallel multi-agent system built on three principles:

**No waves.** Traditional agent pipelines run in sequential waves — one tier finishes before the next starts. Blitz-Swarm fires all agents simultaneously. Every agent reads from and writes to a shared live blackboard. The swarm thinks as a unit, not a pipeline.

**Shared hierarchical memory.** Agents don't just coordinate — they accumulate. Every task the swarm completes makes it smarter on future tasks. Memory is organized across three tiers (interaction traces, query patterns, distilled insights) using the G-Memory architecture (NeurIPS 2025).

**Consensus-driven convergence.** The swarm doesn't stop on a timer or after a fixed number of rounds. It stops when all agents agree the output meets quality bar. Dissent is preserved, not suppressed.

This is infrastructure for the AI field. Not a wrapper. Not a product. A reference architecture for how parallel agent swarms should work.

---

## First Target Task: Research + Summarize a Technical Topic

Given a topic string, Blitz-Swarm deploys a parallel agent swarm that collectively researches, cross-validates, critiques, and synthesizes a high-quality technical summary.

No single agent produces the output. The output *emerges* from the swarm's collective iteration over shared memory.

**Output format:** Structured markdown — core concepts, key findings, implementation implications, open questions, source confidence ratings, and a dissent section preserving minority views.

---

## How Agents Are Defined

Agent count is **dynamic — decided by the orchestrator at runtime based on task complexity.** The orchestrator analyzes the topic, estimates domain breadth, and spawns the minimum number of agents needed to achieve coverage without redundancy.

### Agent Roles

| Role | Count | Responsibility |
|------|-------|----------------|
| **Researcher** | 2–4 | Deep-dives into an assigned subtopic. Writes findings to blackboard. |
| **Critic** | 1–2 | Reads all researcher outputs. Flags gaps, contradictions, low-confidence claims. |
| **Synthesizer** | 1 | Integrates all findings into a coherent structured summary. |
| **Fact-Checker** | 1 | Cross-validates specific claims across researcher outputs. |
| **Quality Judge** | 1 | Scores synthesized output on coverage, accuracy, clarity, depth. Casts consensus vote. |

**Typical spawn count:** 6–8 agents for a standard technical topic.
For a narrow topic (e.g. "SQLite WAL mode internals"): 4 agents.
For a broad topic (e.g. "multi-agent memory architectures"): up to 12 agents.

### Agent Definition Schema

```python
@dataclass
class BlitzAgent:
    id: str                  # e.g. "researcher_01"
    role: str                # "researcher" | "critic" | "synthesizer" | "fact_checker" | "quality_judge"
    subtopic: str            # Assigned scope (e.g. "embedding strategies for memory retrieval")
    system_prompt: str       # Full role instruction injected at invocation
    max_iterations: int = 3  # Max rounds this agent participates in
```

---

## Subprocess Architecture

Blitz-Swarm agents are **stateless CLI invocations.** The orchestrator is the only persistent process — it owns all state, all memory, all coordination. Agents receive context in their prompt and return structured output. They never touch storage directly.

This design is model-agnostic. Any CLI-invocable model works as an agent backend.

### Invocation Pattern

```python
import subprocess

def invoke_agent(agent: BlitzAgent, context: str, task: str) -> str:
    prompt = f"""
{agent.system_prompt}

## Your Assigned Subtopic
{agent.subtopic}

## Shared Memory Context (current blackboard state)
{context}

## Task
{task}

## Output Format
Provide your findings, then close with a memory block:
```memory
{{
  "key_findings": ["finding 1", "finding 2"],
  "confidence": 0.0-1.0,
  "gaps_identified": ["gap 1"],
  "quality_vote": "ready" | "needs_work",
  "quality_notes": "reason for vote"
}}
```
"""
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=120
    )
    return result.stdout
```

### Orchestrator Lifecycle

```
1. SPAWN       — Analyze topic, define agents, initialize blackboard
2. BLAST       — All agents invoked simultaneously via asyncio.gather()
3. WRITE       — Agent outputs parsed, memory blocks queued to Redis Stream
4. CONSOLIDATE — MemoryWriter commits to SQLite + LanceDB, updates G-Memory tiers
5. CHECK       — Evaluate consensus across all quality votes
6. ITERATE     — If no consensus: re-invoke agents with updated blackboard context
7. FINALIZE    — Synthesizer produces final document, output saved to disk
```

### Parallel Invocation

```python
import asyncio

async def blast_all_agents(agents: list[BlitzAgent], context: str, task: str):
    tasks = [
        asyncio.to_thread(invoke_agent, agent, context, task)
        for agent in agents
    ]
    return await asyncio.gather(*tasks)
```

All agents receive the **same blackboard snapshot** at the start of each round. No agent waits for another. Between rounds, the orchestrator consolidates all outputs before the next blast.

---

## Convergence Condition

**Blitz-Swarm stops when all voting agents reach unanimous "ready" consensus.**

### Consensus Algorithm

```python
def check_consensus(agent_outputs: list[dict]) -> bool:
    voters = [o for o in agent_outputs if o.get("quality_vote") is not None]
    if not voters:
        return False
    return all(v["quality_vote"] == "ready" for v in voters)
```

### Convergence Rules

| Condition | Action |
|-----------|--------|
| All voters say `"ready"` | ✅ Converged — finalize output |
| Any voter says `"needs_work"` | 🔄 Re-iterate with updated blackboard |
| Max iterations reached (default: 5) | ⚠️ Force-finalize with quality warning |
| One holdout after 3 rounds, all others ready | 🔍 Override + log dissent |

### What Triggers "needs_work"

- Factual contradiction between researcher outputs not yet resolved
- Critical subtopic with zero coverage
- Synthesizer output that misrepresents a finding
- Confidence below 0.6 on a core claim

### Dissent Preservation

Every `"needs_work"` vote and `quality_notes` entry is preserved in the final output under `## Dissent & Open Questions`. The swarm surfaces disagreement rather than hiding it. Minority views are first-class output.

---

## Memory Architecture

Blitz-Swarm implements G-Memory's three-tier hierarchical memory (NeurIPS 2025, arXiv 2506.07398), adapted for a subprocess-based parallel swarm.

### Storage Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Working memory | Redis (Hash + Streams + Pub/Sub) | Live blackboard, write queue, event bus |
| Persistent graphs | SQLite + WAL | G-Memory three tiers (Interaction, Query, Insight graphs) |
| Semantic retrieval | LanceDB (embedded) | Vector similarity search for query recall |
| Embedding model | all-MiniLM-L6-v2 (384-dim) | Loaded once in orchestrator, never in agents |

### The Three Memory Tiers

**Tier 1 — Interaction Graph:** Raw agent communication traces per task. Every utterance, every causal link between outputs. The finest-grained record of how the swarm solved a problem.

**Tier 2 — Query Graph:** Task-level nodes connected by semantic similarity. 1-hop expansion at retrieval time surfaces related past tasks without noise (2+ hops degrade performance).

**Tier 3 — Insight Graph:** LLM-distilled generalizations extracted from interaction traces. Hyper-edges connect insights through the queries that validated them. This is the swarm's long-term learned knowledge — it compounds across tasks.

### Write Serialization

All memory writes funnel through a single `MemoryWriter` process consuming a Redis Stream. Agents never write to storage directly. This eliminates SQLite write contention at any agent count.

### Retrieval Pipeline (per task)

```
1. Embed new query → cosine similarity search → top-2 historical queries
2. 1-hop graph expansion → expand candidate set
3. Upward traversal → collect insights whose Ω set overlaps candidates
4. LLM relevance scoring → select top-3 most relevant historical interactions
5. LLM sparsification → extract essential subgraph, discard noise
6. Inject context → build agent prompt with insights + interaction fragments
```

Full implementation details in the companion research documents.

---

## Design Principles

**Model-agnostic.** Any CLI-invocable model can be an agent backend. The architecture doesn't assume Claude, GPT, or any specific provider.

**No frameworks.** Vanilla Python + asyncio + subprocess. No LangGraph, no CrewAI, no AutoGen. Every component is explicit and inspectable.

**Infrastructure, not product.** Blitz-Swarm is a reference implementation and research platform. It is meant to be studied, forked, adapted, and improved — not wrapped in a UI and sold.

**Privacy-first.** All memory is local. No telemetry. No external APIs required beyond the model backend.

**Compounding intelligence.** Each task makes the swarm smarter. The insight tier distills accumulated experience into generalizable knowledge that improves future task performance without retraining any model weights.

---

## Repository Structure

```
blitz-swarm/
├── BLITZ-SWARM.md          # This file
├── README.md               # Project overview + quickstart
├── orchestrator.py         # Main entrypoint + agent spawning + consensus loop
├── agents.py               # BlitzAgent dataclass + role prompt definitions
├── memory/
│   ├── writer.py           # MemoryWriter (single-writer daemon)
│   ├── reader.py           # MemoryReader (retrieval pipeline)
│   ├── schema.sql          # SQLite schema — G-Memory three tiers
│   └── models.py           # Dataclasses: Utterance, QueryNode, InsightNode, InsightEdge
├── blackboard.py           # Redis interface (read / write / subscribe)
├── embedder.py             # Embedding model wrapper (MiniLM, loaded once at startup)
├── consensus.py            # Convergence logic + dissent logging
└── output/
    └── {topic}_{timestamp}.md   # Final research summaries
```

---

## Implementation Constraints

- Python only — no other languages in core
- No agent frameworks — vanilla asyncio + subprocess throughout
- External services: Redis (single instance, `docker run redis`), SQLite (file), LanceDB (embedded, zero config)
- Single machine target — no distributed infrastructure required
- Model backend: any CLI-invocable model; reference implementation uses Claude Code CLI
- Code style: explicit over clever, constants at file tops, no unnecessary abstractions

---

## Status

🔨 **In development.** Research phase complete. Implementation in progress.

Reference research:
- Blackboard architecture + storage layer: `docs/memory-architecture-research.md`
- G-Memory hierarchical memory: `docs/g-memory-implementation-guide.md`

---

## License

MIT. Build on it. Break it. Make it better.
