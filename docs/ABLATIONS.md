# Ablations and Design Decisions

> **Superseded 2026-10-11 (v0.3.0).** This file describes the v0.1 system as of 2026-03-14 and is kept as history. For what the code does today see `README.md`, `docs/MODELS.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/IMPLEMENTATION_STATUS.md` and `docs/LIMITATIONS.md`. `metrics.jsonl` referenced below was archived to `docs/data/metrics-2026-03_06.jsonl`; live runs write `output/metrics.jsonl`.

> **No ablation has been run since either.** Of the six decisions: (1) is now *staged* round 1, not fully simultaneous; (2) Tiers 2–3 are implemented and wired (v0.3.0), still unvalidated; (3) LanceDB is optional and off the default path — the store is SQLite FTS5 with optional vectors in SQLite; (4) MiniLM is optional, a hash stub stands in without it; (5) holdout override unchanged, never triggered live; (6) in-memory fallback unchanged.

No formal ablation studies have been conducted. This document records the rationale behind key design decisions.

## 1. Parallel vs. Sequential Execution

**Decision:** All agents fire simultaneously, not in waves.

**Rationale:** Research Swarm's sequential pipeline imposes ordering that may cause anchoring — later agents are influenced by earlier agents' framing. Parallel execution avoids this by design.

**Status:** Design bet. No empirical comparison between parallel and sequential on identical topics.

## 2. G-Memory Three-Tier Architecture

**Decision:** Adopted G-Memory's hierarchical model (Zhang et al., NeurIPS 2025) — interaction traces, semantic graphs, distilled insights.

**Rationale:** Flat memory stores (simple vector databases) lose temporal and causal structure. G-Memory's three tiers preserve the distinction between raw data, task-level patterns, and cross-task generalizations.

**Status:** Tier 1 implemented. Tiers 2-3 proposed. The architecture is motivated by published research (20.89% improvement in task success reported by Zhang et al.) but not validated in this codebase.

## 3. Redis + SQLite + LanceDB Stack

**Decision:** Three storage backends for three access patterns.

**Rationale:**
- Redis: hot-path coordination (~0.1ms latency), pub/sub notifications
- SQLite+WAL: durable persistence with concurrent reads
- LanceDB: embedded vector search without infrastructure

**Status:** Redis blackboard implemented. SQLite storage partially implemented. LanceDB integration exists in embedder.py but retrieval pipeline not connected.

## 4. MiniLM-L6-v2 for Embeddings

**Decision:** Use all-MiniLM-L6-v2 (22M parameters, 384 dimensions) for semantic similarity.

**Rationale:** Small enough to run on CPU without GPU infrastructure. Fast enough for real-time embedding during agent execution. Accuracy is "good enough" for coarse-grained similarity — the embedder is the first filter, not the final decision.

**Status:** Implemented. No comparison against alternative embedding models.

## 5. Holdout Override After 3 Rounds

**Decision:** If one agent dissents while all others agree for 3+ rounds, override the holdout and preserve their dissent in the output.

**Rationale:** Prevents one agent from blocking convergence indefinitely while still documenting their concerns.

**Status:** Implemented but never triggered (neither run reached the override condition).

## 6. No-Redis Fallback

**Decision:** Support running without Redis using an in-memory blackboard.

**Rationale:** Reduces setup friction for first-time users. Redis is a dependency that not everyone has installed.

**Status:** Implemented. The successful run (Run 1) may have used this mode.
