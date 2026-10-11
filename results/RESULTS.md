# Results

> **Superseded 2026-10-11 (v0.3.0).** This file describes the v0.1 system as of 2026-03-14 and is kept as history. For what the code does today see `README.md`, `docs/MODELS.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/IMPLEMENTATION_STATUS.md` and `docs/LIMITATIONS.md`. `metrics.jsonl` referenced below was archived to `docs/data/metrics-2026-03_06.jsonl`; live runs write `output/metrics.jsonl`.

> n=2 from 2026-03-14 on v0.1.0. No later run table exists; the only later live numbers are the four 2026-05-10 mythos-bench rows (also stale). The first v0.3.0 results will come from `bench/runner.py` (`docs/ROADMAP.md` item 1).

## Run Table (2 runs)

| Run | Topic | Rounds | Consensus | Override | Quality | Coverage | Accuracy | Clarity | Depth | Cost | Time | Agents | Errors |
|-----|-------|--------|-----------|----------|---------|----------|----------|---------|-------|------|------|--------|--------|
| 1 | SQLite WAL mode internals | 3 | No | No | 7.0 | 6 | 9 | 8 | 5 | $2.26 | 926s | 16 | 1 timeout |
| 2 | Consensus in distributed systems | 3 | No | No | 0.0 | 0 | 0 | 0 | 0 | $0.47 | 268s | 16 | 12 errors |

## Run 1 Details: SQLite WAL mode internals

| Round | Agents OK | Errors | Timeouts | Ready Votes | Total Voters | Avg Confidence | Cost | Time |
|-------|-----------|--------|----------|-------------|--------------|----------------|------|------|
| 1 | 5/5 | 0 | 0 | 1/3 | 3 | 0.91 | $0.74 | 267s |
| 2 | 4/5 | 0 | 1 | 1/3 | 3 | 0.91 | $0.61 | 300s |
| 3 | 5/5 | 0 | 0 | 0/3 | 3 | 0.91 | $0.79 | 231s |
| Synthesis | 1/1 | 0 | 0 | — | — | — | $0.12 | 101s |

**Observations:** High agent confidence (0.91) but low consensus (never more than 1/3 ready). Suggests agents are confident in their individual findings but the judge has high standards. Round 2 had one timeout. Overall output was usable despite no formal consensus.

## Run 2 Details: Consensus in distributed systems

| Round | Agents OK | Errors | Timeouts | Ready Votes | Total Voters | Avg Confidence | Cost | Time |
|-------|-----------|--------|----------|-------------|--------------|----------------|------|------|
| 1 | 3/5 | 2 | 0 | 1/3 | 3 | 0.31 | $0.47 | 243s |
| 2 | 0/5 | 5 | 0 | 0/3 | 3 | 0.00 | $0.00 | 3s |
| 3 | 0/5 | 5 | 0 | 0/3 | 3 | 0.00 | $0.00 | 3s |
| Synthesis | 0/1 | 1 | 0 | — | — | — | $0.00 | 2s |

**Observations:** Catastrophic cascade. Round 1 had 2/5 agent errors and low confidence (0.31). Rounds 2-3 failed completely (all agents errored in ~3 seconds, suggesting immediate parse/context failure). Total wasted cost: $0.47.

## Aggregate Statistics

Not meaningful with n=2. Provided for completeness:

| Metric | Run 1 | Run 2 |
|--------|-------|-------|
| Success | Yes | No |
| Total cost | $2.26 | $0.47 |
| Total time | 926s | 268s |
| Agent invocations | 16 | 16 |
| Agent errors | 1 | 12 |
| Consensus reached | No | No |
