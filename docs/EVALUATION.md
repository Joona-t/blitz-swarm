# Evaluation

> **Superseded 2026-10-11 (v0.3.0).** This file describes the v0.1 system as of 2026-03-14 and is kept as history. For what the code does today see `README.md`, `docs/MODELS.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/IMPLEMENTATION_STATUS.md` and `docs/LIMITATIONS.md`. `metrics.jsonl` referenced below was archived to `docs/data/metrics-2026-03_06.jsonl`; live runs write `output/metrics.jsonl`.

> Judge dimensions and consensus metrics below are still the ones `metrics.py` records; the judge ensemble (`mechanisms/judge_ensemble.py`) adds per-judge rubric scores and a halt reason. Still automatic, still no human evaluation.

## Scoring System

The quality judge agent scores each round's collective output on 4 dimensions (0-10):

| Metric | Definition | What It Does NOT Capture |
|--------|-----------|--------------------------|
| Coverage | Breadth of topic dimensions addressed | Whether important dimensions were missed entirely |
| Accuracy | Correctness of technical claims | Subtle errors the judge LLM may not catch |
| Clarity | Readability and coherence of output | Whether the structure aids comprehension for specific audiences |
| Depth | Technical depth of analysis | Whether depth is proportional to importance |

## Consensus Metrics

| Metric | Definition |
|--------|-----------|
| Rounds to convergence | Number of blast rounds before unanimous "ready" |
| Ready votes per round | Count of agents voting "ready" |
| Avg confidence per round | Mean of agent confidence scores (0.0-1.0) |
| Override applied | Whether holdout was overridden after 3+ rounds |

## Automatic vs. Human Metrics

All metrics are automatic. No human evaluation has been conducted. No inter-rater reliability exists.

## Current Data

With 2 runs, no aggregate statistics are meaningful. See results/RESULTS.md for raw data.
