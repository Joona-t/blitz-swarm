# Artifacts (v0.3.0, 2026-10-11)

What exists in the tree at commit 36767cd. The 2026-03-14 version of this file is superseded.

## Core system

| File | Purpose | Status |
|---|---|---|
| `orchestrator.py` | `run_swarm()` lifecycle, `--mode mythos` dispatch, `--list-models`, dry run | wired, tested |
| `agents.py` | `plan_agents` (planner tier, heuristic fallback), `BlitzAgent`, `max_agents` enforcement | wired, 19 tests |
| `consensus.py` | unanimity, holdout override, `final_votes`, dissent, convergence report | wired, 9 tests |
| `blackboard.py` | Redis blackboard + in-memory fallback, one interface, read back per round | wired, 14 tests |
| `context.py` | untrusted-data fences for agent output and prior findings | wired |
| `config.py` | strict `blitz.toml` loader (`ConfigError` on unknown key / bad type) | wired, 26 tests |
| `metrics.py` | per-run record → `output/metrics.jsonl`; pricing from config, `None` → `n/a` | wired, 28 tests |
| `embedder.py` | optional MiniLM; deterministic hash stub without it | optional, 12 tests |
| `backends/` | contract, `models.py` resolver, registry + fallback, 6 adapters | wired, 92 tests |
| `mechanisms/` | cascade guard, judge ensemble, selector synthesis | wired (max profile), 67 tests |
| `gmemory/` | the memory store (Tier 1/2/3, FTS5, promotion, eviction, migration) | wired, 66 tests |
| `memory/` | 32-line re-export shim | compat only |
| `mythos/` + `mythos.toml` | planner → executors → verifier mode | wired, 27 tests |
| `prompts/` | loader + `general/` + `crypto/` presets | wired, 24 tests |

## Instruments (tested, not run live)

| File | Purpose | Status |
|---|---|---|
| `bench/slate_v1.toml`, `bench/slate_upgrade.toml` | 30-prompt slate + upgrade mini-slate | no live run |
| `bench/runner.py`, `detectors.py`, `mast_regression.py`, `stats.py` | runner (MAST detectors now run), scoreboard, stats | 160 tests; `bench/runs/` empty |
| `bench/mast_scoreboard.md` | synthetic 14-scenario detector coverage (9/14 at v0.1.1) | synthetic |
| `mythos-bench/` | TOML task specs, runner, adversarial hidden tests | 4 result rows, 2026-05-10, stale |
| `jobs/blitz_upgrade.py`, `jobs/candidates.py` | 10M-token upgrade harness | 84 tests; live run incomplete |
| `scripts/migrate_memory_db.py` | idempotent memory.db migration with backup | tested |
| `scripts/nightly-gmemory-aggregate.sh` | Tier-3 nightly promotion sweep (pure SQLite) | never scheduled |
| `scripts/optimize_prompts.py` | GEPA driver, gated by `[evolve] enabled` | experimental |

## Experimental / contrib

| File | Status |
|---|---|
| `evolve/`, `heterogeneity/` | EXPERIMENTAL, UNWIRED, gated |
| `contrib/managed_agents/` | paid API, opt-in, unimported, call not implemented |

## Configuration

| File | Purpose |
|---|---|
| `blitz.toml` | the only place model names live; every section documented in README "Configuration" |
| `mythos.toml` | Mythos overlay: backend, role tiers, policies |
| `pyproject.toml` | package metadata, version 0.3.0, optional extras |

## Run data

| File | Contents |
|---|---|
| `docs/data/metrics-2026-03_06.jsonl` | 11 archived v0.1–v0.2 run records (was tracked `metrics.jsonl`) |
| `output/` (gitignored) | run documents, `metrics.jsonl`, `scratch/<run_id>/`, `mythos/<run>/` |
| `memory.db` (gitignored) | gmemory SQLite store; legacy files get a `.pre-gmemory.bak` on first open |
| `mythos-bench/results/runs.jsonl`, `comparison.md` | 4 rows, 2026-05-10 |

## Documentation

| File | Purpose |
|---|---|
| `README.md` | current truth |
| `docs/MODELS.md` | model/tier/provider resolution with real command output |
| `docs/CLAIMS_AND_EVIDENCE.md` | one row per README claim → evidence or "not measured" |
| `docs/IMPLEMENTATION_STATUS.md` | wired / tested-not-run / experimental / removed |
| `docs/LIMITATIONS.md` | 20 numbered limitations |
| `docs/ROADMAP.md` | what is next |
| `docs/BIBLIOGRAPHY.md` | arXiv-verified anchors |
| `docs/plans/2026-10-11-fable-remediation.md` | plan of record for v0.3.0 |
| `docs/research/PHASE_0..3_*.md` | 2026-05-09 design deep dives (bannered; dangling paths marked) |
| `docs/RESEARCH_LOG.md`, `METHODOLOGY.md`, `RESEARCH_QUESTIONS.md`, `EVALUATION.md`, `ABLATIONS.md`, `results/RESULTS.md` | 2026-03-14 research docs, bannered as superseded, log extended |
| `BUGS_AND_ITERATIONS.md` | BUG-001..005, ITER-001/002, ITER-MYTHOS-001..003, ITER-AUDIT-1/2 |
| `BLITZ-SWARM.md`, `MYTHOS-SWARM*.md`, `research.md`, `plan.md`, `*.md` literature reviews at root | founding design docs, historical |
