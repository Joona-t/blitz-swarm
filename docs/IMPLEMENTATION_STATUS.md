# Implementation Status (v0.3.0, 2026-10-11)

Regenerated from the v0.3.0 tree (36767cd + WP5 tests). "Wired" means the orchestrator or Mythos runner calls it on a default run; "tested" is the per-file count from `pytest --co -q`. The 2026-03-14 version of this file is superseded.

## Wired and tested (runs on `python orchestrator.py "topic"`)

| Component | Files | Tests | Notes |
|---|---|---|---|
| Agent planning | `agents.py` | 19 (`test_agents_planner`) | LLM plan on the `planner` tier; heuristic fallback logs loudly; `max_agents` enforced |
| Staged round 1 + parallel rounds 2+ | `orchestrator.py` `run_swarm` | 17 + 6 + 19 | `asyncio.to_thread` + `gather`; staging banner printed |
| Blackboard (Redis + in-memory, read back) | `blackboard.py` | 14 | same interface; `read_round_outputs` feeds every context |
| Consensus: unanimity, holdout override, unresolved dissent, convergence report | `consensus.py` | 9 | |
| Secondary KS-stop halt | `orchestrator._judge_ks_halt` | in wp3 tests | only with `[judge_ensemble] halt = true` |
| Untrusted-data fences | `context.py` | in wp3 tests | agent output + prior findings |
| Strict config loader | `config.py` | 26 | unknown key / bad type / undefined provider raise |
| Model resolution (role → tier → model) | `backends/models.py` | 38 + 3 (grep guard) | |
| Embedder (optional MiniLM, hash stub fallback) | `embedder.py` | 12 | |
| Adapters: claude_cli, codex_cli, gemini_cli, ollama_http, openai_http, generic_cli | `backends/adapters/` | 27 + 12 + 12 | strict Codex schema; `--effort`; fallback chain |
| Cascade guard (heuristic screen by default; LLM hooks on `max`) | `mechanisms/cascade_guard.py` | 23 | |
| Judge ensemble (N=3, KS-stop recorded) | `mechanisms/judge_ensemble.py` | 23 | enabled in the `max` profile |
| Selector synthesis (Bradley-Terry pairwise) | `mechanisms/selector_synth.py` | 21 | enabled in the `max` profile |
| Prompt presets + MAR personas | `prompts/` | 24 | `persona_critics = false` by default |
| G-Memory store: Tier 1/2/3, FTS5 retrieval, promotion gate, eviction, migration | `gmemory/` | 31 + 35 | stdlib only; vectors/LLM hooks optional |
| Metrics record | `metrics.py` | 28 | `output/metrics.jsonl`; cost `None` → `n/a` |
| Mythos mode: planner → executors → verifier, replans, token ceiling, subprocess test gate, memory | `mythos/` | 19 + 8 | default backend claude, deep tier = `fable` |

## Implemented and tested, not run live

| Component | Files | Tests | Status |
|---|---|---|---|
| Bench slate + runner + MAST detectors + stats | `bench/` | 89 + 46 + 25 | no live run; `bench/runs/` empty |
| mythos-bench harness | `mythos-bench/` | smoke via `test_mythos_smoke` | 4 result rows from 2026-05-10, stale |
| 10M-token upgrade job | `jobs/` | 50 + 34 | live run 2026-06-10 stalled at phase 1/5 |
| Migration + nightly aggregation scripts | `scripts/migrate_memory_db.py`, `scripts/nightly-gmemory-aggregate.sh` | migration tested | nightly script never scheduled |

## Experimental, gated, unwired

| Component | Files | Tests | Gate |
|---|---|---|---|
| GEPA adapter + driver | `evolve/gepa_adapter.py`, `scripts/optimize_prompts.py` | in `test_phase3` (36) | `[evolve] enabled = true` + `gepa` package |
| AFlow MCTS | `evolve/aflow_search.py` | " | `[evolve] enabled` |
| Meta-loop config patches (L2) | `evolve/meta_loop.py` | " | `[evolve] enabled`; L3 human audit |
| CLI router across providers | `heterogeneity/cli_router.py`, `routing_table.toml` | " | `[evolve] enabled` |

## Contrib (paid API, opt-in)

| Component | Files | Tests | Status |
|---|---|---|---|
| Anthropic Managed Agents adapter | `contrib/managed_agents/adapter.py` | 13 | scaffold; `run_task` raises `NotImplementedError` with a client |

## Removed in v0.3.0

- `memory/reader.py`, `memory/writer.py`, `memory/schema.sql` logic (Redis single-writer daemon, LanceDB L2 read) — `memory/` is a 32-line re-export shim.
- `mythos/policies.MODEL_ALIASES`, `SwarmConfig.default_model`, `agents.ROLE_MODEL_OVERRIDES`, every literal model name in `.py` files.
- Tracked `metrics.jsonl` (archived to `docs/data/`).
- The fake `completed` path in the managed-agents adapter.

## Not implemented (never was)

- Web search / tool use for agents (agents are tool-less by design in consensus mode).
- CRDT conflict resolution, arbiter agents, temporal validity windows, delta compression (listed in `BLITZ-SWARM.md` 2026-03; never started).
- Cross-provider judge ensembles.
- NEO / iLTN neuro-symbolic reasoning (deferred in v0.2, still deferred).
