# Claims and Evidence (v0.3.0, 2026-10-11)

One row per claim the README makes. "Evidence" is a `file:line` at commit 36767cd, a test, or a command output; "not measured" means no artifact exists. The previous version of this file (2026-03-14) is superseded.

| # | README claim | Evidence | Strength |
|---|---|---|---|
| 1 | Agents run as subprocesses of local CLIs; no paid token API on a default path | `backends/adapters/cli.py` (subprocess), `ollama.py`, `openai_http.py` (local HTTP); `contrib/managed_agents/` is the only API path and is unimported (`grep -rn contrib --include='*.py'` → 0 non-test hits) | Strong — code |
| 2 | 745 passed / 14 skipped / 0 failed with only `pytest` + `pytest-asyncio`; no model call | `pytest -q` in a clean venv on 2026-10-11 (6.1 s); `pytest --co -q` → 759 collected (667 / 14 at 36767cd before WP5) | Strong — reproduced |
| 3 | No `.py` outside `tests/` contains a model name | `tests/test_no_hardcoded_models.py` (3 tests, passing) | Strong — automated gate |
| 4 | One resolver with a fixed precedence; never a silent default | `backends/models.py:9-21` docstring, `resolve_model` steps 1–6, `tests/test_models.py` (38) | Strong — code + tests |
| 5 | `--list-models` prints the role → tier → model table without a model call | `orchestrator.py:1739-1742` → `format_models_table`; output pasted in `docs/MODELS.md §3` | Strong — reproduced |
| 6 | Mythos planner/verifier = `fable` (Fable 5.1), effort `max`, on claude, with no flags | `mythos.toml [models] backend = "claude"`, `planner_model = "deep"`; `blitz.toml [backend.providers.claude.tiers] deep = "fable"`, `.effort deep = "max"`; `python orchestrator.py "x" --mode mythos --dry-run` output in `docs/MODELS.md §3`; `tests/test_mythos_smoke.py::test_role_tier_resolution_order` | Strong — reproduced |
| 7 | Backend precedence `--backend`/`BLITZ_BACKEND` > `mythos.toml` > `blitz.toml` | `mythos/_invoke.py:55-70` | Strong — code |
| 8 | Six adapters; new CLI / local OpenAI server / Ollama plug in via config only | `backends/registry.py` `_ensure_builtins` (6 `register_backend` calls); `blitz.toml` providers `mycli`, `local`, `ollama`; `tests/test_adapters_v030.py` (27) | Strong — code + tests; `gemini_cli` unverified live |
| 9 | `[backend] fallback` retries once on unavailable-class errors only | `backends/registry.py:128-180` `FallbackBackend.call`; `backends.core.is_unavailable_error` | Strong — code |
| 10 | Round 1 staged (researchers → evaluators → judges); round 2+ one blast | `orchestrator.py:82-85` banner, `:1306-1381`; `run_swarm` docstring `:1166-1174` | Strong — code |
| 11 | Blackboard is the only round store and is read back for every context | `orchestrator.py:576-582`, `:1327`, `:1390`, `:1403`; `tests/test_orchestrator_wp3.py::test_run_swarm_rounds_come_from_the_blackboard_and_calls_stay_in_scratch` | Strong — code + test |
| 12 | Stopping rule = evaluator unanimity + holdout override after round 3; KS-stop secondary behind `halt=true` | `consensus.py` `check_consensus`, `should_override_holdout`; `orchestrator.py:731-745`; `blitz.toml [judge_ensemble] halt = false`; `tests/test_consensus.py` (9) | Strong — code + tests |
| 13 | Dissent section reports unresolved votes only | `consensus.final_votes` | Strong — code |
| 14 | Judge/selector calls fan out concurrently | `orchestrator.py:900-913`, `:1039-1052`; `tests/test_orchestrator_wp3.py::test_judge_ensemble_votes_fan_out_concurrently`, `::test_selector_pairwise_judgements_fan_out_concurrently` | Strong — code + test |
| 15 | Injected outputs and memory are fenced as untrusted data | `context.py:24-32`; `orchestrator.build_context` | Strong — code |
| 16 | Agents run tool-less in a scratch cwd, never the repo root | `orchestrator.py:80` (`CONSENSUS_MAX_TURNS = 1`), `:88-91` (`scratch_cwd`), `:1232` print | Strong — code |
| 17 | `max_agents` enforced; strict TOML loader | `agents.py:303-320`; `config.py:214-260`; `tests/test_config.py` (26), `tests/test_agents_planner.py` (19) | Strong — code + tests |
| 18 | gmemory is the only memory implementation and is wired into both modes | `orchestrator.py:749-790`, `:1268-1279`; `mythos/runner.py:83-150`; `memory/__init__.py` is a re-export shim | Strong — code |
| 19 | Retrieval works with stdlib `sqlite3` + FTS5 only; missing embedder never empties a result or aborts a write | `gmemory/store.py:1-14`; `tests/test_memory_store.py` (31, run without `lancedb`/`sentence-transformers`) | Strong — tests |
| 20 | Legacy `memory.db` is backed up to `*.pre-gmemory.bak` then migrated | `gmemory/store.py:129-144`; `scripts/migrate_memory_db.py`; `tests/test_memory_store.py::test_open_backs_up_legacy_db_once_before_migrating`, `::test_migrate_legacy_db_is_idempotent_and_keeps_rows`, `::test_migrate_script_cli_on_a_copy` | Strong — code + test |
| 21 | N=3 promotion gate and eviction every `cycle_frequency` tasks run on the live path | `gmemory/store.py:211-322` | Strong — code; field effect **not measured** |
| 22 | Memory makes later runs better ("compounding") | — | **Not measured** |
| 23 | Cascade guard filters errored/raw/blocked outputs before the next context | `orchestrator.py:231-258` `_apply_guard` / `_filter_guarded`; `tests/test_cascade_guard.py` (23) | Strong — code + tests |
| 24 | `[guard] mode = "max"` runs LLM claim decomposition; `balanced` is a labelled heuristic | `mechanisms/cascade_guard.py:13-23`, `:333`; `orchestrator.py:1214` | Strong — code; `max` not run live |
| 25 | Mythos verifier demotes low-confidence passes; with `verifier_run_tests = true` (opt-in, off by default: executes model-written code) it also runs shipped tests in a subprocess | `mythos/verifier.py:109-195`; `mythos.toml [policies] verifier_*`; `tests/test_wp4_bench_evolve_guard_verifier.py::test_verifier_mechanical_check_demotes_pass_when_artifact_tests_fail`, `::test_verifier_confidence_gate` | Strong — code + tests |
| 26 | Token ceiling applies always; cost ceiling only when cost known; unknown cost printed `n/a` | `mythos/policies.py:26-52` `CostBudget`; `mythos/runner.py:24-29`; `tests/test_mythos_tokens.py` (8) | Strong — code + tests |
| 27 | Bench runner runs MAST detectors and says when it did not | `bench/runner.py:116-125`, `:424-432`, `:668`; `tests/test_runner_functional.py` (25) | Strong — code + tests; never run live |
| 28 | Exact Student-t for df < 30 without scipy | `bench/stats.py:144-162`; `tests/bench/test_stats.py` (11) | Strong — code + tests |
| 29 | p = 0.0 is treated as significant in the upgrade gate | `jobs/blitz_upgrade.py:1886-1891`; `tests/test_wp4_bench_evolve_guard_verifier.py::test_decision_rule_treats_p_zero_as_significant_and_none_as_not` | Strong — code + test |
| 30 | `evolve/` + `heterogeneity/` are experimental and refuse unless `[evolve] enabled = true` | `evolve/__init__.py:1-47`; `heterogeneity/cli_router.py:174-176`; `tests/test_phase3.py` (36) | Strong — code + tests |
| 31 | `contrib/managed_agents` is opt-in, paid, unimported, network call not implemented | `contrib/managed_agents/README.md`; `adapter.py` `run_task`; `tests/contrib/test_managed_agents.py` (13) | Strong — code |
| 32 | Mythos is cheaper/faster than consensus on artifact tasks | `mythos-bench/results/comparison.md` (4 rows, 2026-05-10, pre-v0.3 code path) | **Weak — stale, n=4, not regenerated** |
| 33 | Any mechanism improves output quality over v0.1 | — | **Not measured** (no live bench run) |
| 34 | MAST detector coverage 9/14 | `bench/mast_scoreboard.md` (synthetic scenarios, `bench/mast_regression.py`) | Moderate — synthetic, not live |
| 35 | Paper-stated numbers (G-Memory +20.89 %, blackboard MAS +13–57 %, cascade defense 0.32 → 0.89, selection 0.81 vs 0.51) | `docs/BIBLIOGRAPHY.md`, titles verified against arXiv 2026-10-11 | Their results, not ours — never replicated here |
| 36 | 10M-token upgrade harness works end-to-end | `jobs/` (84 tests, mocked); live run 2026-06-10 stalled at phase 1/5 | **Not demonstrated live** |
