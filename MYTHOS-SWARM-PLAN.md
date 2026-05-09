# Mythos Swarm — Plan

> Granular implementation plan. Decisions on §6 hard questions filled in. Built per CLAUDE.md workflow.
> Companion: `MYTHOS-SWARM-RESEARCH.md`. Author: Claude (Opus 4.7) | Date: 2026-05-09.

---

## Decisions on §6 Hard Questions (autonomous defaults — Joona to override if wrong)

| Q | Decision | Why |
|---|---|---|
| Q1 — model alias | `mythos` is a string alias resolved via `mythos/policies.py::resolve_model()`. Default mapping: `"mythos" → ("opus", effort="max")`. Future-proof: change the dict when a real Mythos lands. | Decouples code from naming churn. CLI probe confirms `--model opus` works and routes to `claude-opus-4-7`. |
| Q2 — folder | `~/Claude x LoveSpark/blitz-swarm/mythos/` (in-repo, peer mode) | Joona explicitly said "make a mythos folder" — done. Extract to its own repo later if it earns it. |
| Q3 — first test | A small algorithmic correctness task — **"implement & verify a binary search with explicit loop invariants"** — runnable as Python, self-checkable, no Lean toolchain needed. Plus a Python 2→3 migration toy as second example. | Lean 4 toolchain isn't installed; binary search gives the same "proof discipline" signal at zero setup cost. |
| Q4 — cost ceiling | Hard cap **$5.00 per task**, configurable via `--cost-ceiling` flag and `mythos.toml [policies] cost_ceiling_usd`. Task aborts and reports partial output if exceeded. | Defensive default. Single Mythos turn already cost $0.33 in probe. 15× headroom feels right for v1. |
| Q5 — memory | Shared G-Memory blackboard with mode-tagged inserts (`mode="mythos"` on every utterance/insight). | Insights compound across modes. Filter at query time. |
| Q6 — verifier failure | Loop back to planner with verifier feedback for **up to `max_replans=3`** rounds. After that, fail loudly with the partial artifact + verification trace saved. | Matches the prior research's recommended replan loop without unbounded cost. |
| Q7 — CLI feasibility | Confirmed: `claude -p --model opus --effort max` is the Mythos invocation. Probed, returned cleanly. | Done. |

---

## Architecture

```
Task spec ────► Mythos Planner (opus + effort=max)
                  │  decomposes into N sub-specs + invariants
                  ▼
                Sub-spec list ────────►  Sonnet Executors (parallel, asyncio.gather)
                                           │  one per sub-spec
                                           ▼
                                         Executor outputs
                                           │
                                           ▼
                                    Mythos Verifier (opus + effort=max)
                                           │  checks outputs against spec + invariants
                                           ▼
                                ┌──────────┴──────────┐
                                │                     │
                          PASS  ▼                NEEDS_WORK ▼
                          Artifact assembly      Replan with verifier notes
                          (final deliverable)    (back to Planner, max 3 rounds)
```

Re-uses blitz-swarm primitives:
- `heterogeneity.cli_router.CLIRouter` for the actual subprocess invocations (all flow through claude CLI)
- `metrics.MetricsCollector` for cost/token tracking
- `blackboard.Blackboard` (optional, --no-redis works) for shared state

Distinct from blitz-swarm consensus mode:
- No critic/fact-checker/quality-judge roles. Verifier IS the gate.
- No "ready/needs_work" voting across heterogeneous evaluators. Verifier returns structured `pass | needs_work` with a list of failed invariants and required fixes.
- Planner re-spec loop, not blast-iterate.
- Output is an **artifact** + verification trace, not a markdown summary.

---

## File Layout (final)

```
~/Claude x LoveSpark/blitz-swarm/
├── mythos/
│   ├── __init__.py            # Public surface: run_mythos, MythosResult
│   ├── policies.py            # Model aliases, cost ceiling, replan budget, timeouts
│   ├── planner.py             # decompose() + replan()
│   ├── executor.py            # run_executor() — single sub-spec implementation
│   ├── verifier.py            # verify() — spec/invariant check
│   ├── artifact.py            # assemble() — bundle final deliverable to disk
│   ├── runner.py              # orchestration: planner→executors→verifier→loop
│   ├── schemas.py             # JSON schemas for planner/executor/verifier outputs
│   └── prompts/
│       ├── planner.md
│       ├── executor.md
│       └── verifier.md
├── orchestrator.py            # ADD: --mode mythos flag dispatching to mythos.runner
├── mythos.toml                # NEW: mythos-specific config overlay
├── mythos_swarm.py            # Thin shim: python mythos_swarm.py "task"
├── MYTHOS-SWARM.md            # User-facing doc
├── MYTHOS-SWARM-RESEARCH.md   # (already exists)
├── MYTHOS-SWARM-PLAN.md       # (this file)
└── tests/
    └── test_mythos_smoke.py   # Import + dry-run + policy-resolution tests (no CLI calls)
```

---

## Output Format

```
output/mythos/<task_slug>_<timestamp>/
├── plan.md                    # Mythos planner output (sub-specs + invariants)
├── executor_00_output.md
├── executor_01_output.md
├── ...
├── verification.md            # Verifier final report (pass/fail per invariant)
├── final_artifact.{md|py|...} # The actual deliverable
└── metrics.json               # Cost, tokens, replans, wall clock
```

---

## Granular TODO Checklist

Implement in this order. Mark each ✅ as it lands.

### Phase 1 — Skeleton + policies
- [ ] **T1.1** Create `mythos/__init__.py` with public exports stub
- [ ] **T1.2** Create `mythos/policies.py` with `MODEL_ALIASES` dict, `resolve_model()`, `CostBudget` class, `MythosConfig` dataclass
- [ ] **T1.3** Create `mythos.toml` at repo root with `[policies]`, `[models]` sections
- [ ] **T1.4** Wire `mythos.toml` loader into `policies.py::load_mythos_config()`

### Phase 2 — Schemas + prompts
- [ ] **T2.1** Create `mythos/schemas.py` with `PLANNER_SCHEMA`, `EXECUTOR_SCHEMA`, `VERIFIER_SCHEMA` (JSON Schema dicts)
- [ ] **T2.2** Write `mythos/prompts/planner.md` — decompose into sub-specs + invariants
- [ ] **T2.3** Write `mythos/prompts/executor.md` — implement one sub-spec faithfully
- [ ] **T2.4** Write `mythos/prompts/verifier.md` — gate check, list failed invariants

### Phase 3 — Core agents
- [ ] **T3.1** `mythos/planner.py::decompose(task, config) -> Plan`
- [ ] **T3.2** `mythos/planner.py::replan(task, prev_plan, executor_outputs, verification, config) -> Plan`
- [ ] **T3.3** `mythos/executor.py::run_executor(sub_spec, config) -> ExecutorOutput`
- [ ] **T3.4** `mythos/verifier.py::verify(task, plan, executor_outputs, config) -> Verification`

### Phase 4 — Runner (the loop)
- [ ] **T4.1** `mythos/runner.py::run_mythos(task, config) -> MythosResult` — full plan→exec→verify→replan loop with cost tracking
- [ ] **T4.2** `mythos/artifact.py::assemble(result, output_dir) -> Path` — write all artifacts to disk
- [ ] **T4.3** Wire `MetricsCollector` for cost/token telemetry per phase
- [ ] **T4.4** Hard cost-ceiling enforcement; abort gracefully with partial artifact

### Phase 5 — CLI integration
- [ ] **T5.1** Add `--mode mythos` flag to `orchestrator.py` argparser
- [ ] **T5.2** Add `--max-replans`, `--cost-ceiling` flags (mythos-mode only)
- [ ] **T5.3** Dispatch logic: `if args.mode == "mythos": asyncio.run(mythos.runner.run_mythos(...))`
- [ ] **T5.4** Default `--mode consensus` keeps current behavior intact (zero regression)

### Phase 6 — Shim + docs
- [ ] **T6.1** Create `mythos_swarm.py` thin shim → invokes orchestrator with `--mode mythos`
- [ ] **T6.2** Write `MYTHOS-SWARM.md` user doc (quickstart, examples, design notes, cost expectations)

### Phase 7 — Tests + smoke
- [ ] **T7.1** Create `tests/test_mythos_smoke.py`: imports clean, policy resolution works, dry-run prints expected agent layout
- [ ] **T7.2** Run smoke test
- [ ] **T7.3** Run `python orchestrator.py --mode mythos --help` to confirm CLI surface

### Phase 8 — Trail (rule #11)
- [ ] **T8.1** Update `BUGS_AND_ITERATIONS.md` with `ITER-MYTHOS-001` entry: motivation, design choices, deferred items
- [ ] **T8.2** Stage + commit (no Co-Authored-By per rule #6) with reasoning in commit message
- [ ] **T8.3** Push to remote (`git push origin main`)

### Deferred to v2 (NOT in v1 scope)
- Live full-task run with real cost (the swarm itself shouldn't run a Mythos task during build — too expensive). v2: run on the binary-search test target, capture metrics, write a head-to-head vs blitz-swarm consensus mode as a separate doc.
- Lean 4 / formal-verification target.
- Domain-A preset (Mythos-as-domain-preset) — could land later if Joona wants the cheap variant.
- Cross-CLI heterogeneity for Mythos (`codex` for executor on code tasks, `gemini` for fact-grounding). Hooks are there; activation is v2.
- Persona-typed verifiers (factual / logical / counterfactual gating). v2.
- Cascade-guard escalation policy. v2.

---

## Risk Recap (from research §7)

- **R2 cost blow-up:** mitigated by hard ceiling + per-phase cost telemetry written to `metrics.json` for every run, plus cache-aware prompts.
- **R4 rule #10:** Mythos invokes only via `claude` CLI. No `ANTHROPIC_API_KEY` paths. Audit checklist before commit.
- **R5 import boundary:** `mythos/` only imports from blitz-swarm public surfaces (`config`, `metrics`, `heterogeneity.cli_router`, `blackboard`). No reaching into `agents.ROLE_PROMPTS` or other internals.

---

## Convention Compliance Audit (CLAUDE.md hard rules)

- ✅ Rule #1 — plan.md exists (this file) before any code lands.
- ✅ Rule #2 — folder location confirmed by Joona ("make a mythos folder").
- N/A Rule #3 — no UI in this build.
- N/A Rule #4 — no UI.
- N/A Rule #5 — not a Chrome extension.
- ✅ Rule #6 — commit will not include Co-Authored-By.
- ✅ Rule #7 — research.md exists (`MYTHOS-SWARM-RESEARCH.md`).
- ✅ Rule #8 — Joona said "execute everything" → implementation green light.
- N/A Rule #9 — not an extension; no manifest version to bump.
- ✅ Rule #10 — all LLM calls via Joona's existing `claude` CLI. No paid API key path.
- ✅ Rule #11 — BUGS_AND_ITERATIONS entry will land same session (T8.1).

---

*End of plan.md. Beginning implementation now.*
