# Mythos Swarm

> Hierarchical orchestrator-executor mode for blitz-swarm.
> Mythos planner + Sonnet executors + Mythos verifier — for tasks where reasoning *is* the deliverable.

---

## What it is

Mythos Swarm is a peer mode of [BLITZ-SWARM](BLITZ-SWARM.md), in the same repo. While blitz-swarm consensus mode runs N peer researchers + critics in flat parallel and votes on quality, Mythos mode is a **hierarchy**:

```
Mythos Planner (extended-thinking Opus)
    ↓ decomposes the task into N independent sub-specs + global invariants
Sonnet Executors (parallel)
    ↓ each implements one sub-spec, self-checking against acceptance criteria
Mythos Verifier (extended-thinking Opus)
    ↓ gates: pass | needs_work, with per-invariant evidence
   either ACCEPT and emit artifact
   or REPLAN with verifier feedback (up to N rounds)
```

**Why this shape?** From the prior `/blitz-swarm` research on Opus 4.7 + Mythos token-worthiness: extended-thinking models are wasted on parallel breadth but uniquely strong on long-horizon coherence. The right architecture is one deep planner + cheap parallel executors + one deep verifier — not a flat swarm.

**Killer use cases** (where Mythos mode beats consensus mode):
- Legacy codebase modernization (full-repo behavioral equivalence)
- Multi-hop security audit + remediation
- Non-deterministic production debug (race conditions, distributed consensus)
- Compiler / interpreter construction
- Formal verification (Lean 4 / TLA+ / Coq)

---

## Quickstart

```bash
# Dry-run shows the config without making any CLI calls
python mythos_swarm.py "Implement & verify binary search with loop invariants" --dry-run

# Real run (will spend $$ — see cost ceiling below)
python mythos_swarm.py "Implement & verify binary search with loop invariants"

# Custom budget
python mythos_swarm.py "task spec" --cost-ceiling 2.0 --max-replans 2

# Equivalent invocation via the orchestrator directly
python orchestrator.py --mode mythos "task spec"
```

Output lives at `output/mythos/<task_slug>_<timestamp>/`:

```
output/mythos/<slug>_<ts>/
├── plan.md            # Initial planner decomposition
├── plan.json
├── executor_00_<spec_id>.md
├── executor_01_<spec_id>.md
├── ...
├── verification.md    # Verifier gate report
├── verification.json
├── plan_round1.md     # If replan happened
├── verification_round1.md
├── final_artifact.md  # Human-readable assembled deliverable
├── metrics.json       # Cost / tokens / rounds
└── run.json           # Machine-readable run summary
```

---

## Configuration

Two layers. CLI flags override `mythos.toml` overrides defaults.

### `mythos.toml` (repo root)

```toml
[models]
planner_model  = "mythos"   # "mythos" alias resolves to opus + effort=max
executor_model = "sonnet"   # cheap parallelism
verifier_model = "mythos"

[policies]
max_replans       = 3
max_executors     = 8
cost_ceiling_usd  = 5.0
planner_timeout_s = 600
executor_timeout_s = 300
verifier_timeout_s = 600
output_dir        = "./output/mythos"
use_redis         = false
use_memory        = true
```

### Model aliases (`mythos/policies.py::MODEL_ALIASES`)

| Alias    | Model      | Effort | Use case                             |
|----------|------------|--------|--------------------------------------|
| `mythos` | opus       | max    | Planner / verifier deep reasoning    |
| `opus`   | opus       | —      | Plain Opus orchestration             |
| `sonnet` | sonnet     | —      | Workhorse executor                   |
| `haiku`  | haiku      | —      | Cheapest executor                    |

When Anthropic ships a real Mythos model, change one entry in `MODEL_ALIASES` and the whole swarm picks it up.

### CLI flags (`mythos`-mode)

| Flag                | Effect                                                 |
|---------------------|--------------------------------------------------------|
| `--cost-ceiling N`  | Hard $ ceiling per task (overrides toml)               |
| `--max-replans N`   | Replan budget                                          |
| `--dry-run`         | Print config, do not invoke any CLI                    |
| `--no-redis`        | Skip Redis blackboard                                  |

---

## Cost expectations

A single Mythos call (`opus + effort=max`) cost **$0.33 for an empty round-trip** in our probe. Real runs scale with reasoning depth and context size.

Rough heuristic for v1:
- **Planner:** $0.30 – $1.50 per call (1–2 calls per replan round)
- **Executors:** $0.05 – $0.30 each, in parallel; 4–8 typical
- **Verifier:** $0.50 – $2.00 per call (sees full deliverables)
- **One round (no replan):** ~$1 – $4
- **Three rounds (worst case):** ~$3 – $12

The default ceiling is **$5.00**. Override with `--cost-ceiling`. The run aborts gracefully and emits a partial artifact when the ceiling is hit.

---

## Design notes

- **Distinct from consensus mode.** No critic/fact-checker/quality-judge voting. The verifier IS the gate. The planner replans. That's it.
- **Output is an artifact, not a summary.** Mythos produces the deliverable (code, proof, refactor) plus the verification trace. Blitz-swarm consensus mode produces a research summary; Mythos mode produces work.
- **Stateless agents.** Like blitz-swarm, every Mythos role is a stateless `claude -p` subprocess invocation. The runner owns state.
- **Cost telemetry from day 1.** Every phase logs `cost_usd` to `metrics.json`. No black-box spending.
- **Hard cost ceiling.** Defensive default. The Mythos research warned about extended-thinking blow-ups; the runner enforces it.
- **Rule #10 compliance.** All LLM calls go through Joona's existing `claude` CLI subscription. No paid API key path. No exceptions.

---

## Architecture overview

```
mythos/
├── __init__.py        # public surface: run_mythos, MythosResult, load_mythos_config
├── policies.py        # MODEL_ALIASES, CostBudget, MythosConfig, load_mythos_config
├── schemas.py         # PLANNER_SCHEMA, EXECUTOR_SCHEMA, VERIFIER_SCHEMA (JSON)
├── _invoke.py         # Shared `claude -p` wrapper with cost extraction
├── planner.py         # decompose() + replan()
├── executor.py        # run_executor + parallel runner
├── verifier.py        # verify()
├── runner.py          # the orchestration loop (plan→exec→verify→replan)
├── artifact.py        # writes plan/executors/verification/final_artifact to disk
└── prompts/
    ├── planner.md
    ├── executor.md
    └── verifier.md
```

Imports only from blitz-swarm public surfaces (`config`, optional `metrics`, optional `blackboard`). Clean import boundary so it can be extracted to `lovespark-mythos-swarm` later if it earns it.

---

## Future work (v2+)

- **Cross-CLI heterogeneity.** Route executors to `codex` for code, `gemini` for fact-grounding (hooks already exist in `heterogeneity/`).
- **Persona-typed verifiers.** Factual / logical / counterfactual gates run in parallel.
- **Cascade-guard escalation.** If verifier confidence is low, escalate to a second Mythos verifier with even higher effort.
- **Head-to-head benchmark.** Run blitz-swarm consensus mode and Mythos mode on the same task spec; compare cost / correctness / wall clock. Required before declaring Mythos mode "worth it".
- **Lean 4 / formal verification target.** The cleanest Mythos use case once the toolchain is plumbed in.

---

## License

MIT. Same as blitz-swarm.
