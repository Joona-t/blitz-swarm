# Mythos Swarm — Research

> Draft research doc per CLAUDE.md workflow (research → plan → annotate → "implement it all"). **Do not implement yet.**
> Author: Claude (Opus 4.7) | Date: 2026-05-09 | Source-of-truth for the Mythos Swarm design discussion.

---

## 1. What "Mythos" Means In This Context

Carrying forward from the prior `/blitz-swarm` run on Opus 4.7 + Mythos token-worthiness, **Mythos** is most plausibly an **extended-thinking variant of Opus 4.7** — i.e., a model that consumes a large internal reasoning budget (100K+ thinking tokens) before producing visible output. Cost economics 3–5× standard Opus, value proposition: tasks where the *reasoning chain itself* is the deliverable.

Two implications govern the swarm design:

1. **Mythos shines on long-horizon coherence + multi-hop logical inference** (formal verification, legacy modernization, multi-hop security audits, distributed systems debugging, compiler construction).
2. **Mythos is wasteful on parallel breadth-first work** (boilerplate generation, test stub writing, doc summarization, routine refactors). Sonnet beats it on token-per-quality there.

**The architectural punchline from the prior research:** the optimal pattern is **Mythos as orchestrator, Sonnet/Haiku as executors**, with prompt caching mandatory. ~90% of Mythos quality at 30–40% of full-Mythos cost. *(That ratio is researcher estimate, not benchmark — needs validation.)*

---

## 2. What Joona Already Has (Prior Art)

Blitz-Swarm at `~/Claude x LoveSpark/blitz-swarm/` is **not just a research summarizer** — it's a generic parallel-agent platform with:

| Capability | Where | What it gives us for Mythos |
|---|---|---|
| Domain prompt presets | `prompts/<domain>/{role}.md` | A new `prompts/mythos/` preset is a 1-line config flip |
| Per-role model overrides | `agents.py::ROLE_MODEL_OVERRIDES` | Easy to route synth/judge/critic to Mythos, researchers to Sonnet |
| Cross-CLI heterogeneity | `heterogeneity/cli_router.py` + `routing_table.toml` | Mythos could even be a different CLI (e.g. via Claude Code's `--model` flag) |
| Persona-typed critics | `prompts/general/critic_*.md` + `assign_personas` | MAR mechanism (arXiv 2512.20845) — useful for Mythos's deeper critic role |
| Selector-synth, judge-ensemble, cascade-guard | `mechanisms/` | Could be Mythos-evaluated for higher-stakes tasks |
| Evolve loops | `evolve/aflow_search.py`, `gepa_adapter.py`, `meta_loop.py` | Self-improving prompts — high token, Mythos-justified |
| Shared blackboard, consensus, dissent | `blackboard.py`, `consensus.py` | Already battle-tested |
| Metrics + cost tracking | `metrics.py` → `metrics.jsonl` | Critical for proving Mythos ROI |

**Constraint observed:** today the heterogeneity router defaults *researchers* to claude (heaviest workload on the heaviest-by-default agent). Inverting this — researchers on cheap models, orchestration on Mythos — is the Mythos pattern.

---

## 3. Three Plausible Designs

### Design A — **Mythos as a Domain Preset** (additive, smallest change)

A new `prompts/mythos/` preset + a `[swarm] domain = "mythos"` mode + a model-routing override that pushes synthesizer / quality_judge / critic to a `mythos` model alias and keeps researchers on `sonnet`.

```toml
# blitz.toml
[swarm]
domain = "mythos"

[mythos.models]
researcher = "sonnet"
critic = "opus"             # or "claude-opus-4-7-thinking" once available
fact_checker = "sonnet"
quality_judge = "opus"
synthesizer = "opus-thinking" # the actual Mythos call
```

- **Pros:** Zero new code paths. Reuses everything: blackboard, consensus, dissent, metrics. `--domain mythos` is the only orchestrator change. Maximum reuse.
- **Cons:** Doesn't change the *topology* — still flat consensus, still researchers do the heavy work in parallel. Mythos's strength (long-horizon hierarchical orchestration) is underused.
- **Best for:** Validating "is Mythos worth it on our existing tasks" cheaply.
- **Effort:** ~1 day. Mostly prompt-writing + config.

### Design B — **Hierarchical Orchestrator-Executor Swarm** (medium change, faithful to the research)

A new top-level mode that implements the pattern from §4 of the prior research output: **one Mythos planner** decomposes the task into specs, **N Sonnet executors** run in parallel per sub-spec, **one Mythos verifier** gates outputs, loops until the planner accepts.

```
Mythos planner    →  decomposes task into N sub-specs
  ↓
Sonnet executors  →  parallel implementation per sub-spec (N agents)
  ↓
Mythos verifier   →  cross-checks all outputs against original spec + invariants
  ↓
Replan if needed; otherwise emit
```

- **Pros:** Faithful to the strongest finding in the prior research. Suits the killer use cases (legacy modernization, multi-hop security audit, distributed debug, compiler construction) — *not* generic research tasks. Different topology = different product.
- **Cons:** New code paths (planner / executor / verifier roles aren't currently first-class). Different consensus model — no "ready/needs_work" voting; instead, planner-driven re-spec loops. Output format different too (artifact + verification trace, not a markdown summary).
- **Best for:** The actual Mythos-worthy use cases the prior swarm identified.
- **Effort:** ~1 week. Real architecture work.

### Design C — **Cost-Aware Sub-Swarm Inside Blitz** (smallest delta, biggest leverage)

A `mechanisms/mythos_subswarm.py` that any role can invoke for a single deep task. The orchestrator calls it whenever a single agent hits a hard subtask (e.g., a researcher's confidence drops below 0.5, or the critic flags a multi-hop reasoning gap). The sub-swarm spends 10× the tokens to produce one high-quality answer, then returns it to the parent swarm.

- **Pros:** Surgical. Mythos consumed only where the prior research says it's actually worth it — not on every turn. Great economics.
- **Cons:** Adds dynamic dispatch logic (when to invoke). Hard to predict cost a priori. Requires an "escalation policy" — itself a small ML problem.
- **Best for:** Living with current swarm but giving it a Mythos escape hatch.
- **Effort:** ~3 days for v1, but the escalation policy will iterate forever.

---

## 4. Recommendation (open to override)

**Build Design B as a new top-level mode `mythos-swarm`, but inside the existing `blitz-swarm` repo** — not a new project. Reasons:

1. The prior research strongly suggests the *topology* is the differentiator, not the model choice. Design A re-skins; Design B does the actual thing.
2. Reusing blitz-swarm's blackboard / metrics / heterogeneity / config infrastructure means we ship faster and keep one codebase to maintain. Forking would split cognitive load and the existing investment in mechanisms/, evolve/, gmemory/.
3. Inside the repo, Mythos becomes a *peer mode* alongside the consensus mode — selectable via CLI flag `--mode mythos` or via a new `mythos.toml` overlay. The user picks based on task type.
4. If Mythos-mode proves out, **then** consider extracting to its own repo (`lovespark-mythos-swarm`) for public release. Premature extraction loses the option.

**Folder location proposal (CONFIRM BEFORE IMPLEMENTATION):**
- `~/Claude x LoveSpark/blitz-swarm/mythos/` — new package inside the repo
- Entrypoint: `python orchestrator.py --mode mythos "task spec"` *(extend existing CLI, don't fork)*
- Output: `output/mythos/{task}_{timestamp}/{plan.md, executor_outputs/, verification.md, final_artifact}`
- Optional: keep a thin `mythos_swarm.py` shim at repo root for the muscle-memory `python mythos_swarm.py "task"` invocation.

---

## 5. Killer-Use-Case Targets (the swarm has to be good at these to justify itself)

From the prior `/blitz-swarm` run, ranked by combined intensity × worthiness:

| # | Use case | Why Mythos | Mythos role |
|---|---|---|---|
| 1 | Legacy codebase modernization (full-repo) | Behavioral equivalence across 800K–3M tokens | Planner: holds the global migration plan; verifier: behavioral-equivalence check |
| 2 | Multi-hop security audit + auto-remediation | 5+ step taint chains | Planner: maps attack surfaces; verifier: traces remediations |
| 3 | Non-deterministic prod debug (race / consensus) | Holds full logs + traces + repo + hypotheses | Planner: hypothesis tree; verifier: invariant re-check |
| 4 | Compiler / interpreter construction | NP-hard graph-coloring, HM unification | Planner: type-system design; verifier: type-check the generated IR |
| 5 | Formal verification (Lean 4 / TLA+ / Coq) | Proof search = canonical extended-thinking task | Planner: lemma decomposition; verifier: tactic correctness |

**Test target for v1:** pick ONE of these (likely #5 — formal verification of a tiny Lean theorem, since it's the cleanest pass/fail signal and bounded in scope). Run blitz-swarm consensus mode AND mythos mode on the same task. Compare cost, quality, success.

---

## 6. Hard Questions That Aren't Resolved

These need Joona's call before plan.md:

| Q | Options | My lean |
|---|---|---|
| Q1 | Is "Mythos" the actual model name we route to, or a placeholder until Anthropic ships one? | "Mythos" as a config alias that maps to whatever real model we have (`opus`, `claude-opus-4-7`, future `claude-mythos-*`). Decouples our code from naming changes. | Config alias — maximum future-proofing |
| Q2 | Should mythos-mode be inside blitz-swarm or a sibling project? | (a) Inside blitz-swarm/ (b) `mythos-swarm/` sibling (c) New private repo `lovespark-mythos-swarm` | (a) — extract later if it earns it |
| Q3 | First test target use case? | (a) Formal verification of a Lean 4 theorem (b) Migrate one small Python 2 file to 3 (c) Audit one LoveSpark extension end-to-end for vulns | (a) — cleanest signal, smallest scope |
| Q4 | Cost ceiling per task? | Hard cap (`$X/task`) or soft warning? | Hard cap at $5/task for v1, override flag for power use |
| Q5 | Memory: does mythos-mode share the G-Memory blackboard with consensus mode, or have its own tier? | Shared memory means insights compound across modes (good); separate means cleaner experimentation | Shared, with a `mode=mythos` tag on every insight for filtering |
| Q6 | Verifier failure → what? | (a) Loop back to planner with verifier notes (b) Fail loudly to user (c) Fall back to consensus mode (graceful degradation) | (a) with max 3 replan rounds, then (b) |
| Q7 | Is "Mythos" available via `claude -p --model <id>` today, or do we need a different CLI/SDK? | Empirical — needs a quick CLI probe before plan.md | Probe in plan phase — drives feasibility |

---

## 7. Risks

- **R1 — Mythos isn't a thing yet.** If `claude -p --model mythos` (or whatever the real ID is) errors, the whole project blocks until Anthropic ships. Mitigation: build with config alias; default the alias to `opus` so the swarm runs today and "becomes Mythos" the day a real model lands.
- **R2 — Cost blow-up.** Extended-thinking models can spend 100K+ tokens per call. Three planner-replan cycles × two verifier passes × ten executors = a $50 task on a small spec. Mitigation: hard cost ceiling, mandatory prompt caching, telemetry from day 1.
- **R3 — No measurable win over flat consensus.** Mythos mode might just be slower + pricier blitz-swarm with the same output. Mitigation: §5 head-to-head benchmark is mandatory before we declare victory.
- **R4 — Rule #10 ("no paid LLM API, ever").** Mythos via `claude` CLI = uses Joona's existing Claude Max subscription = OK. **Anything that requires direct API billing = STOP.** All Mythos invocations must be CLI-driven. Documented for future-Claude.
- **R5 — Hidden coupling to blitz-swarm internals.** If we extract later, undocumented coupling will hurt. Mitigation: enforce a clean import boundary — `mythos/` only imports from blitz-swarm's *public* surfaces (config, agents, blackboard, metrics).

---

## 8. What v1 Looks Like (concrete preview)

```
~/Claude x LoveSpark/blitz-swarm/
  mythos/
    __init__.py
    planner.py          # Mythos planner: decompose spec into sub-specs
    executor.py         # Sonnet executor: implement one sub-spec
    verifier.py         # Mythos verifier: check sub-outputs against spec+invariants
    artifact.py         # Aggregator: assemble executor outputs into final artifact
    policies.py         # Cost ceilings, replan budget, escalation rules
    prompts/
      planner.md
      executor.md
      verifier.md
  prompts/mythos/       # Optional: also expose as a domain preset for Design A users
    ...
  mythos_swarm.py       # Thin entrypoint shim → python orchestrator.py --mode mythos
  MYTHOS-SWARM.md       # User-facing doc once we ship
```

CLI:
```
python orchestrator.py --mode mythos "Prove that Lean's List.length_append is correct"
python orchestrator.py --mode mythos "Migrate scripts/legacy.py from Python 2 to 3"
python orchestrator.py --mode mythos "Audit Extensions/lovespark-focus for OWASP Top 10"
```

Output:
```
output/mythos/<task>_<ts>/
  plan.md                  # Mythos planner output
  executor_00_output.md
  executor_01_output.md
  ...
  verification.md          # Mythos verifier final report
  final_artifact.{ext}     # The actual deliverable
  metrics.json             # Cost, tokens, rounds, quality
```

Success metric for v1: **on the formal-verification test (§5), beat blitz-swarm consensus mode on either correctness OR cost** (ideally both). If neither, the architecture is wrong and we either fix or shelve.

---

## 9. Next Steps (after Joona's annotations)

1. Joona reads this doc, marks it up inline (especially §6 hard questions, §4 folder location, §5 test target).
2. Claude reads annotations, produces `MYTHOS-SWARM-PLAN.md` with a granular todo checklist.
3. Joona approves plan.
4. Joona says "implement it all" → Claude builds it.
5. v1 lands; head-to-head benchmark runs; we either ship or rethink.

---

*End of research.md. **Do not implement yet.** Awaiting Joona's annotations.*
