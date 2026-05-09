# Phase 3 Recursive Self-Improvement — Implementation Deep Dive

**Status:** Frozen research output from background-task agent run 2026-05-09.
**Source:** Phase 3 GEPA + Aflow + meta-loop deep-dive subagent.
**Consumed by:** `evolve/`, `heterogeneity/`, `managed_agents/`, `scripts/optimize_prompts.py` implementation.

---

## 0. Frame and Recursion Map

The recursion ladder is hard-capped at three live levels plus a frozen meta layer:

```
L0  base swarm (orchestrator, agents, judge)            — runs every research task
L1  prompt + architecture evolution (GEPA, AFlow)       — runs nightly on bench slate
L2  evolution-strategy evolution (meta_loop)            — runs weekly, mutates L1 hyperparams
L3  human audit + freeze                                — Joona reviews L2 changes
```

**Recursion graph:**

```
                     ┌──────── L3 audit ────────┐
                     │           ↑              │
                     v           │              │
   ┌── L2 meta_loop ──┐ ←── meta-insights (insight graph: meta:*) ──┘
   │      │     ↑     │
   │ proposes  bench_score deltas
   │ config   │
   v         v
┌──L1a GEPA──┐  ┌──L1b AFlow──┐
│ prompts    │  │ architecture │
└──┬─────────┘  └──┬───────────┘
   │               │
   v               v
   ┌── L0 swarm (Phase 0..3) ──┐
   │  agents, orchestrator,    │
   │  blackboard, judge,       │
   │  cli_router               │
   └──┬───────────────────────┘
      │
      v
   bench.score(config, slate) → {dim: float, ...}
```

### Loop budgets (hard caps from `blitz.toml [evolve]`)

| Level | Frequency | Wall-clock cap | $$ cap (Managed-Agents) | Token cap (CLI) |
|-------|-----------|----------------|-------------------------|----------------|
| L0    | per task  | 300 s          | $0 (CLI)                | unbounded      |
| L1a   | nightly   | 4 h            | $40                     | "soft 200M"    |
| L1b   | nightly   | 4 h            | $40                     | "soft 200M"    |
| L2    | weekly    | 12 h           | $200                    | "soft 1B"      |
| L3    | manual    | n/a            | n/a                     | n/a            |

"Token unbounded" is honored within a level's cap; caps bumpable from `blitz.toml`.

---

## 1. Component 1 — GEPA Prompt Evolution

### 1.1 Mechanism summary

GEPA (Genetic-Pareto, arXiv 2507.19457, ICLR 2026 oral) replaces gradient-based RL with reflective natural-language mutation of prompt artifacts. Beats GRPO by 6-20% with up to 35× fewer rollouts; beats MIPROv2 by 10%+.

Two ideas:

1. **Reflective mutation.** Reflection LM reads the trace (input → rollout → score → judge feedback) and rewrites the prompt module. Guided rewriting using "actionable side information" (ASI).
2. **Pareto frontier preservation.** Keep candidates that win on at least one bench instance. Sampling from frontier prevents collapse.

Library at github.com/gepa-ai/gepa: `gepa.optimize(seed_candidate, trainset, valset, adapter, ...)` for compound-AI; `gepa.optimize_anything(seed_candidate, evaluator, ...)` for any text artifact.

### 1.2 File layout

```
blitz-swarm/
  scripts/optimize_prompts.py              # CLI entrypoint
  evolve/
    __init__.py
    gepa_adapter.py                        # GEPAAdapter -> swarm
    bench_evaluator.py                     # wraps bench.score for GEPA
  prompts/general/                         # *.md role prompts (existing)
  evolve/runs/<timestamp>/                 # generation logs, candidates, pareto
```

### 1.3 Concrete API

```python
@dataclass
class SwarmTask:
    task_id: str
    topic: str
    domain: str
    expected_traits: dict
    judge_rubric: dict

@dataclass
class SwarmTrajectory:
    task: SwarmTask
    role_outputs: dict[str, str]
    judge_breakdown: dict[str, float]
    final_score: float
    failure_modes: list[str]

class BlitzGEPAAdapter(GEPAAdapter[SwarmTask, SwarmTrajectory, str]):
    def __init__(self, swarm_runner, judge_ensemble, domain_filter=None): ...

    def evaluate(self, candidate: dict[str, str], batch: list[SwarmTask],
                 capture_traces: bool = True) -> EvaluationBatch: ...

    def make_reflective_dataset(self, candidate, eval_batch,
                                 components_to_update: list[str]) -> dict[str, list[dict]]: ...
```

```python
# scripts/optimize_prompts.py
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate", default="bench/slate.jsonl")
    ap.add_argument("--max-metric-calls", type=int, default=1500)
    ap.add_argument("--reflection-lm", default="claude-cli:opus")
    ap.add_argument("--components", nargs="+",
                    default=["researcher", "critic_factual", "critic_logical", "synthesizer"])
    ap.add_argument("--domain-filter", default=None)
    ap.add_argument("--out", default="evolve/runs/")
    args = ap.parse_args()

    seed = {comp: Path(f"prompts/general/{comp}.md").read_text() for comp in args.components}
    train, val = load_slate(args.slate, split=(0.7, 0.3), seed=42)

    adapter = BlitzGEPAAdapter(SwarmRunner(), JudgeEnsemble(n=3), args.domain_filter)
    result = gepa.optimize(
        seed_candidate=seed,
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=args.reflection_lm,
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        skip_perfect_score=True,
        max_metric_calls=args.max_metric_calls,
        use_merge=True,
        max_merge_invocations=8,
        track_best_outputs=True,
        run_dir=f"{args.out}/{int(time.time())}",
        seed=42,
    )
    write_pareto_for_human_review(result)
```

### 1.4 Integration contract

What Phase 0/1 must expose:
- `bench.load_slate(path, split, seed) -> (list[SwarmTask], list[SwarmTask])`
- `agents.SwarmRunner.run_with_prompts(task, prompt_dict) -> SwarmTrajectory` — runner accepts dict overriding role prompts at runtime (no monkey-patching files).
- `consensus.JudgeEnsemble.score(traj, rubric) -> float` and `JudgeEnsemble.breakdown(traj) -> dict[str, float]`.
- `JudgeEnsemble.failure_modes(traj) -> list[str]` — extracts named failure tags. **Single most load-bearing piece for GEPA quality.**

### 1.5 Eval design

- **Primary:** mean weighted quality on held-out val slate.
- **Secondary:** Pareto-frontier coverage.
- **Tertiary:** intra-domain robustness (variance not increased).
- **Promotion threshold:** any frontier candidate beats seed by ≥ 0.4 aggregate AND no domain regresses by > 0.3 → flagged.
- **Stopping rule:** no improvement on val for 200 metric calls OR `max_metric_calls` OR `max_reflection_cost`.

### 1.6 Failure modes + safety

| Failure | Detection | Mitigation |
|---|---|---|
| Prompt collapses to gibberish exploiting judge | Judge ensemble disagreement spike (`std(judges) > 1.5`) | Reject; log `meta:judge-exploit` |
| Reflection LM injects unsafe instructions | Static lint against deny-list | Strip and re-mutate; halt after 3 reverts |
| Mode collapse (frontier shrinks) | Frontier diversity < 0.3 of initial | Force restart from random ancestor |
| Pareto pollution from over-fit | Train-val gap > 1.5 points | Drop before promotion |
| API/CLI cost overrun | `max_reflection_cost` + `max_metric_calls` | Hard stop |
| Reflection LM unavailable | Subprocess timeout > 60s | Fallback to local sonnet; `meta:reflection-fallback` |

Rollback: every candidate's hash + parent + diff logged. `cp parent_prompt.md prompts/general/<role>.md`.

### 1.7 Test plan

```python
def test_adapter_evaluate_returns_one_score_per_task(): ...
def test_adapter_reflective_dataset_filters_to_components(): ...
def test_seed_candidate_loads_all_roles(): ...
def test_pareto_frontier_grows_over_iterations(): ...
def test_max_metric_calls_terminates_run(): ...
def test_judge_disagreement_rejects_candidate(): ...
def test_resume_from_run_dir_continues_iteration(): ...
def test_domain_filter_restricts_slate_subset(): ...
```

---

## 2. Component 2 — AFlow Architectural Search

### 2.1 Mechanism summary

AFlow (arXiv 2410.10762, ICLR 2025 oral, FoundationAgents/AFlow) frames workflow optimization as MCTS over code-represented graphs. Each MCTS node is a complete workflow. Six built-in operators: Generate, Format, Review, Revise, Ensemble, Test, Programmer. **5.7% average improvement over baselines, smaller models match GPT-4o-quality at 4.55% cost.**

A2Flow (arXiv 2511.20693, AAAI 2026) extends with self-adaptive abstraction operators auto-extracted from expert traces. **v0.3 stretch.**

MCTS loop:
1. **Selection** — soft-mixed-probability over visited nodes (UCB1 with score-weighted prior).
2. **Expansion** — LLM proposes code mutation of workflow.
3. **Simulation** — execute candidate against bench slate.
4. **Backpropagation** — propagate score up tree, update visit counts.

### 2.2 File layout

```
blitz-swarm/evolve/
  aflow_search.py
  aflow/
    __init__.py
    workflow_graph.py      # SwarmGraph dataclass, code IR
    operators.py           # AddResearcher, RemoveCritic, AddDebate, SwapSynth, ...
    mcts.py                # MCTS node, selection, backprop
    mutator.py             # LLM-driven mutation proposer
    executor.py            # Compile graph -> orchestrator config -> run -> score
  evolve/runs_aflow/<timestamp>/tree.json
```

### 2.3 Concrete API

```python
class RoleKind(Enum):
    RESEARCHER = "researcher"
    CRITIC_FACTUAL = "critic_factual"
    CRITIC_LOGICAL = "critic_logical"
    SYNTHESIZER = "synthesizer"
    SELECTOR_SYNTH = "selector_synth"
    DEBATER = "debater"
    JUDGE = "judge"

@dataclass
class Node:
    id: str
    kind: RoleKind
    prompt_ref: str
    cli: str = "claude-cli:sonnet"

@dataclass
class Edge:
    src: str
    dst: str
    payload: str = "findings"

@dataclass
class SwarmGraph:
    nodes: list[Node]
    edges: list[Edge]
    rounds: int = 4
    consensus_strategy: str = "judge_select"
    def fingerprint(self) -> str: ...
    def to_orchestrator_config(self) -> dict: ...
```

```python
class Operator(Protocol):
    name: str
    def applicable(self, g: SwarmGraph) -> bool: ...
    def apply(self, g: SwarmGraph, ctx: dict) -> SwarmGraph: ...

class AddResearcher: ...
class RemoveCritic: ...
class AddDebateRound: ...
class SwapSynthForSelector: ...
class AddJudgeEnsemble: ...
class IncreaseRounds: ...
class FuseRoles: ...
class SwapCLI: ...

OPERATOR_REGISTRY: list[type[Operator]] = [...]
```

```python
@dataclass
class MCTSNode:
    graph: SwarmGraph
    parent: "MCTSNode | None"
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_score: float = 0.0
    untried_ops: list[type[Operator]] = field(default_factory=list)

    def ucb(self, c: float = 1.4) -> float: ...

class AFlowSearch:
    def __init__(self, seed_graph, evaluator, mutator, max_iters=200,
                 max_depth=6, soft_mix=0.7, score_threshold=0.85): ...
    def run(self) -> list[SwarmGraph]: ...
```

### 2.4 Search loop

```python
def search(seed: SwarmGraph, slate, evaluator, max_iters=200):
    root = MCTSNode(seed, parent=None, untried_ops=list(OPERATOR_REGISTRY))
    history = []
    for i in range(max_iters):
        leaf = select(root)                              # soft-UCB descent
        new_op = pick_op(leaf, mutator)                  # LLM picks op + args
        child_graph = new_op.apply(leaf.graph, ctx={"hist": history})
        if seen(child_graph): continue                   # dedup by fingerprint
        score = evaluator.score(child_graph, slate)
        child = MCTSNode(child_graph, parent=leaf, untried_ops=list(OPERATOR_REGISTRY))
        child.visits = 1; child.total_score = score
        leaf.children.append(child)
        backprop(child, score)
        history.append({"i": i, "fp": child_graph.fingerprint(),
                        "score": score, "op": new_op.name})
        if early_stop(history): break
    return rank_frontier(root)
```

### 2.5 Integration contract

- `agents.SwarmRunner.run_from_graph(graph: SwarmGraph, task) -> SwarmTrajectory`. **Biggest Phase 0/1 ask** — runner must compile any valid `SwarmGraph` into executable workflow at runtime.
- `bench.score(graph, slate, seed=42) -> dict[str, float]`
- `orchestrator.compile_graph(graph) -> Workflow` — pure, side-effect-free.
- `metrics.log_aflow_step(graph_fp, score, op_name, parent_fp)`.

### 2.6 Eval design

- **Primary:** best-frontier-graph mean bench score vs seed graph.
- **Secondary:** cost-efficiency frontier — score per CLI-token spent.
- **Diversity:** unique graph fingerprints in top-10 / total visits ≥ 0.4.
- **Promotion threshold:** beats seed by ≥ 0.5 aggregate AND uses ≤ 1.3× seed's cost.

### 2.7 Failure modes

| Failure | Detection | Mitigation |
|---|---|---|
| Invalid workflow (cycle, orphan) | `SwarmGraph.validate()` | Reject before scoring |
| Degenerate branch (unbounded depth) | `max_depth=6` enforced | Hard cap |
| Mutator LLM proposes same op every time | Repetition counter | Force epsilon-random |
| Score variance per-graph too high | Re-score top with seed=43, 44; reject if std > 1.0 | |
| Cost runaway | Per-graph cost cap | Skip score, mark "too expensive" |
| Graph exploits judge | Same as GEPA | Same |

### 2.8 Test plan

```python
def test_seed_graph_validates(): ...
def test_each_operator_preserves_validity(): ...  # parametrized
def test_fingerprint_canonical(): ...
def test_mcts_ucb_explores_unvisited_first(): ...
def test_backprop_updates_ancestors(): ...
def test_dedup_skips_seen_graph(): ...
def test_max_depth_enforced(): ...
def test_resume_from_tree_json(): ...
```

---

## 3. Component 3 — Meta-Insight Loop

### 3.1 Mechanism summary

Insights tagged `meta:` describe the swarm itself. Meta-loop reads, proposes config patches, validates against bench, auto-merges if safe.

Configuration patches are typed schema (TOML keys + bounded numeric ranges + enum strings). Auto-merge gated by regression bound.

### 3.2 File layout

```
blitz-swarm/evolve/
  meta_loop.py
  meta/
    __init__.py
    schema.py          # ConfigPatch dataclass + validators
    proposer.py        # LLM that turns insights -> patches
    validator.py       # bench-driven gate
    merger.py          # writes blitz.toml, commits, rolls back
  evolve/runs_meta/<timestamp>/
```

### 3.3 Concrete API

```python
class PatchScope(Enum):
    GLOBAL = "global"
    DOMAIN = "domain"
    TOPIC = "topic"

@dataclass
class ConfigPatch:
    patch_id: str
    scope: PatchScope
    scope_value: str | None
    key_path: str
    old_value: object
    new_value: object
    rationale: str
    source_insights: list[str]
    risk_class: str               # "low" / "medium" / "high"

ALLOWED_KEY_PATHS = {
    "swarm.max_rounds":            (int, range(1, 9)),
    "swarm.max_agents":            (int, range(2, 25)),
    "consensus.judge_ensemble.n":  (int, range(1, 8)),
    "consensus.strategy":          (str, {"vote", "judge_select", "synthesize"}),
    "memory.top_k_retrieval":      (int, range(1, 12)),
    "evolve.gepa.max_metric_calls":(int, range(100, 5000)),
    "evolve.aflow.max_iters":      (int, range(20, 600)),
}
```

### 3.4 Main loop

```python
def run_meta_loop():
    insights = memory.fetch_insights(tag_prefix="meta:", min_confidence=0.6, freshness_days=14)
    patches = proposer.propose_patches(insights)

    candidates = []
    for p in patches:
        if not schema.validate(p):
            log("rejected_unsafe_path", p); continue
        baseline = bench.score(load_swarm_config(), SLATE)
        with apply_patch_temporarily(p):
            after = bench.score(load_swarm_config(), SLATE)
        candidates.append((p, baseline, after))

    for p, baseline, after in candidates:
        delta = aggregate(after) - aggregate(baseline)
        regressions = per_dim_regressions(baseline, after, bound=0.3)
        if delta >= 0.4 and not regressions and p.risk_class != "high":
            merger.merge(p)
            memory.add_insight(f"meta-applied:{p.patch_id}", confidence=delta)
        else:
            queue_for_human_review(p, baseline, after)
```

### 3.5 blitz.toml additions

```toml
[evolve]
backend = "cli"            # "cli" or "managed_agents"
auto_merge_threshold = 0.4
regression_bound = 0.3
high_risk_paths = ["swarm.max_rounds", "consensus.strategy"]
gepa_max_cost_usd = 40
aflow_max_iters = 200
meta_loop_freq = "weekly"
```

### 3.6 Eval design

- **Bench delta:** `aggregate(after) - aggregate(baseline)` ≥ `auto_merge_threshold`.
- **Regression bound:** no per-dim drop ≥ 0.3.
- **Replication:** rerun with different bench seed; require positive delta on both.
- **Holdout:** 20% slate never seen by GEPA/AFlow/meta-loop, used for final acceptance.

### 3.7 Failure modes

| Failure | Detection | Mitigation |
|---|---|---|
| Patch escapes allow-list | `schema.validate` fail | Reject, log `meta-loop:unsafe-path` |
| Patch oscillation | Loop detector on `(key_path, new_value)` last 4 patches | 30-day cool-down |
| Bench gaming | 20% holdout never seen | Reject if holdout regresses |
| Drift from human intent | All `risk_class=high` go to L3 | Hard rule |
| Insight fabrication | Cross-check `source_insights` exist in `memory.db` | Reject |
| Bench cost runaway | Per-meta-loop budget | Hard cap |
| Multiple patches conflict | Apply one at a time; revalidate from scratch | |

Rollback: `merger.merge(p)` writes `blitz.toml.bak.<timestamp>`. Revert: `cp blitz.toml.bak.<ts> blitz.toml && git commit`.

### 3.8 Test plan

```python
def test_schema_rejects_unknown_key(): ...
def test_schema_rejects_out_of_range_int(): ...
def test_proposer_cites_real_insight_ids(): ...
def test_apply_patch_temporarily_restores_on_exception(): ...
def test_auto_merge_threshold_gate(): ...
def test_regression_bound_blocks_per_dim_drop(): ...
def test_oscillation_detector_30_day_cooldown(): ...
def test_holdout_regression_blocks_merge(): ...
```

---

## 4. Component 4 — Cross-CLI Heterogeneity

### 4.1 Mechanism summary

Maryanskyy 2603.20324 + Diversity for the Win (OpenReview ptUxbqOGrC): heterogeneous models help iff paired with judge-driven selection. Diverse-team-with-judge-selection wins 81% vs 51.2% homogeneous. Synthesis-aggregation diversity: 0/42 tasks.

Phase 1 ships `selector_synth`. Phase 3 ships heterogeneity layer that exploits it. Routes role calls to local CLIs: `claude -p`, `codex exec`, `gemini -p`. **All subscription-based — Rule #10 compliant.**

### 4.2 File layout

```
blitz-swarm/heterogeneity/
  __init__.py
  cli_router.py
  routing_table.toml
  cli_adapters/
    claude_cli.py
    codex_cli.py
    gemini_cli.py
```

### 4.3 Concrete API

```python
@dataclass
class CLICall:
    cli_id: str
    model: str | None
    prompt: str
    schema: dict | None
    timeout_s: int

@dataclass
class CLIResult:
    cli_id: str
    text: str
    structured: dict | None
    elapsed_s: float
    fallback_used: bool

class CLIAdapter(Protocol):
    cli_id: str
    def is_available(self) -> bool: ...
    def call(self, c: CLICall) -> CLIResult: ...

class ClaudeCLI:
    cli_id = "claude"
    def is_available(self): return shutil.which("claude") is not None
    def call(self, c: CLICall) -> CLIResult:
        cmd = ["claude", "-p", c.prompt]
        if c.schema:
            cmd += ["--json-schema", json.dumps(c.schema)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=c.timeout_s)
        return CLIResult("claude", out.stdout, _parse(out.stdout, c.schema), ...)

class CodexCLI:
    cli_id = "codex"
    def is_available(self): return shutil.which("codex") is not None
    def call(self, c: CLICall) -> CLIResult:
        out = subprocess.run(["codex", "exec", "-"], input=c.prompt,
                             capture_output=True, text=True, timeout=c.timeout_s)
        return CLIResult("codex", out.stdout, _try_json(out.stdout), ...)

class GeminiCLI:
    cli_id = "gemini"
    def is_available(self): return shutil.which("gemini") is not None
    def call(self, c: CLICall) -> CLIResult:
        out = subprocess.run(["gemini", "-p", c.prompt],
                             capture_output=True, text=True, timeout=c.timeout_s)
        return CLIResult("gemini", out.stdout, _try_json(out.stdout), ...)

class CLIRouter:
    def __init__(self, table_path="heterogeneity/routing_table.toml"):
        self.table = _load_routing_table(table_path)
        self.adapters = {a.cli_id: a for a in [ClaudeCLI(), CodexCLI(), GeminiCLI()]
                         if a.is_available()}
        if "claude" not in self.adapters:
            raise RuntimeError("claude CLI required as fallback")

    def route(self, role: str, domain: str, prompt: str, schema: dict | None,
              timeout_s: int = 120) -> CLIResult:
        cli_id = self.table.get((domain, role)) or self.table.get(("*", role)) or "claude"
        adapter = self.adapters.get(cli_id)
        fallback = False
        if adapter is None:
            adapter = self.adapters["claude"]
            fallback = True
        ...
```

### 4.4 Routing table

```toml
# heterogeneity/routing_table.toml
[default]
researcher       = "claude"
critic_factual   = "gemini"     # web grounding
critic_logical   = "claude"
synthesizer      = "claude"
selector_synth   = "claude"
debater          = "codex"      # adversarial
judge            = "claude"

[domain.code]
researcher       = "codex"
critic_factual   = "claude"
synthesizer      = "claude"

[domain.logical-reasoning]
researcher       = "claude"
critic_logical   = "claude"
debater          = "gemini"

[domain.factual]
researcher       = "gemini"
critic_factual   = "gemini"
```

### 4.5 Integration contract

- `agents.Agent.invoke(prompt, schema)` replaced with router-aware: `CLIRouter.route(self.role, task.domain, prompt, schema)`.
- `metrics.log_cli_call(role, domain, cli_id, elapsed_s, fallback)`.
- `consensus.JudgeEnsemble` unchanged; selector_synth makes heterogeneity pay off.

### 4.6 Eval design

- Comparison: all-claude baseline vs heterogeneous on 30-task slate.
- Threshold: heterogeneous wins by ≥ 0.5 aggregate AND no domain regresses by > 0.3.
- Per-CLI attribution: which CLI handled which call recorded in `metrics.jsonl`.

### 4.7 Failure modes

| Failure | Detection | Mitigation |
|---|---|---|
| CLI missing | `is_available()` check | Fall back to claude, log `cli-fallback` |
| CLI rate-limit / login expired | Subprocess returns non-zero or "login required" | Fall back, surface `cli-login-stale` |
| Subprocess hang | `timeout_s` enforced | Kill, fall back |
| Schema differences (codex doesn't honor JSON schema flag) | `_try_json` parses output | Mark `structured=None`; consumer handles |

### 4.8 Test plan

```python
def test_router_falls_back_to_claude_when_codex_missing(): ...
def test_routing_table_domain_override_wins_over_default(): ...
def test_subprocess_timeout_triggers_fallback(): ...
def test_metrics_logged_for_every_call(): ...
def test_each_adapter_parses_or_returns_structured_None(): ...
```

---

## 5. Component 5 — Anthropic Managed Agents Adapter (Optional)

### 5.1 Mechanism summary

May 7 2026: Anthropic shipped multiagent sessions + Outcomes in public beta. Header `managed-agents-2026-04-01`. Coordinator agent decomposes task, delegates to up to 20 specialist agents that share filesystem and persist events. Outcomes is separate-context grader pushing through iterations until rubric met.

This is **opt-in** because it costs API dollars (Rule #10). For Joona's R&D, paid usage is fine when explicitly enabled.

### 5.2 Trade-offs

| Dimension | CLI backend | Managed-Agents backend |
|---|---|---|
| Cost | $0 (subscription) | $$$ (API metered) |
| Max parallel specialists | OS limit (~50) | 20 (Anthropic cap) |
| Shared filesystem | Local FS (free) | Managed container (ephemeral) |
| Persistence | `memory.db` + blackboard | Server-side events |
| Long-running | Limited by laptop | Cloud, hours of runtime |
| Outcomes grader | We build via judge ensemble | Provided, separate-context |
| Rule #10 | Yes (default) | No (opt-in for owner R&D) |

### 5.3 File layout

```
blitz-swarm/managed_agents/
  __init__.py
  adapter.py
  agent_specs/             # one per role; created idempotently
    researcher.json
    critic_factual.json
    ...
  outcomes/
    swarm_outcome.md
```

### 5.4 Concrete API

```python
import anthropic
from anthropic import Anthropic

BETA = "managed-agents-2026-04-01"

class ManagedAgentsAdapter:
    def __init__(self, client: Anthropic, env_id: str, agent_ids: dict[str, str]): ...

    @classmethod
    def bootstrap(cls, client, role_specs, multiagent_config) -> "ManagedAgentsAdapter":
        env = client.beta.environments.create(
            extra_headers={"anthropic-beta": BETA},
            packages=["python-3.11", "node-20"],
            network={"egress": "restricted"})
        agent_ids = {}
        for role, spec in role_specs.items():
            a = client.beta.agents.create(
                extra_headers={"anthropic-beta": BETA},
                name=f"blitz-{role}", model=spec["model"],
                system_prompt=spec["system_prompt"],
                tools=spec.get("tools", []))
            agent_ids[role] = a.id
        coordinator = client.beta.agents.create(
            extra_headers={"anthropic-beta": BETA},
            name="blitz-coordinator", model="claude-opus",
            system_prompt=multiagent_config["coordinator_prompt"],
            multiagent={"agents": [{"type": "agent", "id": v} for v in agent_ids.values()]})
        agent_ids["coordinator"] = coordinator.id
        return cls(client, env.id, agent_ids)

    def run_task(self, task, rubric_md=None, max_iter=6) -> dict:
        session = self.client.beta.sessions.create(
            extra_headers={"anthropic-beta": BETA},
            agent_id=self.agent_ids["coordinator"],
            environment_id=self.env_id,
            title=f"blitz:{task.task_id}")
        if rubric_md:
            self.client.beta.sessions.events.create(
                session_id=session.id, extra_headers={"anthropic-beta": BETA},
                define_outcome={"description": task.topic, "rubric": rubric_md,
                                "max_iterations": max_iter})
        self.client.beta.sessions.events.create(
            session_id=session.id, extra_headers={"anthropic-beta": BETA},
            user_message={"content": task.topic})
        events = list(self.client.beta.sessions.events.stream(
            session_id=session.id, extra_headers={"anthropic-beta": BETA}))
        return self._collect_outputs(session.id, events)
```

### 5.5 blitz.toml

```toml
[evolve.managed_agents]
api_key_env = "ANTHROPIC_API_KEY"
beta_header = "managed-agents-2026-04-01"
max_specialists = 20
session_max_runtime_min = 60
spend_cap_usd_per_run = 5
spend_cap_usd_per_day = 50
```

### 5.6 Failure modes

| Failure | Detection | Mitigation |
|---|---|---|
| Beta header missing | API rejects with 400 | Hard fail at adapter init |
| Spend cap exceeded | Pre-call accounting via `metrics.estimated_spend()` | Abort session, emit `managed-agents:budget` |
| Coordinator depth > 1 | Anthropic ignores; we enforce in spec building | Static check |
| > 20 specialists | Schema validation | Reject |
| Network egress denied | Pre-flight tool list check | Surface to Joona |
| Outcomes grader infinite-loops | `max_iterations` enforced | Hard cap |
| Cost vs CLI regression | `$_per_score_point` logged | If 3× CLI cost without quality lift, auto-flip backend back to CLI |

### 5.7 Test plan

```python
def test_bootstrap_creates_one_agent_per_role(): ...
def test_run_task_with_rubric_calls_define_outcome(): ...
def test_spend_cap_aborts_session(): ...
def test_backend_switch_via_blitz_toml(): ...
def test_trajectory_schema_matches_cli_backend(): ...
```

---

## 6. Component 6 — Bench Harness Contract (Phase 0 deliverable)

```python
@dataclass
class BenchScore:
    aggregate: float
    per_dim: dict[str, float]
    per_task: list[dict]
    cost_usd: float
    elapsed_s: float
    seed: int

def load_slate(path="bench/slate.jsonl", split=(0.7, 0.3), seed=42) -> tuple[list, list]: ...
def score(swarm_config, slate, seed=42, backend="cli") -> BenchScore: ...
def regression_check(baseline, candidate, bound=0.3) -> list[str]: ...
```

### 6.1 Reproducibility requirements

- Same `seed` → same `BenchScore.aggregate` within ±2%, ≥ 95% of time, over 5 reruns.
- Slate file hash-pinned in CHANGELOG.
- Judge ensemble random sub-judge ordering also seeded.
- Holdout slate (`bench/holdout.jsonl`) never touched by GEPA/AFlow training.

### 6.2 Test plan

```python
def test_load_slate_returns_correct_split_sizes(): ...
def test_score_deterministic_within_2pct_over_5_reruns(): ...
def test_regression_check_flags_per_dim_drop(): ...
def test_holdout_never_appears_in_train_or_val(): ...
def test_slate_hash_pinned_in_changelog(): ...
```

---

## 7. The Recursion Structure

### 7.1 What L2 specifically mutates

L1 hyperparams L2 is allowed to touch:

| L1 hyperparam | Range | L2 trigger condition |
|---|---|---|
| `evolve.gepa.max_metric_calls` | 100..5000 | 3+ insights "GEPA stopped early" or "never converged" |
| `evolve.gepa.minibatch_size` | 4..32 | Reflection cost trends high without quality lift |
| `evolve.aflow.max_iters` | 20..600 | "Frontier still expanding at last iter" |
| `evolve.aflow.max_depth` | 3..10 | Deep mutations consistently outperform shallow |
| `evolve.aflow.soft_mix` | 0.3..0.95 | Tree exploitation/exploration balance off |
| `consensus.judge_ensemble.n` | 1..8 | Judge variance > threshold on specific domain |
| `swarm.max_rounds` | 1..9 | Round-N marginal quality gain trend |

Anything outside this allow-list goes to L3.

### 7.2 What L3 alone can change

- Which CLIs are in routing table (adding new CLI = L3).
- Which roles exist (adding `verifier`, removing `debater`).
- The bench slate itself.
- Auto-merge thresholds (`auto_merge_threshold`, `regression_bound`).
- The `high_risk_paths` list itself.
- Switching backend `cli` ↔ `managed_agents`.

### 7.3 Why we cap at L3 (no L4)

A live L4 would mutate the meta-loop's own gating thresholds. That's the RSI runaway scenario the ICLR 2026 RSI workshop highlights. Fix is structural: don't let the system rewrite its own safety bounds. L3 is humans-in-the-loop and that's where recursion stops by design.

### 7.4 Insight tag taxonomy

```
meta:gepa-converged         L2 reads
meta:gepa-stopped-early     L2 reads
meta:aflow-frontier-active  L2 reads
meta:judge-variance-high    L2 reads + L1 reads (for ensemble n)
meta:cli-fallback           L2 reads (routing-table flag for L3)
meta:bench-instability      L3 escalation only
meta:judge-exploit          L3 escalation only (potential adversarial)
meta:applied:<patch_id>     L2 writes after merge
meta:reverted:<patch_id>    L2 writes after rollback
```

---

## 8. Cross-Component Recursion Examples

### Example 1 — L0 failure → L1a fix → L2 generalization

1. L0 task on logical-reasoning fails: `accuracy=4.1` (low).
2. Insight: `meta:judge-variance-high domain=logical-reasoning`.
3. Nightly L1a (GEPA) on logical-reasoning subset proposes new `critic_logical.md` with stronger sourcing. Pareto-promoted.
4. Bench score on logical-reasoning rises by 0.8 aggregate.
5. Insight: `meta:gepa-converged domain=logical-reasoning lift=0.8`.
6. Weekly L2 reads: GEPA converged 3× in a row at K=1500 with full slack remaining. Proposes `evolve.gepa.max_metric_calls = 800` for logical-reasoning (saves cost).
7. Bench validates: aggregate unchanged, cost down 47%. Auto-merged.

### Example 2 — L1b discovers new architecture → L2 freezes it

1. AFlow's MCTS finds graph: `researcher → debater(codex) → critic_factual(gemini) → selector_synth(claude)`. 0.9 aggregate lift.
2. Promoted; new default.
3. AFlow keeps exploring; 4 nightly runs no further improvement.
4. Insight: `meta:aflow-frontier-active=false`.
5. L2 drops `evolve.aflow.max_iters` 200 → 80 to save bench-cost.

### Example 3 — Component 4 + judge_select payoff

1. L0 runs diverse-team-without-selector → quality drops below all-claude. (Maryanskyy effect.)
2. Insight: `meta:judge-variance-high domain=*` + `meta:cli-fallback claude->codex`.
3. L1b's AFlow proposes `SwapSynthForSelector` operator.
4. Bench: heterogeneous + selector beats homogeneous baseline by 0.7. Promoted.
5. Synergy of Component 4 + selector_synth becomes documented in insight graph.

### Example 4 — L2 fails its own gate, L3 escalation

1. L2 proposes `consensus.strategy = "vote"` (high-risk).
2. Schema validates; aggregate up 0.4 BUT one domain regresses 0.5.
3. L2's regression bound (0.3) blocks. Routed to L3 review queue.
4. Joona reviews, decides not to merge.
5. Insight: `meta:l3-rejected patch_id=...`.

---

## 9. End-to-End Cost Model

| Layer | Cost driver | Default cap | Typical actual |
|-------|------------|-------------|----------------|
| L0 | per-task subprocess | unbounded (CLI) | $0 |
| L1a | reflection LM (GEPA) | 1500 metric calls | 1100 calls |
| L1b | bench scores (AFlow) | 200 iters × 30 tasks | 4500 scores |
| L2 | bench reruns × N patches | 5 × 30 × 2 seeds | 300 scores |

If `backend = "managed_agents"`:
- L0 ≈ $0.15 per task
- L1a/b/L2 multiplied by per-call API cost
- Daily cap `spend_cap_usd_per_day = 50` enforced

"Token unbounded" is honored: caps are bumpable in `blitz.toml`. Without caps the system is unsafe.

---

## 10. Phasing into Phase 0/1/2/3

| Phase | Components delivered | New artifacts |
|-------|----------------------|--------------|
| 0 | Bench (Comp 6) | `bench/`, scoring API |
| 1 | Selector + judge ensemble | `consensus.py` upgrades |
| 2 | Insight graph + meta tags | `gmemory/` schema, tag taxonomy |
| 3a | GEPA (Comp 1) | `scripts/optimize_prompts.py`, `evolve/gepa_adapter.py` |
| 3b | AFlow (Comp 2) | `evolve/aflow_search.py`, `evolve/aflow/` |
| 3c | Meta-loop (Comp 3) | `evolve/meta_loop.py`, `evolve/meta/` |
| 3d | Heterogeneity (Comp 4) | `heterogeneity/`, routing table |
| 3e | Managed Agents (Comp 5, opt-in) | `managed_agents/` |

Implementation order: GEPA before AFlow (GEPA's eval design exercises bench thoroughly). Meta-loop last (depends on enough insights existing).

---

## 11. Sources

- [GEPA (arXiv 2507.19457)](https://arxiv.org/abs/2507.19457) — ICLR 2026 oral.
- [GEPA OpenReview](https://openreview.net/forum?id=RQm2KQTM5r)
- [gepa-ai/gepa GitHub](https://github.com/gepa-ai/gepa)
- [AFlow (arXiv 2410.10762)](https://arxiv.org/abs/2410.10762)
- [FoundationAgents/AFlow GitHub](https://github.com/FoundationAgents/AFlow)
- [A2Flow (arXiv 2511.20693)](https://arxiv.org/abs/2511.20693)
- [When Agents Disagree (arXiv 2603.20324)](https://arxiv.org/abs/2603.20324)
- [Diversity for the Win](https://openreview.net/forum?id=ptUxbqOGrC)
- [EvoMAS (arXiv 2602.06511)](https://arxiv.org/abs/2602.06511)
- [EvoAgentX (arXiv 2507.03616)](https://arxiv.org/abs/2507.03616v2)
- [AlphaEvolve (arXiv 2506.13131)](https://arxiv.org/abs/2506.13131)
- [Claude Managed Agents Overview](https://platform.claude.com/docs/en/managed-agents/overview)
- [Claude Managed Agents Multiagent](https://platform.claude.com/docs/en/managed-agents/multi-agent)
- [Anthropic agents updates 2026-05-07](https://9to5mac.com/2026/05/07/anthropic-updates-claude-managed-agents-with-three-new-features/)
- [ICLR 2026 RSI Workshop](https://recursive-workshop.github.io/)
- [Codex CLI noninteractive](https://developers.openai.com/codex/noninteractive)
- [Gemini CLI shell tool](https://google-gemini.github.io/gemini-cli/docs/tools/shell.html)
