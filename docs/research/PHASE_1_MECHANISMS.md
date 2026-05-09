# Phase 1 Mechanism Upgrades — Implementation Deep Dive

**Status:** Frozen research output from background-task agent run 2026-05-09.
**Source:** Phase 1 mechanism deep-dive subagent.
**Consumed by:** `mechanisms/`, `prompts/`, `agents.py`, `consensus.py`, `orchestrator.py` implementation.

---

## Mechanism 1 — Genealogy-graph cascade defense

### 1.1 Paper summary

Xie et al., "From Spark to Fire" (arXiv:2603.04474v1, 4 Mar 2026).

**Propagation model.** Collaboration graph G with adjacency A. Each agent i has infection state s_i(t) ∈ [0,1]:

```
s_i(t+1) = (1 − δ) · s_i(t) + (1 − s_i(t)) · f_i({s_j(t)}_{j ∈ N(i)}, G)

f_i^prod(t) = 1 − Π_{j ∈ N(i)} (1 − β · a_ij · s_j(t))
```

Early-stage risk criterion: **R ≈ β · ρ(A) / δ**, where ρ(A) is the spectral radius of the adjacency matrix. R > 1 → supercritical → cascade.

**Three vulnerability classes:** cascade amplification (concurrent tainted mentions multiply), topological fragility (error growth aligns with principal eigenvector — hub injections dominate; LangGraph showed 10.31× hub/leaf Impact Factor), consensus inertia (early errors crystallise into intermediate artifacts).

**Defense — Lineage Graph governance layer.** Message-layer middleware:

1. **Decompose** outbound messages into atomic claims c_1…c_k (single subject-predicate-object).
2. **Screen** each c_i against persistent Lineage Graph L = (V, E):
   - **Green** — entailed by trusted history → release.
   - **Red** — contradicts trusted history → block + rollback.
   - **Yellow** — novel/unverified → routed by policy (Speed / Balanced / Strict).
3. **Verify** Yellow claims via external evidence + adjudication LLM, then promote (→ Green) or reject (→ Red).

**Numbers (paper-stated):** Reflection baseline 0.32 → Speed 0.89 → Balanced 0.93 → Strict 0.94 BICR across 6 frameworks (LangChain, MetaGPT, AutoGen, CAMEL, CrewAI, LangGraph).

### 1.2 Concrete API

**File:** `mechanisms/cascade_guard.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

ClaimVerdict = Literal["green", "yellow", "red"]
GuardMode = Literal["speed", "balanced", "strict", "off"]

@dataclass
class AtomicClaim:
    id: str                                # uuid4
    text: str
    source_agent: str
    source_round: int
    parent_claim_ids: list[str] = field(default_factory=list)
    verdict: ClaimVerdict = "yellow"
    supports: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    verify_notes: str = ""

@dataclass
class GuardConfig:
    mode: GuardMode = "balanced"
    decompose_model: str = "haiku"
    adjudicate_model: str = "sonnet"
    max_claims_per_msg: int = 12
    yellow_release_threshold: float = 0.7   # Speed: 0.5, Balanced: 0.7, Strict: 0.9
    cache_ttl_s: int = 600

class CascadeGuard:
    def __init__(self, cfg: GuardConfig, blackboard=None): ...
    def decompose(self, agent_output: dict) -> list[AtomicClaim]: ...
    def screen(self, claim: AtomicClaim) -> ClaimVerdict: ...
    def verify_yellow(self, claim: AtomicClaim) -> ClaimVerdict: ...
    def on_agent_output(self, output: dict, round_n: int) -> dict: ...
    def on_agent_error(self, agent_id: str, round_n: int, reason: str): ...
    def filter_context(self, outputs: list[dict]) -> list[dict]: ...
    def round_summary(self) -> dict: ...
```

**Wiring point in `orchestrator.py`:** before `_write_to_blackboard`, call `guard.on_agent_output`. Replace `build_context(...)` with `build_context(guard.filter_context(...), ...)`. On any output where `_error == True`, call `guard.on_agent_error`.

**Schema additions to AGENT_OUTPUT_SCHEMA:** `parent_message_ids: array[string]`, `agent_error_state: bool` (both optional).

### 1.3 Modes

- **speed**: skip `verify_yellow`, threshold 0.5 (release on any non-contradiction)
- **balanced**: verify Yellow at hubs only (synthesizer, judge), threshold 0.7
- **strict**: verify all Yellow, block on any unresolved, threshold 0.9

### 1.4 Edge cases

1. Decomposer LLM fails → treat whole message as one Yellow claim.
2. Cache leakage between runs → extend `Blackboard.cleanup()` to `lineage:*`.
3. Adjudicator hallucinates contradictions → 2-of-3 mini-vote on Red verdicts in Strict mode.
4. First round has no Lineage Graph → everything Yellow → speed mode = v0.1 behavior.
5. Self-supporting echo chamber → tag claims from same round as "lateral", only earlier-round Green count as anchors.
6. Researcher language/dialect mangles decomposer → rate-limit retries, mark Yellow + log on second failure.
7. Synthesizer is the hub → always run with `verify_yellow=True` regardless of mode.
8. Blocked output starves round → if >50% blocked, rerun those agents with Red feedback as new context (cap 1 rerun/round).
9. Memory unbounded → cap at N=2000 claims, evict oldest non-Green.
10. Off mode → must be tested as identity passthrough.

### 1.5 Test cases

```python
def test_decompose_simple_findings(monkeypatch): ...
def test_screen_contradiction_marks_red(guard): ...
def test_screen_entailment_marks_green(guard): ...
def test_error_taints_descendants(guard): ...
def test_filter_context_omits_blocked(guard): ...
def test_speed_mode_skips_yellow_verify(guard_speed, mock_verify): ...
def test_strict_mode_blocks_red(guard_strict): ...
def test_lineage_persists_across_rounds(guard): ...
def test_off_mode_is_identity(guard_off): ...
def test_decomposer_failure_falls_back_to_yellow(guard, broken_llm): ...
```

### 1.6 Cost

Per round, 5 agents × 8 claims/agent:
- Decompose: 5 × 800 = 4k tokens
- Screen: 40 claims × ~2k batched = 8k tokens
- Verify (Balanced): ~30% Yellow → 12 calls × 1.2k = 14k tokens
- **~26k tokens per round at haiku rates ≈ $0.01–0.03 per round, +3–8s wall clock**

Speed mode: ~12k tokens, +1–3s. Strict: ~50k tokens, +10–15s. Baseline 5-agent round uses 50–80k tokens; guard adds 10–60% overhead.

### 1.7 Failure-mode interactions

- **Guard ↔ Judge ensemble (M2):** judges receive a separate `_blocked_claims` digest; dissent stays visible.
- **Guard ↔ Selection synthesis (M3):** add `redact_red()` for partial taints — red sentences stripped, output passes.
- **Guard ↔ Persona critics (M4):** persona-critic outputs tagged with `bypass_guard=True` for screening; their findings don't enter Lineage Graph as anchors.

### 1.8 Latest extensions

- INFA-Guard (arXiv 2601.14667, Jan 2026) — three-state model (infected/benign/attacker).
- MAS-Shield (arXiv 2511.22924, Nov 2025) — learned classifier pre-filter.
- XG-Guard (arXiv 2512.18733, Dec 2025) — explainability via sentence + token bi-level encoding.
- CASCADE (arXiv 2604.17125, Apr 2026) — cascaded hybrid defense for prompt-injection in MCP.

---

## Mechanism 2 — Multi-judge debate with adaptive stability

### 2.1 Paper summary

Hu et al., "Multi-Agent Debate for LLM Judges with Adaptive Stability Detection" (arXiv:2510.12697, NeurIPS 2025).

**Setup.** N judges produce binary correctness votes (or rubric-bucketed scores) per debate round. Score-share at round t modeled as time-varying Beta-Binomial mixture:

```
S^t  ~  w^t · BB(k, α_1^t, β_1^t) + (1 − w^t) · BB(k, α_2^t, β_2^t)
```

Two latent components capture "confident-correct" vs "uncertain/wrong" judge populations. Parameters fit by EM each round.

**KS-test stability:**
```
D_t = sup_{ψ ∈ [0,1]} | F^t(ψ) − F^{t-1}(ψ) |
```

Halt when **D_t < 0.05 for 2 consecutive rounds.**

**Defaults:** N = 7 judges, max k = 10 rounds, temperature 1.0. LLMBar: 77.75% (majority vote) → 81.83% (debate) [paper-stated].

**Cross-reference: Autorubric** (arXiv 2603.00077): per-criterion atomic LLM calls, CANNOT_ASSESS verdict, verdict-balanced few-shot.

### 2.2 Concrete API

**File:** `mechanisms/judge_ensemble.py`

```python
@dataclass
class JudgeConfig:
    n_judges: int = 3                     # blitz-swarm budget; paper used 7
    max_rounds: int = 5                   # paper used 10; we cap at 5
    ks_threshold: float = 0.05
    ks_consecutive: int = 2
    min_rounds: int = 2
    rubric_path: str = "prompts/general/quality_rubric.yaml"
    seed_strategy: str = "prompt_seed"    # "prompt_seed" | "model_mix" | "both"
    judge_models: list[str] = field(
        default_factory=lambda: ["sonnet", "sonnet", "haiku"]
    )
    em_max_iter: int = 100
    em_tol: float = 1e-6
    score_buckets: int = 11
    log_path: str = "judge_ensemble.jsonl"

@dataclass
class JudgeVote:
    judge_id: str
    round: int
    rubric_scores: dict[str, float]
    aggregate_score: float
    score_bucket: int
    rationale: str

@dataclass
class StabilityState:
    round: int
    w: float
    alpha1: float; beta1: float
    alpha2: float; beta2: float
    ks_stat: float
    consecutive_below_threshold: int
    halted: bool
    final_decision: float | None

class JudgeEnsemble:
    def __init__(self, cfg: JudgeConfig): ...
    def round(self, candidate: str, debate_history: str) -> StabilityState: ...
    def is_stable(self) -> bool: ...
    def final_score(self) -> float | None: ...
    def majority_vote(self) -> str: ...
```

### 2.3 Pseudocode

```python
def _update_state(self, votes: list[JudgeVote]) -> StabilityState:
    scores = [v.score_bucket for v in votes]
    k = self.cfg.score_buckets - 1
    t = len(self.history)

    w, a1, b1, a2, b2 = em_fit_bb_mixture(
        scores, k=k, max_iter=self.cfg.em_max_iter, tol=self.cfg.em_tol,
        warm_start=self._last_params())

    ks = 0.0
    if t >= 2:
        prev = self._mixture_cdf_prev()
        curr = mixture_cdf(w, a1, b1, a2, b2, k)
        ks = max(abs(p - c) for p, c in zip(prev, curr))

    last = self.state_log[-1] if self.state_log else None
    consec = (last.consecutive_below_threshold if last else 0)
    consec = consec + 1 if ks < self.cfg.ks_threshold else 0

    halted = (
        t >= self.cfg.min_rounds
        and consec >= self.cfg.ks_consecutive
    ) or t >= self.cfg.max_rounds

    return StabilityState(...)
```

### 2.4 Edge cases

1. N=1 fallback → single judge; KS undefined → halt after `min_rounds`.
2. All judges agree round 1 → variance ≈ 0 → BB collapse → detect σ²<1e-4 and halt early with confidence flag.
3. Judge timeout → exclude vote; if N-1 < 2, halt with degraded flag.
4. Mixture component collapse (w → 0/1) → re-init with random perturbation; if collapses again, fall back to single Beta.
5. Score discretization → clamp to k.
6. Debate history bloat → cap each judge's prior rationale at 200 tokens.
7. Rubric drift → hash-pin rubric in run metadata.
8. Two-component multimodality from biased seeds → rotate seed assignments across rounds.
9. min_rounds vs ks_consecutive interaction → earliest halt is round 3 with defaults.
10. JSON parse failure → reuse `_parse_agent_output` retry; on failure, treat as missing vote.

### 2.5 Test cases

```python
def test_em_recovers_known_bb_mixture(): ...
def test_ks_below_threshold_two_rounds_halts(ensemble): ...
def test_min_rounds_prevents_early_halt(ensemble): ...
def test_max_rounds_force_halt(ensemble_no_convergence): ...
def test_judge_timeout_excludes_vote(ensemble, mock_timeout_judge): ...
def test_rubric_grounded_scoring(ensemble, rubric): ...
def test_unanimous_round_does_not_crash(ensemble): ...
def test_seed_rotation_across_rounds(ensemble): ...
def test_state_log_persistence(ensemble, tmp_path): ...
def test_warm_start_reuses_prior_params(): ...
```

### 2.6 Cost

N=3 (sonnet × 2 + haiku × 1), 4-round debate: ~50k tokens, **$0.05–$0.15, +20–30s**. v0.1 single judge ~3k tokens / 5s. Ensemble ~15× tokens, ~5× wall clock at N=3 / 4 rounds. Gate behind `judge_ensemble.enabled = false` by default.

### 2.7 Failure-mode interactions

- **Ensemble ↔ Cascade guard (M1):** add "guard transparency" appendix in judge prompt.
- **Ensemble ↔ Selection synthesizer (M3):** run ensemble per candidate at low N=2, then full N=3 on selected.
- **Ensemble ↔ Persona critics (M4):** persona critics' findings as inputs to judges; judges remain separate role.

### 2.8 Latest extensions

- Autorubric (arXiv 2603.00077, Feb 2026) — rubric-grounding inside each judge call.
- LLM-Rubric (arXiv 2501.00274) — multidimensional calibrated rubric eval.
- Stop Overvaluing Multi-Agent Debate (arXiv 2502.08788) — supports `min_rounds=2, max_rounds=5` cap.
- Adaptive Heterogeneous Multi-Agent Debate (Springer 2025) — informs `judge_models` mix.

---

## Mechanism 3 — Selection-bottleneck synthesizer

### 3.1 Paper summary

Maryanskyy, "When Agents Disagree: The Selection Bottleneck" (arXiv:2603.20324, Mar 2026).

**Core claim.** Output quality from MAS depends on aggregator, not agent diversity alone:

```
Q(T, s) = s · O(T) + (1 − s) · M(T)
```

s ∈ [0,1] is selection skill. Crossover threshold:

```
s* = (μ_best − M(T_d)) / (O(T_d) − M(T_d))
```

Empirically `s* ≈ 0.567 [95% CI 0.48–0.65]`. Synthesis ≈ 0; selection ≈ 0.7.

**Results (paper-stated):** Diverse 3-agent + judge-selection (Bradley-Terry): **0.810 win rate.** Homogeneous + same judge: 0.512. Synthesis-based: 0.179 (worse than single baseline in 82% of comparisons). "Synthesis averages, selection picks."

**Selection mechanism.** Pairwise comparisons across all candidate pairs, three judges, randomized presentation order to control position bias. Bradley-Terry MLE.

### 3.2 Concrete API

**File:** `mechanisms/selector_synth.py`

```python
@dataclass
class SelectorConfig:
    granularity: Literal["whole", "section", "paragraph"] = "section"
    n_judges: int = 3
    judge_models: list[str] = field(
        default_factory=lambda: ["sonnet", "sonnet", "haiku"]
    )
    pairwise_random_order: bool = True
    bt_regularization: float = 1e-3
    min_section_chars: int = 200
    section_split_strategy: str = "markdown_h2"
    transition_strategy: str = "smooth"
    smooth_model: str = "haiku"
    cache_path: str = "selector_cache.jsonl"

@dataclass
class Span:
    id: str
    source_agent: str
    source_round: int
    heading: str | None
    text: str
    char_start: int
    char_end: int

@dataclass
class PairwiseVerdict:
    judge_id: str
    span_a_id: str
    span_b_id: str
    winner: Literal["a", "b", "tie"]
    rationale: str

@dataclass
class SelectionResult:
    selected_span_ids: list[str]
    bt_scores: dict[str, float]
    sections_assembled: list[Span]
    final_text: str
    diagnostics: dict

class SelectorSynth:
    def __init__(self, cfg: SelectorConfig): ...
    def synthesize(self, researcher_outputs, topic, guard_filter=None) -> SelectionResult: ...
    def split_into_spans(self, output: dict) -> list[Span]: ...
    def cluster_spans_by_topic(self, spans: list[Span]) -> dict[str, list[Span]]: ...
    def select_within_cluster(self, cluster: list[Span]) -> Span: ...
    def assemble(self, ordered_spans: list[Span], topic: str) -> str: ...
```

### 3.3 Pairwise judge prompt

```
You compare two candidate research excerpts for the topic "{topic}".

[A]
{span_a.text}

[B]
{span_b.text}

Which is better, judging on:
1. Factual accuracy and citation rigor
2. Specificity and actionability
3. Coverage of the sub-topic
4. Clarity for an implementing engineer

Output JSON: {"winner": "a"|"b"|"tie", "rationale": "<2 sentences>"}.
```

### 3.4 Bradley-Terry MLE

Use `choix.mm_pairwise` (Minorization-Maximization, no scipy required) or implement with iterative updates. Ties contribute 0.5 to each side. Add regularization `reg * I` to avoid runaway skills.

### 3.5 Edge cases

1. Single researcher output → return directly with bt_score=1.0.
2. All candidates identical → all ties → BT skills equal → pick first by stable hash, log warning.
3. Span-cluster mismatch → if cluster-count > 2× expected sections, retry with `granularity="whole"`.
4. Judge produces non-JSON → reuse `_parse_agent_output` retry; on second failure, treat as "tie".
5. BT MLE non-convergent → graph disconnected; add tiny epsilon win for every pair.
6. Judge agreement = 0 → all flip on every pair → fall back to "highest avg confidence" researcher.
7. Topic-drift in spans → include source agent's `subtopic` in clustering.
8. Smoothed assembly hallucinates content → "DO NOT add new facts" + post-check (n-gram overlap > 90%).
9. Cost explosion for many candidates → cap at 5; for >5, "qualifier round" first.
10. Section ordering → cluster ordering via topic-similarity to up-front TOC generated by haiku.

### 3.6 Test cases

```python
def test_split_markdown_h2_into_spans(): ...
def test_pairwise_verdicts_cached(): ...
def test_bt_mle_recovers_known_skills(): ...
def test_tie_handling(): ...
def test_select_within_single_cluster_picks_max(): ...
def test_assemble_concat_preserves_spans_byte_for_byte(): ...
def test_assemble_smooth_does_not_lose_facts(): ...
def test_position_bias_is_controlled(mocker): ...
def test_single_researcher_short_circuit(): ...
def test_homogeneous_team_warns(): ...
```

### 3.7 Cost

4 researchers × 3 sections = 12 spans. Per cluster: 3 pairs × 3 judges = 9 calls. 3 clusters: 27 calls × ~1.5k tokens = 40k tokens, parallel ~10–15s. Smooth assembly: 1 call ~3k tokens, 5s. **Total ~45k tokens, +15–25s, $0.10–$0.30 per swarm.** v0.1 single synthesizer ~5k tokens, 8s.

Granularity=whole: 6 pairs × 3 judges = 18 calls. ~30k tokens, +10–15s. **Recommended default.**

### 3.8 Failure-mode interactions

- **Selector ↔ Cascade guard (M1):** receives only Green-majority candidates; degenerates to passthrough on single candidate.
- **Selector ↔ Judge ensemble (M2):** share `judge_models` config; ensemble debate is iterative, selector is one-shot pairwise.
- **Selector ↔ Persona critics (M4):** filter to `role == "researcher"` before splitting into spans.

### 3.9 Latest extensions

- MoA / Self-MoA literature (Wang et al. 2024, Li et al. 2024) — the foil.
- `choix` library (lucasmaystre/choix) — standard Python BT impl.
- Counterfactual Debating with Preset Stances (COLING 2025) — handles intentionally adversarial candidates.
- Debate-to-Write (COLING 2025) — persona-driven assembly.

---

## Mechanism 4 — Persona-typed critics (MAR)

### 4.1 Paper summary

Onat Ozer et al., "MAR: Multi-Agent Reflexion" (arXiv:2512.20845, Dec 2025).

**Problem.** Single-agent Reflexion (Shinn et al. 2023) suffers *degeneration of thought*: same model self-critiquing tends to confirm its prior misconception.

**Solution.** Replace single critic with N personas (3–4) each with distinct prompt template. After failed attempt, every persona writes a diagnostic. Personas debate up to 2 rounds. Judge synthesizes into "Consensus Reflection."

**Persona templates (paper-stated):**

*HotPotQA (4 personas):*
- Verifier — checks factual correctness
- Skeptic — assumes hallucinations
- Logician — strict spec compliance
- Creative — proposes unforeseen angles

*HumanEval (3 personas):*
- Senior Engineer — clean, efficient, correct code
- QA Engineer — edge cases, input validation
- Code Reviewer — bugs, syntax, style

**Numbers:** HumanEval baseline 67.1% → Reflexion 76.4% → **MAR 82.6%** (+6.2). HotPotQA ReAct 32% → Reflexion 44% → **MAR 47%** (+3).

**Mapping to blitz-swarm:** factual / logical / counterfactual / steelman set:

- **Factual** — verifies numbers, citations, paper-claim correspondence
- **Logical** — checks internal consistency, spec compliance
- **Counterfactual** — what-if-this-were-false; finds load-bearing assumptions
- **Steelman** — defends the strongest version of dissent (round 2+ when dissent exists)

### 4.2 Concrete API

**Files:**
- `prompts/general/critic_factual.md`
- `prompts/general/critic_logical.md`
- `prompts/general/critic_counterfactual.md`
- `prompts/general/critic_steelman.md`
- `prompts/general/researcher.md`, `fact_checker.md`, `quality_judge.md`, `synthesizer.md`
- `prompts/crypto/...` (domain overlay)

**Refactor `agents.py`:**

```python
@dataclass
class PromptSet:
    role: str
    persona: str | None
    template_path: Path
    system: str
    user_template: str

class PromptLoader:
    def __init__(self, root: Path, domain: str = "general"): ...
    def load(self, role: str, persona: str | None = None) -> PromptSet: ...
    def list_personas(self, role: str) -> list[str]: ...
    def reload(self): ...

PERSONA_ASSIGNMENT = {
    1: ["factual"],
    2: ["factual", "counterfactual"],
    3: ["factual", "logical", "counterfactual"],
}

def assign_personas(critic_count: int, round_n: int,
                    has_unresolved_dissent: bool) -> list[str]:
    base = PERSONA_ASSIGNMENT[min(critic_count, 3)]
    if round_n >= 2 and has_unresolved_dissent:
        base = base[:-1] + ["steelman"]
    return base
```

### 4.3 BlitzAgent extension

```python
@dataclass
class BlitzAgent:
    id: str
    role: str
    subtopic: str
    system_prompt: str
    model: str = "sonnet"
    max_iterations: int = 3
    persona: str | None = None        # NEW
    prompt_path: str | None = None    # NEW (for tracebacks)
```

### 4.4 Counterfactual critic prompt sketch

```
You are a Counterfactual Critic in a research swarm.
Your job is to find load-bearing assumptions in the researcher findings.

Method:
1. List the 5 strongest claims in the findings.
2. For each, ask: "If this claim is wrong, what conclusions collapse?"
3. Identify the claim with the highest blast-radius if false.
4. Demand additional evidence for that claim — not the others.

Persona rules:
- Do not be merely contrarian. Pick the load-bearing claim, not nitpicks.
- Cite specific evidence the researcher *should* have provided.
- Vote 'needs_work' only if a high-blast-radius claim is unsupported.
```

### 4.5 Edge cases

1. Prompt file missing → fall back to inline ROLE_PROMPTS, log warning.
2. Persona name collision with role → don't allow `persona == "critic"`.
3. All critics duplicate factual angle → override to default 3-persona set.
4. Steelman votes always ready → skip steelman in round 1.
5. Persona drift across runs → hash-pin prompt files in metrics.jsonl.
6. Counterfactual breaks consensus permanently → pair with holdout-override after 3 rounds.
7. Domain mismatch → loader resolution: `prompts/{domain}/{role}_{persona}.md` then fallback to general.
8. Persona-specific fields → optional `load_bearing_claim_id`, default null.
9. Round 1 with no prior critique → fine; topic + findings only.
10. Cost asymmetry → cap steelman output at 800 tokens.

### 4.6 Test cases

```python
def test_prompt_loader_finds_general_critic_factual(): ...
def test_assign_personas_round1_no_dissent(): ...
def test_assign_personas_round2_with_dissent_introduces_steelman(): ...
def test_critic_outputs_carry_persona_tag(swarm_run): ...
def test_counterfactual_critic_finds_load_bearing_claim(mocker): ...
def test_persona_prompt_hash_recorded(metrics_log): ...
def test_steelman_skipped_round1(swarm_run): ...
def test_loader_fallback_to_inline_on_missing_file(tmp_path): ...
def test_domain_overlay_resolution(loader_crypto): ...
def test_persona_dissent_grouped_in_output(final_doc): ...
```

### 4.7 Cost

Same number of subprocess calls vs v0.1 (no extra invocations). Per critic call: +50 input tokens (~$0.0001). Across 3 critics × 3 rounds: +450 tokens, **negligible cost**.

Real cost is decision latency: persona-driven critique surfaces more dissent → more rounds. Avg consensus round drifts from 2.5 (v0.1) to ~3.0 (v0.2). **+25–40s wall clock per swarm.**

### 4.8 Failure-mode interactions

- **Personas ↔ Cascade guard (M1):** counterfactual claims may flag as Red against trusted lineage by design — bypass guard for critic writes; include in lineage as `verdict="red", source_role="critic_counterfactual"`.
- **Personas ↔ Judge ensemble (M2):** include `persona` in rubric criteria.
- **Personas ↔ Selection synthesizer (M3):** BT scoring includes critic-objection prior — pre-down-weight researcher spans that received `needs_work` from `critic_factual`.

### 4.9 Latest extensions

- MARPO (arXiv 2512.22832) — reflective policy optimization for MARL.
- Counterfactual Debating with Preset Stances (COLING 2025).
- Debate-to-Write (COLING 2025).
- Reflexion (Shinn et al. 2023, arXiv 2303.11366) — predecessor.

---

## Cross-mechanism integration notes

### Shared blackboard keys (extend `Blackboard.cleanup()` to delete all):
- `blackboard:lineage` — JSON-serialized Lineage Graph (M1)
- `blackboard:judge_state:{round}` — StabilityState log (M2)
- `blackboard:selection_cache` — pairwise verdict cache (M3)
- `blackboard:persona_assignments:{round}` — which personas active (M4)

### Schema additions to AGENT_OUTPUT_SCHEMA (agents.py):

```python
"persona": {"type": "string", "description": "Persona tag (critic only)."},
"parent_claim_ids": {"type": "array", "items": {"type": "string"}},
"agent_error_state": {"type": "boolean"},
"load_bearing_claim_id": {"type": "string"},
```

All optional, default null/empty/false.

### Config schema (`blitz.toml`):

```toml
[guard]
enabled = true
mode = "balanced"
yellow_release_threshold = 0.7

[judge_ensemble]
enabled = true
n_judges = 3
max_rounds = 5
ks_threshold = 0.05
ks_consecutive = 2
min_rounds = 2

[selector]
enabled = true
granularity = "section"
n_judges = 3

[personas]
enabled = true
default_set = ["factual", "logical", "counterfactual"]
domain = "general"
```

### Failure cascading hierarchy

1. Guard runs first (per-output, on write).
2. Personas drive critic round (round-by-round role assignment).
3. Judge ensemble replaces single judge.
4. Selector replaces synthesizer (one-shot at end).

Each can be disabled independently. Disable order under cost pressure: judge ensemble → selector → personas → guard.

---

## Suggested implementation order

1. **Personas (M4) first** — pure refactor, no new logic, no extra subprocess calls. Forces prompt-loading discipline. ~2 days.
2. **Cascade guard (M1) second** — most independent, biggest defensive value. Ship in `mode="speed"` first. ~4 days.
3. **Selection synthesizer (M3) third** — replaces existing component, cleanly bounded. ~3 days.
4. **Judge ensemble (M2) last** — most complex (EM, KS, mixture model), highest cost, biggest behavior shift. ~5 days.

**Total: ~14 engineer-days for v0.2.** Each independently shippable behind config flag.

---

## Risk flags

- **Cost blowup.** Naive enable of all four ~3-5× per-run cost. Hard cap via `[budget] max_cost_usd_per_run` and abort.
- **EM non-convergence in M2.** Cap iterations, log warnings, fall back to majority vote.
- **Guard over-blocking.** Mandatory `--guard-audit` flag prints every Red verdict before run can complete. Manual `--guard-override claim_id`.
- **Persona prompt rot.** SHA-pinning + persona regression suite — same input × same persona produces stable critique structure.
- **Selection homogeneity collapse.** Diagnostic flag, fall back to v0.1 synthesis when span-similarity > 0.9.
- **Cross-mechanism deadlock.** Detection: zero-input mechanism → abort with structured error.
- **Memory schema drift.** Add nullable columns, write one-shot migration script.

---

## Sources

- [From Spark to Fire (arXiv 2603.04474)](https://arxiv.org/abs/2603.04474v1)
- [Multi-Agent Debate for LLM Judges (arXiv 2510.12697)](https://arxiv.org/abs/2510.12697)
- [When Agents Disagree (arXiv 2603.20324)](https://arxiv.org/abs/2603.20324)
- [MAR: Multi-Agent Reflexion (arXiv 2512.20845)](https://arxiv.org/abs/2512.20845)
- [Autorubric (arXiv 2603.00077)](https://arxiv.org/abs/2603.00077)
- [INFA-Guard (arXiv 2601.14667)](https://arxiv.org/abs/2601.14667)
- [Reflexion (arXiv 2303.11366)](https://arxiv.org/abs/2303.11366)
- [choix Bradley-Terry library](https://pypi.org/project/choix/)
- [Stop Overvaluing Multi-Agent Debate (arXiv 2502.08788)](https://arxiv.org/pdf/2502.08788)
