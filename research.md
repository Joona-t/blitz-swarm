# Research — Blitz-Swarm v0.2 "Frontier"

**Author:** Joona Tyrninoksa
**Status:** Living document. Updated as the four parallel paper-survey agents return findings.
**Scope:** v0.2.0 — methodology-first frontier upgrade with recursive self-improvement.
**Last updated:** 2026-05-09

---

## 0. Executive summary

Blitz-Swarm v0.1.1 is a parallel multi-agent research swarm with five hand-coded role prompts, a single Claude Sonnet quality judge, a Tier-1 interaction-trace memory, and an explicit ROADMAP of unbuilt mechanisms. v0.2 closes every documented gap **and** adds a recursive self-improvement loop so the swarm proposes upgrades to itself and the bench validates them.

The upgrade lands in five phases over an estimated 5–7 weeks of background work:

| Phase | Theme | Key outputs |
|---|---|---|
| 0 — Bench | Methodology backbone | 30-prompt slate, runner, MAST regression suite, parallel-vs-sequential bench, baseline run |
| 1 — Mechanism | Frontier mechanisms | cascade_guard, judge_ensemble (KS-stop), selector_synth, persona critics, generalized prompts, preset registry |
| 2 — Memory | G-Memory Tier 2/3 | query_graph, insight_graph, GAM-style promotion gate, retrieval pipeline, distillation cron |
| 3 — Recursion | Self-improvement loop | GEPA (prompt evolution), Aflow (architecture evolution), meta-loop, cross-CLI heterogeneity, optional Managed Agents adapter |
| 4 — Methodology | Docs + paper-grade results | BIBLIOGRAPHY.md, README v0.2, n≥20 results across slate, optional short technical report |

**The Karpathy principle binds all five phases:** Phase 0 produces the metric; every later phase A/B-tests against it. The bench is fitness; recursive evolution is gradient ascent.

---

## 1. Approach: A+ Frontier (token-unconstrained methodology-first)

The original "Approach A" pitched a conservative methodology-first sequence. The user explicitly upgraded the brief to *latest, best, evolving, recursive of improvement, we are building the future*. The frontier translation:

1. **Methodology first stays.** Phase 0 still ships before any mechanism work — without a stable bench, every later A/B test is theater and the recursion loop optimizes for nothing.
2. **Mechanism set goes frontier.** Every paper at applicability ≥ 8 from the survey lands. Token budget is not a cap.
3. **Memory tier goes frontier.** G-Memory Tier 2 *and* Tier 3 ship in v0.2, not Tier 2 in v0.2 + Tier 3 in v0.3. Backed by GAM-style promotion gates so Tier 3 doesn't fill with noise.
4. **Recursion ships in v0.2.** Cross-CLI heterogeneity, GEPA, Aflow, and the meta-loop are pulled in from v0.3 / v0.4 originally proposed.
5. **Anthropic Managed Agents adapter ships as an opt-in alternative orchestrator.** Beta header `managed-agents-2026-04-01`. Default stays local-CLI for rule-#10 compliance and zero-API-cost open-source ergonomics.
6. **Models ladder up.** Default `sonnet` for researchers/critics, optional `opus-4-7` for synthesizer/judge, configurable per role.

**What "Frontier" does NOT mean:** model fine-tuning. The agent survey confirmed there's no 2025–2026 paper at applicability ≥ 6 that adds compositional reasoning to a frozen-LLM swarm without fine-tuning. NEO Neural Theorizer (KAIST) and iLTN-style neuro-symbolic hybrids are flagged as **deferred to v0.3 or beyond**, contingent on a feasibility study.

---

## 2. Latest methods catalog (May 2026)

Each row is keyed to the gap it addresses and the file it lands in. Sources verified against arXiv abstracts and ICLR/NeurIPS proceedings via web search; numeric claims are paper-stated unless explicitly re-verified.

| # | Method | Source | Year/Mo | Paper-stated effect | Lands in | Gap |
|---|---|---|---|---|---|---|
| 1 | Genealogy-graph cascade defense | Xie et al. arXiv 2603.04474 | Mar 2026 | 0.32 → 0.89 defense rate across 6 frameworks | `mechanisms/cascade_guard.py` | Cascading failures |
| 2 | Multi-judge debate + KS-stop | Hu et al. arXiv 2510.12697 (NeurIPS 2025) | Oct 2025 | KS<0.05 stops; outperforms majority voting | `mechanisms/judge_ensemble.py` | Judge calibration |
| 3 | Autorubric | arXiv 2603.00077 | Feb 2026 | RiceChem 80% with 5-shot calibration | `mechanisms/judge_ensemble.py` | Judge drift |
| 4 | Selection-bottleneck synth | Maryanskyy arXiv 2603.20324 | Mar 2026 | Diverse + judge-selection 0.81 vs MoA 0.51 | `mechanisms/selector_synth.py` | Cross-model + synth quality |
| 5 | MAR persona critics | arXiv 2512.20845 | Dec 2025 | HumanEval +6.2, HotPotQA +3 | `prompts/general/critic_*.md` | Structured debate |
| 6 | Token-Level RR collaboration | Liu et al. arXiv 2604.17139 | Apr 2026 | Defeats adversarial-majority collapse | `mechanisms/selector_synth.py` (paragraph-level) | Consensus robustness |
| 7 | Empirical comparison harness | Shen et al. arXiv 2603.29632 | Mar 2026 | Subagent-parallel vs agent-team trade-off characterized | `bench/parallel_vs_sequential.py` | No baseline |
| 8 | Scaling-architecture predictor | Google/MIT arXiv 2512.08296 | Dec 2025 | 87% optimal-architecture prediction on unseen tasks | `bench/runner.py` (architecture routing) | No baseline |
| 9 | MAST taxonomy (14 modes) | Cemri et al. arXiv 2503.13657 (NeurIPS 2025) | Mar 2025 | Catalog + κ=0.77 LLM-as-judge auto-annotator | `bench/mast_regression.py` | Cascading + spec |
| 10 | G-Memory Tier 1/2/3 | Zhang et al. arXiv 2506.07398 (NeurIPS 2025 spotlight) | Jun 2025 | +20.89% embodied, +10.12% knowledge QA | `gmemory/{query,insight}_graph.py` | Memory unbuilt |
| 11 | GAM hierarchical promotion gate | arXiv 2604.12285 | Apr 2026 | Promotion only with N≥3 confirming neighbors | `gmemory/promotion.py` | Tier 3 noise |
| 12 | A-MEM Zettelkasten links | Xu et al. arXiv 2502.12110 (NeurIPS 2025) | Feb 2025 | 2× on multi-hop reasoning | `gmemory/insight_graph.py` (link policy) | Insight quality |
| 13 | GEPA reflective prompt evolution | arXiv 2507.19457 (ICLR 2026 oral) | Jul 2025 | Beats RL +6–20%, 35× fewer rollouts than GRPO | `scripts/optimize_prompts.py` | Hand-written prompts |
| 14 | Aflow MCTS workflow synthesis | DeepWisdom (arXiv ID per subagent) | 2024+ | MCTS over agent workflows | `evolve/aflow_search.py` | Static architecture |
| 15 | AlphaEvolve LLM evolutionary search | Google DeepMind | May 2025 | LLM-driven code/algorithm evolution | `evolve/meta_loop.py` (inspiration only) | Recursive substrate |
| 16 | Salemi/Google blackboard MAS | arXiv 2510.01285 | Oct 2025 | +13–57% end-to-end task success | already foundational; refresh `BLITZ-SWARM.md` | Architectural validation |
| 17 | LbMAS token efficiency | Han & Zhang arXiv 2507.01701 | Jul 2025 | Lower token cost vs sequential | `BLITZ-SWARM.md` citation | Cost-efficiency claim |
| 18 | InfoDeepSeek RAG benchmark | arXiv 2505.15872 | May 2025 | 245 hand-curated questions, ACC/IA@k/EEU/IC | (deferred — Phase 4 web search) | Web grounding |
| 19 | Anthropic Managed Agents multiagent sessions | Anthropic announcement | May 7 2026 | Lead-agent + 20 specialists; persistent events | `managed_agents/adapter.py` (opt-in) | Alternative backend |
| 20 | Claude Opus 4.7 | Anthropic GA | May 2026 | Long-running SWE perf, higher-res vision | `blitz.toml` per-role model knobs | Frontier model |

**Papers explicitly NOT included (rationale):**

- Grounding-vs-Compositionality / iLTN (arXiv 2604.26521) — requires neural-symbolic fine-tuning. Deferred to v0.3+.
- Diversity for the Win (OpenReview ptUxbqOGrC) — +47% AIME claim flagged as benchmark-specific tail effect; Maryanskyy's selection-bottleneck story is the cleaner causal mechanism.
- DSPy / MIPROv2 — useful concept, framework weight too high for vanilla-Python blitz-swarm; GEPA standalone is the correct integration.
- AutoGen v0.4 / Microsoft Agent Framework / OpenAI Agents SDK — replaced or competing frameworks; out of scope per design philosophy ("no frameworks").
- LLMs-as-Judges Comprehensive Survey (arXiv 2412.05579) — orientation reading only; no novel mechanism.

---

## 3. Most powerful systems (deployable May 2026)

### Models

| Model | Vendor | CLI access | Role assignment in v0.2 default config |
|---|---|---|---|
| Claude Opus 4.7 | Anthropic | `claude --model opus-4-7` | optional `synthesizer`, optional `quality_judge` |
| Claude Sonnet 4.6 | Anthropic | `claude --model sonnet` | default `researcher`, `critic`, `fact_checker` |
| Claude Haiku 4.5 | Anthropic | `claude --model haiku` | retrieval relevance scoring + insight extraction (cheap, high-volume) |
| GPT-5.x via Codex | OpenAI | `codex exec` | optional heterogeneous `researcher` slot |
| Gemini 2.5 Pro | Google | `gemini -p` | optional heterogeneous `fact_checker` (free Google grounding) |

### Frameworks (intentionally avoided in core)

- Microsoft Agent Framework (MAF) — successor to AutoGen v0.4, enterprise-flavored.
- OpenAI Agents SDK — replaced Swarm March 2026; OpenAI-API-coupled.
- LangGraph / CrewAI / MetaGPT — adds dependency surface for no benefit at our scale.

### Self-improvement substrates (integrated)

- **GEPA** (gepa-ai/gepa) — reflective prompt evolution, used standalone.
- **Aflow** — MCTS workflow generation, ported standalone.
- **Anthropic Managed Agents** (May 7 beta) — opt-in adapter only; default stays CLI.

### Rule-#10 compliance

Every model used in the **core v0.2 build** is reachable via Joona's existing local CLI subscriptions (Claude Pro/Max, ChatGPT Plus, Gemini, etc.). The Managed Agents adapter is opt-in and explicitly flagged as API-cost-bearing.

---

## 4. Recursive self-improvement design

### The loop

```
                     ┌──────────────────────────────────────┐
                     │ Bench (Phase 0): fitness function    │
                     │   - 30-prompt slate                  │
                     │   - quality_judge ensemble (Phase 1) │
                     │   - paired t-tests, bootstrap CIs    │
                     └────────────┬─────────────────────────┘
                                  │ scores
            ┌─────────────────────┼─────────────────────────┐
            ▼                     ▼                         ▼
      GEPA (prompts)        Aflow (architecture)       Manual A/B
      ─────────────        ──────────────────────      ───────────
      Evolves              Mutates: add/remove        Joona-driven
      role prompts         agents, debate rounds,      one-off
      against bench        topology                    experiments
            │                     │                         │
            └─────────────────────┴─────────────────────────┘
                                  │ candidates
                                  ▼
                     ┌──────────────────────────────────────┐
                     │ Bench validation                     │
                     │   - Run candidate on slate           │
                     │   - Compare to baseline              │
                     │   - Cohen's d ≥ 0.3, paired-t p<0.05 │
                     └────────────┬─────────────────────────┘
                                  │ wins
                                  ▼
                     ┌──────────────────────────────────────┐
                     │ Insight graph (Phase 2 Tier 3)       │
                     │   - Tag winning configs with         │
                     │     `meta:` namespace                │
                     │   - Distill cross-task insights      │
                     └────────────┬─────────────────────────┘
                                  │ meta-insights
                                  ▼
                     ┌──────────────────────────────────────┐
                     │ Meta-loop (Phase 3)                  │
                     │   - Reads `meta:` insights           │
                     │   - Proposes config edits            │
                     │   - Auto-merges if guards pass       │
                     └────────────┬─────────────────────────┘
                                  │
                                  └─→ next generation
```

### Recursion bounds (safety)

- **Level 0** — base swarm produces research output for Joona's actual queries.
- **Level 1** — GEPA + Aflow propose mutations; bench validates.
- **Level 2** — meta-loop proposes mutations to *the evolution loops themselves* (e.g., GEPA's K iterations, Aflow's MCTS depth).
- **Level 3 — humans audit.** No autonomous changes beyond Level 2. Meta-meta-loops require explicit `--allow-l3` flag plus written confirmation.

### Auto-merge guards

A candidate config replaces baseline only if **all** of:
- Cohen's d ≥ 0.3 on the bench slate aggregate quality
- No dimension regression > 5% (coverage, accuracy, clarity, depth)
- MAST regression suite passes ≥ 12/14 modes
- Total cost ≤ 1.5× baseline cost
- Paired t-test p < 0.05

Otherwise: candidate is logged to `evolve/candidates.jsonl` for human review, never auto-merged.

### Token-budget safety

Despite "token spenditure is not an issue":
- `evolve.budget_per_generation_usd = 50` default cap (configurable)
- `evolve.max_concurrent_evals = 4` to prevent rate-limit storms
- Each generation logs cumulative cost; if cap hit, generation halts and waits for human ack

---

## 5. Per-phase research

> Each subsection below has a **DEEP DIVE** placeholder that gets filled with the corresponding background subagent's output when it returns. The placeholders explicitly say which subagent is responsible.

### 5.1 Phase 0 — Bench harness

**Subagent in flight:** Phase 0 bench harness deep-dive (background).

**What we already know:**
- 30-prompt slate covering easy / medium / hard / adversarial / multi-domain / compositional.
- Each prompt tagged with parallelizable-vs-sequential class per Google scaling paper 2512.08296.
- Runner with parallel execution, rate limits, JSONL log, resume capability.
- MAST regression: 14 named tests, each injects a failure mode and asserts containment.
- Parallel-vs-sequential per Shen 2603.29632 methodology.
- Cost cap, reproducibility schema, paired t-tests, Cohen's d, bootstrap CIs.

**Open questions for the deep dive:**
- Should the 30-prompt slate be MIT-released as a separate `blitz-swarm-bench` repo to seed a community benchmark?
- What's the cost of one full slate run on Sonnet 4.6 vs Opus 4.7?
- How do we estimate per-prompt time budgets without running first?

**DEEP DIVE: [pending — subagent integrating]**

### 5.2 Phase 1 — Mechanism upgrades

**Subagent in flight:** Phase 1 mechanism deep-dive (background).

**What we already know:**
- 4 mechanisms target ≥ 8 applicability papers from §2.
- Cascade guard adds `parent_message_ids` chain to blackboard writes.
- Judge ensemble runs N=3 judges with Beta-Binomial mixture and KS-test stop.
- Selector synth replaces blended synthesis with judge-driven span selection.
- Persona critics: 3-4 templates rotated across critic slots.
- Prompts move from inline strings in `agents.py` to `prompts/general/*.md` and `prompts/crypto/*.md` with a preset registry.

**DEEP DIVE: [pending — subagent integrating]**

### 5.3 Phase 2 — G-Memory Tier 2/3

**Subagent in flight:** Phase 2 G-Memory deep-dive (background).

**What we already know (from `Memory architecture for a parallel AI agent swarm.md` and `Building Hierarchicl agent Memory from G-Memory's blueprint.md`):**
- Tier 2 = query graph (task nodes, kNN cosine ≥ 0.7, k=5, 1-hop expansion).
- Tier 3 = insight graph with hyperedges connecting insights to validating queries.
- GAM promotion: insight only created if N ≥ 3 query-graph neighbors share pattern.
- Retrieval pipeline: embed → kNN → 1-hop expand → upward traversal → LLM relevance score → LLM sparsify → inject as `## Relevant prior findings`.
- LanceDB for vector ANN, SQLite + WAL for graph.
- Daily distillation cron; per-task lightweight extraction.
- Meta-tags: insights tagged `meta:` are queried by the recursion loop.

**DEEP DIVE: [pending — subagent integrating]**

### 5.4 Phase 3 — Recursive self-improvement

**Subagent in flight:** Phase 3 GEPA + Aflow + meta-loop deep-dive (background).

**What we already know:**
- GEPA evolves prompts against bench. Standalone library; no DSPy dependency.
- Aflow does MCTS over architectural mutations (add/remove agents, debate rounds, topology).
- Meta-loop reads `meta:` insights from Tier 3 and proposes config edits.
- Cross-CLI heterogeneity: `claude` / `codex` / `gemini` routing per Maryanskyy's selection-bottleneck thesis.
- Managed Agents adapter optional (beta header `managed-agents-2026-04-01`).
- Recursion bounded at Level 3.
- Auto-merge guards: Cohen's d ≥ 0.3, no dim regression > 5%, MAST ≥ 12/14, cost ≤ 1.5×, p < 0.05.

**DEEP DIVE: [pending — subagent integrating]**

### 5.5 Phase 4 — Methodology + docs

**No external subagent — synthesized from paper survey.**

- BIBLIOGRAPHY.md: 13 new + 9 existing citations, all with arXiv links and one-line annotations.
- README v0.2: replaces "n=2 anecdote" disclaimer with n ≥ 20 results table, parallel-vs-sequential bench chart, MAST regression scoreboard.
- Optional short technical report (3-5 page PDF via `make-pdf` skill) summarizing v0.2 findings for archival.
- BLITZ-SWARM.md updated to reflect built (not proposed) Tier 2/3 + recursion loop.

---

## 6. Bibliography (target state for `docs/BIBLIOGRAPHY.md`)

Full bibliography with arXiv links and one-line annotations lives in `docs/BIBLIOGRAPHY.md` once Phase 4 completes. Below is the citation key list:

### Multi-agent architecture
- Salemi et al. 2025 — Google blackboard MAS [arXiv 2510.01285]
- Han & Zhang 2025 — LbMAS [arXiv 2507.01701]
- Qian et al. 2025 — MacNet [ICLR 2025]
- Wu et al. 2025 — Memory in MAS survey [TechRxiv]
- Sagirova et al. 2025 — SRMT [arXiv 2501.13200]

### Mechanism upgrades
- Xie et al. 2026 — Genealogy-graph cascade defense [arXiv 2603.04474]
- Cemri et al. 2025 — MAST taxonomy [arXiv 2503.13657, NeurIPS 2025]
- Hu et al. 2025 — Multi-judge debate + KS-stop [arXiv 2510.12697, NeurIPS 2025]
- Autorubric 2026 [arXiv 2603.00077]
- Maryanskyy 2026 — Selection-bottleneck [arXiv 2603.20324]
- MAR 2025 — Multi-Agent Reflexion [arXiv 2512.20845]
- Liu et al. 2026 — Token-Level RR [arXiv 2604.17139]
- Du et al. 2024 — Multi-agent debate [ICML 2024]

### Memory
- Zhang et al. 2025 — G-Memory [arXiv 2506.07398, NeurIPS 2025 spotlight]
- Xu et al. 2025 — A-MEM [arXiv 2502.12110, NeurIPS 2025]
- GAM 2026 [arXiv 2604.12285]
- AgeMem 2026 [arXiv 2601.01885]
- MAGMA 2026 [arXiv 2601.03236]

### Recursive self-improvement
- GEPA 2025 — Reflective prompt evolution [arXiv 2507.19457, ICLR 2026 oral]
- Aflow — MCTS workflow synthesis [DeepWisdom — verify arXiv ID via subagent]
- AlphaEvolve 2025 — Google DeepMind LLM evolutionary search [Google blog]

### Methodology
- Shen et al. 2026 — Empirical comparison harness [arXiv 2603.29632]
- Google scaling 2025 — Architecture predictor [arXiv 2512.08296]
- InfoDeepSeek 2025 — RAG benchmark [arXiv 2505.15872]

### Frontier models / systems
- Anthropic Managed Agents — multiagent sessions public beta (May 7 2026)
- Claude Opus 4.7 GA (May 2026)

---

## 7. Open questions

1. **No purpose-built "research synthesis swarm" benchmark exists.** GAIA / AgentBench / Gaia2 skew toward task-completion. v0.2 should ship a 30-prompt slate as a candidate community benchmark and invite contributions.
2. **Cross-CLI heterogeneity has no public empirical data.** All cross-model MAS papers test API-based heterogeneity. Joona is positioned to publish first measurements of CLI-subprocess heterogeneity.
3. **Insight-graph distillation cadence unstudied.** G-Memory and GAM describe distillation but don't answer "how often". v0.2 instruments this and reports the cost-vs-freshness curve.
4. **Judge drift across model versions.** Sonnet 4.5 → 4.6 → ... shifts the 0–10 distribution. v0.2 logs raw judge scores per model version to surface this empirically; Phase 4 includes a short note in the technical report.
5. **Compositional reasoning with frozen LLMs.** No 2025–2026 paper at applicability ≥ 6 shows how to inject neural-symbolic compositional structure into an LLM-only synthesizer without fine-tuning. NEO Neural Theorizer and iLTN both require model training. **Deferred.**
6. **Recursion stability.** No prior work demonstrates a 3-level recursive self-improvement MAS with empirical safety bounds. v0.2 ships Level 2 + manual gate at Level 3; Level 3+ is open research.

---

## 8. Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-09 | Approach A+ (frontier methodology-first) selected | User upgraded scope to "latest, best, evolving, recursive, future-building". Methodology-first preserved because recursion needs fitness signal. |
| 2026-05-09 | Generalize prompts; crypto becomes a preset | User answered "Generalize + presets". Matches BLITZ-SWARM.md framing; allows future presets (`code`, `science`, etc.). |
| 2026-05-09 | Default backend = local CLI; Managed Agents = opt-in adapter | Rule #10 (no paid API in core); Managed Agents is May 7 public beta and API-bearing. |
| 2026-05-09 | Recursion bounded at Level 2; Level 3+ requires explicit flag | Safety bound on autonomous self-modification. |
| 2026-05-09 | NEO / neuro-symbolic compositional reasoning deferred | No applicable paper without fine-tuning; out of scope for v0.2. |
| 2026-05-09 | Tier 2 *and* Tier 3 ship in v0.2 (not split across versions) | Token-unconstrained budget; user wants the future, not incrementalism. |
| 2026-05-09 | v0.1.1 committed as separate release before v0.2 work begins | Preserves WIP (crypto specialization + retry loop + metrics) on the main timeline. |

---

## 9. Provenance

- **Background subagents:** four parallel paper-survey + implementation deep-dive agents launched 2026-05-09 to populate §5 deep-dives.
- **Initial paper survey:** 2026-05-09, ~2,400-word filtered survey covering 13 high-applicability papers across 9 of 10 documented gaps.
- **Existing in-repo literature reviews:**
  - `Memory architecture for a parallel AI agent swarm.md` — blackboard + storage stack + retention.
  - `Building Hierarchicl agent Memory from G-Memory's blueprint.md` — G-Memory three-tier algorithms in Python.
- **In-repo research-grade docs:** `docs/METHODOLOGY.md`, `docs/RESEARCH_QUESTIONS.md`, `docs/RESEARCH_LOG.md`, `docs/IMPLEMENTATION_STATUS.md`, `docs/LIMITATIONS.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/EVALUATION.md`, `docs/ROADMAP.md`, `docs/ABLATIONS.md`, `docs/ARTIFACTS.md`.

This research document is the **source of truth** for v0.2 planning. `plan.md` is derivative — it operationalizes what's here.
