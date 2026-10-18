# Bibliography — Blitz-Swarm v0.3

All citations grouped by what they affect. Titles were re-fetched from `arxiv.org/abs/<id>` on 2026-10-11 and replaced where the earlier short label did not match the paper (audit research-docs-05). Anchors that drive shipped code are bolded; "(exp.)" marks anchors whose module is experimental/unwired in v0.3.0.

---

## Multi-agent architecture (foundational)

- **Salemi et al. 2025 — "LLM-Based Multi-Agent Blackboard System for Information Discovery in Data Science."** [arXiv 2510.01285](https://arxiv.org/abs/2510.01285). +13–57 % relative improvement on end-to-end task success with a shared blackboard [paper-stated]. The pattern Blitz-Swarm's `blackboard.py` builds on.
- **Han & Zhang 2025 — "Exploring Advanced LLM Multi-Agent Systems Based on Blackboard Architecture."** [arXiv 2507.01701](https://arxiv.org/abs/2507.01701). Blackboard coordination uses fewer tokens than alternative MAS approaches at competitive quality [paper-stated].
- Qian et al. 2025 — MacNet. [ICLR 2025](https://openreview.net/forum?id=K3n5jPkrU6). Scales MAS to 1,000+ agents via final-artifact-only message passing.
- Wu et al. 2025 — survey on memory in LLM-based multi-agent systems. *No verified link* — the earlier entry carried a placeholder TechRxiv URL, now removed. Cited for the hybrid-memory-topology finding; treat as unverified until a DOI is found.
- Sagirova et al. 2025 — "SRMT: Shared Memory for Multi-agent Lifelong Pathfinding." [arXiv 2501.13200](https://arxiv.org/abs/2501.13200). Shared-memory cross-attention enables emergent coordination without explicit protocols.

## Mechanisms (`mechanisms/`, `prompts/`)

- **Xie et al. 2026 — "From Spark to Fire: Modeling and Mitigating Error Cascades in LLM-Based Multi-Agent Collaboration."** [arXiv 2603.04474](https://arxiv.org/abs/2603.04474). Genealogy-graph cascade defense; defense success 0.32 → 0.89 across six MAS frameworks [paper-stated]. `mechanisms/cascade_guard.py` (heuristic screen by default; LLM hooks in `max` mode).
- **Cemri et al. 2025 — "Why Do Multi-Agent LLM Systems Fail?"** [arXiv 2503.13657](https://arxiv.org/abs/2503.13657). NeurIPS 2025. MAST taxonomy, 14 failure modes. Drives `bench/detectors.py` + `bench/mast_regression.py` (9/14 detected on synthetic scenarios).
- **Hu et al. 2025 — "Multi-Agent Debate for LLM Judges with Adaptive Stability Detection."** [arXiv 2510.12697](https://arxiv.org/abs/2510.12697). NeurIPS 2025. Beta-Binomial mixture + KS halt. `mechanisms/judge_ensemble.py` uses an empirical-CDF KS instead of the parametric mixture (documented deviation, `PHASE_1_MECHANISMS.md` §2).
- Rao & Callison-Burch 2026 — "Autorubric: A Unifying Framework for Rubric-Based LLM Evaluation on Non-Verifiable Tasks." [arXiv 2603.00077](https://arxiv.org/abs/2603.00077). Rubric API in `judge_ensemble`.
- **Maryanskyy et al. 2026 — "When Agents Disagree: The Selection Bottleneck in Multi-Agent LLM Pipelines."** [arXiv 2603.20324](https://arxiv.org/abs/2603.20324). Diverse generation + judge selection 0.81 win rate vs MoA-style synthesis 0.51 [paper-stated]. `mechanisms/selector_synth.py`.
- **Ozer et al. 2025 — "MAR: Multi-Agent Reflexion Improves Reasoning Abilities in LLMs."** [arXiv 2512.20845](https://arxiv.org/abs/2512.20845). Persona-typed critics; HumanEval +6.2, HotPotQA +3 [paper-stated]. `prompts/general/critic_{factual,logical,counterfactual,steelman}.md`, `persona_critics` flag.
- Liu et al. 2026 — "The Consensus Trap: Rescuing Multi-Agent LLMs from Adversarial Majorities via Token-Level Collaboration." [arXiv 2604.17139](https://arxiv.org/abs/2604.17139). (Earlier label "Token-Level Round-Robin" was the mechanism, not the title.) Fallback path in `selector_synth` for split votes.
- Du et al. 2024 — "Improving Factuality and Reasoning in Language Models through Multiagent Debate." [arXiv 2305.14325](https://arxiv.org/abs/2305.14325). ICML 2024. Foundational; operationally superseded by MAR + Hu 2510.12697.

## Memory (`gmemory/`)

- **Zhang et al. 2025 — "G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems."** [arXiv 2506.07398](https://arxiv.org/abs/2506.07398). NeurIPS 2025. Three-tier graph hierarchy (interaction / query / insight); +20.89 % embodied action, +10.12 % knowledge QA [paper-stated, not replicated here]. `gmemory/store.py` and friends — wired in v0.3.0.
- Xu et al. 2025 — "A-MEM: Agentic Memory for LLM Agents." [arXiv 2502.12110](https://arxiv.org/abs/2502.12110). Zettelkasten-style link evolution; reference for distillation prompts.
- **Wu et al. 2026 — "GAM: Hierarchical Graph-based Agentic Memory for LLM Agents."** [arXiv 2604.12285](https://arxiv.org/abs/2604.12285). LLM-discrimination at session boundaries; adapted to the N=3 distinct-query gate in `gmemory/promotion.py` (documented deviation).
- Yu et al. 2026 — "Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents." [arXiv 2601.01885](https://arxiv.org/abs/2601.01885). (Earlier label "AgeMem".) RL-trained memory ops; reference for future learned eviction.
- Jiang et al. 2026 — "MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents." [arXiv 2601.03236](https://arxiv.org/abs/2601.03236). Reference for future Tier-3 axis expansion.

## Recursive self-improvement (`evolve/`, exp.)

- **Agrawal et al. 2025 — "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning."** (exp.) [arXiv 2507.19457](https://arxiv.org/abs/2507.19457). [ICLR 2026 oral](https://openreview.net/forum?id=RQm2KQTM5r). Library [gepa-ai/gepa](https://github.com/gepa-ai/gepa). `evolve/gepa_adapter.py`, `scripts/optimize_prompts.py` — gated by `[evolve] enabled`.
- **Zhang et al. 2024 — "AFlow: Automating Agentic Workflow Generation."** (exp.) [arXiv 2410.10762](https://arxiv.org/abs/2410.10762). [ICLR 2025 oral](https://openreview.net/forum?id=z5uVAKwmjf). `evolve/aflow_search.py` — gated.
- Zhao et al. 2025 — "A²Flow: Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators." [arXiv 2511.20693](https://arxiv.org/abs/2511.20693). Not implemented.
- AlphaEvolve 2025 — [DeepMind blog](https://deepmind.google/blog/alphaevolve/). Inspirational reference for the meta-loop's exploration ceiling.
- "Diversity for the Win" — [OpenReview ptUxbqOGrC](https://openreview.net/forum?id=ptUxbqOGrC). Heterogeneous LLM compositions; treated with caution per Maryanskyy's selection-bottleneck story.

## Methodology (`bench/`)

- **Shen et al. 2026 — "An Empirical Study of Multi-Agent Collaboration for Automated Research."** [arXiv 2603.29632](https://arxiv.org/abs/2603.29632). Subagent-parallel vs agent-team paradigms under fixed compute. The parallel-vs-sequential harness it motivates (`bench/parallel_vs_sequential.py` in the Phase 0 design) **was never written**; `bench/runner.py` is `swarm_fn`-injectable so it can be.
- **Cemri et al. 2025** (above) — `bench/mast_regression.py`.
- Kim et al. 2025 — "Towards a Science of Scaling Agent Systems." [arXiv 2512.08296](https://arxiv.org/abs/2512.08296). (Earlier label "Google scaling — architecture predictor".) 87 % optimal-architecture prediction on unseen tasks [paper-stated]. Drives the slate's `parallelizable` annotation.
- Xi et al. 2025 — "InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation." [arXiv 2505.15872](https://arxiv.org/abs/2505.15872). Reference for future web-grounded retrieval.

## Frontier systems referenced

- Anthropic Managed Agents — multiagent sessions + Outcomes (May 2026 public beta, header `managed-agents-2026-04-01`). `contrib/managed_agents/adapter.py` — paid API, opt-in, call not implemented.
- Claude Code CLI model aliases (`fable`, `opus`, `sonnet`, `haiku`) — the strings `blitz.toml` maps tiers to for the `claude` provider. `fable` = Fable 5.1 (`claude-fable-5-1`), the Mythos default.
- Microsoft Agent Framework / OpenAI Agents SDK — intentionally not used (framework weight; API coupling).

## Not implemented (deferred since v0.2)

- NEO Neural Theorizer (KAIST 2026) — compositional reasoning; requires fine-tuning.
- Shahid & Rothe 2026 — "Grounding vs. Compositionality: On the Non-Complementarity of Reasoning in Neuro-Symbolic Systems." [arXiv 2604.26521](https://arxiv.org/abs/2604.26521). (The earlier entry labelled this id "iLTN"; that is not the paper's title.) Out of scope.
- DSPy / MIPROv2 — GEPA standalone was the chosen cut.

---

## Citation key

- **Bold** = drives shipped code (module, gate, schema). (exp.) = module exists but is gated/unwired.
- Plain = context, design rationale or verification.
- Numeric claims are paper-stated unless a deep-dive doc marks them re-verified. None of them has been replicated in this repo (`docs/CLAIMS_AND_EVIDENCE.md` row 35).
