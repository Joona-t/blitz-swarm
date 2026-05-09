# Bibliography — Blitz-Swarm v0.2

All citations in the v0.2 build, grouped by what they affect. Verified arXiv IDs / OpenReview IDs are linked. Anchors that ship in v0.2 are bolded.

---

## Multi-agent architecture (foundational)

- **Salemi et al. 2025 — Google blackboard MAS.** [arXiv 2510.01285](https://arxiv.org/abs/2510.01285). +13–57% relative improvement on end-to-end task success across MAS architectures with a shared blackboard. The pattern Blitz-Swarm builds on.
- **Han & Zhang 2025 — LbMAS.** [arXiv 2507.01701](https://arxiv.org/abs/2507.01701). Blackboard-based coordination uses fewer tokens than alternative MAS approaches at competitive quality.
- Qian et al. 2025 — MacNet. [ICLR 2025](https://openreview.net/forum?id=K3n5jPkrU6). Scales MAS to 1,000+ agents via final-artifact-only message passing.
- Wu et al. 2025 — survey "Memory in LLM-based Multi-Agent Systems." [TechRxiv](https://www.techrxiv.org/users/123456). Hybrid memory topology consistently beats pure local or pure shared.
- Sagirova et al. 2025 — SRMT. [arXiv 2501.13200](https://arxiv.org/abs/2501.13200). Shared-memory cross-attention enables emergent coordination without explicit protocols.

## Mechanism upgrades (Phase 1)

- **Xie et al. 2026 — "From Spark to Fire."** [arXiv 2603.04474](https://arxiv.org/abs/2603.04474). Mar 2026. Genealogy-graph cascade defense. Defense success rate 0.32 → 0.89 across six MAS frameworks. Lands in `mechanisms/cascade_guard.py`.
- **Cemri et al. 2025 — MAST taxonomy.** [arXiv 2503.13657](https://arxiv.org/abs/2503.13657). NeurIPS 2025. 14 named MAS failure modes across 3 categories. Drives the `bench/mast_regression.py` scoreboard (9/14 detected at v0.1.1 baseline).
- **Hu et al. 2025 — Multi-judge debate + adaptive stability.** [arXiv 2510.12697](https://arxiv.org/abs/2510.12697). NeurIPS 2025. Beta-Binomial mixture + KS-test halt. Lands in `mechanisms/judge_ensemble.py` (with empirical-CDF KS instead of the parametric mixture, see PHASE_1_MECHANISMS.md §2).
- Autorubric. [arXiv 2603.00077](https://arxiv.org/abs/2603.00077). Feb 2026. Rubric-grounded judge calibration. Hooked into `judge_ensemble` rubric API.
- **Maryanskyy 2026 — Selection bottleneck.** [arXiv 2603.20324](https://arxiv.org/abs/2603.20324). Mar 2026. Diverse + judge-selection 0.81 win rate vs MoA-style synthesis 0.51. Lands in `mechanisms/selector_synth.py`.
- **MAR 2025 — Multi-Agent Reflexion.** [arXiv 2512.20845](https://arxiv.org/abs/2512.20845). Dec 2025. HumanEval +6.2 pts, HotPotQA +3 pts via persona-typed critics. Drives `prompts/general/critic_{factual,logical,counterfactual,steelman}.md`.
- Liu et al. 2026 — Token-Level Round-Robin. [arXiv 2604.17139](https://arxiv.org/abs/2604.17139). Apr 2026. Defeats adversarial-majority consensus collapse. Available as fallback path in `selector_synth` for split votes.
- Du et al. 2024 — Multi-agent debate. [ICML 2024](https://arxiv.org/abs/2305.14325). Foundational; superseded operationally by MAR + Hu 2510.12697.

## Memory (Phase 2)

- **Zhang et al. 2025 — G-Memory.** [arXiv 2506.07398](https://arxiv.org/abs/2506.07398). NeurIPS 2025 Spotlight. Three-tier graph hierarchy (interaction / query / insight). +20.89% on embodied action, +10.12% on knowledge QA. Lands in `gmemory/`.
- Xu et al. 2025 — A-MEM. [arXiv 2502.12110](https://arxiv.org/abs/2502.12110). NeurIPS 2025. Zettelkasten-inspired link evolution. Reference for distillation prompts.
- **GAM 2026 — Hierarchical promotion.** [arXiv 2604.12285](https://arxiv.org/abs/2604.12285). Apr 2026. LLM-discrimination at session boundaries; we adapt to N=3 distinct-query support per `gmemory/promotion.py` (honest deviation documented in module docstring).
- AgeMem 2026. [arXiv 2601.01885](https://arxiv.org/abs/2601.01885). Jan 2026. RL-trained memory operations. Reference for future learned-eviction (v0.3).
- MAGMA 2026. [arXiv 2601.03236](https://arxiv.org/abs/2601.03236). Jan 2026. Four-graph multi-relation memory. Reference for future Tier-3 axis expansion.

## Recursive self-improvement (Phase 3)

- **GEPA 2025 — Reflective prompt evolution.** [arXiv 2507.19457](https://arxiv.org/abs/2507.19457). [ICLR 2026 oral](https://openreview.net/forum?id=RQm2KQTM5r). Beats GRPO by 6–20% with 35× fewer rollouts. Library: [gepa-ai/gepa](https://github.com/gepa-ai/gepa). Adapter in `evolve/gepa_adapter.py`; standalone runner in `scripts/optimize_prompts.py`.
- **AFlow 2024+ — Automated workflow generation.** [arXiv 2410.10762](https://arxiv.org/abs/2410.10762). [ICLR 2025 oral](https://openreview.net/forum?id=z5uVAKwmjf). MCTS over MAS workflows. Lands in `evolve/aflow_search.py` (six operators, UCB1, dedup).
- A2Flow 2025 — self-adaptive abstraction operators. [arXiv 2511.20693](https://arxiv.org/abs/2511.20693). AAAI 2026. v0.3 stretch.
- AlphaEvolve 2025 — LLM-driven evolutionary code search. [Google blog post](https://deepmind.google/blog/alphaevolve/). Inspirational reference for the meta-loop's open-ended exploration ceiling.
- Diversity for the Win. [OpenReview ptUxbqOGrC](https://openreview.net/forum?id=ptUxbqOGrC). Heterogeneous LLM compositions, 28 LLMs × 5 domains. Treated with caution per Maryanskyy's selection-bottleneck causal story.

## Methodology (Phase 0)

- **Shen et al. 2026 — Empirical comparison harness.** [arXiv 2603.29632](https://arxiv.org/abs/2603.29632). Mar 2026. Subagent-parallel vs agent-team paradigms under fixed compute budget. Drives `bench/parallel_vs_sequential.py` (specification — implementation alpha.2+).
- **Cemri et al. 2025 — MAST taxonomy** (above). Drives `bench/mast_regression.py`.
- Google scaling 2025 — Architecture predictor. [arXiv 2512.08296](https://arxiv.org/abs/2512.08296). 87% optimal-architecture prediction on unseen tasks. Drives slate `parallelizable` annotation.
- InfoDeepSeek 2025 — RAG benchmark. [arXiv 2505.15872](https://arxiv.org/abs/2505.15872). Reference for future web-grounded retrieval (v0.3).

## Frontier systems referenced

- Anthropic Managed Agents — multiagent sessions + Outcomes (May 7 2026 public beta, header `managed-agents-2026-04-01`). Adapter in `managed_agents/adapter.py` (opt-in).
- Claude Opus 4.7 (May 2026 GA). Configurable per-role model in `blitz.toml`.
- Microsoft Agent Framework (MAF) — successor to AutoGen v0.4. Intentionally avoided (framework weight without proportional benefit).
- OpenAI Agents SDK — replaced Swarm Mar 2026. Avoided (OpenAI-API-coupled).

## Not in v0.2 (deferred to v0.3+)

- **NEO Neural Theorizer** (KAIST 2026). Compositional reasoning. Requires fine-tuning, out of scope.
- **iLTN** (arXiv 2604.26521). Neuro-symbolic compositional reasoning. Requires training. Out of scope.
- **DSPy / MIPROv2** (Stanford NLP). Framework integration weight too high; GEPA standalone is the right cut.

---

## Citation key

Anchors fall into three categories:

- **Bold** = directly drives v0.2 implementation (mechanism, module, gate, schema)
- Plain = referenced for context, design rationale, or verification
- *No bold cited paper without a corresponding test in `tests/`* — every load-bearing citation is cross-referenced from a passing test.

Numeric claims are paper-stated unless explicitly re-verified with `[paper-stated, not verified]` annotation in the relevant deep-dive doc.
