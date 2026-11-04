# Frontier scan: what's new since the 27 Oct audit, and what transfers

2026-11-04. Research only; **not implementing yet.** Covers roughly Aug–Nov 2026 plus earlier work the 27 Oct brief missed. Nothing from that brief is repeated. Tags: **[PR]** peer-reviewed · **[P]** preprint · **[V]** vendor. Every arXiv ID below was opened on arxiv.org by the scan; I re-opened 2608.06940, 2608.17994 and 2608.20274 myself and the numbers match. Items marked *(snippet)* were seen only in search results.

## Headline

**No paper in this window overturns "debate ≈ voting at equal compute."** Three new compute-matched studies (Dylan 2609.04217, Liu 2610.27488, Jwalapuram 2606.13003) all land on the same side; the one claimed win (Cao 2610.36848) is on GPU kernels with an execution oracle. The real movement is in *measurement* and *gating*: when a panel can help at all, which items to spend verification on, when to stop, and what memory granularity transfers. Those are the breakthroughs worth borrowing, and several are directly implementable in blitz-swarm because the vote logs already exist.

## The five transfers, ranked by evidence × applicability

### 1. Spend verification only where the vote is pivotal; measure the co-failure ceiling first
- **Shu, *Blind to the Pivotal Vote*, 2608.06940 [P, Aug 2026].** An external signal (test execution) left "effective votes" unchanged (−0.04, CI [−0.10, +0.02]) yet gave **+10.4 to +23.3 pp on one-vote-split queries** and zero elsewhere. A majority-side replacement rule raised HumanEval+/MBPP+ from 82.44% → 85.62% while calling the signal on **16.2%** of queries.
- **Chen J., *Co-Failure Ceiling*, 2606.27288 [P].** Any route/vote/MoA that returns one member's answer is capped at 1−β, β = probability all members fail together. Across 67 models / 21 providers: β ≈ 0.05 (math), 0.08 (code), 0.13 (GPQA). Pairwise correlation ρ cannot identify β. Combining rarely beats the best single model without a query-level signal.
- **Kohli, *Nine Judges, Two Effective Votes*, 2605.29800 [P].** 9 frontier judges from 7 families ≈ 2.18 effective votes; best single judge matched or beat the panel in every condition.
- **Jha, *Decorrelation Is Not Complementarity*, 2608.16190 [P].** Cross-lineage panels no better at matched skill (p=0.13); skill predicts agreement (ρ=0.84). Counterpoint: Teng C²-MAS 2609.38274 [P] claims complementarity-aware team selection helps (no numbers in abstract).

**For blitz-swarm.** This is three mutually consistent results with numbers and it needs nothing new: the orchestrator already has per-item votes. (a) On the Phase-1 eval set, estimate β (all-researchers-wrong rate) with a Clopper–Pearson interval; if β is near the single-agent error rate, the fan-out cannot pay. (b) Route the expensive step (retrieval fact-check, `deep`-tier re-research) **only to one-vote-margin items**, not every item every round. (c) It also revises the 27 Oct advice "diversify judge families": do it *by measured skill on your task*, not by lineage.

### 2. Judge → "accept / retrieve-and-rejudge / abstain" with a stated error rate, after deterministic gates
- **Badshah, Emami, Sajjad, *Judge, Retrieve, or Abstain*, 2608.17994 [PR, COLM 2026].** Thresholds calibrated on a held-out set (Clopper–Pearson) so FDR among accepted verdicts ≤ α; uncertain items go to a retrieval-augmented judge with a second calibrated threshold; guarantee holds across the two-stage routing.
- **Zhou, *More Convincing, Not More Correct*, 2607.05904 [P].** Self-play against a reference-free judge drove pass rate 0.72 → 0.94 with accuracy flat at 0.20; a three-family ensemble still accepted 55% of exploits; **having the judge commit to its own answer before seeing the candidate cut false-positive rate 0.719 → 0.012.**
- **Afoud, VReAct, 2609.31937 [P, pilot n=10].** 8 of 9 corrections came from zero-cost deterministic gates; adding a planner *hurt* once verification existed.
- **Aly, 2609.30328 [P, WiML@NeurIPS 2026].** A multi-agent code judge was grounded only when evidence was independent of the answer and differed between candidates; gating on that raised accuracy 20.7% → 36.9% while answering half the items.
- **Dastidar, 2607.17044 [P].** In a production agent, swapping the small verifier for the generating frontier model removed most rescues: verifier ≠ generator.

**For blitz-swarm.** Replace `quality_vote ∈ {ready, needs_work}` with a three-way verdict at a declared α; order the pipeline as deterministic gates (schema valid, citation resolves, quoted span found in source) → blind-solve-then-compare for checkable sub-claims → LLM judge → retrieval re-judge → abstain. Abstentions go to the Dissent section instead of forcing a vote. The verifier model must differ from the researcher model.

### 3. Anytime-valid stopping instead of KS halting; return the best round, not the last
- **Ota, CITE, 2605.05873 [PR, NeurIPS 2026 per RIKEN AIP list].** Certifies the *mode* of a discrete answer distribution under arbitrary data-driven stopping, rate independent of answer-set size. Certifies agreement, not correctness.
- **Yin, *When Is Enough Enough*, 2610.04756 [P, Oct 2026].** Anytime-valid restart detector on paired per-item outcomes + change-point selection of an earlier artifact: stopped at round 4 of 40, **−91.6% tokens**, 82.43% vs 82.00%.
- **Liu Y., EMS, 2604.02863 [P].** Order judges by reliability, stop when the lead is mathematically safe: −35% agents, −44% tokens, accuracy preserved on five benchmarks.
- **Shrivastava, 2606.27009 [P, n=60].** Embedding-convergence stopper −38% tokens at parity, but the oracle best-round was +0.115 above every policy: the open problem is *which round to return*.
- **Morandi, SPRT governor, 2605.19193 [P].** Worked on GSM8K (3.7× fewer calls), collapsed on MMLU (hit the cap on 99.5% of items at 2.1× cost). Any sequential governor needs a calibration-divergence check.
- **Wu, *Ladders in Chaos*, 2608.28496 [PR, Findings EMNLP 2026].** For translation (closest to open-ended writing), a short sequential refinement chain beats wide best-of-N at small budgets; quality can fall at large budgets.

**For blitz-swarm.** Drop the KS rule (inert at N=3, AUDIT M1). Use EMS-style safe-lead stopping on the judge ensemble, CITE for any discrete verdict, and Yin's change-point rule for GEPA runs. Keep every round's artifact and select the return round by a held-out score, not recency. For the synthesizer, a 2–3-step refinement chain with a depth cap rather than wide best-of-N.

### 4. Memory: subtask-level text skills with a utility score; task-level skills are net negative
- **Feng et al., *Break It Down, Pass It On*, 2608.20274 [P, v2 Oct 2026].** Task-level skills push agents **below the no-memory baseline**; subtask-level skills improve on average; text skills transfer better than code; a utility score = specificity × abstractness predicts transfer from descriptions alone.
- **Wang, OEP, 2605.18930 [P].** Reflective memory over-generalises clean edge cases into risk-averse rules; black-box poisoning ASR > 50% on GPT-4o agents; beats LLM-audit defences. *(snippet)* MemSecBench 2607.27080, revocation study 2609.08258.

**For blitz-swarm.** G-Memory Tier 3 currently distils whole-task insights, which is the granularity shown to hurt. Store subtask-level text skills, compute the utility score before retrieval, keep provenance on every write, and require a transfer test (does the skill help on a held-out related task?) before an insight becomes a rule. This also answers the 27 Oct "test the one memory claim that matters" item with a concrete acceptance test.

### 5. Instrument the synthesizer; resolve disagreement by crux, not by debate
- **Hirsch et al., *Who is the Agent to Blame?*, 2608.24306 [PR, EMNLP 2026 main].** Per-invocation faithfulness checks in deep-research agents: **84.7% of final-report errors originate at the orchestrator/synthesis step** (31% hallucination, rest citation); two simple interventions gave +5% citation recall.
- **Jiang, *Collaborative Disagreement Resolution*, 2607.01251 [PR, ICML 2026].** Agents locate the crux and present evidence instead of arguing sides: non-expert judge accuracy 62.1% vs 49.2% for standard debate.
- **Chen Y., ColMAD, 2510.20963 [PR, NeurIPS 2026].** Competitive debate up to −15 pp vs single agent; collaborative variant fixes it.
- **Carro, 2610.28539 [P, Oct 2026]** and **Elasky, 2605.27483 [P]:** debate improves *human* judges; weaker LLM judges gain nothing or lose; one independent critique recovers most of any benefit; rebuttal rounds add nothing.
- **Fadnavis, 2605.29116 [P]:** an aggregator that reads full traces recovers correct answers even when all agents agree.
- **He, Minority Sentinel, 2606.29270 [P, SIGIR workshop]:** ~1 in 4 divergent cases hides a correct minority; a LightGBM meta-classifier over vote-log "fingerprints" flips with 81.2% precision; an LLM-judge override had negative net gain.

**For blitz-swarm.** The synthesizer is the dominant error source and the least instrumented stage: run the citation resolver and claim-support check on *its* output, not only on researchers'. Replace multi-round critic/researcher back-and-forth with one independent critique + a crux-isolation step the judge resolves with evidence. Feed the synthesizer researcher reasoning and evidence, not just conclusions. Train a small non-LLM classifier on vote logs to flag likely-correct minorities (cheap, needs the Phase-1 data).

## Other results worth recording

- **Compute-matched negatives.** Dylan 2609.04217 [P]: planner–executor–critic over one frozen 7B, equal calls: ALFWorld 0.769 vs 0.754 (p=0.80), 1.8× calls; planner and critic prompts evolved to empty. Liu 2610.27488 [P, 300 memos, 4 models]: a 3-agent team raised recall +0.08–0.10 but also false alarms; balanced accuracy Δ ≤ 0.02 at 2.2–2.4× tokens; **single agent ×3 with OR beat the team**; showing agents the requirements they weren't assigned reduced false alarms → give each researcher the full scope, not just its slice. Jwalapuram 2606.13003 [P]: AFlow-style auto-generated MAS lost to CoT-SC at up to 10× cost → direct evidence against `evolve/aflow_search.py`.
- **Reward hacking in optimisation loops.** Afrin 2608.08002 [P]: common-mode judge error is not identifiable from judge disagreement; needs an external anchor. Kenton (DeepMind) 2608.17776 [P]: a 150-word critique cap prevented judge hacking in RLAIF. Nguyen 2609.02889 [P]: harness-evolution value sat in one slot (reflection/control, +0.119); splitting the budget over four slots left all empty → run GEPA on one prompt slot at a time. *(snippet)* HackProbe 2609.04665, "Not an Oracle" 2609.02246 (judge agreement 80% on tuning subset vs 52% on full suite).
- **Verification in MAS generally.** Venkataramani, MAS-ProVe [PR, ICML 2026]: process verification "does not consistently improve performance" across six frameworks; LLM judges > reward models; verification-tuned judges > general ones.
- **Byzantine framing.** Lee & Karimireddy 2605.09076 [PR, EMNLP 2026]: filter-and-refine with (F+1)-robustness; criticises confidence-weighted BFT for trusting self-reported confidence → cascade_guard must not weight by agents' own `confidence`.
- **Protocols.** Governance-gaps paper 2606.31498 *(snippet)*: MCP/A2A/ACP/ANP cover none of voting, dissent preservation, escalation or audit. Nothing changes the harness plan in [HARNESS-RESEARCH.md](HARNESS-RESEARCH.md).
- **Parallels.** Scalable Delphi, Lorenz 2602.08889 *(snippet)*: anonymous estimates → moderator rationale summary → revision, avoiding the persuasion effects Carro and Chen report; this is the mechanism-level fix for blitz's open blackboard. Peer-review RCT, Kim 2609.19420 [ICML 2026, n>17k]: LLM-use policy had near-zero effect on decisions; 22.5% of "banned" reviewers used LLMs anyway → audit compliance rather than legislate.

## Looks like hype or too thin to act on

- Evolving Agent Teams 2610.36848: one domain, execution oracle, no text transfer.
- JudgePanel 2608.29168 ("14B beats 70B"): single preprint, no code.
- C²-MAS complementarity selection: contradicted at matched skill by Jha; no numbers.
- CP-WBFT confidence-weighted consensus: relies on self-reported confidence.
- Any AFlow-class auto-generated workflow claim.
- Debate-as-oversight for LLM judges: three Oct 2026 results say it helps humans, not LLM judges.
- SPRT governors without a calibration check.

## What was searched and not found

No compute-matched multi-agent win on open-ended research writing with human evaluation; no NeurIPS 2026 main-track list beyond RIKEN AIP's own; no ICLR 2027 submissions public; no Anthropic/OpenAI/DeepMind engineering post on multi-agent research systems in Aug–Nov 2026.

## How this changes the 27 Oct proposal

| 27 Oct item | Revision |
|---|---|
| Phase 1 baselines B0–B2 | Add **B3 = single agent ×N with OR/union** (Liu). Add β and n_eff estimates to the report. |
| Phase 2 grounding | Resolver and claim-support run on the **synthesizer output too** (Hirsch). Fact-check routed to pivotal items first (Shu). |
| Phase 3 "independent blinded scoring" | Becomes **gates → blind-solve → judge → retrieve → abstain at α** (Badshah, Zhou, VReAct); one critique + crux step instead of rounds (Jiang, Elasky). |
| Phase 3 "KS as diagnostic" | **EMS safe-lead + CITE**; select return round by held-out score (Ota, Liu Y., Shrivastava). |
| Phase 4 GEPA/AFlow | **Drop AFlow** (Jwalapuram). GEPA: one slot at a time, 150-word critique cap, change-point stop (Nguyen, Kenton, Yin). |
| Phase 4 memory | Subtask-level text skills + utility score + transfer test (Feng, Wang). |
| "Diversify judge families" | **Diversify by measured skill**, lineage is not enough (Jha, Kohli). |
| Mythos planner | Treat as an ablation candidate: two results show planners hurt once verification exists (VReAct, Dylan). |

## Citations

- Dylan https://arxiv.org/abs/2609.04217 · Liu P. https://arxiv.org/abs/2610.27488 · Cao https://arxiv.org/abs/2610.36848 · Jwalapuram https://arxiv.org/abs/2606.13003 · Chen Y. https://arxiv.org/abs/2510.20963 (NeurIPS 2026 per https://aip.riken.jp/news/neurips2026/)
- Badshah https://arxiv.org/abs/2608.17994 · Jiang https://arxiv.org/abs/2607.01251 · Elasky https://arxiv.org/abs/2605.27483 · Carro https://arxiv.org/abs/2610.28539 · Kenton https://arxiv.org/abs/2608.17776 · Aly https://arxiv.org/abs/2609.30328 · Afoud https://arxiv.org/abs/2609.31937 · Dastidar https://arxiv.org/abs/2607.17044 · Hirsch https://arxiv.org/abs/2608.24306 · Zheng https://arxiv.org/abs/2609.22223 · Zhou https://arxiv.org/abs/2607.05904 · Qian https://arxiv.org/abs/2608.29168
- Ota https://arxiv.org/abs/2605.05873 · Yin https://arxiv.org/abs/2610.04756 · Morandi https://arxiv.org/abs/2605.19193 · Liu Y. https://arxiv.org/abs/2604.02863 · Shrivastava https://arxiv.org/abs/2606.27009 · Wu https://arxiv.org/abs/2608.28496 · Fadnavis https://arxiv.org/abs/2605.29116 · He https://arxiv.org/abs/2606.29270
- Feng https://arxiv.org/abs/2608.20274 · Wang https://arxiv.org/abs/2605.18930
- Nguyen https://arxiv.org/abs/2609.02889 · Afrin https://arxiv.org/abs/2608.08002 · Wahi https://arxiv.org/abs/2609.25848
- Chen J. https://arxiv.org/abs/2606.27288 · Kohli https://arxiv.org/abs/2605.29800 · Shu https://arxiv.org/abs/2608.06940 · Jha https://arxiv.org/abs/2608.16190 · Teng https://arxiv.org/abs/2609.38274 · Zhu https://arxiv.org/abs/2608.19802 · Lee https://arxiv.org/abs/2605.09076 · Venkataramani https://arxiv.org/abs/2602.03053
- Lorenz https://arxiv.org/abs/2602.08889 · Kim https://arxiv.org/abs/2609.19420 · Governance gaps https://arxiv.org/abs/2606.31498 · MCP vs A2A https://arxiv.org/abs/2607.23884
- Snippet-only: 2609.04665, 2609.02246, 2607.27080, 2609.08258, 2608.12337
