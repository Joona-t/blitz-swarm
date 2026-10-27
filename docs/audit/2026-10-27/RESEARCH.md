# What the literature says about the design choices in Blitz-Swarm

Research brief, 2026-10-27. Source tags: **[PR]** peer-reviewed venue · **[P]** arXiv preprint · **[V]** vendor or practitioner write-up.

Every arXiv ID listed in the sources was opened on arxiv.org during this research. I re-checked 2604.02460, 2605.00914 and 2610.30769 a second time. Numbers are taken from abstracts and official pages, not reproduced. Items marked *(from memory)* were not opened this session.

---

## 1. Does a multi-agent system beat a strong single agent at equal compute?

**Mostly no, once you control for compute.**

- Debate does not reliably beat self-consistency or ensembling, and it is sensitive to hyperparameters (Smit et al. [PR, ICML 2024]).
- A single agent with a strong prompt nearly matches the best discussion method (Wang et al. [PR, ACL 2024]).
- At a matched number of responses, debate does not beat self-consistency (Huang et al. [PR, ICLR 2024]).
- Majority voting explains most of the gains attributed to debate. The paper also proves debate is a martingale: it can't raise expected correctness unless the updates are biased toward the right answer (Choi et al. [PR, NeurIPS 2025]).
- At equal thinking-token budgets, a single agent matches or beats five multi-agent architectures on multi-hop QA. Some budget controls inflate the apparent multi-agent advantage (Tran & Kiela 2604.02460 [P]).
- Simple baselines such as repeated sampling often sit on the cost/accuracy Pareto frontier ahead of complex agents (Kapoor et al. [PR, TMLR]; HAL [PR, ICLR 2026]).

**Where the sources conflict:**

- Anthropic's multi-agent research system beat single-agent Opus by 90.2% on an internal eval [V]. That comparison was **not compute-matched**: token usage alone explained 80% of the variance on BrowseComp, and the system used about 15× the tokens of chat. Its gain also came from parallel *tool-using search*, which Blitz-Swarm doesn't have.
- Across 260 configurations, the effect of a multi-agent setup ranges from +81% on decomposable tasks to −70% on sequential tasks. Centralized verification contains errors better than independent agents (Kim et al. 2512.08296 [P]).
- MAST finds multi-agent gains are often minimal. Failures cluster in system design, inter-agent misalignment and verification (Cemri et al. [PR, NeurIPS 2025 D&B]).

**Untested.** Nobody has published a compute-matched comparison on *long-form research synthesis without tools*, which is exactly Blitz-Swarm's task. **Strength:** moderate to strong for "debate ≈ voting at equal compute"; weak for synthesis specifically.

## 2. Conformity, herding, and stopping when everyone agrees

**The sources agree that agents drift toward agreement, and that this costs correct answers.**

- Models switch from correct to incorrect answers after reading their peers' reasoning (Wynn et al. [PR workshop, ICML 2025]).
- In homogeneous teams, up to 85.5% of agents adopt the majority answer. Peer rationales destabilize up to 70% of correct answers. Plurality voting discards correct answers that were already in the pool, an "oracle gap" of up to 32.3 points. Debate cost 2.1–3.4× the tokens of isolated self-correction, for equal or lower accuracy (Bertalanič & Fortuna 2605.00914 [P]).
- Voting beats consensus on reasoning tasks by 13.2%. Consensus wins on knowledge tasks by only 2.8%. More discussion rounds before the vote reduce performance (Kaesberg et al. [PR, ACL Findings 2025]).
- LLM populations herd more than humans and correct cascades less often (Hashemi et al. 2610.30769 [P, Oct 2026]).
- A single injected error can cascade, and once a consensus forms it resists correction (Xie et al. 2603.04474 [P]).

**Implication for Blitz-Swarm's loop.** The unanimous-"ready" stop rule plus a holdout override is the configuration these results warn about. It selects for agreement, not correctness. No paper tests unanimity-stopping exactly, so this is an inference from a consistent mechanism, not a direct result.

**Diversity.**

- Mixing model families helps more than running high-temperature copies of one model (ReConcile [PR, ACL 2024]).
- A diverse team scored 0.810 against 0.512 (near chance) for a homogeneous team (Maryanskyy et al. 2603.20324 [Applied Sciences 2026]).
- Persona prompts alone don't improve objective performance (Zheng et al., EMNLP Findings 2024 *(from memory)*).
- In homogeneous teams, more initial diversity actually *increased* conformity (2605.00914).

**Strength:** moderate. The results are consistent, but mostly come from small models on multiple-choice and math tasks.

## 3. How reliable are LLM judges?

**Well-replicated biases:**

- Position bias, verbosity bias and self-enhancement bias (Zheng et al. [PR, NeurIPS 2023 D&B]; Wang et al. "Not Fair Evaluators", ACL 2024 *(from memory)*).
- Self-preference, which grows with the model's ability to recognize its own text (Panickssery et al. [PR, NeurIPS 2024]).
- Preference leakage: judges favor generators from their own family (Li et al. [PR, ICLR 2026]).
- Agreement with humans varies by task, so judges must be validated per task (Bavaresco et al., JUDGE-BENCH [PR, ACL 2025]).

**Panels.** A panel of judges from *disjoint model families* tracks humans better than one large judge, and costs about 7× less (Verga et al. 2404.18796 [P]). The benefit comes from family diversity. Three Sonnet calls don't buy it.

**Adaptive stopping.** Hu et al. 2510.12697 [P] exists. It describes debating judges, a Beta-Binomial mixture model and KS-based stopping, and reports gains over majority voting. It is a single preprint with no independent replication. It is also in tension with Choi and Kaesberg, who find little gain from debate over voting.

**Conflicting practitioner advice.** Anthropic [V] found a single rubric-based judge call the most consistent and the closest to human judgment. Autorubric (Rao & Callison-Burch 2603.00077 [P]) finds no mitigation configuration that works for every criterion.

## 4. Grounding and fabricated citations (the strongest evidence in this brief)

- **Parametric-only citation is unreliable.**
  - 55% of GPT-3.5 references and 18% of GPT-4 references were fabricated. Of the real ones, 24–43% contained substantive errors (Walters & Wilder [PR, Sci Rep 2023]).
  - GPT-4o fabricated 78–90% of research citations. A retrieval-augmented 8B model cited about as accurately as human experts (OpenScholar, Asai et al. [PR, Nature 2026]).
- **Retrieval helps but doesn't solve the problem.**
  - 50–90% of LLM responses are not fully supported by their own cited sources. Even GPT-4o with web search left about 30% of statements unsupported (SourceCheckup, Wu et al. [PR, Nat Commun 2025]).
  - 3–13% of URLs produced by search-augmented and deep-research agents never existed. A liveness check against the live web and the Wayback Machine cut non-resolving URLs to under 1% (Rao et al. 2604.03173 [P]).
- **Self-correction without external feedback doesn't improve reasoning** (Huang et al. [PR, ICLR 2024]). A fact-checker agent with no tools can catch internal inconsistencies, but it can't tell whether a paper exists.

**Implication.** Blitz-Swarm's researcher prompts require specific papers and measured numbers while forbidding tools. That is the regime with the worst measured fabrication rates.

## 5. Evaluating changes with small samples

- Treat eval prompts as a sample from a population. Use paired differences on the same prompts, clustered standard errors and an up-front power analysis (Miller 2411.00640 [P, Anthropic]). For significance-testing practice, see Dror et al., ACL 2018 *(from memory)*.
- Seed-to-seed variance can swamp small differences (Madaan et al. 2406.10229 [P]). Run multiple seeds.
- About 20 real queries were enough to detect *large* effects early in development [V, Anthropic]. Small effects need many more.
- Report cost and accuracy together, as a Pareto frontier over repeated runs (Kapoor; HAL).
- When the judge shares the generator's model family, results are biased in the generator's favor (Li et al.; Panickssery et al.).

**Untested.** I found no published guidance on the minimum n for comparing long-form research reports.

## 6. Citation check on the repo's own bibliography

All 11 cited arXiv IDs exist.

- **AFlow (2410.10762)** is misattributed to "Liu". The first author is Jiayi Zhang.
- **Maryanskyy (2603.20324)** supports *selection* over synthesis. In that paper, synthesis was never preferred over the baseline.
- **MAR (2512.20845)** is multi-agent Reflexion with persona debaters. "Persona critics" is a loose description of it.
- **Shen (2603.29632)** studies agents doing automated ML optimization, not literature synthesis. It transfers only weakly to this use.
- The other 7 match how the repo describes them.

---

## Design implications, ranked by strength of evidence

1. **Ground the research.** Give researchers retrieval. Check every reference by deterministic DOI, arXiv or URL lookup, and check each claim against its source. This is the strongest and most consistent evidence in the brief. (§4)
2. **Compare against equal-compute single-agent baselines before adding mechanisms.** If the swarm doesn't beat best-of-N on a cost-vs-quality plot, the extra machinery isn't paying for itself. (§1)
3. **Replace unanimity with independent, blinded scoring and a fixed 1–2 revision rounds.** Log answers that flip from correct to incorrect, and check that correct minority findings survive into the final output. (§2)
4. **Diversify across model families, not personas,** especially for judging. Keep the judge out of the generator's family. (§2, §3)
5. **Select first, then patch,** rather than freely re-synthesizing. (§2, Maryanskyy)
6. **Debias pairwise judging:** score both orderings, control for length, use per-criterion rubrics, and calibrate on 30–50 human-labeled items. (§3)
7. **Treat KS adaptive halting as experimental.** It rests on a single preprint, and at N=3 it is mathematically inert (see AUDIT M1). (§3)
8. **Optimize GEPA/AFlow only against a held-out judge from a different model family plus grounded metrics.** Otherwise they optimize toward pleasing Sonnet. (§3, §5)
9. **Keep centralized verification, and test it by fault injection:** seed a fabricated claim and measure how often it reaches the final output. (§1 Kim; §2 Xie)
10. **Use paired prompts, run a power analysis, run ≥3 seeds, report bootstrap CIs and cost per run, and spot-check with humans.** (§5)

---

## Sources

Peer-reviewed unless tagged.

- Smit et al., *Should we be going MAD?*, ICML 2024 — https://proceedings.mlr.press/v235/smit24a.html
- Wang et al., *Rethinking the Bounds of LLM Reasoning*, ACL 2024 — https://aclanthology.org/2024.acl-long.331
- Huang et al., *LLMs Cannot Self-Correct Reasoning Yet*, ICLR 2024 — https://arxiv.org/abs/2310.01798
- Choi, Zhu, Li, *Debate or Vote*, NeurIPS 2025 — https://arxiv.org/abs/2508.17536
- [P] Tran & Kiela — https://arxiv.org/abs/2604.02460
- [P] Kim et al. — https://arxiv.org/abs/2512.08296
- Cemri et al., MAST, NeurIPS 2025 D&B — https://arxiv.org/abs/2503.13657
- Kapoor et al., *AI Agents That Matter* — https://arxiv.org/abs/2407.01502
- HAL, ICLR 2026 — https://arxiv.org/abs/2510.11977
- [V] Anthropic, *How we built our multi-agent research system* — https://www.anthropic.com/engineering/multi-agent-research-system
- [V] Cognition, *Don't Build Multi-Agents* — https://cognition.com/blog/dont-build-multi-agents
- Wynn, Satija, Hadfield (ICML 2025 workshop) — https://arxiv.org/abs/2509.05396
- [P] Bertalanič & Fortuna — https://arxiv.org/abs/2605.00914
- Kaesberg et al., ACL Findings 2025 — https://arxiv.org/abs/2502.19130
- [P] Hashemi et al. — https://arxiv.org/abs/2610.30769
- [P] Xie et al. — https://arxiv.org/abs/2603.04474
- Chen, Saha, Bansal, ReConcile, ACL 2024 — https://arxiv.org/abs/2309.13007
- Maryanskyy et al., Applied Sciences 2026 — https://arxiv.org/abs/2603.20324
- Zheng et al., *Judging LLM-as-a-Judge*, NeurIPS 2023 D&B — https://arxiv.org/abs/2306.05685
- Panickssery et al., NeurIPS 2024 — https://arxiv.org/abs/2404.13076
- Li et al., *Preference Leakage*, ICLR 2026 — https://arxiv.org/abs/2502.01534
- [P] Verga et al., *Replacing Judges with Juries* — https://arxiv.org/abs/2404.18796
- Bavaresco et al., JUDGE-BENCH, ACL 2025 — https://arxiv.org/abs/2406.18403
- [P] Hu et al. — https://arxiv.org/abs/2510.12697
- [P] Rao & Callison-Burch, Autorubric — https://arxiv.org/abs/2603.00077
- Walters & Wilder, Sci Rep 2023 — https://doi.org/10.1038/s41598-023-41032-5
- Asai et al., OpenScholar, Nature 2026 — https://doi.org/10.1038/s41586-025-10072-4
- Wu et al., SourceCheckup, Nat Commun 2025 — https://arxiv.org/abs/2402.02008
- [P] Rao, Wong, Callison-Burch — https://arxiv.org/abs/2604.03173
- [P] DeepResearch Bench — https://arxiv.org/abs/2506.11763
- [P] Miller, *Adding Error Bars to Evals* — https://arxiv.org/abs/2411.00640
- [P] Madaan et al. — https://arxiv.org/abs/2406.10229
