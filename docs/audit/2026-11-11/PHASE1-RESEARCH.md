# Phase 1 research: the measuring stick

2026-11-11. Research for proposal Phase 1 ([../2026-10-27/PROPOSAL.md](../2026-10-27/PROPOSAL.md), revised by [../2026-11-04/FRONTIER-RESEARCH.md](../2026-11-04/FRONTIER-RESEARCH.md)). **Not implementing yet.** Evidence labels: **[measured]** from this repo's own run data, **[read]** from the code, **[lit]** from the two research briefs (citations there).

## The question Phase 1 answers

Does the swarm produce a more accurate research report than one agent given the same generation budget? Nothing in the repo can answer that today, for three reasons found below.

## 1. What the existing bench can and cannot tell us

**The seam is already there.** `bench/runner.py` takes an injected `swarm_fn(text, max_rounds=…, use_redis=…) -> Path | (Path, rounds)` (`runner.py:124`, `:410`, `:862`). A baseline arm is just another function with that signature; no runner redesign is needed. Pairing against an earlier run already works (`--baseline`, exercised live on 2026-11-04). [read]

**Problem 1: the primary metric rewards citations that merely look right.** `research_quality_composite` = 0.45 coverage + 0.30 citations + 0.15 novelty + 0.10 low-dedup (`detectors.py:257`). The citation term is `wellformed_fraction × min(1, distinct/3)`: an arXiv ID that *matches the ID format* and a URL that parses score full marks. Nothing checks that the paper exists. In the slate run the swarm, which has no tools and writes from memory, scored **0.90 on citations** on average. [measured, `bench/runs/20261104T212924_4e57c4496ae2_v1`] The literature says tool-less models fabricate 18–90% of references. [lit] So 30% of the primary score is currently free credit for confident formatting, and it will favour whichever arm cites most fluently.

**Problem 2: coverage is keyword recall.** `expected_coverage` is 4–6 phrases per prompt; the score is the fraction that appear. It rewards mentioning a topic, not getting it right, and longer outputs win by volume. The swarm's outputs are long (median **84k output tokens** per prompt across all agents; final documents of 140–320 lines). [measured] Length must be reported next to every score.

**Problem 3: the LLM-judge score is the swarm's own judge.** `quality.avg` comes from the swarm's quality judge inside the run. A single-agent arm has no judge, so the column would be empty for baselines, and where it exists it is a participant grading its own team. It cannot be the comparison metric.

**What the existing data gives us for calibration.** [measured, n=10, one configuration]
- Swarm arm, profile `max`, 2 rounds: mean 476 s and $3.17 reported cost per prompt (Claude-side only; Codex is uncosted), 2.2M input tokens per prompt, of which almost all is per-call CLI overhead (LIMITATIONS 20).
- Output tokens per prompt: median 84,172, range 33,739–149,336. This is the only compute number that reflects *generation work*, so it is the one to match across arms.
- Same-prompt rerun noise (n=2 pairs, 2026-11-04): quality Δ +0.1 / −0.4, composite Δ ≈ 0.002, cost −16% / +11%.

## 2. Arms: what "equal compute" can mean here

Agent calls are stateless subprocesses (`max_turns=1`, ephemeral). A "single agent with a bigger budget" therefore has to be a chain of calls where each call receives the previous draft. That is a fair model of one agent revising its own work, and it is what Wu et al. (EMNLP Findings 2026) found beats wide best-of-N at small budgets for open-ended writing. [lit]

| Arm | Construction | Calls | Budget rule |
|---|---|---|---|
| **B0** | one researcher call | 1 | none (floor) |
| **B1** | draft → self-critique → revise, repeated | 3–9 | stop when cumulative **output tokens** reach the swarm's median for that tier |
| **B2** | N independent drafts, cross-family judge picks one (pairwise, both orders) | N + judge | N = swarm researcher count for the prompt |
| **B3** | N independent drafts, one merge call that must keep every distinct claim | N + 1 | same N |
| **S** | the swarm as shipped on `next/v0.4` | ~20–40 | as configured |

Why output tokens and not dollars or input tokens: dollars are unknown for Codex and notional on a subscription; input tokens are ~98% harness overhead that scales with *call count*, which would make any multi-call arm look expensive for reasons unrelated to thinking. Calls, wall-clock and reported cost are still recorded for every arm so a cost-vs-quality plot can be drawn on any axis. [measured for the overhead; design choice for the rule]

B3 is the arm Liu (2610.27488) found beats a 3-agent team on balanced accuracy; without it a swarm win could be "N samples win", not "interaction wins". [lit]

All generation in every arm uses the **same provider and tier** (the default provider's `standard` tier). Only the judge is a different family.

## 3. Metrics that an arm cannot game by writing fluently

**M1. Citation resolution (deterministic, no LLM).** Extract arXiv IDs, DOIs and URLs. Resolve against public, key-less endpoints: the arXiv export API (`id_list`), Crossref (`/works/<doi>`), and for URLs an HTTP HEAD with a Wayback availability fallback. Classify each as `resolved` / `resolved-but-title-mismatch` / `not-found` / `unverifiable` (network error). Score = resolved-and-matching ÷ cited, plus the raw count. This directly replaces the format-only citation term. Rao et al. report 3–13% dead URLs even from search-backed agents and a 6–79× reduction from a liveness check. [lit]
- Sends only the cited identifiers to arXiv/Crossref/archive.org. No user data, no key, free. Results cached on disk so reruns are offline and reproducible.
- **Needs a known-bad fixture** (studio ledger: "checks that cannot fail"): a report with one real arXiv ID, one fabricated ID in valid format, one real DOI with a wrong title. The test passes only if the three are classified differently.

**M2. Blind pairwise preference vs B1 (cross-family judge).** For each prompt the judge sees two anonymised reports and a per-criterion rubric (accuracy, completeness against the prompt, specificity, internal consistency) and returns a winner per criterion. Each pair is judged in **both orders**; a pair that flips with order counts as a tie. Length of both reports is logged. The judge runs on a provider pinned by `[models] judge_provider`, different from the generator family (self-preference and preference leakage are well replicated [lit]). Output: win/tie/loss per arm vs B1 and a Bradley-Terry score across arms (the BT code exists in `mechanisms/selector_synth.py` and matches scipy [audit]).
- Zhou's result: a judge that commits to its own answer first cuts false positives from 0.72 to 0.01. [lit] For long-form reports the equivalent is cheap: the judge first lists the 5 facts it would expect in a correct answer, *then* reads the two reports. Included as a rubric step.

**M3. Coverage, unchanged, but reported with length** and as coverage per 1,000 output characters.

**M4. Co-failure on coverage items (β).** For multi-draft arms: the fraction of `expected_coverage` items that **no** draft mentions. If β for N researchers is close to a single draft's miss rate, fan-out cannot help coverage on that prompt (Chen J., co-failure ceiling). [lit] Computable from data the runner already keeps.

**M5. Cost block per arm**: calls, output tokens, input tokens, wall-clock, reported cost with its policy.

**Deferred to Phase 2**: claim-level support against fetched source text. It needs retrieval to be meaningful; checking claims against abstracts alone would mostly return "not in abstract".

## 4. Calibration against a human

A judge metric nobody has checked against a person is the "proxy metric" failure again. Bavaresco et al.: judge agreement varies by task and must be validated per task. [lit] Plan: a terminal tool shows Joona 30 anonymised report pairs (drawn across arms and tiers), one at a time, asks which is better per criterion, and writes `labels.jsonl`. Agreement with the judge is reported as Cohen's κ with a bootstrap CI. Below ~0.6 the rubric is revised before any judged result is trusted; the threshold is a convention. Estimated 2–3 hours of reading for 30 pairs of long reports; it can be done in sittings.

## 5. Sample size and what it costs in usage

Power for a paired design at α=0.05, 80% (computed 2026-10-27 with scipy): n=20 detects d_z ≥ 0.66; n=34 detects 0.5. Same-prompt noise on the composite was tiny in the one rerun pair we have, which suggests seeds matter less than prompts for the composite; for judged preference it is unknown.

Measured cost of the swarm arm is ~8 min and ~84k output tokens per prompt at profile `max`. Baseline arms are far cheaper: B0 is one call (~1–2 min); B1 at the matched budget is by construction about the swarm's generation work but in 3–9 calls instead of 20–40, so much less harness overhead. Rough totals for **20 prompts, 1 seed**: S ≈ 2.7 h of wall-clock (10 of the 20 already exist from 2026-11-04 on unchanged swarm code), B0–B3 together ≈ 2–3 h, judging ≈ 20 prompts × 4 arm-pairs × 2 orders = 160 short judge calls. These are estimates from one configuration, not measurements of the baseline arms.

The binding constraint is the weekly subscription limit, not dollars. The slate run on 2026-11-04 consumed 22M input tokens for 10 prompts.

## 6. Risks

- **Prompt slate is small and partly author-written with the swarm in mind.** 30 prompts, 15 easy / 10 medium / 5 hard. Results generalise to this slate only.
- **Coverage keywords were written before any arm ran**, but by the same author who built the swarm; they may fit its section style.
- **The judge is still an LLM.** Cross-family reduces self-preference; it does not make the judge right. M1 is the only fully objective metric.
- **Network dependence for M1.** Mitigated by the cache and by an `unverifiable` class that is excluded from the ratio and reported.
- **Budget matching by output tokens** undercounts thinking tokens the CLIs do not report. Recorded as a limitation; the calls and wall-clock columns make it visible.
