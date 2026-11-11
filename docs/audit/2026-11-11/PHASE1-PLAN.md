# Phase 1 plan: baseline arms and metrics that can lose

2026-11-11. From [PHASE1-RESEARCH.md](PHASE1-RESEARCH.md). **Waits for Joona's annotations and "implement it all".** Branch: continue on `next/v0.4`. No paid APIs: generation and judging go through the installed `claude` / `codex` CLIs; citation checks use key-less public endpoints. Every WP ends in a test that fails before and passes after; every new check ships a known-bad fixture.

## Data flow

```
slate prompt ──► arm_fn(prompt) ──► report.md + calls.jsonl (tokens, wall, cost per call)
                                         │
              ┌──────────────────────────┼──────────────────────────────┐
              ▼                          ▼                              ▼
   M1 citation resolver       M3/M4 coverage, length, β      M2 pairwise judge vs B1
   (cache: bench/cache/)      (existing detectors)           (both orders, pinned provider)
              └──────────────────────────┴──────────────────────────────┘
                                         ▼
                    bench/runs/<id>/{results.jsonl, summary.json, stats.md}
                                         ▼
                    bench/compare.py ──► comparison.md (paired stats + cost/quality table)
```

## Work packages

| WP | Scope | Files | Done when |
|---|---|---|---|
| **1. Arm interface** | `bench/arms.py`: `ARMS = {"S", "B0", "B1", "B2", "B3"}`, each an async fn with the `swarm_fn` signature returning `(Path, rounds)`. A shared `_call()` wraps `backend.call` and appends one row per call to `<run_dir>/calls/<prompt>.jsonl` (role, tokens, elapsed, cost, cost_policy, provider). `--arm` flag on `bench.runner`; run id gains the arm. | `bench/arms.py`, `bench/runner.py` | `tests/bench/test_arms.py`: with a fake backend each arm makes the expected number of calls, writes a report, and the call log sums match `results.jsonl`. `--arm S` reproduces today's behaviour byte-for-byte on a fake swarm. |
| **2. B0, B1** | B0: one researcher-prompt call. B1: draft → critique → revise loop; stops when cumulative output tokens ≥ `--output-token-budget` (default: per-tier medians read from a named S run's `results.jsonl`), min 3 calls, max 9. Each call gets the previous draft and critique in the prompt. | `bench/arms.py` | Fake backend returning fixed token counts: B1 stops at the first call that crosses the budget; never fewer than 3 or more than 9 calls. |
| **3. B2, B3** | N independent drafts (N from the heuristic planner's researcher count for that prompt). B2: BT selection over whole drafts via the cross-family judge, both orders. B3: one merge call with a "keep every distinct claim, do not summarise away" instruction. Drafts are kept on disk for M4. | `bench/arms.py` | Fake judge preferring a marked draft: B2 returns it. B3 output contains a unique sentinel sentence from each draft. |
| **4. M1 citation resolver** | `bench/citations.py`: extract arXiv IDs / DOIs / URLs; resolve via arXiv export API, Crossref, HTTP HEAD + Wayback; title match by normalised token overlap against the nearest quoted or bracketed title in the report; classes `resolved`, `title_mismatch`, `not_found`, `unverifiable`. On-disk cache `bench/cache/citations.json` (committed, so reruns are offline). `--offline` uses the cache only. Rate-limited, 3 s arXiv courtesy delay, stdlib `urllib`. | `bench/citations.py`, `bench/detectors.py` | `tests/bench/test_citations.py` with a fake HTTP layer **and a known-bad fixture**: real ID → resolved; valid-format fabricated ID → not_found; real DOI with wrong title → title_mismatch; network error → unverifiable and excluded from the ratio. One opt-in live test (`BLITZ_LIVE_NET=1`) resolves `1706.03762` and rejects a fabricated ID. |
| **5. Composite v2** | `research_quality_composite` gains `citations_resolved` and a `composite_v2` where the 0.30 citation weight uses M1's ratio × richness instead of well-formedness. v1 stays in the output for comparability with the 2026-11-04 runs. Length (chars, words) and coverage-per-1k-chars added to each row. | `bench/detectors.py`, `bench/runner.py` | A report citing 3 fabricated-but-well-formed IDs scores 0.9 on v1 citations and 0.0 on v2; one with 3 real IDs scores the same on both. |
| **6. M2 pairwise judge** | `bench/judge.py`: for a prompt and two report paths, build the blinded rubric prompt (expected-facts-first step, four criteria), call the pinned judge provider with a strict schema, run both orders, return per-criterion winner with order-flip → tie. Refuses to run when the judge provider equals the generator provider unless `--allow-same-family`. Verdicts cached by content hash. | `bench/judge.py` | Fake judge that always answers "A": every pair becomes a tie (position bias caught). Fake judge preferring a marker: marker wins in both orders. Same-provider config raises. |
| **7. M4 co-failure** | `beta_coverage(drafts, expected)`: fraction of expected items no draft mentions; per-draft miss rate alongside. Reported for S (researcher round-1 outputs), B2, B3. | `bench/detectors.py` | Hand-built drafts with a known all-miss item give the exact β. |
| **8. Compare** | `bench/compare.py --runs <dir>…`: aligns runs by prompt id, emits `comparison.md` with per-arm means and bootstrap CIs for M1, composite v2, coverage, length; paired t / d_z vs B1 for each arm; M2 win/tie/loss and BT scores; cost block (calls, output tokens, wall, reported cost + policy); β table. States n and the smallest detectable effect at that n. | `bench/compare.py`, `bench/stats.py` | Synthetic runs with a planted +0.1 composite gain: the table shows it with the right sign, n, and CI; prompts missing from any arm are listed, not silently dropped. |
| **9. Human labels** | `bench/label.py`: terminal tool; shows a pair (anonymised, order randomised, paged), asks per-criterion preference, appends to `labels.jsonl`, resumable. `bench/compare.py --labels` reports Cohen's κ judge-vs-human with bootstrap CI. | `bench/label.py`, `bench/compare.py` | Scripted stdin session writes the expected rows; κ on a constructed set equals a hand-computed value. |
| **10. Budget and preflight** | `--output-token-budget` and `--max-calls` gates in the runner (a token gate, since Codex cost is unknown). Preflight before any live arm: both CLIs answer a 1-token ping, judge provider ≠ generator provider, cache dir writable, free disk > 2 GB. A 1-prompt smoke of every arm runs before any multi-prompt job. | `bench/runner.py`, `bench/preflight.py` | Preflight fails with a named reason for each missing condition (tested with fakes). |
| **11. Pilot run + write-up** | Run the pilot (below), write `docs/audit/<date>/PHASE1-RESULTS.md` with the comparison table, what was and was not detected, and the decision-rule outcome. Update README limitations and `RESEARCH_LOG.md`. | docs, `bench/runs/` | Results file exists, cites run ids, and every number in it is reproducible with `bench/compare.py` from committed artefacts. |

Order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 10 → smoke → 9 → 11.

## Pilot design (needs your call, see Questions)

- **Prompts:** the 20 prompts = the 10 already scored for S on 2026-11-04 (s001, s005, s006, s011, s012, s017, s018, s024, s025, s030) + 10 more chosen by fixed rule (lowest unused id per tier to reach 8 easy / 8 medium / 4 hard).
- **Arms:** B0, B1, B2, B3, S. Generator: Codex `standard`. Judge: Claude `standard`.
- **Seeds:** 1 to start. A second seed only for arms whose CI vs B1 straddles zero.
- **Gates:** `--output-token-budget` per arm, `--max-calls`, one arm at a time, sequential prompts.

## Decision rule (fixed before the run)

- **S beats B1** only if the paired CI excludes zero in S's favour on **M1 or composite v2**, *and* S does not lose M2. Then Phase 2 builds on the swarm.
- **Otherwise** the README says so, the default becomes the best-scoring cheaper arm plus a verifier, and the swarm stays as an experimental mode. Either result is written up.
- At n=20 only large effects (d_z ≥ 0.66) are detectable. "No detectable difference" is a possible and reportable outcome, not a failure of the run.

## Out of scope

Retrieval / search tools for agents (Phase 2), claim-level support against source text (Phase 2), deliberation-loop changes (Phase 3), any change to the swarm itself during the pilot.

## Questions for you

1. **Usage budget.** The pilot is roughly 5–6 hours of wall-clock and a large share of a weekly limit (the 10-prompt S run on 2026-11-04 used 22M input tokens). OK to run it in one or two overnight sittings, or do you want a smaller first cut (e.g. 12 prompts, B0/B1/S only)?
2. **Swarm profile for S.** The existing 10 runs used profile `max` (guard + judge ensemble + selector). Keep `max` so they can be reused, or re-run S at `balanced`?
3. **Labelling.** 30 pairs of long reports is 2–3 hours of your reading. Acceptable, or should the tool show only the first ~150 lines of each report to cut it down (with that limitation stated)?
4. **Cache in git.** `bench/cache/citations.json` holds only identifiers and titles from arXiv/Crossref. Commit it (reproducible offline) or keep it local?
