# Proposal: make Blitz-Swarm better by making it measurable

2026-10-27. Builds on [AUDIT.md](AUDIT.md) (what's wrong) and [RESEARCH.md](RESEARCH.md) (what the evidence says).

## The thesis

Blitz-Swarm has outrun its evidence. It has eleven paper-anchored mechanisms. None has run on real output, and the one loop that has run has never been compared against a single agent. The research says that comparison is where most multi-agent systems lose (RESEARCH §1).

So the next version should add **no new mechanisms**. It should earn the right to keep each existing one:

1. **Be able to tell whether a change is better.** That means a bench, baselines and a judge from a different model family.
2. **Fix the input problem.** Ground the research, so "accuracy" has something to be measured against.
3. **Fix the loop the evidence argues against**, then keep only the mechanisms that pay for themselves in ablation.

Each phase ends in a measurement, not a feature list. Each one has an explicit rule for what to keep and what to cut.

---

## Where to build: keep `main` intact

`main` doesn't wire up any v0.2 mechanism (AUDIT H1). PR #2 does, merges cleanly and has 3× the tests. Starting from `main` would mean redoing PR #2.

**Recommendation.** Cut an integration branch `next/v0.4` from the PR #2 tip (`origin/fable/audit-remediation`). Do all work below there. `main` stays exactly as it is until Phase 2 produces evidence worth merging. PR #2 itself stays open as the record of that work. If you'd rather review and merge PR #2 into `main` first, the Phase 0 fixes can go onto that branch instead. The choice changes the order of work, not its content.

---

## Phase 0: make it safe and honest (small, no live runs)

Every item has a test that fails before the fix and passes after.

| Task | Fixes | Notes |
|---|---|---|
| Delete `--dangerously-skip-permissions` from every remaining call site. Agents run with tools disabled (or an explicit allowlist once Phase 1 adds search) in a temp cwd. | H4 | PR #2 already does this for consensus agents. Still to cover: the planner, memory LLM ops and mythos. Move `mythos-bench/adversarial/` outside the swarm's cwd, and run `run_pytest` in a throwaway container or a separate user. |
| Fence untrusted text. Wrap prior-agent output and memory in clearly delimited, labeled blocks with an instruction that they are data. | H4 | This cuts the risk of prompt injection without eliminating it. Pair it with the tool lockdown. |
| Enforce the output schema: pass `--json-schema` and read `structured_output`. Count a schema failure as an *error*, not as a `needs_work` vote. | H5, H3 | Use the same fix for the planner (`agents.py:_llm_plan`, `_split_subtopics_llm`). |
| Rewrite `build_context`: newest round first, a separate budget per role, the final round passed in full, and critic text never truncated to zero. | H2 | Add a test asserting the synthesizer context contains the last round and at least one critique. |
| Persist the judge ensemble across swarm rounds: one scoring per swarm round, with no inner `while` loop. | PR #2 #1, M1 | Until it is validated, record the KS statistic as a diagnostic only and never halt on it. |
| Packaging: add `prompts` and the `.toml` configs, add a `blitz` console script, test a wheel install in CI. Remove the `/Users/darkfire/...` path. Make `--dry-run` fully model-free. | PR #2 #4–6 | |
| Namespace blackboard keys per `run_id`. | M10 | |
| **Docs truth pass:** README badge and version, RESEARCH_LOG vs `metrics.jsonl`, the stale LIMITATIONS #10, the AFlow first author, the Maryanskyy framing, the BCa claim, the power table. | AUDIT §Low | Use the cheapest honest wording: "implemented, not yet evaluated". |

**Done when** the tests are green, a wheel installs and runs `--dry-run` in a clean venv, and `grep -r dangerously-skip` returns nothing.

---

## Phase 1: build the measuring stick

This is the core of the proposal. Nothing after it means anything without it.

### 1a. The arms (equal-compute comparison, RESEARCH §1)

| Arm | What it is |
|---|---|
| **B0** | One Sonnet call, same researcher prompt. |
| **B1** | A single agent given the swarm's **median token budget**: draft → self-critique → revise, in one context. |
| **B2** | Best-of-N: N independent Sonnet drafts (N = the swarm's researcher count), with one picked by the cross-family judge. |
| **S** | The swarm as it stands after Phase 0 (PR #2 defaults with the judge loop fixed). |

Without B1 and B2, a "swarm wins" result can't be told apart from "more tokens win".

### 1b. The metrics: grounded first, judged second

1. **Citation resolvability.** Deterministic, no LLM. Extract every reference and resolve it against arXiv, Crossref/DOI and URL liveness (HEAD plus the Wayback Machine as fallback). Report fabricated, resolvable-but-wrong-metadata and correct. This is the one metric no judge can game (RESEARCH §4).
2. **Claim support.** For a sample of K=10 atomic claims per report, have a cross-family judge check each claim against the *fetched* source text, or mark it "unsupported/unverifiable".
3. **Pairwise preference.** A cross-family judge (the repo already has codex/gpt as a backend) compares the arm's report with B1's, run **in both orders**, with a per-criterion rubric and length reported alongside. Never use a Claude judge for Claude output (RESEARCH §3).
4. **Cost and wall-clock** from the envelope. Fix the token counting first (AUDIT, metrics.py).
5. **Loop diagnostics.** Rounds, vote trajectory, and how often a claim present in round k disappears from the final output. That last one is the "oracle gap" (RESEARCH §2).

### 1c. Calibration

Hand-label 30 report pairs, roughly 2–3 hours of your time. If agreement between the judge and you is below about 0.6 (Cohen's κ), fix the rubric before trusting any judged metric. The 0.6 threshold is a convention, not a law.

### 1d. Statistics and budget

- Paired design on the same prompts, ≥3 seeds per (arm, prompt), bootstrap CIs, cost per run reported (RESEARCH §5).
- **Pilot:** 20 prompts from `bench/slate_v1.toml`. At n=20 paired, 80% power at α=0.05 detects an effect of about d_z ≥ 0.66, so the pilot only catches **large** effects. Detecting d_z=0.5 needs about 34 prompts.
- **Cost estimate** (not measured): the observed swarm cost was $0.47–3.70 per run, so 20 prompts × 3 seeds ≈ $30–220 for arm S, and much less for B0–B2. On a CLI subscription the binding limit will be rate limits, not dollars. Fix the runner budget accounting before launching (AUDIT M5).
- Fix `bench/runner.py` first: real spend accounting, resume by `run_id`, errors reported separately rather than scored as 0.

**Done when** `python -m bench.runner --arms B0 B1 B2 S --prompts pilot20 --seeds 3` produces one report with a cost-vs-quality plot and CIs.

### Decision rule after Phase 1

- **If S doesn't beat B1/B2 on grounded metrics at comparable cost:** say so in the README. Make Phase 2 grounding plus a *single-agent-plus-verifier* topology (essentially Mythos's shape) the default, and keep the swarm as an experimental mode. That is still a useful, publishable result.
- **If S wins:** continue to Phase 2 with S as the base.

---

## Phase 2: ground the research (biggest expected gain, RESEARCH §4)

1. **Researchers get search.** Port the Research Swarm search MCP (ROADMAP item 7) behind an allowlisted tool, so agents get read-only search and fetch and nothing else.
2. **A citation resolver as a pipeline stage, not an agent.** Run the Phase 1b resolver on every researcher output before it reaches the blackboard. Unresolvable references are stripped and logged, and the claim is marked unsourced.
3. **The fact-checker gets the fetched sources** and grades claim support, instead of "verifying" from memory.
4. **Change the prompts:** "cite only sources you retrieved; mark anything else as unsourced". This replaces "cite specific papers with authors and years".
5. This gives CascadeGuard a real root of trust. Claims grounded by the resolver and fact-checker become the green anchors it has never had (AUDIT H7).

**Measure:** re-run the pilot for B1+search and S+search. Expect a large gain in citation resolvability for both. The interesting question is whether S keeps any lead once B1 can also search.

---

## Phase 3: fix the deliberation loop (RESEARCH §2–3)

Ablate each change on its own against the Phase 2 arm.

| Change | Replaces | Evidence |
|---|---|---|
| **Independent blinded scoring.** Evaluators don't see each other's votes. Use a rubric score, not ready/needs_work. | Unanimous `ready` | Kaesberg; Choi; Wynn |
| **A fixed 1–2 revision rounds**, with the stop decided by score improvement < ε or by budget. | Holdout override at round 3 | Kaesberg (more rounds hurt); Bertalanič |
| **At least one evaluator from a different model family.** Cut persona-only diversity if it doesn't help in ablation. | 3× Sonnet judges, persona critics | Verga; ReConcile; Panickssery; Li |
| **Select then patch.** Fix selector clustering (embedding similarity, not exact `##` headings), pick the best draft per section, then let the synthesizer graft only attributed, grounded claims from the other drafts. Keep critique and dissent in the output. | Free synthesis / concatenation | Maryanskyy; AUDIT M2, PR #2 #2 |
| **A minority-claim survival check.** Any grounded claim present in some draft but missing from the final output gets listed in the Dissent section. | — | Bertalanič oracle gap |
| **Fault injection.** Seed one fabricated claim into one researcher. Measure how often it reaches the final output, with and without the guard. | — | Xie; Kim |

**Keep or cut:** a change stays only if its paired CI excludes zero in the right direction on a grounded metric, or if it holds quality steady while cutting cost by ≥20%.

---

## Phase 4: the self-improvement layer, after it has something true to optimize

- **GEPA/AFlow** only optimize against the Phase 1 metrics, scored by the cross-family judge, on a held-out prompt split that the optimizer never sees. If they optimize against a Sonnet judge, they will learn to please Sonnet (RESEARCH §3).
- **meta_loop** auto-merge requires a paired test with real n, every check present (a missing check fails closed) and a cost check (AUDIT M7). Until then, it only opens a PR.
- **Memory:** fix retrieval (embed the query, use the correct distance, add a relevance threshold), fix FTS5 escaping and the schema migration (AUDIT H8, M3, M4), and stop writing synthesizer output into Tier 1. Then test the one memory claim that matters: on a second run of a related topic, does retrieved memory improve grounded metrics compared with an empty memory? If not, turn it off by default.

---

## What I'd cut or freeze now

- **The KS halting rule** becomes a diagnostic only. At N=3 it can't fire meaningfully (AUDIT M1), and the method rests on one preprint.
- **MAST "9/14 detected"** comes out of the README until the detectors' false-positive rates are measured on the Phase 1 traces.
- **`managed_agents/`** is a stub. Mark it as one, or move it to `contrib/` (PR #2 already does this).
- **Defaults:** the cheapest configuration that the evidence supports, not `quality_profile="max"`.

---

## Order of work and size

| Phase | Size | Needs live runs? | Your time |
|---|---|---|---|
| 0 Safety and honesty | ~1–2 sessions | No | Review only |
| 1 Measuring stick | ~2–3 sessions plus a pilot run | Yes (pilot) | ~3 h of labeling |
| 2 Grounding | ~2 sessions plus a re-run | Yes | Review |
| 3 Loop fixes | ~1 session per ablation | Yes | Review |
| 4 Self-improvement | Only if Phase 3 leaves something to tune | Yes | Review |

Session counts are rough estimates for Claude Code sessions, not commitments.

## The question this whole plan answers

*Does a swarm of agents that read each other's work produce a more accurate research report than one agent given the same budget and the same tools?*

Nobody has published an answer for long-form synthesis (RESEARCH §1, "untested"). Blitz-Swarm already has most of the machinery to answer it honestly, and either answer is worth having.
