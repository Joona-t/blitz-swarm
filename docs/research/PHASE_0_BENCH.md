# Phase 0 Bench Harness — Implementation Deep Dive

**Status:** Frozen research output from background-task agent run 2026-05-09.
**Source:** Phase 0 bench-harness deep-dive subagent.
**Consumed by:** `bench/` implementation.

---

## 0. Preamble — why this spec is the foundation

Phase 0 is the fitness signal. Without a stable bench, every later phase (mechanism A/B, GEPA, Aflow, RSI meta-loop) is open-loop optimization against noise. The Karpathy frame — *if you can't measure it, you can't improve it* — is binding here. So this document is engineered to one specific test: **can two engineers, working from this doc alone, ship a runnable bench in 5 days?** If yes, Phase 1 unblocks.

Three load-bearing constraints inherited from the existing codebase (verified by reading `orchestrator.py`, `agents.py`, `metrics.py`, `consensus.py`, `config.py`, `blitz.toml`):

1. **Agents are stateless `claude` CLI subprocesses** invoked from the orchestrator (lines 76–105 of `orchestrator.py`). The bench must drive `run_swarm(topic, …)` (orchestrator.py:555), not reach inside.
2. **`metrics.jsonl` already records per-run quality scores, cost, wall-clock, rounds-to-consensus** (metrics.py:55–84). The bench reads this — it does *not* duplicate the schema.
3. **Crypto-trading angles are wired into agent planning** via `_CRYPTO_KEYWORDS` (agents.py:193). The slate must avoid those keywords for general-research prompts unless we want crypto-specialized planning. Five of our prompts will deliberately *include* those keywords as a regression test for the specialization path.

Anchor papers (verified, not hallucinated):

- **MAST** — Cemri et al., arXiv 2503.13657, *Why Do Multi-Agent LLM Systems Fail?* — 14 failure modes (FM-1.1 … FM-3.3) clustered into 3 categories.
- **Shen et al.** — arXiv 2603.29632 — compares subagent (parallel + post-hoc consolidation) vs agent-team (sequential handoffs).
- **Towards a Science of Scaling Agent Systems** — arXiv 2512.08296.
- **DeepResearch Bench** — arXiv 2506.11763.
- **DeepScholar-Bench** — arXiv 2508.20033.
- **BrowseComp / BrowseComp-Plus** — arXiv 2504.12516, 2508.06600.
- **MultiAgentBench** — arXiv 2503.01935 (ACL 2025).
- **Gaia2** — arXiv 2602.11964 (ICLR 2026).

---

## 1. Bench slate — `bench/slate_v1.toml`

### 1.1 Design axes

Each prompt has six metadata fields:

| Field | Values | Used by |
|---|---|---|
| `tier` | easy / medium / hard | aggregation, CI cost-throttle |
| `domain` | technical / open-ended / adversarial / multi-domain / compositional | per-class disaggregation |
| `parallelizable` | true / false | parallel-vs-sequential paired analysis |
| `tool_heavy` | true / false | scaling-paper diagnostic |
| `expected_coverage` | list of 3–8 keyword strings | judge completeness rubric |
| `budget_usd` | 0.05 / 0.20 / 0.60 | per-prompt timeout × cost cap |

Tier counts: **15 easy, 10 medium, 5 hard = 30**. Hold-out slate `slate_v1_held.toml` = 20 paraphrased / domain-shifted prompts kept *out of any GEPA loop* until v0.2 ships.

Adversarial: 5 prompts. Multi-domain: 6. Compositional: 5.

### 1.2 The 30 prompts

Format: `id | tier | domain | par | tool | $ | prompt`

| id | tier | dom | par | tool | $ | prompt |
|---|---|---|---|---|---|---|
| s001 | easy | tech | T | F | 0.05 | "Explain SQLite WAL mode internals: WAL file structure, checkpointing, concurrency guarantees." |
| s002 | easy | tech | T | F | 0.05 | "How does Bitcoin's SegWit (BIP141) improve transaction throughput? Cover witness data separation, malleability fix, and effective block-size implications." |
| s003 | easy | tech | T | F | 0.05 | "What does HTTP/3's QUIC transport layer change relative to TCP? Cover head-of-line blocking, 0-RTT handshakes, and connection migration." |
| s004 | easy | tech | T | F | 0.05 | "Explain the CAP theorem and provide one concrete production system per partition tolerance trade-off (CP, AP)." |
| s005 | easy | tech | T | F | 0.05 | "How does Postgres MVCC work? Cover xmin/xmax, vacuum, and the relationship to long-running transactions." |
| s006 | easy | open | T | F | 0.05 | "What is the current state of small-language-model (SLM) research relative to frontier LLMs? Touch on Phi-3, Qwen, and Gemma families." |
| s007 | easy | open | T | F | 0.05 | "Summarize capabilities and known weaknesses of MLA (multi-head latent attention) as in DeepSeek-V2." |
| s008 | easy | open | T | F | 0.05 | "What is RAG and what are its three main failure modes in production?" |
| s009 | easy | tech | T | F | 0.05 | "Explain how Raft achieves consensus: leader election, log replication, and safety arguments." |
| s010 | easy | tech | T | F | 0.05 | "How does Rust's borrow checker prevent data races at compile time? Cover ownership, lifetimes, Send/Sync." |
| s011 | easy | tech | T | F | 0.05 | "Summarize OpenAI's structured outputs (JSON Schema) end-to-end including constrained decoding." |
| s012 | easy | tech | T | F | 0.05 | "What is FlashAttention-3 and what are its key throughput claims relative to FA2?" |
| s013 | easy | tech | T | F | 0.05 | "Explain Speculative Decoding: draft model, target model, acceptance ratio, and where it helps." |
| s014 | easy | open | T | F | 0.05 | "Summarize the design of GRPO (Group Relative Policy Optimization) used in DeepSeekMath." |
| s015 | easy | tech | T | F | 0.05 | "How does Apache Iceberg's metadata layer differ from Hive's? Cover snapshots, manifests, and ACID semantics." |
| s016 | med | adv | T | F | 0.20 | "Compare strongest arguments for and against Universal Basic Income, including empirical results from Finland (2017–2018) and Stockton SEED (2019)." |
| s017 | med | adv | F | F | 0.20 | "What is the academic disagreement on whether scaling laws are evidence for or against AGI on a near-term timeline? Steelman both sides." |
| s018 | med | adv | T | F | 0.20 | "Compare the case for and against Modern Monetary Theory (MMT) as a policy framework, citing Kelton vs Krugman/Summers critiques." |
| s019 | med | adv | T | F | 0.20 | "Strongest argument that LLM hallucinations are intrinsic vs strongest argument they are eliminable? Cite specific 2024–2026 research." |
| s020 | med | multi | T | F | 0.20 | "How does perpetual-funding-rate carry interact with on-chain liquidity dynamics during deleveraging cascades? Combine market microstructure, behavioural-econ, and crypto." |
| s021 | med | multi | F | F | 0.20 | "Connect transformer attention sparsity, neuroscience papers on cortical sparse coding, and energy-efficiency claims for analog/in-memory compute." |
| s022 | med | comp | F | F | 0.20 | "First, list top 3 reasons LightGBM tends to outperform XGBoost on tabular data. Then, evaluate which apply to crypto-return prediction at 1-hour bar resolution." |
| s023 | med | comp | F | F | 0.20 | "First, summarize G-Memory (NeurIPS 2025) thesis. Then, evaluate whether its three-tier architecture would still hold if you replaced the LLM with a 7B-class small model." |
| s024 | med | comp | F | F | 0.20 | "First, summarize the MAST taxonomy. Then, classify three named multi-agent failure modes from published agent-system blog posts (Anthropic, DeepMind, OpenAI 2025) into MAST FM codes." |
| s025 | med | tech | T | F | 0.20 | "Explain Mixture-of-Experts routing instabilities (auxiliary-loss-free balancing, expert collapse) with reference to DeepSeek-V3 and OLMoE." |
| s026 | hard | open | F | F | 0.60 | "State of AI alignment research as of mid-2026? Cover scalable oversight, mechanistic interpretability, debate, RLAIF, and constitutional approaches; identify strongest open empirical disagreements." |
| s027 | hard | comp | F | F | 0.60 | "Position on whether multi-agent LLM systems beat single-agent baselines on long-horizon research, citing MAST (Cemri 2025), Google scaling (2512.08296), MultiAgentBench, and Shen et al. (2603.29632). Resolve apparent disagreements." |
| s028 | hard | adv | F | F | 0.60 | "Case for and against current public-key crypto being broken by NISQ-era quantum hardware in next 10 years. Steelman both sides using NIST PQC migration data and recent quantum-volume papers." |
| s029 | hard | multi | F | F | 0.60 | "Synthesize: how do GPU memory-tier hierarchies (HBM, SRAM), KV-cache compression (MQA/GQA/MLA), and speculative decoding *jointly* affect inference economics for a 70B-class model serving 100k QPS?" |
| s030 | hard | comp | F | F | 0.60 | "Three-part: (1) state the bitter lesson; (2) describe strongest empirical counter-evidence in 2024–2026 (e.g., test-time-compute scaling, structured prompting); (3) defend a position on which view will dominate AI research in 2027." |

### 1.3 Why these specific prompts

- s001 / s002 are the same prompts as the existing `output/` directory's two completed runs. Day 1 of bench operation produces a *retrospective baseline*.
- s016–s019 are the explicit adversarial prompts; s028 is the 5th (hard). They map to MAST FM-3.3 (incorrect verification).
- s020 is a **deliberate trip-wire for `_CRYPTO_KEYWORDS`** (agents.py:193) — should route to crypto-specialized agents.
- s022–s024 are compositional — they map to scaling-paper *sequential* tasks where parallel multi-agent should show predicted -70% performance hit.
- s027 is meta — the swarm researches itself. Deliberately triggers the failure mode from Run 2 ("Self-referential topics may trigger a feedback loop"). If our error-isolation fix in Phase 1 worked, s027 should now succeed.

### 1.4 TOML schema

```toml
schema_version = 1
slate_id = "v1"
created_utc = "2026-05-09"
prompt_count = 30
sha256 = "<computed at write time>"

[[prompt]]
id = "s001"
tier = "easy"
domain = "technical"
parallelizable = true
tool_heavy = false
budget_usd = 0.05
expected_coverage = ["WAL file structure", "checkpointing", "rollback journal vs WAL", "concurrency guarantees", "fsync semantics"]
text = "Explain SQLite WAL mode internals: how the WAL file is structured, when checkpointing happens, and what concurrency guarantees it provides."
# ... 29 more entries identical shape
```

---

## 2. Bench runner — `bench/runner.py`

### 2.1 Lifecycle

```
load_slate -> filter_by_args -> for each prompt:
  estimate_cost -> if budget_remaining < estimate: skip & log
  acquire_semaphore (parallel cap)
  run_blitz_swarm(topic=prompt.text, ...)  # invokes existing run_swarm
  load_resulting_metrics_record
  load_resulting_md_output
  judge_with_v01_judge(prompt, output)   # baseline
  optionally judge_with_v02_ensemble(prompt, output)  # behind --judge=ensemble flag
  write per-prompt JSONL row
-> aggregate -> write summary.json -> write stats.md -> render charts
```

Resumable: per-prompt rows append to `bench/runs/<run_id>/results.jsonl` with `fsync` after each write. Restarting skips prompts whose `id` is already in the file.

### 2.2 Module skeleton

Full Python module skeleton with:

- `BenchPrompt`, `BenchConfig`, `PromptResult` dataclasses
- `load_slate`, `filter_slate`, `_git_sha` helpers
- `_run_one_prompt` async function with budget gate, retry, partial-output recovery
- `run_bench` async orchestrator with semaphore + budget-lock + atomic JSONL appends
- CLI argparse main

(See full code in original deep-dive output preserved in agent transcript at `/private/tmp/claude-501/-Users-darkfire/.../tasks/a7c30054dc80bf3a5.output` — implementation will reproduce.)

### 2.3 Sequential baseline — `run_sequential_swarm`

Lives next to `run_swarm` in `orchestrator.py` (or in a new `sequential.py` to keep blast logic clean). Implements Shen et al.'s *agent-team* paradigm: researcher → critic → fact-checker → synthesizer, single pass each, with handoffs. Same agents, same prompts, same model, same total token budget — *only the topology changes*.

---

## 3. MAST regression suite — `bench/mast_regression.py`

The 14 failure modes per arXiv 2503.13657, mapped 1:1 to pytest cases. Each test:
(a) injects the failure deterministically by monkey-patching `invoke_agent` or `blast_agents`,
(b) asserts the orchestrator either *contains*, *recovers*, or *flags* it.

**FC1 — Specification & System Design:**
- FM-1.1 Disobey Task Specification
- FM-1.2 Disobey Role Specification
- FM-1.3 Step Repetition
- FM-1.4 Loss of Conversation History
- FM-1.5 Unaware of Termination

**FC2 — Inter-Agent Misalignment:**
- FM-2.1 Conversation Reset
- FM-2.2 Fail to Ask for Clarification
- FM-2.3 Task Derailment
- FM-2.4 Information Withholding
- FM-2.5 Ignored Other Agents' Input
- FM-2.6 Reasoning-Action Mismatch

**FC3 — Verification & Termination:**
- FM-3.1 Premature Termination
- FM-3.2 No or Incomplete Verification
- FM-3.3 Incorrect Verification

Detector heuristics (live in `bench/detectors.py`, ~150 LOC, rule-based, no LLM calls):

```
FM-1.2: role-output mismatch (researcher findings <50% match agents.py role prompt)
FM-1.3: round-N findings == round-N-1 findings (Jaccard > 0.95)
FM-2.1: convergence not monotone (oscillating ready_votes)
FM-2.4: key_points empty in researcher output
FM-2.5: critic context present but researcher findings unchanged across rounds
FM-2.6: confidence < 0.4 paired with vote == "ready"
FM-3.1: timeout in any round AND output produced (silent partial)
FM-3.2: synthesizer ran but no quality_judge in agent roster
FM-3.3: avg quality > 8 AND output length < 500 chars
```

---

## 4. Parallel-vs-sequential bench — `bench/parallel_vs_sequential.py`

### 4.1 Methodology (per Shen 2603.29632 + Google 2512.08296)

Two configurations against the **same 30-prompt slate**, **same total compute budget**, **same agent roster definition**, **same models**, **same seed**:

- **Run A — Parallel (subagent paradigm):** existing `run_swarm`.
- **Run B — Sequential (agent-team paradigm):** new `run_sequential_swarm`. Researcher → critic → fact-checker → synthesizer. Each sees prior agent's full output; no parallel concurrency.

Compute-budget parity is the load-bearing constraint. We enforce it by:
- Cap parallel run at total tokens T and wall-clock W.
- Sequential run gets the *same* T and W. Because sequential is cheaper per-prompt, we run it with `max_rounds=4` of self-refinement to consume parity tokens.

### 4.2 Pre-registration

**We expect parallel to win on `parallelizable=true` and lose on `parallelizable=false`**. If that's not what we see, we update our priors, not the bench.

---

## 5. Reproducibility schema — `summary.json`

Diff-friendly JSON capturing:
- code: git_sha, git_status_clean, version, python_version, platform
- models: per-role model assignment
- config: blitz_toml_sha256, blitz_toml_inline, bench_config
- slate: slate_id, sha256, prompts run/skipped/timed out/errored
- aggregate_quality: per-dim mean / stddev / ci95
- by_tier, by_domain breakdowns
- consensus stats
- cost: total_usd, tokens, by_tier
- mast_flags_summary: counts per FM-X.Y
- regression_vs_baseline: delta_avg_quality, delta_avg_cost, passed (bool)

Two runs' summary.json files compared with `git diff --no-index` produce a readable delta.

---

## 6. Statistical analysis spec — `stats.md`

| Question | Test |
|---|---|
| Per-dimension mean and uncertainty | Mean + Welch's CI95 (or bootstrap if n<15 in a cell) |
| Bootstrap CI implementation | `scipy.stats.bootstrap` with 10,000 resamples, BCa interval, fixed seed |
| Two-run A/B comparison | **Paired t-test** on per-prompt avg_quality (`scipy.stats.ttest_rel`) |
| A/B with skewed scores | Wilcoxon signed-rank fallback when Shapiro-Wilk p<0.05 |
| Effect size | Cohen's d_z for paired data |
| Cross-tier comparison | One-way ANOVA across tiers, then Tukey HSD if F-stat significant |
| Cost regression vs baseline | Mann-Whitney U (cost is right-skewed) |
| MAST flag rate change | Fisher's exact test (small counts) |
| Multiple-comparison correction | Benjamini-Hochberg FDR at q=0.10 |
| Power | n=14 needed for d_z=0.5 at α=0.05, power=0.8 → our n=30 detects d_z≥0.35 |

---

## 7. Cost model

Anthropic pricing as of May 2026:
- Sonnet 4.6: **$3.00 / $15.00** per million in/out tokens
- Opus 4.7: **$5.00 / $25.00** per million in/out tokens (with ~1.20× tokenizer bloat factor)
- Cache reads: 0.10× input price (90% discount)
- Batch API: 50% off, async — not used by the bench

### 7.1 Realistic per-tier budget

| Tier | Agents | Rounds | Models | Est. per-prompt | Slate count | Subtotal |
|---|---|---|---|---|---|---|
| easy | 4 | 2 | Sonnet only | $0.08 | 15 | $1.20 |
| medium | 6 | 3 | Sonnet + Opus judge | $0.45 | 10 | $4.50 |
| hard | 8 | 4 | Sonnet + Opus judge + Opus synth | $1.40 | 5 | $7.00 |
| **Total full slate** | | | | | **30** | **$12.70** |

Plus 25% safety margin for retries + ensemble judge: **~$16 per full slate run.**

### 7.2 Yearly forecast

- Daily smoke (5 prompts): $0.40 × 365 = $146/yr
- Weekly full slate: $16 × 52 = $832/yr
- Per-PR CI smoke: $0.40 × 150 PRs/yr = $60/yr

**Total realistic bench cost: ~$1,050/yr** (and against Joona's existing Claude Max sub via `claude -p`, this is effectively $0).

### 7.3 Cost gates

- `--budget` default $8 — enough for a smoke (3 prompts × $2.50 worst-case).
- Per-tier override flags: `--easy-only`, `--no-hard`.
- Pre-flight estimator refuses to start if `(estimated_total > 1.5 × budget)` unless `--force`.

---

## 8. CI integration

**Yes, run on every PR — but on a *smoke slate*, not the full slate.**

```yaml
# .github/workflows/bench-smoke.yml
name: bench-smoke
on:
  pull_request:
    paths: ['orchestrator.py', 'agents.py', 'consensus.py', 'metrics.py', 'bench/**']
jobs:
  smoke:
    runs-on: macos-14
    timeout-minutes: 25
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install -e '.[dev]' jsonlines scipy matplotlib
      - run: python bench/runner.py --filter s001 s003 s016 s022 s026 --budget 3.0
      - run: python bench/regression_check.py
      - uses: actions/upload-artifact@v4
        with: { name: bench-results, path: bench/runs/ }
```

Five prompts: one easy (s001), one easy-medium (s003), one adversarial (s016), one compositional (s022), one hard (s026). Touches every domain × tier diagonally. ~$2.50, ~12 min.

**MAST regression suite** runs as a *separate* CI job using **stubbed `invoke_agent`** (no real LLM calls), making it free + fast (~30s). Runs on every commit, not just PRs.

Full slate runs nightly via cron-style scheduled workflow on `main` only.

---

## 9. Open-source bench publication

### 9.1 Yes, MIT-release it

The bench has *more* value as a public artifact than as a private one — it lets blitz-swarm be benchmarked *by other multi-agent systems*.

### 9.2 New repo: `Joona-t/blitz-bench`

```
blitz-bench/
├── slate_v1.toml             # the 30 prompts
├── slate_v1_held.toml        # 20 held-out paraphrases
├── runner_adapter.py         # abstract BenchAdapter
├── adapters/
│   ├── blitz_swarm.py
│   ├── research_swarm.py
│   └── single_agent.py
├── mast_regression.py
├── stats.py
├── README.md
└── LICENSE                   # MIT
```

### 9.3 Positioning vs prior art

Tagline: *"The first multi-agent research bench that fails the system, not just measures the output."*

| Bench | Tasks | Eval | What blitz-bench adds |
|---|---|---|---|
| **GAIA** | 466 single-answer tool-use | string match | Multi-agent failure modes |
| **GAIA2** | 1120 async smartphone-env | action verifier | Async + tool-use complementary |
| **DeepResearch Bench** | 100 PhD reports | RACE+FACT | DRA closed-source; we target MAS internals |
| **DeepScholar-Bench** | live ArXiv synth | 3 dims | We adopt their 3 dims as judge axes |
| **MultiAgentBench** | scenario-based | KPI milestones | They benchmark topology; we benchmark failure modes |
| **MAST-Data** | 1600 traces | failure annotations | We operationalize as live regression |
| **blitz-bench v1** | **30 synth-research** | **4-dim judge + MAST + parallel-vs-seq** | **first MAS bench combining process + output metrics** |

### 9.4 Calibration plan

- 100 DeepResearch tasks → blitz-swarm → measure correlation between RACE score and our avg_quality. If ρ > 0.7, our bench is in the same ballpark.
- Top 50 GAIA tasks → blitz-swarm → measure agreement on accuracy/coverage.

---

## 10. Implementation week — concrete plan

| Day | Deliverable | Lines | Owner |
|---|---|---|---|
| Mon | `bench/slate_v1.toml` (30 prompts) + `bench/__init__.py` + skeleton tests | 350 LOC TOML + 30 LOC stub | data-only |
| Tue | `bench/runner.py` + `bench/detectors.py` (MAST flags only) | ~600 LOC Python | runner runs against 5-prompt smoke |
| Wed | `bench/mast_regression.py` (14 pytest cases) + `bench/sequential.py` | ~500 LOC Python | suite passes locally, no LLM cost |
| Thu | `bench/parallel_vs_sequential.py` + `bench/stats.py` | ~250 LOC Python | first paired run executes |
| Fri | Charts, summary.json polish, GitHub Actions, README, MIT release scaffold | ~150 LOC + docs | bench/runner.py ships green |

---

## 11. The ten things that will go wrong

1. `metrics.jsonl` schema records `coverage` as `int` in some rows and `float` in others. Runner must coerce to float.
2. `_CRYPTO_KEYWORDS` will trip s020 unexpectedly. *Feature, not bug.*
3. Judge LLM is non-deterministic. Bootstrap CI is the answer; if too wide, ensemble across Sonnet+Opus.
4. Run 2's "self-referential topic causes cascading failure" *will* re-occur on s027. Bench treats this as a feature flag.
5. Cost will overshoot v0.1 estimates. Build the budget gate first.
6. `tomllib` lacks dump; use `tomli-w` for any auto-write of TOML. Or never auto-write.
7. `asyncio.as_completed` order is non-deterministic; results.jsonl rows must include `prompt_id` (never positional).
8. macOS Metal subprocess starvation — `parallel_prompts >= 4` will rate-limit. Default 2.
9. Held-out slate must *never* be touched by GEPA. Add a hash-checking pre-commit hook that fails if held opens during a GEPA run.
10. Opus 4.7's 35% tokenizer bloat means 4.6 budget estimates are wrong. Bake `MODEL_TOKENIZER_BLOAT` into `runner.py:estimate_cost`.

---

## 12. What this unblocks

- **Phase 1** gets a paired t-test signal within 4 hours of any orchestrator change.
- **Phase 2** (G-Memory) gets a closed-form fitness signal: `summary.json.aggregate_quality.avg.mean - regression_vs_baseline.delta_avg_quality`.
- **Phase 3** (Aflow / GEPA / meta-loop) gets a multi-objective signal: `(avg_quality, cost_usd, mast_flag_count, consensus_rate)`. NSGA-II readymade Pareto frontier.
- **Phase 4** (RSI meta-loop) gets the closed-form fitness that lets it self-improve without LLM-judge variance noise drowning the gradient.

---

## 13. Sources

- [arXiv:2503.13657 — Why Do Multi-Agent LLM Systems Fail? (MAST)](https://arxiv.org/abs/2503.13657)
- [arXiv:2603.29632 — Empirical Study of Multi-Agent Collaboration for Automated Research](https://arxiv.org/abs/2603.29632)
- [arXiv:2512.08296 — Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296)
- [arXiv:2506.11763 — DeepResearch Bench](https://arxiv.org/abs/2506.11763)
- [arXiv:2508.20033 — DeepScholar-Bench](https://arxiv.org/abs/2508.20033)
- [arXiv:2508.06600 — BrowseComp-Plus](https://arxiv.org/abs/2508.06600)
- [arXiv:2504.12516 — BrowseComp](https://arxiv.org/abs/2504.12516)
- [arXiv:2503.01935 — MultiAgentBench](https://arxiv.org/abs/2503.01935)
- [arXiv:2602.11964 — Gaia2](https://arxiv.org/abs/2602.11964)
- [arXiv:2311.12983 — GAIA](https://arxiv.org/abs/2311.12983)
- [arXiv:2308.03688 — AgentBench](https://arxiv.org/abs/2308.03688)
- [Anthropic API Pricing 2026 — FinOut](https://www.finout.io/blog/anthropic-api-pricing)
- [BrowseComp — OpenAI announcement](https://openai.com/index/browsecomp/)
- [DeepScholar-Bench — UC Berkeley Sky Lab](https://sky.cs.berkeley.edu/project/deepscholar-bench/)
