# Limitations (v0.3.0, 2026-10-11)

Numbered so other docs can cite them. Each item says what is true at commit 36767cd and what artifact (if any) backs it. "Not measured" means exactly that.

## Data

1. **No live bench run exists, on any version.** `bench/runner.py` is `swarm_fn`-injectable and tested with fakes (160 tests); `bench/runs/` and `bench/reports/` are empty. No n≥20 slate run has been executed against Codex, Claude, Gemini, Ollama or a single-agent baseline. Every number in `docs/research/PHASE_0_BENCH.md` is a plan.

2. **mythos-bench results are stale.** `mythos-bench/results/runs.jsonl` has 4 rows dated 2026-05-10, produced on the pre-v0.3 Claude `claude -p` path that the backend refactor replaced. They are kept as history. Regenerating them requires live LLM runs and was deferred (plan: "Explicitly deferred").

3. **The archived run records are not a dataset.** `docs/data/metrics-2026-03_06.jsonl` holds 11 v0.1–v0.2 records including a 2026-06-18 run where 4/5 agents errored every round (quality 0). n is too small for any statistic; `results/RESULTS.md` (n=2, 2026-03-14) is historical.

4. **Memory "compounding" is not measured.** The store round-trips, promotes and evicts in tests on temp databases (`tests/test_memory_store.py`, 31). No before/after comparison across real runs exists.

5. **The quality judge is self-reported.** Scores come from an LLM judge on the same backend as the agents. No human evaluation, no inter-rater reliability.

## Architecture

6. **Cascade guard is a heuristic screen by default.** `[guard] mode = "balanced"` runs word-overlap screening and prints `heuristic screen` (`mechanisms/cascade_guard.py:333`). Only `mode = "max"` runs LLM claim decomposition on the `guard` tier. The `max` path is tested with mocked hooks, not live.

7. **The judge "ensemble" is N calls to one model.** With a single provider the three judges are the same `judge` tier model. Cross-provider judges would need `[backend] fallback`-style routing that does not exist for judges. Empirical-CDF KS instead of the Beta-Binomial mixture of Hu 2510.12697 — documented deviation.

8. **Round 1 is staged, not simultaneous.** Researchers → evaluators → judges is a data dependency. Only rounds 2+ are a single blast. The docs used to say "no waves"; that was false and is now printed at run start.

9. **KS-stop is off by default.** `[judge_ensemble] halt = false`; the swarm stops on evaluator unanimity, the holdout override, or `max_rounds`. The ensemble's halt signal is recorded, not acted on, unless you flip the key.

10. **Codex cost telemetry is absent.** Cost is `n/a` unless `[backend.providers.<id>.pricing]` is configured. The Mythos `cost_ceiling_usd` is inert on such providers; the `token_ceiling` is not.

11. **Mythos verifier trust is mostly the LLM's word.** The confidence floor demotes weak passes; running the deliverable's own tests in a subprocess is opt-in (`verifier_run_tests`, off by default because it executes model-written code without a sandbox, `mythos/verifier.py:109-195`); a verifier that writes both the code and its tests can still bless itself. The 2026-05-10 `monetary_decimal` probe did not trigger a false pass; one probe is not evidence.

12. **No web retrieval.** Agents run tool-less (`--max-turns 1`, `tools = None`). Research output depends on the backend model's knowledge.

13. **Selector synthesis cost scales with spans.** Pairwise Bradley-Terry selection grows with sections × candidates; `candidate_cap` and `granularity` bound it, latency not measured live.

## Experimental / incomplete

14. **`evolve/` and `heterogeneity/` have never run.** Gated by `[evolve] enabled = false`; GEPA needs the optional `gepa` package; AFlow and the meta-loop have only unit tests. The recursion ladder L1/L2 is a design.

15. **The 10M-token upgrade job did not complete.** Live run of 2026-06-10 stalled at phase 1/5 after 829k tokens; its generic fixes (zero-signature detection, circuit breaker) are merged (1a4876c), the run is not.

16. **`contrib/managed_agents/` is a scaffold.** Paid Anthropic API, opt-in, unimported, and the network call is not implemented.

17. **Gemini adapter unverified.** Not installed on the reference machine; `blitz.toml` model ids for it are placeholders.

## External validity

18. **Local CLI behaviour drifts.** Adapters were built against `codex-cli 0.160.0` (BUG-002 verification, 2026-10-07) and the Claude CLI that accepts `--effort`; flag changes upstream require adapter updates. Fallback once-per-call masks outages only between configured providers.

19. **`fable` is a CLI alias.** The Mythos default resolves to whatever `claude --model fable` routes to on your account; `blitz.toml` can pin `claude-fable-5-1` instead.

20. **Solo-operator evaluation.** All design decisions and qualitative judgements are one person's; no external review of the architecture or the audit remediation has happened beyond the automated skeptic pass (WP7).
