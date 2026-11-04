# Status of `next/v0.4` — written for the morning after (2026-11-04/05)

Everything below is on branch `next/v0.4` (cut from PR #2's tip `86ada20`; `main` untouched).
CI (`tests.yml`: pytest 3.12/3.13/3.14, wheel install, leak gate) is green on every push since
the WP9 commit except one intermediate red push that was fixed within minutes and is logged in
`BUGS_AND_ITERATIONS.md` ITER-V04. Tests: **872 passed, 16 skipped**. Commits end-to-end tested
through `scripts/hooks/pre-commit` (pytest exit code).

## What was built (PLAN.md WP1–WP9, all done)

| WP | Delivered | Proof |
|---|---|---|
| 1 Capabilities | `BackendCapabilities` schema/tool/cost/effort/max_turns; `AgentCall.requires`; `PolicyBackend` refuses or degrades loudly | `tests/test_capabilities.py` |
| 2 Schema policy | native → validate (+1 retry) → refuse; brace-scan removed; schema failure = error, not a vote | `tests/test_schema_policy.py` |
| 3 Tool policy | Codex read-only/never on no-tools; Claude refuses no-tools on a build without `--tools`; HTTP refuse tool requests; bypass-flag lint | `tests/test_tool_policy.py` |
| 4 Cost | `reported | priced | unknown`; cache tokens counted; Codex JSONL usage; `>= $x` print, never `$0` | `tests/test_cost_policy.py` |
| 5 ACP adapter | stdlib JSON-RPC ACP v1 client; permission gate = tool policy; `{model}` argv | `tests/test_acp_adapter.py`, fake agent |
| 6 Conformance | 6 calls × 8 adapters behind PolicyBackend against `tests/fakes` | `tests/test_conformance.py` |
| 7 Provider pins | `[models] <role>_provider`; `RoleRouter`; `--list-models` provider column + capability table | `tests/test_provider_pins.py` |
| 8 Packaging/CI | prompts + TOML data files, `blitz` script, config lookup order, 3.14 + wheel + leak-gate CI, no home paths | wheel installed in a clean venv, `--dry-run` exit 0 |
| 9 Docs truth | README harness table generated from declarations; `AGENTS.md` rule table | `tests/test_docs_truth.py` |

## What the live runs showed (details: `docs/RESEARCH_LOG.md`, 2026-11-04 entries)

- **Run A** (cheap profile): routing works; Codex synthesizer timed out at 300 s → `synthesizer_timeout_seconds`, findings fallback.
- **Run B** (balanced, 2 rounds): judge ensemble now 1 scoring/round; guard promoted claims; selector output carries critique/dissent; **but** judge/pairwise calls bypassed the judge pin and were invisible to metrics → fixed (`653747e`); Codex researchers at `high` effort timed out half the time → `medium`.
- **Run C**: no timeouts, 258 s, `quality_judge → claude`, selector cost reported. Cross-family judging is live.
- **Bench smokes**: first live `bench.runner` runs on any version (2 prompts, then paired `--baseline`): budget, aggregate, paired stats all work; n=2, so no claim.
- **Slate run** `bench/runs/20261104T212924_4e57c4496ae2_v1` (23:29–00:58 EET): 10 of 30 prompts scored before the $30 budget gate stopped it (20 `budget_exhausted`, excluded from means); self-judged q 7.53 (CI95 7.13–7.86), composite 0.856, $31.74 reported, 22M input tokens. First baseline dataset for `--baseline` pairing; no comparison arm. A full 30-prompt arm at this configuration is ~$90–100.

## Audited defects fixed beyond the plan (ITER-V04b)

Judge-ensemble loop, context budgets (H2), selector dissent, dry-run model-free, per-run blackboard
namespace (M10), Tier-1 trace excludes synthesizer (P4), cascade-guard corroboration root of trust
(H7), selector heading clustering (M2), `avg_quality` zeros, meta_loop fail-closed (M7), verifier
pass-vs-failed-specs (M11), bench budget/aggregate/git-sha (M5), Codex `--skip-git-repo-check` and
JSONL error text, research-log corrections.

## Still open, honestly

- No n≥20 comparison against a single-agent baseline (proposal Phase 1). The slate run is the swarm
  arm only; B0–B3 arms are not implemented. That is the next piece of real work and needs your go.
- KS halting is inert at N=3 (documented); EMS/CITE stopping is Phase 3.
- Codex cost is `unknown` until a `[backend.providers.codex.pricing]` table exists.
- Gemini and ACP adapters are conformance-tested, not live-verified (no agent installed here).
- Per-call CLI overhead (~44k input tokens per Claude call, ~20k per Codex call) dominates token
  counts (LIMITATIONS 20).

## Decisions waiting for you

1. Merge `next/v0.4` into `main` (PR not opened; say the word), or keep iterating on the branch.
2. Start proposal Phase 1 (baseline arms B0–B3 + cross-family judge metric + citation resolver).
3. Whether to keep `judge_provider = "claude"` pins in the shipped `blitz.toml` (currently only in
   the live-run copies; the shipped file has them commented out).
