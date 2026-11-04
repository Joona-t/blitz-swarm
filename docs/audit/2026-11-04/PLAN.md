# Plan: any harness, any provider

2026-11-04. Derived from [HARNESS-RESEARCH.md](HARNESS-RESEARCH.md) and the revision table in [FRONTIER-RESEARCH.md](FRONTIER-RESEARCH.md). **Waits for Joona's annotations and "implement it all" before any code.** Each work package (WP) ends in a test that fails before and passes after. No paid APIs: every provider is a local CLI, a local HTTP server, or a subscription CLI already on the machine.

## Base

Branch `next/v0.4` cut from `origin/fable/audit-remediation` (PR #2 tip, `86ada20`). `main` stays untouched. Reason: PR #2 already has `backends/{core,models,registry}.py` and six adapters; `main` has none ([HARNESS-RESEARCH §1](HARNESS-RESEARCH.md)).

Contract we keep, verbatim from `backends/core.py` on that branch: `AgentCall`, `AgentResult`, `RuntimePolicy`, `BackendCapabilities`, `AgentBackend` Protocol. Changes below are additive to those dataclasses.

## Design rules

1. **Capabilities are declared, requirements are checked.** A feature that needs schema enforcement or no-tools says so; the registry refuses or degrades *loudly* (log line + `AgentResult.metadata["degraded"]`), never silently.
2. **One schema policy.** `native` (harness enforces) → `validate` (prompt + client-side `validate_json_schema` + one retry) → `unsupported` (refuse the call). Regex brace-matching is removed as a silent fallback (AUDIT H5).
3. **Tools default to none.** `AgentCall.tools=None` must map to a real deny on every adapter, or the adapter is marked `tool_sandbox=False` and the registry refuses no-tools calls on it.
4. **Cost is `reported | priced | unknown`.** `cost_usd=None` stays "unknown"; a pricing table turns it into `priced`; never 0.
5. **Provider pins are config.** Per-role provider in `blitz.toml [models]`, so judges and critics can sit on a different family from researchers (FRONTIER §1, §2: pick by measured skill).

## Work packages

| WP | Scope | Files | Done when |
|---|---|---|---|
| **1. Capability negotiation** | Extend `BackendCapabilities` with `schema_policy: native\|validate\|none`, `tool_policy: allowlist\|sandbox\|none`, `cost_policy: reported\|priced\|unknown`, `effort: bool`, `max_turns: bool`. Add `AgentCall.requires: set[str]`. Registry checks `requires ⊆ capabilities` at `get()`; mismatch → `BackendCapabilityError` or a logged degrade when `allow_degrade=True`. | `backends/core.py`, `backends/registry.py`, each adapter's `capabilities()` | `tests/test_capabilities.py`: a call requiring `schema=native` against the Gemini adapter raises; against Claude passes; against generic-CLI with no schema placeholder degrades to `validate` with a log line. |
| **2. Schema policy** | One `apply_schema_policy(call, caps)` in `core.py`: native → pass through; validate → append schema to prompt, validate result, retry once, else `errored=True, error="schema"`. Remove brace-regex fallback from `orchestrator._parse_agent_output` and `parse_json_loose` callers. | `backends/core.py`, `orchestrator.py`, `agents.py` (planner), `mythos/_invoke.py` | `tests/test_schema_policy.py` with fake binaries: malformed output is an error, not a `needs_work` vote; planner reads `structured_output` (AUDIT H3). |
| **3. Tool policy per adapter** | Claude: `--tools ""` / allowlist, `--max-turns`. Codex: `--sandbox read-only -a never` for no-tools, `workspace-write` for allowlist; `--yolo` forbidden (lint). Gemini: no documented allowlist → `tool_policy=none`, no-tools calls refused unless `[backend.providers.gemini] trust_prompt_only=true` is set explicitly. HTTP/Ollama: no tools by construction → `allowlist` with empty set. | `backends/adapters/*.py` | `tests/test_tool_policy.py` asserts the exact argv per adapter for `tools=None` and `tools=["WebSearch"]`; `grep -rn "yolo\|dangerously" --include=*.py . \| grep -v tests` is empty and enforced by a test. |
| **4. Cost normalisation** | `AgentResult.cost_policy` field; Claude envelope → `reported`; Codex JSONL `token_count` events → tokens + `priced` if table; Gemini `stats` → tokens; local → `unknown`. Fix `metrics.py` input-token under-count (AUDIT Low) by summing cache tokens. | `backends/adapters/*.py`, `metrics.py` | `tests/test_cost_policy.py`: fixture envelopes for each harness produce the right policy and token sums; a run with any `unknown` cost prints "cost: unknown", never `$0.0000`. |
| **5. ACP adapter** | `backends/adapters/acp.py` using `agent-client-protocol` (PyPI, optional extra `acp`). `initialize` → `session/new {cwd, mcpServers}` → `session/prompt` → collect `agent_message_chunk`; `usage_update` → tokens; `stopReason` → error taxonomy (`max_tokens`, `refusal`, `cancelled`); deny every `session/request_permission` when `tools=None`, allow by name otherwise. Schema policy `validate`. Config: `[backend.providers.<id>] adapter="acp" command=[...]`. | new file, `registry.py`, `blitz.toml` | `tests/test_acp_adapter.py` against a fake ACP agent (stdio JSON-RPC script in `tests/fakes/`): prompt round-trip, permission denied when no tools, `refusal` → `errored`. Live smoke against `gemini` or `opencode` if installed, skipped otherwise. **Python ≤3.14 constraint**: CI matrix adds 3.14; 3.15 skips WP5 tests with a reason. |
| **6. Conformance suite** | Six canonical `AgentCall`s (plain, schema, schema-with-nulls, no-tools, timeout, binary-unavailable) run through *every* registered adapter against fakes in `tests/fakes/`. Asserts identical `AgentResult` shape, error taxonomy, `cost_policy`, and that `raw_stdout` is preserved. New adapters must pass it to be listed in `README`. | `tests/test_conformance.py`, `tests/fakes/{claude,codex,gemini,acp,openai}` | Parametrised over `registered_backends()`; 6 × N green. |
| **7. Provider pins + list-models** | `[models] <role>_provider = "<id>"` overrides `[backend] default` per role; `--list-models` prints role → provider → model → capabilities. Default config: researchers on the default provider, `judge`/`critic` on a *different* installed provider when one is available, else same provider with a printed warning. | `backends/models.py`, `config.py`, `blitz.toml`, `orchestrator.py` | `tests/test_provider_pins.py`; `python orchestrator.py --list-models` runs model-free. |
| **8. Packaging + CI** | `prompts` in `packages`, `.toml` as package-data, `blitz` console script, wheel-install job in CI (not only `-e`), CI matrix 3.12/3.13/3.14. Remove `/Users/darkfire` path from `jobs/blitz_upgrade.py` (make it config). | `pyproject.toml`, `.github/workflows/ci.yml`, `jobs/blitz_upgrade.py` | Clean venv: `pip install dist/*.whl && blitz --dry-run --no-llm-plan "x"` exits 0. Leak-gate grep is a CI step. |
| **9. Docs truth** | README "Supported harnesses" table generated from the conformance suite; mark Gemini "conformance-tested, not live-verified" until someone runs it. BUGS_AND_ITERATIONS entry per WP. | `README.md`, `BUGS_AND_ITERATIONS.md`, `docs/IMPLEMENTATION_STATUS.md` | A doc test asserts every adapter in `registered_backends()` appears in the README table. |

Order: 1 → 2 → 3 → 4 → 6 (suite against existing adapters) → 5 (ACP, must pass 6) → 7 → 8 → 9. WP 6 before WP 5 so the new adapter is born against the gate.

## Out of scope here (tracked in the 27 Oct proposal)

Cost-aware escalation between tiers (FRONTIER: CRE-Router, UCCI) needs Phase-1 metrics first. Pivotal-vote gating, judge abstention, anytime-valid stopping and memory granularity are Phase 2–4 items; this plan only makes them *portable* by giving them a provider layer to sit on.

## Acceptance for the whole plan

- Conformance suite green for Claude, Codex, Gemini, generic-CLI, OpenAI-HTTP, Ollama, ACP (fakes), with live smokes for whichever of `claude`, `codex`, `gemini` are installed (skips otherwise, with reasons printed).
- A consensus run with `[models] judge_provider` set to a different installed provider completes and records both providers in `metrics.jsonl`.
- No `dangerously`/`yolo` in production code; wheel installs clean; leak gate clean.
- Frontier reference for comparison: the ACP registry's listed agents. Concrete criterion: any agent on that registry that runs locally can be added to blitz by *config only* (an `[backend.providers.<id>]` block with `adapter="acp"`) and passes the conformance suite.

## Estimates

Sizes are rough Claude Code session counts, not commitments: WP1–4 ≈ 1–2 sessions; WP6 ≈ 1; WP5 ≈ 1–2 (plus a live smoke if an ACP agent is installed); WP7–9 ≈ 1. Nothing here costs model tokens except the optional live smokes.
