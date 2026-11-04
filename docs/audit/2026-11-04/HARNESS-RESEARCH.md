# Any harness, any provider: research for the backend layer

2026-11-04. Research only; **not implementing yet.** Companion to [../2026-10-27/AUDIT.md](../2026-10-27/AUDIT.md). Tags: **[PR]** peer-reviewed · **[P]** preprint · **[V]** vendor/official docs · **[read]** confirmed in repo code.

## 1. What already exists in the repo

`main` has no abstraction: every call site builds a `claude -p ...` argv by hand (`orchestrator.py:76-84`, `agents.py:130-137, 515-523`, `memory/reader.py`, `memory/writer.py`, `mythos/_invoke.py`) [read].

PR #2 (`origin/fable/audit-remediation`) already carries a backend layer [read]:

| File | Size | What it holds |
|---|---|---|
| `backends/core.py` | 12.3 kB | `AgentBackend` Protocol, `AgentCall`, `AgentResult`, `BackendCapabilities`, `RuntimePolicy`, loose-JSON parsing, strict-schema conversion |
| `backends/models.py` | 8.1 kB | `resolve_model(role, backend_id, cfg)`: role → tier → provider-native model string, with env overrides |
| `backends/registry.py` | 11.3 kB | `BackendRegistry`, `FallbackBackend` (one retry on the next provider), factories |
| `backends/adapters/cli.py` | 15.5 kB | `ClaudeCLIBackend`, `CodexLocalBackend`, `GeminiCLIBackend` |
| `backends/adapters/generic_cli.py` | 5.1 kB | argv-template adapter: any CLI, configured in TOML, no code |
| `backends/adapters/openai_http.py` | 6.4 kB | OpenAI-compatible HTTP (`/v1/chat/completions`) |
| `backends/adapters/ollama.py` | 4.2 kB | Ollama HTTP |
| `blitz.toml [backend.providers.*]` | — | the only place model names live; tiers `fast/standard/deep` per provider |

Every production call site on that branch goes through `backend.call(AgentCall(...))` and `resolve_model(...)` (`orchestrator.py:307`, `agents.py:544`, `mythos/_invoke.py:114`, `mechanisms/cascade_guard.py:248`, `gmemory/llm_ops.py:72`) [read]. `grep dangerously` on that branch returns nothing outside tests [read].

**So "any provider" is ~80% built on PR #2 and 0% on `main`.** The work is to finish and harden that layer, not to design a new one. The gaps:

1. **Capability negotiation is static.** The Claude adapter probes `claude --help` for `--tools`/`--json-schema` (`cli.py:208-236`), but nothing advertises "supports schema", "supports tools allowlist", "reports cost" to the orchestrator. Mechanisms that need a schema (judge, selector) have no way to know when the provider can't enforce one and must validate client-side.
2. **Structured output is handled three different ways.** Claude: `--json-schema` → `structured_output` field [V]. Codex: `--output-schema PATH` + `-o FILE`, strict-mode schema (`additionalProperties:false`) [V]. Gemini CLI: `--output-format json` gives `{response, stats, error}` with **no** user schema [V]. HTTP adapters: provider-specific `response_format`. The repo has `to_strict_output_schema` for Codex [read]; it needs a single "schema policy" per capability (native / prompt-and-validate / unsupported).
3. **Tool policy differs per harness.** Claude: `--tools ""` or allowlist. Codex: `--sandbox read-only|workspace-write|danger-full-access` + `-a never`. Gemini: no documented tool allowlist flag in headless mode [V, from mirrors; verify against the English reference]. The orchestrator needs one `RuntimePolicy` (already a dataclass in `core.py`) that each adapter maps to its flags, and **refuses** when it cannot honour "no tools" rather than silently running with tools.
4. **Cost/usage normalisation.** Claude envelope: `total_cost_usd`, `usage`. Codex JSONL: token counts per event, no USD. Gemini: `stats`. Ollama/local: nothing. `pricing_for()` exists [read]; cost must be "reported | priced from table | unknown", never 0.
5. **No conformance test.** Nothing runs the same `AgentCall` through each adapter against a fake binary and checks the `AgentResult` shape. That is the test that makes "any harness" a claim rather than a hope.
6. **Gemini adapter unverified** (the branch's own comment says so).

## 2. What's new outside the repo (Aug–Nov 2026)

### Agent Client Protocol (ACP) — the harness side

- Zed's **Agent Client Protocol** is a JSON-RPC-over-stdio protocol between a *client* and a *coding agent* (the harness). v1 is stable; v2 is a published draft; Rust/TypeScript SDKs are 1.0; a Python SDK `agent-client-protocol` is on PyPI (0.12.1 Aug 2026, 1.0.0rc3 Oct 12 2026, Python 3.10–3.14) [V].
- The ACP agent registry lists **Gemini CLI, OpenCode, Goose, Kiro, Qwen Code, Kimi CLI, Mistral Vibe, OpenHands, Cline, Cursor, Copilot (preview)** directly, and **Claude ("Claude Agent", via Zed's SDK adapter) and Codex CLI (via ACP's adapter)** through adapters [V].
- Turn model: `session/new {cwd, mcpServers}` → `session/prompt {sessionId, prompt[]}` → streamed `session/update` (`agent_message_chunk`, `tool_call`, `tool_call_update`, `usage_update`, `plan`) → response with `stopReason ∈ {end_turn, max_tokens, max_turn_requests, refusal, cancelled}`; `session/cancel`; `session/request_permission` for tool approval [V].
- **What ACP does not give us:** no JSON-schema/structured-output facility and no standard model selector in `session/new` (only unnamed "mode, model, or session configuration state" on load/resume) [V]. So an ACP backend would still need prompt-and-validate for schemas and per-agent config for model choice.

**Implication.** One `ACPBackend` adapter would cover a dozen harnesses at once, including every future one that joins the registry, and it gives us the **permission hook** (`session/request_permission`) as a real tool gate instead of per-CLI flags. It becomes the *default* adapter; the hand-written Claude/Codex/Gemini adapters stay as the fast path where their native flags (schema enforcement, cost) beat ACP. Python ≤3.14 constraint matters: this machine runs 3.15, PR #2 CI tests 3.12/3.13.

### MCP / A2A / ANP — the tool and agent-to-agent side

- Ehtesham et al., *A survey of agent interoperability protocols* (arXiv 2505.02279) [P]: MCP = agent→tool, ACP(IBM, a different "ACP") = REST messaging, A2A = agent↔agent via Agent Cards, ANP = decentralised discovery. Nothing newer than mid-2025 at survey level surfaced; the 2026 hits were auto-generated digests.
- For blitz the relevant one is **MCP for search/retrieval tools** (Phase 2 of the 10-27 proposal): ACP sessions take an `mcpServers` list, so a grounded researcher = ACP session + one read-only search MCP server, on any harness.

### Cost-aware routing and cascades — the provider side

All [P], none agent-specific, all on QA/math:

- **CRE-Router**, Moslem et al. 2606.27457 (v2 Oct 23 2026): cluster queries → route each cluster to the cheapest adequate model → quality-estimator escalation to the strong model; needs only task-correctness labels; pools of 2–9 local models. Accuracy "comparable to the strong model" at lower cost (no numbers in abstract).
- **UCCI**, 2605.18796: calibrated-uncertainty thresholds are cost-optimal under stated assumptions; beats entropy thresholds and FrugalGPT-style baselines.
- **RouteNLP**, 2604.23577: conformal escalation thresholds; 8-week pilot, 58% cost cut.
- **RLCascadeRouter** 2608.15817, **Signed Rescue Routing** 2609.07786 (route on what the strong model would *change*), **Drift-aware routing** 2609.00662 (budgets as shadow prices).

**Implication.** The repo's `fast/standard/deep` tiers are the right primitive. The papers argue for **escalation after a cheap attempt, gated by a calibrated check**, rather than static role→tier. That maps onto blitz directly: researchers on `fast`, escalate a subtopic to `deep` only when the fact-checker's grounded score for it is low. Needs Phase-1 metrics first; without a quality estimator there is nothing to calibrate.

### Mixing model families — why "any provider" matters for quality, not just portability

- de Curtò & de Zarzà, *Collective Intelligence with Foundation Models*, 2607.07729 [P, single chapter, numbers from abstract]: heterogeneous solvers 0.64 vs 0.54 step-wise accuracy; "model diversity was the main driver", redundant same-model sampling gave modest gains.
- MAS-ProVe, Venkataramani et al. 2602.03053 [PR, ICML 2026]: process verification across 6 MAS frameworks "does not consistently improve performance"; LLM-judges beat reward models; verification-fine-tuned judges beat general ones; context length vs performance trade-off.
- Consistent with the 10-27 brief (ReConcile, Verga PoLL, Maryanskyy): **the measurable benefit of a provider layer is cross-family judging and critique.** Portability alone doesn't move accuracy.

## 3. Design conclusions (for plan.md)

1. **Keep PR #2's `AgentBackend` contract; add an explicit `BackendCapabilities` negotiation** (`schema: native|validate|none`, `tools: allowlist|sandbox|none`, `cost: reported|priced|unknown`, `effort`, `max_context`). Orchestrator features declare what they *require*; the registry refuses or degrades loudly.
2. **Add `ACPBackend`** (Python SDK, stdio JSON-RPC) as the universal harness adapter; map `RuntimePolicy.tools=[]` to denying every `session/request_permission`.
3. **One schema policy**: native enforcement where the harness has it (Claude `--json-schema`, Codex `--output-schema`, OpenAI HTTP `response_format`); otherwise prompt + client-side `validate_json_schema` + one retry. Never regex-brace-match as a silent fallback (AUDIT H5).
4. **Conformance suite**: the same 6 `AgentCall`s (plain, schema, schema-with-nulls, no-tools, timeout, unavailable) through every adapter against fake binaries/servers, asserting identical `AgentResult` shape, error taxonomy and cost field semantics. This is the gate for calling a new harness "supported".
5. **Cross-family placement as config, not code**: `[models] judge_provider = "codex"` style per-role provider pins, so judges and critics can sit on a different family from researchers (RESEARCH §3 of 10-27).
6. **Escalation tiers (later)**: `fast` first, `deep` on a calibrated low score, once Phase-1 metrics exist.

## Sources

- ACP: https://agentclientprotocol.com (intro, `protocol/v1/prompt-turn`, `protocol/v1/session-setup`, `get-started/agents`, `llms.txt`) [V]; PyPI `agent-client-protocol` https://pypi.org/project/agent-client-protocol/ [V]
- Claude Code headless: https://code.claude.com/docs/ (headless; `--json-schema` → `structured_output`) [V]
- Codex CLI `exec` flags: https://learn.chatgpt.com/docs/developer-commands?surface=cli (`--output-schema`, `--json`, `-o`, `--sandbox`, `-a never`, `-c`, `-C`) [V]
- Gemini CLI headless: https://geminicli.com/docs/cli/headless (`--output-format json` → `{response, stats, error}`) [V, mirrors; verify]
- Ehtesham et al., interoperability survey: https://arxiv.org/abs/2505.02279 [P]
- Moslem et al., CRE-Router: https://arxiv.org/abs/2606.27457 [P] · UCCI https://arxiv.org/abs/2605.18796 · RouteNLP https://arxiv.org/abs/2604.23577 · RLCascadeRouter https://arxiv.org/abs/2608.15817 · Signed Rescue Routing https://arxiv.org/abs/2609.07786 · Drift-aware https://arxiv.org/abs/2609.00662 [all P]
- de Curtò & de Zarzà: https://arxiv.org/abs/2607.07729 [P]
- Venkataramani et al., MAS-ProVe, ICML 2026: https://arxiv.org/abs/2602.03053 [PR]
