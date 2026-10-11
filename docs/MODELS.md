# Models, tiers and providers (v0.3.0)

How Blitz-Swarm decides which model runs which role, and how to plug in a new provider without touching code. Everything below was read from the code at commit 36767cd; the command outputs are pasted verbatim from runs on 2026-10-11 (none of them invokes a model).

## 1. The contract

- **Code names roles and tiers.** Roles: `researcher`, `critic`, `fact_checker`, `quality_judge`, `synthesizer`, `judge`, `planner`, `executor`, `verifier`, `memory_ops`, `guard` (plus `*` for anything else). Tiers are abstract strings; the shipped config uses `fast`, `standard`, `deep`.
- **Config maps the two.** `blitz.toml [models]` maps role → tier; `[backend.providers.<id>.tiers]` maps tier → the provider's native model string; `[backend.providers.<id>.effort]` optionally maps tier → effort.
- **One resolver.** `backends.models.resolve_model(role, backend_id, cfg, override=None, *, tier=None)` returns `ResolvedModel(model, tier, effort, backend_id, source)` (`backends/models.py`, `resolve_model`). Every call site in the repo goes through it: `orchestrator.py` (`_resolved_for_agent`, judge/selector/guard tiers), `agents.py` (planner tier), `mythos/_invoke.py`, `mechanisms/*` (model injected by caller), `metrics.py` (pricing), `jobs/*`, `mythos-bench/*`, `heterogeneity/cli_router.py`.
- **No literal model names in code.** `tests/test_no_hardcoded_models.py` greps every non-test `.py`/`.toml` except `blitz.toml` and `mythos.toml` and fails on `opus|sonnet|haiku|fable|gpt-*|claude-*|gemini-*|qwen*|llama*`. `docs/` is not scanned, which is why this file can name models.

## 2. Resolution order

From `backends/models.py:9-21`, applied per role:

| # | Source | Example |
|---|---|---|
| 1 | CLI `--model <model\|tier>` or env `BLITZ_MODEL` | `--model deep` → the backend's `deep` tier; `--model claude-fable-5-1` → taken verbatim |
| 2 | env `BLITZ_MODEL_<ROLE>` (role upper-cased, non-alphanumerics → `_`) | `BLITZ_MODEL_PLANNER=opus` |
| 2b | explicit `tier=` request — strict: an undefined tier raises | `mythos.toml [models] planner_model = "deep"` |
| 3 | `[models].<role>` → tier → `[backend.providers.<id>.tiers].<tier>` | `synthesizer = "deep"` |
| 4 | `[models].default` → tier → provider tiers | `default = "standard"` |
| 5 | provider legacy single `model` key | `[backend.providers.x] model = "..."` |
| 6 | `ModelResolutionError` naming role, backend and tier | never a silent default |

Effort: `[backend.providers.<id>.effort].<tier>` if set, else the provider-wide `reasoning_effort`, else none. The Claude adapter forwards it as `--effort`; the Codex adapter maps it to `-c model_reasoning_effort=`.

### Backend selection

- Consensus mode: `--backend <id>` / `BLITZ_BACKEND` > `blitz.toml [backend] default` (`codex` as shipped).
- Mythos mode: `--backend` / `BLITZ_BACKEND` > `mythos.toml [models] backend` (`"claude"` as shipped) > `blitz.toml [backend] default` (`mythos/_invoke.py:55-70`).
- `[backend] fallback = [...]`: on an errored result of the "unavailable" class (CLI missing, server down, rate-limited) the first available fallback provider is called **once**, with the model re-resolved for *that* provider through its own tier table (`backends/registry.py`, `FallbackBackend`). Timeouts and schema failures never fall back. The result carries `fallback_used=True`.

## 3. What the shipped config resolves to

`python orchestrator.py --list-models --backend claude`:

```
Resolved models for backend='claude' (default tier='standard')
role            tier      model                       effort  source
critic          standard  sonnet                      high    tier
executor        fast      haiku                       high    tier
fact_checker    standard  sonnet                      high    tier
guard           fast      haiku                       high    tier
judge           standard  sonnet                      high    tier
memory_ops      fast      haiku                       high    tier
planner         deep      fable                       max     tier
quality_judge   standard  sonnet                      high    tier
researcher      standard  sonnet                      high    tier
synthesizer     deep      fable                       max     tier
verifier        deep      fable                       max     tier
*               standard  sonnet                      high    default_tier
Override: --model <model|tier>, env BLITZ_MODEL, env BLITZ_MODEL_<ROLE>; '*' = any role not listed under [models].
```

`python orchestrator.py --list-models --backend codex`:

```
Resolved models for backend='codex' (default tier='standard')
role            tier      model                       effort  source
critic          standard  gpt-5.5                     high    tier
executor        fast      gpt-5.5                     high    tier
fact_checker    standard  gpt-5.5                     high    tier
guard           fast      gpt-5.5                     high    tier
judge           standard  gpt-5.5                     high    tier
memory_ops      fast      gpt-5.5                     high    tier
planner         deep      gpt-5.5                     high    tier
quality_judge   standard  gpt-5.5                     high    tier
researcher      standard  gpt-5.5                     high    tier
synthesizer     deep      gpt-5.5                     high    tier
verifier        deep      gpt-5.5                     high    tier
*               standard  gpt-5.5                     high    default_tier
Override: --model <model|tier>, env BLITZ_MODEL, env BLITZ_MODEL_<ROLE>; '*' = any role not listed under [models].
```

`python orchestrator.py "x" --mode mythos --dry-run` (no flags → `mythos.toml` picks `claude`):

```
MYTHOS DRY RUN — x

Config:
  planner_model     = deep  (tier)
  executor_model    = fast  (tier)
  verifier_model    = deep  (tier)
  resolved          = planner=fable (claude:deep, effort=max) | executor=haiku (claude:fast, effort=high) | verifier=fable (claude:deep, effort=max)
  max_replans       = 3
  max_executors     = 8
  cost_ceiling_usd  = 5.0
  token_ceiling     = 2000000
  output_dir        = ./output/mythos
  backend           = claude
  sandbox           = read-only

No CLI calls will be made. Use without --dry-run to execute.
```

`python orchestrator.py "x" --mode mythos --dry-run --backend codex`:

```
  resolved          = planner=gpt-5.5 (codex:deep, effort=high) | executor=gpt-5.5 (codex:fast, effort=high) | verifier=gpt-5.5 (codex:deep, effort=high)
  backend           = codex
```

`fable` is the Claude CLI alias for Fable 5.1 (full id `claude-fable-5-1`); either string is accepted by `claude --model`, and `blitz.toml` says which it uses.

## 4. Environment variables

| Variable | Effect |
|---|---|
| `BLITZ_BACKEND` | provider id, same as `--backend` |
| `BLITZ_MODEL` | model or tier for every role, same as `--model` (step 1) |
| `BLITZ_MODEL_<ROLE>` | model or tier for one role (step 2), e.g. `BLITZ_MODEL_PLANNER`, `BLITZ_MODEL_SYNTHESIZER`, `BLITZ_MODEL_MEMORY_OPS` |
| `BLITZ_SANDBOX` | sandbox mode passed to the adapter (`read-only` as shipped) |
| `BLITZ_FEATURE_OVERRIDES` | the upgrade job's per-arm feature toggles (`orchestrator._feature_enabled`) |

Global `BLITZ_MODEL` is ignored on a fallback hop — it names a model on the primary provider.

## 5. Adapters

| Adapter id | Class | Talks to | Notes |
|---|---|---|---|
| `claude_cli` | `backends.adapters.cli.ClaudeCLIBackend` | `claude -p` | forwards `--effort`; prompts > 100 KB go via stdin, not argv; `is_error` envelope honoured |
| `codex_cli` | `backends.adapters.cli.CodexLocalBackend` | `codex exec` | strict `--output-schema` conversion (BUG-002); effort → `model_reasoning_effort`; read-only sandbox |
| `gemini_cli` | `backends.adapters.cli.GeminiCLIBackend` | `gemini` | not verified on the reference machine |
| `ollama_http` | `backends.adapters.ollama.OllamaHTTPBackend` | `POST /api/generate` | |
| `openai_http` | `backends.adapters.openai_http.OpenAIHTTPBackend` | `POST <base_url>/chat/completions` | any local OpenAI-compatible server; optional `api_key_env`; reuses the strict-schema conversion for `response_format` |
| `generic_cli` | `backends.adapters.generic_cli.GenericCLIBackend` | your argv template | placeholders `{model} {prompt} {prompt_file} {system_prompt} {schema_file} {output_file} {effort} {cwd}` |

Register a provider by adding a `[backend.providers.<id>]` table with `adapter = "<id above>"` and a `tiers` table. The registry is keyed by adapter id and knows nothing about specific providers (`backends/registry.py:1-10`). Custom adapters: `backends.register_backend(adapter_id, factory)`.

## 6. Recipes

See the three recipes in `README.md` ("Model & provider configuration"), copied from `blitz.toml`: `generic_cli` argv template, `openai_http` local server, `ollama_http`.

Add pricing to get cost in the run summary (otherwise `n/a`, never `0`):

```toml
[backend.providers.claude.pricing."claude-fable-5-1"]
input_per_mtok = 15.0
output_per_mtok = 75.0
```

Numbers above are the commented-out example in `blitz.toml`, not a quoted price.

## 7. Rule #10

Every adapter above is a subprocess of, or a local HTTP call to, software the user already runs on their own subscription or hardware. The only paid-API code in the repo is `contrib/managed_agents/` (opt-in, unimported, network call not implemented).
