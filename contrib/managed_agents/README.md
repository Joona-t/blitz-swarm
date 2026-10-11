# contrib/managed_agents — Anthropic Managed Agents adapter (OPT-IN, PAID API)

**Status: scaffold. The network call is not implemented.** `run_task` raises
`NotImplementedError` whenever a client is attached; without a client it returns a
`status: "stub"` dry-run record. Nothing in blitz-swarm imports this package.

## Why this lives under `contrib/`

LoveSpark rule #10: **no paid LLM API on any default path.** Every blitz-swarm role
runs through the user's own local CLI or local endpoint (`[backend.providers.*]`).
Managed Agents is Anthropic's server-side multi-agent beta
(`managed-agents-2026-04-01`) and bills per token through `ANTHROPIC_API_KEY`.
That makes it the one explicit **rule-#10 exception** in the repo, so it is:

- **opt-in** — you must import `contrib.managed_agents.adapter` yourself and set the
  API key env var; no `blitz.toml` key enables it;
- **fenced** — spend caps per run / per day (`SpendTracker`) fail closed;
- **honest** — no fake "completed" path. The old adapter charged the spend tracker
  and returned `status: completed` without calling anything (audit finding
  backends-02). It now refuses.

## What is implemented

| Piece | File | Tested |
|---|---|---|
| `AgentSpec`, `ManagedAgentsConfig` (model injected by caller, never a literal) | `adapter.py` | yes |
| `validate_spec` (<= `max_specialists`), `check_coordinator_depth` (= 1) | `adapter.py` | yes |
| `SpendTracker` per-run / per-day caps | `adapter.py` | yes |
| `ManagedAgentsAdapter.bootstrap` (requires the API key env var) | `adapter.py` | yes |
| `run_task` dry-run stub / `NotImplementedError` with the call shape | `adapter.py` | yes |
| Session create / events stream | — | **not implemented** |

Tests: `tests/contrib/test_managed_agents.py` (no network).

## If you implement the call

Follow the call shape in `run_task`'s docstring, record spend only from the
provider's reported usage, keep the coordinator depth cap, and keep this package
out of every default code path.
