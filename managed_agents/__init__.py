"""Anthropic Managed Agents adapter (opt-in alternative orchestrator).

May 7 2026: Managed Agents shipped multiagent sessions + Outcomes in
public beta under header `managed-agents-2026-04-01`. Up to 20 specialist
agents under a coordinator, persistent events, shared filesystem,
LLM-judge "Outcomes" for iterative refinement.

This adapter is OPT-IN — default backend stays local CLI. Activated by
`blitz.toml [evolve] backend = "managed_agents"`. Honors a per-run and
per-day spend cap; fail-closed if the cap is hit.
"""
