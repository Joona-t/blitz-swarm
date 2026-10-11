"""Anthropic Managed Agents adapter — OPT-IN, PAID API (rule-#10 exception).

Lives under contrib/ because it is the one place in blitz-swarm that would
talk to a paid, metered API. It is never on a default path: nothing imports
it unless you do. Read contrib/managed_agents/README.md before enabling.

May 7 2026: Managed Agents shipped multiagent sessions + Outcomes in
public beta under header `managed-agents-2026-04-01`. Up to 20 specialist
agents under a coordinator, persistent events, shared filesystem,
LLM-judge "Outcomes" for iterative refinement.

The network call is NOT implemented (`ManagedAgentsAdapter.run_task` raises
NotImplementedError with the documented call shape) so nobody mistakes the
spend-cap scaffold for a working orchestrator.
"""
