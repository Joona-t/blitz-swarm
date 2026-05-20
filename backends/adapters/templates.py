"""Adapter templates for API/framework harnesses.

Adapters for OpenAI API, OpenAI Agents SDK, LangGraph, CrewAI, and Aider
should keep the public contract stateless:

    AgentCall -> adapter-owned execution -> AgentResult

If the harness has sessions, threads, memory, graphs, or crews, keep that
state inside the adapter. Honor AgentCall.ephemeral by creating an
isolated one-shot run and discarding harness state before returning.
"""

from __future__ import annotations

from backends.core import AgentBackend, AgentCall, AgentResult


class OneCallHarnessAdapterTemplate:
    """Minimal shape for future harness adapters.

    This is intentionally not registered. Copy the shape into a concrete
    adapter and normalize all harness-specific errors into AgentResult.
    """

    backend_id = "template"
    adapter_kind = "template"

    def is_available(self) -> bool:
        return False

    def call(self, call: AgentCall) -> AgentResult:
        raise NotImplementedError("Copy this template into a concrete adapter.")


__all__ = ["AgentBackend", "AgentCall", "AgentResult", "OneCallHarnessAdapterTemplate"]
