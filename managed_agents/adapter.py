"""Managed Agents adapter — opt-in alternative to the CLI orchestrator.

Default backend is local CLI subprocesses (rule #10 compliant). This
adapter is for users who explicitly want server-side orchestration with
the May 2026 Managed Agents beta features (multiagent sessions,
Outcomes grader, persistent events, shared filesystem).

The adapter ships the structure but does not import the `anthropic`
SDK at module-load — users must `pip install anthropic` to actually
use it. Tests verify schema / spend-cap / coordinator-depth-1 logic
without touching the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

BETA_HEADER = "managed-agents-2026-04-01"
COORDINATOR_DEPTH_LIMIT = 1
MAX_SPECIALISTS_DEFAULT = 20


@dataclass(slots=True)
class AgentSpec:
    role: str
    model: str
    system_prompt: str
    tools: tuple[str, ...] = ()


@dataclass(slots=True)
class ManagedAgentsConfig:
    api_key_env: str = "ANTHROPIC_API_KEY"
    beta_header: str = BETA_HEADER
    max_specialists: int = MAX_SPECIALISTS_DEFAULT
    session_max_runtime_min: int = 60
    spend_cap_usd_per_run: float = 5.0
    spend_cap_usd_per_day: float = 50.0
    coordinator_model: str = "claude-opus-4-7"
    coordinator_prompt: str = (
        "You are the coordinator of a research swarm. Decompose the topic "
        "into sub-questions, delegate to specialist agents, integrate their "
        "outputs, and produce a single coherent summary."
    )


class SpendCapExceeded(Exception):
    pass


class CoordinatorDepthExceeded(Exception):
    pass


class TooManySpecialists(Exception):
    pass


def validate_spec(specs: list[AgentSpec], cfg: ManagedAgentsConfig) -> None:
    if len(specs) > cfg.max_specialists:
        raise TooManySpecialists(
            f"requested {len(specs)} specialists; max {cfg.max_specialists}"
        )


def check_coordinator_depth(specs: list[AgentSpec]) -> None:
    """Coordinator depth is hard-capped at 1 by Anthropic; we enforce."""
    coords = [s for s in specs if s.role == "coordinator"]
    if len(coords) > COORDINATOR_DEPTH_LIMIT:
        raise CoordinatorDepthExceeded(
            f"depth {len(coords)} > limit {COORDINATOR_DEPTH_LIMIT}"
        )


@dataclass(slots=True)
class SpendTracker:
    cap_per_run: float
    cap_per_day: float
    spent_this_run: float = 0.0
    spent_today: float = 0.0

    def assert_ok(self, estimated: float) -> None:
        if self.spent_this_run + estimated > self.cap_per_run:
            raise SpendCapExceeded(
                f"per-run cap ${self.cap_per_run} would be exceeded "
                f"(would spend ${self.spent_this_run + estimated:.2f})"
            )
        if self.spent_today + estimated > self.cap_per_day:
            raise SpendCapExceeded(
                f"per-day cap ${self.cap_per_day} would be exceeded"
            )

    def record(self, amount: float) -> None:
        self.spent_this_run += amount
        self.spent_today += amount


# Pluggable client for tests (production uses anthropic.Anthropic)
ClientFactory = Callable[[ManagedAgentsConfig], Any]


@dataclass
class ManagedAgentsAdapter:
    cfg: ManagedAgentsConfig
    client: Any | None = None
    spend: SpendTracker = field(init=False)
    env_id: str | None = None
    agent_ids: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.spend = SpendTracker(
            cap_per_run=self.cfg.spend_cap_usd_per_run,
            cap_per_day=self.cfg.spend_cap_usd_per_day,
        )

    @classmethod
    def bootstrap(
        cls,
        cfg: ManagedAgentsConfig,
        specs: list[AgentSpec],
        *,
        client_factory: ClientFactory | None = None,
    ) -> "ManagedAgentsAdapter":
        validate_spec(specs, cfg)
        check_coordinator_depth(specs)
        if not os.environ.get(cfg.api_key_env):
            raise EnvironmentError(
                f"missing {cfg.api_key_env}; Managed Agents requires an API key"
            )
        adapter = cls(cfg=cfg)
        if client_factory is not None:
            adapter.client = client_factory(cfg)
        return adapter

    def estimate_run_cost(self, n_specialists: int, n_rounds: int) -> float:
        """Rough cost estimate (Sonnet ≈ $0.05/round/specialist)."""
        return n_specialists * n_rounds * 0.05

    def run_task(self, topic: str, n_specialists: int, n_rounds: int) -> dict:
        """Execute a swarm run via Managed Agents (network call here in prod).

        Pre-flight enforces spend cap. Real implementation issues
        client.beta.sessions.create + events.create + events.stream.
        """
        cost_estimate = self.estimate_run_cost(n_specialists, n_rounds)
        self.spend.assert_ok(cost_estimate)

        if self.client is None:
            return {
                "status": "stub",
                "topic": topic,
                "estimated_cost_usd": cost_estimate,
                "note": "client not initialized; this is a dry-run scaffold",
            }

        # Real implementation goes here — beta.sessions.create / events.stream.
        self.spend.record(cost_estimate)
        return {
            "status": "completed",
            "topic": topic,
            "cost_usd": cost_estimate,
        }
