"""Mythos Swarm — policies, model aliases, cost budget.

The model alias layer decouples our code from Anthropic's naming churn.
"mythos" today maps to ("opus", effort="max"); the day a real Mythos model
ships, we change one dict entry and the whole swarm picks it up.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------

# alias -> (claude --model value, --effort value or None)
# CLI probe (2026-05-09) confirmed: --model opus routes to claude-opus-4-7,
# --effort accepts {low, medium, high, xhigh, max}. Mythos = opus + effort max.
MODEL_ALIASES: dict[str, tuple[str, str | None]] = {
    "mythos": ("opus", "max"),     # extended-thinking Opus; the killer model
    "opus": ("opus", None),        # plain Opus 4.7 — orchestration without thinking budget
    "sonnet": ("sonnet", None),    # workhorse executor
    "haiku": ("haiku", None),      # cheapest executor
}


def resolve_model(alias: str) -> tuple[str, str | None]:
    """Map an alias to (model, effort) for `claude -p` invocation.

    Unknown aliases pass through unchanged with effort=None — lets users
    pass concrete model IDs (e.g. "claude-opus-4-7") directly.
    """
    if alias in MODEL_ALIASES:
        return MODEL_ALIASES[alias]
    return (alias, None)


# ---------------------------------------------------------------------------
# Cost budget — hard ceiling enforcement
# ---------------------------------------------------------------------------


@dataclass
class CostBudget:
    """Track cumulative cost across a Mythos run; abort when ceiling hit."""
    ceiling_usd: float = 5.0
    spent_usd: float = 0.0

    def charge(self, amount_usd: float) -> None:
        self.spent_usd += amount_usd

    def remaining(self) -> float:
        return max(0.0, self.ceiling_usd - self.spent_usd)

    def exceeded(self) -> bool:
        return self.spent_usd >= self.ceiling_usd


# ---------------------------------------------------------------------------
# Mythos config
# ---------------------------------------------------------------------------


@dataclass
class MythosConfig:
    # Models (aliases — resolved via resolve_model at call time)
    planner_model: str = "mythos"
    executor_model: str = "sonnet"
    verifier_model: str = "mythos"

    # Loop budget
    max_replans: int = 3
    max_executors: int = 8         # cap on parallel sub-specs per round

    # Cost ceiling
    cost_ceiling_usd: float = 5.0

    # Timeouts
    planner_timeout_s: int = 600   # extended-thinking can take a while
    executor_timeout_s: int = 300
    verifier_timeout_s: int = 600

    # Output
    output_dir: str = "./output/mythos"

    # Feature flags
    use_redis: bool = False        # default off; Mythos runs are usually one-shot
    use_memory: bool = True        # G-Memory shared blackboard with mode tag


def load_mythos_config(path: Path | None = None) -> MythosConfig:
    """Load mythos.toml overlay; return defaults if missing."""
    cfg = MythosConfig()

    if path is None:
        path = Path(__file__).parent.parent / "mythos.toml"

    if not path.exists():
        return cfg

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except Exception:
        return cfg

    models = raw.get("models", {}) or {}
    for k in ("planner_model", "executor_model", "verifier_model"):
        if k in models:
            setattr(cfg, k, models[k])

    policies = raw.get("policies", {}) or {}
    for k in (
        "max_replans", "max_executors", "cost_ceiling_usd",
        "planner_timeout_s", "executor_timeout_s", "verifier_timeout_s",
        "output_dir", "use_redis", "use_memory",
    ):
        if k in policies:
            setattr(cfg, k, policies[k])

    return cfg
