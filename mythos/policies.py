"""Mythos Swarm — role tiers, cost/token budget, config overlay.

Mythos names ROLES (planner / executor / verifier). Each role's model comes
from `backends.models.resolve_model(role, backend_id, cfg, override=<tier>)`
where `<tier>` is the optional `[models]` value in mythos.toml ("deep",
"fast", ...). The tier resolves through the active backend's
`[backend.providers.<id>.tiers]` table in blitz.toml, so switching
`--backend` switches every Mythos model with it. No model name lives here.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

MYTHOS_ROLES: tuple[str, ...] = ("planner", "executor", "verifier")


# ---------------------------------------------------------------------------
# Budget — cost ceiling (only when cost is known) + token ceiling (always)
# ---------------------------------------------------------------------------


@dataclass
class CostBudget:
    """Track cumulative spend across a Mythos run; abort when a ceiling hits.

    `ceiling_usd` applies only to cost the provider actually reported (an
    unknown cost is charged as nothing, never as 0 toward a false "fine").
    `token_ceiling` (input + output tokens) applies always; 0 disables it.
    """
    ceiling_usd: float = 5.0
    spent_usd: float = 0.0
    token_ceiling: int = 0
    tokens_spent: int = 0

    def charge(self, amount_usd: float | None, tokens: int = 0) -> None:
        if amount_usd is not None:
            self.spent_usd += amount_usd
        self.tokens_spent += max(0, int(tokens or 0))

    def remaining(self) -> float:
        return max(0.0, self.ceiling_usd - self.spent_usd)

    def cost_exceeded(self) -> bool:
        return self.spent_usd >= self.ceiling_usd

    def tokens_exceeded(self) -> bool:
        return self.token_ceiling > 0 and self.tokens_spent >= self.token_ceiling

    def exceeded(self) -> bool:
        return self.cost_exceeded() or self.tokens_exceeded()


# ---------------------------------------------------------------------------
# Mythos config
# ---------------------------------------------------------------------------


@dataclass
class MythosConfig:
    # Per-role TIER override (mythos.toml [models]). Empty = use blitz.toml
    # [models].<role>. Env BLITZ_MODEL / BLITZ_MODEL_<ROLE> beat both.
    backend: str = ""              # mythos.toml [models] backend: provider id used by --mode mythos
                                   # unless --backend / BLITZ_BACKEND is given ("" = blitz.toml default)
    planner_model: str = ""
    executor_model: str = ""
    verifier_model: str = ""

    # Loop budget
    max_replans: int = 3
    max_executors: int = 8         # cap on parallel sub-specs per round

    # Ceilings
    cost_ceiling_usd: float = 5.0  # applies only when the provider reports cost
    token_ceiling: int = 2_000_000 # input + output tokens; applies always

    # Timeouts
    planner_timeout_s: int = 600   # extended-thinking can take a while
    executor_timeout_s: int = 300
    verifier_timeout_s: int = 600

    # Verifier gates (backends-04). The LLM verdict alone never passes a round:
    #  - mechanical check: when a deliverable contains a test file, the
    #    artifact's own tests run in a subprocess and must pass;
    #  - confidence gate: a "pass" below `verifier_min_confidence` is demoted
    #    to needs_work.
    verifier_run_tests: bool = True
    verifier_test_timeout_s: int = 120
    verifier_min_confidence: float = 0.6

    # Output
    output_dir: str = "./output/mythos"

    # Feature flags
    use_redis: bool = False        # default off; Mythos runs are usually one-shot
    use_memory: bool = True        # G-Memory shared blackboard with mode tag (WP2 wires it)

    def tier_for(self, role: str) -> str | None:
        """The mythos.toml tier override for `role`, or None."""
        return getattr(self, f"{role}_model", "") or None


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
    for k in ("backend", "planner_model", "executor_model", "verifier_model"):
        if k in models:
            setattr(cfg, k, str(models[k]))

    policies = raw.get("policies", {}) or {}
    for k in (
        "max_replans", "max_executors", "cost_ceiling_usd", "token_ceiling",
        "planner_timeout_s", "executor_timeout_s", "verifier_timeout_s",
        "verifier_run_tests", "verifier_test_timeout_s", "verifier_min_confidence",
        "output_dir", "use_redis", "use_memory",
    ):
        if k in policies:
            setattr(cfg, k, policies[k])

    return cfg
