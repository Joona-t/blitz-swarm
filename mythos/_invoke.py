"""Shared invocation for Mythos roles.

One helper: resolve the role's model (tier -> provider model string via
backends.models.resolve_model), build the configured backend (with the
[backend] fallback chain) and return parsed JSON + cost/token telemetry.

Kept inside mythos/ because Mythos has different output schemas, longer
timeouts, and a different retry policy than blitz-swarm consensus mode.

Model precedence for a role (see backends/models.py):
  --model / BLITZ_MODEL  >  BLITZ_MODEL_<ROLE>  >  mythos.toml [models] tier
  >  blitz.toml [models].<role> tier  >  [models].default tier  >  provider
  legacy `model`  >  ModelResolutionError. The provider-wide `model` key never
  shadows the role tier (the v0.2 `provider.model or model` bug).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from backends import AgentCall, make_provider_backend
from backends.models import ENV_OVERRIDE, ResolvedModel, resolve_model, role_env_var
from config import load_config

# Path to prompts/ relative to this file
PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class InvokeResult:
    parsed: dict | None         # parsed JSON output (None on failure)
    raw_stdout: str             # full stdout for debugging
    cost_usd: float             # from the provider envelope; 0.0 when unknown
    input_tokens: int
    output_tokens: int
    elapsed_s: float
    error: str | None = None    # populated on failure
    _envelope: dict | None = field(default=None, repr=False)

    @property
    def tokens(self) -> int:
        return int(self.input_tokens or 0) + int(self.output_tokens or 0)


def load_prompt(name: str) -> str:
    """Load prompts/<name>.md."""
    path = PROMPTS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def resolve_role_model(model_role: str, tier: str | None, cfg, backend_id: str | None = None) -> ResolvedModel:
    """Resolve `model_role` honouring the mythos.toml tier override.

    `tier` (mythos.toml `<role>_model`) wins over blitz.toml `[models]` but
    loses to the explicit `--model` / BLITZ_MODEL / BLITZ_MODEL_<ROLE>
    overrides, so the documented precedence holds.
    """
    backend_id = backend_id or os.environ.get("BLITZ_BACKEND") or cfg.backend.default or None
    var = role_env_var(model_role)
    env_override = os.environ.get(ENV_OVERRIDE) or (os.environ.get(var) if var else None)
    if env_override:
        return resolve_model(model_role, backend_id, cfg)
    return resolve_model(model_role, backend_id, cfg, override=tier or None)


def invoke(
    *,
    role: str,
    system_prompt: str,
    user_prompt: str,
    schema_json: str,
    model_role: str,
    tier: str | None = None,
    timeout_s: int,
    max_turns: int = 4,
) -> InvokeResult:
    """Invoke a single Mythos role through the configured local backend.

    `role` is the label for logs/errors (e.g. "executor[spec-3]"); `model_role`
    is the resolver role ("planner" | "executor" | "verifier"); `tier` is the
    optional mythos.toml tier override for it.

    Returns InvokeResult with parsed dict or error. Never raises on CLI
    failure — wraps the failure into the result so the runner can decide
    what to do (replan, abort, etc.).
    """
    start = time.monotonic()
    try:
        cfg = load_config()
        resolved = resolve_role_model(model_role, tier, cfg)
        provider = cfg.backend.get_provider(resolved.backend_id)
        sandbox = os.environ.get("BLITZ_SANDBOX") or provider.sandbox
        backend = make_provider_backend(resolved.backend_id, cfg, sandbox=sandbox)
        schema = json.loads(schema_json)
    except Exception as e:
        return InvokeResult(
            parsed=None,
            raw_stdout="",
            cost_usd=0.0,
            input_tokens=0,
            output_tokens=0,
            elapsed_s=time.monotonic() - start,
            error=f"{role} backend setup exception: {e!r}",
        )

    res = backend.call(AgentCall(
        role=role,
        prompt=user_prompt,
        system_prompt=system_prompt,
        schema=schema,
        model=resolved.model,
        timeout_s=timeout_s,
        sandbox=sandbox,
        approval_policy=provider.approval_policy,
        reasoning_effort=resolved.effort or provider.reasoning_effort or None,
        ephemeral=provider.ephemeral,
    ))

    parsed = res.parsed
    err = res.error
    if not err and not parsed:
        err = f"{role} returned no parseable JSON"

    return InvokeResult(
        parsed=parsed,
        raw_stdout=res.raw_stdout,
        cost_usd=float(res.cost_usd or 0.0),
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
        elapsed_s=res.elapsed_s,
        error=err,
        _envelope={
            "backend_id": res.backend_id,
            "model": res.model or resolved.model,
            "tier": resolved.tier,
            "effort": resolved.effort,
            "validation_errors": res.validation_errors,
        },
    )
