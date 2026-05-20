"""Shared invocation for Mythos roles.

Wraps the configured local backend into a single helper that returns
parsed JSON + cost telemetry. Codex is the default backend for local
runs; Claude remains available through the backend abstraction.

Kept inside mythos/ because Mythos has different output schemas, longer
timeouts, and a different retry policy than blitz-swarm consensus mode.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from backends import AgentCall, make_backend, parse_json_loose
from config import load_config
from .policies import resolve_model

# Path to prompts/ relative to this file
PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class InvokeResult:
    parsed: dict | None         # parsed JSON output (None on failure)
    raw_stdout: str             # full stdout for debugging
    cost_usd: float             # from CLI envelope
    input_tokens: int
    output_tokens: int
    elapsed_s: float
    error: str | None = None    # populated on failure
    _envelope: dict | None = field(default=None, repr=False)


def load_prompt(name: str) -> str:
    """Load prompts/<name>.md."""
    path = PROMPTS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def invoke(
    *,
    role: str,
    system_prompt: str,
    user_prompt: str,
    schema_json: str,
    model_alias: str,
    timeout_s: int,
    max_turns: int = 4,
) -> InvokeResult:
    """Invoke a single Mythos role through the configured local backend.

    Returns InvokeResult with parsed dict or error. Never raises on CLI
    failure — wraps the failure into the result so the runner can decide
    what to do (replan, abort, etc.).
    """
    start = time.monotonic()
    try:
        model, effort = resolve_model(model_alias)
        cfg = load_config()
        backend_id = os.environ.get("BLITZ_BACKEND") or cfg.backend.default or "codex"
        provider = getattr(cfg.backend, backend_id, cfg.backend.codex)
        sandbox = os.environ.get("BLITZ_SANDBOX") or provider.sandbox
        backend_model = provider.model or model
        if backend_id == "claude":
            backend_model = model
        backend = make_backend(
            backend_id,
            model=backend_model,
            reasoning_effort=provider.reasoning_effort,
            sandbox=sandbox,
            approval_policy=provider.approval_policy,
            ephemeral=provider.ephemeral,
        )
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
        model=backend_model,
        timeout_s=timeout_s,
        sandbox=sandbox,
        approval_policy=provider.approval_policy,
        reasoning_effort=effort or provider.reasoning_effort,
        ephemeral=provider.ephemeral,
    ))

    parsed = res.parsed or parse_json_loose(res.text)
    err = res.error
    if not err and not parsed:
        err = f"{role} returned no parseable JSON"

    return InvokeResult(
        parsed=parsed,
        raw_stdout=res.raw_stdout,
        cost_usd=res.cost_usd,
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
        elapsed_s=res.elapsed_s,
        error=err,
        _envelope={
            "backend_id": res.backend_id,
            "model": res.model,
            "validation_errors": res.validation_errors,
        },
    )


def _parse_envelope(stdout: str) -> dict | None:
    """Parse the outer CLI JSON envelope."""
    if not stdout.strip():
        return None
    try:
        env = json.loads(stdout)
        if isinstance(env, dict):
            return env
    except json.JSONDecodeError:
        pass
    return None


def _parse_inner_result(envelope: dict | None, role: str) -> dict | None:
    """Extract the role's structured JSON from the envelope.

    Order of preference:
      1. envelope["structured_output"] — set when --json-schema is used and
         the model emits via tool call (the documented happy path).
      2. envelope["result"] as dict — fallback if CLI ever inlines it.
      3. envelope["result"] as string — try direct JSON parse, then embedded
         {…} extraction (last resort).
    """
    if not envelope:
        return None

    so = envelope.get("structured_output")
    if isinstance(so, dict):
        return so

    inner = envelope.get("result")
    if isinstance(inner, dict):
        return inner
    if isinstance(inner, str) and inner.strip():
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            import re
            m = re.search(r"\{.*\}", inner, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
    return None
