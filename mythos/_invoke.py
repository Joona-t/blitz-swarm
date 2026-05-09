"""Shared subprocess invocation for Mythos roles.

Wraps `claude -p --model X [--effort Y] --json-schema Z` into a single
helper that returns parsed JSON + cost telemetry. Mirrors the parsing
contract from orchestrator.py::invoke_agent (CLI envelope unwrapping).

Kept inside mythos/ because Mythos has different output schemas, longer
timeouts, and a different retry policy than blitz-swarm consensus mode.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

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
    max_turns: int = 1,
) -> InvokeResult:
    """Invoke a single Mythos role via `claude -p`.

    Returns InvokeResult with parsed dict or error. Never raises on CLI
    failure — wraps the failure into the result so the runner can decide
    what to do (replan, abort, etc.).
    """
    model, effort = resolve_model(model_alias)

    cmd = [
        "claude",
        "-p", user_prompt,
        "--system-prompt", system_prompt,
        "--output-format", "json",
        "--model", model,
        "--max-turns", str(max_turns),
        "--json-schema", schema_json,
        "--dangerously-skip-permissions",
    ]
    if effort:
        cmd.extend(["--effort", effort])

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return InvokeResult(
            parsed=None,
            raw_stdout="",
            cost_usd=0.0,
            input_tokens=0,
            output_tokens=0,
            elapsed_s=time.monotonic() - start,
            error=f"{role} timed out after {timeout_s}s",
        )
    except Exception as e:
        return InvokeResult(
            parsed=None,
            raw_stdout="",
            cost_usd=0.0,
            input_tokens=0,
            output_tokens=0,
            elapsed_s=time.monotonic() - start,
            error=f"{role} subprocess exception: {e!r}",
        )

    elapsed = time.monotonic() - start

    if proc.returncode != 0:
        return InvokeResult(
            parsed=None,
            raw_stdout=proc.stdout,
            cost_usd=0.0,
            input_tokens=0,
            output_tokens=0,
            elapsed_s=elapsed,
            error=f"{role} exit {proc.returncode}: {proc.stderr[:300]}",
        )

    envelope = _parse_envelope(proc.stdout)
    cost = float(envelope.get("total_cost_usd", 0.0)) if envelope else 0.0
    usage = (envelope or {}).get("usage", {}) or {}
    in_tok = int(usage.get("input_tokens", 0))
    out_tok = int(usage.get("output_tokens", 0))

    parsed = _parse_inner_result(envelope, role)

    return InvokeResult(
        parsed=parsed,
        raw_stdout=proc.stdout,
        cost_usd=cost,
        input_tokens=in_tok,
        output_tokens=out_tok,
        elapsed_s=elapsed,
        error=None if parsed else f"{role} returned no parseable JSON",
        _envelope=envelope,
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
    """Extract the role's structured JSON from the envelope's `result` field."""
    if not envelope:
        return None
    inner = envelope.get("result")
    if isinstance(inner, dict):
        return inner
    if isinstance(inner, str):
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            # Try to find an embedded JSON object
            import re
            m = re.search(r"\{.*\}", inner, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
    return None
