"""Unified local agent invocation backends.

The orchestrator owns state. Backends only turn one prompt into one
structured result with consistent telemetry. Codex is the default local
backend; Claude/Gemini remain optional compatibility backends.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_CODEX_REASONING = "high"
DEFAULT_SANDBOX = "read-only"
DEFAULT_APPROVAL = "never"


@dataclass(slots=True)
class AgentCall:
    role: str
    prompt: str
    system_prompt: str = ""
    schema: dict | None = None
    model: str | None = None
    timeout_s: int = 120
    cwd: Path | None = None
    sandbox: str = DEFAULT_SANDBOX
    approval_policy: str = DEFAULT_APPROVAL
    reasoning_effort: str | None = None
    ephemeral: bool = True


@dataclass(slots=True)
class AgentResult:
    backend_id: str
    model: str | None
    text: str
    raw_stdout: str
    parsed: dict | None
    elapsed_s: float
    errored: bool = False
    error: str | None = None
    fallback_used: bool = False
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    validation_errors: list[str] = field(default_factory=list)


class AgentBackend(Protocol):
    backend_id: str

    def is_available(self) -> bool: ...

    def call(self, call: AgentCall) -> AgentResult: ...


def _combined_prompt(call: AgentCall) -> str:
    if call.system_prompt:
        return (
            f"## System Instructions\n{call.system_prompt}\n\n"
            f"## User Task\n{call.prompt}"
        )
    return call.prompt


def parse_json_loose(text: str) -> dict | None:
    """Parse direct JSON, fenced JSON, or the largest embedded object."""
    if not text or not text.strip():
        return None
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*\n(.*?)\n```", stripped, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(stripped[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def validate_json_schema(data: dict | None, schema: dict | None) -> list[str]:
    """Small JSON-schema subset validator for agent result contracts.

    The project does not depend on jsonschema. This covers the schema
    features used by Blitz/Mythos: object, required, string/number/integer/
    boolean/array, enum, minimum, maximum, and nested array item objects.
    """
    if schema is None:
        return []
    if data is None:
        return ["result is not parseable JSON"]
    return _validate_value(data, schema, "$")


def _validate_value(value, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected object"]
        for key in schema.get("required", []) or []:
            if key not in value:
                errors.append(f"{path}.{key}: missing required field")
        props = schema.get("properties", {}) or {}
        for key, subschema in props.items():
            if key in value:
                errors.extend(_validate_value(value[key], subschema, f"{path}.{key}"))
        return errors
    if expected == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array"]
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                errors.extend(_validate_value(item, item_schema, f"{path}[{i}]"))
        return errors
    if expected == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string")
    elif expected == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{path}: expected number")
    elif expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path}: expected integer")
    elif expected == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean")

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        errors.append(f"{path}: expected one of {enum!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")
    return errors


def _extract_last_codex_agent_message(stdout: str) -> str:
    last = ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, dict) else None
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str):
                last = text
    return last


def _read_text_if_present(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return ""


@dataclass
class CodexLocalBackend:
    model: str = DEFAULT_CODEX_MODEL
    reasoning_effort: str = DEFAULT_CODEX_REASONING
    sandbox: str = DEFAULT_SANDBOX
    approval_policy: str = DEFAULT_APPROVAL
    ephemeral: bool = True
    backend_id: str = "codex"

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    def call(self, call: AgentCall) -> AgentResult:
        if not self.is_available():
            return AgentResult(
                backend_id=self.backend_id,
                model=call.model or self.model,
                text="",
                raw_stdout="",
                parsed=None,
                elapsed_s=0.0,
                errored=True,
                error="codex CLI not found on PATH",
            )

        model = call.model or self.model
        sandbox = call.sandbox or self.sandbox
        approval = call.approval_policy or self.approval_policy
        prompt = _combined_prompt(call)

        with tempfile.TemporaryDirectory(prefix="blitz-codex-") as td:
            tmp = Path(td)
            output_path = tmp / "last_message.txt"
            schema_path: Path | None = None
            if call.schema:
                schema_path = tmp / "schema.json"
                schema_path.write_text(json.dumps(call.schema), encoding="utf-8")

            cmd = [
                "codex",
                "exec",
                "-c", f"approval_policy={json.dumps(approval)}",
                "--model", model,
                "--sandbox", sandbox,
                "--json",
                "--output-last-message", str(output_path),
            ]
            if call.reasoning_effort or self.reasoning_effort:
                cmd.extend([
                    "-c",
                    f"model_reasoning_effort={json.dumps(call.reasoning_effort or self.reasoning_effort)}",
                ])
            if call.cwd:
                cmd.extend(["--cd", str(call.cwd)])
            if call.ephemeral and self.ephemeral:
                cmd.append("--ephemeral")
            if schema_path is not None:
                cmd.extend(["--output-schema", str(schema_path)])
            cmd.append("-")

            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=call.timeout_s,
                    cwd=str(call.cwd) if call.cwd else None,
                )
            except subprocess.TimeoutExpired as exc:
                return AgentResult(
                    backend_id=self.backend_id,
                    model=model,
                    text="",
                    raw_stdout=exc.stdout if isinstance(exc.stdout, str) else "",
                    parsed=None,
                    elapsed_s=time.monotonic() - start,
                    errored=True,
                    error=f"codex timed out after {call.timeout_s}s",
                )
            except Exception as exc:
                return AgentResult(
                    backend_id=self.backend_id,
                    model=model,
                    text="",
                    raw_stdout="",
                    parsed=None,
                    elapsed_s=time.monotonic() - start,
                    errored=True,
                    error=f"codex subprocess exception: {exc!r}",
                )

            elapsed = time.monotonic() - start
            final_text = _read_text_if_present(output_path)
            if not final_text:
                final_text = _extract_last_codex_agent_message(proc.stdout)
            if not final_text:
                final_text = proc.stdout.strip()
            parsed = parse_json_loose(final_text)
            validation_errors = validate_json_schema(parsed, call.schema)
            stderr_preview = (proc.stderr or "")[:500]
            error = None
            if proc.returncode != 0:
                error = f"codex exit {proc.returncode}: {stderr_preview}"
            elif validation_errors:
                error = "; ".join(validation_errors[:3])

            return AgentResult(
                backend_id=self.backend_id,
                model=model,
                text=final_text,
                raw_stdout=proc.stdout,
                parsed=parsed,
                elapsed_s=elapsed,
                errored=proc.returncode != 0 or bool(validation_errors),
                error=error,
                validation_errors=validation_errors,
            )


@dataclass
class ClaudeCLIBackend:
    model: str = "sonnet"
    backend_id: str = "claude"

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    def call(self, call: AgentCall) -> AgentResult:
        if not self.is_available():
            return AgentResult(
                backend_id=self.backend_id, model=call.model or self.model,
                text="", raw_stdout="", parsed=None, elapsed_s=0.0,
                errored=True, error="claude CLI not found on PATH",
            )
        model = call.model or self.model
        cmd = [
            "claude", "-p", call.prompt,
            "--output-format", "json",
            "--model", model,
        ]
        if call.system_prompt:
            cmd.extend(["--system-prompt", call.system_prompt])
        if call.schema:
            cmd.extend(["--json-schema", json.dumps(call.schema)])
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=call.timeout_s,
                cwd=str(call.cwd) if call.cwd else None,
            )
        except subprocess.TimeoutExpired:
            return AgentResult(
                backend_id=self.backend_id, model=model, text="", raw_stdout="",
                parsed=None, elapsed_s=time.monotonic() - start, errored=True,
                error=f"claude timed out after {call.timeout_s}s",
            )
        elapsed = time.monotonic() - start
        envelope = parse_json_loose(proc.stdout)
        text = proc.stdout.strip()
        parsed = None
        cost = 0.0
        in_tok = 0
        out_tok = 0
        if envelope:
            cost = float(envelope.get("total_cost_usd", 0.0) or 0.0)
            usage = envelope.get("usage", {}) or {}
            in_tok = int(usage.get("input_tokens", 0) or 0)
            out_tok = int(usage.get("output_tokens", 0) or 0)
            if isinstance(envelope.get("structured_output"), dict):
                parsed = envelope["structured_output"]
                text = json.dumps(parsed)
            elif isinstance(envelope.get("result"), dict):
                parsed = envelope["result"]
                text = json.dumps(parsed)
            elif isinstance(envelope.get("result"), str):
                text = envelope["result"]
                parsed = parse_json_loose(text)
        if parsed is None:
            parsed = parse_json_loose(text)
        validation_errors = validate_json_schema(parsed, call.schema)
        error = None
        if proc.returncode != 0:
            error = f"claude exit {proc.returncode}: {(proc.stderr or '')[:500]}"
        elif validation_errors:
            error = "; ".join(validation_errors[:3])
        return AgentResult(
            backend_id=self.backend_id, model=model, text=text,
            raw_stdout=proc.stdout, parsed=parsed, elapsed_s=elapsed,
            errored=proc.returncode != 0 or bool(validation_errors),
            error=error, cost_usd=cost, input_tokens=in_tok,
            output_tokens=out_tok, validation_errors=validation_errors,
        )


@dataclass
class GeminiCLIBackend:
    model: str | None = None
    backend_id: str = "gemini"

    def is_available(self) -> bool:
        return shutil.which("gemini") is not None

    def call(self, call: AgentCall) -> AgentResult:
        if not self.is_available():
            return AgentResult(
                backend_id=self.backend_id, model=call.model or self.model,
                text="", raw_stdout="", parsed=None, elapsed_s=0.0,
                errored=True, error="gemini CLI not found on PATH",
            )
        prompt = _combined_prompt(call)
        cmd = ["gemini", "-p", prompt]
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=call.timeout_s,
                cwd=str(call.cwd) if call.cwd else None,
            )
        except subprocess.TimeoutExpired:
            return AgentResult(
                backend_id=self.backend_id, model=call.model or self.model,
                text="", raw_stdout="", parsed=None,
                elapsed_s=time.monotonic() - start, errored=True,
                error=f"gemini timed out after {call.timeout_s}s",
            )
        elapsed = time.monotonic() - start
        text = proc.stdout.strip()
        parsed = parse_json_loose(text)
        validation_errors = validate_json_schema(parsed, call.schema)
        error = None
        if proc.returncode != 0:
            error = f"gemini exit {proc.returncode}: {(proc.stderr or '')[:500]}"
        elif validation_errors:
            error = "; ".join(validation_errors[:3])
        return AgentResult(
            backend_id=self.backend_id, model=call.model or self.model,
            text=text, raw_stdout=proc.stdout, parsed=parsed,
            elapsed_s=elapsed, errored=proc.returncode != 0 or bool(validation_errors),
            error=error, validation_errors=validation_errors,
        )


class BackendRegistry:
    """Select a backend by id and optionally fail over to configured fallbacks."""

    def __init__(
        self,
        backends: list[AgentBackend] | None = None,
        *,
        default_backend: str = "codex",
        fallback_backend: str | None = None,
    ):
        self.backends = {
            b.backend_id: b for b in (
                backends or [CodexLocalBackend(), ClaudeCLIBackend(), GeminiCLIBackend()]
            )
        }
        self.default_backend = default_backend
        self.fallback_backend = fallback_backend

    def get(self, backend_id: str | None = None) -> AgentBackend:
        chosen = backend_id or self.default_backend
        backend = self.backends.get(chosen)
        if backend and backend.is_available():
            return backend
        if self.fallback_backend:
            fallback = self.backends.get(self.fallback_backend)
            if fallback and fallback.is_available():
                return fallback
        if backend:
            return backend
        return CodexLocalBackend()


def make_backend(backend_id: str = "codex", **kwargs) -> AgentBackend:
    if backend_id == "codex":
        return CodexLocalBackend(**kwargs)
    if backend_id == "claude":
        return ClaudeCLIBackend(model=kwargs.get("model", "sonnet"))
    if backend_id == "gemini":
        return GeminiCLIBackend(model=kwargs.get("model"))
    raise ValueError(f"unknown backend: {backend_id}")
