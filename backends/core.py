"""Core backend contract for Blitz-Swarm.

The swarm core only knows how to send one AgentCall and receive one
AgentResult. CLI tools, APIs, and stateful harnesses are adapters behind
that contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_CODEX_REASONING = "high"
DEFAULT_SANDBOX = "read-only"
DEFAULT_APPROVAL = "never"


@dataclass(slots=True)
class RuntimePolicy:
    timeout_s: int = 120
    cwd: Path | None = None
    sandbox: str = DEFAULT_SANDBOX
    approval_policy: str = DEFAULT_APPROVAL
    reasoning_effort: str | None = None
    ephemeral: bool = True

    @classmethod
    def from_call(cls, call: "AgentCall") -> "RuntimePolicy":
        return cls(
            timeout_s=call.timeout_s,
            cwd=call.cwd,
            sandbox=call.sandbox,
            approval_policy=call.approval_policy,
            reasoning_effort=call.reasoning_effort,
            ephemeral=call.ephemeral,
        )


@dataclass(slots=True)
class AgentCall:
    role: str
    prompt: str
    system_prompt: str = ""
    schema: dict | None = None
    model: str | None = None
    # Measured live claude calls run 150-240s; 120 starved them (C5b).
    timeout_s: int = 600
    cwd: Path | None = None
    sandbox: str = DEFAULT_SANDBOX
    approval_policy: str = DEFAULT_APPROVAL
    reasoning_effort: str | None = None
    ephemeral: bool = True
    metadata: dict = field(default_factory=dict)

    @property
    def policy(self) -> RuntimePolicy:
        return RuntimePolicy.from_call(self)


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
    adapter_kind: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def raw_text(self) -> str:
        return self.raw_stdout or self.text


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    adapter_kind: str
    structured_output: bool = False
    schema_enforcement: bool = False
    streaming: bool = False
    tool_sandbox: bool = False
    stateful: bool = False
    local: bool = True
    supports_ephemeral: bool = True


class AgentBackend(Protocol):
    backend_id: str

    def is_available(self) -> bool: ...

    def call(self, call: AgentCall) -> AgentResult: ...


def combined_prompt(call: AgentCall) -> str:
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
    """Small JSON-schema subset validator for agent result contracts."""
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


def normalize_result(
    *,
    backend_id: str,
    adapter_kind: str,
    model: str | None,
    text: str,
    raw_stdout: str = "",
    schema: dict | None = None,
    elapsed_s: float = 0.0,
    errored: bool = False,
    error: str | None = None,
    cost_usd: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    metadata: dict | None = None,
) -> AgentResult:
    parsed = parse_json_loose(text)
    validation_errors = validate_json_schema(parsed, schema)
    final_error = error
    if not final_error and validation_errors:
        final_error = "; ".join(validation_errors[:3])
    return AgentResult(
        backend_id=backend_id,
        adapter_kind=adapter_kind,
        model=model,
        text=text,
        raw_stdout=raw_stdout,
        parsed=parsed,
        elapsed_s=elapsed_s,
        errored=errored or bool(validation_errors),
        error=final_error,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        validation_errors=validation_errors,
        metadata=metadata or {},
    )
