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


DEFAULT_SANDBOX = "read-only"
DEFAULT_APPROVAL = "never"

# Error substrings that mean "this provider cannot serve right now" (CLI
# missing, server down, rate-limited/overloaded). Timeouts and schema
# failures are NOT in this class: the provider answered, badly.
UNAVAILABLE_MARKERS = (
    "not found on path",
    "connection refused",
    "request failed",
    "http 429",
    "http 502",
    "http 503",
    "http 504",
    "unavailable",
    "overloaded",
    "rate limit",
    "rate_limit",
)


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
    cost_usd: float | None = None     # None = provider did not report cost (never 0)
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


def is_unavailable_error(result: "AgentResult") -> bool:
    """True when `result` errored for an 'unavailable'-class reason.

    `[backend] fallback` is tried exactly once per call on this class only.
    """
    if not result.errored:
        return False
    err = (result.error or "").lower()
    return any(marker in err for marker in UNAVAILABLE_MARKERS)


def to_strict_output_schema(schema: dict) -> dict:
    """Convert a loose JSON schema into OpenAI strict structured-output form.

    `codex exec --output-schema` (and OpenAI-compatible `response_format`
    json_schema with strict=true) rejects any object node that lacks
    `additionalProperties: false` or that does not list every property in
    `required` (400 invalid_json_schema, codex exit 1). Our schemas are
    written loosely (optional keys, free-form objects), so normalise them
    here instead of at every call site:

    - every object gets `additionalProperties: false`;
    - every property is listed in `required`; properties that were
      optional in the source become nullable (`type: [t, "null"]`, and
      `null` is appended to any `enum`) so the model can still omit them;
    - a free-form object (no `properties`) becomes an empty strict object,
      the only object shape strict mode can express without keys.

    `strip_strict_nulls` undoes the nullable widening on the parsed result.
    See BUGS_AND_ITERATIONS.md BUG-002.
    """
    return _strict_node(schema)


def _strict_node(node):
    if not isinstance(node, dict):
        return node
    out = dict(node)
    if out.get("type") == "object":
        props = out.get("properties") or {}
        originally_required = set(out.get("required") or [])
        new_props = {}
        for key, sub in props.items():
            strict_sub = _strict_node(sub)
            if key not in originally_required:
                strict_sub = _make_nullable(strict_sub)
            new_props[key] = strict_sub
        out["properties"] = new_props
        out["required"] = list(new_props.keys())
        out["additionalProperties"] = False
    elif out.get("type") == "array" and isinstance(out.get("items"), dict):
        out["items"] = _strict_node(out["items"])
    return out


def _make_nullable(node: dict) -> dict:
    node = dict(node)
    t = node.get("type")
    if isinstance(t, str) and t != "null":
        node["type"] = [t, "null"]
    elif isinstance(t, list) and "null" not in t:
        node["type"] = [*t, "null"]
    if isinstance(node.get("enum"), list) and None not in node["enum"]:
        node["enum"] = [*node["enum"], None]
    return node


def strip_strict_nulls(value, schema: dict | None):
    """Drop `null` values for keys the *source* schema marked optional.

    Strict mode forces the model to emit every key, using null for the ones
    it would have omitted. Removing them restores the loose contract that
    `validate_json_schema` and downstream `.get(key, default)` callers expect.
    """
    if not isinstance(schema, dict) or not isinstance(value, (dict, list)):
        return value
    if schema.get("type") == "object" and isinstance(value, dict):
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        cleaned = {}
        for key, item in value.items():
            if item is None and key not in required:
                continue
            cleaned[key] = strip_strict_nulls(item, props.get(key))
        return cleaned
    if schema.get("type") == "array" and isinstance(value, list):
        return [strip_strict_nulls(item, schema.get("items")) for item in value]
    return value


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
    cost_usd: float | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    metadata: dict | None = None,
    parsed: dict | None = None,
    strict_nulls: bool = False,
) -> AgentResult:
    """Build an AgentResult from raw text.

    `parsed` short-circuits JSON extraction when the adapter already has the
    object; `strict_nulls=True` strips nulls a strict-mode schema forced in.
    """
    if parsed is None:
        parsed = parse_json_loose(text)
    if strict_nulls:
        parsed = strip_strict_nulls(parsed, schema)
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
