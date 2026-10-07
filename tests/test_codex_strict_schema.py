"""Regression: Codex --output-schema must be OpenAI strict-mode valid.

codex exec sends --output-schema as a strict json_schema response_format.
The API rejects any object node without `additionalProperties: false` or
with a property missing from `required` (400 invalid_json_schema, codex
exit 1), which made every agent / judge / pairwise call on the default
backend error out. See BUGS_AND_ITERATIONS.md BUG-002.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import backends
from backends import (
    AgentCall,
    CodexLocalBackend,
    strip_strict_nulls,
    to_strict_output_schema,
    validate_json_schema,
)


def _strict_violations(node, path="$") -> list[str]:
    """Check the strict structured-output rules codex/OpenAI enforce."""
    errors: list[str] = []
    if not isinstance(node, dict):
        return errors
    types = node.get("type")
    types = types if isinstance(types, list) else [types]
    if "object" in types:
        props = node.get("properties")
        if props is None:
            errors.append(f"{path}: object without properties")
            props = {}
        if node.get("additionalProperties") is not False:
            errors.append(f"{path}: additionalProperties must be false")
        if sorted(node.get("required") or []) != sorted(props):
            errors.append(f"{path}: required must list every property")
        for key, sub in props.items():
            errors.extend(_strict_violations(sub, f"{path}.{key}"))
    if "array" in types and isinstance(node.get("items"), dict):
        errors.extend(_strict_violations(node["items"], f"{path}[]"))
    return errors


def _real_schemas() -> dict[str, dict]:
    import agents
    import orchestrator
    from mythos import schemas as mythos_schemas

    return {
        "AGENT_OUTPUT_SCHEMA": agents.AGENT_OUTPUT_SCHEMA,
        "PLANNING_SCHEMA": json.loads(agents.PLANNING_SCHEMA),
        "JUDGE_VOTE_SCHEMA": orchestrator.JUDGE_VOTE_SCHEMA,
        "judge_vote_schema(dims)": orchestrator._judge_vote_schema(
            ("accuracy", "coverage", "clarity")
        ),
        "PAIRWISE_VERDICT_SCHEMA": orchestrator.PAIRWISE_VERDICT_SCHEMA,
        "PLANNER_SCHEMA": mythos_schemas.PLANNER_SCHEMA,
        "EXECUTOR_SCHEMA": mythos_schemas.EXECUTOR_SCHEMA,
        "VERIFIER_SCHEMA": mythos_schemas.VERIFIER_SCHEMA,
    }


@pytest.mark.parametrize("name", list(_real_schemas().keys()))
def test_every_real_schema_converts_to_strict_valid(name):
    strict = to_strict_output_schema(_real_schemas()[name])
    assert _strict_violations(strict) == []


def test_raw_schemas_were_not_strict_valid():
    # Guards the test itself: the checker must flag the loose originals.
    assert _strict_violations(_real_schemas()["AGENT_OUTPUT_SCHEMA"])


def test_conversion_does_not_mutate_source():
    import agents

    before = json.dumps(agents.AGENT_OUTPUT_SCHEMA, sort_keys=True)
    to_strict_output_schema(agents.AGENT_OUTPUT_SCHEMA)
    assert json.dumps(agents.AGENT_OUTPUT_SCHEMA, sort_keys=True) == before


def test_optional_fields_become_nullable_and_nulls_are_stripped():
    schema = {
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": ["a", "b"]},
            "note": {"type": "string", "enum": ["x", "y"]},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"k": {"type": "string"}, "v": {"type": "number"}},
                    "required": ["k"],
                },
            },
        },
        "required": ["winner"],
    }
    strict = to_strict_output_schema(schema)
    assert strict["properties"]["note"]["type"] == ["string", "null"]
    assert None in strict["properties"]["note"]["enum"]
    assert strict["properties"]["winner"]["type"] == "string"

    model_out = {"winner": "a", "note": None, "items": [{"k": "z", "v": None}]}
    cleaned = strip_strict_nulls(model_out, schema)
    assert cleaned == {"winner": "a", "items": [{"k": "z"}]}
    assert validate_json_schema(cleaned, schema) == []


def test_codex_backend_writes_strict_schema_and_strips_nulls(monkeypatch, tmp_path):
    import agents

    monkeypatch.setattr(backends.shutil, "which", lambda name: "/usr/bin/codex")
    seen = {}

    def fake_run(cmd, input, capture_output, text, timeout, cwd):
        schema_path = Path(cmd[cmd.index("--output-schema") + 1])
        seen["schema"] = json.loads(schema_path.read_text())
        out = Path(cmd[cmd.index("--output-last-message") + 1])
        payload = {key: None for key in seen["schema"]["properties"]}
        payload.update({
            "findings": "f", "key_points": [], "confidence": 0.5,
            "gaps_identified": [], "quality_vote": "ready",
        })
        out.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(backends.subprocess, "run", fake_run)
    result = CodexLocalBackend().call(AgentCall(
        role="researcher", prompt="p", schema=agents.AGENT_OUTPUT_SCHEMA,
        cwd=tmp_path, timeout_s=5,
    ))
    assert _strict_violations(seen["schema"]) == []
    assert not result.errored, result.error
    assert None not in result.parsed.values()
