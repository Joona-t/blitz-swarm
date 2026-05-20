"""Tests for unified local agent backends."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import backends
from backends import AgentCall, CodexLocalBackend, validate_json_schema


AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "quality_vote": {"type": "string", "enum": ["ready", "needs_work"]},
    },
    "required": ["findings", "confidence", "quality_vote"],
}


def _ok_payload(**overrides):
    payload = {
        "findings": "ok",
        "confidence": 0.8,
        "quality_vote": "ready",
    }
    payload.update(overrides)
    return payload


def test_codex_backend_uses_output_file_schema_and_safe_flags(monkeypatch, tmp_path):
    calls = {}

    monkeypatch.setattr(backends.shutil, "which", lambda name: "/usr/bin/codex")

    def fake_run(cmd, input, capture_output, text, timeout, cwd):
        calls["cmd"] = cmd
        calls["input"] = input
        calls["cwd"] = cwd
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        schema_path = Path(cmd[cmd.index("--output-schema") + 1])
        assert json.loads(schema_path.read_text()) == AGENT_SCHEMA
        output_path.write_text(json.dumps(_ok_payload()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(backends.subprocess, "run", fake_run)

    result = CodexLocalBackend(model="gpt-test").call(AgentCall(
        role="researcher",
        prompt="hello",
        system_prompt="system",
        schema=AGENT_SCHEMA,
        cwd=tmp_path,
        reasoning_effort="xhigh",
        timeout_s=5,
    ))

    cmd = calls["cmd"]
    assert cmd[:2] == ["codex", "exec"]
    assert "--json" in cmd
    assert "--output-last-message" in cmd
    assert "--output-schema" in cmd
    assert "--ephemeral" in cmd
    assert "--cd" in cmd
    assert "--ask-for-approval" not in cmd
    assert any(arg == 'approval_policy="never"' for arg in cmd)
    assert any(arg == 'model_reasoning_effort="xhigh"' for arg in cmd)
    assert cmd[-1] == "-"
    assert "## System Instructions" in calls["input"]
    assert result.parsed == _ok_payload()
    assert not result.errored


def test_codex_backend_falls_back_to_jsonl_agent_message(monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda name: "/usr/bin/codex")

    def fake_run(cmd, **kwargs):
        stdout = "\n".join([
            json.dumps({"item": {"type": "agent_message", "text": "not json"}}),
            json.dumps({"item": {"type": "agent_message", "text": json.dumps(_ok_payload())}}),
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(backends.subprocess, "run", fake_run)

    result = CodexLocalBackend().call(AgentCall(
        role="researcher",
        prompt="hello",
        schema=AGENT_SCHEMA,
    ))

    assert result.parsed == _ok_payload()
    assert not result.errored


def test_codex_backend_prefers_final_file_over_malformed_intermediate(monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda name: "/usr/bin/codex")

    def fake_run(cmd, **kwargs):
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text(json.dumps(_ok_payload(findings="final")), encoding="utf-8")
        stdout = json.dumps({
            "item": {
                "type": "agent_message",
                "text": json.dumps({"findings": "intermediate"}),
            },
        })
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(backends.subprocess, "run", fake_run)

    result = CodexLocalBackend().call(AgentCall(
        role="researcher",
        prompt="hello",
        schema=AGENT_SCHEMA,
    ))

    assert result.parsed["findings"] == "final"
    assert not result.errored


def test_codex_backend_schema_validation_marks_error(monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda name: "/usr/bin/codex")

    def fake_run(cmd, **kwargs):
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text(json.dumps({"findings": "missing fields"}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(backends.subprocess, "run", fake_run)

    result = CodexLocalBackend().call(AgentCall(
        role="researcher",
        prompt="hello",
        schema=AGENT_SCHEMA,
    ))

    assert result.errored
    assert any("missing required field" in err for err in result.validation_errors)


def test_codex_backend_missing_cli_fails_clearly(monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda name: None)

    result = CodexLocalBackend().call(AgentCall(role="researcher", prompt="hello"))

    assert result.errored
    assert result.error == "codex CLI not found on PATH"


def test_codex_backend_timeout(monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda name: "/usr/bin/codex")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=3, output="partial")

    monkeypatch.setattr(backends.subprocess, "run", fake_run)

    result = CodexLocalBackend().call(AgentCall(
        role="researcher",
        prompt="hello",
        timeout_s=3,
    ))

    assert result.errored
    assert "timed out" in result.error


def test_validate_json_schema_subset():
    errors = validate_json_schema(
        {"name": "x", "score": 11, "items": [1, "bad"]},
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["x", "y"]},
                "score": {"type": "number", "minimum": 0, "maximum": 10},
                "items": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["name", "score", "missing"],
        },
    )

    assert "$.missing: missing required field" in errors
    assert "$.score: above maximum 10" in errors
    assert "$.items[1]: expected integer" in errors
