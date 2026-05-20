"""Tests for unified local agent backends."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

import backends
from backends.adapters import cli as cli_adapters
from backends.adapters import ollama as ollama_adapters
from backends import AgentCall, CodexLocalBackend, validate_json_schema
from config import load_config


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

    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/codex")

    def fake_run(cmd, input, capture_output, text, timeout, cwd):
        calls["cmd"] = cmd
        calls["input"] = input
        calls["cwd"] = cwd
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        schema_path = Path(cmd[cmd.index("--output-schema") + 1])
        assert json.loads(schema_path.read_text()) == AGENT_SCHEMA
        output_path.write_text(json.dumps(_ok_payload()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli_adapters.subprocess, "run", fake_run)

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
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/codex")

    def fake_run(cmd, **kwargs):
        stdout = "\n".join([
            json.dumps({"item": {"type": "agent_message", "text": "not json"}}),
            json.dumps({"item": {"type": "agent_message", "text": json.dumps(_ok_payload())}}),
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(cli_adapters.subprocess, "run", fake_run)

    result = CodexLocalBackend().call(AgentCall(
        role="researcher",
        prompt="hello",
        schema=AGENT_SCHEMA,
    ))

    assert result.parsed == _ok_payload()
    assert not result.errored


def test_codex_backend_prefers_final_file_over_malformed_intermediate(monkeypatch):
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/codex")

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

    monkeypatch.setattr(cli_adapters.subprocess, "run", fake_run)

    result = CodexLocalBackend().call(AgentCall(
        role="researcher",
        prompt="hello",
        schema=AGENT_SCHEMA,
    ))

    assert result.parsed["findings"] == "final"
    assert not result.errored


def test_codex_backend_schema_validation_marks_error(monkeypatch):
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/codex")

    def fake_run(cmd, **kwargs):
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text(json.dumps({"findings": "missing fields"}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli_adapters.subprocess, "run", fake_run)

    result = CodexLocalBackend().call(AgentCall(
        role="researcher",
        prompt="hello",
        schema=AGENT_SCHEMA,
    ))

    assert result.errored
    assert any("missing required field" in err for err in result.validation_errors)


def test_codex_backend_missing_cli_fails_clearly(monkeypatch):
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: None)

    result = CodexLocalBackend().call(AgentCall(role="researcher", prompt="hello"))

    assert result.errored
    assert result.error == "codex CLI not found on PATH"


def test_codex_backend_timeout(monkeypatch):
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/codex")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=3, output="partial")

    monkeypatch.setattr(cli_adapters.subprocess, "run", fake_run)

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


def test_registry_makes_builtin_and_custom_backends():
    class FakeBackend:
        backend_id = "fake"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def is_available(self):
            return True

        def call(self, call):
            return backends.AgentResult(
                backend_id=self.backend_id,
                adapter_kind="fake_adapter",
                model=call.model,
                text="{}",
                raw_stdout="{}",
                parsed={},
                elapsed_s=0.0,
            )

    backends.register_backend(
        "fake_adapter",
        lambda **kw: FakeBackend(**kw),
        aliases=("fake",),
        replace=True,
    )

    builtin = backends.make_backend("codex")
    custom = backends.make_backend("fake", model="m")

    assert builtin.adapter_kind == "codex_cli"
    assert custom.kwargs["backend_id"] == "fake"
    assert custom.kwargs["model"] == "m"
    assert "fake_adapter" in backends.registered_backends()


def test_backend_registry_falls_back_when_default_unavailable():
    class FakeBackend:
        def __init__(self, backend_id, available):
            self.backend_id = backend_id
            self.available = available

        def is_available(self):
            return self.available

        def call(self, call):
            raise AssertionError("not used")

    registry = backends.BackendRegistry(
        [
            FakeBackend("primary", False),
            FakeBackend("fallback", True),
        ],
        default_backend="primary",
        fallback_backend=["fallback"],
    )

    assert registry.get().backend_id == "fallback"


def test_ollama_backend_fake_http(monkeypatch):
    calls = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "response": json.dumps(_ok_payload()),
                "prompt_eval_count": 4,
                "eval_count": 5,
                "done": True,
            }).encode()

    def fake_urlopen(req, timeout):
        calls["url"] = req.full_url
        calls["body"] = json.loads(req.data.decode())
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(ollama_adapters.urllib.request, "urlopen", fake_urlopen)

    result = backends.OllamaHTTPBackend(model="local-model").call(AgentCall(
        role="researcher",
        prompt="hello",
        schema=AGENT_SCHEMA,
        timeout_s=7,
    ))

    assert calls["url"].endswith("/api/generate")
    assert calls["body"]["model"] == "local-model"
    assert calls["body"]["format"] == AGENT_SCHEMA
    assert calls["timeout"] == 7
    assert result.parsed == _ok_payload()
    assert result.adapter_kind == "ollama_http"
    assert result.input_tokens == 4
    assert result.output_tokens == 5


def test_backend_config_supports_provider_registry_and_legacy_aliases(tmp_path):
    cfg_path = tmp_path / "blitz.toml"
    cfg_path.write_text(textwrap.dedent("""
        [backend]
        default = "ollama"
        fallback = ["codex"]

        [backend.providers.ollama]
        adapter = "ollama_http"
        model = "llama3.2"
        base_url = "http://127.0.0.1:11434"

        [backend.codex]
        model = "gpt-test"
        reasoning_effort = "xhigh"
    """), encoding="utf-8")

    cfg = load_config(cfg_path)

    assert cfg.backend.default == "ollama"
    assert cfg.backend.fallback == ["codex"]
    assert cfg.backend.get_provider("ollama").adapter == "ollama_http"
    assert cfg.backend.get_provider("ollama").base_url == "http://127.0.0.1:11434"
    assert cfg.backend.get_provider("codex").model == "gpt-test"
    assert cfg.backend.codex.reasoning_effort == "xhigh"
