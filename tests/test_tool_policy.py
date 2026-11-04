"""WP3: tool policy per adapter.

`AgentCall.tools=None` means NO tools and must map to a real deny on every
adapter (or be refused). The exact argv is asserted so a flag regression is
a test failure, not a silent change of blast radius.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from backends import AgentCall
from backends.adapters import cli as cli_adapters
from backends.adapters.cli import ClaudeCLIBackend, CodexLocalBackend
from backends.adapters.ollama import OllamaHTTPBackend
from backends.adapters.openai_http import OpenAIHTTPBackend

REPO = Path(__file__).resolve().parents[1]


def _codex_argv(monkeypatch, call: AgentCall) -> list[str]:
    seen = {}
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/codex")

    def fake_run(cmd, input, capture_output, text, timeout, cwd):
        seen["cmd"] = cmd
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli_adapters.subprocess, "run", fake_run)
    CodexLocalBackend(model="gpt-test", sandbox="workspace-write", approval_policy="on-request").call(call)
    return seen["cmd"]


def test_codex_no_tools_forces_read_only_and_never(monkeypatch):
    cmd = _codex_argv(monkeypatch, AgentCall(role="r", prompt="p", tools=None))
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert '"never"' in cmd[cmd.index("-c") + 1]


def test_codex_tool_list_keeps_the_calls_sandbox(monkeypatch):
    # The call's sandbox is authoritative (orchestrator passes the resolved one).
    cmd = _codex_argv(monkeypatch, AgentCall(role="r", prompt="p", tools=["shell"], sandbox="workspace-write"))
    assert cmd[cmd.index("--sandbox") + 1] == "workspace-write"


def test_codex_no_tools_overrides_a_looser_call_sandbox(monkeypatch):
    cmd = _codex_argv(monkeypatch, AgentCall(role="r", prompt="p", tools=None, sandbox="workspace-write"))
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"


def test_codex_refuses_full_access(monkeypatch):
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/codex")
    spawned = []
    monkeypatch.setattr(cli_adapters.subprocess, "run", lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(AssertionError("spawned")))
    result = CodexLocalBackend(model="m").call(
        AgentCall(role="r", prompt="p", tools=["shell"], sandbox="danger-full-access"))
    assert spawned == []
    assert result.errored and "danger-full-access" in result.error


def _claude_argv(monkeypatch, call: AgentCall, help_text: str) -> list[str] | None:
    seen = {}
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/claude")
    cli_adapters.reset_claude_probe_cache()

    def fake_run(cmd, **kw):
        if cmd[:2] == ["claude", "--help"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=help_text, stderr="")
        seen["cmd"] = cmd
        env = {"result": {"a": "x"}, "total_cost_usd": 0.01, "usage": {"input_tokens": 1, "output_tokens": 1}}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(env), stderr="")

    monkeypatch.setattr(cli_adapters.subprocess, "run", fake_run)
    seen["result"] = ClaudeCLIBackend(model="sonnet").call(call)
    cli_adapters.reset_claude_probe_cache()
    return seen


def test_claude_no_tools_sends_empty_tools_flag(monkeypatch):
    seen = _claude_argv(monkeypatch, AgentCall(role="r", prompt="p", tools=None), "--tools --max-turns")
    cmd = seen["cmd"]
    assert cmd[cmd.index("--tools") + 1] == ""


def test_claude_allowlist_is_comma_joined(monkeypatch):
    seen = _claude_argv(monkeypatch, AgentCall(role="r", prompt="p", tools=["WebSearch", "WebFetch"]), "--tools")
    cmd = seen["cmd"]
    assert cmd[cmd.index("--tools") + 1] == "WebSearch,WebFetch"


def test_claude_refuses_no_tools_when_build_lacks_flag(monkeypatch):
    seen = _claude_argv(monkeypatch, AgentCall(role="r", prompt="p", tools=None), "no such flag here")
    assert "cmd" not in seen, "must not spawn a tool-enabled claude for a no-tools call"
    assert seen["result"].errored and "--tools" in seen["result"].error


def test_http_adapters_refuse_tool_requests():
    r = OpenAIHTTPBackend(base_url="http://127.0.0.1:1", model="m").call(AgentCall(role="r", prompt="p", tools=["x"]))
    assert r.errored and "no tools" in r.error
    r = OllamaHTTPBackend(model="m").call(AgentCall(role="r", prompt="p", tools=["x"]))
    assert r.errored and "no tools" in r.error


def test_no_permission_bypass_flags_in_production_code():
    """`--dangerously-skip-permissions`, `--yolo`, `danger-full-access` as a
    default: none may appear outside tests except in a refusal message."""
    offenders = []
    for path in REPO.rglob("*.py"):
        if "tests" in path.parts or ".venv" in path.parts or "scratch" in path.parts:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if ("dangerously" in line or "--yolo" in line or "bypass-approvals" in line) and "not allowed" not in line and "refuse" not in line.lower():
                offenders.append(f"{path.relative_to(REPO)}:{i}: {line.strip()[:80]}")
    assert offenders == [], "\n".join(offenders)
