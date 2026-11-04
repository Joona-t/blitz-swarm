"""WP4: cost normalisation.

Every adapter says HOW it knows the cost: reported | priced | unknown.
Unknown is never printed as $0; cache tokens count as input; Codex JSONL
usage events are summed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from backends import AgentCall, COST_REPORTED, COST_UNKNOWN
from backends.adapters import cli as cli_adapters
from backends.adapters.cli import ClaudeCLIBackend, CodexLocalBackend, GeminiCLIBackend, _codex_error_message, _codex_usage


def test_codex_usage_sums_jsonl_events():
    stdout = "\n".join([
        json.dumps({"type": "thread.started"}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 50}}),
        "not json",
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}),
    ])
    assert _codex_usage(stdout) == (1010, 400, 55)
    assert _codex_usage("") == (0, 0, 0)


def test_codex_result_carries_tokens_and_unknown_cost(monkeypatch):
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/codex")
    events = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 7, "cached_input_tokens": 2, "output_tokens": 3}})

    def fake_run(cmd, input, capture_output, text, timeout, cwd):
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text('{"a": 1}', encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout=events + "\n", stderr="")

    monkeypatch.setattr(cli_adapters.subprocess, "run", fake_run)
    r = CodexLocalBackend(model="m").call(AgentCall(role="r", prompt="p", tools=[]))
    assert (r.input_tokens, r.output_tokens) == (7, 3)
    assert r.cost_usd is None and r.cost_policy == COST_UNKNOWN
    assert r.metadata["cached_input_tokens"] == 2


def _claude(monkeypatch, envelope: dict):
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/claude")
    cli_adapters.reset_claude_probe_cache()

    def fake_run(cmd, **kw):
        if cmd[:2] == ["claude", "--help"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="--tools", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(envelope), stderr="")

    monkeypatch.setattr(cli_adapters.subprocess, "run", fake_run)
    r = ClaudeCLIBackend(model="m").call(AgentCall(role="r", prompt="p"))
    cli_adapters.reset_claude_probe_cache()
    return r


def test_claude_counts_cache_tokens_as_input_and_reports_cost(monkeypatch):
    r = _claude(monkeypatch, {
        "result": {"a": 1}, "total_cost_usd": 0.0123,
        "usage": {"input_tokens": 5, "cache_creation_input_tokens": 3000, "cache_read_input_tokens": 12000, "output_tokens": 40},
    })
    assert r.input_tokens == 15005 and r.output_tokens == 40
    assert r.cost_usd == 0.0123 and r.cost_policy == COST_REPORTED
    assert r.metadata["cache_read_input_tokens"] == 12000


def test_claude_without_cost_field_is_unknown_not_zero(monkeypatch):
    r = _claude(monkeypatch, {"result": {"a": 1}, "usage": {"input_tokens": 5, "output_tokens": 1}})
    assert r.cost_usd is None and r.cost_policy == COST_UNKNOWN


def test_gemini_envelope_tokens_and_unknown_cost(monkeypatch):
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/gemini")
    env = {"response": '{"a": 1}', "stats": {"input_tokens": 9, "output_tokens": 4}}
    monkeypatch.setattr(cli_adapters.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(env), stderr=""))
    r = GeminiCLIBackend(model="m").call(AgentCall(role="r", prompt="p", tools=[]))
    assert r.parsed == {"a": 1} and (r.input_tokens, r.output_tokens) == (9, 4)
    assert r.cost_policy == COST_UNKNOWN and r.cost_usd is None


def test_metrics_round_policy_and_lower_bound_flag(tmp_path):
    from metrics import MetricsCollector
    mc = MetricsCollector()
    mc.start_run("run", "topic")
    mc.start_round("r1")
    mc.end_round("r1", [
        {"_cost_usd": 0.5, "_input_tokens": 10, "_output_tokens": 1},
        {"_cost_usd": None, "_input_tokens": 10, "_output_tokens": 1, "_backend_id": "nobody", "_model": "m"},
    ])
    rm = mc.run.rounds["r1"]
    assert rm["cost_policy"] == "unknown" and rm["cost_usd"] == 0.5
    assert rm["providers"] == ["nobody"] and rm["providers_by_role"] == {"?": "nobody"}
    assert mc.run.cost_incomplete is True
    mc.start_round("r2")
    mc.end_round("r2", [{"_cost_usd": 0.25, "_input_tokens": 1, "_output_tokens": 1}])
    assert mc.run.rounds["r2"]["cost_policy"] == "reported"


def test_codex_error_message_comes_from_jsonl_when_stderr_is_empty(monkeypatch):
    stream = "\n".join([
        json.dumps({"type": "turn.started"}),
        json.dumps({"type": "error", "message": "The 'x' model is not supported when using Codex with a ChatGPT account."}),
        json.dumps({"type": "turn.failed", "error": {"message": "The 'x' model is not supported when using Codex with a ChatGPT account."}}),
    ])
    assert "not supported" in _codex_error_message(stream)
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(cli_adapters.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout=stream, stderr=""))
    r = CodexLocalBackend(model="x").call(AgentCall(role="r", prompt="p", tools=[]))
    assert r.errored and "not supported" in r.error
