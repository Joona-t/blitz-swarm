"""v0.3.0 adapter behaviour: Claude --effort probe + stdin for big prompts,
generic_cli command templates, openai_http chat/completions, data-driven
registry, and the once-per-call [backend] fallback. Every subprocess /
urlopen boundary is mocked; nothing here invokes a model."""

from __future__ import annotations

import json
import subprocess
import textwrap
import urllib.error
from pathlib import Path

import pytest

import backends
from backends import AgentCall, AgentResult, FallbackBackend, make_provider_backend
from backends.adapters import cli as cli_adapters
from backends.adapters import generic_cli as generic_adapters
from backends.adapters import openai_http as openai_adapters
from backends.adapters.templates import render_command
from config import load_config

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "note": {"type": "string"}},
    "required": ["answer"],
}


def _claude_envelope(parsed: dict) -> str:
    return json.dumps({
        "structured_output": parsed,
        "total_cost_usd": 0.01,
        "usage": {"input_tokens": 3, "output_tokens": 4},
    })


@pytest.fixture(autouse=True)
def _reset_claude_probe():
    cli_adapters.reset_claude_probe_cache()
    yield
    cli_adapters.reset_claude_probe_cache()


# ---------------------------------------------------------------------------
# Claude: --effort probe (cached) + stdin for prompts > 100 KB
# ---------------------------------------------------------------------------


def _fake_claude(monkeypatch, *, help_text: str, calls: list):
    monkeypatch.setattr(cli_adapters.shutil, "which", lambda name: "/usr/bin/claude")

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[:2] == ["claude", "--help"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=help_text, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=_claude_envelope({"answer": "ok"}), stderr="")

    monkeypatch.setattr(cli_adapters.subprocess, "run", fake_run)


def test_claude_forwards_effort_when_help_advertises_it_and_probes_once(monkeypatch):
    calls: list = []
    _fake_claude(monkeypatch, help_text="  --effort <level>  Effort level (low, medium, high, xhigh, max)", calls=calls)
    backend = backends.ClaudeCLIBackend()

    r1 = backend.call(AgentCall(role="planner", prompt="p", model="deep-model", reasoning_effort="max", schema=SCHEMA))
    r2 = backend.call(AgentCall(role="planner", prompt="p", model="deep-model", reasoning_effort="max"))

    assert not r1.errored and r1.parsed == {"answer": "ok"}
    assert r1.metadata["effort"] == "max"
    probes = [c for c, _ in calls if c[:2] == ["claude", "--help"]]
    assert len(probes) == 1, "claude --help must be probed once and cached"
    model_calls = [c for c, _ in calls if c[:2] != ["claude", "--help"]]
    assert len(model_calls) == 2
    for cmd in model_calls:
        assert cmd[cmd.index("--effort") + 1] == "max"
        assert cmd[cmd.index("--model") + 1] == "deep-model"
    assert "--json-schema" in model_calls[0] and "--json-schema" not in model_calls[1]
    assert r2.cost_usd == 0.01 and r2.input_tokens == 3 and r2.output_tokens == 4


def test_claude_omits_effort_when_cli_does_not_support_it(monkeypatch):
    calls: list = []
    _fake_claude(monkeypatch, help_text="Usage: claude [options] (no effort flag here)", calls=calls)

    r = backends.ClaudeCLIBackend().call(AgentCall(role="planner", prompt="p", model="m", reasoning_effort="max"))

    assert not r.errored
    cmd = [c for c, _ in calls if c[:2] != ["claude", "--help"]][0]
    assert "--effort" not in cmd


def test_claude_no_effort_means_no_probe_and_no_flag(monkeypatch):
    calls: list = []
    _fake_claude(monkeypatch, help_text="--effort", calls=calls)

    backends.ClaudeCLIBackend().call(AgentCall(role="r", prompt="p", model="m"))

    assert all(c[:2] != ["claude", "--help"] for c, _ in calls)
    assert "--effort" not in calls[0][0]


def test_claude_small_prompt_goes_on_argv_with_devnull_stdin(monkeypatch):
    calls: list = []
    _fake_claude(monkeypatch, help_text="", calls=calls)

    r = backends.ClaudeCLIBackend().call(AgentCall(role="r", prompt="short prompt", model="m"))

    cmd, kwargs = calls[0]
    assert cmd[:3] == ["claude", "-p", "short prompt"]
    assert kwargs.get("stdin") is subprocess.DEVNULL and "input" not in kwargs
    assert r.metadata["prompt_via"] == "argv"


def test_claude_large_prompt_goes_via_stdin(monkeypatch):
    calls: list = []
    _fake_claude(monkeypatch, help_text="", calls=calls)
    big = "x" * (cli_adapters.CLAUDE_STDIN_PROMPT_BYTES + 1)

    r = backends.ClaudeCLIBackend().call(AgentCall(role="r", prompt=big, model="m", system_prompt="sys"))

    cmd, kwargs = calls[0]
    assert big not in cmd
    assert cmd[:2] == ["claude", "-p"] and cmd[2] == "--output-format"
    assert kwargs["input"] == big and "stdin" not in kwargs
    assert cmd[cmd.index("--system-prompt") + 1] == "sys"
    assert r.metadata["prompt_via"] == "stdin"


def test_claude_without_model_is_explicit_error(monkeypatch):
    calls: list = []
    _fake_claude(monkeypatch, help_text="", calls=calls)

    r = backends.ClaudeCLIBackend().call(AgentCall(role="r", prompt="p"))

    assert r.errored and "no model resolved" in r.error
    assert calls == []


# ---------------------------------------------------------------------------
# generic_cli
# ---------------------------------------------------------------------------


def test_render_command_fills_placeholders_and_drops_empty_flag_pairs():
    template = ["tool", "--model", "{model}", "--schema", "{schema_file}", "--effort={effort}", "-f", "{prompt_file}"]
    out = render_command(template, {"model": "m", "schema_file": None, "effort": None, "prompt_file": "/p"})
    assert out == ["tool", "--model", "m", "-f", "/p"]
    with pytest.raises(ValueError):
        render_command(["tool", "{bogus}"], {})


def test_generic_cli_prompt_file_schema_and_output_file(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(generic_adapters.shutil, "which", lambda name: "/usr/bin/mycli")

    def fake_run(cmd, input, capture_output, text, timeout, cwd):
        calls["cmd"] = cmd
        calls["input"] = input
        prompt_path = Path(cmd[cmd.index("--prompt-file") + 1])
        schema_path = Path(cmd[cmd.index("--schema") + 1])
        assert "## System Instructions" in prompt_path.read_text()
        assert json.loads(schema_path.read_text()) == SCHEMA
        Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps({"answer": "from-file"}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ignored stdout", stderr="")

    monkeypatch.setattr(generic_adapters.subprocess, "run", fake_run)

    backend = backends.GenericCLIBackend(
        backend_id="mycli",
        command=["mycli", "--model", "{model}", "--prompt-file", "{prompt_file}",
                 "--schema", "{schema_file}", "--out", "{output_file}", "--effort", "{effort}"],
    )
    r = backend.call(AgentCall(role="r", prompt="hi", system_prompt="sys", schema=SCHEMA, model="m1", cwd=tmp_path))

    assert calls["cmd"][:3] == ["mycli", "--model", "m1"]
    assert "--effort" not in calls["cmd"]          # no effort -> pair dropped
    assert calls["input"] is None                  # prompt went through the file
    assert not r.errored and r.parsed == {"answer": "from-file"}
    assert r.adapter_kind == "generic_cli" and r.backend_id == "mycli"


def test_generic_cli_pipes_prompt_on_stdin_when_template_has_no_prompt_placeholder(monkeypatch):
    calls = {}
    monkeypatch.setattr(generic_adapters.shutil, "which", lambda name: "/usr/bin/mycli")

    def fake_run(cmd, input, **kwargs):
        calls["cmd"], calls["input"] = cmd, input
        return subprocess.CompletedProcess(cmd, 0, stdout='{"answer": "stdout"}', stderr="")

    monkeypatch.setattr(generic_adapters.subprocess, "run", fake_run)

    r = backends.GenericCLIBackend(command=["mycli", "-m", "{model}", "--effort", "{effort}"]).call(
        AgentCall(role="r", prompt="hi", model="m", reasoning_effort="high", schema=SCHEMA)
    )
    assert calls["cmd"] == ["mycli", "-m", "m", "--effort", "high"]
    assert calls["input"] == "hi"
    assert r.parsed == {"answer": "stdout"}


def test_generic_cli_error_paths(monkeypatch):
    r = backends.GenericCLIBackend(backend_id="x").call(AgentCall(role="r", prompt="p", model="m"))
    assert r.errored and "no 'command' template" in r.error

    monkeypatch.setattr(generic_adapters.shutil, "which", lambda name: None)
    r = backends.GenericCLIBackend(command=["ghost"]).call(AgentCall(role="r", prompt="p", model="m"))
    assert r.errored and "not found on PATH" in r.error and backends.is_unavailable_error(r)

    monkeypatch.setattr(generic_adapters.shutil, "which", lambda name: "/usr/bin/t")
    r = backends.GenericCLIBackend(command=["t"]).call(AgentCall(role="r", prompt="p"))
    assert r.errored and "no model resolved" in r.error

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="boom")

    monkeypatch.setattr(generic_adapters.subprocess, "run", fake_run)
    r = backends.GenericCLIBackend(command=["t"]).call(AgentCall(role="r", prompt="p", model="m"))
    assert r.errored and r.error.startswith("t exit 2: boom")


# ---------------------------------------------------------------------------
# openai_http
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, body: dict, status: int = 200):
        self._body = json.dumps(body).encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def test_openai_http_chat_completions_strict_schema_and_usage(monkeypatch):
    calls = {}

    def fake_urlopen(req, timeout):
        calls["url"] = req.full_url
        calls["headers"] = dict(req.header_items())
        calls["body"] = json.loads(req.data.decode())
        calls["timeout"] = timeout
        return _Resp({
            "choices": [{"message": {"role": "assistant", "content": json.dumps({"answer": "a", "note": None})}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        })

    monkeypatch.setattr(openai_adapters.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("LOCAL_TOKEN", "secret")

    backend = backends.OpenAIHTTPBackend(backend_id="local", base_url="http://127.0.0.1:8080/v1/", api_key_env="LOCAL_TOKEN")
    r = backend.call(AgentCall(role="r", prompt="hi", system_prompt="sys", schema=SCHEMA,
                               model="local-model", reasoning_effort="low", timeout_s=9))

    assert calls["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert calls["timeout"] == 9
    assert calls["headers"].get("Authorization") == "Bearer secret"
    body = calls["body"]
    assert body["model"] == "local-model"
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == backends.to_strict_output_schema(SCHEMA)
    assert body["reasoning_effort"] == "low"
    assert not r.errored, r.error
    assert r.parsed == {"answer": "a"}          # strict null stripped
    assert (r.input_tokens, r.output_tokens) == (11, 7)
    assert r.adapter_kind == "openai_http"


def test_openai_http_no_key_env_sends_no_auth_header(monkeypatch):
    calls = {}

    def fake_urlopen(req, timeout):
        calls["headers"] = dict(req.header_items())
        return _Resp({"choices": [{"message": {"content": "plain text"}}]})

    monkeypatch.setattr(openai_adapters.urllib.request, "urlopen", fake_urlopen)
    r = backends.OpenAIHTTPBackend(base_url="http://localhost:1").call(AgentCall(role="r", prompt="p", model="m"))
    assert "Authorization" not in calls["headers"]
    assert not r.errored and r.text == "plain text"


def test_openai_http_errors_are_classified(monkeypatch):
    r = backends.OpenAIHTTPBackend().call(AgentCall(role="r", prompt="p", model="m"))
    assert r.errored and "no base_url" in r.error

    r = backends.OpenAIHTTPBackend(base_url="http://localhost:1").call(AgentCall(role="r", prompt="p"))
    assert r.errored and "no model resolved" in r.error

    def refused(req, timeout):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(openai_adapters.urllib.request, "urlopen", refused)
    r = backends.OpenAIHTTPBackend(base_url="http://localhost:1").call(AgentCall(role="r", prompt="p", model="m"))
    assert r.errored and "request failed" in r.error and backends.is_unavailable_error(r)

    def http503(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)

    monkeypatch.setattr(openai_adapters.urllib.request, "urlopen", http503)
    r = backends.OpenAIHTTPBackend(base_url="http://localhost:1").call(AgentCall(role="r", prompt="p", model="m"))
    assert r.errored and "HTTP 503" in r.error and backends.is_unavailable_error(r)

    def http400(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, None)

    monkeypatch.setattr(openai_adapters.urllib.request, "urlopen", http400)
    r = backends.OpenAIHTTPBackend(base_url="http://localhost:1").call(AgentCall(role="r", prompt="p", model="m"))
    assert r.errored and not backends.is_unavailable_error(r)


def test_openai_http_is_available_probes_models(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        return _Resp({"data": []})

    monkeypatch.setattr(openai_adapters.urllib.request, "urlopen", fake_urlopen)
    assert backends.OpenAIHTTPBackend(base_url="http://localhost:1/v1").is_available()
    assert seen["url"] == "http://localhost:1/v1/models"
    assert not backends.OpenAIHTTPBackend().is_available()


# ---------------------------------------------------------------------------
# Registry: data-driven by adapter id + provider config passthrough
# ---------------------------------------------------------------------------


PROVIDERS_TOML = """
[models]
default = "standard"
planner = "deep"

[backend]
default = "primary"
fallback = ["second", "third"]

[backend.providers.primary]
adapter = "codex_cli"
reasoning_effort = "high"
[backend.providers.primary.tiers]
standard = "p-std"
deep = "p-deep"

[backend.providers.second]
adapter = "claude_cli"
[backend.providers.second.tiers]
standard = "s-std"
deep = "s-deep"
[backend.providers.second.effort]
deep = "max"

[backend.providers.third]
adapter = "ollama_http"
base_url = "http://127.0.0.1:11434"
[backend.providers.third.tiers]
standard = "t-std"

[backend.providers.tool]
adapter = "generic_cli"
command = ["tool", "{model}"]
knob = "v"
[backend.providers.tool.tiers]
standard = "tool-std"

[backend.providers.srv]
adapter = "openai_http"
base_url = "http://127.0.0.1:8080/v1"
api_key_env = "SRV_TOKEN"
[backend.providers.srv.tiers]
standard = "srv-std"

[backend.providers.broken]
model = "x"
"""


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "blitz.toml"
    path.write_text(textwrap.dedent(PROVIDERS_TOML), encoding="utf-8")
    return load_config(path)


def test_registry_builds_every_adapter_from_provider_tables(cfg):
    assert set(backends.registered_backends()) >= {
        "codex_cli", "claude_cli", "gemini_cli", "ollama_http", "generic_cli", "openai_http",
    }
    tool = make_provider_backend("tool", cfg, with_fallback=False)
    assert isinstance(tool, backends.GenericCLIBackend)
    assert tool.command == ["tool", "{model}"] and tool.metadata == {"knob": "v"} and tool.backend_id == "tool"

    srv = make_provider_backend("srv", cfg, with_fallback=False)
    assert isinstance(srv, backends.OpenAIHTTPBackend)
    assert srv.base_url == "http://127.0.0.1:8080/v1" and srv.api_key_env == "SRV_TOKEN"

    primary = make_provider_backend(None, cfg, with_fallback=False)
    assert isinstance(primary, backends.CodexLocalBackend)
    assert primary.model is None and primary.reasoning_effort == "high"

    with pytest.raises(ValueError, match="no 'adapter' key"):
        make_provider_backend("broken", cfg)
    with pytest.raises(ValueError, match="Unknown backend 'ghost'"):
        make_provider_backend("ghost", cfg)


def test_make_provider_backend_wraps_fallback_chain(cfg):
    wrapped = make_provider_backend("primary", cfg)
    assert isinstance(wrapped, FallbackBackend)
    assert wrapped.backend_id == "primary"
    assert [fid for fid, _ in wrapped.fallbacks] == ["second", "third"]
    cfg.backend.fallback = []
    assert isinstance(make_provider_backend("primary", cfg), backends.CodexLocalBackend)


# ---------------------------------------------------------------------------
# Fallback: once per call, only on "unavailable"-class errors
# ---------------------------------------------------------------------------


class _Fake:
    def __init__(self, backend_id, *, available=True, error=None):
        self.backend_id = backend_id
        self.adapter_kind = "fake"
        self.available = available
        self.error = error
        self.calls: list[AgentCall] = []

    def is_available(self):
        return self.available

    def call(self, call):
        self.calls.append(call)
        return AgentResult(
            backend_id=self.backend_id, model=call.model, text="", raw_stdout="",
            parsed=None if self.error else {"ok": True}, elapsed_s=0.0,
            errored=bool(self.error), error=self.error,
        )


def test_fallback_hops_once_and_re_resolves_model_for_the_fallback_provider(cfg, monkeypatch):
    monkeypatch.setenv("BLITZ_MODEL", "cli-override-for-primary")
    primary = _Fake("primary", error="codex CLI not found on PATH")
    second = _Fake("second", available=False)
    third = _Fake("third")
    logs: list[str] = []
    fb = FallbackBackend(primary, [("second", second), ("third", third)], cfg, log=logs.append)

    r = fb.call(AgentCall(role="planner", prompt="p", model="cli-override-for-primary", reasoning_effort="high"))

    assert not r.errored and r.fallback_used
    assert r.backend_id == "third"
    assert second.calls == []                       # unavailable fallback skipped without a call
    assert len(primary.calls) == 1 and len(third.calls) == 1
    hop = third.calls[0]
    assert hop.model == "t-std"                     # planner=deep missing on third -> default tier, via tiers (override ignored)
    assert hop.reasoning_effort is None
    assert r.metadata["fallback_from"] == "primary"
    assert "codex CLI not found on PATH" in r.metadata["fallback_reason"]
    assert any("falling back once" in m for m in logs)


def test_fallback_uses_tier_effort_of_fallback_provider(cfg):
    primary = _Fake("primary", error="openai_http HTTP 503")
    second = _Fake("second")
    fb = FallbackBackend(primary, [("second", second)], cfg, log=lambda m: None)

    r = fb.call(AgentCall(role="planner", prompt="p", model="p-deep"))

    assert r.backend_id == "second" and second.calls[0].model == "s-deep"
    assert second.calls[0].reasoning_effort == "max"


def test_fallback_not_taken_for_timeouts_or_schema_errors(cfg):
    second = _Fake("second")
    for err in ("codex timed out after 5s", "$.answer: missing required field", "codex exit 1: boom"):
        primary = _Fake("primary", error=err)
        fb = FallbackBackend(primary, [("second", second)], cfg, log=lambda m: None)
        r = fb.call(AgentCall(role="planner", prompt="p", model="p-deep"))
        assert r.errored and r.error == err and not r.fallback_used and r.backend_id == "primary"
    assert second.calls == []


def test_fallback_returns_fallback_error_without_a_second_hop(cfg):
    primary = _Fake("primary", error="claude CLI not found on PATH")
    second = _Fake("second", error="claude CLI not found on PATH")
    third = _Fake("third")
    fb = FallbackBackend(primary, [("second", second), ("third", third)], cfg, log=lambda m: None)

    r = fb.call(AgentCall(role="r", prompt="p", model="p-std"))

    assert r.errored and r.backend_id == "second" and r.fallback_used
    assert third.calls == []                        # exactly one hop per call


def test_fallback_skips_provider_without_resolvable_model(cfg):
    cfg.backend.providers["second"].tiers = {}      # second can no longer resolve anything
    primary = _Fake("primary", error="codex CLI not found on PATH")
    second = _Fake("second")
    third = _Fake("third")
    logs: list[str] = []
    fb = FallbackBackend(primary, [("second", second), ("third", third)], cfg, log=logs.append)

    r = fb.call(AgentCall(role="r", prompt="p", model="p-std"))

    assert r.backend_id == "third" and second.calls == []
    assert any("skipped" in m for m in logs)


def test_fallback_backend_availability_is_any_of_chain(cfg):
    fb = FallbackBackend(_Fake("a", available=False), [("b", _Fake("b", available=False))], cfg)
    assert not fb.is_available()
    fb = FallbackBackend(_Fake("a", available=False), [("b", _Fake("b"))], cfg)
    assert fb.is_available()


# ---------------------------------------------------------------------------
# Orchestrator wiring (minimum): _model_for_agent + --list-models
# ---------------------------------------------------------------------------


def test_orchestrator_model_for_agent_uses_resolver(monkeypatch):
    from agents import BlitzAgent
    import orchestrator

    cfg = load_config()
    agent = BlitzAgent(id="synthesizer", role="synthesizer", subtopic="t", system_prompt="s", model="legacy-hint")
    provider = cfg.backend.get_provider("claude")
    assert orchestrator._model_for_agent(agent, "claude", provider, cfg) == "fable"
    monkeypatch.setenv("BLITZ_MODEL_SYNTHESIZER", "fast")
    assert orchestrator._model_for_agent(agent, "claude", provider, cfg) == "haiku"
    monkeypatch.setenv("BLITZ_MODEL", "everything")
    assert orchestrator._model_for_agent(agent, "codex", provider, cfg) == "everything"


def test_orchestrator_list_models_flag_prints_table_without_topic(monkeypatch, capsys):
    import orchestrator

    monkeypatch.setattr("sys.argv", ["orchestrator.py", "--list-models", "--backend", "claude"])
    orchestrator.main()
    out = capsys.readouterr().out
    assert "backend='claude'" in out
    assert "planner" in out and "fable" in out and "max" in out


def test_orchestrator_model_flag_sets_global_override(monkeypatch, capsys):
    import orchestrator

    # setenv first so monkeypatch restores the ORIGINAL (absent) value at
    # teardown; main() then overwrites it in-process.
    monkeypatch.setenv("BLITZ_MODEL", "")
    monkeypatch.setattr("sys.argv", ["orchestrator.py", "--list-models", "--backend", "codex", "--model", "deep"])
    orchestrator.main()
    out = capsys.readouterr().out
    assert __import__("os").environ.get("BLITZ_MODEL") == "deep"
    assert "override" in out


def test_orchestrator_requires_topic_without_list_models(monkeypatch):
    import orchestrator

    monkeypatch.setattr("sys.argv", ["orchestrator.py"])
    with pytest.raises(SystemExit):
        orchestrator.main()
