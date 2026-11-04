"""WP1: capability negotiation.

A call that requires something the adapter cannot provide must raise
before any process is spawned; a provider may waive a named requirement
with `allow_degrade`, which logs and tags the result, never silently.
"""

from __future__ import annotations

import pytest

import backends
from backends import (
    AgentCall,
    AgentResult,
    BackendCapabilities,
    BackendCapabilityError,
    PolicyBackend,
    check_requirements,
    implied_requirements,
)
from backends.adapters.cli import ClaudeCLIBackend, CodexLocalBackend, GeminiCLIBackend
from backends.adapters.generic_cli import GenericCLIBackend
from backends.adapters.ollama import OllamaHTTPBackend
from backends.adapters.openai_http import OpenAIHTTPBackend

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


class _Recorder:
    """Adapter stand-in: records calls, returns a fixed result."""

    def __init__(self, caps: BackendCapabilities, *, parsed=None, text="", errors=()):
        self.backend_id = "rec"
        self.adapter_kind = caps.adapter_kind
        self.capabilities = caps
        self.calls: list[AgentCall] = []
        self._parsed = parsed
        self._text = text
        self._errors = list(errors)

    def is_available(self):
        return True

    def call(self, call):
        self.calls.append(call)
        return AgentResult(
            backend_id=self.backend_id, adapter_kind=self.adapter_kind, model=call.model,
            text=self._text, raw_stdout=self._text, parsed=self._parsed, elapsed_s=0.0,
            errored=bool(self._errors), error="; ".join(self._errors) or None,
            validation_errors=self._errors,
        )


def test_every_builtin_adapter_declares_the_three_policies():
    for adapter in (CodexLocalBackend(), ClaudeCLIBackend(), GeminiCLIBackend(),
                    GenericCLIBackend(command=["tool", "{model}"]), OllamaHTTPBackend(),
                    OpenAIHTTPBackend(base_url="http://127.0.0.1:1")):
        caps = adapter.capabilities
        assert caps.schema_policy in (backends.SCHEMA_NATIVE, backends.SCHEMA_VALIDATE, backends.SCHEMA_NONE)
        assert caps.tool_policy in (backends.TOOL_ALLOWLIST, backends.TOOL_SANDBOX, backends.TOOL_NONE)
        assert caps.cost_policy in (backends.COST_REPORTED, backends.COST_PRICED, backends.COST_UNKNOWN)


def test_requirement_vocabulary_matches_adapter_facts():
    claude = ClaudeCLIBackend().capabilities
    gemini = GeminiCLIBackend().capabilities
    assert check_requirements(AgentCall(role="r", prompt="p", requires=frozenset({"schema:native", "no_tools", "cost:reported"})), claude) == []
    assert check_requirements(AgentCall(role="r", prompt="p", requires=frozenset({"schema:native"})), gemini) == ["schema:native"]
    assert check_requirements(AgentCall(role="r", prompt="p", requires=frozenset({"no_tools"})), gemini) == ["no_tools"]
    # an unknown word never passes
    assert check_requirements(AgentCall(role="r", prompt="p", requires=frozenset({"telepathy"})), claude) == ["telepathy"]
    # no capabilities object meets nothing
    assert check_requirements(AgentCall(role="r", prompt="p", requires=frozenset({"schema"})), None) == ["schema"]


def test_call_shape_implies_requirements():
    plain = AgentCall(role="r", prompt="p", tools=["WebSearch"])
    assert implied_requirements(plain) == frozenset()
    with_schema_no_tools = AgentCall(role="r", prompt="p", schema=SCHEMA)
    assert implied_requirements(with_schema_no_tools) == frozenset({"schema", "no_tools"})


def test_generic_cli_capabilities_follow_the_template():
    native = GenericCLIBackend(command=["t", "--schema", "{schema_file}", "--effort", "{effort}"]).capabilities
    assert native.schema_policy == backends.SCHEMA_NATIVE and native.effort is True
    prompted = GenericCLIBackend(command=["t", "{prompt_file}"]).capabilities
    assert prompted.schema_policy == backends.SCHEMA_VALIDATE and prompted.effort is False
    assert prompted.tool_policy == backends.TOOL_NONE


def test_policy_backend_refuses_before_calling_the_adapter():
    rec = _Recorder(GeminiCLIBackend().capabilities)
    pb = PolicyBackend(rec)
    with pytest.raises(BackendCapabilityError, match="no_tools"):
        pb.call(AgentCall(role="researcher", prompt="p", model="m"))   # tools=None -> no_tools implied
    assert rec.calls == []


def test_policy_backend_degrades_loudly_when_allowed():
    rec = _Recorder(GeminiCLIBackend().capabilities, parsed={"a": "x"}, text='{"a": "x"}')
    lines: list[str] = []
    pb = PolicyBackend(rec, allow_degrade=("no_tools",), log=lines.append)
    result = pb.call(AgentCall(role="researcher", prompt="p", model="m"))
    assert len(rec.calls) == 1
    assert result.metadata["degraded"] == ["no_tools"]
    assert lines and "DEGRADED" in lines[0] and "no_tools" in lines[0]


def test_policy_backend_schema_native_requirement_against_validate_adapter():
    rec = _Recorder(GeminiCLIBackend().capabilities)
    pb = PolicyBackend(rec, allow_degrade=("no_tools",))
    with pytest.raises(BackendCapabilityError, match="schema:native"):
        pb.call(AgentCall(role="judge", prompt="p", model="m", schema=SCHEMA,
                          requires=frozenset({"schema:native"})))


def test_provider_allow_degrade_is_read_from_the_toml(tmp_path, monkeypatch):
    from config import load_config
    toml = tmp_path / "blitz.toml"
    toml.write_text(
        '[backend]\ndefault = "g"\n'
        '[backend.providers.g]\nadapter = "gemini_cli"\nallow_degrade = ["no_tools"]\n'
        '[backend.providers.g.tiers]\nfast = "gemini-x"\nstandard = "gemini-x"\ndeep = "gemini-x"\n',
        encoding="utf-8",
    )
    cfg = load_config(toml)
    pb = backends.make_provider_backend("g", cfg, with_fallback=False)
    assert isinstance(pb, PolicyBackend)
    assert pb.allow_degrade == ("no_tools",)
    assert isinstance(pb.inner, GeminiCLIBackend)
