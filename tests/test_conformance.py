"""WP6: adapter conformance suite.

The same six canonical calls go through EVERY registered adapter (wrapped
in PolicyBackend exactly as `make_provider_backend` wraps it) against the
fake harnesses in tests/fakes. Identical AgentResult shape, the same
error taxonomy, and the same cost/schema vocabulary are asserted. A new
adapter is "supported" when it is added to ADAPTERS here and passes.
"""

from __future__ import annotations

import dataclasses

import pytest

import backends
from backends import (
    AgentCall,
    AgentResult,
    BackendCapabilityError,
    PolicyBackend,
    is_unavailable_error,
)
from backends.adapters.cli import ClaudeCLIBackend, CodexLocalBackend, GeminiCLIBackend
from backends.adapters.generic_cli import GenericCLIBackend
from backends.adapters.ollama import OllamaHTTPBackend
from backends.adapters.openai_http import OpenAIHTTPBackend
from tests.fakes import CONFORMANCE_SCHEMA, GOOD_PAYLOAD, PLAIN_TEXT, FakeWorld, install

# adapter_id -> factory. Keep in sync with backends.registry builtins; the
# test below fails if the registry knows an adapter this table does not.
ADAPTERS = {
    "codex_cli": lambda: CodexLocalBackend(model="m"),
    "claude_cli": lambda: ClaudeCLIBackend(model="m"),
    "gemini_cli": lambda: GeminiCLIBackend(model="m"),
    "generic_cli": lambda: GenericCLIBackend(command=["tool", "--model", "{model}", "--prompt-file", "{prompt_file}"], model="m"),
    "generic_cli[native-schema]": lambda: GenericCLIBackend(command=["tool", "{model}", "{prompt_file}", "{schema_file}"], model="m"),
    "ollama_http": lambda: OllamaHTTPBackend(model="m"),
    "openai_http": lambda: OpenAIHTTPBackend(base_url="http://127.0.0.1:9/v1", model="m"),
}
# Adapters that cannot restrict tools get the waiver a provider would set.
DEGRADE = {"gemini_cli": ("no_tools",), "generic_cli": ("no_tools",), "generic_cli[native-schema]": ("no_tools",)}

VOCAB_SCHEMA = {backends.SCHEMA_NATIVE, backends.SCHEMA_VALIDATE, ""}
VOCAB_COST = {backends.COST_REPORTED, backends.COST_PRICED, backends.COST_UNKNOWN}


def _backend(adapter_id: str) -> PolicyBackend:
    return PolicyBackend(ADAPTERS[adapter_id](), allow_degrade=DEGRADE.get(adapter_id, ()), log=lambda m: None)


def _assert_shape(r: AgentResult):
    assert isinstance(r, AgentResult)
    for f in dataclasses.fields(AgentResult):
        assert hasattr(r, f.name)
    assert isinstance(r.text, str) and isinstance(r.raw_stdout, str)
    assert isinstance(r.errored, bool)
    assert r.error is None or isinstance(r.error, str)
    assert r.errored == (r.error is not None), "errored and error must agree"
    assert r.cost_usd is None or isinstance(r.cost_usd, float)
    assert r.cost_policy in VOCAB_COST
    assert (r.cost_policy == backends.COST_REPORTED) == isinstance(r.cost_usd, float)
    assert r.schema_mode in VOCAB_SCHEMA
    assert isinstance(r.input_tokens, int) and isinstance(r.output_tokens, int)
    assert isinstance(r.validation_errors, list) and isinstance(r.metadata, dict)


def test_registry_builtins_are_all_covered():
    covered = {k.split("[")[0] for k in ADAPTERS}
    # other tests register throwaway adapters into the process-wide registry
    builtins = {b for b in backends.registered_backends() if not b.startswith(("fake", "test"))}
    assert builtins <= covered, "add the new adapter to ADAPTERS"


@pytest.mark.parametrize("adapter_id", sorted(ADAPTERS))
def test_plain_call(monkeypatch, adapter_id):
    install(monkeypatch, FakeWorld())
    r = _backend(adapter_id).call(AgentCall(role="r", prompt="say hi", tools=[]))
    _assert_shape(r)
    assert not r.errored, r.error
    assert PLAIN_TEXT in r.text
    assert r.parsed is None
    assert r.schema_mode == ""
    assert r.raw_stdout, "raw harness output must be preserved"
    if not adapter_id.startswith("generic_cli"):   # an argv template carries no usage channel
        assert r.input_tokens > 0 and r.output_tokens > 0


@pytest.mark.parametrize("adapter_id", sorted(ADAPTERS))
def test_schema_call(monkeypatch, adapter_id):
    install(monkeypatch, FakeWorld())
    r = _backend(adapter_id).call(AgentCall(role="r", prompt="say hi", schema=CONFORMANCE_SCHEMA, tools=[]))
    _assert_shape(r)
    assert not r.errored, r.error
    assert r.parsed == GOOD_PAYLOAD, "optional null from strict mode must be stripped"
    assert r.schema_mode in (backends.SCHEMA_NATIVE, backends.SCHEMA_VALIDATE)
    assert r.validation_errors == []


@pytest.mark.parametrize("adapter_id", sorted(ADAPTERS))
def test_no_tools_call(monkeypatch, adapter_id):
    world = FakeWorld()
    install(monkeypatch, world)
    caps = ADAPTERS[adapter_id]().capabilities
    call = AgentCall(role="r", prompt="say hi", tools=None)
    if caps.tool_policy == backends.TOOL_NONE:
        with pytest.raises(BackendCapabilityError):
            PolicyBackend(ADAPTERS[adapter_id](), log=lambda m: None).call(call)
        assert world.spawned == [] and world.requests == []
        r = _backend(adapter_id).call(call)            # with the waiver
        assert r.metadata.get("degraded") == ["no_tools"]
        return
    r = _backend(adapter_id).call(call)
    _assert_shape(r)
    assert not r.errored, r.error
    for cmd in world.spawned:
        if cmd[0] == "claude":
            assert cmd[cmd.index("--tools") + 1] == ""
        if cmd[0] == "codex":
            assert cmd[cmd.index("--sandbox") + 1] == "read-only"


@pytest.mark.parametrize("adapter_id", sorted(ADAPTERS))
def test_timeout_call(monkeypatch, adapter_id):
    install(monkeypatch, FakeWorld("timeout"))
    r = _backend(adapter_id).call(AgentCall(role="r", prompt="slow", tools=[], timeout_s=1))
    _assert_shape(r)
    assert r.errored and "timed out" in r.error
    assert not is_unavailable_error(r), "a timeout is not an availability failure (no fallback)"


@pytest.mark.parametrize("adapter_id", sorted(ADAPTERS))
def test_unavailable_call(monkeypatch, adapter_id):
    install(monkeypatch, FakeWorld("unavailable"))
    b = _backend(adapter_id)
    assert b.is_available() is False
    r = b.call(AgentCall(role="r", prompt="hi", tools=[]))
    _assert_shape(r)
    assert r.errored
    assert is_unavailable_error(r), f"unavailable error must trigger fallback: {r.error!r}"


@pytest.mark.parametrize("adapter_id", sorted(ADAPTERS))
def test_no_model_is_an_explicit_error(monkeypatch, adapter_id):
    install(monkeypatch, FakeWorld())
    inner = ADAPTERS[adapter_id]()
    inner.model = None
    r = PolicyBackend(inner, allow_degrade=("no_tools",), log=lambda m: None).call(AgentCall(role="r", prompt="hi", tools=[]))
    _assert_shape(r)
    assert r.errored and "no model" in r.error
