"""WP5: ACP adapter against the fake agent in tests/fakes/acp_agent.py.

Also a live smoke that runs only when an ACP-capable agent is installed
(`gemini --experimental-acp`), skipped with a reason otherwise.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

import backends
from backends import AgentCall, PolicyBackend
from backends.adapters.acp import ACPBackend, _tool_allowed
from tests.fakes import CONFORMANCE_SCHEMA, GOOD_PAYLOAD, PLAIN_TEXT

FAKE = Path(__file__).parent / "fakes" / "acp_agent.py"


def _backend(mode: str, **kw) -> ACPBackend:
    return ACPBackend(command=[sys.executable, str(FAKE), "{model}"], model="m",
                      metadata={"env": {"ACP_FAKE_MODE": mode}, **kw})


def test_registered_and_capabilities():
    assert "acp" in backends.registered_backends()
    caps = ACPBackend().capabilities
    assert caps.schema_policy == backends.SCHEMA_VALIDATE
    assert caps.tool_policy == backends.TOOL_ALLOWLIST
    assert caps.cost_policy == backends.COST_UNKNOWN


def test_plain_prompt_round_trip():
    r = _backend("ok").call(AgentCall(role="r", prompt="hi", tools=[], timeout_s=20))
    assert not r.errored, r.error
    assert r.text == PLAIN_TEXT                      # chunks concatenated
    assert (r.input_tokens, r.output_tokens) == (21, 6)
    assert r.cost_usd is None and r.cost_policy == backends.COST_UNKNOWN
    assert r.metadata["stop_reason"] == "end_turn"
    assert '"session/prompt"' in r.raw_stdout or "session/update" in r.raw_stdout


def test_schema_goes_through_validate_policy():
    pb = PolicyBackend(_backend("ok"), log=lambda m: None)
    r = pb.call(AgentCall(role="r", prompt="hi", schema=CONFORMANCE_SCHEMA, tools=[], timeout_s=20))
    assert not r.errored, r.error
    assert r.parsed == GOOD_PAYLOAD and r.schema_mode == "validate"


def test_no_tools_denies_permission_requests():
    r = _backend("permission").call(AgentCall(role="r", prompt="run ls", tools=None, timeout_s=20))
    assert not r.errored, r.error
    assert r.text == "DENIED"
    assert r.metadata["permissions"] == [{"tool": "Run shell: ls", "allowed": False}]
    assert r.metadata["tool_calls"][0]["status"] == "failed"


def test_allowlist_grants_matching_tool():
    r = _backend("permission").call(AgentCall(role="r", prompt="run ls", tools=["execute"], timeout_s=20))
    assert r.text == "TOOL RAN"
    assert r.metadata["permissions"] == [{"tool": "Run shell: ls", "allowed": True}]
    r = _backend("permission").call(AgentCall(role="r", prompt="run ls", tools=["WebSearch"], timeout_s=20))
    assert r.text == "DENIED"


def test_tool_matching_rules():
    tc = {"kind": "execute", "title": "Run shell: ls"}
    assert _tool_allowed(tc, None) is False
    assert _tool_allowed(tc, []) is False
    assert _tool_allowed(tc, ["execute"]) and _tool_allowed(tc, ["shell"]) and _tool_allowed(tc, ["*"])
    assert not _tool_allowed(tc, ["fetch"])


def test_refusal_is_an_error_with_stop_reason():
    r = _backend("refusal").call(AgentCall(role="r", prompt="hi", tools=[], timeout_s=20))
    assert r.errored and "refusal" in r.error and r.metadata["stop_reason"] == "refusal"
    assert not backends.is_unavailable_error(r)


def test_timeout_kills_the_agent():
    r = _backend("hang").call(AgentCall(role="r", prompt="hi", tools=[], timeout_s=1))
    assert r.errored and "timed out" in r.error
    assert not backends.is_unavailable_error(r)


def test_agent_crash_is_reported_with_stderr():
    r = _backend("crash").call(AgentCall(role="r", prompt="hi", tools=[], timeout_s=20))
    assert r.errored and "exited (code 3)" in r.error and "crashing on purpose" in r.error


def test_missing_binary_is_unavailable_class():
    b = ACPBackend(command=["definitely-not-a-real-agent-binary"], model="m")
    assert b.is_available() is False
    r = b.call(AgentCall(role="r", prompt="hi", tools=[]))
    assert r.errored and backends.is_unavailable_error(r)


def test_model_placeholder_reaches_argv():
    assert _backend("ok")._argv("gemini-x")[-1] == "gemini-x"


def test_provider_table_builds_acp_backend(tmp_path):
    from config import load_config
    toml = tmp_path / "blitz.toml"
    toml.write_text(
        '[backend]\ndefault = "a"\n'
        f'[backend.providers.a]\nadapter = "acp"\ncommand = ["{sys.executable}", "{FAKE}", "{{model}}"]\n'
        'mcp_servers = []\n'
        '[backend.providers.a.tiers]\nfast = "m"\nstandard = "m"\ndeep = "m"\n',
        encoding="utf-8",
    )
    cfg = load_config(toml)
    pb = backends.make_provider_backend("a", cfg, with_fallback=False)
    assert isinstance(pb.inner, ACPBackend)
    r = pb.call(AgentCall(role="r", prompt="hi", model="m", tools=[], timeout_s=20))
    assert not r.errored and r.text == PLAIN_TEXT


@pytest.mark.skipif(shutil.which("gemini") is None, reason="gemini CLI not installed; live ACP smoke skipped")
def test_live_gemini_acp_smoke():
    """Costs one short Gemini call. Only proves the handshake; skipped unless
    the agent is on PATH."""
    b = ACPBackend(command=["gemini", "--experimental-acp"], model=os.environ.get("BLITZ_ACP_SMOKE_MODEL", "flash"))
    r = b.call(AgentCall(role="r", prompt="Reply with the single word pong.", tools=None, timeout_s=120))
    assert not r.errored, r.error
    assert "pong" in r.text.lower()
