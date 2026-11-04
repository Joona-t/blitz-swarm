"""WP2: one schema policy.

native   -> schema forwarded untouched, adapter/harness enforces it.
validate -> schema appended to the prompt, reply validated client-side,
            one retry quoting the errors, then an explicit error.
A malformed reply is an ERROR, never a `needs_work` vote; prose with a
brace in it is not "parsed JSON".
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
    parse_json_loose,
    prepare_schema_call,
)

SCHEMA = {
    "type": "object",
    "properties": {"findings": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["findings", "confidence"],
}

NATIVE = BackendCapabilities(adapter_kind="n", schema_policy=backends.SCHEMA_NATIVE, tool_policy=backends.TOOL_ALLOWLIST)
VALIDATE = BackendCapabilities(adapter_kind="v", schema_policy=backends.SCHEMA_VALIDATE, tool_policy=backends.TOOL_ALLOWLIST)
NONE = BackendCapabilities(adapter_kind="z", schema_policy=backends.SCHEMA_NONE, tool_policy=backends.TOOL_ALLOWLIST)


class _Scripted:
    """Adapter stand-in returning one scripted reply per call, validating
    like `normalize_result` would."""

    def __init__(self, replies: list[str]):
        self.backend_id = "s"
        self.adapter_kind = "scripted"
        self.capabilities = VALIDATE
        self.replies = list(replies)
        self.calls: list[AgentCall] = []

    def is_available(self):
        return True

    def call(self, call):
        self.calls.append(call)
        text = self.replies.pop(0)
        return backends.normalize_result(
            backend_id=self.backend_id, adapter_kind=self.adapter_kind, model=call.model,
            text=text, raw_stdout=text, schema=call.schema,
        )


def test_brace_scan_fallback_is_gone():
    assert parse_json_loose('Sure! Here is {"findings": "x"} for you') is None
    assert parse_json_loose('{"findings": "x"}') == {"findings": "x"}
    assert parse_json_loose('```json\n{"findings": "x"}\n```') == {"findings": "x"}
    assert parse_json_loose("") is None


def test_prepare_schema_call_modes():
    call = AgentCall(role="r", prompt="task", schema=SCHEMA, tools=[])
    sent, mode = prepare_schema_call(call, NATIVE)
    assert mode == "native" and sent.prompt == "task" and sent.schema == SCHEMA
    sent, mode = prepare_schema_call(call, VALIDATE)
    assert mode == "validate" and "Output contract" in sent.prompt and '"findings"' in sent.prompt
    assert sent.schema == SCHEMA  # kept for client-side validation
    with pytest.raises(BackendCapabilityError):
        prepare_schema_call(call, NONE)
    assert prepare_schema_call(AgentCall(role="r", prompt="t", tools=[]), NONE) == (
        AgentCall(role="r", prompt="t", tools=[]), "",
    )


def test_validate_mode_retries_once_then_errors():
    adapter = _Scripted(["I think the answer is 42.", "still prose"])
    pb = PolicyBackend(adapter)
    result = pb.call(AgentCall(role="r", prompt="t", model="m", schema=SCHEMA, tools=[]))
    assert len(adapter.calls) == 2
    assert "did not validate" in adapter.calls[1].prompt
    assert result.errored is True
    assert result.error.startswith("schema:")
    assert result.schema_mode == "validate"
    assert result.metadata["schema_retry"] is True
    assert result.parsed is None


def test_validate_mode_retry_can_succeed():
    adapter = _Scripted(["nope", '{"findings": "ok", "confidence": 0.5}'])
    pb = PolicyBackend(adapter)
    result = pb.call(AgentCall(role="r", prompt="t", model="m", schema=SCHEMA, tools=[]))
    assert len(adapter.calls) == 2
    assert result.errored is False and result.parsed == {"findings": "ok", "confidence": 0.5}
    assert result.metadata["schema_retry"] is True


def test_native_mode_does_not_touch_the_prompt_or_retry():
    adapter = _Scripted(['{"findings": "ok", "confidence": 1}'])
    adapter.capabilities = NATIVE
    pb = PolicyBackend(adapter)
    result = pb.call(AgentCall(role="r", prompt="t", model="m", schema=SCHEMA, tools=[]))
    assert adapter.calls[0].prompt == "t" and len(adapter.calls) == 1
    assert result.schema_mode == "native" and result.errored is False


def test_schema_failure_is_an_error_not_a_vote(monkeypatch, tmp_path):
    """Orchestrator path: a reply that fails the schema on both tries must
    come back as `_error`, not as a `needs_work` vote (AUDIT H5)."""
    import orchestrator
    from agents import BlitzAgent

    class _Bad:
        backend_id = "bad"
        adapter_kind = "scripted"
        capabilities = NATIVE

        def is_available(self):
            return True

        def call(self, call):
            return AgentResult(
                backend_id="bad", adapter_kind="scripted", model="m", text="prose",
                raw_stdout="prose", parsed=None, elapsed_s=0.1, errored=True,
                error="schema: $.findings: missing required field",
                validation_errors=["$.findings: missing required field"],
            )

    monkeypatch.setattr(orchestrator, "_backend_settings",
                        lambda **kw: (orchestrator.load_config(), "bad", _FakeProvider(), "read-only"))
    monkeypatch.setattr(orchestrator, "_resolved_for_agent",
                        lambda agent, backend_id, cfg: _FakeResolved())
    agent = BlitzAgent(id="r1", role="researcher", subtopic="s", system_prompt="sys", model="m")
    out = orchestrator.invoke_agent(agent, "", "task", backend=_Bad(), scratch_dir=tmp_path)
    assert out.get("_error") is True
    assert out.get("quality_vote") != "ready"
    assert "schema" in (out.get("findings") or "").lower()


class _FakeProvider:
    approval_policy = "never"
    reasoning_effort = None
    ephemeral = True


class _FakeResolved:
    model = "m"
    effort = None
