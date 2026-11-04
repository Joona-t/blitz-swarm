"""Fixes from the first live cross-provider run (2026-11-04)."""

from __future__ import annotations

import orchestrator
from agents import BlitzAgent
from backends import AgentResult


def test_final_output_falls_back_to_last_round_with_researchers():
    rounds = [
        [{"role": "researcher", "agent_id": "r1", "findings": "WAL appends.", "quality_vote": None},
         {"role": "critic", "agent_id": "c1", "findings": "ok", "quality_vote": "needs_work", "quality_notes": "n", "confidence": 0.5}],
        [{"role": "synthesizer", "agent_id": "s", "findings": "[ERROR] codex timed out", "_error": True}],
    ]
    doc = orchestrator.format_final_output("t", rounds, synthesizer_output=rounds[1][0])
    assert "## Research Findings" in doc and "WAL appends." in doc


def test_synthesizer_gets_its_own_timeout(monkeypatch, tmp_path):
    seen = {}

    class _B:
        backend_id = "b"
        adapter_kind = "fake"
        capabilities = None

        def is_available(self):
            return True

        def call(self, call):
            seen[call.role] = call.timeout_s
            return AgentResult(backend_id="b", adapter_kind="fake", model="m", text="{}", raw_stdout="{}",
                               parsed={"findings": "x", "key_points": [], "confidence": 1, "quality_vote": "ready"},
                               elapsed_s=0.0)

    monkeypatch.setattr(orchestrator, "_backend_settings",
                        lambda **kw: (orchestrator.load_config(), "b", type("P", (), {"approval_policy": "never", "reasoning_effort": None, "ephemeral": True})(), "read-only"))
    monkeypatch.setattr(orchestrator, "_resolved_for_agent", lambda a, b, c: type("R", (), {"model": "m", "effort": None})())
    monkeypatch.setattr(orchestrator, "AGENT_TIMEOUT_SECONDS", 100)
    monkeypatch.setattr(orchestrator, "SYNTHESIZER_TIMEOUT_SECONDS", 250)
    for role in ("researcher", "synthesizer"):
        orchestrator.invoke_agent(BlitzAgent(id=role, role=role, subtopic="s", system_prompt="p", model="m"),
                                  "", "task", backend=_B(), scratch_dir=tmp_path)
    assert seen == {"researcher": 100, "synthesizer": 250}


def test_timeout_errors_are_counted_as_timeouts():
    from metrics import MetricsCollector
    mc = MetricsCollector()
    mc.start_run("r", "t")
    mc.start_round("synthesis")
    mc.end_round("synthesis", [{"_error": True, "findings": "[ERROR] codex timed out after 300s"}])
    assert mc.run.rounds["synthesis"]["agents_timeout"] == 1 and mc.run.rounds["synthesis"]["agents_error"] == 0
