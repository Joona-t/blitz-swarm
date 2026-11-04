"""PR #2 defect 2: the selector path must keep evaluator notes, dissent and
the no-consensus warning in the final document."""

from __future__ import annotations

import asyncio

import orchestrator
from config import load_config
from mechanisms import selector_synth as ss


def test_selector_output_carries_critique_and_dissent(monkeypatch):
    cfg = load_config()

    class _Result:
        final_text = "## Findings\n\nSelected span text."
        selected_span_ids = ["a"]
        diagnostics = {"judge_calls": 0}

    async def fake_synth(self, outputs, *, topic, guard_filter=None):
        return _Result()

    monkeypatch.setattr(ss.SelectorSynth, "synthesize_async", fake_synth)
    monkeypatch.setattr(orchestrator, "_make_pairwise_judge_fn", lambda **kw: (lambda *a, **k: None))
    monkeypatch.setattr(orchestrator, "resolve_model", lambda *a, **k: type("R", (), {"model": "m", "effort": None})())

    feedback = [
        {"role": "critic", "agent_id": "c1", "quality_vote": "needs_work", "quality_notes": "Missing failure modes.", "dissent": "I disagree that WAL is always faster."},
        {"role": "fact_checker", "agent_id": "f1", "quality_vote": "ready", "quality_notes": "Claims check out.", "dissent": ""},
        {"role": "critic", "agent_id": "c2", "quality_vote": "needs_work", "quality_notes": "errored", "dissent": "x", "_error": True},
    ]
    out = asyncio.run(orchestrator._run_selector_synthesis(
        topic="t", researcher_outputs=[{"role": "researcher", "agent_id": "r1", "findings": "## Findings\nx"}],
        guard=None, backend=None, cfg=cfg, backend_id="codex", provider=None, sandbox="read-only",
        feedback_outputs=feedback, warning="The swarm did not reach full consensus.",
    ))
    text = out["findings"]
    assert text.startswith("## Findings")
    assert "## Critique and dissent" in text
    assert "did not reach full consensus" in text
    assert "Missing failure modes." in text and "Claims check out." in text
    assert "I disagree that WAL is always faster." in text
    assert "errored" not in text, "errored evaluators are excluded"
    assert "c1" in out["dissent"] and out["quality_vote"] == "ready"
