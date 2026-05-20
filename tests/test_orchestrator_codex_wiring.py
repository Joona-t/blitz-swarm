"""Regression tests for harness-agnostic orchestrator wiring."""

from __future__ import annotations

from pathlib import Path

from agents import BlitzAgent
from backends import AgentCall, AgentResult
from config import load_config
from mechanisms.cascade_guard import CascadeGuard
from mechanisms.cascade_guard import GuardConfig as CascadeGuardConfig
from orchestrator import (
    _apply_guard,
    _filter_guarded,
    _prune_low_contribution_agents,
    _run_judge_ensemble,
    _run_selector_synthesis,
    build_context,
    invoke_agent,
)


class FakeBackend:
    backend_id = "codex"

    def __init__(self):
        self.calls: list[AgentCall] = []

    def is_available(self) -> bool:
        return True

    def call(self, call: AgentCall) -> AgentResult:
        self.calls.append(call)
        if call.role == "judge_ensemble":
            parsed = {
                "aggregate_score": 8.0,
                "rationale": "stable",
                "rubric_scores": {
                    "coverage": 8,
                    "accuracy": 8,
                    "clarity": 8,
                    "depth": 8,
                },
            }
        elif call.role == "selector":
            parsed = {"winner": "a", "rationale": "more specific"}
        else:
            parsed = {
                "findings": "## Finding\n\nBackend routed output.",
                "key_points": ["routed"],
                "confidence": 0.9,
                "gaps_identified": [],
                "quality_vote": "ready",
                "quality_notes": "ok",
                "dissent": "",
            }
        return AgentResult(
            backend_id=self.backend_id,
            model=call.model,
            text="",
            raw_stdout="{}",
            parsed=parsed,
            elapsed_s=0.01,
        )


def _agent(agent_id: str, role: str) -> BlitzAgent:
    return BlitzAgent(
        id=agent_id,
        role=role,
        subtopic="topic",
        system_prompt="system",
        model="sonnet",
    )


def test_invoke_agent_uses_backend_layer_and_codex_model():
    cfg = load_config()
    fake = FakeBackend()
    agent = _agent("researcher_00", "researcher")

    output = invoke_agent(
        agent,
        context="",
        task="research",
        backend=fake,
        backend_id="codex",
        sandbox="read-only",
        cfg=cfg,
    )

    assert output["findings"].startswith("## Finding")
    assert output["backend_id"] == "codex"
    assert fake.calls[0].model == cfg.backend.codex.model
    assert fake.calls[0].sandbox == "read-only"
    assert fake.calls[0].approval_policy == "never"


def test_cascade_guard_excludes_errored_output_from_next_context():
    guard = CascadeGuard(CascadeGuardConfig(mode="balanced"))
    outputs = _apply_guard(
        guard,
        [
            {
                "agent_id": "researcher_bad",
                "role": "researcher",
                "findings": "[ERROR] failed",
                "_error": True,
            },
            {
                "agent_id": "researcher_good",
                "role": "researcher",
                "findings": "Good finding.",
                "key_points": ["good"],
                "confidence": 0.8,
            },
        ],
        round_n=1,
    )

    context = build_context(_filter_guarded(guard, outputs), for_role="critic")

    assert "researcher_bad" not in context
    assert "Good finding" in context


def test_judge_ensemble_produces_quality_judge_vote_with_fake_backend():
    cfg = load_config()
    cfg.judge_ensemble.n_judges = 2
    cfg.judge_ensemble.min_rounds = 1
    fake = FakeBackend()

    output = _run_judge_ensemble(
        topic="SQLite WAL",
        candidate_outputs=[
            {
                "agent_id": "researcher_00",
                "role": "researcher",
                "findings": "Useful research.",
                "key_points": ["useful"],
                "confidence": 0.8,
            }
        ],
        history_outputs=[],
        backend=fake,
        cfg=cfg,
        backend_id="codex",
        provider=cfg.backend.get_provider("codex"),
        sandbox="read-only",
    )

    assert output["role"] == "quality_judge"
    assert output["_judge_ensemble"] is True
    assert output["quality_vote"] == "ready"
    assert output["coverage_score"] == 8


def test_selector_synthesis_replaces_final_blended_synth_with_fake_backend():
    cfg = load_config()
    cfg.selector.n_judges = 1
    fake = FakeBackend()

    output = _run_selector_synthesis(
        topic="SQLite WAL",
        researcher_outputs=[
            {
                "agent_id": "researcher_00",
                "role": "researcher",
                "findings": "## Internals\n\nSQLite WAL appends frames before checkpointing.",
                "key_points": ["frames"],
                "confidence": 0.8,
            }
        ],
        guard=None,
        backend=fake,
        cfg=cfg,
        backend_id="codex",
        provider=cfg.backend.get_provider("codex"),
        sandbox="read-only",
    )

    assert output["role"] == "synthesizer"
    assert output["_selector_synth"] is True
    assert "SQLite WAL appends frames" in output["findings"]


def test_agent_dropout_preserves_researcher_and_critic_floors():
    agents = [
        _agent("researcher_00", "researcher"),
        _agent("researcher_01", "researcher"),
        _agent("researcher_02", "researcher"),
        _agent("critic", "critic"),
    ]
    outputs = [
        {"agent_id": "researcher_00", "role": "researcher", "_error": True, "findings": ""},
        {"agent_id": "researcher_01", "role": "researcher", "_error": True, "findings": ""},
        {"agent_id": "researcher_02", "role": "researcher", "_error": True, "findings": ""},
        {"agent_id": "critic", "role": "critic", "_error": True, "findings": ""},
    ]

    pruned = _prune_low_contribution_agents(agents, outputs, selector_enabled=True)

    assert sum(1 for a in pruned if a.role == "researcher") >= 2
    assert sum(1 for a in pruned if a.role == "critic") >= 1


def test_no_direct_claude_subprocess_calls_remain_on_active_paths(repo_root: Path):
    active_paths = [
        "orchestrator.py",
        "agents.py",
        "memory/reader.py",
        "memory/writer.py",
        "mythos/_invoke.py",
    ]
    for rel in active_paths:
        text = (repo_root / rel).read_text(encoding="utf-8")
        assert '"claude", "-p"' not in text
        assert "--dangerously-skip-permissions" not in text
        assert 'if backend_id == "claude"' not in text
        assert "getattr(cfg.backend" not in text
        assert "CodexLocalBackend" not in text
        assert "ClaudeCLIBackend" not in text
        assert "GeminiCLIBackend" not in text
