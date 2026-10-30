"""Opus orchestrates, Sonnet researches: model roles and the planner's structured-output parsing."""

import json
import subprocess

import agents
from config import SwarmConfig


def _models(plan):
    return {a.id: a.model for a in plan}


def test_default_split_is_opus_orchestrator_sonnet_workers():
    assert SwarmConfig().orchestrator_model == "opus"
    assert SwarmConfig().default_model == "sonnet"
    models = _models(agents.plan_agents("how do swarms reach consensus", use_llm=False, domain="general"))
    assert models["quality_judge"] == "opus"
    assert models["synthesizer"] == "opus"
    workers = {m for i, m in models.items() if i not in ("quality_judge", "synthesizer")}
    assert workers == {"sonnet"}


def test_orchestrator_model_comes_from_config(monkeypatch):
    class Cfg:
        class swarm:
            orchestrator_model = "claude-opus-5-5"
    monkeypatch.setattr("config.load_config", lambda *a, **k: Cfg)
    assert agents._orchestrator_model() == "claude-opus-5-5"


def _fake_run(envelope, calls):
    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(envelope), stderr="")
    return run


def test_planner_reads_structured_output_and_uses_orchestrator_model(monkeypatch):
    plan = {"researcher_count": 9, "critic_count": 0, "needs_fact_checker": True, "subtopics": ["a", "b"]}
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run({"result": "", "structured_output": plan}, calls))
    out = agents._llm_plan("any topic")
    assert out["researcher_count"] == 6 and out["critic_count"] == 1   # clamped
    cmd = calls[0]
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert "--json-schema" in cmd


def test_planner_ignores_keys_on_the_envelope_itself(monkeypatch):
    """Regression: the old code looked for planning keys on the envelope, so every run fell back."""
    envelope = {"researcher_count": 4, "critic_count": 1, "needs_fact_checker": True, "result": ""}
    monkeypatch.setattr(subprocess, "run", _fake_run(envelope, []))
    assert agents._llm_plan("any topic") is None


def test_planner_envelope_error_falls_back(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run({"is_error": True, "subtype": "error_max_turns"}, []))
    assert agents._llm_plan("t") is None
    assert agents._split_subtopics_llm("t", 3) == agents._split_subtopics_heuristic("t", 3)
