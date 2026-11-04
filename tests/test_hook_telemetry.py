"""Judge-ensemble and selector hook calls carry provider and cost telemetry
and are routed through the `judge` role pin (live run 2026-11-04)."""

from __future__ import annotations

import asyncio

import orchestrator
from backends import AgentCall, AgentResult
from config import load_config


class _Backend:
    def __init__(self, backend_id="pinned", cost=0.01):
        self.backend_id = backend_id
        self.adapter_kind = "fake"
        self.capabilities = None
        self.cost = cost
        self.calls: list[AgentCall] = []

    def is_available(self):
        return True

    def call(self, call):
        self.calls.append(call)
        parsed = {"aggregate_score": 8.0, "rationale": "ok", "rubric_scores": {}} if call.role == "judge_ensemble" \
            else {"winner": "a", "rationale": "r"}
        return AgentResult(backend_id=self.backend_id, adapter_kind="fake", model=call.model, text="{}",
                           raw_stdout="{}", parsed=parsed, elapsed_s=0.0, cost_usd=self.cost,
                           input_tokens=100, output_tokens=10)


class _Provider:
    approval_policy = "never"
    reasoning_effort = None
    ephemeral = True


def test_judge_ensemble_output_carries_telemetry_per_swarm_round(tmp_path):
    cfg = load_config()
    cfg.judge_ensemble.n_judges = 3
    b = _Backend()
    kw = dict(topic="t", backend=b, cfg=cfg, backend_id="codex", provider=_Provider(), sandbox="read-only",
              scratch_dir=tmp_path)
    out1, ens = asyncio.run(orchestrator._run_judge_ensemble(candidate_outputs=[{"role": "researcher", "agent_id": "r", "findings": "x"}],
                                                             history_outputs=[], **kw))
    assert all(c.model_role == "judge" for c in b.calls), "ensemble calls must route via the judge pin"
    assert out1["_backend_id"] == "pinned" and out1["_hook_calls"] == 3
    assert out1["_cost_usd"] == 0.03 and out1["_input_tokens"] == 300
    out2, _ = asyncio.run(orchestrator._run_judge_ensemble(candidate_outputs=[{"role": "researcher", "agent_id": "r", "findings": "y"}],
                                                           history_outputs=[], ensemble=ens, **kw))
    assert out2["_hook_calls"] == 3 and out2["_cost_usd"] == 0.03, "second round reports its own delta, not the total"


def test_unknown_cost_in_any_hook_call_makes_the_total_unknown():
    t = orchestrator._CallTally()
    t.add(AgentResult(backend_id="b", adapter_kind="f", model="m", text="", raw_stdout="", parsed=None, elapsed_s=0, cost_usd=0.5))
    t.add(AgentResult(backend_id="b", adapter_kind="f", model="m", text="", raw_stdout="", parsed=None, elapsed_s=0, cost_usd=None))
    tel = orchestrator._CallTally.telemetry(t.snapshot())
    assert tel["_cost_usd"] is None and tel["_hook_calls"] == 2
