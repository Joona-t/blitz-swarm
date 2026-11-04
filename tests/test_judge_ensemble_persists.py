"""The judge ensemble persists across swarm rounds and scores each round
exactly once (n_judges calls), instead of re-scoring the same text in a
`while not is_stable()` loop (AUDIT 2026-10-27; PR #2 defect 1)."""

from __future__ import annotations

import asyncio

import orchestrator
from config import load_config
from mechanisms.judge_ensemble import JudgeVote


def test_one_scoring_per_swarm_round_and_state_carries_over():
    cfg = load_config()
    cfg.judge_ensemble.n_judges = 3
    calls: list[tuple[str, int]] = []

    def judge_fn(candidate, debate_history, judge_id, model, seed, rubric_dims) -> JudgeVote:
        calls.append((judge_id, seed))
        return JudgeVote(judge_id=judge_id, round_n=0, rubric_scores={d: 8.0 for d in rubric_dims},
                         aggregate_score=8.0, score_bucket=8, rationale="ok", model=model, seed=seed)

    outs1 = [{"role": "researcher", "agent_id": "r1", "findings": "round one text"}]
    outs2 = [{"role": "researcher", "agent_id": "r1", "findings": "round two text"}]
    kw = dict(topic="t", backend=None, cfg=cfg, backend_id="codex", provider=None,
              sandbox="read-only", judge_fn=judge_fn)

    out1, ens = asyncio.run(orchestrator._run_judge_ensemble(candidate_outputs=outs1, history_outputs=[], **kw))
    assert len(calls) == 3, "exactly n_judges calls for swarm round 1"
    assert out1["role"] == "quality_judge" and out1["quality_vote"] == "ready"
    assert len(out1["_judge_state_log"]) == 1

    out2, ens2 = asyncio.run(orchestrator._run_judge_ensemble(candidate_outputs=outs2, history_outputs=outs1, ensemble=ens, **kw))
    assert ens2 is ens
    assert len(calls) == 6, "exactly n_judges more calls for swarm round 2, no inner loop"
    assert len(out2["_judge_state_log"]) == 2 and out2["_judge_state_log"][1]["round_n"] == 2
