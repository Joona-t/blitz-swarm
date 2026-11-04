"""AUDIT 2026-10-27 H2: the synthesizer must see the final round and the
critique. Sections have their own budgets, newest round first."""

from __future__ import annotations

import orchestrator


def _outs(n_rounds=3, n_res=4, findings_len=1500):
    outs = []
    for r in range(1, n_rounds + 1):
        for i in range(n_res):
            outs.append({"role": "researcher", "agent_id": f"r{i}_round{r}", "_round": r,
                         "findings": f"R{r}-{i} " + "x" * findings_len, "key_points": ["k"], "confidence": 0.9})
        outs.append({"role": "critic", "agent_id": f"c_round{r}", "_round": r, "findings": f"CRITIC{r} " + "c" * 400})
        outs.append({"role": "quality_judge", "agent_id": f"j_round{r}", "_round": r, "findings": f"JUDGE{r} " + "j" * 300})
    return outs


def test_synthesizer_sees_final_round_and_all_feedback_sections():
    ctx = orchestrator.build_context(_outs(), for_role="synthesizer")
    assert len(ctx) <= orchestrator.MAX_CONTEXT_CHARS * 1.1
    for i in range(4):
        assert f"r{i}_round3" in ctx, "every final-round researcher must be present"
    assert "CRITIC3" in ctx and "JUDGE3" in ctx
    assert "### Critic Feedback" in ctx and "### Quality Judge Assessment" in ctx
    # newest first: round 3 header appears before round 1
    assert ctx.index("r0_round3") < ctx.index("r0_round1") if "r0_round1" in ctx else True


def test_researchers_see_critique_even_when_findings_are_long():
    ctx = orchestrator.build_context(_outs(findings_len=5000), for_role="researcher")
    assert "CRITIC3" in ctx
    assert "JUDGE" not in ctx            # researchers never see judge text


def test_section_never_silently_empty_but_budget_holds():
    outs = _outs(n_rounds=1, n_res=1, findings_len=50_000)
    ctx = orchestrator.build_context(outs, for_role="synthesizer")
    assert "truncated" in ctx
    assert "CRITIC1" in ctx and "JUDGE1" in ctx
    assert len(ctx) <= orchestrator.MAX_CONTEXT_CHARS * 1.1


def test_spare_budget_flows_to_researchers_when_no_feedback_yet():
    outs = [{"role": "researcher", "agent_id": "r0", "_round": 1, "findings": "y" * 9000, "key_points": [], "confidence": 0.5}]
    ctx = orchestrator.build_context(outs, for_role="critic")
    assert ctx.count("y") >= int(orchestrator.MAX_CONTEXT_CHARS * 0.9)
