"""agents.py heuristic planner: subtopic splitting, `[swarm] max_agents`
enforcement (_cap_plan), roster shape from plan_agents(use_llm=False), and
the loud fallback when LLM planning fails. `_llm_plan` is monkeypatched in
every use_llm=True test so no backend is ever invoked."""

from __future__ import annotations

import pytest

import agents as agents_mod
from agents import (
    CRYPTO_TRADING_ANGLES,
    GENERIC_ANGLES,
    HEURISTIC_PLAN,
    BlitzAgent,
    _cap_plan,
    _split_subtopics_heuristic,
    plan_agents,
)

KW = dict(use_llm=False, domain="general", persona_critics=False)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Any accidental LLM planning call fails loudly instead of spawning a CLI."""
    def boom(*a, **kw):
        raise AssertionError("_llm_plan must not be invoked by this test")
    monkeypatch.setattr(agents_mod, "_llm_plan", boom)


# ---------------------------------------------------------------------------
# _split_subtopics_heuristic
# ---------------------------------------------------------------------------


def test_split_generic_angles_cycle_and_keep_topic_prefix():
    subs = _split_subtopics_heuristic("graph databases", 6)
    assert len(subs) == 6
    assert all(s.startswith("graph databases — focusing on ") for s in subs)
    assert subs[0].endswith(GENERIC_ANGLES[0])
    assert subs[4].endswith(GENERIC_ANGLES[0])          # wraps after 4 generic angles
    assert subs[5].endswith(GENERIC_ANGLES[1])


@pytest.mark.parametrize("topic", ["BTC momentum strategies", "Sharpe ratio of a backtest", "Crypto risk"])
def test_split_uses_crypto_angles_for_trading_topics(topic):
    subs = _split_subtopics_heuristic(topic, 2)
    assert subs[0].endswith(CRYPTO_TRADING_ANGLES[0])
    assert subs[1].endswith(CRYPTO_TRADING_ANGLES[1])


def test_split_zero_count_is_empty():
    assert _split_subtopics_heuristic("x", 0) == []


# ---------------------------------------------------------------------------
# _cap_plan
# ---------------------------------------------------------------------------


def test_cap_plan_leaves_a_fitting_plan_alone():
    assert _cap_plan(4, 2, True, 12) == (4, 2, True)
    assert _cap_plan(4, 2, True, 9) == (4, 2, True)     # 4+2+1+2 == 9 exactly


def test_cap_plan_trims_researchers_then_critics_then_fact_checker():
    assert _cap_plan(4, 2, True, 8) == (3, 2, True)
    assert _cap_plan(4, 2, True, 7) == (2, 2, True)
    assert _cap_plan(4, 2, True, 6) == (1, 2, True)     # researchers floor at 1
    assert _cap_plan(4, 2, True, 5) == (1, 1, True)     # then critics floor at 1
    assert _cap_plan(4, 2, True, 4) == (1, 1, False)    # then the fact-checker goes


def test_cap_plan_floor_is_four_slots():
    with pytest.raises(ValueError, match=r"max_agents = 3 cannot fit the minimum swarm"):
        _cap_plan(1, 1, False, 3)


# ---------------------------------------------------------------------------
# plan_agents(use_llm=False)
# ---------------------------------------------------------------------------


def test_heuristic_roster_shape_ids_and_order():
    roster = plan_agents("graph databases", max_agents=12, **KW)
    assert HEURISTIC_PLAN == {"researcher_count": 2, "critic_count": 1,
                              "needs_fact_checker": True, "subtopics": None}
    assert [a.id for a in roster] == ["researcher_00", "researcher_01", "critic",
                                      "fact_checker", "quality_judge", "synthesizer"]
    assert [a.role for a in roster] == ["researcher", "researcher", "critic",
                                        "fact_checker", "quality_judge", "synthesizer"]
    assert all(isinstance(a, BlitzAgent) and a.system_prompt for a in roster)
    assert roster[0].subtopic.endswith(GENERIC_ANGLES[0])
    assert roster[1].subtopic.endswith(GENERIC_ANGLES[1])
    assert all(a.subtopic == "graph databases" for a in roster[2:])


def test_heuristic_roster_loads_prompts_from_disk_with_sha():
    roster = plan_agents("t", max_agents=12, **KW)
    for a in roster:
        assert a.prompt_path and a.prompt_path.endswith(".md"), a.id
        assert a.prompt_sha256 and len(a.prompt_sha256) == 64, a.id


@pytest.mark.parametrize("max_agents, expected_ids", [
    (6, ["researcher_00", "researcher_01", "critic", "fact_checker", "quality_judge", "synthesizer"]),
    (5, ["researcher_00", "critic", "fact_checker", "quality_judge", "synthesizer"]),
    (4, ["researcher_00", "critic", "quality_judge", "synthesizer"]),
])
def test_max_agents_trims_the_heuristic_roster_and_prints_a_notice(capsys, max_agents, expected_ids):
    roster = plan_agents("t", max_agents=max_agents, **KW)
    assert [a.id for a in roster] == expected_ids
    assert len(roster) <= max_agents
    out = capsys.readouterr().out
    if max_agents < 6:
        assert f"max_agents={max_agents}: plan trimmed to" in out
    else:
        assert "plan trimmed" not in out


def test_max_agents_below_minimum_swarm_raises():
    with pytest.raises(ValueError, match="cannot fit the minimum swarm"):
        plan_agents("t", max_agents=3, **KW)


def test_max_agents_none_reads_blitz_toml(monkeypatch, tmp_path):
    cfg = tmp_path / "blitz.toml"
    cfg.write_text("[swarm]\nmax_agents = 4\n", encoding="utf-8")
    monkeypatch.setenv("BLITZ_CONFIG", str(cfg))
    roster = plan_agents("t", use_llm=False, domain="general", persona_critics=False)
    assert len(roster) == 4 and "fact_checker" not in [a.id for a in roster]


def test_crypto_topic_gets_crypto_subtopics_even_in_general_domain():
    roster = plan_agents("bitcoin funding rate arbitrage", max_agents=12, **KW)
    assert roster[0].subtopic.endswith(CRYPTO_TRADING_ANGLES[0])


# ---------------------------------------------------------------------------
# use_llm=True with a mocked planner (never a real backend)
# ---------------------------------------------------------------------------


def test_llm_plan_failure_falls_back_to_heuristic_with_stderr_warning(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(agents_mod, "_llm_plan",
                        lambda topic, *, backend_id=None, sandbox=None: calls.append((topic, backend_id, sandbox)))
    roster = plan_agents("t", use_llm=True, domain="general", persona_critics=False,
                         backend_id="bk", sandbox="read-only", max_agents=12)
    assert calls == [("t", "bk", "read-only")]
    assert len(roster) == 6
    err = capsys.readouterr().err
    assert err.startswith("WARNING: LLM agent planning failed")
    assert "falling back to the heuristic plan" in err


def test_llm_plan_counts_and_subtopics_are_honoured_and_capped(monkeypatch, capsys):
    monkeypatch.setattr(agents_mod, "_llm_plan", lambda *a, **kw: {
        "researcher_count": 4, "critic_count": 2, "needs_fact_checker": False,
        "subtopics": ["alpha", "beta", "gamma", "delta", "epsilon"],
    })
    roster = plan_agents("T", use_llm=True, domain="general", persona_critics=False, max_agents=12)
    ids = [a.id for a in roster]
    assert ids == ["researcher_00", "researcher_01", "researcher_02", "researcher_03",
                   "critic_00", "critic_01", "quality_judge", "synthesizer"]
    assert [a.subtopic for a in roster[:4]] == [f"T — focusing on {s}" for s in ("alpha", "beta", "gamma", "delta")]
    assert "plan trimmed" not in capsys.readouterr().out

    capped = plan_agents("T", use_llm=True, domain="general", persona_critics=False, max_agents=5)
    assert [a.id for a in capped] == ["researcher_00", "critic_00", "critic_01", "quality_judge", "synthesizer"]
    assert "max_agents=5: plan trimmed to 1 researchers, 2 critics, fact_checker=no" in capsys.readouterr().out
    assert capped[0].subtopic == "T — focusing on alpha"


def test_llm_plan_with_too_few_subtopics_uses_heuristic_angles(monkeypatch):
    monkeypatch.setattr(agents_mod, "_llm_plan", lambda *a, **kw: {
        "researcher_count": 3, "critic_count": 1, "needs_fact_checker": True, "subtopics": ["only one"],
    })
    roster = plan_agents("T", use_llm=True, domain="general", persona_critics=False, max_agents=12)
    subs = [a.subtopic for a in roster if a.role == "researcher"]
    assert len(subs) == 3
    assert subs == [f"T — focusing on {GENERIC_ANGLES[i]}" for i in range(3)]
