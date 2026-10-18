"""consensus.py: unanimity, holdout override timing, dissent extraction,
final-vote filtering, formatting."""

from __future__ import annotations

from consensus import (
    check_consensus,
    extract_dissent,
    final_votes,
    format_convergence_report,
    format_dissent_section,
    should_override_holdout,
)


def _vote(agent_id: str, vote: str | None, *, role="critic", conf=0.8, notes="", dissent=""):
    out = {"agent_id": agent_id, "role": role, "confidence": conf,
           "quality_notes": notes, "dissent": dissent, "findings": "f"}
    if vote is not None:
        out["quality_vote"] = vote
    return out


# --- unanimity -------------------------------------------------------------

def test_consensus_requires_every_voter_ready():
    assert check_consensus([_vote("a", "ready"), _vote("b", "ready")]) is True
    assert check_consensus([_vote("a", "ready"), _vote("b", "needs_work")]) is False


def test_consensus_false_with_no_voters_and_ignores_non_voters():
    assert check_consensus([]) is False
    assert check_consensus([_vote("r", None, role="researcher")]) is False
    # A non-voting researcher does not break unanimity among voters.
    assert check_consensus([_vote("r", None, role="researcher"), _vote("c", "ready")]) is True


# --- holdout override timing -------------------------------------------------

def test_holdout_override_only_from_round_3_with_exactly_one_holdout():
    votes = [_vote("a", "ready"), _vote("b", "ready"), _vote("c", "needs_work")]
    assert should_override_holdout(1, votes) is False
    assert should_override_holdout(2, votes) is False
    assert should_override_holdout(3, votes) is True
    assert should_override_holdout(7, votes) is True


def test_holdout_override_needs_two_voters_and_a_single_dissenter():
    assert should_override_holdout(3, [_vote("only", "needs_work")]) is False
    two_holdouts = [_vote("a", "ready"), _vote("b", "needs_work"), _vote("c", "needs_work")]
    assert should_override_holdout(3, two_holdouts) is False
    assert should_override_holdout(3, [_vote("a", "ready"), _vote("b", "ready")]) is False


# --- dissent extraction ----------------------------------------------------

def test_extract_dissent_collects_needs_work_with_notes():
    d = extract_dissent([
        _vote("a", "ready"),
        _vote("b", "needs_work", role="fact_checker", conf=0.4, notes="numbers off", dissent="disagree"),
        _vote("r", None, role="researcher"),
    ])
    assert d == [{
        "agent_id": "b", "role": "fact_checker", "quality_notes": "numbers off",
        "dissent": "disagree", "confidence": 0.4,
    }]


def test_final_votes_keeps_only_each_agents_last_vote():
    rounds = [
        [_vote("critic", "needs_work", notes="r1"), _vote("fc", "needs_work", notes="r1")],
        [_vote("critic", "ready", notes="r2"), _vote("fc", "needs_work", notes="r2")],
        [_vote("critic", "ready", notes="r3"), _vote("fc", "ready", notes="r3")],
    ]
    finals = final_votes(rounds)
    assert [(v["agent_id"], v["quality_notes"]) for v in finals] == [("critic", "r3"), ("fc", "r3")]
    # An agent that flipped to ready is NOT a dissenter (core-loop-08).
    assert extract_dissent(finals) == []


def test_final_votes_reports_an_unresolved_holdout_once():
    rounds = [
        [_vote("critic", "needs_work", notes="r1"), _vote("fc", "needs_work", notes="r1")],
        [_vote("critic", "ready", notes="r2"), _vote("fc", "needs_work", notes="still wrong")],
    ]
    dissent = extract_dissent(final_votes(rounds))
    assert [(d["agent_id"], d["quality_notes"]) for d in dissent] == [("fc", "still wrong")]


# --- formatting ---------------------------------------------------------------

def test_format_dissent_section_empty_and_populated():
    assert format_dissent_section([]) == ""
    text = format_dissent_section(extract_dissent([
        _vote("fc", "needs_work", role="fact_checker", conf=0.25, notes="N", dissent="D"),
    ]))
    assert text.startswith("## Dissent & Open Questions")
    assert "### fact_checker (fc)" in text
    assert "**Concern:** N" in text and "**Dissent:** D" in text
    assert "**Confidence:** 25.0%" in text


def test_format_convergence_report_per_round_status():
    report = format_convergence_report([
        [_vote("a", "ready"), _vote("b", "needs_work", conf=0.6)],
        [_vote("a", "ready"), _vote("b", "ready", conf=0.6)],
        [],
    ])
    lines = [l for l in report.splitlines() if l.startswith("**Round")]
    assert lines[0] == "**Round 1:** 1/2 ready | avg confidence: 70.0%"
    assert lines[1] == "**Round 2:** CONSENSUS | avg confidence: 70.0%"
    assert lines[2] == "**Round 3:** 0/0 ready | avg confidence: 0.0%"
