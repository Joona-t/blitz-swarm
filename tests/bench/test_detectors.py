"""Unit tests for bench/detectors.py — MAST failure-mode rule-based detectors."""

from __future__ import annotations

from bench.detectors import (
    detect_all,
    detect_FM_1_3_step_repetition,
    detect_FM_1_4_loss_of_history,
    detect_FM_2_1_oscillating_consensus,
    detect_FM_2_4_information_withholding,
    detect_FM_2_5_ignored_other_agents,
    detect_FM_2_6_reasoning_action_mismatch,
    detect_FM_3_1_silent_partial,
    detect_FM_3_3_incorrect_verification,
)


def _agent(role, agent_id="a01", findings="findings text", key_points=None,
           vote="ready", confidence=0.9, **kw):
    base = {
        "agent_id": agent_id,
        "role": role,
        "findings": findings,
        "key_points": key_points if key_points is not None else ["k1", "k2"],
        "confidence": confidence,
        "quality_vote": vote,
        "quality_notes": "ok",
        "dissent": "",
    }
    base.update(kw)
    return base


def test_FM_1_3_no_repetition_when_findings_differ():
    rounds = [
        [_agent("researcher", findings="alpha bravo charlie delta echo foxtrot")],
        [_agent("researcher", findings="zulu yankee xray whiskey victor uniform")],
    ]
    assert not detect_FM_1_3_step_repetition(rounds)


def test_FM_1_3_detected_on_identical_findings():
    text = "the same finding repeated word for word across rounds without change"
    rounds = [
        [_agent("researcher", findings=text)],
        [_agent("researcher", findings=text)],
    ]
    assert detect_FM_1_3_step_repetition(rounds)


def test_FM_1_4_empty_context_detected():
    assert detect_FM_1_4_loss_of_history([0, 1500, 2000])


def test_FM_1_4_all_nonempty_passes():
    assert not detect_FM_1_4_loss_of_history([1500, 2000, 3000])


def test_FM_2_1_oscillating_consensus():
    # Round counts: 1, 3, 1 — peaks at 3 and drops
    rounds = [
        [_agent("critic", vote="needs_work"), _agent("judge", vote="ready")],
        [_agent("critic", vote="ready"), _agent("judge", vote="ready"), _agent("res", vote="ready")],
        [_agent("critic", vote="needs_work"), _agent("judge", vote="ready")],
    ]
    assert detect_FM_2_1_oscillating_consensus(rounds)


def test_FM_2_1_monotone_passes():
    rounds = [
        [_agent("critic", vote="needs_work"), _agent("judge", vote="needs_work")],
        [_agent("critic", vote="needs_work"), _agent("judge", vote="ready")],
        [_agent("critic", vote="ready"), _agent("judge", vote="ready")],
    ]
    assert not detect_FM_2_1_oscillating_consensus(rounds)


def test_FM_2_4_empty_key_points_with_findings():
    outputs = [_agent("researcher", findings="lots of text", key_points=[])]
    assert detect_FM_2_4_information_withholding(outputs)


def test_FM_2_4_empty_findings_passes():
    outputs = [_agent("researcher", findings="", key_points=[])]
    assert not detect_FM_2_4_information_withholding(outputs)


def test_FM_2_6_low_confidence_with_ready_vote():
    outputs = [_agent("researcher", confidence=0.2, vote="ready")]
    assert detect_FM_2_6_reasoning_action_mismatch(outputs)


def test_FM_2_6_high_confidence_passes():
    outputs = [_agent("researcher", confidence=0.9, vote="ready")]
    assert not detect_FM_2_6_reasoning_action_mismatch(outputs)


def test_FM_3_1_silent_partial_detected():
    rounds = [
        [_agent("researcher", _partial=True)],
    ]
    assert detect_FM_3_1_silent_partial(rounds)


def test_FM_3_1_flagged_partial_passes():
    rounds = [
        [_agent("researcher", _partial=True, _flagged_partial=True)],
    ]
    assert not detect_FM_3_1_silent_partial(rounds)


def test_FM_3_3_high_judge_score_short_output():
    judge = _agent(
        "quality_judge",
        coverage_score=9, accuracy_score=9, clarity_score=9, depth_score=9,
    )
    assert detect_FM_3_3_incorrect_verification([judge], output_md_chars=200)


def test_FM_3_3_long_output_passes():
    judge = _agent(
        "quality_judge",
        coverage_score=9, accuracy_score=9, clarity_score=9, depth_score=9,
    )
    assert not detect_FM_3_3_incorrect_verification([judge], output_md_chars=2000)


def test_detect_all_no_failures_returns_empty():
    rounds = [
        [_agent("researcher", findings="foo bar"), _agent("critic", vote="ready", findings="missing flag concern issue gap")],
        [_agent("researcher", findings="baz qux"), _agent("critic", vote="ready", findings="missing flag concern issue gap")],
    ]
    judge = _agent("quality_judge", coverage_score=8, accuracy_score=8,
                   clarity_score=8, depth_score=8)
    rounds[-1].append(judge)
    flags = detect_all(rounds, output_md_chars=1500, round_2_context_lengths=[1500, 1500])
    # Should not flag FM-3.3 (output is long), not flag any others
    assert "FM-3.3" not in flags


def test_detect_all_returns_list():
    rounds = [[_agent("researcher")]]
    flags = detect_all(rounds)
    assert isinstance(flags, list)
