"""Tests for the MAST regression scoreboard."""

from __future__ import annotations

import json

import pytest

from bench.mast_regression import (
    SCENARIOS,
    ScoreboardRow,
    format_scoreboard,
    run_scenario,
    score_all,
)


def test_14_scenarios_registered():
    assert len(SCENARIOS) == 14
    codes = {s.code for s in SCENARIOS}
    expected_codes = {f"FM-{a}.{b}" for (a, b) in [
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 1), (3, 2), (3, 3),
    ]}
    assert codes == expected_codes


def test_each_scenario_runs_without_error():
    for scenario in SCENARIOS:
        row = run_scenario(scenario)
        assert isinstance(row, ScoreboardRow)
        assert row.code == scenario.code


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.code for s in SCENARIOS])
def test_detectable_scenarios_fire_detector(scenario):
    """For scenarios marked detectable=True, the detector MUST fire."""
    if not scenario.detectable:
        pytest.skip("scenario not expected to be detectable by current rules")
    row = run_scenario(scenario)
    assert row.detector_fired, (
        f"{scenario.code} ({scenario.name}) marked as detectable but detector did not fire. "
        f"Notes: {scenario.notes}"
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.code for s in SCENARIOS])
def test_non_detectable_scenarios_do_not_fire(scenario):
    """For scenarios marked detectable=False, the detector must NOT fire (no false positives)."""
    if scenario.detectable:
        pytest.skip("scenario expected to fire — covered by other test")
    row = run_scenario(scenario)
    assert not row.detector_fired, (
        f"{scenario.code} unexpectedly fired its own detector. "
        f"If detector was added, update Scenario.detectable=True."
    )


def test_score_all_returns_14_rows():
    rows = score_all()
    assert len(rows) == 14


def test_format_scoreboard_includes_each_FM():
    rows = score_all()
    text = format_scoreboard(rows, version_label="test")
    for row in rows:
        assert row.code in text


def test_format_scoreboard_header():
    rows = score_all()
    text = format_scoreboard(rows, version_label="vTEST")
    assert "vTEST" in text
    assert "Detected:" in text
    assert "Coverage by category" in text


def test_baseline_v01_detection_count():
    """v0.1.1 baseline: exactly 9 of 14 scenarios should be detected.

    This is the expected detector coverage. If this number changes, either:
    (a) a new detector was added → update SCENARIOS[i].detectable + this test
    (b) a detector regressed → fix the detector
    """
    rows = score_all()
    detected = sum(1 for r in rows if r.detector_fired)
    assert detected == 9, (
        f"expected 9 detected scenarios at v0.1.1 baseline, got {detected}. "
        f"Update test_baseline_v01_detection_count if scoreboard intentionally changed."
    )
