"""Unit tests for mechanisms.selector_synth."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from mechanisms.selector_synth import (
    PairwiseVerdict,
    SelectorConfig,
    SelectorSynth,
    Span,
    bt_mle,
    mock_pairwise_judge,
    split_into_spans,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output(agent_id: str, findings: str, *, round_n: int = 1) -> dict:
    return {
        "agent_id": agent_id,
        "role": "researcher",
        "findings": findings,
        "_round": round_n,
        "key_points": [],
        "confidence": 0.9,
        "quality_vote": "ready",
    }


def _span(span_id: str, text: str = "x" * 100, heading: str | None = None) -> Span:
    return Span(
        id=span_id, source_agent="r", source_round=1,
        heading=heading, text=text,
    )


# ---------------------------------------------------------------------------
# Span splitting
# ---------------------------------------------------------------------------


def test_split_whole_returns_one_span():
    out = _output("r", "## A\nbody1\n\n## B\nbody2")
    spans = split_into_spans(out, granularity="whole")
    assert len(spans) == 1
    assert spans[0].heading is None


def test_split_section_by_h2():
    findings = (
        "## Section A\n"
        + "alpha " * 50
        + "\n\n"
        + "## Section B\n"
        + "bravo " * 50
    )
    out = _output("r", findings)
    spans = split_into_spans(out, granularity="section", min_section_chars=50)
    assert len(spans) == 2
    assert spans[0].heading == "Section A"
    assert spans[1].heading == "Section B"


def test_split_section_no_headings_fallback():
    out = _output("r", "no headings here, just prose")
    spans = split_into_spans(out, granularity="section")
    assert len(spans) == 1
    assert spans[0].heading is None


def test_split_paragraph():
    out = _output("r", "para1\n\npara2\n\npara3")
    spans = split_into_spans(out, granularity="paragraph")
    assert len(spans) == 3
    assert spans[0].text == "para1"
    assert spans[2].text == "para3"


def test_split_empty_findings_returns_empty():
    out = _output("r", "")
    assert split_into_spans(out) == []


# ---------------------------------------------------------------------------
# Bradley-Terry MLE
# ---------------------------------------------------------------------------


def test_bt_mle_uniform_when_all_ties():
    items = ["a", "b", "c"]
    wins = {
        ("a", "b"): 1.5, ("b", "a"): 1.5,
        ("a", "c"): 1.5, ("c", "a"): 1.5,
        ("b", "c"): 1.5, ("c", "b"): 1.5,
    }
    skill = bt_mle(items, wins)
    # All scores roughly equal
    for s in skill.values():
        assert abs(s - 1.0) < 0.1


def test_bt_mle_clear_winner():
    items = ["a", "b", "c"]
    wins = {
        ("a", "b"): 5.0, ("b", "a"): 0.0,
        ("a", "c"): 5.0, ("c", "a"): 0.0,
        ("b", "c"): 3.0, ("c", "b"): 2.0,
    }
    skill = bt_mle(items, wins)
    assert skill["a"] > skill["b"]
    assert skill["b"] > skill["c"]


def test_bt_mle_single_item():
    skill = bt_mle(["only"], {})
    assert skill == {"only": 1.0}


def test_bt_mle_empty():
    assert bt_mle([], {}) == {}


def test_bt_mle_normalized_sum_equals_n():
    items = ["x", "y", "z"]
    wins = {("x", "y"): 1.0, ("y", "z"): 1.0, ("z", "x"): 1.0}
    skill = bt_mle(items, wins)
    assert sum(skill.values()) == pytest.approx(len(items), abs=0.1)


# ---------------------------------------------------------------------------
# Mock pairwise judge
# ---------------------------------------------------------------------------


def test_mock_judge_picks_longer_span():
    a = _span("a", text="short")
    b = _span("b", text="much longer text that definitely outweighs the short one " * 5)
    v = mock_pairwise_judge(a, b, "topic", "judge_00", 0)
    assert v.winner == "b"


def test_mock_judge_ties_similar_lengths():
    a = _span("a", text="x" * 100)
    b = _span("b", text="x" * 102)
    v = mock_pairwise_judge(a, b, "topic", "judge_00", 0)
    assert v.winner == "tie"


# ---------------------------------------------------------------------------
# SelectorSynth full flow
# ---------------------------------------------------------------------------


def test_synthesize_empty_inputs():
    synth = SelectorSynth(SelectorConfig())
    result = synth.synthesize([], "topic")
    assert result.final_text == "[no researcher inputs]"
    assert result.diagnostics["reason"] == "empty_inputs"


def test_synthesize_single_researcher_short_circuits():
    """One input → no pairwise selection needed."""
    out = _output("r1", "## Section A\n" + "long content " * 30)
    synth = SelectorSynth(SelectorConfig(granularity="whole"))
    result = synth.synthesize([out], "topic")
    assert len(result.selected_span_ids) == 1
    assert "long content" in result.final_text


def test_synthesize_picks_winner_per_cluster():
    """Two researchers with matching headings → selector picks one per heading."""
    a = _output("r1", "## Section A\n" + "a content " * 30
                + "\n\n## Section B\n" + "b1 " * 50)
    b = _output("r2", "## Section A\n" + "alternative A " * 30
                + "\n\n## Section B\nshort")
    synth = SelectorSynth(SelectorConfig(granularity="section"))
    result = synth.synthesize([a, b], "topic")
    # Two clusters (Section A, Section B) → 2 selected spans
    assert len(result.selected_span_ids) == 2
    assert "Section A" in result.final_text
    assert "Section B" in result.final_text


def test_synthesize_concat_preserves_content():
    out = _output("r1",
                  "## H1\nfact_one_text\n\n## H2\nfact_two_text")
    synth = SelectorSynth(SelectorConfig(granularity="section",
                                          min_section_chars=1,
                                          transition_strategy="concat"))
    result = synth.synthesize([out], "topic")
    assert "fact_one_text" in result.final_text
    assert "fact_two_text" in result.final_text


def test_synthesize_uses_bt_to_pick_winner():
    """When one span clearly dominates, BT should pick it."""
    long_winner = "## Result\n" + "winning content " * 40
    short_loser = "## Result\nshort"
    out_a = _output("r1", long_winner)
    out_b = _output("r2", short_loser)
    synth = SelectorSynth(SelectorConfig(granularity="section",
                                          min_section_chars=1))
    result = synth.synthesize([out_a, out_b], "topic")
    # The longer span should win (mock judge picks length)
    assert "winning content" in result.final_text
    assert result.final_text.count("short") <= 1  # not assembled


def test_synthesize_diagnostics_judge_calls():
    out_a = _output("r1", "## A\n" + "x " * 50)
    out_b = _output("r2", "## A\n" + "y " * 50)
    cfg = SelectorConfig(granularity="section", min_section_chars=1, n_judges=3)
    synth = SelectorSynth(cfg)
    result = synth.synthesize([out_a, out_b], "topic")
    # 1 cluster of 2 spans = 1 pair × 3 judges = 3 calls
    assert result.diagnostics["judge_calls"] == 3


def test_synthesize_with_guard_filter():
    """guard_filter is applied before splitting."""
    a = _output("r1", "## A\n" + "kept " * 50)
    b = _output("r2", "## A\n" + "filtered " * 50)
    b["_blocked"] = True

    def fake_filter(outputs):
        return [o for o in outputs if not o.get("_blocked")]

    synth = SelectorSynth(SelectorConfig(granularity="section",
                                          min_section_chars=1))
    result = synth.synthesize([a, b], "topic", guard_filter=fake_filter)
    assert "filtered" not in result.final_text
    assert "kept" in result.final_text


def test_synthesize_homogeneous_collapse_warning():
    """Identical spans across researchers → diagnostics flag homogeneous_collapse."""
    txt = "identical content " * 30
    a = _output("r1", txt)
    b = _output("r2", txt)
    synth = SelectorSynth(SelectorConfig(granularity="whole"))
    result = synth.synthesize([a, b], "topic")
    assert result.diagnostics.get("warning") == "homogeneous_collapse"


# ---------------------------------------------------------------------------
# Custom pairwise judge
# ---------------------------------------------------------------------------


def test_custom_pairwise_judge_is_called():
    calls = []
    def custom(span_a, span_b, topic, judge_id, seed):
        calls.append((span_a.id, span_b.id, judge_id))
        return PairwiseVerdict(
            judge_id=judge_id, span_a_id=span_a.id, span_b_id=span_b.id,
            winner="a",
        )
    out_a = _output("r1", "## A\n" + "alpha " * 30)
    out_b = _output("r2", "## A\n" + "beta " * 30)
    synth = SelectorSynth(SelectorConfig(granularity="section",
                                          min_section_chars=1, n_judges=2),
                           pairwise_judge_fn=custom)
    synth.synthesize([out_a, out_b], "topic")
    # 1 pair × 2 judges = 2 calls
    assert len(calls) == 2
