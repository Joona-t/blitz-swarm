"""Unit tests for mechanisms.cascade_guard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from mechanisms.cascade_guard import (
    AtomicClaim,
    CascadeGuard,
    GuardConfig,
    HUB_ROLES,
    heuristic_decompose,
    heuristic_screen,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_output(role="researcher", agent_id="r01", findings="x" * 100,
                  vote="ready", confidence=0.9, **kw):
    base = {
        "agent_id": agent_id,
        "role": role,
        "findings": findings,
        "key_points": ["k1"],
        "confidence": confidence,
        "quality_vote": vote,
        "quality_notes": "ok",
        "dissent": "",
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------


def test_heuristic_decompose_sentence_split():
    text = "First sentence. Second sentence. Third sentence."
    claims = heuristic_decompose(text)
    assert len(claims) == 3
    assert claims[0] == "First sentence."


def test_heuristic_decompose_max_claims_cap():
    text = ". ".join(f"Sentence number {i}" for i in range(20)) + "."
    claims = heuristic_decompose(text, max_claims=5)
    assert len(claims) == 5


def test_heuristic_decompose_empty():
    assert heuristic_decompose("") == []
    assert heuristic_decompose("   ") == []


def test_heuristic_screen_no_neighbors_returns_yellow():
    claim = AtomicClaim(id="c1", text="some claim", source_agent="r01",
                        source_round=1)
    assert heuristic_screen(claim, []) == "yellow"


def test_heuristic_screen_overlap_returns_green():
    anchor = AtomicClaim(id="a1", text="The WAL file structure is documented",
                         source_agent="r0", source_round=0, verdict="green")
    claim = AtomicClaim(id="c1", text="The WAL file structure is documented",
                        source_agent="r1", source_round=1)
    assert heuristic_screen(claim, [anchor]) == "green"


def test_heuristic_screen_negation_with_overlap_returns_red():
    anchor = AtomicClaim(id="a1", text="The WAL file structure is documented",
                         source_agent="r0", source_round=0, verdict="green")
    claim = AtomicClaim(id="c1",
                        text="The WAL file structure is not documented",
                        source_agent="r1", source_round=1)
    assert heuristic_screen(claim, [anchor]) == "red"


# ---------------------------------------------------------------------------
# CascadeGuard — basic flow
# ---------------------------------------------------------------------------


def test_off_mode_passthrough():
    g = CascadeGuard(GuardConfig(mode="off"))
    out = _agent_output(findings="hello world")
    result = g.on_agent_output(out, round_n=1)
    assert result is out
    assert "_blocked" not in result
    # Even off-mode tags _round so context-builders can identify the round
    assert result.get("_round") == 1


def test_on_agent_output_decomposes_and_tags():
    g = CascadeGuard(GuardConfig(mode="speed"))
    out = _agent_output(findings="First fact. Second fact. Third fact.")
    g.on_agent_output(out, round_n=1)
    assert "_claim_ids" in out
    assert len(out["_claim_ids"]) == 3
    assert out["_claim_counts"]["yellow"] >= 1


def test_speed_mode_does_not_block_red():
    """Speed mode releases all yellows and never blocks on Red."""
    g = CascadeGuard(GuardConfig(mode="speed"))
    # Plant a Green anchor in round 0
    g.lineage["a1"] = AtomicClaim(
        id="a1", text="The WAL file is structured documented",
        source_agent="r0", source_round=0, verdict="green",
    )
    out = _agent_output(
        findings="The WAL file is not structured documented.",
    )
    g.on_agent_output(out, round_n=1)
    # Even though the heuristic detects a contradiction (red), speed
    # mode does not block. block_on_red is True but speed mode is
    # excluded from the block path.
    assert "_blocked" not in out


def test_balanced_mode_blocks_red():
    g = CascadeGuard(GuardConfig(mode="balanced"))
    g.lineage["a1"] = AtomicClaim(
        id="a1", text="The WAL file is structured documented",
        source_agent="r0", source_round=0, verdict="green",
    )
    out = _agent_output(
        findings="The WAL file is not structured documented.",
    )
    g.on_agent_output(out, round_n=1)
    assert out.get("_blocked") is True
    assert "_block_reason" in out


def test_filter_context_drops_blocked():
    g = CascadeGuard(GuardConfig(mode="balanced"))
    out_a = _agent_output(agent_id="a", findings="x")
    out_b = _agent_output(agent_id="b", findings="y")
    out_a["_blocked"] = True
    filtered = g.filter_context([out_a, out_b])
    assert filtered == [out_b]


def test_filter_context_drops_tainted_agent_round():
    g = CascadeGuard(GuardConfig(mode="balanced"))
    out_a = _agent_output(agent_id="a", findings="x")
    out_a["_round"] = 1
    out_b = _agent_output(agent_id="b", findings="y")
    out_b["_round"] = 1
    g.tainted.add(("a", 1))
    filtered = g.filter_context([out_a, out_b])
    assert filtered == [out_b]


def test_off_mode_filter_is_identity():
    g = CascadeGuard(GuardConfig(mode="off"))
    out_a = _agent_output(agent_id="a", findings="x")
    out_a["_blocked"] = True   # off mode ignores
    out_b = _agent_output(agent_id="b", findings="y")
    filtered = g.filter_context([out_a, out_b])
    assert len(filtered) == 2


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_on_agent_error_marks_agent_tainted():
    g = CascadeGuard(GuardConfig(mode="balanced"))
    g.on_agent_error("r01", round_n=1, reason="agent_error")
    assert ("r01", 1) in g.tainted


def test_error_taints_descendants():
    g = CascadeGuard(GuardConfig(mode="balanced"))
    # Round 1: erroring agent r01 produces claim c1
    g.lineage["c1"] = AtomicClaim(
        id="c1", text="root claim", source_agent="r01", source_round=1,
        verdict="green",
    )
    # Round 2: r02 produces c2 citing c1 as parent
    g.lineage["c2"] = AtomicClaim(
        id="c2", text="downstream claim", source_agent="r02", source_round=2,
        parent_claim_ids=["c1"], verdict="green",
    )
    # Round 3: r03 produces c3 citing c2
    g.lineage["c3"] = AtomicClaim(
        id="c3", text="further downstream", source_agent="r03", source_round=3,
        parent_claim_ids=["c2"], verdict="green",
    )
    n = g.on_agent_error("r01", round_n=1, reason="boom")
    assert n == 2
    assert ("r02", 2) in g.tainted
    assert ("r03", 3) in g.tainted
    assert g.lineage["c2"].verdict == "yellow"
    assert g.lineage["c3"].verdict == "yellow"


def test_error_cascade_does_not_demote_red():
    g = CascadeGuard(GuardConfig(mode="balanced"))
    g.lineage["c1"] = AtomicClaim(
        id="c1", text="root", source_agent="r01", source_round=1, verdict="green",
    )
    g.lineage["c2"] = AtomicClaim(
        id="c2", text="downstream", source_agent="r02", source_round=2,
        parent_claim_ids=["c1"], verdict="red",
    )
    g.on_agent_error("r01", round_n=1, reason="boom")
    # Red stays Red even when its parent's author errors
    assert g.lineage["c2"].verdict == "red"


# ---------------------------------------------------------------------------
# Full round flow with errored agent (the v0.1.0 Run-2 cascade scenario)
# ---------------------------------------------------------------------------


def test_full_round_flow_isolates_errored_agent():
    """Simulates a round where one researcher errors and a critic in
    the same round produces output. After error propagation,
    filter_context excludes the errored agent's downstream claims."""
    g = CascadeGuard(GuardConfig(mode="balanced"))

    # Round 1: r01 errors immediately, r02 is fine
    out_r01 = _agent_output(agent_id="r01", findings="garbage", _error=True)
    out_r02 = _agent_output(agent_id="r02",
                            findings="Real finding. Properly structured.")
    g.on_agent_output(out_r01, round_n=1)
    g.on_agent_output(out_r02, round_n=1)

    # Round 2: filter_context should drop the errored output but keep r02
    all_outputs = [out_r01, out_r02]
    filtered = g.filter_context(all_outputs)
    assert out_r02 in filtered
    assert out_r01 not in filtered


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


def test_eviction_drops_oldest_non_green_when_capped():
    g = CascadeGuard(GuardConfig(mode="speed", max_lineage_size=5,
                                  max_claims_per_msg=3))
    # 7 yellow claims spread across rounds (3 + 3 + 1 = 7)
    for i in range(3):
        out = _agent_output(agent_id="r1",
                            findings="A. B. C.")
        g.on_agent_output(out, round_n=i)
    # Eviction triggers on each on_agent_output past 5
    assert len(g.lineage) <= 5


# ---------------------------------------------------------------------------
# Round summary
# ---------------------------------------------------------------------------


def test_round_summary_counts_verdicts():
    g = CascadeGuard(GuardConfig(mode="speed"))
    out = _agent_output(findings="A. B. C.")
    g.on_agent_output(out, round_n=1)
    summary = g.round_summary(1)
    # All yellow at speed (no green anchors yet)
    assert summary.yellow == 3
    assert summary.green == 0
    assert summary.red == 0


def test_round_summary_for_unknown_round_is_empty():
    g = CascadeGuard(GuardConfig(mode="speed"))
    summary = g.round_summary(99)
    assert summary.green == summary.yellow == summary.red == 0


# ---------------------------------------------------------------------------
# Hub-only verification
# ---------------------------------------------------------------------------


def test_hub_roles_constant():
    assert "synthesizer" in HUB_ROLES
    assert "quality_judge" in HUB_ROLES
    assert "researcher" not in HUB_ROLES


# ---------------------------------------------------------------------------
# Pluggable LLM hooks
# ---------------------------------------------------------------------------


def test_custom_decompose_fn_is_called():
    calls = []
    def custom_decompose(findings):
        calls.append(findings)
        return ["custom claim"]
    g = CascadeGuard(GuardConfig(mode="speed"),
                     decompose_fn=custom_decompose)
    out = _agent_output(findings="anything")
    g.on_agent_output(out, round_n=1)
    assert calls == ["anything"]


def test_custom_screen_fn_is_called():
    def custom_screen(claim, neighbors):
        return "red"
    g = CascadeGuard(GuardConfig(mode="balanced"),
                     screen_fn=custom_screen)
    out = _agent_output(findings="anything claim")
    g.on_agent_output(out, round_n=1)
    # Red claim under balanced mode should block
    assert out.get("_blocked") is True
