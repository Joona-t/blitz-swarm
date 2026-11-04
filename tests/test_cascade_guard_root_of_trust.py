"""AUDIT H7: the guard had no root of trust, so nothing ever became Green
and nothing could ever be Red. Corroboration at round close provides it."""

from __future__ import annotations

from mechanisms.cascade_guard import CascadeGuard, GuardConfig


def _out(agent, text, round_n=1, role="researcher"):
    return {"agent_id": agent, "role": role, "findings": text, "_round": round_n}


def test_corroborated_claims_become_green_and_enable_red_next_round():
    g = CascadeGuard(GuardConfig(mode="strict"))
    a = g.on_agent_output(_out("r1", "WAL mode appends committed pages to a separate log file."), 1)
    b = g.on_agent_output(_out("r2", "WAL mode appends committed pages to a separate log file."), 1)
    c = g.on_agent_output(_out("r3", "Checkpoints copy pages back into the main database."), 1)
    assert g.round_summary(1).green == 0, "no promotion before the round closes"
    promoted = g.close_round(1)
    assert promoted == 2
    s = g.round_summary(1)
    assert s.promoted == 2 and s.green == 2
    # the uncorroborated claim stays yellow
    assert [g.lineage[i].verdict for i in c["_claim_ids"]] == ["yellow"]
    # round 2: a contradiction of the now-Green anchor is Red and blocked
    d = g.on_agent_output(_out("r4", "WAL mode does not append committed pages to a separate log file.", round_n=2), 2)
    assert d.get("_blocked") is True
    assert g.round_summary(2).red >= 1


def test_same_agent_repeating_itself_does_not_corroborate():
    g = CascadeGuard(GuardConfig(mode="balanced"))
    g.on_agent_output(_out("r1", "Readers never block the single writer in WAL mode."), 1)
    g.on_agent_output(_out("r1", "Readers never block the single writer in WAL mode."), 1)
    assert g.close_round(1) == 0


def test_close_round_is_idempotent_and_off_when_inactive():
    g = CascadeGuard(GuardConfig(mode="balanced"))
    g.on_agent_output(_out("r1", "Fsync happens on the WAL file at commit."), 1)
    g.on_agent_output(_out("r2", "Fsync happens on the WAL file at commit."), 1)
    assert g.close_round(1) == 2
    before = g.round_summary(1).promoted
    assert g.close_round(1) == 0 and g.round_summary(1).promoted == before
    off = CascadeGuard(GuardConfig(mode="off"))
    assert off.close_round(1) == 0
