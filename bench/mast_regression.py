"""MAST regression scoreboard — 14 named failure-mode scenarios.

Each scenario constructs synthetic round outputs that model one MAST
failure mode from Cemri et al. arXiv 2503.13657 (NeurIPS 2025). The
scoreboard runs every scenario through `bench.detectors.detect_all`
and reports which failure modes the current detectors catch.

Goal: produce `bench/mast_scoreboard.md` — a published artifact showing
v0.1.x → v0.2.x detector coverage progression.

Why pure-synthetic and not orchestrator-injection:
- Synthetic scenarios run in milliseconds and require zero API spend.
- They unambiguously document what each failure mode "looks like" as
  data, which is the spec the orchestrator must learn to surface.
- Orchestrator-injection tests (with monkey-patched invoke_agent and a
  real run_swarm) live in Phase 1 once cascade_guard is wired into the
  blackboard write path.

Run:
    python -m bench.mast_regression                 # prints scoreboard to stdout
    python -m bench.mast_regression --write         # writes bench/mast_scoreboard.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Make package importable when running as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.detectors import FM_NAMES, detect_all


# ---------------------------------------------------------------------------
# Helpers — synthetic agent outputs
# ---------------------------------------------------------------------------


def _agent(role: str, *, agent_id: str = "a01", findings: str = "x" * 200,
           key_points: list[str] | None = None, vote: str = "ready",
           confidence: float = 0.9, **kw) -> dict:
    base = {
        "agent_id": agent_id,
        "role": role,
        "findings": findings,
        "key_points": key_points if key_points is not None else ["k1", "k2"],
        "confidence": confidence,
        "gaps_identified": [],
        "quality_vote": vote,
        "quality_notes": "ok",
        "dissent": "",
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Scenario:
    """One MAST failure-mode scenario with expected detector behavior."""

    code: str                         # e.g. "FM-1.3"
    name: str
    category: str                     # "FC1" | "FC2" | "FC3"
    build: Callable[[], dict]         # returns kwargs for detect_all
    detectable: bool                  # is this catchable by current detectors?
    notes: str


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _scenario_FM_1_1() -> dict:
    """FM-1.1 — Disobey Task Specification: agent ignores topic entirely."""
    return {
        "rounds": [[
            _agent("researcher", findings="Here's a recipe for chicken parmesan."),
            _agent("critic", vote="needs_work",
                   findings="Researcher output is off-topic: missing flag concern issue gap"),
        ]],
        "output_md_chars": 1500,
    }


def _scenario_FM_1_2() -> dict:
    """FM-1.2 — Disobey Role Specification: critic produces research-style output."""
    return {
        "rounds": [[
            _agent("researcher", findings="The WAL file is structured. Checkpointing happens."),
            _agent("critic",
                   findings="Here is a novel finding I discovered.",  # zero critique words
                   vote="ready"),
        ]],
        "output_md_chars": 1500,
    }


def _scenario_FM_1_3() -> dict:
    """FM-1.3 — Step Repetition: round-N findings byte-identical to round-(N-1)."""
    text = ("the same finding text repeated word for word across rounds "
            "without any change at all because the agent is stuck and is not "
            "responding to the critic feedback that was provided in the prior round")
    return {
        "rounds": [
            [_agent("researcher", agent_id="r01", findings=text)],
            [_agent("researcher", agent_id="r01", findings=text)],
        ],
        "output_md_chars": 1500,
    }


def _scenario_FM_1_4() -> dict:
    """FM-1.4 — Loss of Conversation History: round-2 agent saw empty context."""
    return {
        "rounds": [[_agent("researcher")]],
        "output_md_chars": 1500,
        "round_2_context_lengths": [0, 1500, 1500],  # one agent received nothing
    }


def _scenario_FM_1_5() -> dict:
    """FM-1.5 — Unaware of Termination: max-rounds with all-ready votes that don't halt.

    Pure-synthetic detection of FM-1.5 is hard — it's an orchestrator-level
    failure (orchestrator failed to halt despite consensus). Detection lives
    in the orchestrator integration tests, not the synthetic scoreboard.
    """
    return {
        "rounds": [
            [_agent("critic", vote="ready"), _agent("judge", vote="ready")],
            [_agent("critic", vote="ready"), _agent("judge", vote="ready")],
            [_agent("critic", vote="ready"), _agent("judge", vote="ready")],
        ],
        "output_md_chars": 1500,
    }


def _scenario_FM_2_1() -> dict:
    """FM-2.1 — Conversation Reset (proxy: oscillating ready-vote progress)."""
    return {
        "rounds": [
            [_agent("critic", vote="needs_work"), _agent("judge", vote="ready")],     # 1
            [_agent("critic", vote="ready"), _agent("judge", vote="ready"),
             _agent("res", vote="ready")],                                            # 3
            [_agent("critic", vote="needs_work"), _agent("judge", vote="ready")],     # 1
        ],
        "output_md_chars": 1500,
    }


def _scenario_FM_2_2() -> dict:
    """FM-2.2 — Failure to Ask for Clarification.

    Detection requires checking that ambiguous topics produce dissent.
    Pure-synthetic version: no detector; flagged in orchestrator integration
    tests with a deliberately-ambiguous topic.
    """
    return {
        "rounds": [[_agent("researcher", findings="x" * 500)]],
        "output_md_chars": 1500,
    }


def _scenario_FM_2_3() -> dict:
    """FM-2.3 — Task Derailment: synthesizer goes off-topic."""
    return {
        "rounds": [[
            _agent("researcher"),
            _agent("synthesizer", findings="Let's discuss kittens instead of databases."),
            _agent("quality_judge", vote="needs_work",
                   findings="Synthesizer derailed.",
                   coverage_score=2, accuracy_score=2, clarity_score=2, depth_score=2),
        ]],
        "output_md_chars": 1500,
    }


def _scenario_FM_2_4() -> dict:
    """FM-2.4 — Information Withholding: researcher returned findings but no key_points."""
    return {
        "rounds": [[
            _agent("researcher",
                   findings="Substantial research findings text that is non-empty.",
                   key_points=[]),
        ]],
        "output_md_chars": 1500,
    }


def _scenario_FM_2_5() -> dict:
    """FM-2.5 — Ignored Other Agents' Input: critic feedback ignored across rounds."""
    text = ("Same researcher findings as before with no changes made even though "
            "the critic explicitly flagged the missing coverage area in the prior round")
    return {
        "rounds": [
            [_agent("researcher", agent_id="r01", findings=text),
             _agent("critic", vote="needs_work",
                    findings="Researcher missed the coverage gap on x — flag concern issue")],
            [_agent("researcher", agent_id="r01", findings=text)],   # unchanged
        ],
        "output_md_chars": 1500,
    }


def _scenario_FM_2_6() -> dict:
    """FM-2.6 — Reasoning-Action Mismatch: low confidence with ready vote."""
    return {
        "rounds": [[
            _agent("researcher", confidence=0.2, vote="ready",
                   findings="I'm uncertain and would defer to others."),
        ]],
        "output_md_chars": 1500,
    }


def _scenario_FM_3_1() -> dict:
    """FM-3.1 — Premature Termination (proxy: silent partial output)."""
    return {
        "rounds": [[
            _agent("researcher", _partial=True, findings="partial findings"),
        ]],
        "output_md_chars": 1500,
    }


def _scenario_FM_3_2() -> dict:
    """FM-3.2 — No or Incomplete Verification: synthesizer ran without quality_judge.

    Detection is structural — the orchestrator's plan_agents must include
    a quality_judge. If absent, the bench detector inspects the agent
    roster (recorded in metrics.jsonl) and flags. Pure-synthetic version:
    no detector; this is in orchestrator integration tests.
    """
    return {
        "rounds": [[_agent("researcher"), _agent("synthesizer")]],  # no judge
        "output_md_chars": 1500,
    }


def _scenario_FM_3_3() -> dict:
    """FM-3.3 — Incorrect Verification: judge gives 10/10 to thin output."""
    return {
        "rounds": [[
            _agent("synthesizer", findings="N/A.", key_points=[]),
            _agent("quality_judge",
                   coverage_score=10, accuracy_score=10,
                   clarity_score=10, depth_score=10,
                   vote="ready"),
        ]],
        "output_md_chars": 100,  # very short output
    }


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------


SCENARIOS: list[Scenario] = [
    Scenario("FM-1.1", FM_NAMES["FM-1.1"], "FC1", _scenario_FM_1_1, detectable=False,
             notes="Requires LLM-based topic-relevance judgment; no rule-based detector."),
    Scenario("FM-1.2", FM_NAMES["FM-1.2"], "FC1", _scenario_FM_1_2, detectable=True,
             notes="Heuristic keyword overlap with role-template vocabulary."),
    Scenario("FM-1.3", FM_NAMES["FM-1.3"], "FC1", _scenario_FM_1_3, detectable=True,
             notes="Jaccard 5-gram > 0.95 between rounds for same agent."),
    Scenario("FM-1.4", FM_NAMES["FM-1.4"], "FC1", _scenario_FM_1_4, detectable=True,
             notes="Round-2+ agent received zero-length context."),
    Scenario("FM-1.5", FM_NAMES["FM-1.5"], "FC1", _scenario_FM_1_5, detectable=False,
             notes="Orchestrator-level failure; needs integration test."),
    Scenario("FM-2.1", FM_NAMES["FM-2.1"], "FC2", _scenario_FM_2_1, detectable=True,
             notes="Non-monotone ready_votes across rounds (oscillation)."),
    Scenario("FM-2.2", FM_NAMES["FM-2.2"], "FC2", _scenario_FM_2_2, detectable=False,
             notes="Requires ambiguity detection; orchestrator integration."),
    Scenario("FM-2.3", FM_NAMES["FM-2.3"], "FC2", _scenario_FM_2_3, detectable=False,
             notes="Synthesizer-content drift; needs LLM judgment beyond rule heuristics."),
    Scenario("FM-2.4", FM_NAMES["FM-2.4"], "FC2", _scenario_FM_2_4, detectable=True,
             notes="Researcher findings non-empty but key_points list empty."),
    Scenario("FM-2.5", FM_NAMES["FM-2.5"], "FC2", _scenario_FM_2_5, detectable=True,
             notes="Critic feedback present but researcher findings unchanged."),
    Scenario("FM-2.6", FM_NAMES["FM-2.6"], "FC2", _scenario_FM_2_6, detectable=True,
             notes="Confidence < 0.4 paired with quality_vote=='ready'."),
    Scenario("FM-3.1", FM_NAMES["FM-3.1"], "FC3", _scenario_FM_3_1, detectable=True,
             notes="_partial flag set without _flagged_partial."),
    Scenario("FM-3.2", FM_NAMES["FM-3.2"], "FC3", _scenario_FM_3_2, detectable=False,
             notes="Roster-level check; needs metrics.jsonl `agents_used` field."),
    Scenario("FM-3.3", FM_NAMES["FM-3.3"], "FC3", _scenario_FM_3_3, detectable=True,
             notes="Judge avg > 8 with output_md_chars < 500."),
]


# ---------------------------------------------------------------------------
# Scoreboard generation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ScoreboardRow:
    code: str
    name: str
    category: str
    expected_detectable: bool
    detected: bool
    detector_fired: bool   # True if FM code appeared in detect_all output
    notes: str


def run_scenario(scenario: Scenario) -> ScoreboardRow:
    kwargs = scenario.build()
    flags = detect_all(**kwargs)
    detector_fired = scenario.code in flags
    return ScoreboardRow(
        code=scenario.code,
        name=scenario.name,
        category=scenario.category,
        expected_detectable=scenario.detectable,
        detected=detector_fired,
        detector_fired=detector_fired,
        notes=scenario.notes,
    )


def score_all() -> list[ScoreboardRow]:
    return [run_scenario(s) for s in SCENARIOS]


def format_scoreboard(rows: list[ScoreboardRow], *, version_label: str = "v0.1.1") -> str:
    detected = sum(1 for r in rows if r.detected)
    expected = sum(1 for r in rows if r.expected_detectable)
    total = len(rows)

    lines = [
        f"# MAST Regression Scoreboard — {version_label}",
        "",
        f"**Detected: {detected}/{total}** ({expected} marked as detectable; "
        f"{total - expected} require orchestrator integration or LLM judgment).",
        "",
        f"Source: {len(SCENARIOS)} synthetic scenarios from "
        f"`bench/mast_regression.py`. Detectors live in `bench/detectors.py`.",
        "",
        "| Code | Name | FC | Expected | Fired | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        expected = "yes" if r.expected_detectable else "no"
        fired = "YES" if r.detector_fired else "no"
        lines.append(
            f"| {r.code} | {r.name} | {r.category} | {expected} | {fired} | {r.notes} |"
        )
    lines.append("")
    lines.append("## Coverage by category")
    lines.append("")
    by_cat: dict[str, list[ScoreboardRow]] = {}
    for r in rows:
        by_cat.setdefault(r.category, []).append(r)
    for cat in sorted(by_cat):
        cat_rows = by_cat[cat]
        cat_detected = sum(1 for r in cat_rows if r.detected)
        cat_expected = sum(1 for r in cat_rows if r.expected_detectable)
        lines.append(f"- **{cat}** ({cat_detected}/{len(cat_rows)} detected, "
                     f"{cat_expected} marked detectable)")
    lines.append("")
    lines.append("## Path to full coverage")
    lines.append("")
    lines.append("- FM-1.1, FM-2.3 require LLM-based content judgment (Phase 1: judge_ensemble).")
    lines.append("- FM-1.5, FM-2.2, FM-3.2 require orchestrator integration "
                 "(Phase 1: roster + halting hooks in `run_swarm`).")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="MAST regression scoreboard")
    ap.add_argument("--write", action="store_true",
                    help="Write to bench/mast_scoreboard.md instead of stdout")
    ap.add_argument("--version-label", default="v0.1.1")
    ap.add_argument("--json", action="store_true", help="Output JSON instead of markdown")
    args = ap.parse_args()

    rows = score_all()

    if args.json:
        out = json.dumps([
            {"code": r.code, "name": r.name, "category": r.category,
             "expected_detectable": r.expected_detectable,
             "detected": r.detected, "notes": r.notes}
            for r in rows
        ], indent=2)
    else:
        out = format_scoreboard(rows, version_label=args.version_label)

    if args.write:
        target = Path(__file__).parent / (
            "mast_scoreboard.json" if args.json else "mast_scoreboard.md"
        )
        target.write_text(out)
        print(f"wrote {target}")
    else:
        print(out)


if __name__ == "__main__":
    main()
