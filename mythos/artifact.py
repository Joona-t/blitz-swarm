"""Mythos Swarm — artifact assembly.

Writes a complete record of the run to disk:
- plan.md, plan.json
- executor_NN_output.md (one per spec)
- verification.md, verification.json
- final_artifact.md (human-readable assembly)
- metrics.json (cost, tokens, replans, wall clock)
- run.json (machine-readable summary of the entire run)
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .planner import Plan
from .verifier import Verification


def _slug(text: str, max_len: int = 50) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())[:max_len]
    return s.strip("_") or "task"


def make_run_dir(task: str, base_output_dir: str) -> Path:
    """Create output/mythos/<slug>_<timestamp>/ and return the path."""
    base = Path(base_output_dir)
    if not base.is_absolute():
        base = Path(__file__).parent.parent / base_output_dir
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{_slug(task)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_plan(run_dir: Path, plan: Plan, round_n: int) -> None:
    suffix = f"_round{round_n}" if round_n > 0 else ""
    (run_dir / f"plan{suffix}.json").write_text(
        json.dumps(plan.raw or plan.__dict__, indent=2, default=str),
        encoding="utf-8",
    )
    md = [
        f"# Plan{f' (replan round {round_n})' if round_n > 0 else ''}",
        "",
        f"**Confidence:** {plan.confidence:.2f}",
        "",
        "## Task Summary",
        plan.task_summary or "_(empty)_",
        "",
        "## Sub-Specs",
    ]
    for s in plan.sub_specs:
        md.append(f"### {s.get('id', '?')}: {s.get('title', '')}")
        md.append("")
        md.append(s.get("description", ""))
        md.append("")
        md.append(f"**Deliverable:** {s.get('deliverable', '')}")
        md.append("")
        md.append("**Acceptance Criteria:**")
        for c in s.get("acceptance_criteria", []):
            md.append(f"- {c}")
        md.append("")
    md.append("## Global Invariants")
    for inv in plan.global_invariants:
        md.append(f"- {inv}")
    md.append("")
    md.append("## Verification Strategy")
    md.append(plan.verification_strategy or "_(none)_")
    md.append("")
    md.append("## Rationale")
    md.append(plan.rationale or "_(none)_")
    if plan.error:
        md.append("")
        md.append(f"**ERROR:** {plan.error}")
    (run_dir / f"plan{suffix}.md").write_text("\n".join(md), encoding="utf-8")


def write_executors(run_dir: Path, executor_outputs: list, round_n: int) -> None:
    suffix = f"_round{round_n}" if round_n > 0 else ""
    for i, o in enumerate(executor_outputs):
        md = [
            f"# Executor Output: {o.spec_id}",
            "",
            f"**Confidence:** {o.confidence:.2f} | "
            f"**Cost:** ${o.cost_usd:.4f} | "
            f"**Wall:** {o.elapsed_s:.1f}s",
            "",
            "## Deliverable",
            o.deliverable or "_(empty)_",
            "",
            "## Self-Check",
        ]
        for c in o.acceptance_check:
            mark = "✅" if c.get("satisfied") else "❌"
            md.append(f"- {mark} {c.get('criterion', '')}")
            if c.get("evidence"):
                md.append(f"  - _evidence:_ {c['evidence']}")
        if o.concerns:
            md.append("")
            md.append("## Concerns")
            for c in o.concerns:
                md.append(f"- {c}")
        if o.error:
            md.append("")
            md.append(f"**ERROR:** {o.error}")
        (run_dir / f"executor_{i:02d}_{o.spec_id}{suffix}.md").write_text(
            "\n".join(md), encoding="utf-8",
        )


def write_verification(run_dir: Path, verification: Verification, round_n: int) -> None:
    suffix = f"_round{round_n}" if round_n > 0 else ""
    (run_dir / f"verification{suffix}.json").write_text(
        json.dumps(verification.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    md = [
        f"# Verification{f' (round {round_n})' if round_n > 0 else ''}",
        "",
        f"**Verdict:** `{verification.verdict}` | "
        f"**Confidence:** {verification.confidence:.2f}",
        "",
        "## Summary",
        verification.summary or "_(none)_",
        "",
        "## Invariants",
    ]
    for ir in verification.invariant_results:
        status = ir.get("status", "?")
        mark = {"pass": "✅", "fail": "❌", "unverifiable": "⚠️"}.get(status, "?")
        md.append(f"- {mark} **{status}** — {ir.get('invariant', '')}")
        if ir.get("evidence"):
            md.append(f"  - _evidence:_ {ir['evidence']}")
    md.append("")
    md.append("## Sub-Specs")
    for sr in verification.spec_results:
        status = sr.get("status", "?")
        mark = {"pass": "✅", "partial": "⚠️", "fail": "❌"}.get(status, "?")
        md.append(f"- {mark} **{sr.get('spec_id', '?')}** — {status}")
        for issue in sr.get("issues", []) or []:
            md.append(f"  - {issue}")
    if verification.required_fixes:
        md.append("")
        md.append("## Required Fixes")
        for fix in verification.required_fixes:
            md.append(f"- {fix}")
    if verification.error:
        md.append("")
        md.append(f"**ERROR:** {verification.error}")
    (run_dir / f"verification{suffix}.md").write_text("\n".join(md), encoding="utf-8")


def write_final_artifact(
    run_dir: Path,
    task: str,
    plan: Plan,
    executor_outputs: list,
    verification: Verification,
    status: str,
) -> None:
    """Assemble a single human-readable final artifact from executor deliverables."""
    md = [
        f"# Mythos Swarm Output",
        "",
        f"**Status:** `{status}`",
        f"**Final verdict:** `{verification.verdict}`",
        "",
        "## Task",
        task,
        "",
        "## Plan Summary",
        plan.task_summary or "_(empty)_",
        "",
        "## Assembled Deliverables",
    ]
    for o in executor_outputs:
        md.append(f"### {o.spec_id}")
        md.append("")
        md.append(o.deliverable or "_(empty)_")
        md.append("")
    md.append("## Verification")
    md.append(verification.summary or "_(none)_")
    if verification.required_fixes:
        md.append("")
        md.append("**Required Fixes (unresolved):**")
        for fix in verification.required_fixes:
            md.append(f"- {fix}")
    (run_dir / "final_artifact.md").write_text("\n".join(md), encoding="utf-8")


def write_metrics(run_dir: Path, metrics: dict) -> None:
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8",
    )


def write_run_summary(run_dir: Path, summary: dict) -> None:
    (run_dir / "run.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
