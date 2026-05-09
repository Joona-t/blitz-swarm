"""Mythos Planner — decomposes a task into sub-specs + invariants.

Plain agent: one shot of `claude -p --model opus --effort max` with the
planner.md system prompt and the planner JSON schema. Replanning passes
the prior plan + executor outputs + verifier feedback as additional
context so the planner can fix what failed without redoing what worked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import _invoke
from .policies import MythosConfig
from .schemas import PLANNER_SCHEMA_JSON


@dataclass
class Plan:
    task_summary: str
    sub_specs: list[dict]
    global_invariants: list[str]
    verification_strategy: str
    rationale: str
    confidence: float
    raw: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    error: str | None = None

    @classmethod
    def from_parsed(cls, parsed: dict, cost_usd: float) -> "Plan":
        return cls(
            task_summary=parsed.get("task_summary", ""),
            sub_specs=list(parsed.get("sub_specs", [])),
            global_invariants=list(parsed.get("global_invariants", [])),
            verification_strategy=parsed.get("verification_strategy", ""),
            rationale=parsed.get("rationale", ""),
            confidence=float(parsed.get("confidence", 0.0)),
            raw=parsed,
            cost_usd=cost_usd,
        )

    @classmethod
    def errored(cls, error: str, cost_usd: float = 0.0) -> "Plan":
        return cls(
            task_summary="",
            sub_specs=[],
            global_invariants=[],
            verification_strategy="",
            rationale="",
            confidence=0.0,
            cost_usd=cost_usd,
            error=error,
        )


def decompose(task: str, config: MythosConfig) -> Plan:
    """Initial decomposition pass."""
    system = _invoke.load_prompt("planner")
    user = (
        "## Task\n"
        f"{task}\n\n"
        "## Your job\n"
        f"Decompose this task into at most {config.max_executors} parallel sub-specs "
        "with clear acceptance criteria and global invariants the verifier can check.\n\n"
        "Return only the JSON object matching the planner schema."
    )

    res = _invoke.invoke(
        role="planner",
        system_prompt=system,
        user_prompt=user,
        schema_json=PLANNER_SCHEMA_JSON,
        model_alias=config.planner_model,
        timeout_s=config.planner_timeout_s,
    )

    if res.parsed is None:
        return Plan.errored(res.error or "planner returned no JSON", cost_usd=res.cost_usd)
    return Plan.from_parsed(res.parsed, cost_usd=res.cost_usd)


def replan(
    task: str,
    prev_plan: Plan,
    executor_outputs: list[dict],
    verification: dict,
    config: MythosConfig,
) -> Plan:
    """Replan after a failed verification.

    Pass enough context for the planner to diagnose what failed without
    redoing what already worked.
    """
    system = _invoke.load_prompt("planner")

    # Compact the executor outputs — we don't need full deliverables to replan,
    # we need to know which spec_ids passed/failed and the verifier's notes.
    exec_summary = "\n".join(
        f"- {o.get('spec_id', '?')}: confidence={o.get('confidence', 0):.2f}, "
        f"concerns={o.get('concerns', [])[:3]}"
        for o in executor_outputs
    ) or "(no executor outputs)"

    failed_invariants = [
        ir for ir in verification.get("invariant_results", [])
        if ir.get("status") in ("fail", "unverifiable")
    ]
    failed_specs = [
        sr for sr in verification.get("spec_results", [])
        if sr.get("status") in ("fail", "partial")
    ]

    user = (
        "## Task\n"
        f"{task}\n\n"
        "## Previous Plan Summary\n"
        f"{prev_plan.task_summary}\n\n"
        f"Sub-specs ({len(prev_plan.sub_specs)}): "
        f"{[s.get('id') for s in prev_plan.sub_specs]}\n"
        f"Invariants: {prev_plan.global_invariants}\n"
        f"Verification strategy: {prev_plan.verification_strategy}\n\n"
        "## Executor Output Summary\n"
        f"{exec_summary}\n\n"
        "## Verifier Verdict: needs_work\n"
        f"Summary: {verification.get('summary', '(none)')}\n\n"
        f"Failed invariants: {failed_invariants}\n\n"
        f"Failed specs: {failed_specs}\n\n"
        f"Required fixes: {verification.get('required_fixes', [])}\n\n"
        "## Your job\n"
        "Produce a NEW plan that addresses the verifier's required fixes. "
        "Preserve sub-specs that passed; rewrite or split sub-specs that failed. "
        "Tighten invariants the verifier flagged as ambiguous.\n\n"
        "Return only the JSON object matching the planner schema."
    )

    res = _invoke.invoke(
        role="planner-replan",
        system_prompt=system,
        user_prompt=user,
        schema_json=PLANNER_SCHEMA_JSON,
        model_alias=config.planner_model,
        timeout_s=config.planner_timeout_s,
    )

    if res.parsed is None:
        return Plan.errored(res.error or "replanner returned no JSON", cost_usd=res.cost_usd)
    return Plan.from_parsed(res.parsed, cost_usd=res.cost_usd)
