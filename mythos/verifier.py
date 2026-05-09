"""Mythos Verifier — gates the swarm's work.

Reads the original task, the planner's plan, and all executor outputs.
Returns a structured verdict (pass | needs_work) with per-invariant and
per-spec breakdowns plus required_fixes the planner can act on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import _invoke
from .policies import MythosConfig
from .schemas import VERIFIER_SCHEMA_JSON


@dataclass
class Verification:
    verdict: str                       # "pass" | "needs_work"
    invariant_results: list[dict]
    spec_results: list[dict]
    required_fixes: list[str]
    summary: str
    confidence: float
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    raw: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "invariant_results": self.invariant_results,
            "spec_results": self.spec_results,
            "required_fixes": self.required_fixes,
            "summary": self.summary,
            "confidence": self.confidence,
            "cost_usd": round(self.cost_usd, 4),
            "elapsed_s": round(self.elapsed_s, 1),
            "error": self.error,
        }


def verify(
    task: str,
    plan,                              # mythos.planner.Plan
    executor_outputs: list,            # list[ExecutorOutput]
    config: MythosConfig,
) -> Verification:
    system = _invoke.load_prompt("verifier")

    # Build the verifier's view: full plan + full executor deliverables.
    # We send full deliverables (no truncation) — the verifier needs them
    # to actually check.
    plan_view = {
        "task_summary": plan.task_summary,
        "sub_specs": plan.sub_specs,
        "global_invariants": plan.global_invariants,
        "verification_strategy": plan.verification_strategy,
    }

    exec_view = [o.to_dict() for o in executor_outputs]

    user = (
        "## Original task\n"
        f"{task}\n\n"
        "## Plan\n"
        f"{json.dumps(plan_view, indent=2)}\n\n"
        "## Executor outputs\n"
        f"{json.dumps(exec_view, indent=2)}\n\n"
        "## Your job\n"
        "Decide pass | needs_work. For each global invariant and each sub-spec, "
        "give an evidence-cited status. If any failure, list concrete required_fixes "
        "the planner can use to replan.\n\n"
        "Return only the JSON object matching the verifier schema."
    )

    res = _invoke.invoke(
        role="verifier",
        system_prompt=system,
        user_prompt=user,
        schema_json=VERIFIER_SCHEMA_JSON,
        model_alias=config.verifier_model,
        timeout_s=config.verifier_timeout_s,
    )

    if res.parsed is None:
        # Verifier failure is treated as needs_work — better safe than wrong.
        return Verification(
            verdict="needs_work",
            invariant_results=[],
            spec_results=[],
            required_fixes=[
                "Verifier could not produce a structured verdict; "
                "re-run with simpler verification strategy or smaller sub-specs."
            ],
            summary=res.error or "verifier returned no JSON",
            confidence=0.0,
            cost_usd=res.cost_usd,
            elapsed_s=res.elapsed_s,
            error=res.error,
        )

    parsed = res.parsed
    return Verification(
        verdict=parsed.get("verdict", "needs_work"),
        invariant_results=list(parsed.get("invariant_results", [])),
        spec_results=list(parsed.get("spec_results", [])),
        required_fixes=list(parsed.get("required_fixes", [])),
        summary=parsed.get("summary", ""),
        confidence=float(parsed.get("confidence", 0.0)),
        cost_usd=res.cost_usd,
        elapsed_s=res.elapsed_s,
        raw=parsed,
    )
