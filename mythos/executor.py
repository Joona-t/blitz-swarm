"""Mythos Executor — implements one sub-spec.

Sonnet-class agent (per default config). Sees ONE sub-spec and produces
a concrete deliverable + self-check. Multiple executors run in parallel
via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from . import _invoke
from .policies import MythosConfig
from .schemas import EXECUTOR_SCHEMA_JSON


@dataclass
class ExecutorOutput:
    spec_id: str
    deliverable: str
    acceptance_check: list[dict]
    concerns: list[str]
    confidence: float
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    raw: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        d = {
            "spec_id": self.spec_id,
            "deliverable": self.deliverable,
            "acceptance_check": self.acceptance_check,
            "concerns": self.concerns,
            "confidence": self.confidence,
            "cost_usd": round(self.cost_usd, 4),
            "elapsed_s": round(self.elapsed_s, 1),
        }
        if self.error:
            d["error"] = self.error
        return d


def run_executor(sub_spec: dict, config: MythosConfig) -> ExecutorOutput:
    """Synchronous executor invocation. Wrap with asyncio.to_thread to parallelize."""
    spec_id = sub_spec.get("id", "unknown")
    system = _invoke.load_prompt("executor")
    user = (
        "## Your assigned sub-spec\n"
        f"{json.dumps(sub_spec, indent=2)}\n\n"
        "## Your job\n"
        "Produce the deliverable described above. Self-check against every "
        "acceptance criterion. Surface any concerns the verifier should know about.\n\n"
        "Return only the JSON object matching the executor schema."
    )

    res = _invoke.invoke(
        role=f"executor[{spec_id}]",
        system_prompt=system,
        user_prompt=user,
        schema_json=EXECUTOR_SCHEMA_JSON,
        model_alias=config.executor_model,
        timeout_s=config.executor_timeout_s,
    )

    if res.parsed is None:
        return ExecutorOutput(
            spec_id=spec_id,
            deliverable="",
            acceptance_check=[],
            concerns=[],
            confidence=0.0,
            cost_usd=res.cost_usd,
            elapsed_s=res.elapsed_s,
            error=res.error or "executor returned no JSON",
        )

    parsed = res.parsed
    return ExecutorOutput(
        spec_id=parsed.get("spec_id", spec_id),
        deliverable=parsed.get("deliverable", ""),
        acceptance_check=list(parsed.get("acceptance_check", [])),
        concerns=list(parsed.get("concerns", [])),
        confidence=float(parsed.get("confidence", 0.0)),
        cost_usd=res.cost_usd,
        elapsed_s=res.elapsed_s,
        raw=parsed,
    )


async def run_executors_parallel(
    sub_specs: list[dict],
    config: MythosConfig,
) -> list[ExecutorOutput]:
    """Run all executors in parallel via asyncio.to_thread."""
    if not sub_specs:
        return []
    coros = [
        asyncio.to_thread(run_executor, spec, config)
        for spec in sub_specs
    ]
    return list(await asyncio.gather(*coros))
