"""GEPA adapter — wraps the gepa-ai/gepa library against blitz-swarm.

Anchor: arXiv 2507.19457 (ICLR 2026 oral) "GEPA: Reflective Prompt
Evolution Can Outperform Reinforcement Learning."

This module ships the adapter layer (no GEPA dependency required at
import time). The actual gepa.optimize() call lives in
scripts/optimize_prompts.py — that file imports gepa lazily and only
runs when Joona invokes the script.

The adapter is responsible for:
    1. Loading swarm role prompts as a candidate dict
    2. Wrapping the bench as an evaluator
    3. Translating GEPA's reflective dataset format into per-role
       feedback drawn from quality_judge breakdowns
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(slots=True)
class SwarmTask:
    task_id: str
    topic: str
    domain: str
    expected_traits: dict
    judge_rubric: dict


@dataclass(slots=True)
class SwarmTrajectory:
    task: SwarmTask
    role_outputs: dict[str, str]
    judge_breakdown: dict[str, float]
    final_score: float
    failure_modes: list[str] = field(default_factory=list)


# Pluggable hooks
RunWithPromptsFn = Callable[[SwarmTask, dict[str, str]], SwarmTrajectory]
ScoreFn = Callable[[SwarmTrajectory, dict], float]


@dataclass(slots=True)
class BlitzGEPAAdapter:
    """Adapter exposing the GEPA-required surface.

    GEPA expects:
        adapter.evaluate(candidate, batch, capture_traces) ->
            EvaluationBatch(outputs, scores, trajectories)
        adapter.make_reflective_dataset(candidate, eval_batch, components) ->
            dict[component_name, list[reflection_example]]

    We provide a lightweight in-process implementation. The full
    gepa.api.GEPAAdapter integration lives in scripts/optimize_prompts.py.
    """

    run_fn: RunWithPromptsFn
    score_fn: ScoreFn

    def evaluate(
        self,
        candidate: dict[str, str],
        batch: list[SwarmTask],
        *,
        capture_traces: bool = True,
    ) -> dict:
        outputs: list[str] = []
        scores: list[float] = []
        trajectories: list[SwarmTrajectory | None] = []
        for task in batch:
            traj = self.run_fn(task, candidate)
            score = self.score_fn(traj, task.judge_rubric)
            outputs.append(traj.role_outputs.get("synthesizer", ""))
            scores.append(score)
            trajectories.append(traj if capture_traces else None)
        return {
            "outputs": outputs,
            "scores": scores,
            "trajectories": trajectories,
        }

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: dict,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """Per-component reflection examples GEPA's reflection_lm rewrites."""
        result: dict[str, list[dict]] = {comp: [] for comp in components_to_update}
        for traj, score, out in zip(
            eval_batch["trajectories"], eval_batch["scores"], eval_batch["outputs"]
        ):
            if traj is None:
                continue
            for comp in components_to_update:
                role_output = traj.role_outputs.get(comp, "")
                result[comp].append({
                    "Inputs": {"task": traj.task.topic, "domain": traj.task.domain},
                    "Generated Output": role_output[:2000],
                    "Feedback": self._build_feedback(traj, comp, score),
                })
        return result

    def _build_feedback(
        self, traj: SwarmTrajectory, component: str, score: float,
    ) -> str:
        """Build < 1500 token feedback string from judge breakdown."""
        parts: list[str] = []
        parts.append(f"Score: {score:.2f}")
        if traj.judge_breakdown:
            dims = ", ".join(
                f"{k}={v:.1f}" for k, v in traj.judge_breakdown.items()
            )
            parts.append(f"Per-dim: {dims}")
        if traj.failure_modes:
            parts.append("Failure modes: " + ", ".join(traj.failure_modes))
        # Component-specific hint
        if "researcher" in component:
            parts.append("Suggestion: deeper citations + more numeric specifics.")
        elif "critic" in component:
            parts.append("Suggestion: name the load-bearing claim, demand evidence.")
        elif "synthesizer" in component:
            parts.append("Suggestion: preserve dissent; resolve contradictions explicitly.")
        return "\n".join(parts)


def load_role_prompts(
    role_files: dict[str, Path],
) -> dict[str, str]:
    """Load role prompts from disk into a candidate dict for GEPA."""
    return {role: path.read_text(encoding="utf-8") for role, path in role_files.items()}
