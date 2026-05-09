"""Mythos Swarm — main orchestration loop.

Plan → execute (parallel) → verify → either accept or replan, up to
max_replans rounds. Hard cost ceiling enforced. Every phase writes its
artifacts to disk so a partial run is still useful for debugging.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import artifact as artifact_mod
from . import planner as planner_mod
from . import verifier as verifier_mod
from .executor import ExecutorOutput, run_executors_parallel
from .planner import Plan
from .policies import CostBudget, MythosConfig
from .verifier import Verification


@dataclass
class MythosResult:
    task: str
    status: str                        # "passed" | "exhausted_replans" | "cost_exceeded" | "planner_failed"
    rounds: int                        # number of plan-exec-verify cycles run
    final_plan: Plan
    final_executor_outputs: list[ExecutorOutput]
    final_verification: Verification
    cost_usd: float
    wall_clock_s: float
    run_dir: Path
    history: list[dict] = field(default_factory=list)   # per-round summary

    def summary_dict(self) -> dict:
        return {
            "task": self.task,
            "status": self.status,
            "rounds": self.rounds,
            "cost_usd": round(self.cost_usd, 4),
            "wall_clock_s": round(self.wall_clock_s, 1),
            "run_dir": str(self.run_dir),
            "final_verdict": self.final_verification.verdict,
            "history": self.history,
        }


async def run_mythos(task: str, config: MythosConfig | None = None) -> MythosResult:
    """Run the full Mythos loop on a task.

    Lifecycle:
        plan → execute → verify
        if needs_work and budget allows → replan → execute → verify (up to max_replans)
        else → finalize
    """
    cfg = config or MythosConfig()
    budget = CostBudget(ceiling_usd=cfg.cost_ceiling_usd)

    run_dir = artifact_mod.make_run_dir(task, cfg.output_dir)
    t_start = time.monotonic()

    print(f"\n{'='*60}")
    print(f"MYTHOS SWARM — {task}")
    print(f"Run dir: {run_dir}")
    print(f"Budget:  ${budget.ceiling_usd:.2f} | "
          f"max replans: {cfg.max_replans} | "
          f"max executors: {cfg.max_executors}")
    print(f"Models:  planner={cfg.planner_model} | "
          f"executor={cfg.executor_model} | "
          f"verifier={cfg.verifier_model}")
    print(f"{'='*60}\n")

    history: list[dict] = []

    # --- Initial planning round ---
    print(f"--- Round 0: Planner ---")
    t0 = time.monotonic()
    plan = planner_mod.decompose(task, cfg)
    budget.charge(plan.cost_usd)
    print(f"  Planner done [{time.monotonic()-t0:.1f}s] "
          f"cost=${plan.cost_usd:.4f} budget=${budget.spent_usd:.2f}/${budget.ceiling_usd:.2f}")

    if plan.error or not plan.sub_specs:
        # Couldn't even produce an initial plan
        empty_v = Verification(
            verdict="needs_work",
            invariant_results=[],
            spec_results=[],
            required_fixes=[],
            summary=f"Planner failed: {plan.error or 'no sub-specs returned'}",
            confidence=0.0,
        )
        artifact_mod.write_plan(run_dir, plan, round_n=0)
        artifact_mod.write_verification(run_dir, empty_v, round_n=0)
        artifact_mod.write_final_artifact(
            run_dir, task, plan, [], empty_v, status="planner_failed",
        )
        result = MythosResult(
            task=task, status="planner_failed", rounds=0,
            final_plan=plan, final_executor_outputs=[],
            final_verification=empty_v,
            cost_usd=budget.spent_usd,
            wall_clock_s=time.monotonic() - t_start,
            run_dir=run_dir,
            history=history,
        )
        artifact_mod.write_run_summary(run_dir, result.summary_dict())
        _print_summary(result)
        return result

    artifact_mod.write_plan(run_dir, plan, round_n=0)

    # Cap sub-specs at max_executors (defensive — schema also enforces)
    if len(plan.sub_specs) > cfg.max_executors:
        print(f"  Plan had {len(plan.sub_specs)} sub-specs; capping to {cfg.max_executors}")
        plan.sub_specs = plan.sub_specs[: cfg.max_executors]

    # --- Iterate: execute → verify → maybe replan ---
    executor_outputs: list[ExecutorOutput] = []
    verification: Verification | None = None
    round_n = 0

    for round_n in range(cfg.max_replans + 1):
        # --- Cost ceiling check before executors ---
        if budget.exceeded():
            return _finalize_aborted(
                "cost_exceeded", task, plan, executor_outputs, verification,
                budget, round_n, run_dir, history, t_start,
            )

        # --- Execute (parallel) ---
        print(f"--- Round {round_n}: Executors ({len(plan.sub_specs)} parallel) ---")
        t0 = time.monotonic()
        executor_outputs = await run_executors_parallel(plan.sub_specs, cfg)
        for o in executor_outputs:
            budget.charge(o.cost_usd)
        elapsed = time.monotonic() - t0
        round_exec_cost = sum(o.cost_usd for o in executor_outputs)
        print(f"  Executors done [{elapsed:.1f}s] cost=${round_exec_cost:.4f} "
              f"budget=${budget.spent_usd:.2f}/${budget.ceiling_usd:.2f}")
        artifact_mod.write_executors(run_dir, executor_outputs, round_n=round_n)

        if budget.exceeded():
            return _finalize_aborted(
                "cost_exceeded", task, plan, executor_outputs, verification,
                budget, round_n, run_dir, history, t_start,
            )

        # --- Verify ---
        print(f"--- Round {round_n}: Verifier ---")
        t0 = time.monotonic()
        verification = verifier_mod.verify(task, plan, executor_outputs, cfg)
        budget.charge(verification.cost_usd)
        print(f"  Verifier done [{time.monotonic()-t0:.1f}s] "
              f"verdict={verification.verdict} cost=${verification.cost_usd:.4f} "
              f"budget=${budget.spent_usd:.2f}/${budget.ceiling_usd:.2f}")
        artifact_mod.write_verification(run_dir, verification, round_n=round_n)

        history.append({
            "round": round_n,
            "verdict": verification.verdict,
            "exec_cost_usd": round(round_exec_cost, 4),
            "verifier_cost_usd": round(verification.cost_usd, 4),
            "cumulative_cost_usd": round(budget.spent_usd, 4),
        })

        # --- Decide: pass, replan, or out of rounds ---
        if verification.passed:
            return _finalize_passed(
                task, plan, executor_outputs, verification,
                budget, round_n, run_dir, history, t_start,
            )

        if round_n >= cfg.max_replans:
            print(f"  Max replans ({cfg.max_replans}) reached. Finalizing as exhausted_replans.\n")
            break

        if budget.exceeded():
            return _finalize_aborted(
                "cost_exceeded", task, plan, executor_outputs, verification,
                budget, round_n, run_dir, history, t_start,
            )

        # --- Replan ---
        print(f"--- Round {round_n+1}: Replanner ---")
        t0 = time.monotonic()
        prev_plan = plan
        plan = planner_mod.replan(
            task, prev_plan,
            [o.to_dict() for o in executor_outputs],
            verification.to_dict(),
            cfg,
        )
        budget.charge(plan.cost_usd)
        print(f"  Replan done [{time.monotonic()-t0:.1f}s] "
              f"cost=${plan.cost_usd:.4f} budget=${budget.spent_usd:.2f}/${budget.ceiling_usd:.2f}")
        artifact_mod.write_plan(run_dir, plan, round_n=round_n + 1)

        if plan.error or not plan.sub_specs:
            print(f"  Replan failed: {plan.error or 'no sub-specs'}. Finalizing.")
            break

        if len(plan.sub_specs) > cfg.max_executors:
            plan.sub_specs = plan.sub_specs[: cfg.max_executors]

    # Fell through the loop = exhausted replans
    return _finalize_aborted(
        "exhausted_replans", task, plan, executor_outputs, verification,
        budget, round_n, run_dir, history, t_start,
    )


def _finalize_passed(
    task, plan, executor_outputs, verification,
    budget, round_n, run_dir, history, t_start,
) -> MythosResult:
    artifact_mod.write_final_artifact(
        run_dir, task, plan, executor_outputs, verification, status="passed",
    )
    result = MythosResult(
        task=task,
        status="passed",
        rounds=round_n + 1,
        final_plan=plan,
        final_executor_outputs=executor_outputs,
        final_verification=verification,
        cost_usd=budget.spent_usd,
        wall_clock_s=time.monotonic() - t_start,
        run_dir=run_dir,
        history=history,
    )
    artifact_mod.write_run_summary(run_dir, result.summary_dict())
    artifact_mod.write_metrics(run_dir, _build_metrics(history, budget, t_start))
    _print_summary(result)
    return result


def _finalize_aborted(
    status, task, plan, executor_outputs, verification,
    budget, round_n, run_dir, history, t_start,
) -> MythosResult:
    if verification is None:
        verification = Verification(
            verdict="needs_work",
            invariant_results=[],
            spec_results=[],
            required_fixes=[f"Run aborted before verification ({status})"],
            summary=f"Aborted: {status}",
            confidence=0.0,
        )
    artifact_mod.write_final_artifact(
        run_dir, task, plan, executor_outputs, verification, status=status,
    )
    result = MythosResult(
        task=task,
        status=status,
        rounds=round_n + 1 if executor_outputs else 0,
        final_plan=plan,
        final_executor_outputs=executor_outputs,
        final_verification=verification,
        cost_usd=budget.spent_usd,
        wall_clock_s=time.monotonic() - t_start,
        run_dir=run_dir,
        history=history,
    )
    artifact_mod.write_run_summary(run_dir, result.summary_dict())
    artifact_mod.write_metrics(run_dir, _build_metrics(history, budget, t_start))
    _print_summary(result)
    return result


def _build_metrics(history, budget, t_start) -> dict:
    return {
        "rounds": history,
        "total_cost_usd": round(budget.spent_usd, 4),
        "cost_ceiling_usd": budget.ceiling_usd,
        "wall_clock_s": round(time.monotonic() - t_start, 1),
    }


def _print_summary(result: MythosResult) -> None:
    print(f"\n{'='*60}")
    print(f"MYTHOS RUN COMPLETE — status: {result.status}")
    print(f"  rounds: {result.rounds}")
    print(f"  cost:   ${result.cost_usd:.4f} (ceiling: ${result.final_plan.cost_usd + result.cost_usd:.2f})")
    print(f"  wall:   {result.wall_clock_s:.1f}s")
    print(f"  output: {result.run_dir}")
    print(f"{'='*60}\n")
