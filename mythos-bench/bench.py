#!/usr/bin/env python3
"""mythos-bench CLI.

    python mythos-bench/bench.py list
    python mythos-bench/bench.py run <task_id> --mode mythos|consensus
    python mythos-bench/bench.py verify <task_id> --mode mythos|consensus --run-dir <path>
    python mythos-bench/bench.py backfill <task_id> --mode mythos|consensus --run-dir <path>
    python mythos-bench/bench.py compare
    python mythos-bench/bench.py results

`run` runs the swarm AND verifies in one shot.
`backfill` records an existing already-completed run dir as a result.
`compare` writes results/comparison.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Allow `python mythos-bench/bench.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import (  # noqa: E402
    ARTIFACTS_DIR,
    BENCH_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    RESULTS_LOG,
    RunInfo,
    TaskSpec,
    append_result,
    list_tasks,
    parse_metrics_from_log,
    read_results,
    run_task,
    synthesize_verdict,
    verify_run,
)


# ───────────────────────────────────────────────────────────────────────────
# Commands
# ───────────────────────────────────────────────────────────────────────────

def cmd_list(args) -> int:
    tasks = list_tasks()
    print(f"\n{len(tasks)} task(s) in {BENCH_DIR.relative_to(REPO_ROOT)}/tasks/:\n")
    for t in tasks:
        adv = " [adversarial]" if t.adversarial_file else ""
        print(f"  {t.id:<24}  {t.title}{adv}")
    print()
    return 0


def cmd_run(args) -> int:
    task = TaskSpec.load(args.task_id)
    run = run_task(task, args.mode)

    log_text = run.cli_stdout_path.read_text()
    metrics = parse_metrics_from_log(log_text, args.mode)

    print(f"\n[bench] orchestrator exit={run.exit_code}, parsing & verifying...")
    verify_data = verify_run(task, run, run_adversarial=not args.skip_adversarial)
    verdict = synthesize_verdict(task, metrics, verify_data)

    record = _build_record(task, run, metrics, verify_data, verdict)
    append_result(record)
    _print_run_summary(record)

    return 0 if verdict == "passed" else 1


def cmd_backfill(args) -> int:
    """Record an existing run dir as a result without re-running the swarm."""
    task = TaskSpec.load(args.task_id)
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"[bench] error: run dir not found: {run_dir}")
        return 2

    if args.mode == "mythos":
        output_path = run_dir / "final_artifact.md"
    else:
        # Consensus run dir contains a single .md file; pick the one matching the task slug
        candidates = list(run_dir.glob("*.md"))
        if not candidates:
            print(f"[bench] error: no .md output in {run_dir}")
            return 2
        output_path = candidates[0]

    run = RunInfo(
        task_id=task.id,
        mode=args.mode,
        started_at=args.started_at or "(backfilled)",
        finished_at=args.finished_at or datetime.now().isoformat(timespec="seconds"),
        exit_code=0,
        cli_stdout_path=Path(args.log_path).resolve() if args.log_path else Path("/dev/null"),
        run_dir=run_dir,
        output_path=output_path,
    )

    log_text = run.cli_stdout_path.read_text() if run.cli_stdout_path.exists() else ""
    metrics = parse_metrics_from_log(log_text, args.mode) if log_text else {}
    # Permit metrics overrides for backfill (we have them from prior reports)
    for k in ("cost_usd", "wall_clock_s", "rounds", "invocations", "verifier_verdict", "consensus_reached"):
        v = getattr(args, k, None)
        if v is not None:
            metrics[k] = v

    print(f"[bench] backfilling {task.id} | mode={args.mode} | run_dir={run_dir}")
    verify_data = verify_run(task, run, run_adversarial=not args.skip_adversarial)
    verdict = synthesize_verdict(task, metrics, verify_data)

    record = _build_record(task, run, metrics, verify_data, verdict)
    record["backfilled"] = True
    append_result(record)
    _print_run_summary(record)
    return 0


def cmd_compare(args) -> int:
    """Aggregate results.jsonl into a markdown comparison table."""
    rows = read_results()
    if not rows:
        print("[bench] no results yet — run some tasks first.")
        return 1

    md = ["# mythos-bench — comparison\n",
          f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from {RESULTS_LOG.relative_to(REPO_ROOT)} ({len(rows)} run(s))._\n",
          "## All runs\n",
          "| Task | Mode | Verdict | Cost | Wall | Inv | Rounds | Verifier | Primary | Adversarial |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cost = f"${r.get('cost_usd', 0):.2f}" if r.get('cost_usd') is not None else "—"
        wall = f"{r.get('wall_clock_s', 0):.0f}s" if r.get('wall_clock_s') is not None else "—"
        inv = str(r.get('invocations', '—'))
        rounds = str(r.get('rounds', '—'))
        verifier = r.get('verifier_verdict') or ("yes" if r.get('consensus_reached') else "no" if r.get('consensus_reached') is False else "—")
        pp = r['primary_pytest']
        primary = f"{pp['passed']}p/{pp['failed']}f/{pp['errors']}e"
        ap = r.get('adversarial_pytest')
        adv = f"{ap['passed']}p/{ap['failed']}f/{ap['errors']}e" if ap else "—"
        md.append(f"| {r['task_id']} | {r['mode']} | `{r['verdict']}` | {cost} | {wall} | {inv} | {rounds} | {verifier} | {primary} | {adv} |")

    # Per-task head-to-head
    md.append("\n## Head-to-head per task\n")
    by_task: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by_task.setdefault(r['task_id'], {}).setdefault(r['mode'], []).append(r)
    for task_id, by_mode in by_task.items():
        if 'mythos' in by_mode and 'consensus' in by_mode:
            m = by_mode['mythos'][-1]
            c = by_mode['consensus'][-1]
            md.append(f"### {task_id}\n")
            md.append(f"- **Mythos**: {m['verdict']} | ${m.get('cost_usd', 0):.2f} | {m.get('wall_clock_s', 0):.0f}s | {m.get('invocations', '?')} invocations | rounds={m.get('rounds', '?')}")
            md.append(f"- **Consensus**: {c['verdict']} | ${c.get('cost_usd', 0):.2f} | {c.get('wall_clock_s', 0):.0f}s | {c.get('invocations', '?')} invocations | rounds={c.get('rounds', '?')}")
            if m.get('cost_usd') and c.get('cost_usd'):
                delta = (m['cost_usd'] - c['cost_usd']) / c['cost_usd'] * 100
                md.append(f"- Cost delta: Mythos {delta:+.0f}% vs consensus")

    out_path = RESULTS_DIR / "comparison.md"
    out_path.write_text("\n".join(md) + "\n")
    print(f"[bench] wrote {out_path}")
    return 0


def cmd_results(args) -> int:
    rows = read_results()
    print(json.dumps(rows, indent=2, default=str))
    return 0


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _build_record(task: TaskSpec, run: RunInfo, metrics: dict, verify_data: dict, verdict: str) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task_id": task.id,
        "mode": run.mode,
        "exit_code": run.exit_code,
        "verdict": verdict,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "cost_usd": metrics.get("cost_usd"),
        "wall_clock_s": metrics.get("wall_clock_s"),
        "rounds": metrics.get("rounds"),
        "invocations": metrics.get("invocations"),
        "verifier_verdict": metrics.get("verifier_verdict"),
        "consensus_reached": metrics.get("consensus_reached"),
        "primary_pytest": verify_data["primary_pytest"],
        "adversarial_pytest": verify_data.get("adversarial_pytest"),
        "run_dir": str(run.run_dir) if run.run_dir else None,
        "output_path": str(run.output_path) if run.output_path else None,
        "log_path": str(run.cli_stdout_path),
        "extracted_files": verify_data.get("extracted_files", []),
        "workdir": verify_data.get("workdir"),
    }


def _print_run_summary(record: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"  task:       {record['task_id']}")
    print(f"  mode:       {record['mode']}")
    print(f"  verdict:    {record['verdict']}")
    if record.get('cost_usd') is not None:
        print(f"  cost:       ${record['cost_usd']:.2f}")
    if record.get('wall_clock_s') is not None:
        print(f"  wall:       {record['wall_clock_s']:.1f}s")
    pp = record['primary_pytest']
    print(f"  primary:    {pp['passed']} passed / {pp['failed']} failed / {pp['errors']} errors")
    ap = record.get('adversarial_pytest')
    if ap:
        print(f"  adversarial:{ap['passed']} passed / {ap['failed']} failed / {ap['errors']} errors")
    print(f"  workdir:    {record.get('workdir', '(none)')}")
    print(f"{'─'*60}\n")


# ───────────────────────────────────────────────────────────────────────────
# argparse
# ───────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="mythos-bench: head-to-head harness for blitz-swarm modes")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    p_run = sub.add_parser("run", help="run swarm + verify + log")
    p_run.add_argument("task_id")
    p_run.add_argument("--mode", choices=["consensus", "mythos"], required=True)
    p_run.add_argument("--skip-adversarial", action="store_true")

    p_back = sub.add_parser("backfill", help="record an existing completed run as a result")
    p_back.add_argument("task_id")
    p_back.add_argument("--mode", choices=["consensus", "mythos"], required=True)
    p_back.add_argument("--run-dir", required=True)
    p_back.add_argument("--log-path", default=None)
    p_back.add_argument("--cost-usd", type=float, default=None)
    p_back.add_argument("--wall-clock-s", type=float, default=None, dest="wall_clock_s")
    p_back.add_argument("--rounds", type=int, default=None)
    p_back.add_argument("--invocations", type=int, default=None)
    p_back.add_argument("--verifier-verdict", default=None, dest="verifier_verdict")
    p_back.add_argument("--consensus-reached", type=lambda v: v.lower() == "true", default=None, dest="consensus_reached")
    p_back.add_argument("--started-at", default=None, dest="started_at")
    p_back.add_argument("--finished-at", default=None, dest="finished_at")
    p_back.add_argument("--skip-adversarial", action="store_true")

    sub.add_parser("compare", help="generate comparison.md")
    sub.add_parser("results", help="dump results.jsonl as JSON")

    args = p.parse_args()
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "backfill":
        return cmd_backfill(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "results":
        return cmd_results(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
