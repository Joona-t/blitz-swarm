"""Blitz-Swarm bench runner — executes a slate against the swarm.

Lifecycle:
    1. load_slate
    2. filter by args
    3. for each prompt:
         a. budget gate
         b. invoke run_swarm (or injected swarm_fn for tests)
         c. find matching metrics.jsonl record
         d. read final markdown output
         e. compute coverage hits, MAST flags
         f. write per-prompt JSONL row (atomic, fsync)
    4. aggregate -> summary.json
    5. write stats.md

Design notes:
- `swarm_fn` is injected so tests can mock without spending $.
- Per-prompt rows are appended to results.jsonl with fsync; the runner
  is resumable: re-running skips prompt_ids already present in the file.
- Cost tracking uses `total_cost_usd` from `metrics.jsonl` (CLI envelope).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

# Make the blitz-swarm package root importable when running as a module.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench import BenchPrompt, BenchSlate, load_slate
from bench.detectors import detect_all
from bench.stats import bootstrap_ci, paired_t_test, cohens_d


BENCH_ROOT = Path(__file__).resolve().parent
RUNS_DIR = BENCH_ROOT / "runs"
METRICS_PATH = _REPO_ROOT / "metrics.jsonl"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BenchConfig:
    """Runtime config for one bench run."""

    slate_path: Path
    slate_filter_ids: tuple[str, ...] = ()
    slate_filter_tiers: tuple[str, ...] = ()
    slate_filter_domains: tuple[str, ...] = ()
    parallel_prompts: int = 2
    per_prompt_timeout_s: int = 1200
    total_budget_usd: float = 8.00
    max_rounds: int = 4
    use_redis: bool = False
    seed: int = 42
    started_utc: str = ""
    git_sha: str = ""

    def filter_slate(self, slate: BenchSlate) -> list[BenchPrompt]:
        return slate.filter(
            ids=self.slate_filter_ids or None,
            tiers=self.slate_filter_tiers or None,
            domains=self.slate_filter_domains or None,
        )


@dataclass(slots=True)
class PromptResult:
    """One row of bench/runs/<run_id>/results.jsonl."""

    prompt_id: str
    run_id: str
    swarm_run_id: str
    prompt_sha256: str
    output_md_path: str
    started_utc: str
    elapsed_s: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    consensus_reached: bool
    rounds_to_consensus: int
    quality: dict          # {coverage, accuracy, clarity, depth, avg}
    coverage_hits: int
    coverage_total: int
    timeout: bool
    error: str | None
    mast_flags: list[str]


# Type alias for the swarm function (injected for testing)
SwarmFn = Callable[..., Awaitable[Path]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str:
    """Return short git SHA, or 'nogit' if unavailable."""
    try:
        sha = os.popen("git -C " + str(_REPO_ROOT) + " rev-parse HEAD 2>/dev/null").read().strip()
        return sha[:12] if sha else "nogit"
    except Exception:
        return "nogit"


def _run_id(cfg: BenchConfig, slate: BenchSlate) -> str:
    """Deterministic run_id: started_utc + git_sha + slate_id."""
    timestamp = cfg.started_utc.replace(":", "").replace("-", "").split("+")[0]
    return f"{timestamp}_{cfg.git_sha}_{slate.slate_id}"


def _resumable_ids(run_dir: Path) -> set[str]:
    """Read existing results.jsonl and return prompt_ids already done."""
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return set()
    done: set[str] = set()
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "prompt_id" in row:
                    done.add(row["prompt_id"])
            except json.JSONDecodeError:
                continue
    return done


def _append_jsonl(path: Path, row: dict) -> None:
    """Atomic append + fsync."""
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _coverage_count(md: str, expected: list[str]) -> int:
    """Count how many expected_coverage keywords are present (case-insensitive)."""
    if not md or not expected:
        return 0
    md_lower = md.lower()
    hits = 0
    for kw in expected:
        if kw.lower() in md_lower:
            hits += 1
    return hits


def _find_metrics_record_for_topic(
    topic: str,
    *,
    metrics_path: Path = METRICS_PATH,
    after_timestamp: float | None = None,
) -> dict | None:
    """Find the most recent metrics.jsonl record matching the topic.

    If after_timestamp is given, only rows with timestamp > after_timestamp
    are considered (used to avoid picking up an old run with the same topic).
    """
    if not metrics_path.exists():
        return None
    matches: list[dict] = []
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("topic") != topic:
                continue
            if after_timestamp is not None and float(row.get("timestamp", 0)) <= after_timestamp:
                continue
            matches.append(row)
    if not matches:
        return None
    return max(matches, key=lambda r: float(r.get("timestamp", 0)))


def _build_quality_dict(record: dict | None) -> dict:
    """Extract quality scores from a metrics.jsonl record."""
    if not record:
        return {"coverage": 0, "accuracy": 0, "clarity": 0, "depth": 0, "avg": 0.0}
    return {
        "coverage": float(record.get("coverage", 0) or 0),
        "accuracy": float(record.get("accuracy", 0) or 0),
        "clarity": float(record.get("clarity", 0) or 0),
        "depth": float(record.get("depth", 0) or 0),
        "avg": float(record.get("avg_quality", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Per-prompt execution
# ---------------------------------------------------------------------------


async def run_one_prompt(
    prompt: BenchPrompt,
    cfg: BenchConfig,
    run_dir: Path,
    *,
    swarm_fn: SwarmFn,
    metrics_path: Path = METRICS_PATH,
) -> PromptResult:
    """Run a single prompt through the swarm and collect its result."""
    started = time.monotonic()
    started_utc_iso = _utc()
    pre_run_timestamp = time.time()

    try:
        out_path = await asyncio.wait_for(
            swarm_fn(
                prompt.text,
                max_rounds=cfg.max_rounds,
                use_redis=cfg.use_redis,
            ),
            timeout=cfg.per_prompt_timeout_s,
        )
    except asyncio.TimeoutError:
        return _timeout_result(prompt, run_dir, started, started_utc_iso)
    except Exception as e:
        return _error_result(prompt, run_dir, started, started_utc_iso, repr(e))

    elapsed = round(time.monotonic() - started, 1)

    record = _find_metrics_record_for_topic(
        prompt.text, metrics_path=metrics_path, after_timestamp=pre_run_timestamp,
    )

    md = ""
    if isinstance(out_path, Path) and out_path.exists():
        try:
            md = out_path.read_text(encoding="utf-8")
        except OSError:
            md = ""

    rounds_record = record.get("rounds", {}) if record else {}
    rounds_list: list[list[dict]] = []  # we don't have full per-agent outputs in metrics
    flags: list[str] = []  # rule-based detectors need agent-level outputs; deferred
    # If we have a rounds dict from metrics, we can at least populate FM-3.3 / FM-3.1 hooks
    # from aggregate signals. Full detector coverage requires hooking inside run_swarm.

    quality = _build_quality_dict(record)
    coverage_hits = _coverage_count(md, prompt.expected_coverage)

    return PromptResult(
        prompt_id=prompt.id,
        run_id=run_dir.name,
        swarm_run_id=record.get("run_id", "") if record else "",
        prompt_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
        output_md_path=str(out_path) if isinstance(out_path, Path) else "",
        started_utc=started_utc_iso,
        elapsed_s=elapsed,
        cost_usd=float(record.get("total_cost_usd", 0) or 0) if record else 0.0,
        input_tokens=int(record.get("total_input_tokens", 0) or 0) if record else 0,
        output_tokens=int(record.get("total_output_tokens", 0) or 0) if record else 0,
        consensus_reached=bool(record.get("consensus_reached", False)) if record else False,
        rounds_to_consensus=int(record.get("rounds_to_consensus", 0) or 0) if record else 0,
        quality=quality,
        coverage_hits=coverage_hits,
        coverage_total=len(prompt.expected_coverage),
        timeout=False,
        error=None,
        mast_flags=flags,
    )


def _timeout_result(prompt: BenchPrompt, run_dir: Path, started: float, started_utc_iso: str) -> PromptResult:
    return PromptResult(
        prompt_id=prompt.id,
        run_id=run_dir.name,
        swarm_run_id="",
        prompt_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
        output_md_path="",
        started_utc=started_utc_iso,
        elapsed_s=round(time.monotonic() - started, 1),
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        consensus_reached=False,
        rounds_to_consensus=0,
        quality={"coverage": 0, "accuracy": 0, "clarity": 0, "depth": 0, "avg": 0.0},
        coverage_hits=0,
        coverage_total=len(prompt.expected_coverage),
        timeout=True,
        error="timeout",
        mast_flags=["FM-3.1"],
    )


def _error_result(
    prompt: BenchPrompt, run_dir: Path, started: float, started_utc_iso: str, msg: str,
) -> PromptResult:
    return PromptResult(
        prompt_id=prompt.id,
        run_id=run_dir.name,
        swarm_run_id="",
        prompt_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
        output_md_path="",
        started_utc=started_utc_iso,
        elapsed_s=round(time.monotonic() - started, 1),
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        consensus_reached=False,
        rounds_to_consensus=0,
        quality={"coverage": 0, "accuracy": 0, "clarity": 0, "depth": 0, "avg": 0.0},
        coverage_hits=0,
        coverage_total=len(prompt.expected_coverage),
        timeout=False,
        error=msg,
        mast_flags=[],
    )


# ---------------------------------------------------------------------------
# Aggregation + summary
# ---------------------------------------------------------------------------


def _read_results(run_dir: Path) -> list[dict]:
    path = run_dir / "results.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def aggregate(run_dir: Path, slate: BenchSlate, cfg: BenchConfig) -> dict:
    """Aggregate per-prompt rows into a single summary.json. Returns the dict."""
    rows = _read_results(run_dir)

    # Per-dim aggregates
    by_dim: dict[str, dict[str, float]] = {}
    for dim in ("avg", "coverage", "accuracy", "clarity", "depth"):
        vals = [r["quality"].get(dim, 0) for r in rows if r.get("quality")]
        if not vals:
            by_dim[dim] = {"mean": 0.0, "stddev": 0.0, "ci95": [0.0, 0.0]}
            continue
        mean = float(statistics.mean(vals))
        sd = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
        ci_lo, ci_hi = bootstrap_ci(vals, ci=95.0, n_resamples=2000, seed=cfg.seed)
        by_dim[dim] = {"mean": round(mean, 3), "stddev": round(sd, 3),
                       "ci95": [round(ci_lo, 3), round(ci_hi, 3)]}

    # By tier
    by_tier: dict[str, dict] = {}
    for tier in ("easy", "medium", "hard"):
        tier_rows = [r for r in rows if slate.by_id(r["prompt_id"]).tier == tier
                     if r.get("prompt_id") and any(p.id == r["prompt_id"] for p in slate.prompts)]
        if tier_rows:
            avgs = [r["quality"].get("avg", 0) for r in tier_rows]
            costs = [r.get("cost_usd", 0) for r in tier_rows]
            by_tier[tier] = {
                "n": len(tier_rows),
                "avg_quality": round(statistics.mean(avgs), 3) if avgs else 0,
                "avg_cost": round(statistics.mean(costs), 4) if costs else 0,
            }
        else:
            by_tier[tier] = {"n": 0, "avg_quality": 0, "avg_cost": 0}

    # MAST flag summary
    flag_counts: dict[str, int] = {}
    for r in rows:
        for fm in r.get("mast_flags", []) or []:
            flag_counts[fm] = flag_counts.get(fm, 0) + 1

    summary = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_utc": cfg.started_utc,
        "ended_utc": _utc(),
        "code": {
            "git_sha": cfg.git_sha,
            "blitz_swarm_version": "0.2.0-alpha",
            "python_version": sys.version.split()[0],
        },
        "config": {
            "slate_path": str(cfg.slate_path),
            "max_rounds": cfg.max_rounds,
            "use_redis": cfg.use_redis,
            "seed": cfg.seed,
            "total_budget_usd": cfg.total_budget_usd,
            "per_prompt_timeout_s": cfg.per_prompt_timeout_s,
        },
        "slate": {
            "slate_id": slate.slate_id,
            "sha256": slate.sha256,
            "prompt_count": len(slate),
            "prompts_run": len([r for r in rows if not r.get("error")]),
            "prompts_skipped": 0,
            "prompts_timed_out": len([r for r in rows if r.get("timeout")]),
            "prompts_errored": len([r for r in rows if r.get("error") and not r.get("timeout")]),
        },
        "aggregate_quality": by_dim,
        "by_tier": by_tier,
        "consensus": {
            "rate": round(
                sum(1 for r in rows if r.get("consensus_reached")) / max(len(rows), 1), 3,
            ),
            "avg_rounds_to_consensus": round(
                statistics.mean([r.get("rounds_to_consensus", 0) for r in rows]) if rows else 0,
                3,
            ),
        },
        "cost": {
            "total_usd": round(sum(r.get("cost_usd", 0) for r in rows), 4),
            "input_tokens_total": sum(r.get("input_tokens", 0) for r in rows),
            "output_tokens_total": sum(r.get("output_tokens", 0) for r in rows),
        },
        "mast_flags_summary": flag_counts,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def write_stats_md(run_dir: Path, summary: dict) -> Path:
    """Write a one-page stats.md report."""
    lines = [
        f"# Bench run {run_dir.name}",
        "",
        f"Started: {summary['started_utc']}  →  Ended: {summary['ended_utc']}",
        f"Slate: {summary['slate']['slate_id']} (sha256={summary['slate']['sha256'][:12]}...)",
        f"Prompts: ran {summary['slate']['prompts_run']} / {summary['slate']['prompt_count']}, "
        f"timed out {summary['slate']['prompts_timed_out']}, "
        f"errored {summary['slate']['prompts_errored']}",
        "",
        "## Aggregate quality (0-10)",
        "",
        "| Dim | Mean | SD | CI95 |",
        "|---|---|---|---|",
    ]
    for dim, stats_dict in summary["aggregate_quality"].items():
        ci = stats_dict["ci95"]
        lines.append(f"| {dim} | {stats_dict['mean']} | {stats_dict['stddev']} | "
                     f"[{ci[0]}, {ci[1]}] |")
    lines.append("")
    lines.append("## By tier")
    lines.append("")
    lines.append("| Tier | N | Avg quality | Avg cost ($) |")
    lines.append("|---|---|---|---|")
    for tier, st in summary["by_tier"].items():
        lines.append(f"| {tier} | {st['n']} | {st['avg_quality']} | {st['avg_cost']} |")
    lines.append("")
    lines.append("## MAST flags")
    lines.append("")
    if summary["mast_flags_summary"]:
        lines.append("| FM | Count |")
        lines.append("|---|---|")
        for fm in sorted(summary["mast_flags_summary"]):
            lines.append(f"| {fm} | {summary['mast_flags_summary'][fm]} |")
    else:
        lines.append("_No MAST flags raised._")
    lines.append("")
    lines.append("## Cost")
    lines.append("")
    cost = summary["cost"]
    lines.append(f"- Total: ${cost['total_usd']}")
    lines.append(f"- Input tokens:  {cost['input_tokens_total']:,}")
    lines.append(f"- Output tokens: {cost['output_tokens_total']:,}")
    lines.append("")
    path = run_dir / "stats.md"
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


async def run_bench(
    cfg: BenchConfig,
    *,
    swarm_fn: SwarmFn | None = None,
    metrics_path: Path = METRICS_PATH,
) -> Path:
    """Top-level: load slate, run prompts, aggregate. Returns the run dir."""
    if swarm_fn is None:
        # Lazy import: only require orchestrator when actually running.
        from orchestrator import run_swarm
        swarm_fn = run_swarm

    cfg.git_sha = cfg.git_sha or _git_sha()
    cfg.started_utc = cfg.started_utc or _utc()
    slate = load_slate(cfg.slate_path)
    pending_all = cfg.filter_slate(slate)

    run_id = _run_id(cfg, slate)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resume detection
    done_ids = _resumable_ids(run_dir)
    pending = [p for p in pending_all if p.id not in done_ids]

    # Snapshot config + slate hash for reproducibility
    (run_dir / "config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
    }, indent=2))

    sem = asyncio.Semaphore(max(cfg.parallel_prompts, 1))
    budget_remaining = [cfg.total_budget_usd]
    budget_lock = asyncio.Lock()
    results_path = run_dir / "results.jsonl"

    async def _bounded(p: BenchPrompt) -> PromptResult:
        async with sem:
            async with budget_lock:
                if budget_remaining[0] < p.budget_usd:
                    return _error_result(
                        p, run_dir, time.monotonic(), _utc(),
                        "budget_exhausted",
                    )
                budget_remaining[0] -= p.budget_usd
            result = await run_one_prompt(
                p, cfg, run_dir,
                swarm_fn=swarm_fn,
                metrics_path=metrics_path,
            )
            async with budget_lock:
                # Refund unused budget
                budget_remaining[0] += max(p.budget_usd - result.cost_usd, 0)
            return result

    if pending:
        for coro in asyncio.as_completed([_bounded(p) for p in pending]):
            row = await coro
            _append_jsonl(results_path, asdict(row))
            print(f"  [{row.prompt_id}] q={row.quality.get('avg', '?')} "
                  f"cov={row.coverage_hits}/{row.coverage_total} "
                  f"${row.cost_usd:.3f} {row.elapsed_s:.0f}s "
                  f"{'TIMEOUT' if row.timeout else ('ERR' if row.error else 'ok')}")

    summary = aggregate(run_dir, slate, cfg)
    write_stats_md(run_dir, summary)
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Blitz-Swarm bench runner")
    ap.add_argument("--slate", default=str(BENCH_ROOT / "slate_v1.toml"))
    ap.add_argument("--filter-id", action="append", default=[], dest="filter_ids")
    ap.add_argument("--filter-tier", action="append", default=[], dest="filter_tiers",
                    choices=["easy", "medium", "hard"])
    ap.add_argument("--filter-domain", action="append", default=[], dest="filter_domains")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--per-prompt-timeout", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-redis", action="store_true")
    args = ap.parse_args()

    cfg = BenchConfig(
        slate_path=Path(args.slate),
        slate_filter_ids=tuple(args.filter_ids),
        slate_filter_tiers=tuple(args.filter_tiers),
        slate_filter_domains=tuple(args.filter_domains),
        parallel_prompts=args.parallel,
        per_prompt_timeout_s=args.per_prompt_timeout,
        total_budget_usd=args.budget,
        max_rounds=args.max_rounds,
        use_redis=args.use_redis,
        seed=args.seed,
    )
    run_dir = asyncio.run(run_bench(cfg))
    print(f"\nrun complete: {run_dir}")


if __name__ == "__main__":
    main()
