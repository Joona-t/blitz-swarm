#!/usr/bin/env python3
"""Resumable, budget-aware driver for the 10M-token blitz-swarm upgrade job.

The upgrade job is a five-phase pipeline that takes the current blitz-swarm,
measures it, researches frontier techniques, implements candidates, verifies
them against the bench with statistical significance, and gates the winners
into a commit on the CURRENT branch.

    baseline  -> research -> implement -> verify -> gate

Each phase has a SOFT token sub-budget; the whole job has a HARD ceiling
(default 10M tokens). The driver is resumable at the granularity of a "cell"
(one unit of work — e.g. ``(topic, seed)`` for baseline, or
``(technique, topic, seed)`` for verify). Every completed cell is checkpointed
into a JSON manifest. On ``--resume`` already-done cells are skipped, so an
interrupted run picks up exactly where it left off and never repeats work.

If the hard ceiling is hit mid-phase, the active phase is marked ``paused``,
the manifest is saved, and the process exits 0 after printing the resume
command. Re-running with ``--resume`` continues from the paused phase.

DRY-RUN CONTRACT (critical for testing):
    ``--dry-run`` runs the FULL state machine end-to-end without spawning a
    single subprocess or LLM call. Every orchestrator/research/mythos/bench
    invocation funnels through one helper — ``_invoke(kind, **kw)`` — which in
    dry-run returns a deterministic fake result plus a simulated token count
    and DOES NOT touch ``subprocess``. This makes the whole control flow
    (phase ordering, checkpointing, resume, budget pausing) testable for free.

Pure stdlib only: argparse, hashlib, json, pathlib, subprocess, tomllib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Constants — phase order, soft sub-budgets, hard ceiling
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_DIR = JOBS_DIR / "state"
DEFAULT_STATE_FILE = DEFAULT_STATE_DIR / "upgrade_state.json"

# Ensure the repo root is importable when this module is run as a SCRIPT
# (``python3 jobs/blitz_upgrade.py``), where sys.path[0] is ``jobs/`` rather
# than the repo root. Without this, sibling packages — ``bench`` (stats,
# runner), ``orchestrator``, ``mythos`` — fail to import: the dry-run would
# silently fall back to a delta-only decision (missing the cohens_d/p/CI stats)
# and live mode could not dispatch at all. Tests already put the root on the
# path via conftest; this makes the standalone CLI behave identically.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# repo-root metrics.jsonl — the swarm's own per-run metrics (see metrics.py).
METRICS_PATH = REPO_ROOT / "metrics.jsonl"

# Phases run strictly in this order. Index in the list == execution rank.
PHASES: tuple[str, ...] = ("baseline", "research", "implement", "verify", "gate")

# Soft per-phase token sub-budgets. These are advisory checkpoints used to
# log/report progress; only HARD_CEILING actually halts the job.
SOFT_TOKEN_BUDGETS: dict[str, int] = {
    "baseline": 400_000,
    "research": 2_000_000,
    "implement": 3_000_000,
    "verify": 4_000_000,
    "gate": 600_000,
}

HARD_CEILING: int = 10_000_000

# Status vocabulary for a phase entry in the manifest.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_PAUSED = "paused"

# Per-cell simulated token cost used in dry-run (deterministic, cheap).
_DRY_RUN_TOKENS_PER_CELL = 1_000

# Default research topics for the upgrade (used when the slate file is absent
# or for the research phase, which is technique-discovery rather than slate
# scoring). Kept small + deterministic so dry-run is fast and reproducible.
DEFAULT_RESEARCH_TOPICS: tuple[str, ...] = (
    "latest parallel multi-agent swarm orchestration techniques 2025-2026",
    "self-consistency and majority-vote ensembling for LLM agent swarms",
    "MAST failure modes and rule-based detectors for multi-agent systems",
    "verifier-gated hierarchical planner-executor agent architectures",
    "GEPA / prompt-evolution optimization against a held-out bench",
    "cost-aware model routing (haiku/sonnet/opus) for agent swarms",
)

# Fallback baseline topics if no slate file is present (dry-run friendly).
_FALLBACK_BASELINE_TOPICS: tuple[str, ...] = (
    "Explain SQLite WAL mode internals and concurrency guarantees.",
    "Compare RAG vs fine-tuning for domain question answering.",
    "Design a parallel agent memory system with provenance.",
)


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------


class BudgetTracker:
    """Tracks cumulative token spend against the hard ceiling.

    Two modes:
      * real      — best-effort reads cumulative tokens from metrics.jsonl
                    (input+output across all runs) and adds the driver's own
                    accounted spend (cells that were charged this process).
      * dry-run   — ignores metrics.jsonl entirely and simulates spend by
                    accumulating ``charge()`` calls only. Fully deterministic.

    The driver charges tokens through :meth:`charge` for every unit of work so
    that resume + pause behaviour is identical in both modes.

    B2 — resume double-count fix. In real mode the metrics.jsonl floor is a
    one-time snapshot of historical spend (input+output across all prior runs).
    Without care it double-counts on resume: the very tokens that were in
    metrics.jsonl when the FIRST process ran were *also* charged into
    ``tokens_spent`` and restored as ``initial_spent`` on the SECOND process —
    and if metrics.jsonl has since GROWN, re-reading it would add even more on
    top. The fix: snapshot the floor ONCE (in the first process), persist it in
    the manifest as ``metrics_floor_at_start``, and on every resume reuse that
    persisted value via ``metrics_floor=`` instead of re-reading the now-larger
    file. ``charge`` tracks this-process incremental spend on top of the
    restored ``initial_spent`` so a phase re-reading the file never inflates the
    ceiling.
    """

    def __init__(
        self,
        hard_ceiling: int = HARD_CEILING,
        *,
        dry_run: bool = False,
        metrics_path: Path = METRICS_PATH,
        initial_spent: int = 0,
        metrics_floor: int | None = None,
    ) -> None:
        self.hard_ceiling = int(hard_ceiling)
        self.dry_run = bool(dry_run)
        self.metrics_path = Path(metrics_path)
        # Tokens charged by this process (the resumable, authoritative counter
        # persisted in the manifest as ``tokens_spent``).
        self._charged = int(initial_spent)
        # Additive floor of pre-existing historical spend (real mode only).
        #   * dry-run            -> always 0 (metrics.jsonl ignored entirely).
        #   * real, fresh run    -> snapshot metrics.jsonl ONCE here.
        #   * real, resume       -> reuse the persisted snapshot the caller
        #                           passes in via ``metrics_floor`` so a grown
        #                           metrics.jsonl cannot re-inflate the ceiling
        #                           (B2: no double-count across processes).
        if self.dry_run:
            self._metrics_floor = 0
        elif metrics_floor is not None:
            self._metrics_floor = int(metrics_floor)
        else:
            self._metrics_floor = self._read_metrics_tokens()

    # -- public API --------------------------------------------------------

    def charge(self, tokens: int) -> None:
        """Account ``tokens`` of spend against the ceiling."""
        self._charged += max(int(tokens), 0)

    def spent(self) -> int:
        """Total tokens spent (charged this process + metrics floor)."""
        return self._charged + self._metrics_floor

    def charged(self) -> int:
        """Tokens charged by this process only (persisted to the manifest)."""
        return self._charged

    def metrics_floor(self) -> int:
        """The historical-spend floor snapshot (persisted once as
        ``metrics_floor_at_start`` so resume reuses it instead of re-reading the
        grown metrics.jsonl — see B2)."""
        return self._metrics_floor

    def remaining(self) -> int:
        """Tokens left before the hard ceiling (never negative)."""
        return max(self.hard_ceiling - self.spent(), 0)

    def exhausted(self) -> bool:
        """True once spend has reached or exceeded the hard ceiling."""
        return self.spent() >= self.hard_ceiling

    def would_exceed(self, tokens: int) -> bool:
        """True if charging ``tokens`` more would breach the ceiling."""
        return (self.spent() + max(int(tokens), 0)) > self.hard_ceiling

    # -- internals ---------------------------------------------------------

    def _read_metrics_tokens(self) -> int:
        """Best-effort sum of input+output tokens across all metrics.jsonl runs.

        Tolerant of a missing or partially-written file: any unparsable line is
        skipped. Returns 0 if the file is absent.
        """
        if not self.metrics_path.exists():
            return 0
        total = 0
        try:
            with self.metrics_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total += int(rec.get("total_input_tokens", 0) or 0)
                    total += int(rec.get("total_output_tokens", 0) or 0)
        except OSError:
            return total
        return total


# ---------------------------------------------------------------------------
# Job state manifest
# ---------------------------------------------------------------------------


class JobState:
    """Loads / saves / checkpoints the phase manifest JSON.

    The manifest is the single source of truth for resumability. Every mutating
    method is idempotent and persists immediately, so calling them repeatedly
    (e.g. re-marking a cell already done) is safe and a no-op.

    Schema (``upgrade_state.json``)::

        {
          "started_at": <float|str>,        # passed in, NOT time.time at import
          "branch": "<git branch>",
          "hard_ceiling": 10000000,
          "tokens_spent": <int>,            # authoritative, process-charged
          "phases": {
            "<name>": {
              "status": "pending|running|done|paused",
              "tokens": <int>,              # tokens charged within this phase
              "artifacts": [<str>, ...],    # output paths produced
              "cells_done": [<str>, ...]    # cell ids completed (for resume)
            }, ...
          }
        }
    """

    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = Path(path)
        self.data = data

    # -- construction ------------------------------------------------------

    @classmethod
    def load_or_init(
        cls,
        path: Path,
        *,
        started_at: float | str,
        branch: str,
        hard_ceiling: int = HARD_CEILING,
    ) -> "JobState":
        """Load an existing manifest, or initialise a fresh one on disk.

        ``started_at`` is supplied by the caller (never ``time.time()`` at
        import) so the manifest is deterministic and test-injectable.
        """
        path = Path(path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            if data.get("phases"):
                state = cls(path, data)
                state._ensure_phases()  # tolerate older/partial manifests
                return state
        # Fresh manifest.
        data = {
            "started_at": started_at,
            "branch": branch,
            "hard_ceiling": int(hard_ceiling),
            "tokens_spent": 0,
            "phases": {
                name: {
                    "status": STATUS_PENDING,
                    "tokens": 0,
                    "artifacts": [],
                    "cells_done": [],
                }
                for name in PHASES
            },
        }
        state = cls(path, data)
        state.save()
        return state

    def _ensure_phases(self) -> None:
        """Backfill any missing phase entries / keys (forward-compat)."""
        self.data.setdefault("phases", {})
        for name in PHASES:
            entry = self.data["phases"].setdefault(name, {})
            entry.setdefault("status", STATUS_PENDING)
            entry.setdefault("tokens", 0)
            entry.setdefault("artifacts", [])
            entry.setdefault("cells_done", [])
        self.data.setdefault("tokens_spent", 0)
        self.data.setdefault("hard_ceiling", HARD_CEILING)

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        """Atomically persist the manifest to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def checkpoint(self) -> None:
        """Alias for :meth:`save` — explicit checkpoint after a unit of work."""
        self.save()

    # -- phase accessors ---------------------------------------------------

    def phase(self, name: str) -> dict[str, Any]:
        return self.data["phases"][name]

    def status(self, name: str) -> str:
        return self.phase(name)["status"]

    def is_done(self, name: str) -> bool:
        return self.status(name) == STATUS_DONE

    def cells_done(self, name: str) -> set[str]:
        return set(self.phase(name).get("cells_done", []))

    def cell_results(self, name: str) -> dict[str, dict]:
        """Persisted per-cell result payloads (survive --resume), for rebuilding
        phase-terminal computations like verify's significance pass (B6)."""
        return dict(self.phase(name).get("cell_results", {}))

    def set_status(self, name: str, status: str) -> None:
        self.phase(name)["status"] = status
        self.save()

    # -- mutation (idempotent) --------------------------------------------

    def mark_cell_done(self, name: str, cell_id: str, *, tokens: int = 0,
                       artifact: str | None = None, result: dict | None = None) -> bool:
        """Record ``cell_id`` as completed for ``name``.

        Idempotent: if the cell is already recorded, nothing is charged and the
        method returns False. Returns True when the cell was newly recorded
        (so the caller can charge the budget exactly once).
        """
        entry = self.phase(name)
        done = entry.setdefault("cells_done", [])
        if cell_id in done:
            return False
        done.append(cell_id)
        entry["tokens"] = int(entry.get("tokens", 0)) + max(int(tokens), 0)
        self.data["tokens_spent"] = int(self.data.get("tokens_spent", 0)) + max(int(tokens), 0)
        if artifact:
            arts = entry.setdefault("artifacts", [])
            if artifact not in arts:
                arts.append(artifact)
        if result is not None:
            # Persist a compact per-cell result so phase-terminal computations
            # (verify significance) can be rebuilt on --resume when the cell
            # itself is skipped (B6).
            entry.setdefault("cell_results", {})[cell_id] = result
        self.save()
        return True

    def add_artifact(self, name: str, artifact: str) -> None:
        arts = self.phase(name).setdefault("artifacts", [])
        if artifact and artifact not in arts:
            arts.append(artifact)
            self.save()

    def sync_tokens_from(self, tracker: "BudgetTracker") -> None:
        """Persist the tracker's process-charged tokens into the manifest."""
        self.data["tokens_spent"] = tracker.charged()
        self.save()


# ---------------------------------------------------------------------------
# The single invocation chokepoint
# ---------------------------------------------------------------------------


def _invoke(kind: str, *, dry_run: bool, tracker: BudgetTracker,
            **kw: Any) -> dict[str, Any]:
    """The ONE place any external swarm/LLM/bench work is dispatched.

    Every phase routes its real work through here. This is what makes the whole
    driver testable: in dry-run we return a deterministic fake result and a
    simulated token count and NEVER touch ``subprocess`` or any LLM.

    Parameters
    ----------
    kind:
        One of ``"consensus"``, ``"self_consistency"``, ``"research"``,
        ``"mythos"``, ``"bench_ablation"``, ``"git"``, ``"gh_pr"``.
    dry_run:
        When True, simulate. Returns a deterministic stub.
    tracker:
        The :class:`BudgetTracker`; ``_invoke`` does NOT charge it (the caller
        charges per cell so resume accounting stays in one place) but it is
        threaded through for symmetry / future use.
    kw:
        Arbitrary keyword payload describing the work (topic, seed, technique,
        flags, ...). Echoed back in the result for traceability.

    Returns
    -------
    dict with at least::

        {"ok": bool, "kind": str, "tokens": int, "artifact": str|None,
         "result": <kind-specific payload>}
    """
    if dry_run:
        return _invoke_dry(kind, **kw)
    return _invoke_live(kind, tracker=tracker, **kw)


def _stable_hash(key: Any) -> int:
    """Process-stable hash of an arbitrary key (B4).

    The builtin ``hash()`` is salted by ``PYTHONHASHSEED`` for str/bytes/tuples,
    so two subprocesses with different seeds produce DIFFERENT dry-run token and
    quality numbers — which breaks the dry-run determinism contract (identical
    manifests across processes). We use a SHA-256 of ``repr(key)`` truncated to
    32 bits instead: identical across processes, OSes and Python builds.
    """
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _invoke_dry(kind: str, **kw: Any) -> dict[str, Any]:
    """Deterministic simulation — no subprocess, no LLM, no filesystem reads.

    The fake token count and quality numbers are derived from a STABLE
    (PYTHONHASHSEED-independent) hash of the payload so repeated dry-runs — even
    in separate processes — produce byte-identical state manifests (B4).
    """
    if kind == "decide":
        # Pure, zero-cost gate decision cell (mirrors _invoke_live).
        return {"ok": True, "kind": kind, "tokens": 0, "artifact": None,
                "result": {"decide": True, "dry_run": True}}
    if kind == "regression":
        # Pure, zero-cost regression-gate cell (mirrors _invoke_live). Real test
        # execution happens in the phase's on_result callback, not here, so the
        # dry-run path never shells out (B5).
        return {"ok": True, "kind": kind, "tokens": 0, "artifact": None,
                "result": {"regression": True, "dry_run": True}}
    seed = int(kw.get("seed", 0))
    topic = str(kw.get("topic", kw.get("task", "")))
    technique = str(kw.get("technique", ""))
    arm = str(kw.get("arm", ""))
    # Stable pseudo-token count: base per-cell cost nudged by a payload hash so
    # different cells differ but are reproducible across processes (B4).
    h = _stable_hash((kind, topic, technique, arm, seed)) % 500
    tokens = _DRY_RUN_TOKENS_PER_CELL + h
    fake_path = f"<dry-run:{kind}:{technique or arm or 'x'}:{str(seed)}>"
    # A deterministic fake quality score so the gate decision rule exercises
    # both keep / drop branches across techniques. The base is per-(technique,
    # arm); a small seed-dependent jitter makes the K seeds DIFFER so the
    # significance/stats path (paired t-test, Cohen's d, bootstrap CI) is
    # actually exercised rather than degenerating to zero variance (G5).
    base = 6.0 + (_stable_hash((technique, arm, kind)) % 30) / 10.0
    jitter = (_stable_hash(("seed-jitter", technique, arm, kind, seed)) % 100) / 100.0
    quality = base + jitter
    return {
        "ok": True,
        "kind": kind,
        "tokens": tokens,
        "artifact": fake_path,
        "result": {
            "topic": topic,
            "technique": technique,
            "arm": arm,
            "seed": seed,
            "avg_quality": round(quality, 3),
            "status": "passed" if kind == "mythos" else "ok",
            "dry_run": True,
        },
    }


def _invoke_live(kind: str, *, tracker: BudgetTracker, **kw: Any) -> dict[str, Any]:
    """Real-mode dispatch. Best-effort wiring against the recon API map.

    Each branch is marked with ``# LIVE:`` so the real integration points are
    unmistakable. Tokens are read back from the swarm's metrics.jsonl after the
    run (the CLI envelope is the source of truth); when a backend doesn't report
    usage the token count falls back to 0 and the phase still checkpoints.
    """
    import asyncio  # local import: dry-run never needs asyncio

    topic = kw.get("topic", kw.get("task", ""))
    seed = kw.get("seed", 0)

    if kind == "decide":
        # The gate decision cell does no external work — the keep/drop logic is
        # pure and runs in the phase's on_result callback. This branch just
        # gives the budget loop a uniform, zero-cost invocation.
        return {"ok": True, "kind": kind, "tokens": 0, "artifact": None,
                "result": {"decide": True}}

    if kind == "regression":
        # The regression-gate cell's real work (running the test suite) happens
        # in the phase's on_result callback via _run_regression_suite so the
        # subprocess use is in ONE place and dry-run never reaches it. This is a
        # uniform, zero-cost invocation for the budget loop.
        return {"ok": True, "kind": kind, "tokens": 0, "artifact": None,
                "result": {"regression": True}}

    if kind in ("consensus", "self_consistency"):
        # LIVE: run the current consensus swarm. run_swarm is async and returns
        # a Path to the saved markdown report. Cost/tokens are NOT on the return
        # value — they are appended to repo-root metrics.jsonl; we read the last
        # matching record back to attribute spend.
        from orchestrator import run_swarm  # noqa: PLC0415

        pre_ts = _now_epoch()
        out_path = asyncio.run(
            run_swarm(
                topic,
                max_rounds=int(kw.get("max_rounds", 4)),
                use_redis=bool(kw.get("use_redis", False)),
                quality_profile=kw.get("quality_profile"),
                backend_id=kw.get("backend_id"),
            )
        )
        rec = _read_metrics_record_for_topic(topic, after_ts=pre_ts)
        tokens = _tokens_of(rec)
        return {
            "ok": True,
            "kind": kind,
            "tokens": tokens,
            "artifact": str(out_path) if out_path else None,
            "result": {
                "topic": topic,
                "seed": seed,
                "avg_quality": float(rec.get("avg_quality", 0) or 0) if rec else 0.0,
                "consensus_reached": bool(rec.get("consensus_reached", False)) if rec else False,
                "cost_usd": float(rec.get("total_cost_usd", 0) or 0) if rec else 0.0,
            },
        }

    if kind == "research":
        # LIVE: research-swarm lives OUTSIDE this repo at
        # "/Users/darkfire/Claude x LoveSpark/research-swarm/orchestrator.py"
        # (see recon). It is a separate program with no --mode flag. We shell
        # out to it and grep the "Done. Output at:" line for the report path.
        research_orch = Path(
            "/Users/darkfire/Claude x LoveSpark/research-swarm/orchestrator.py"
        )
        if not research_orch.exists():
            return {"ok": False, "kind": kind, "tokens": 0, "artifact": None,
                    "result": {"error": "research-swarm orchestrator not found",
                               "topic": topic}}
        pre_ts = _now_epoch()
        proc = subprocess.run(  # noqa: PLW1510  (we inspect returncode below)
            [sys.executable, str(research_orch), str(topic)],
            cwd=str(research_orch.parent),
            capture_output=True,
            text=True,
        )
        out_path = _grep_output_path(proc.stdout)
        # G7: research-swarm tracks its own metrics.jsonl (keyed by topic) next
        # to its orchestrator. Best-effort attribute that run's tokens so the
        # governor isn't blind to research spend. If the file/record is missing
        # we charge 0 but emit a clear WARNING (never silently blind).
        research_metrics = research_orch.parent / "metrics.jsonl"
        rec = _read_metrics_record_for_topic(
            str(topic), after_ts=pre_ts, metrics_path=research_metrics)
        tokens = _tokens_of(rec)
        if tokens == 0:
            _warn(f"research token attribution unavailable for topic "
                  f"{topic!r} (metrics={research_metrics}); charging 0 — "
                  f"governor undercounts research spend")
        return {
            "ok": proc.returncode == 0,
            "kind": kind,
            "tokens": tokens,
            "artifact": out_path,
            "result": {"topic": topic, "returncode": proc.returncode,
                       "cost_usd": float(rec.get("total_cost_usd", 0) or 0) if rec else 0.0},
        }

    if kind == "mythos":
        # LIVE: mythos hierarchical planner/executor/verifier. Call run_mythos
        # directly (the CLI does not surface the MythosResult). Honour the
        # per-candidate cost ceiling so a runaway candidate can't drain budget.
        from mythos import load_mythos_config, run_mythos  # noqa: PLC0415

        cfg = load_mythos_config()
        if kw.get("cost_ceiling") is not None:
            cfg.cost_ceiling_usd = float(kw["cost_ceiling"])
        if kw.get("max_replans") is not None:
            cfg.max_replans = int(kw["max_replans"])
        cfg.use_redis = bool(kw.get("use_redis", False))
        result = asyncio.run(run_mythos(str(kw.get("task", topic)), cfg))
        summary = result.summary_dict()
        # G7: mythos persists tokens in <run_dir>/metrics.json (and cost in $ in
        # run.json). Best-effort attribute the token spend; if metrics.json has
        # no token counts, WARN and charge 0 rather than silently undercount.
        tokens = _read_mythos_tokens(result.run_dir)
        if tokens == 0:
            _warn(f"mythos token attribution unavailable (run_dir="
                  f"{result.run_dir}); charging 0 — governor undercounts "
                  f"implement spend (cost was ${summary.get('cost_usd', 0)})")
        return {
            "ok": result.status == "passed",
            "kind": kind,
            "tokens": tokens,
            "artifact": str(result.run_dir),
            "result": summary,
        }

    if kind == "bench_ablation":
        # LIVE: run the bench harness on the upgrade slate for one ablation arm
        # (mechanism on vs off) at one seed. run_bench is async and returns the
        # run dir; the per-prompt quality/cost live in results.jsonl + summary.
        from bench.runner import BenchConfig, run_bench  # noqa: PLC0415

        cfg = BenchConfig(
            slate_path=Path(kw["slate"]),
            parallel_prompts=int(kw.get("parallel", 2)),
            total_budget_usd=float(kw.get("budget", 8.0)),
            max_rounds=int(kw.get("max_rounds", 4)),
            use_redis=bool(kw.get("use_redis", False)),
            seed=int(seed),
            backend_id=kw.get("backend_id"),  # claude-only honored in verify too
        )
        run_dir = asyncio.run(run_bench(cfg))
        summary = _read_bench_summary(run_dir)
        return {
            "ok": True,
            "kind": kind,
            "tokens": _bench_tokens(summary),
            "artifact": str(run_dir),
            "result": {
                "arm": kw.get("arm", ""),
                "seed": seed,
                "avg_quality": _bench_avg_quality(summary),
                "summary_path": str(run_dir / "summary.json"),
            },
        }

    if kind == "git":
        # LIVE: stage + commit on the CURRENT branch. NEVER checkout/commit to
        # main — the caller guards the branch; here we only run the given argv.
        argv = list(kw.get("argv", []))
        proc = subprocess.run(  # noqa: PLW1510
            ["git", "-C", str(REPO_ROOT), *argv],
            capture_output=True, text=True,
        )
        return {"ok": proc.returncode == 0, "kind": kind, "tokens": 0,
                "artifact": None,
                "result": {"argv": argv, "returncode": proc.returncode,
                           "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}}

    if kind == "gh_pr":
        # LIVE: best-effort `gh pr create`. Failure is non-fatal (the commit on
        # the current branch is the durable artifact).
        argv = list(kw.get("argv", ["pr", "create", "--fill"]))
        proc = subprocess.run(  # noqa: PLW1510
            ["gh", *argv], cwd=str(REPO_ROOT),
            capture_output=True, text=True,
        )
        return {"ok": proc.returncode == 0, "kind": kind, "tokens": 0,
                "artifact": proc.stdout.strip() or None,
                "result": {"returncode": proc.returncode,
                           "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}}

    raise ValueError(f"_invoke: unknown kind {kind!r}")


# ---------------------------------------------------------------------------
# Live-mode helpers (only exercised in real runs; dry-run never reaches them)
# ---------------------------------------------------------------------------


def _now_epoch() -> float:
    import time  # local import keeps module import side-effect free
    return time.time()


def _warn(msg: str) -> None:
    """Emit a clear WARNING to stderr (token-attribution gaps, etc. — G7)."""
    print(f"WARNING: {msg}", file=sys.stderr)


def _read_mythos_tokens(run_dir: Path | str) -> int:
    """Best-effort input+output tokens for a mythos run (G7).

    Mythos persists ``metrics.json`` in its run dir with token counts; run.json
    only carries cost in dollars. Tolerant of missing files / keys (returns 0,
    the caller WARNs). Accepts several plausible key spellings.
    """
    run_dir = Path(run_dir)
    for fname in ("metrics.json", "run.json"):
        p = run_dir / fname
        if not p.exists():
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rec, dict):
            continue
        tot = 0
        for k in ("total_input_tokens", "input_tokens", "total_output_tokens",
                  "output_tokens"):
            tot += int(rec.get(k, 0) or 0)
        if tot:
            return tot
    return 0


def _read_metrics_record_for_topic(topic: str, *, after_ts: float | None = None,
                                   metrics_path: Path = METRICS_PATH) -> dict | None:
    """Last metrics.jsonl record whose topic matches, newer than ``after_ts``."""
    if not metrics_path.exists():
        return None
    matches: list[dict] = []
    with metrics_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("topic") != topic:
                continue
            if after_ts is not None and float(rec.get("timestamp", 0)) <= after_ts:
                continue
            matches.append(rec)
    if not matches:
        return None
    return max(matches, key=lambda r: float(r.get("timestamp", 0)))


def _tokens_of(rec: dict | None) -> int:
    if not rec:
        return 0
    return int(rec.get("total_input_tokens", 0) or 0) + int(rec.get("total_output_tokens", 0) or 0)


def _grep_output_path(stdout: str) -> str | None:
    """Pull the report path from a 'Done. Output at: <path>' stdout line."""
    for line in stdout.splitlines():
        line = line.strip()
        for marker in ("Done. Output at:", "OUTPUT SAVED:"):
            if line.startswith(marker):
                return line[len(marker):].strip()
    return None


def _read_bench_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _bench_tokens(summary: dict) -> int:
    cost = summary.get("cost", {}) if summary else {}
    return int(cost.get("input_tokens_total", 0) or 0) + int(cost.get("output_tokens_total", 0) or 0)


def _bench_avg_quality(summary: dict) -> float:
    """The PRIMARY fitness signal for a bench run (G1).

    Per the shared contract with runner.py, the functional composite is the
    authoritative quality measure:

        summary["functional_composite"] = {
            "mean": <float 0..1 over prompts>,
            "per_prompt": {prompt_id: composite},
        }

    The LLM-judge scores (``summary["aggregate_quality"]``) are now SECONDARY
    and used ONLY as a fallback when ``functional_composite`` is absent (e.g. an
    older summary written before runner.py emitted it). We never silently mix
    the two scales: if the composite exists we return it, otherwise we fall back
    to the judge average and the caller treats it as best-effort.
    """
    if summary:
        fc = summary.get("functional_composite")
        if isinstance(fc, dict) and fc.get("mean") is not None:
            return float(fc.get("mean", 0) or 0)
    # Fallback: LLM-judge avg (secondary signal).
    aq = summary.get("aggregate_quality", {}) if summary else {}
    avg = aq.get("avg", {}) if isinstance(aq, dict) else {}
    return float(avg.get("mean", 0) or 0)


# ---------------------------------------------------------------------------
# Slate / topic loading
# ---------------------------------------------------------------------------


def _load_slate_topics(slate_path: Path) -> list[str]:
    """Return the prompt texts from a bench slate TOML, or a fallback list.

    The driver does not depend on the bench package for this — it parses the
    slate directly with tomllib so dry-run has zero heavy imports. If the slate
    file is absent (e.g. ``slate_upgrade.toml`` not yet authored), a small
    deterministic fallback set is used so the state machine still runs.
    """
    slate_path = Path(slate_path)
    if not slate_path.exists():
        return list(_FALLBACK_BASELINE_TOPICS)
    try:
        raw = tomllib.loads(slate_path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return list(_FALLBACK_BASELINE_TOPICS)
    prompts = raw.get("prompt", [])
    topics = [str(row.get("text", "")).strip() for row in prompts if row.get("text")]
    return topics or list(_FALLBACK_BASELINE_TOPICS)


# ---------------------------------------------------------------------------
# Phase implementations
#
# Each phase:
#   * enumerates its cells (deterministic order)
#   * skips cells already in state.cells_done(phase)  (resume)
#   * before each cell, checks the budget; on exhaustion -> pause + exit 0
#   * routes work through _invoke (no subprocess in dry-run)
#   * checkpoints after every cell
# ---------------------------------------------------------------------------


class _Paused(Exception):
    """Raised internally when the hard ceiling halts a phase mid-flight."""

    def __init__(self, phase: str) -> None:
        super().__init__(phase)
        self.phase = phase


# Zero-cost invocation kinds: these cells do NO external/LLM work and so must
# reserve 0 against the ceiling (B1/B3). The decision + regression gate cells
# are pure; their real effects run in on_result callbacks.
_ZERO_COST_KINDS: frozenset[str] = frozenset({"decide", "regression"})


def _cell_known_cost(invoke_kind: str, payload: dict[str, Any], *,
                     dry_run: bool) -> int:
    """The cell's KNOWN cost used by the pre-flight budget gate (B1).

    The previous gate reserved a fixed ``_DRY_RUN_TOKENS_PER_CELL`` (1000) for
    EVERY cell regardless of real cost. That made the ceiling soft (it could be
    breached by a cell costing more than 1000) AND deadlocked zero-cost terminal
    cells like ``gate:decide`` near the ceiling (B3). The reservation now
    reflects what we actually know:

      * zero-cost kinds (``decide``/``regression``) reserve 0;
      * a payload may declare ``known_cost`` (e.g. a real cell with a known
        budget) which is honoured exactly;
      * dry-run cells reserve their deterministic simulated cost
        (``_DRY_RUN_TOKENS_PER_CELL`` baseline);
      * otherwise (real mode, unknown cost) we conservatively reserve the
        baseline so the gate still trips before a runaway — but real mode ALSO
        enforces a TRULY hard per-cell cap after the work (see _run_cells).
    """
    if invoke_kind in _ZERO_COST_KINDS:
        return 0
    if "known_cost" in payload:
        return max(int(payload.get("known_cost", 0) or 0), 0)
    return _DRY_RUN_TOKENS_PER_CELL


def _run_cells(
    state: JobState,
    tracker: BudgetTracker,
    phase: str,
    cells: list[tuple[str, dict[str, Any]]],
    *,
    dry_run: bool,
    invoke_kind: str,
    on_result: Callable[[str, dict[str, Any]], None] | None = None,
    defer_done: bool = False,
) -> None:
    """Drive one phase's cell loop with resume + budget-pause semantics.

    ``cells`` is a list of ``(cell_id, payload)``. ``payload`` is forwarded to
    ``_invoke``. Already-done cells are skipped. Before executing a cell the
    budget is checked against the cell's KNOWN cost (B1); if charging it would
    breach the ceiling the phase is paused and ``_Paused`` is raised (the caller
    turns that into a clean exit 0).

    Hard ceiling (B1): the ceiling is TRULY hard. In real mode, if the work's
    ACTUAL token cost would push spend past the ceiling, the cell is ABORTED
    (its tokens are NOT charged and it is NOT recorded done) and the phase
    pauses — we never charge past the ceiling.

    Zero-cost terminal cells (B3): a zero-cost cell (e.g. ``gate:decide``)
    reserves 0 and is exempt from the exhaustion check, so spending within the
    old fixed 1000-token band of the ceiling can no longer false-pause and
    deadlock the deliverable.

    ``defer_done`` (B6): when True, the phase is NOT marked DONE here — the
    caller marks it DONE only AFTER its terminal side effect (verify's
    significance write, gate's commit) succeeds, so a crash before that side
    effect resumes the phase instead of skipping it.
    """
    state.set_status(phase, STATUS_RUNNING)
    done = state.cells_done(phase)

    for cell_id, payload in cells:
        if cell_id in done:
            continue  # resume: skip completed cell, no duplicate work

        known_cost = _cell_known_cost(invoke_kind, payload, dry_run=dry_run)
        # Budget gate BEFORE doing the work, using the cell's KNOWN cost. A
        # zero-cost cell (known_cost == 0) is exempt from the exhaustion check
        # so a terminal zero-cost deliverable still runs at the ceiling (B3).
        if known_cost > 0 and (tracker.exhausted()
                               or tracker.would_exceed(known_cost)):
            state.set_status(phase, STATUS_PAUSED)
            state.sync_tokens_from(tracker)
            raise _Paused(phase)

        res = _invoke(invoke_kind, dry_run=dry_run, tracker=tracker, **payload)
        tokens = int(res.get("tokens", 0) or 0)

        # Hard per-cell cap (B1): if the ACTUAL cost would breach the ceiling,
        # abort the cell without charging or recording it, and pause. This makes
        # the ceiling truly hard even when the real cost exceeded what we knew
        # up front (the pre-flight reservation can only estimate).
        if tokens > 0 and tracker.would_exceed(tokens):
            _warn(f"[{phase}] cell {cell_id} cost {tokens:,} tokens would "
                  f"breach the hard ceiling "
                  f"(spent={tracker.spent():,}/{tracker.hard_ceiling:,}); "
                  f"aborting cell, not charging past the ceiling")
            state.set_status(phase, STATUS_PAUSED)
            state.sync_tokens_from(tracker)
            raise _Paused(phase)

        tracker.charge(tokens)
        state.mark_cell_done(phase, cell_id, tokens=tokens,
                             artifact=res.get("artifact"), result=res)
        state.sync_tokens_from(tracker)
        if on_result is not None:
            on_result(cell_id, res)

    if not defer_done:
        state.set_status(phase, STATUS_DONE)


def phase_baseline(state: JobState, tracker: BudgetTracker, args: argparse.Namespace) -> None:
    """Measure the CURRENT swarm on the slate x seeds + a K-sample floor.

    Cells: (topic, seed) consensus runs, plus K self-consistency samples on the
    first topic to establish a majority-vote floor (Wang et al. self-consistency
    baseline — see reference_agent_swarm_research.md).

    G3 — self-consistency FLOOR. The K self-consistency cells are aggregated
    into a baseline floor (mean + majority/median over the K samples) and
    persisted as ``baseline_floor_<branch>.json``. The gate consumes this floor
    so a technique whose ``on`` arm does not clear the baseline floor is dropped
    even if it beats its own ``off`` arm — a technique that can't even match the
    current swarm's self-consistency baseline is not an upgrade.
    """
    topics = _load_slate_topics(Path(args.slate))
    seeds = list(range(args.seeds))
    cells: list[tuple[str, dict[str, Any]]] = []
    # LIVE: consensus swarm over slate x seeds.
    for ti, topic in enumerate(topics):
        for seed in seeds:
            cells.append((f"baseline:t{ti}:s{seed}",
                          {"topic": topic, "seed": seed, "max_rounds": args.max_rounds,
                           "backend_id": args.backend}))
    # LIVE: K-sample self-consistency floor on the first topic.
    if topics:
        for k in range(args.seeds):
            cells.append((f"selfconsistency:t0:k{k}",
                          {"topic": topics[0], "seed": 1000 + k, "max_rounds": args.max_rounds,
                           "backend_id": args.backend}))

    # B6-class fix: defer DONE until the floor artifact (baseline's terminal side
    # effect) is written, and rebuild the K samples from PERSISTED cell results so a
    # --resume (which skips already-done cells) can still compute and write the floor
    # instead of marking baseline done with a degenerate/absent floor. Without this,
    # a mid-baseline pause on a multi-window run would leave the gate with no floor
    # (falling back to keep-all). Mirrors the phase_verify fix.
    _run_cells(state, tracker, "baseline", cells, dry_run=args.dry_run,
               invoke_kind="consensus", defer_done=True)

    # G3: aggregate the self-consistency floor from PERSISTED results and persist it.
    sc_samples: list[float] = []
    base_results = state.cell_results("baseline")
    for cell_id, _payload in cells:
        if not cell_id.startswith("selfconsistency:"):
            continue
        res = base_results.get(cell_id)
        if res:
            sc_samples.append(float(res.get("result", {}).get("avg_quality", 0) or 0))

    floor = _aggregate_baseline_floor(sc_samples)
    floor_path = state.path.parent / f"baseline_floor_{_safe(state.data['branch'])}.json"
    floor_path.write_text(json.dumps(floor, indent=2), encoding="utf-8")
    state.add_artifact("baseline", str(floor_path))
    # Terminal side effect landed — now safe to mark baseline done (B6-class).
    state.set_status("baseline", STATUS_DONE)


def _aggregate_baseline_floor(samples: list[float]) -> dict[str, Any]:
    """Aggregate K self-consistency quality samples into a baseline floor (G3).

    Reports both the mean and the majority/median so the gate can pick the
    conservative bar. Never raises on empty input.
    """
    import statistics
    n = len(samples)
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "floor": 0.0, "samples": []}
    mean = statistics.mean(samples)
    median = statistics.median(samples)
    # The floor a technique must clear: the LOWER of mean/median (conservative —
    # a technique should beat the typical self-consistency sample, not just the
    # average inflated by an outlier).
    floor = round(min(mean, median), 4)
    return {
        "n": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "floor": floor,
        "samples": [round(s, 4) for s in samples],
    }


def phase_research(state: JobState, tracker: BudgetTracker, args: argparse.Namespace) -> None:
    """Discover candidate techniques via the research-swarm.

    Cells: (technique-topic, seed). ``--techniques`` caps how many of the
    default research topics to pursue; ``--seeds`` controls repeats.
    """
    n = max(1, min(int(args.techniques), len(DEFAULT_RESEARCH_TOPICS)))
    topics = list(DEFAULT_RESEARCH_TOPICS[:n])
    cells: list[tuple[str, dict[str, Any]]] = []
    # LIVE: research-swarm (separate program; see _invoke 'research').
    for ti, topic in enumerate(topics):
        for seed in range(args.seeds):
            cells.append((f"research:t{ti}:s{seed}", {"topic": topic, "seed": seed}))
    _run_cells(state, tracker, "research", cells, dry_run=args.dry_run,
               invoke_kind="research")


def phase_implement(state: JobState, tracker: BudgetTracker, args: argparse.Namespace) -> None:
    """Implement each candidate technique via mythos (cost-ceilinged).

    Cells: (technique). One mythos run per candidate technique with a per-run
    ``--cost-ceiling`` so a single candidate cannot drain the whole budget.
    """
    n = max(1, min(int(args.techniques), len(DEFAULT_RESEARCH_TOPICS)))
    cells: list[tuple[str, dict[str, Any]]] = []
    # LIVE: mythos per candidate.
    for ti in range(n):
        technique = _technique_id(ti)
        task = (f"Implement and self-verify the blitz-swarm upgrade candidate "
                f"'{technique}' as a toggleable mechanism behind the quality "
                f"profile, with a unit test.")
        cells.append((f"implement:{technique}",
                      {"task": task, "technique": technique,
                       "cost_ceiling": 2.0, "max_replans": 2}))
    _run_cells(state, tracker, "implement", cells, dry_run=args.dry_run,
               invoke_kind="mythos")


def phase_verify(state: JobState, tracker: BudgetTracker, args: argparse.Namespace) -> None:
    """Ablate each mechanism on/off across seeds, then test significance.

    Cells: (technique, arm, seed) where arm in {on, off}. After the cell loop,
    paired significance (stats.paired_t_test + cohens_d) is computed per
    technique from the per-arm quality samples and stashed as an artifact note.
    """
    n = max(1, min(int(args.techniques), len(DEFAULT_RESEARCH_TOPICS)))
    cells: list[tuple[str, dict[str, Any]]] = []
    # LIVE: bench ablation on/off x seeds per technique.
    for ti in range(n):
        technique = _technique_id(ti)
        for arm in ("off", "on"):
            for seed in range(args.seeds):
                cells.append((
                    f"verify:{technique}:{arm}:s{seed}",
                    {"technique": technique, "arm": arm, "seed": seed,
                     "slate": args.slate, "max_rounds": args.max_rounds,
                     "backend_id": args.backend},
                ))

    # B6: defer the phase DONE until AFTER the significance artifact is written.
    # The significance file is verify's terminal side effect; if we marked the
    # phase DONE before writing it, a crash here would lose the artifact yet a
    # --resume would skip verify entirely.
    _run_cells(state, tracker, "verify", cells, dry_run=args.dry_run,
               invoke_kind="bench_ablation", defer_done=True)

    # Rebuild per-(technique, arm) quality samples from the PERSISTED cell
    # results rather than an in-memory collector — so a --resume (which skips
    # the already-done cells) can still recompute and write the significance
    # artifact, instead of marking verify done with no terminal side effect (B6).
    samples: dict[str, dict[str, list[float]]] = {}
    cell_results = state.cell_results("verify")
    for cell_id, _payload in cells:
        res = cell_results.get(cell_id)
        if not res:
            continue
        r = res.get("result", {})
        tech = str(r.get("technique", ""))
        arm = str(r.get("arm", ""))
        if not tech or arm not in ("on", "off"):
            continue
        q = float(r.get("avg_quality", 0) or 0)
        samples.setdefault(tech, {"on": [], "off": []})[arm].append(q)

    # Significance pass (paired on/off). Best-effort: needs >=2 paired samples.
    sig = _compute_significance(samples)
    if sig:
        note = state.path.parent / f"verify_significance_{_safe(state.data['branch'])}.json"
        note.write_text(json.dumps(sig, indent=2), encoding="utf-8")
        state.add_artifact("verify", str(note))
    # Terminal side effect landed (or there were no paired samples to write) —
    # NOW it is safe to mark the phase done (B6).
    state.set_status("verify", STATUS_DONE)


def phase_gate(state: JobState, tracker: BudgetTracker, args: argparse.Namespace) -> None:
    """Decision rule + HARD regression gate + commit the passing mechanisms.

    Two zero-cost gate cells, then optional VCS work:

      1. ``gate:decide``     — applies the significance-driven keep rule
                               (:func:`_apply_decision_rule`; G4 + G3).
      2. ``gate:regression`` — (G2) runs the regression suite
                               (``pytest -q`` + ``bench/mast_regression`` when
                               present) BEFORE committing. If it FAILS, the gate
                               REFUSES to commit and records the failure in
                               ``gate_decision.json``.

    Commit (real mode only) stages an ALLOWLIST — only the ``mechanisms/`` +
    ``tests/`` files the kept techniques touched — never ``git add -A`` (G8),
    then ``git commit`` on the current branch and a best-effort ``gh pr create``.

    NEVER checkout/commit to main: the driver refuses to operate if the current
    branch is main/master.

    B6: the phase is marked DONE only AFTER its terminal side effect (the
    commit, or — in dry-run / refusal paths — after the decision artifact is
    written), so a crash before the commit resumes the gate rather than skipping
    it.
    """
    # Both gate cells route through _run_cells so budget accounting + resume are
    # uniform; their real work is pure / runs in on_result (no LLM, zero cost).
    cells: list[tuple[str, dict[str, Any]]] = [
        ("gate:decide", {}),
        ("gate:regression", {}),
    ]
    decision: dict[str, Any] = {}
    regression: dict[str, Any] = {}

    def _on_gate_cell(cell_id: str, res: dict[str, Any]) -> None:
        if cell_id == "gate:decide":
            decision.update(_apply_decision_rule(state))
        elif cell_id == "gate:regression":
            # G2: hard regression gate — run the test suite before committing.
            # Skipped in dry-run (no subprocess); recorded as such.
            regression.update(
                _run_regression_suite(dry_run=args.dry_run))

    # B6: defer DONE — the gate's terminal side effect (commit / decision write)
    # lands below, not in the cell loop.
    _run_cells(state, tracker, "gate", cells, dry_run=args.dry_run,
               invoke_kind="decide", on_result=_on_gate_cell, defer_done=True)

    # Fold the regression result into the decision record so the artifact is the
    # single source of truth for "what passed and was it safe to commit".
    decision["regression"] = regression

    branch = state.data.get("branch", "")
    kept = decision.get("kept", [])
    regression_ok = bool(regression.get("ok", True))
    decision["committed"] = False

    # Decide whether we are allowed to commit BEFORE writing the artifact so the
    # artifact records the final verdict.
    refuse_reason: str | None = None
    if branch in ("main", "master"):
        refuse_reason = "on main/master"
    elif not kept:
        refuse_reason = "no mechanisms passed the gate"
    elif not regression_ok:
        # G2: tests failed — refuse to commit, record the failure.
        refuse_reason = "regression suite FAILED"

    if refuse_reason:
        decision["refused_commit_reason"] = refuse_reason

    # Persist the decision (now incl. regression result + commit verdict).
    decision_path = state.path.parent / f"gate_decision_{_safe(branch)}.json"
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    state.add_artifact("gate", str(decision_path))

    if branch in ("main", "master"):
        state.add_artifact("gate", "REFUSED: on main/master, no commit made")
        state.set_status("gate", STATUS_DONE)  # B6: terminal effect (refusal) recorded
        return
    if not kept:
        state.add_artifact("gate", "no mechanisms passed the gate; nothing committed")
        state.set_status("gate", STATUS_DONE)
        return
    if not regression_ok:
        state.add_artifact("gate", f"REFUSED: {refuse_reason}; nothing committed")
        state.set_status("gate", STATUS_DONE)
        return

    if args.dry_run:
        # State machine completes; no VCS side effects in dry-run.
        state.add_artifact("gate", f"DRY-RUN would commit kept={kept} on {branch}")
        state.set_status("gate", STATUS_DONE)  # B6: dry-run terminal effect recorded
        return

    # LIVE: stage ONLY the allowlisted files the kept techniques touched, then
    # commit on the current branch only (G8 — never `git add -A`).
    paths = _gate_stage_allowlist(state, kept)
    commit_msg = "feat(swarm): land gated upgrade mechanisms\n\nKept: " + ", ".join(kept)
    _invoke("git", dry_run=False, tracker=tracker, argv=["add", "--", *paths])
    commit = _invoke("git", dry_run=False, tracker=tracker,
                     argv=["commit", "-m", commit_msg])
    state.add_artifact("gate", f"staged: {', '.join(paths)}")
    # LIVE: best-effort PR (non-fatal on failure).
    pr = _invoke("gh_pr", dry_run=False, tracker=tracker,
                 argv=["pr", "create", "--fill", "--head", branch])
    if pr.get("artifact"):
        state.add_artifact("gate", f"PR: {pr['artifact']}")
    # B6: mark DONE only AFTER the commit succeeded. If the commit failed, leave
    # the phase NOT-done so --resume retries the side effect.
    if commit.get("ok"):
        decision["committed"] = True
        decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        state.set_status("gate", STATUS_DONE)
    else:
        _warn(f"[gate] commit failed (rc="
              f"{commit.get('result', {}).get('returncode')}); leaving gate "
              f"not-done so --resume retries the commit")


def _run_regression_suite(*, dry_run: bool) -> dict[str, Any]:
    """G2 — run the HARD regression gate before committing.

    Runs ``[sys.executable, '-m', 'pytest', '-q']`` from the repo root, plus
    ``bench/mast_regression.py`` explicitly when it exists. Returns a dict with
    ``ok`` (all suites passed) and per-suite return codes. In dry-run we do NOT
    shell out (the dry-run contract: zero subprocesses) and report ``skipped``
    with ``ok=True`` so the smoke run completes.
    """
    if dry_run:
        return {"ok": True, "skipped": True, "dry_run": True,
                "reason": "regression suite not run in dry-run (no subprocess)"}

    suites: list[list[str]] = [[sys.executable, "-m", "pytest", "-q"]]
    mast = REPO_ROOT / "bench" / "mast_regression.py"
    if mast.exists():
        suites.append([sys.executable, "-m", "pytest", "-q", str(mast)])

    results: list[dict[str, Any]] = []
    all_ok = True
    for argv in suites:
        proc = subprocess.run(  # noqa: PLW1510 (we inspect returncode)
            argv, cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        ok = proc.returncode == 0
        all_ok = all_ok and ok
        results.append({"argv": argv[1:], "returncode": proc.returncode, "ok": ok})
    return {"ok": all_ok, "skipped": False, "suites": results}


def _gate_stage_allowlist(state: JobState, kept: list[str]) -> list[str]:
    """G8 — the allowlist of paths to ``git add`` for the kept techniques.

    Rather than ``git add -A`` (which over-stages scratch state, the manifest,
    and anything else dirty in the tree), stage only the source the kept
    techniques touched. Each kept technique may declare the files it touched via
    the implement-phase artifacts; absent that, we fall back to the conventional
    mechanism + test directories. ``jobs/state/`` is already gitignored so the
    manifest never leaks in.
    """
    paths: list[str] = []

    # Best-effort: techniques can record touched files as implement artifacts of
    # the form "touched:<relpath>". Collect any that exist.
    for art in state.phase("implement").get("artifacts", []):
        if isinstance(art, str) and art.startswith("touched:"):
            rel = art[len("touched:"):].strip()
            if rel:
                paths.append(rel)

    if not paths:
        # Conventional fallback allowlist: the mechanism + test trees only.
        for d in ("mechanisms", "tests"):
            if (REPO_ROOT / d).is_dir():
                paths.append(d)

    # De-dup, stable order.
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------


def _technique_id(i: int) -> str:
    return f"technique_{i + 1:02d}"


def _safe(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "branch"


def _compute_significance(samples: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    """Paired significance per technique from on/off quality samples (G4).

    For each technique we record the paired statistics the gate's
    significance-driven decision rule needs:

      * ``delta``       — mean_on - mean_off
      * ``t_stat``, ``p_value``, ``df`` — paired t-test
      * ``cohens_d``    — paired effect size (d_z)
      * ``bootstrap_ci`` — bootstrap CI of the paired DIFFERENCE (on - off); the
        gate keeps a technique only when this CI's LOWER bound is > 0, i.e. the
        lift is positive with confidence.

    Uses bench.stats when importable; otherwise records the raw means and a
    plain delta so the gate can still fall back to a delta-only decision. Never
    raises.
    """
    out: dict[str, Any] = {}
    try:
        from bench.stats import bootstrap_ci, cohens_d, paired_t_test  # noqa: PLC0415
        have_stats = True
    except Exception:  # pragma: no cover - stats import is optional
        have_stats = False

    for tech, arms in samples.items():
        on = arms.get("on", [])
        off = arms.get("off", [])
        rec: dict[str, Any] = {
            "n_on": len(on), "n_off": len(off),
            "mean_on": round(sum(on) / len(on), 4) if on else 0.0,
            "mean_off": round(sum(off) / len(off), 4) if off else 0.0,
        }
        rec["delta"] = round(rec["mean_on"] - rec["mean_off"], 4)
        if have_stats and len(on) == len(off) and len(on) >= 2:
            try:
                t, p, df = paired_t_test(on, off)
                rec.update({"t_stat": round(t, 4), "p_value": round(p, 6),
                            "df": df, "cohens_d": round(cohens_d(on, off), 4)})
            except Exception:  # pragma: no cover - degenerate samples
                pass
            try:
                # Bootstrap CI of the paired difference (G4). A fixed seed keeps
                # this deterministic so the manifest/artifact is reproducible.
                diffs = [a - b for a, b in zip(on, off)]
                lo, hi = bootstrap_ci(diffs, ci=95.0, n_resamples=2000, seed=42)
                rec["bootstrap_ci"] = [round(lo, 4), round(hi, 4)]
            except Exception:  # pragma: no cover - degenerate samples
                pass
        out[tech] = rec
    return out


# Significance-driven keep rule thresholds (G4). A technique is kept ONLY if
# every one of these holds — documented here so the rule is auditable:
#   1. delta > 0                  — the ``on`` arm beats the ``off`` arm.
#   2. cohens_d >= MIN_COHENS_D   — the effect is at least small-to-medium.
#   3. p_value <  MAX_P_VALUE     — the lift is statistically significant.
#   4. bootstrap CI lower > 0     — the lift is positive with 95% confidence.
#   5. mean_on >= baseline floor  — (G3) it clears the self-consistency floor.
# A technique missing the stats (degenerate samples / stats import absent)
# falls back to the non-regression delta band so the gate still decides.
_MIN_COHENS_D: float = 0.3
_MAX_P_VALUE: float = 0.05


def _load_baseline_floor(state: JobState) -> float:
    """Read the persisted self-consistency floor for the gate (G3).

    Returns 0.0 (no floor) if the baseline phase wrote no floor artifact.
    """
    for art in state.phase("baseline").get("artifacts", []):
        if "baseline_floor_" in art:
            p = Path(art)
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    return float(data.get("floor", 0) or 0)
                except (json.JSONDecodeError, OSError):
                    return 0.0
    return 0.0


def _apply_decision_rule(state: JobState, *, keep_band: float = 0.5) -> dict[str, Any]:
    """Significance-driven keep rule over the verify artifact (G4 + G3).

    For each technique with full paired statistics we KEEP it only when ALL of:
    ``delta > 0`` AND ``cohens_d >= 0.3`` AND ``p_value < 0.05`` AND the
    bootstrap-CI lower bound ``> 0`` AND the ``on`` arm mean clears the baseline
    self-consistency floor (G3). Techniques that lack stats (degenerate samples
    or no ``bench.stats``) fall back to the non-regression delta band
    (``delta >= -keep_band``) so the gate still has a decision. If no
    significance artifact exists at all, keep every implemented technique
    (smoke-run fallback). The reason for each keep/drop is recorded in
    ``detail`` so the decision is auditable.
    """
    sig_path = None
    for art in state.phase("verify").get("artifacts", []):
        if "verify_significance_" in art:
            sig_path = Path(art)
            break

    baseline_floor = _load_baseline_floor(state)

    kept: list[str] = []
    dropped: list[str] = []
    detail: dict[str, Any] = {}
    fallback = False

    if sig_path and sig_path.exists():
        try:
            sig = json.loads(sig_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            sig = {}
        for tech, rec in sig.items():
            decision = _decide_one_technique(rec, baseline_floor=baseline_floor,
                                             keep_band=keep_band)
            detail[tech] = decision
            if decision["keep"]:
                kept.append(tech)
            else:
                dropped.append(tech)
    else:
        # Fallback: keep every implemented technique.
        fallback = True
        for art in state.phase("implement").get("cells_done", []):
            # cell ids look like "implement:technique_01"
            tech = art.split(":", 1)[-1] if ":" in art else art
            kept.append(tech)

    return {
        "keep_band": keep_band,
        "baseline_floor": baseline_floor,
        "rule": ("delta>0 AND cohens_d>=%.2f AND p<%.2f AND bootstrap_ci_lo>0 "
                 "AND mean_on>=baseline_floor" % (_MIN_COHENS_D, _MAX_P_VALUE)),
        "kept": sorted(set(kept)),
        "dropped": sorted(set(dropped)),
        "detail": detail,
        "fallback_kept_all": fallback,
        "branch": state.data.get("branch", ""),
    }


def _decide_one_technique(rec: dict[str, Any], *, baseline_floor: float,
                          keep_band: float) -> dict[str, Any]:
    """Apply the documented keep rule to ONE technique's significance record.

    Returns a dict with ``keep`` (bool) plus the inputs/reasons (auditable).
    Uses the full stats path when present; otherwise falls back to the
    non-regression delta band.
    """
    delta = float(rec.get("delta", 0) or 0)
    mean_on = float(rec.get("mean_on", 0) or 0)
    has_full_stats = ("cohens_d" in rec and "p_value" in rec
                      and "bootstrap_ci" in rec)
    out: dict[str, Any] = {"delta": delta, "mean_on": mean_on,
                           "baseline_floor": baseline_floor}

    if has_full_stats:
        cohens = float(rec.get("cohens_d", 0) or 0)
        p = float(rec.get("p_value", 1) or 1)
        ci = rec.get("bootstrap_ci", [0.0, 0.0])
        ci_lo = float(ci[0]) if isinstance(ci, (list, tuple)) and ci else 0.0
        clears_floor = mean_on >= baseline_floor
        keep = bool(delta > 0 and cohens >= _MIN_COHENS_D and p < _MAX_P_VALUE
                    and ci_lo > 0 and clears_floor)
        out.update({
            "mode": "significance",
            "cohens_d": cohens, "p_value": p, "ci_lower": ci_lo,
            "clears_floor": clears_floor,
            "keep": keep,
        })
        # Human-readable reason for the drop (auditable).
        if not keep:
            reasons = []
            if not (delta > 0):
                reasons.append("delta<=0")
            if not (cohens >= _MIN_COHENS_D):
                reasons.append(f"cohens_d<{_MIN_COHENS_D}")
            if not (p < _MAX_P_VALUE):
                reasons.append(f"p>={_MAX_P_VALUE}")
            if not (ci_lo > 0):
                reasons.append("ci_lower<=0")
            if not clears_floor:
                reasons.append("below_baseline_floor")
            out["drop_reasons"] = reasons
        return out

    # Fallback: non-regression delta band (no full stats available).
    keep = delta >= -keep_band
    out.update({"mode": "delta_band", "keep": bool(keep)})
    if not keep:
        out["drop_reasons"] = [f"delta<-{keep_band}"]
    return out


# ---------------------------------------------------------------------------
# Phase registry + ordering enforcement
# ---------------------------------------------------------------------------


_PHASE_FUNCS: dict[str, Callable[[JobState, BudgetTracker, argparse.Namespace], None]] = {
    "baseline": phase_baseline,
    "research": phase_research,
    "implement": phase_implement,
    "verify": phase_verify,
    "gate": phase_gate,
}


def _phases_to_run(selected: str) -> list[str]:
    """Resolve the --phase flag to an ordered list of phase names."""
    if selected == "all":
        return list(PHASES)
    if selected not in PHASES:
        raise ValueError(f"unknown phase {selected!r}")
    return [selected]


def _assert_prereqs_done(state: JobState, phase: str) -> None:
    """Enforce phase ordering: every earlier phase must be 'done' first.

    Running a single phase out of order (e.g. ``--phase verify`` before
    ``baseline`` finished) is refused with a clear error.
    """
    idx = PHASES.index(phase)
    for earlier in PHASES[:idx]:
        if not state.is_done(earlier):
            raise SystemExit(
                f"phase ordering violation: cannot run '{phase}' until "
                f"'{earlier}' is done (status={state.status(earlier)}). "
                f"Run earlier phases first or use --phase all."
            )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_job(args: argparse.Namespace, *, started_at: float | str | None = None,
            tracker: BudgetTracker | None = None,
            branch: str | None = None) -> int:
    """Execute the requested phases. Returns a process exit code.

    Resumable + idempotent: safe to call repeatedly with ``--resume``. Returns
    0 on completion OR on a clean budget-pause (the pause is a normal,
    expected outcome, not an error).

    ``branch`` may be injected (tests / callers). When not injected the branch
    is resolved by :func:`_resolve_branch`, which — per B5 — does NOT shell out
    in dry-run: it reuses a persisted manifest branch or a fixed placeholder so
    a real ``--dry-run`` makes ZERO subprocess calls.
    """
    state_path = Path(args.state_file)
    if branch is None:
        branch = _resolve_branch(state_path, dry_run=bool(args.dry_run))

    if started_at is None:
        # Caller (real CLI) supplies the start time; we read it from an existing
        # manifest if resuming, else stamp it now. Never a module-import-time
        # time.time() — that would be baked at import.
        started_at = _read_started_at(state_path) or _now_epoch()

    state = JobState.load_or_init(
        state_path, started_at=started_at, branch=branch,
        hard_ceiling=args.max_tokens,
    )
    # Keep the manifest's hard ceiling in sync with the flag.
    state.data["hard_ceiling"] = int(args.max_tokens)
    state.data["branch"] = branch
    state.save()

    if tracker is None:
        # B2: snapshot the metrics floor ONCE and persist it; on resume reuse
        # the persisted snapshot instead of re-reading a possibly-grown
        # metrics.jsonl (which would double-count). Dry-run never reads metrics.
        if "metrics_floor_at_start" in state.data:
            metrics_floor: int | None = int(state.data["metrics_floor_at_start"])
        else:
            metrics_floor = None  # first run -> tracker snapshots it below
        tracker = BudgetTracker(
            hard_ceiling=int(args.max_tokens),
            dry_run=bool(args.dry_run),
            metrics_path=METRICS_PATH,
            initial_spent=int(state.data.get("tokens_spent", 0)),
            metrics_floor=metrics_floor,
        )
        # Persist the floor on the first run so every later resume reuses it.
        if "metrics_floor_at_start" not in state.data:
            state.data["metrics_floor_at_start"] = int(tracker.metrics_floor())
            state.save()
        # Job-relative budget: --max-tokens is THIS job's own budget. The ceiling
        # check is floor-inclusive (spent = charged + floor), so in real mode lift
        # the ceiling by the pre-job historical floor → the job's own charges are
        # capped at exactly --max-tokens regardless of unrelated prior spend already
        # in metrics.jsonl (other sessions). Dry-run floor is 0, so this is a no-op.
        if not args.dry_run:
            tracker.hard_ceiling = int(args.max_tokens) + int(tracker.metrics_floor())

    phases = _phases_to_run(args.phase)

    print(f"blitz-upgrade: branch={branch} dry_run={args.dry_run} "
          f"resume={args.resume} phases={','.join(phases)} "
          f"ceiling={tracker.hard_ceiling:,} spent={tracker.spent():,}")

    for phase in phases:
        # Skip already-completed phases (idempotent resume).
        if state.is_done(phase):
            print(f"  [{phase}] already done — skipping")
            continue
        # When running a single phase, enforce ordering.
        if args.phase != "all":
            _assert_prereqs_done(state, phase)

        if tracker.exhausted():
            _pause(state, phase, tracker)
            return 0

        print(f"  [{phase}] running (soft budget {SOFT_TOKEN_BUDGETS[phase]:,} tokens)")
        try:
            _PHASE_FUNCS[phase](state, tracker, args)
        except _Paused as paused:
            _pause(state, paused.phase, tracker)
            return 0
        done_cells = len(state.cells_done(phase))
        print(f"  [{phase}] done — {done_cells} cells, "
              f"{state.phase(phase)['tokens']:,} tokens, "
              f"{len(state.phase(phase)['artifacts'])} artifacts")

    state.sync_tokens_from(tracker)
    print(f"blitz-upgrade: complete. total tokens charged={tracker.charged():,} "
          f"(spent incl. history={tracker.spent():,}/{tracker.hard_ceiling:,})")
    return 0


def _pause(state: JobState, phase: str, tracker: BudgetTracker) -> None:
    state.set_status(phase, STATUS_PAUSED)
    state.sync_tokens_from(tracker)
    print(f"  [{phase}] PAUSED — budget ceiling reached "
          f"({tracker.spent():,}/{tracker.hard_ceiling:,} tokens)")
    print("PAUSED — resume with: python3 jobs/blitz_upgrade.py --resume")


# Placeholder branch used in dry-run so the state machine never shells out to
# git (B5). It is recorded in the manifest like any other branch.
_DRY_RUN_BRANCH = "dry-run"


def _resolve_branch(state_path: Path, *, dry_run: bool) -> str:
    """Resolve the working branch WITHOUT shelling out in dry-run (B5).

    The previous code always called :func:`_current_branch` (which runs ``git``)
    even under ``--dry-run``, violating the dry-run contract (zero subprocesses)
    and making a real ``--dry-run`` impossible without monkeypatching. Now:

      * dry-run  -> reuse the branch already persisted in the manifest (resume)
                    or a fixed placeholder; NEVER reads git.
      * real     -> the live :func:`_current_branch` read.
    """
    if dry_run:
        persisted = _read_branch(state_path)
        return persisted or _DRY_RUN_BRANCH
    return _current_branch()


def _read_branch(state_path: Path) -> str | None:
    """Branch recorded in an existing manifest, if any (no git, B5)."""
    if not Path(state_path).exists():
        return None
    try:
        data = json.loads(Path(state_path).read_text(encoding="utf-8"))
        b = data.get("branch")
        return str(b) if b else None
    except (json.JSONDecodeError, OSError):
        return None


def _current_branch() -> str:
    """Current git branch, or 'detached' if unavailable. NEVER returns blank.

    LIVE only — callers must NOT invoke this in dry-run (use
    :func:`_resolve_branch`), which keeps the dry-run contract of zero
    subprocesses (B5).
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True,
        )
        name = proc.stdout.strip()
        return name or "detached"
    except Exception:
        return "detached"


def _read_started_at(state_path: Path) -> float | str | None:
    if not Path(state_path).exists():
        return None
    try:
        data = json.loads(Path(state_path).read_text(encoding="utf-8"))
        return data.get("started_at")
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="blitz_upgrade",
        description="Resumable, budget-aware driver for the 10M-token "
                    "blitz-swarm upgrade job.",
    )
    ap.add_argument("--phase", choices=("all", *PHASES), default="all",
                    help="Which phase to run (default: all, in order).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the saved manifest, skipping done cells.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run the full state machine with NO subprocess/LLM "
                         "calls (deterministic simulation).")
    ap.add_argument("--max-tokens", type=int, default=HARD_CEILING,
                    help=f"Hard token ceiling (default {HARD_CEILING}).")
    ap.add_argument("--techniques", type=int, default=6,
                    help="How many candidate techniques to research/implement.")
    ap.add_argument("--seeds", type=int, default=3,
                    help="Seeds/repeats per cell for statistical power.")
    ap.add_argument("--max-rounds", type=int, default=4,
                    help="Consensus/bench max rounds per run (default 4).")
    ap.add_argument("--backend", default="claude",
                    help="LLM backend for swarm runs: claude (default, honors the "
                         "claude-only job choice) / codex / gemini / ollama. Threaded "
                         "into run_swarm backend_id. NOTE: the swarm's 'sonnet' agents "
                         "require the claude backend; the codex default fails on them.")
    ap.add_argument("--slate", default="bench/slate_upgrade.toml",
                    help="Bench slate TOML for baseline + verify phases.")
    ap.add_argument("--state-file", default=str(DEFAULT_STATE_FILE),
                    help="Path to the JSON state manifest "
                         "(default: jobs/state/upgrade_state.json).")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_job(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
