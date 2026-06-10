"""Deterministic tests for the resumable blitz-swarm upgrade driver.

These tests exercise the FULL state machine in dry-run, which by contract
spawns no subprocess and makes no LLM call. We assert the original five
invariants:

  1. `--dry-run --phase all` completes and marks all 5 phases 'done'.
  2. Re-running with `--resume` is idempotent — no cell is re-executed.
  3. With simulated spend near the hard ceiling, the next phase is 'paused'
     and the run exits cleanly (exit 0).
  4. Phase ordering is enforced (can't run a later phase before earlier done).
  5. In dry-run, subprocess is NEVER invoked (subprocess.run/Popen monkeypatched
     to raise; we assert they were not called).

…plus deterministic coverage for the correctness-bug fixes (B1-B6) and the
scientific-wiring acceptance criteria (G1-G5, G7, G8). Every fix id is covered
by at least one test below; the test name references the id.

Everything is driven through a temp state dir so no repo files are touched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Ensure the repo root is importable (mirror conftest.py).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import blitz_upgrade as bu  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _args(state_file: Path, **over):
    """Build a parsed-args namespace with test-friendly small fan-out."""
    defaults = dict(
        phase="all",
        resume=False,
        dry_run=True,
        max_tokens=bu.HARD_CEILING,
        techniques=3,
        seeds=2,
        max_rounds=4,
        slate="bench/slate_upgrade.toml",  # intentionally absent -> fallback
        state_file=str(state_file),
    )
    defaults.update(over)
    return bu.build_parser().parse_args(_to_argv(defaults))


def _to_argv(d: dict) -> list[str]:
    argv: list[str] = []
    argv += ["--phase", str(d["phase"])]
    if d["resume"]:
        argv.append("--resume")
    if d["dry_run"]:
        argv.append("--dry-run")
    argv += ["--max-tokens", str(d["max_tokens"])]
    argv += ["--techniques", str(d["techniques"])]
    argv += ["--seeds", str(d["seeds"])]
    argv += ["--max-rounds", str(d["max_rounds"])]
    argv += ["--slate", str(d["slate"])]
    argv += ["--state-file", str(d["state_file"])]
    return argv


# A fixed, non-main branch used by the dry-run tests. Injected through the new
# B5 seam (`_resolve_branch`) so the gate phase proceeds and the manifest branch
# is deterministic — without any test shelling out to git.
TEST_BRANCH = "feat/test-branch"


@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    return tmp_path / "state" / "upgrade_state.json"


@pytest.fixture(autouse=True)
def _no_subprocess(request, monkeypatch):
    """Hard guarantee: in these dry-run tests, nothing may shell out.

    Branch resolution is the only thing that would normally read git; we inject
    a fixed non-main branch through `_resolve_branch` (the B5 seam) so the gate
    proceeds, then forbid ANY subprocess use. A counter records calls so the
    no-subprocess test can assert zero.

    Tests that legitimately launch a *child* Python process to prove a property
    (the cross-process determinism / real-dry-run tests) opt out with the
    ``@pytest.mark.spawns_subprocess`` marker — they assert subprocess behaviour
    from the OUTSIDE and must not have the host process's subprocess forbidden.
    Those tests still inject the fixed branch.
    """
    calls = {"run": 0, "popen": 0}

    if request.node.get_closest_marker("spawns_subprocess"):
        # Still inject the deterministic branch, but DO NOT forbid subprocess
        # (the test itself spawns a child process to make its assertion).
        monkeypatch.setattr(bu, "_resolve_branch", lambda *a, **k: TEST_BRANCH)
        return calls

    def _forbidden_run(*a, **k):
        calls["run"] += 1
        raise AssertionError(f"subprocess.run called in dry-run: {a!r}")

    def _forbidden_popen(*a, **k):
        calls["popen"] += 1
        raise AssertionError(f"subprocess.Popen called in dry-run: {a!r}")

    monkeypatch.setattr(bu.subprocess, "run", _forbidden_run)
    monkeypatch.setattr(bu.subprocess, "Popen", _forbidden_popen)
    # Inject the fixed branch via the dry-run-safe resolver (never reads git).
    monkeypatch.setattr(bu, "_resolve_branch", lambda *a, **k: TEST_BRANCH)
    return calls


# ---------------------------------------------------------------------------
# 1. Full dry-run completes, all phases 'done'
# ---------------------------------------------------------------------------


def test_dry_run_all_phases_done(state_file: Path):
    rc = bu.run_job(_args(state_file), started_at=1000.0)
    assert rc == 0

    data = json.loads(state_file.read_text())
    assert data["started_at"] == 1000.0
    assert data["branch"] == TEST_BRANCH
    assert data["hard_ceiling"] == bu.HARD_CEILING

    for phase in bu.PHASES:
        assert data["phases"][phase]["status"] == bu.STATUS_DONE, phase

    # Every phase did real (simulated) cell work; gate's two decision/regression
    # cells are zero-token by design.
    assert data["phases"]["baseline"]["cells_done"], "baseline ran no cells"
    assert data["phases"]["research"]["cells_done"]
    assert data["phases"]["implement"]["cells_done"]
    assert data["phases"]["verify"]["cells_done"]
    assert data["phases"]["gate"]["cells_done"] == ["gate:decide", "gate:regression"]

    # Gate produced a decision artifact that kept something.
    decision_arts = [a for a in data["phases"]["gate"]["artifacts"]
                     if a.endswith(".json")]
    assert decision_arts, "gate wrote no decision artifact"
    decision = json.loads(Path(decision_arts[0]).read_text())
    assert "kept" in decision and "dropped" in decision

    # tokens_spent is the sum of per-phase tokens and is well under the ceiling.
    per_phase = sum(data["phases"][p]["tokens"] for p in bu.PHASES)
    assert data["tokens_spent"] == per_phase
    assert 0 < data["tokens_spent"] < bu.HARD_CEILING


# ---------------------------------------------------------------------------
# 2. Resume is idempotent — no cell re-executed
# ---------------------------------------------------------------------------


def test_resume_is_idempotent(state_file: Path, monkeypatch):
    # First full run.
    assert bu.run_job(_args(state_file), started_at=1000.0) == 0
    first = json.loads(state_file.read_text())
    first_cells = {p: list(first["phases"][p]["cells_done"]) for p in bu.PHASES}
    first_tokens = first["tokens_spent"]

    # Spy on _invoke: on the resume run, it must NOT be called for any
    # consensus/research/mythos/bench cell (all already done). The only thing
    # the gate re-derives is the pure decision/regression cells, which is fine.
    real_invoke = bu._invoke
    seen_kinds: list[str] = []

    def _spy(kind, *, dry_run, tracker, **kw):
        seen_kinds.append(kind)
        return real_invoke(kind, dry_run=dry_run, tracker=tracker, **kw)

    monkeypatch.setattr(bu, "_invoke", _spy)

    # Resume run.
    assert bu.run_job(_args(state_file, resume=True), started_at=1000.0) == 0
    second = json.loads(state_file.read_text())

    # No work-bearing kind should have fired on resume; phases were all done.
    work_kinds = {"consensus", "research", "mythos", "bench_ablation"}
    assert not (set(seen_kinds) & work_kinds), f"re-ran work cells: {seen_kinds}"

    # Cell sets identical (no duplicates, no growth).
    for p in bu.PHASES:
        assert second["phases"][p]["cells_done"] == first_cells[p], p
        # No duplicate cell ids.
        assert len(second["phases"][p]["cells_done"]) == len(set(second["phases"][p]["cells_done"]))

    # Token accounting did not double up.
    assert second["tokens_spent"] == first_tokens
    assert second["started_at"] == first["started_at"]  # start time preserved


def test_resume_mid_phase_skips_done_cells(state_file: Path):
    """Simulate an interrupted phase: pre-seed half of baseline's cells, then
    resume and confirm only the missing cells get executed."""
    # Initialise a manifest with baseline RUNNING and a couple cells already done.
    state = bu.JobState.load_or_init(
        state_file, started_at=2000.0, branch=TEST_BRANCH,
    )
    # Directly mark the deterministic first two cell ids done.
    state.set_status("baseline", bu.STATUS_RUNNING)
    state.mark_cell_done("baseline", "baseline:t0:s0", tokens=1234)
    state.mark_cell_done("baseline", "baseline:t0:s1", tokens=1234)
    pre_tokens = state.data["tokens_spent"]
    pre_done = set(state.cells_done("baseline"))

    # Resume the baseline phase only.
    rc = bu.run_job(_args(state_file, phase="baseline", resume=True), started_at=2000.0)
    assert rc == 0

    after = json.loads(state_file.read_text())
    after_done = set(after["phases"]["baseline"]["cells_done"])
    assert after["phases"]["baseline"]["status"] == bu.STATUS_DONE
    # The pre-seeded cells are still present exactly once, and new cells added.
    assert pre_done <= after_done
    assert len(after_done) > len(pre_done)
    # Pre-seeded cells keep their original token charge (not re-charged).
    assert after["tokens_spent"] > pre_tokens


# ---------------------------------------------------------------------------
# 3. Budget exhaustion pauses the next phase and exits cleanly
# ---------------------------------------------------------------------------


def test_budget_near_ceiling_pauses(state_file: Path, capsys):
    # Tiny ceiling so the very first phase pauses almost immediately. Each
    # dry-run cell costs ~1000 tokens; a 1500-token ceiling leaves room for at
    # most one cell before the pre-flight gate trips.
    rc = bu.run_job(_args(state_file, max_tokens=1500), started_at=3000.0)
    assert rc == 0  # clean exit on pause

    out = capsys.readouterr().out
    assert "PAUSED" in out
    assert "python3 jobs/blitz_upgrade.py --resume" in out

    data = json.loads(state_file.read_text())
    # baseline should be paused (first phase), later phases untouched.
    assert data["phases"]["baseline"]["status"] == bu.STATUS_PAUSED
    for later in ("research", "implement", "verify", "gate"):
        assert data["phases"][later]["status"] == bu.STATUS_PENDING
    # Never exceeded the ceiling.
    assert data["tokens_spent"] <= 1500


def test_budget_pause_then_resume_with_room(state_file: Path):
    # Pause under a tiny ceiling.
    assert bu.run_job(_args(state_file, max_tokens=1500), started_at=3000.0) == 0
    paused = json.loads(state_file.read_text())
    assert paused["phases"]["baseline"]["status"] == bu.STATUS_PAUSED
    paused_cells = len(paused["phases"]["baseline"]["cells_done"])

    # Resume with the full ceiling — the job should now complete.
    assert bu.run_job(_args(state_file, resume=True, max_tokens=bu.HARD_CEILING),
                      started_at=3000.0) == 0
    done = json.loads(state_file.read_text())
    for phase in bu.PHASES:
        assert done["phases"][phase]["status"] == bu.STATUS_DONE, phase
    # No regression of the cell already completed before the pause.
    assert len(done["phases"]["baseline"]["cells_done"]) >= paused_cells


def test_budget_tracker_semantics(tmp_path: Path):
    """Unit-level checks on BudgetTracker (dry-run mode ignores metrics file)."""
    bt = bu.BudgetTracker(hard_ceiling=10_000, dry_run=True)
    assert bt.spent() == 0
    assert bt.remaining() == 10_000
    assert not bt.exhausted()
    bt.charge(4_000)
    assert bt.spent() == 4_000
    assert bt.remaining() == 6_000
    assert bt.would_exceed(7_000)
    assert not bt.would_exceed(6_000)
    bt.charge(6_000)
    assert bt.exhausted()
    assert bt.remaining() == 0

    # Real mode reads metrics.jsonl as an additive floor.
    mp = tmp_path / "metrics.jsonl"
    mp.write_text(
        json.dumps({"total_input_tokens": 100, "total_output_tokens": 50}) + "\n"
        + json.dumps({"total_input_tokens": 10, "total_output_tokens": 5}) + "\n",
        encoding="utf-8",
    )
    bt2 = bu.BudgetTracker(hard_ceiling=1_000, dry_run=False, metrics_path=mp)
    assert bt2.spent() == 165  # 150 + 15 historical floor
    bt2.charge(35)
    assert bt2.spent() == 200
    assert bt2.charged() == 35  # only this-process charge is persisted


# ---------------------------------------------------------------------------
# 4. Phase ordering enforced
# ---------------------------------------------------------------------------


def test_phase_ordering_enforced(state_file: Path):
    # Fresh manifest, nothing done. Asking for 'verify' must be refused.
    with pytest.raises(SystemExit) as ei:
        bu.run_job(_args(state_file, phase="verify"), started_at=4000.0)
    msg = str(ei.value)
    assert "phase ordering violation" in msg
    assert "baseline" in msg  # names the first unmet prerequisite


def test_phase_ordering_allows_next_when_prereqs_done(state_file: Path):
    # Run the first two phases explicitly, in order.
    assert bu.run_job(_args(state_file, phase="baseline"), started_at=4000.0) == 0
    assert bu.run_job(_args(state_file, phase="research"), started_at=4000.0) == 0
    # Now 'implement' is allowed (baseline + research are done).
    assert bu.run_job(_args(state_file, phase="implement"), started_at=4000.0) == 0
    data = json.loads(state_file.read_text())
    assert data["phases"]["implement"]["status"] == bu.STATUS_DONE
    # But 'gate' is still blocked (verify not done).
    with pytest.raises(SystemExit):
        bu.run_job(_args(state_file, phase="gate"), started_at=4000.0)


def test_phases_to_run_ordering():
    assert bu._phases_to_run("all") == list(bu.PHASES)
    assert bu._phases_to_run("verify") == ["verify"]
    # Order constant matches the required spec exactly.
    assert bu.PHASES == ("baseline", "research", "implement", "verify", "gate")


# ---------------------------------------------------------------------------
# 5. No subprocess in dry-run
# ---------------------------------------------------------------------------


def test_dry_run_never_calls_subprocess(state_file: Path, _no_subprocess):
    # The autouse fixture already makes subprocess.run/Popen raise. A full
    # dry-run that completes is itself proof, but we also assert the counters.
    rc = bu.run_job(_args(state_file), started_at=5000.0)
    assert rc == 0
    assert _no_subprocess["run"] == 0
    assert _no_subprocess["popen"] == 0

    # And the manifest fully completed despite zero shelling out.
    data = json.loads(state_file.read_text())
    for phase in bu.PHASES:
        assert data["phases"][phase]["status"] == bu.STATUS_DONE


def test_invoke_dry_is_deterministic():
    """Same payload -> identical simulated result (so manifests are stable)."""
    bt = bu.BudgetTracker(dry_run=True)
    a = bu._invoke("consensus", dry_run=True, tracker=bt, topic="x", seed=1)
    b = bu._invoke("consensus", dry_run=True, tracker=bt, topic="x", seed=1)
    assert a == b
    assert a["tokens"] > 0 and a["ok"] is True
    # 'decide' and 'regression' are always zero-cost.
    d = bu._invoke("decide", dry_run=True, tracker=bt)
    assert d["tokens"] == 0
    r = bu._invoke("regression", dry_run=True, tracker=bt)
    assert r["tokens"] == 0


# ---------------------------------------------------------------------------
# JobState idempotency
# ---------------------------------------------------------------------------


def test_jobstate_checkpoint_idempotent(state_file: Path):
    s = bu.JobState.load_or_init(state_file, started_at=6000.0, branch="b")
    assert s.mark_cell_done("baseline", "c1", tokens=500) is True
    # Re-marking the same cell is a no-op and does not double-charge.
    assert s.mark_cell_done("baseline", "c1", tokens=500) is False
    assert s.phase("baseline")["tokens"] == 500
    assert s.data["tokens_spent"] == 500
    # Reloading from disk yields the same state.
    s2 = bu.JobState.load_or_init(state_file, started_at=9999.0, branch="other")
    # Existing manifest wins over the (different) init args.
    assert s2.data["started_at"] == 6000.0
    assert s2.cells_done("baseline") == {"c1"}
    assert s2.data["tokens_spent"] == 500


# ===========================================================================
# B1 — Hard ceiling is TRULY hard
# ===========================================================================


def test_B1_hard_ceiling_cell_cost_exceeds_remaining(state_file: Path):
    """A cell whose ACTUAL cost exceeds the remaining budget must NOT breach the
    ceiling: it is aborted (not charged, not recorded) and the phase pauses."""
    # Run baseline alone with room for ~a couple cells, then verify spend never
    # exceeds the ceiling regardless of per-cell cost.
    rc = bu.run_job(_args(state_file, phase="baseline", max_tokens=2500),
                    started_at=7000.0)
    assert rc == 0
    data = json.loads(state_file.read_text())
    assert data["phases"]["baseline"]["status"] == bu.STATUS_PAUSED
    # TRULY hard: spend strictly within the ceiling.
    assert data["tokens_spent"] <= 2500


def test_B1_known_cost_reservation_and_hard_abort():
    """The pre-flight reservation reflects the cell's KNOWN cost (0 for zero-cost
    kinds, declared known_cost otherwise), and a real cell whose actual cost
    would breach the ceiling is aborted rather than charged past it."""
    # Zero-cost kinds reserve 0.
    assert bu._cell_known_cost("decide", {}, dry_run=True) == 0
    assert bu._cell_known_cost("regression", {}, dry_run=True) == 0
    # A declared known_cost is honoured exactly.
    assert bu._cell_known_cost("bench_ablation", {"known_cost": 4242},
                               dry_run=False) == 4242
    # Default per-cell reservation otherwise.
    assert bu._cell_known_cost("consensus", {}, dry_run=True) == bu._DRY_RUN_TOKENS_PER_CELL

    # Hard abort: a cell that costs more than what's left is NOT charged past
    # the ceiling. Build a tiny state + tracker and feed a known_cost that fits
    # the pre-flight gate but a fake-injected larger actual cost is impossible
    # in dry-run, so we assert the would_exceed contract directly: charging a
    # cost that breaches must be refused.
    bt = bu.BudgetTracker(hard_ceiling=1000, dry_run=True)
    bt.charge(900)
    assert bt.would_exceed(101)        # 900 + 101 > 1000 -> would breach
    assert not bt.would_exceed(100)    # 900 + 100 == 1000 -> exactly at ceiling


def test_B1_real_mode_cell_cost_over_ceiling_aborts(tmp_path: Path, monkeypatch):
    """Real-mode: a single cell returning more tokens than remaining is aborted
    by the hard per-cell cap — spend never crosses the ceiling."""
    state = bu.JobState.load_or_init(
        tmp_path / "s.json", started_at=1.0, branch=TEST_BRANCH)
    tracker = bu.BudgetTracker(hard_ceiling=1000, dry_run=False, metrics_floor=0)
    tracker.charge(900)  # 100 left

    # An _invoke that reports a 5000-token cost (far over the 100 remaining).
    def _huge(kind, *, dry_run, tracker, **kw):
        return {"ok": True, "kind": kind, "tokens": 5000, "artifact": None,
                "result": {"avg_quality": 7.0}}

    monkeypatch.setattr(bu, "_invoke", _huge)
    cells = [("c0", {"known_cost": 0})]  # known_cost 0 passes the pre-flight gate
    with pytest.raises(bu._Paused):
        bu._run_cells(state, tracker, "baseline", cells, dry_run=False,
                      invoke_kind="bench_ablation")
    # The over-budget cell was NOT charged and NOT recorded.
    assert tracker.spent() == 900
    assert tracker.spent() <= tracker.hard_ceiling
    assert "c0" not in state.cells_done("baseline")
    assert state.status("baseline") == bu.STATUS_PAUSED


# ===========================================================================
# B2 — Real-mode resume must not double-count the metrics floor
# ===========================================================================


def test_B2_metrics_floor_snapshot_persisted_no_double_count(tmp_path: Path, monkeypatch):
    """metrics.jsonl growing between two processes must not double-count: the
    floor is snapshotted ONCE, persisted as metrics_floor_at_start, and reused
    on resume rather than re-read from the now-larger file."""
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps({"total_input_tokens": 1000, "total_output_tokens": 0}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bu, "METRICS_PATH", metrics)
    # Real mode (not dry-run); inject a fixed branch via the B5 seam so no git.
    monkeypatch.setattr(bu, "_resolve_branch", lambda *a, **k: TEST_BRANCH)
    # Forbid actual shelling out — the regression suite / git must never run
    # because we only drive the baseline phase here.
    monkeypatch.setattr(bu.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("subprocess in B2 test")))

    state_file = tmp_path / "state.json"

    # Process 1: real-mode baseline. We stub _invoke so each baseline cell
    # charges a fixed 10 tokens deterministically (no LLM), but the metrics
    # floor (1000) is real.
    def _fixed(kind, *, dry_run, tracker, **kw):
        if kind in ("decide", "regression"):
            return {"ok": True, "kind": kind, "tokens": 0, "artifact": None,
                    "result": {"avg_quality": 7.0}}
        return {"ok": True, "kind": kind, "tokens": 10, "artifact": None,
                "result": {"avg_quality": 7.0}}

    monkeypatch.setattr(bu, "_invoke", _fixed)

    a = _args(state_file, phase="baseline", dry_run=False)
    assert bu.run_job(a, started_at=10.0) == 0
    p1 = json.loads(state_file.read_text())
    floor_at_start = p1["metrics_floor_at_start"]
    assert floor_at_start == 1000  # snapshot of the historical floor
    spent_after_p1 = p1["tokens_spent"]  # the process-charged tokens only

    # metrics.jsonl GROWS between processes (more historical runs appended).
    with metrics.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"total_input_tokens": 5000, "total_output_tokens": 0}) + "\n")

    # Process 2: resume. The persisted floor (1000) must be reused; the grown
    # file (now 6000) must NOT be re-read into the ceiling.
    captured = {}

    real_ctor = bu.BudgetTracker

    def _capture_tracker(*a, **k):
        bt = real_ctor(*a, **k)
        captured["floor"] = bt.metrics_floor()
        captured["spent"] = bt.spent()
        return bt

    monkeypatch.setattr(bu, "BudgetTracker", _capture_tracker)

    a2 = _args(state_file, phase="baseline", dry_run=False, resume=True)
    assert bu.run_job(a2, started_at=10.0) == 0

    # The resumed tracker reused the SNAPSHOT floor (1000), not the grown 6000.
    assert captured["floor"] == 1000, "resume re-read the grown metrics floor (double-count)"
    # spent = persisted process tokens + snapshot floor (no double-count).
    assert captured["spent"] == spent_after_p1 + 1000
    p2 = json.loads(state_file.read_text())
    assert p2["metrics_floor_at_start"] == 1000  # unchanged across resume


# ===========================================================================
# B3 — Zero-cost gate must not deadlock near the ceiling
# ===========================================================================


def test_B3_zero_cost_gate_runs_within_band_of_ceiling(state_file: Path):
    """Spending within the old fixed-1000 band of the ceiling must STILL let the
    zero-cost gate:decide cell run (it reserves 0 and is exempt) — no deadlock,
    the deliverable commits."""
    # Drive baseline..verify normally, then approach the ceiling and run gate.
    for ph in ("baseline", "research", "implement", "verify"):
        assert bu.run_job(_args(state_file, phase=ph), started_at=8000.0) == 0

    # Manually push spend to within <1000 of a tight ceiling, then run gate.
    data = json.loads(state_file.read_text())
    spent = data["tokens_spent"]
    tight_ceiling = spent + 500  # only 500 left — less than the old 1000 reserve
    rc = bu.run_job(_args(state_file, phase="gate", max_tokens=tight_ceiling),
                    started_at=8000.0)
    assert rc == 0
    after = json.loads(state_file.read_text())
    # The gate STILL ran its zero-cost cells and completed (no false pause).
    assert after["phases"]["gate"]["status"] == bu.STATUS_DONE
    assert after["phases"]["gate"]["cells_done"] == ["gate:decide", "gate:regression"]
    # Ceiling not breached (gate is zero-cost).
    assert after["tokens_spent"] <= tight_ceiling


def test_B3_zero_cost_cell_exempt_when_exhausted(tmp_path: Path):
    """Even at/over the ceiling, a zero-cost cell runs (terminal deliverable)."""
    state = bu.JobState.load_or_init(
        tmp_path / "s.json", started_at=1.0, branch=TEST_BRANCH)
    tracker = bu.BudgetTracker(hard_ceiling=1000, dry_run=True)
    tracker.charge(1000)  # exactly exhausted
    assert tracker.exhausted()
    ran = []
    bu._run_cells(state, tracker, "gate", [("gate:decide", {})], dry_run=True,
                  invoke_kind="decide", on_result=lambda cid, r: ran.append(cid))
    # The zero-cost decide cell still executed despite exhaustion.
    assert ran == ["gate:decide"]
    assert "gate:decide" in state.cells_done("gate")


# ===========================================================================
# B4 — Dry-run determinism across processes (no PYTHONHASHSEED salt)
# ===========================================================================


def test_B4_stable_hash_independent_of_pythonhashseed():
    """_stable_hash must be identical regardless of PYTHONHASHSEED (the builtin
    hash() is salted and would differ across processes)."""
    key = ("consensus", "some topic", "technique_02", "on", 3)
    # Known SHA-256-derived value (first 8 hex digits of sha256(repr(key))).
    import hashlib as _h
    expected = int(_h.sha256(repr(key).encode()).hexdigest()[:8], 16)
    assert bu._stable_hash(key) == expected


@pytest.mark.spawns_subprocess
def test_B4_dry_run_identical_across_subprocesses(tmp_path: Path):
    """Two subprocesses with DIFFERENT PYTHONHASHSEED must produce byte-identical
    state manifests (the dry-run determinism contract)."""
    script = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(ROOT)!r})
        from jobs import blitz_upgrade as bu
        argv = ["--phase", "all", "--dry-run",
                "--techniques", "3", "--seeds", "2", "--max-rounds", "4",
                "--slate", "bench/slate_upgrade.toml",
                "--state-file", sys.argv[1]]
        args = bu.build_parser().parse_args(argv)
        # Inject a fixed branch (dry-run already avoids git, but be explicit).
        bu.run_job(args, started_at=1000.0, branch="feat/test-branch")
        data = json.loads(open(sys.argv[1]).read())
        # Print just the reproducible core (phase tokens + cells + spend).
        core = {{
            "tokens_spent": data["tokens_spent"],
            "phases": {{p: {{"tokens": data["phases"][p]["tokens"],
                            "cells_done": data["phases"][p]["cells_done"]}}
                        for p in bu.PHASES}},
        }}
        print(json.dumps(core, sort_keys=True))
        """
    )
    script_path = tmp_path / "run_dry.py"
    script_path.write_text(script, encoding="utf-8")

    def _run(seed: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=seed)
        sf = tmp_path / f"state_{seed}.json"
        proc = subprocess.run(
            [sys.executable, str(script_path), str(sf)],
            capture_output=True, text=True, env=env, cwd=str(ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip().splitlines()[-1]

    out_a = _run("0")
    out_b = _run("12345")
    assert out_a == out_b, "dry-run manifests differ across PYTHONHASHSEED"
    # Sanity: the core has real (non-empty) cell work.
    core = json.loads(out_a)
    assert core["tokens_spent"] > 0
    assert core["phases"]["baseline"]["cells_done"]


# ===========================================================================
# B5 — Real --dry-run (no monkeypatch) shells out ZERO times
# ===========================================================================


@pytest.mark.spawns_subprocess
def test_B5_real_dry_run_invokes_zero_subprocesses(tmp_path: Path):
    """A genuine `--dry-run --phase all` with NO monkeypatching must invoke the
    subprocess module zero times — including branch resolution (B5)."""
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        import subprocess as _sp
        _calls = {{"n": 0}}
        _orig_run, _orig_popen = _sp.run, _sp.Popen
        def _count_run(*a, **k):
            _calls["n"] += 1
            return _orig_run(*a, **k)
        class _CountPopen(_orig_popen):
            def __init__(self, *a, **k):
                _calls["n"] += 1
                super().__init__(*a, **k)
        _sp.run = _count_run
        _sp.Popen = _CountPopen
        from jobs import blitz_upgrade as bu
        # Re-bind the patched module reference the driver actually uses.
        bu.subprocess = _sp
        argv = ["--phase", "all", "--dry-run",
                "--techniques", "3", "--seeds", "2",
                "--slate", "bench/slate_upgrade.toml",
                "--state-file", sys.argv[1]]
        rc = bu.run_job(bu.build_parser().parse_args(argv), started_at=1000.0)
        print("RC", rc)
        print("CALLS", _calls["n"])
        """
    )
    script_path = tmp_path / "run_count.py"
    script_path.write_text(script, encoding="utf-8")
    sf = tmp_path / "state.json"
    proc = subprocess.run(
        [sys.executable, str(script_path), str(sf)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "RC 0" in proc.stdout, proc.stdout + proc.stderr
    assert "CALLS 0" in proc.stdout, f"dry-run shelled out: {proc.stdout}"
    # Manifest completed and branch is the dry-run placeholder (no git read).
    data = json.loads(sf.read_text())
    assert data["branch"] == bu._DRY_RUN_BRANCH
    for phase in bu.PHASES:
        assert data["phases"][phase]["status"] == bu.STATUS_DONE


# ===========================================================================
# B6 — Phase marked DONE only AFTER its terminal side effect
# ===========================================================================


def test_B6_verify_done_only_after_significance_write(state_file: Path, monkeypatch):
    """If verify crashes AFTER its cells but BEFORE writing the significance
    artifact, the phase must NOT be DONE (so resume re-runs the side effect)."""
    for ph in ("baseline", "research", "implement"):
        assert bu.run_job(_args(state_file, phase=ph), started_at=8100.0) == 0

    # Make the significance write blow up (simulate a crash at the side effect).
    boom = {"hit": False}
    real_sig = bu._compute_significance

    def _exploding_sig(samples):
        boom["hit"] = True
        raise RuntimeError("simulated crash before significance write")

    monkeypatch.setattr(bu, "_compute_significance", _exploding_sig)
    with pytest.raises(RuntimeError):
        bu.run_job(_args(state_file, phase="verify"), started_at=8100.0)
    assert boom["hit"]
    crashed = json.loads(state_file.read_text())
    # B6: verify is NOT done (cells ran, but the terminal artifact never landed).
    assert crashed["phases"]["verify"]["status"] != bu.STATUS_DONE
    # The verify cells DID complete (so resume won't re-run the expensive work).
    assert crashed["phases"]["verify"]["cells_done"]

    # Now restore and resume: the side effect lands, phase becomes DONE.
    monkeypatch.setattr(bu, "_compute_significance", real_sig)
    assert bu.run_job(_args(state_file, phase="verify", resume=True),
                      started_at=8100.0) == 0
    fixed = json.loads(state_file.read_text())
    assert fixed["phases"]["verify"]["status"] == bu.STATUS_DONE
    sig_arts = [a for a in fixed["phases"]["verify"]["artifacts"]
                if "verify_significance_" in a]
    assert sig_arts and Path(sig_arts[0]).exists()


def test_baseline_floor_rewritten_on_resume(state_file: Path, monkeypatch):
    """B6-class: if baseline crashes AFTER its cells but BEFORE the self-consistency
    floor write, the phase must NOT be DONE, and a --resume must rebuild + write the
    floor from the persisted cell results (the gate consumes this floor)."""
    boom = {"hit": False}
    real_agg = bu._aggregate_baseline_floor

    def _exploding(samples):
        boom["hit"] = True
        raise RuntimeError("simulated crash before baseline floor write")

    monkeypatch.setattr(bu, "_aggregate_baseline_floor", _exploding)
    with pytest.raises(RuntimeError):
        bu.run_job(_args(state_file, phase="baseline"), started_at=9000.0)
    assert boom["hit"]
    crashed = json.loads(state_file.read_text())
    # baseline is NOT done (cells ran, but the terminal floor artifact never landed)
    assert crashed["phases"]["baseline"]["status"] != bu.STATUS_DONE
    assert crashed["phases"]["baseline"]["cells_done"]

    # resume: the floor is rebuilt from persisted cell results and written; done.
    monkeypatch.setattr(bu, "_aggregate_baseline_floor", real_agg)
    assert bu.run_job(_args(state_file, phase="baseline", resume=True),
                      started_at=9000.0) == 0
    fixed = json.loads(state_file.read_text())
    assert fixed["phases"]["baseline"]["status"] == bu.STATUS_DONE
    floor_arts = [a for a in fixed["phases"]["baseline"]["artifacts"]
                  if "baseline_floor_" in a]
    assert floor_arts and Path(floor_arts[0]).exists()


def test_B6_gate_done_only_after_commit(tmp_path: Path, monkeypatch):
    """In real mode, the gate is marked DONE only AFTER the commit succeeds; a
    failed commit leaves the gate not-done so resume retries it."""
    state_file = tmp_path / "s.json"
    monkeypatch.setattr(bu, "_resolve_branch", lambda *a, **k: TEST_BRANCH)

    # Pre-build a manifest with baseline..verify done and a kept technique so the
    # gate proceeds to the commit step.
    state = bu.JobState.load_or_init(state_file, started_at=1.0, branch=TEST_BRANCH)
    for ph in ("baseline", "research", "implement", "verify"):
        state.set_status(ph, bu.STATUS_DONE)
    state.mark_cell_done("implement", "implement:technique_01", tokens=0)
    # A significance artifact that KEEPS technique_01 via the delta-band fallback.
    sig = {"technique_01": {"n_on": 0, "n_off": 0, "mean_on": 9.0,
                            "mean_off": 8.0, "delta": 1.0}}
    sig_path = state_file.parent / f"verify_significance_{bu._safe(TEST_BRANCH)}.json"
    sig_path.write_text(json.dumps(sig), encoding="utf-8")
    state.add_artifact("verify", str(sig_path))

    # Intercept _invoke: regression passes (handled in on_result), git COMMIT
    # FAILS. Everything routes through _invoke in the gate.
    def _invoke_commit_fails(kind, *, dry_run, tracker, **kw):
        if kind in ("decide", "regression"):
            return {"ok": True, "kind": kind, "tokens": 0, "artifact": None,
                    "result": {}}
        if kind == "git":
            argv = kw.get("argv", [])
            ok = not (argv and argv[0] == "commit")  # add ok, commit fails
            return {"ok": ok, "kind": "git", "tokens": 0, "artifact": None,
                    "result": {"argv": argv, "returncode": 0 if ok else 1}}
        if kind == "gh_pr":
            return {"ok": False, "kind": "gh_pr", "tokens": 0, "artifact": None,
                    "result": {"returncode": 1}}
        return {"ok": True, "kind": kind, "tokens": 0, "artifact": None, "result": {}}

    monkeypatch.setattr(bu, "_invoke", _invoke_commit_fails)
    # Regression suite must report success without shelling out.
    monkeypatch.setattr(bu, "_run_regression_suite",
                        lambda *, dry_run: {"ok": True, "skipped": False, "suites": []})

    a = _args(state_file, phase="gate", dry_run=False)
    assert bu.run_job(a, started_at=1.0) == 0
    after = json.loads(state_file.read_text())
    # Commit failed -> gate NOT done (resume will retry the commit) — B6.
    assert after["phases"]["gate"]["status"] != bu.STATUS_DONE
    decision = json.loads(
        (state_file.parent / f"gate_decision_{bu._safe(TEST_BRANCH)}.json").read_text())
    assert decision["committed"] is False


# ===========================================================================
# G1 — functional_composite is the PRIMARY bench signal (judge fallback)
# ===========================================================================


def test_G1_bench_avg_quality_reads_functional_composite():
    """_bench_avg_quality reads summary['functional_composite']['mean'] as the
    primary signal, falling back to the LLM-judge avg only when it is absent."""
    # functional_composite present -> used (0..1 scale).
    summary = {
        "functional_composite": {"mean": 0.81, "per_prompt": {"p1": 0.8}},
        "aggregate_quality": {"avg": {"mean": 7.4}},
    }
    assert bu._bench_avg_quality(summary) == 0.81

    # functional_composite absent -> judge avg fallback (secondary).
    summary_no_fc = {"aggregate_quality": {"avg": {"mean": 7.4}}}
    assert bu._bench_avg_quality(summary_no_fc) == 7.4

    # Empty summary -> 0.0, never raises.
    assert bu._bench_avg_quality({}) == 0.0


# ===========================================================================
# G3 — Self-consistency floor is aggregated, persisted, and fed to the gate
# ===========================================================================


def test_G3_baseline_floor_persisted_and_consumed(state_file: Path):
    """phase_baseline aggregates the K SINGLE-AGENT floor cells (H1: a true
    Wang-style self-consistency floor, not K more swarm runs) into a floor on
    the 0-1 composite scale, persists baseline_floor_<branch>.json, and the
    gate consumes it."""
    assert bu.run_job(_args(state_file, phase="baseline"), started_at=8200.0) == 0
    data = json.loads(state_file.read_text())
    floor_arts = [a for a in data["phases"]["baseline"]["artifacts"]
                  if "baseline_floor_" in a]
    assert floor_arts, "baseline wrote no floor artifact"
    floor = json.loads(Path(floor_arts[0]).read_text())
    assert floor["n"] == 5  # --floor-k default (M5: separate knob from --seeds)
    assert "floor" in floor and "mean" in floor and "median" in floor
    assert floor["floor"] == round(min(floor["mean"], floor["median"]), 4)
    # H1 scale unification: the floor lives on the SAME 0-1 composite scale the
    # verify samples use (the old 0-10 judge scale made every gate comparison
    # vacuously false).
    assert all(0.0 <= s <= 1.0 for s in floor["samples"])
    # The floor cells are single_agent cells, distinct from the swarm cells.
    assert any(c.startswith("floor:k") for c in data["phases"]["baseline"]["cells_done"])

    # The gate reads it back as a float bar.
    state = bu.JobState.load_or_init(state_file, started_at=8200.0, branch=TEST_BRANCH)
    assert bu._load_baseline_floor(state) == float(floor["floor"])


def test_G3_aggregate_baseline_floor_math():
    f = bu._aggregate_baseline_floor([6.0, 8.0, 7.0])
    assert f["n"] == 3
    assert f["mean"] == 7.0
    assert f["median"] == 7.0
    assert f["floor"] == 7.0
    # Empty input is safe.
    assert bu._aggregate_baseline_floor([]) == {
        "n": 0, "mean": 0.0, "median": 0.0, "floor": 0.0, "samples": []}


def test_G3_below_floor_technique_is_dropped(tmp_path: Path):
    """A technique whose on-arm mean does not clear the baseline floor is dropped
    even with an otherwise-significant positive lift."""
    state = bu.JobState.load_or_init(tmp_path / "s.json", started_at=1.0, branch="b")
    # Baseline floor of 9.0 (high bar).
    floor_path = tmp_path / f"baseline_floor_{bu._safe('b')}.json"
    floor_path.write_text(json.dumps({"floor": 9.0}), encoding="utf-8")
    state.add_artifact("baseline", str(floor_path))
    # Significance: clear positive lift, strong stats, but mean_on=7.5 < 9.0.
    sig = {"techA": {"mean_on": 7.5, "mean_off": 6.0, "delta": 1.5,
                     "cohens_d": 2.0, "p_value": 0.01, "bootstrap_ci": [0.8, 2.2]}}
    sig_path = tmp_path / f"verify_significance_{bu._safe('b')}.json"
    sig_path.write_text(json.dumps(sig), encoding="utf-8")
    state.add_artifact("verify", str(sig_path))
    decision = bu._apply_decision_rule(state)
    assert decision["kept"] == []
    assert decision["dropped"] == ["techA"]
    assert "below_baseline_floor" in decision["detail"]["techA"]["drop_reasons"]


# ===========================================================================
# G4 — Significance-driven decision (p, cohens_d, bootstrap CI), documented
# ===========================================================================


def test_G4_compute_significance_emits_full_stats():
    """_compute_significance emits t_stat, p_value, cohens_d AND bootstrap_ci."""
    samples = {"t": {"on": [8.0, 8.4, 8.2, 8.6], "off": [7.0, 7.1, 6.9, 7.2]}}
    sig = bu._compute_significance(samples)["t"]
    for key in ("delta", "t_stat", "p_value", "df", "cohens_d", "bootstrap_ci"):
        assert key in sig, key
    assert isinstance(sig["bootstrap_ci"], list) and len(sig["bootstrap_ci"]) == 2
    # Strong, consistent positive lift -> CI lower bound > 0.
    assert sig["bootstrap_ci"][0] > 0


def test_G4_decision_rule_requires_all_conditions():
    """KEEP iff delta>0 AND cohens_d>=0.3 AND p<0.05 AND ci_lo>0 AND clears floor.
    Flip any single condition and the technique is dropped."""
    base = {"mean_on": 8.0, "mean_off": 7.0, "delta": 1.0, "cohens_d": 1.5,
            "p_value": 0.01, "bootstrap_ci": [0.5, 1.5]}
    keep = bu._decide_one_technique(dict(base), baseline_floor=7.0, keep_band=0.5)
    assert keep["keep"] is True and keep["mode"] == "significance"

    # delta<=0
    assert not bu._decide_one_technique(
        {**base, "delta": -0.1}, baseline_floor=7.0, keep_band=0.5)["keep"]
    # cohens_d below threshold
    assert not bu._decide_one_technique(
        {**base, "cohens_d": 0.2}, baseline_floor=7.0, keep_band=0.5)["keep"]
    # p too high
    assert not bu._decide_one_technique(
        {**base, "p_value": 0.2}, baseline_floor=7.0, keep_band=0.5)["keep"]
    # CI lower bound <= 0
    assert not bu._decide_one_technique(
        {**base, "bootstrap_ci": [-0.1, 1.5]}, baseline_floor=7.0, keep_band=0.5)["keep"]
    # below baseline floor
    assert not bu._decide_one_technique(
        dict(base), baseline_floor=9.0, keep_band=0.5)["keep"]


def test_G4_decision_rule_documented_in_artifact(state_file: Path):
    """The decision artifact documents the exact rule string (auditable)."""
    for ph in bu.PHASES[:-1]:
        assert bu.run_job(_args(state_file, phase=ph), started_at=8300.0) == 0
    assert bu.run_job(_args(state_file, phase="gate"), started_at=8300.0) == 0
    data = json.loads(state_file.read_text())
    dec_arts = [a for a in data["phases"]["gate"]["artifacts"]
                if "gate_decision_" in a]
    decision = json.loads(Path(dec_arts[0]).read_text())
    assert "cohens_d" in decision["rule"]
    assert "p<" in decision["rule"]
    assert "bootstrap_ci_lo" in decision["rule"]
    assert "baseline_floor" in decision["rule"]


# ===========================================================================
# G5 — Dry-run seed variance + the stats path flips keep->drop on regression
# ===========================================================================


def test_G5_dry_run_seed_variance_exercises_stats(state_file: Path):
    """The K seeds differ in dry-run, so verify_significance_*.json carries
    t_stat/p_value/cohens_d/ci (the stats path is actually exercised)."""
    for ph in ("baseline", "research", "implement", "verify"):
        assert bu.run_job(_args(state_file, phase=ph), started_at=8400.0) == 0
    data = json.loads(state_file.read_text())
    sig_arts = [a for a in data["phases"]["verify"]["artifacts"]
                if "verify_significance_" in a]
    assert sig_arts, "verify wrote no significance artifact"
    sig = json.loads(Path(sig_arts[0]).read_text())
    # At least one technique has the full stats quartet (seeds varied enough).
    full = [t for t, rec in sig.items()
            if all(k in rec for k in ("t_stat", "p_value", "cohens_d", "bootstrap_ci"))]
    assert full, f"no technique exercised the stats path: {sig}"
    # And the per-seed dry-run quality genuinely varies across seeds.
    q = [bu._invoke_dry("bench_ablation", technique="technique_01", arm="on",
                        seed=s)["result"]["avg_quality"] for s in range(3)]
    assert len(set(q)) > 1, "dry-run avg_quality is constant across seeds"


def test_G5_synthetic_regression_flips_keep_to_drop(tmp_path: Path):
    """Injecting a synthetic regression (on < off) flips a kept technique to
    dropped through the real significance + decision path."""
    state = bu.JobState.load_or_init(tmp_path / "s.json", started_at=1.0, branch="b")
    floor_path = tmp_path / f"baseline_floor_{bu._safe('b')}.json"
    floor_path.write_text(json.dumps({"floor": 0.0}), encoding="utf-8")
    state.add_artifact("baseline", str(floor_path))

    # KEEP case: on clearly beats off.
    keep_samples = {"tech": {"on": [8.0, 8.2, 8.1, 8.3], "off": [7.0, 7.1, 6.9, 7.0]}}
    sig_keep = bu._compute_significance(keep_samples)
    sig_path = tmp_path / f"verify_significance_{bu._safe('b')}.json"
    sig_path.write_text(json.dumps(sig_keep), encoding="utf-8")
    state.add_artifact("verify", str(sig_path))
    dec_keep = bu._apply_decision_rule(state)
    assert dec_keep["kept"] == ["tech"], dec_keep

    # DROP case: inject a synthetic regression (on now WORSE than off).
    drop_samples = {"tech": {"on": [6.0, 6.1, 5.9, 6.0], "off": [7.0, 7.1, 6.9, 7.2]}}
    sig_drop = bu._compute_significance(drop_samples)
    sig_path.write_text(json.dumps(sig_drop), encoding="utf-8")
    dec_drop = bu._apply_decision_rule(state)
    assert dec_drop["dropped"] == ["tech"], dec_drop
    assert dec_drop["kept"] == []


# ===========================================================================
# G7 — Token attribution (best-effort) for research + mythos; WARN if blind
# ===========================================================================


def test_G7_mythos_token_attribution_reads_metrics_json(tmp_path: Path):
    """_read_mythos_tokens parses input+output tokens from the run dir's
    metrics.json (best-effort)."""
    run_dir = tmp_path / "mythos_run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"total_input_tokens": 120, "total_output_tokens": 380}),
        encoding="utf-8")
    assert bu._read_mythos_tokens(run_dir) == 500
    # Missing token data -> 0 (the caller WARNs).
    empty = tmp_path / "empty_run"
    empty.mkdir()
    (empty / "run.json").write_text(json.dumps({"cost_usd": 1.23}), encoding="utf-8")
    assert bu._read_mythos_tokens(empty) == 0
    # Missing dir entirely -> 0.
    assert bu._read_mythos_tokens(tmp_path / "nope") == 0


def test_G7_research_token_attribution_by_topic(tmp_path: Path):
    """research-swarm metrics.jsonl tokens are attributed by topic, newest after
    the run's start timestamp."""
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps({"topic": "T", "timestamp": 50.0,
                    "total_input_tokens": 10, "total_output_tokens": 20}) + "\n"
        + json.dumps({"topic": "T", "timestamp": 150.0,
                      "total_input_tokens": 100, "total_output_tokens": 200}) + "\n"
        + json.dumps({"topic": "other", "timestamp": 160.0,
                      "total_input_tokens": 9, "total_output_tokens": 9}) + "\n",
        encoding="utf-8",
    )
    rec = bu._read_metrics_record_for_topic("T", after_ts=100.0, metrics_path=metrics)
    assert bu._tokens_of(rec) == 300  # the newest 'T' record after ts=100


def test_G7_warn_when_attribution_unavailable(capsys):
    """_warn emits a clear WARNING to stderr so the governor isn't silently
    blind to unattributed spend."""
    bu._warn("research token attribution unavailable for X")
    err = capsys.readouterr().err
    assert "WARNING" in err and "attribution unavailable" in err


# ===========================================================================
# G8 — git add allowlist (mechanisms/ + tests/), jobs/state/ gitignored
# ===========================================================================


def test_G8_stage_allowlist_is_scoped_not_dash_A(tmp_path: Path):
    """The gate stages a SCOPED allowlist (mechanisms/ + tests/), never -A."""
    state = bu.JobState.load_or_init(tmp_path / "s.json", started_at=1.0, branch="b")
    paths = bu._gate_stage_allowlist(state, ["technique_01"])
    # Default conventional allowlist (the repo has both dirs).
    assert "mechanisms" in paths
    assert "tests" in paths
    assert "-A" not in paths and "--all" not in paths

    # Explicit touched-file declarations are honoured and de-duped.
    state.add_artifact("implement", "touched:mechanisms/foo.py")
    state.add_artifact("implement", "touched:tests/test_foo.py")
    state.add_artifact("implement", "touched:mechanisms/foo.py")  # dup
    paths2 = bu._gate_stage_allowlist(state, ["technique_01"])
    assert paths2 == ["mechanisms/foo.py", "tests/test_foo.py"]


def test_G8_jobs_state_is_gitignored():
    """jobs/state/ must be gitignored so the manifest never gets over-staged."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "jobs/state/" in gitignore


def test_G8_gate_uses_add_double_dash_paths_not_dash_A(tmp_path: Path, monkeypatch):
    """In real mode the gate issues `git add -- <paths>` (allowlist), and never
    `git add -A`."""
    state_file = tmp_path / "s.json"
    monkeypatch.setattr(bu, "_resolve_branch", lambda *a, **k: TEST_BRANCH)
    state = bu.JobState.load_or_init(state_file, started_at=1.0, branch=TEST_BRANCH)
    for ph in ("baseline", "research", "implement", "verify"):
        state.set_status(ph, bu.STATUS_DONE)
    state.mark_cell_done("implement", "implement:technique_01", tokens=0)
    sig = {"technique_01": {"mean_on": 9.0, "mean_off": 8.0, "delta": 1.0}}
    sig_path = state_file.parent / f"verify_significance_{bu._safe(TEST_BRANCH)}.json"
    sig_path.write_text(json.dumps(sig), encoding="utf-8")
    state.add_artifact("verify", str(sig_path))

    git_argvs: list[list[str]] = []

    def _invoke_capture(kind, *, dry_run, tracker, **kw):
        if kind == "git":
            git_argvs.append(list(kw.get("argv", [])))
            return {"ok": True, "kind": "git", "tokens": 0, "artifact": None,
                    "result": {"returncode": 0}}
        if kind == "gh_pr":
            return {"ok": True, "kind": "gh_pr", "tokens": 0,
                    "artifact": "http://pr", "result": {"returncode": 0}}
        return {"ok": True, "kind": kind, "tokens": 0, "artifact": None, "result": {}}

    monkeypatch.setattr(bu, "_invoke", _invoke_capture)
    monkeypatch.setattr(bu, "_run_regression_suite",
                        lambda *, dry_run: {"ok": True, "skipped": False, "suites": []})

    assert bu.run_job(_args(state_file, phase="gate", dry_run=False),
                      started_at=1.0) == 0
    # The first git call is `add -- <paths>`, never `-A`.
    add_calls = [a for a in git_argvs if a and a[0] == "add"]
    assert add_calls, f"no git add issued: {git_argvs}"
    for a in add_calls:
        assert "-A" not in a, f"gate used `git add -A`: {a}"
        assert a[1] == "--", f"git add not allowlisted with --: {a}"
        assert "mechanisms" in a or "tests" in a


# ===========================================================================
# G2 — Hard regression gate runs tests before committing; refuses on failure
# ===========================================================================


def test_G2_regression_failure_refuses_commit(tmp_path: Path, monkeypatch):
    """If the regression suite FAILS, the gate REFUSES to commit and records the
    failure in gate_decision.json (nothing is committed)."""
    state_file = tmp_path / "s.json"
    monkeypatch.setattr(bu, "_resolve_branch", lambda *a, **k: TEST_BRANCH)
    state = bu.JobState.load_or_init(state_file, started_at=1.0, branch=TEST_BRANCH)
    for ph in ("baseline", "research", "implement", "verify"):
        state.set_status(ph, bu.STATUS_DONE)
    state.mark_cell_done("implement", "implement:technique_01", tokens=0)
    sig = {"technique_01": {"mean_on": 9.0, "mean_off": 8.0, "delta": 1.0}}
    sig_path = state_file.parent / f"verify_significance_{bu._safe(TEST_BRANCH)}.json"
    sig_path.write_text(json.dumps(sig), encoding="utf-8")
    state.add_artifact("verify", str(sig_path))

    committed_argvs: list[list[str]] = []

    def _invoke_track_commit(kind, *, dry_run, tracker, **kw):
        if kind == "git":
            committed_argvs.append(list(kw.get("argv", [])))
        return {"ok": True, "kind": kind, "tokens": 0, "artifact": None, "result": {}}

    monkeypatch.setattr(bu, "_invoke", _invoke_track_commit)
    # Force the regression suite to FAIL (without shelling out).
    monkeypatch.setattr(
        bu, "_run_regression_suite",
        lambda *, dry_run: {"ok": False, "skipped": False,
                            "suites": [{"argv": ["-m", "pytest", "-q"],
                                        "returncode": 1, "ok": False}]})

    assert bu.run_job(_args(state_file, phase="gate", dry_run=False),
                      started_at=1.0) == 0
    # NOTHING was committed (no git commit issued).
    assert not any(a and a[0] == "commit" for a in committed_argvs), committed_argvs
    decision = json.loads(
        (state_file.parent / f"gate_decision_{bu._safe(TEST_BRANCH)}.json").read_text())
    assert decision["committed"] is False
    assert decision["refused_commit_reason"] == "regression suite FAILED"
    assert decision["regression"]["ok"] is False


def test_G2_regression_skipped_in_dry_run():
    """The regression gate is SKIPPED (no subprocess) in dry-run, reported ok."""
    res = bu._run_regression_suite(dry_run=True)
    assert res["ok"] is True
    assert res["skipped"] is True
    assert res["dry_run"] is True


def test_G2_regression_command_targets_pytest(tmp_path: Path, monkeypatch):
    """The real regression gate runs `<python> -m pytest -q` (+ mast_regression
    when present) from the repo root."""
    captured: list[list[str]] = []

    def _fake_run(argv, *a, **k):
        captured.append(list(argv))
        class _P:
            returncode = 0
        return _P()

    monkeypatch.setattr(bu.subprocess, "run", _fake_run)
    res = bu._run_regression_suite(dry_run=False)
    assert res["ok"] is True
    assert captured, "no regression command was run"
    # First suite is the full pytest run.
    assert captured[0][:3] == [sys.executable, "-m", "pytest"]
    assert "-q" in captured[0]
    # bench/mast_regression.py exists in this repo -> a second targeted suite.
    if (bu.REPO_ROOT / "bench" / "mast_regression.py").exists():
        assert any("mast_regression.py" in tok
                   for a in captured for tok in a)


# ===========================================================================
# Audit remediation — C4 failure semantics, C3 calibrator, H2 pairing, C2 toggle
# ===========================================================================


def test_C4_errored_cell_retried_then_succeeds(state_file: Path, monkeypatch):
    """A transiently-errored cell is retried (not marked done/failed on the
    first error) and succeeds on the second attempt."""
    real_invoke = bu._invoke
    flaky = {"failures_left": 1, "calls": 0}

    def _flaky_invoke(kind, **kw):
        res = real_invoke(kind, **kw)
        if kind == "single_agent" and kw.get("seed") == 0:
            flaky["calls"] += 1
            if flaky["failures_left"] > 0:
                flaky["failures_left"] -= 1
                bad = dict(res)
                bad["ok"] = False
                bad["error"] = "transient backend hiccup"
                return bad
        return res

    monkeypatch.setattr(bu, "_invoke", _flaky_invoke)
    assert bu.run_job(_args(state_file, phase="baseline"), started_at=9100.0) == 0
    data = json.loads(state_file.read_text())
    assert "floor:k0" in data["phases"]["baseline"]["cells_done"]
    assert "floor:k0" not in data["phases"]["baseline"].get("cells_failed", [])
    assert flaky["calls"] == 2  # first attempt errored, retry succeeded


def test_C4_persistent_failure_excluded_from_samples(state_file: Path, monkeypatch):
    """A cell that fails every attempt is recorded in cells_failed and its
    data is MISSING from the floor (n drops) — never a poisoning 0.0."""
    real_invoke = bu._invoke

    def _always_fail_k0(kind, **kw):
        res = real_invoke(kind, **kw)
        if kind == "single_agent" and kw.get("seed") == 0:
            bad = dict(res)
            bad["ok"] = False
            bad["error"] = "permanent failure"
            return bad
        return res

    monkeypatch.setattr(bu, "_invoke", _always_fail_k0)
    assert bu.run_job(_args(state_file, phase="baseline"), started_at=9200.0) == 0
    data = json.loads(state_file.read_text())
    assert "floor:k0" in data["phases"]["baseline"].get("cells_failed", [])
    floor_arts = [a for a in data["phases"]["baseline"]["artifacts"]
                  if "baseline_floor_" in a]
    floor = json.loads(Path(floor_arts[0]).read_text())
    assert floor["n"] == 4          # 5 floor cells - 1 failed = 4 samples
    assert 0.0 not in floor["samples"]  # missing, not zero-poisoned


def test_C4_usage_limit_pauses_and_resume_retries(state_file: Path, monkeypatch):
    """A usage-window exhaustion PAUSES the phase (clean exit 0); the same cell
    is retried after resume — never burned as failed."""
    real_invoke = bu._invoke
    limited = {"on": True}

    def _limited_invoke(kind, **kw):
        res = real_invoke(kind, **kw)
        if limited["on"] and kind == "single_agent" and kw.get("seed") == 1:
            bad = dict(res)
            bad["ok"] = False
            bad["error"] = "claude exit 1: usage limit reached for this window"
            return bad
        return res

    monkeypatch.setattr(bu, "_invoke", _limited_invoke)
    assert bu.run_job(_args(state_file, phase="baseline"), started_at=9300.0) == 0
    data = json.loads(state_file.read_text())
    assert data["phases"]["baseline"]["status"] == bu.STATUS_PAUSED
    assert "floor:k1" not in data["phases"]["baseline"]["cells_done"]
    assert "floor:k1" not in data["phases"]["baseline"].get("cells_failed", [])

    # Window reset: resume retries the SAME cell and the phase completes.
    limited["on"] = False
    assert bu.run_job(_args(state_file, phase="baseline", resume=True),
                      started_at=9300.0) == 0
    fixed = json.loads(state_file.read_text())
    assert fixed["phases"]["baseline"]["status"] == bu.STATUS_DONE
    assert "floor:k1" in fixed["phases"]["baseline"]["cells_done"]


def test_C3_calibrator_refuses_infeasible_job(tmp_path: Path, monkeypatch):
    """The pre-flight calibrator projects the whole job from one measured cell
    and refuses (rc=2) when the projection exceeds the budget; the override
    flag lets it proceed."""
    def _fat_invoke(kind, **kw):
        return {"ok": True, "kind": kind, "tokens": 50_000, "artifact": None,
                "result": {"topic": kw.get("topic", ""), "seed": 0,
                           "avg_quality": 5.0}}

    monkeypatch.setattr(bu, "_invoke", _fat_invoke)
    monkeypatch.setattr(bu, "_resolve_branch", lambda *a, **k: TEST_BRANCH)

    sf = tmp_path / "cal" / "upgrade_state.json"
    args = _args(sf, phase="baseline", dry_run=False, max_tokens=200_000)
    rc = bu.run_job(args, started_at=9400.0)
    assert rc == 2  # refused: 50K/swarm-run projects far past 200K
    data = json.loads(sf.read_text())
    assert data["calibration"]["feasible"] is False
    assert data["calibration"]["per_swarm_tokens"] == 50_000

    # Override proceeds past the refusal (fresh state dir).
    sf2 = tmp_path / "cal2" / "upgrade_state.json"
    args2 = _args(sf2, phase="baseline", dry_run=False, max_tokens=200_000)
    args2.ignore_calibration = True
    rc2 = bu.run_job(args2, started_at=9400.0)
    assert rc2 in (0,)  # proceeds (and may pause later on budget, both fine)


def test_H2_per_prompt_pairing_width(state_file: Path):
    """Verify pairs per-(seed, prompt): with seeds=2 and the 4-prompt
    mini-slate default, each technique gets n_on = n_off = 8 paired samples —
    not the old n=seeds=2."""
    for ph in ("baseline", "research", "implement", "verify"):
        assert bu.run_job(_args(state_file, phase=ph), started_at=9500.0) == 0
    data = json.loads(state_file.read_text())
    sig_arts = [a for a in data["phases"]["verify"]["artifacts"]
                if "verify_significance_" in a]
    assert sig_arts
    sig = json.loads(Path(sig_arts[0]).read_text())
    assert sig, "no techniques in significance artifact"
    for tech, rec in sig.items():
        assert rec["n_on"] == 8, (tech, rec["n_on"])   # 2 seeds x 4 prompts
        assert rec["n_off"] == 8
        assert "p_value" in rec and "cohens_d" in rec and "bootstrap_ci" in rec


def test_C2_bench_arm_sets_feature_override_env(monkeypatch, tmp_path: Path):
    """The live bench branch toggles BLITZ_FEATURE_OVERRIDES per arm and
    restores the env afterwards (C2 — arms must actually differ)."""
    import asyncio as _asyncio
    seen: dict[str, str | None] = {}

    async def _fake_run_bench(cfg):
        seen["env"] = os.environ.get("BLITZ_FEATURE_OVERRIDES")
        seen["filter_ids"] = tuple(cfg.slate_filter_ids)
        d = tmp_path / "fake_run"
        d.mkdir(exist_ok=True)
        return d

    import bench.runner as _br
    monkeypatch.setattr(_br, "run_bench", _fake_run_bench)
    monkeypatch.setattr(bu, "_read_bench_summary", lambda run_dir: {
        "functional_composite": {"mean": 0.7, "per_prompt": {"p1": 0.7}},
    })
    import os
    os.environ.pop("BLITZ_FEATURE_OVERRIDES", None)

    tracker = bu.BudgetTracker(hard_ceiling=10_000, dry_run=True)
    res = bu._invoke_live("bench_ablation", tracker=tracker,
                          technique="my-mech", arm="on", seed=1,
                          slate="bench/slate_upgrade.toml",
                          prompt_ids=["p1", "p2"], max_rounds=1)
    assert json.loads(seen["env"]) == {"my-mech": True}
    assert seen["filter_ids"] == ("p1", "p2")
    assert os.environ.get("BLITZ_FEATURE_OVERRIDES") is None  # restored
    assert res["result"]["per_prompt"] == {"p1": 0.7}


# ===========================================================================
# Live-outage hardening (2026-06-10): zero-signature + circuit breaker
# ===========================================================================


def test_consensus_zero_signature_is_errored(monkeypatch, tmp_path: Path):
    """A swarm run that force-finalizes with tokens=0 + quality=0 (all agent
    calls errored) must come back ok=False so C4 retries/excludes it — never
    recorded as a real 0.0 sample. (Live: 14/20 cells silently zeroed.)"""
    import sys as _sys
    import types as _types

    async def _fake_run_swarm(topic, **kw):
        return tmp_path / "out.md"

    orch = _sys.modules.get("orchestrator") or _types.ModuleType("orchestrator")
    monkeypatch.setattr(orch, "run_swarm", _fake_run_swarm, raising=False)
    _sys.modules["orchestrator"] = orch

    tracker = bu.BudgetTracker(hard_ceiling=10_000, dry_run=True)

    # No metrics record at all -> zero signature -> errored.
    monkeypatch.setattr(bu, "_read_metrics_record_for_topic", lambda *a, **k: None)
    res = bu._invoke_live("consensus", tracker=tracker, topic="t", seed=0)
    assert res["ok"] is False
    assert "no usable output" in res["error"]

    # Record with real tokens + quality -> healthy.
    monkeypatch.setattr(bu, "_read_metrics_record_for_topic",
                        lambda *a, **k: {"total_tokens": 5000, "avg_quality": 7.1})
    monkeypatch.setattr(bu, "_tokens_of", lambda rec: 5000)
    res2 = bu._invoke_live("consensus", tracker=tracker, topic="t", seed=0)
    assert res2["ok"] is True and res2["result"]["avg_quality"] == 7.1


def test_circuit_breaker_pauses_after_consecutive_failures(state_file: Path, monkeypatch):
    """3 consecutive cell FAILURES (retries exhausted) trip the breaker: the
    phase PAUSES instead of burning every remaining cell into cells_failed.
    (Live: a claude outage zeroed 14 cells in 10 minutes.)"""
    real_invoke = bu._invoke

    def _backend_down(kind, **kw):
        res = real_invoke(kind, **kw)
        if kind == "single_agent":  # every floor cell fails, repeatedly
            bad = dict(res)
            bad["ok"] = False
            bad["error"] = "backend down"
            return bad
        return res

    monkeypatch.setattr(bu, "_invoke", _backend_down)
    assert bu.run_job(_args(state_file, phase="baseline"), started_at=9600.0) == 0
    data = json.loads(state_file.read_text())
    assert data["phases"]["baseline"]["status"] == bu.STATUS_PAUSED
    failed = data["phases"]["baseline"].get("cells_failed", [])
    assert len(failed) == 3  # breaker tripped before the 4th could burn
    assert "floor:k3" not in failed and "floor:k4" not in failed
