"""mythos-bench library — extraction, verification, result management.

Single-file utility module called by bench.py. Keeps the harness small.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "mythos-bench"
TASKS_DIR = BENCH_DIR / "tasks"
ADVERSARIAL_DIR = BENCH_DIR / "adversarial"
ARTIFACTS_DIR = BENCH_DIR / "artifacts"
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_LOG = RESULTS_DIR / "runs.jsonl"


# ───────────────────────────────────────────────────────────────────────────
# Task spec
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class TaskSpec:
    id: str
    title: str
    prompt: str
    deliverable_files: list[str]
    verification_target: str
    expected_min_pass: int
    adversarial_file: str | None = None  # path under adversarial/
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, task_id: str) -> "TaskSpec":
        path = TASKS_DIR / f"{task_id}.toml"
        if not path.exists():
            raise FileNotFoundError(f"task spec not found: {path}")
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        task = raw["task"]
        deliv = raw.get("deliverables", {})
        verif = raw.get("verification", {})
        adv = verif.get("adversarial", {})
        return cls(
            id=task["id"],
            title=task.get("title", task["id"]),
            prompt=task["prompt"].strip(),
            deliverable_files=list(deliv.get("files", [])),
            verification_target=verif.get("target", ""),
            expected_min_pass=int(verif.get("expected_min_pass", 1)),
            adversarial_file=adv.get("file"),
            raw=raw,
        )


def list_tasks() -> list[TaskSpec]:
    return [TaskSpec.load(p.stem) for p in sorted(TASKS_DIR.glob("*.toml"))]


# ───────────────────────────────────────────────────────────────────────────
# Run dispatch — invoke the orchestrator
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class RunInfo:
    task_id: str
    mode: str                # "consensus" | "mythos"
    started_at: str
    finished_at: str
    exit_code: int
    cli_stdout_path: Path
    run_dir: Path | None     # mythos: output/mythos/<slug>_<ts>/; consensus: parent of saved md
    output_path: Path | None # consensus: the final .md; mythos: final_artifact.md


def run_task(task: TaskSpec, mode: str, *, extra_args: list[str] | None = None) -> RunInfo:
    """Invoke `orchestrator.py --mode <mode>` and capture run metadata.

    Returns RunInfo even on failure — caller can decide what to do with a bad
    exit code.
    """
    if mode not in ("consensus", "mythos"):
        raise ValueError(f"unknown mode: {mode}")

    started_at = datetime.now().isoformat(timespec="seconds")
    log_dir = ARTIFACTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}__{mode}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    cmd = [
        sys.executable, "orchestrator.py",
        "--mode", mode,
        "--no-redis",
        task.prompt,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n[bench] {task.id} | mode={mode} | starting {started_at}")
    print(f"[bench] log → {log_path}")

    with open(log_path, "wb") as f:
        proc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            cwd=REPO_ROOT, timeout=1500,
        )

    finished_at = datetime.now().isoformat(timespec="seconds")
    log_text = log_path.read_text()

    run_dir = _detect_run_dir(mode, log_text)
    output_path = _detect_output_path(mode, log_text)

    return RunInfo(
        task_id=task.id,
        mode=mode,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=proc.returncode,
        cli_stdout_path=log_path,
        run_dir=run_dir,
        output_path=output_path,
    )


_CONSENSUS_OUTPUT_RE = re.compile(r"OUTPUT SAVED:\s*(.+\.md)\s*$", re.MULTILINE)
_MYTHOS_RUN_DIR_RE = re.compile(r"output:\s*(.+/output/mythos/[^\s]+)\s*$", re.MULTILINE)


def _detect_run_dir(mode: str, log_text: str) -> Path | None:
    if mode == "mythos":
        m = _MYTHOS_RUN_DIR_RE.search(log_text)
        return Path(m.group(1)) if m else None
    if mode == "consensus":
        m = _CONSENSUS_OUTPUT_RE.search(log_text)
        return Path(m.group(1)).parent if m else None
    return None


def _detect_output_path(mode: str, log_text: str) -> Path | None:
    if mode == "mythos":
        rd = _detect_run_dir(mode, log_text)
        return rd / "final_artifact.md" if rd else None
    if mode == "consensus":
        m = _CONSENSUS_OUTPUT_RE.search(log_text)
        return Path(m.group(1)) if m else None
    return None


# ───────────────────────────────────────────────────────────────────────────
# Cost / wall / verdict extraction
# ───────────────────────────────────────────────────────────────────────────

def parse_metrics_from_log(log_text: str, mode: str) -> dict:
    """Best-effort metrics extraction from CLI stdout."""
    out = {
        "cost_usd": None,
        "wall_clock_s": None,
        "rounds": None,
        "invocations": None,
        "verifier_verdict": None,
        "consensus_reached": None,
    }
    if mode == "mythos":
        # "Cost: ${budget:.4f}", "wall: {sec:.1f}s", verdict from final summary
        m = re.search(r"cost:\s*\$([\d.]+)", log_text)
        if m: out["cost_usd"] = float(m.group(1))
        m = re.search(r"wall:\s*([\d.]+)s", log_text)
        if m: out["wall_clock_s"] = float(m.group(1))
        m = re.search(r"rounds:\s*(\d+)", log_text)
        if m: out["rounds"] = int(m.group(1))
        m = re.search(r"verdict=(pass|needs_work)", log_text)
        if m: out["verifier_verdict"] = m.group(1)
        # Mythos invocations = 1 planner + N executors + 1 verifier per round
        # Approx via plan rounds × (1 + executors + 1)
        out["invocations"] = log_text.count("Planner done") + log_text.count("Executors done") + log_text.count("Verifier done")
        if "MYTHOS RUN COMPLETE — status: passed" in log_text:
            out["consensus_reached"] = True
        elif "MYTHOS RUN COMPLETE — status:" in log_text:
            out["consensus_reached"] = False
    else:  # consensus
        m = re.search(r"Cost:\s*\$([\d.]+)", log_text)
        if m: out["cost_usd"] = float(m.group(1))
        m = re.search(r"Wall clock:\s*([\d.]+)s", log_text)
        if m: out["wall_clock_s"] = float(m.group(1))
        m = re.search(r"Rounds:\s*(\d+)", log_text)
        if m: out["rounds"] = int(m.group(1))
        m = re.search(r"Total agent invocations:\s*(\d+)", log_text)
        if m: out["invocations"] = int(m.group(1))
        m = re.search(r"Consensus:\s*(yes|no)", log_text)
        if m: out["consensus_reached"] = (m.group(1) == "yes")
    return out


# ───────────────────────────────────────────────────────────────────────────
# Deliverable extraction
# ───────────────────────────────────────────────────────────────────────────

_PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)


def extract_deliverables(task: TaskSpec, run: RunInfo, dest_dir: Path) -> dict[str, Path]:
    """Pull deliverable files out of the run output into dest_dir.

    Mythos: one ```python block per executor_*.md (1:1 with deliverable_files).
    Consensus: blocks live inside the final synthesized .md; we take the
        first N blocks where N == len(deliverable_files).

    Returns {filename: path_written}.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    if run.mode == "mythos" and run.run_dir and run.run_dir.exists():
        executors = sorted(run.run_dir.glob("executor_*_spec_*.md"))
        if len(executors) < len(task.deliverable_files):
            return results
        for fname, exec_md in zip(task.deliverable_files, executors):
            text = exec_md.read_text()
            blocks = _PYTHON_BLOCK_RE.findall(text)
            if not blocks:
                continue
            (dest_dir / fname).write_text(blocks[0])
            results[fname] = dest_dir / fname
        return results

    if run.mode == "consensus" and run.output_path and run.output_path.exists():
        text = run.output_path.read_text()
        blocks = _PYTHON_BLOCK_RE.findall(text)
        if len(blocks) < len(task.deliverable_files):
            return results
        for fname, block in zip(task.deliverable_files, blocks):
            (dest_dir / fname).write_text(block)
            results[fname] = dest_dir / fname
        return results

    return results


# ───────────────────────────────────────────────────────────────────────────
# Verification
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class PytestResult:
    passed: int
    failed: int
    errors: int
    raw_tail: str

    def ok(self) -> bool:
        return self.failed == 0 and self.errors == 0 and self.passed > 0


_PYTEST_SUMMARY_RE = re.compile(
    r"(\d+) passed|(\d+) failed|(\d+) error",
)


def run_pytest(directory: Path, target: str) -> PytestResult:
    """Run pytest in directory; return parsed result."""
    if not (directory / target).exists():
        return PytestResult(passed=0, failed=0, errors=1, raw_tail=f"target not found: {target}")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", target],
        capture_output=True, text=True, cwd=directory, timeout=120,
    )
    out = proc.stdout + proc.stderr
    passed = failed = errors = 0
    summary_line = ""
    for line in out.splitlines():
        if "passed" in line or "failed" in line or "error" in line:
            summary_line = line
            for m in re.finditer(r"(\d+)\s+(passed|failed|error)", line):
                n, kind = int(m.group(1)), m.group(2)
                if kind == "passed":
                    passed = n
                elif kind == "failed":
                    failed = n
                elif kind == "error":
                    errors = n
    tail = "\n".join(out.splitlines()[-15:])
    return PytestResult(passed=passed, failed=failed, errors=errors, raw_tail=tail)


def verify_run(task: TaskSpec, run: RunInfo, *, run_adversarial: bool = True) -> dict:
    """Extract deliverables and run pytest (+ adversarial pytest if configured)."""
    workdir = ARTIFACTS_DIR / f"{task.id}__{run.mode}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    workdir.mkdir(parents=True, exist_ok=True)

    extracted = extract_deliverables(task, run, workdir)
    primary = run_pytest(workdir, task.verification_target) if extracted else PytestResult(0, 0, 1, "no deliverables extracted")

    adv = None
    if run_adversarial and task.adversarial_file and extracted:
        src = ADVERSARIAL_DIR / task.adversarial_file
        if src.exists():
            shutil.copy(src, workdir / src.name)
            adv = run_pytest(workdir, src.name)

    return {
        "workdir": str(workdir),
        "extracted_files": [str(p) for p in extracted.values()],
        "primary_pytest": {
            "passed": primary.passed,
            "failed": primary.failed,
            "errors": primary.errors,
            "ok": primary.ok(),
            "tail": primary.raw_tail,
        },
        "adversarial_pytest": (
            {
                "passed": adv.passed,
                "failed": adv.failed,
                "errors": adv.errors,
                "ok": adv.ok(),
                "tail": adv.raw_tail,
            }
            if adv else None
        ),
    }


# ───────────────────────────────────────────────────────────────────────────
# Result logging
# ───────────────────────────────────────────────────────────────────────────

def append_result(record: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_LOG, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_results() -> list[dict]:
    if not RESULTS_LOG.exists():
        return []
    out = []
    for line in RESULTS_LOG.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ───────────────────────────────────────────────────────────────────────────
# Verdict synthesis — the headline "did this run actually work" call
# ───────────────────────────────────────────────────────────────────────────

def synthesize_verdict(task: TaskSpec, run_metrics: dict, verify_data: dict) -> str:
    """One-word headline for the comparison report.

    - "passed" — primary pytest ok AND (no adversarial OR adversarial ok)
    - "verifier_wrong" — verifier said pass but primary or adversarial failed
                          (only meaningful for mythos)
    - "tests_failed" — primary pytest failed
    - "adversarial_caught" — primary ok but adversarial caught a real bug
                              (only meaningful when adversarial configured)
    - "no_deliverable" — couldn't extract deliverables
    - "errored" — orchestrator exited nonzero
    """
    primary_ok = verify_data["primary_pytest"]["ok"]
    primary_passed = verify_data["primary_pytest"]["passed"]
    adv_data = verify_data.get("adversarial_pytest")
    adv_ok = adv_data["ok"] if adv_data else None

    if not verify_data["extracted_files"]:
        return "no_deliverable"

    if not primary_ok:
        if run_metrics.get("verifier_verdict") == "pass":
            return "verifier_wrong"
        return "tests_failed"

    if primary_passed < task.expected_min_pass:
        return "tests_under_min"

    if adv_data is not None and not adv_ok:
        if run_metrics.get("verifier_verdict") == "pass":
            return "verifier_wrong"
        return "adversarial_caught"

    return "passed"
