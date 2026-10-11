"""Mythos Verifier — gates the swarm's work.

Reads the original task, the planner's plan, and all executor outputs.
Returns a structured verdict (pass | needs_work) with per-invariant and
per-spec breakdowns plus required_fixes the planner can act on.

Two gates sit on top of the LLM verdict (backends-04), so a confident-sounding
model never passes a round by itself:

1. **Mechanical check** — when any deliverable contains a test file
   (`test_*.py` / `*_test.py`, extracted from fenced code blocks that name
   their path), every extracted file is written to a temp dir and the tests
   run there in a subprocess (`python -m pytest -q`, falling back to
   `unittest`). A failing or timed-out run forces `needs_work`.
2. **Confidence gate** — a `pass` whose self-reported confidence is below
   `MythosConfig.verifier_min_confidence` is demoted to `needs_work`.

Both are recorded in `Verification.mechanical` / `gate_notes` and surface in
the run artifacts, so a demotion is always explainable.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import _invoke
from .policies import MythosConfig
from .schemas import VERIFIER_SCHEMA_JSON

_TEST_FILE = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py)$")
# A fenced block whose info string or the line(s) right before it name a path.
_FENCE = re.compile(
    r"(?P<pre>(?:^(?!```)[^\n]*\n){0,2})"
    r"^```(?P<info>[^\n]*)\n(?P<body>.*?)^```[ \t]*$",
    re.M | re.S,
)
_PATH = re.compile(r"(?<![\w./-])((?:[\w.-]+/)*[\w.-]+\.(?:py|txt|toml|cfg|ini|json|md|sh))(?![\w/])")


@dataclass
class Verification:
    verdict: str                       # "pass" | "needs_work"
    invariant_results: list[dict]
    spec_results: list[dict]
    required_fixes: list[str]
    summary: str
    confidence: float
    cost_usd: float | None = None     # None = provider did not report cost
    tokens: int = 0
    elapsed_s: float = 0.0
    raw: dict = field(default_factory=dict)
    error: str | None = None
    llm_verdict: str = ""              # verdict before the gates
    mechanical: dict = field(default_factory=dict)   # run_artifact_tests() result
    gate_notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "llm_verdict": self.llm_verdict,
            "invariant_results": self.invariant_results,
            "spec_results": self.spec_results,
            "required_fixes": self.required_fixes,
            "summary": self.summary,
            "confidence": self.confidence,
            "cost_usd": None if self.cost_usd is None else round(self.cost_usd, 4),
            "elapsed_s": round(self.elapsed_s, 1),
            "error": self.error,
            "mechanical": self.mechanical,
            "gate_notes": self.gate_notes,
        }


# ---------------------------------------------------------------------------
# Mechanical check
# ---------------------------------------------------------------------------


def extract_files(deliverable: str) -> dict[str, str]:
    """Pull `{path: content}` out of fenced code blocks that name their file.

    Recognised: a path in the fence info string (```python test_x.py,
    ```title=test_x.py) or in the one/two lines before the fence
    (`### test_x.py`, `**File: tests/test_x.py**`, `# file: test_x.py`).
    """
    files: dict[str, str] = {}
    for m in _FENCE.finditer(deliverable or ""):
        candidates = _PATH.findall(m.group("info")) + _PATH.findall(m.group("pre"))
        if not candidates:
            continue
        raw = candidates[0]
        if raw.startswith("/") or ".." in raw.split("/"):
            continue                              # never escape the temp dir
        files[raw.lstrip("./")] = m.group("body")
    return files


def run_artifact_tests(executor_outputs: list, *, timeout_s: int = 120) -> dict:
    """Run tests shipped inside the deliverables. Returns a result dict:

    ``{"ran": bool, "passed": bool|None, "returncode": int|None,
       "files": [..], "test_files": [..], "output_tail": str, "reason": str}``

    `ran` is False (and `passed` None) when no test file was found — that is
    "not checked", never "clean".
    """
    files: dict[str, str] = {}
    for o in executor_outputs:
        files.update(extract_files(getattr(o, "deliverable", "") or ""))
    test_files = sorted(p for p in files if _TEST_FILE.search(p))
    result = {"ran": False, "passed": None, "returncode": None,
              "files": sorted(files), "test_files": test_files,
              "output_tail": "", "reason": ""}
    if not test_files:
        result["reason"] = "no test file in deliverables"
        return result

    tmp = Path(tempfile.mkdtemp(prefix="mythos-verify-"))
    try:
        for rel, body in files.items():
            dest = tmp / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
        try:
            import pytest  # noqa: F401
            cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *test_files]
        except ImportError:
            cmd = [sys.executable, "-m", "unittest", "discover", "-p", "test_*.py"]
        try:
            proc = subprocess.run(
                cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout_s,
                env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            result.update(ran=True, passed=False, returncode=None,
                          reason=f"artifact tests timed out after {timeout_s}s")
            return result
        tail = (proc.stdout + proc.stderr)[-2000:]
        result.update(ran=True, passed=proc.returncode == 0, returncode=proc.returncode,
                      output_tail=tail,
                      reason="artifact tests passed" if proc.returncode == 0
                      else f"artifact tests failed (rc={proc.returncode})")
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def apply_gates(verification: Verification, config: MythosConfig, *, mechanical: dict | None) -> Verification:
    """Demote the LLM verdict when the mechanical check or the confidence gate fails."""
    verification.llm_verdict = verification.llm_verdict or verification.verdict
    if mechanical is not None:
        verification.mechanical = mechanical
        if mechanical.get("ran"):
            if mechanical.get("passed"):
                verification.gate_notes.append(
                    f"mechanical: {mechanical['reason']} ({', '.join(mechanical['test_files'])})")
            else:
                verification.gate_notes.append(f"mechanical: {mechanical['reason']} -> needs_work")
                verification.verdict = "needs_work"
                verification.required_fixes.append(
                    f"Mechanical check: {mechanical['reason']}. Output tail:\n"
                    f"{mechanical.get('output_tail', '')}"
                )
        else:
            verification.gate_notes.append(f"mechanical: not run ({mechanical.get('reason', '')})")

    floor = float(getattr(config, "verifier_min_confidence", 0.0) or 0.0)
    if verification.verdict == "pass" and verification.confidence < floor:
        verification.gate_notes.append(
            f"confidence {verification.confidence:.2f} < floor {floor:.2f} -> needs_work")
        verification.verdict = "needs_work"
        verification.required_fixes.append(
            f"Verifier confidence {verification.confidence:.2f} is below the "
            f"{floor:.2f} floor; strengthen evidence per sub-spec before re-verifying."
        )
    return verification


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------


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

    # Mechanical check first: its result is also handed to the LLM verifier
    # as evidence (it must not contradict a red test run).
    mechanical = None
    if getattr(config, "verifier_run_tests", False):
        mechanical = run_artifact_tests(
            executor_outputs, timeout_s=int(getattr(config, "verifier_test_timeout_s", 120)))

    user = (
        "## Original task\n"
        f"{task}\n\n"
        "## Plan\n"
        f"{json.dumps(plan_view, indent=2)}\n\n"
        "## Executor outputs\n"
        f"{json.dumps(exec_view, indent=2)}\n\n"
        + (
            "## Mechanical check (the artifact's own tests, run in a subprocess)\n"
            f"{json.dumps({k: mechanical[k] for k in ('ran', 'passed', 'reason', 'test_files', 'output_tail')}, indent=2)}\n\n"
            if mechanical is not None else ""
        )
        + "## Your job\n"
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
        model_role="verifier",
        tier=config.verifier_model,
        timeout_s=config.verifier_timeout_s,
        mode_backend=config.backend or None,
    )

    if res.parsed is None:
        # Verifier failure is treated as needs_work — better safe than wrong.
        return apply_gates(Verification(
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
            tokens=res.tokens,
            elapsed_s=res.elapsed_s,
            error=res.error,
        ), config, mechanical=mechanical)

    parsed = res.parsed
    return apply_gates(Verification(
        verdict=parsed.get("verdict", "needs_work"),
        invariant_results=list(parsed.get("invariant_results", [])),
        spec_results=list(parsed.get("spec_results", [])),
        required_fixes=list(parsed.get("required_fixes", [])),
        summary=parsed.get("summary", ""),
        confidence=float(parsed.get("confidence", 0.0)),
        cost_usd=res.cost_usd,
        tokens=res.tokens,
        elapsed_s=res.elapsed_s,
        raw=parsed,
    ), config, mechanical=mechanical)
