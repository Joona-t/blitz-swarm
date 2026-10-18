"""WP4 (v0.3.0): bench / evolve gate / upgrade-job p-value / contrib move /
cascade_guard LLM hooks / mythos verifier gates.

No LLM is ever invoked: backends are fakes, `_invoke.invoke` is monkeypatched.
The two verifier tests that run the artifact's own tests spawn a child
python (pytest) on a temp dir — marked `spawns_subprocess`.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import stats as bstats
from bench import runner as brunner
from config import BlitzConfig, ConfigError, EvolveConfig, GUARD_MODES, load_config
from mechanisms import cascade_guard as cg

PY = sys.executable

_CFG_TOML = """
[models]
default = "standard"
guard = "fast"

[backend]
default = "alpha"

[backend.providers.alpha]
adapter = "claude_cli"
[backend.providers.alpha.tiers]
fast = "alpha-fast"
standard = "alpha-std"
deep = "alpha-deep"
[backend.providers.alpha.effort]
fast = "low"

[guard]
enabled = true
mode = "{mode}"
"""


def _cfg(tmp_path: Path, mode: str = "balanced") -> BlitzConfig:
    p = tmp_path / "blitz.toml"
    p.write_text(_CFG_TOML.format(mode=mode), encoding="utf-8")
    return load_config(p)


# ===========================================================================
# bench/stats.py — bench-03: no normal approximation below df=30
# ===========================================================================


@pytest.mark.parametrize("t,df,p_expected", [
    (12.706, 1, 0.05), (4.303, 2, 0.05), (2.776, 4, 0.05), (2.262, 9, 0.05),
    (2.045, 29, 0.05), (63.657, 1, 0.01), (3.250, 9, 0.01),
])
def test_t_two_sided_p_matches_t_table_for_small_df(t, df, p_expected):
    """Critical values from the standard two-sided t table."""
    assert bstats.t_two_sided_p(t, df) == pytest.approx(p_expected, abs=2e-4)


def test_small_df_is_not_the_normal_approximation():
    """At df=3, |t|=2.5: t gives p~0.088 (not significant); the normal
    approximation claims p~0.012. The old code used the latter."""
    p_t = bstats.t_two_sided_p(2.5, 3)
    p_norm = 2.0 * (1.0 - bstats._normal_cdf(2.5))
    assert p_t > 0.05 > p_norm
    assert p_t == pytest.approx(0.0877, abs=2e-3)


def test_normal_approximation_only_at_or_above_30_df():
    assert bstats.NORMAL_APPROX_MIN_DF == 30
    p_norm = 2.0 * (1.0 - bstats._normal_cdf(2.0))
    assert bstats.t_two_sided_p(2.0, 30) == pytest.approx(p_norm)
    assert bstats.t_two_sided_p(2.0, 29) != pytest.approx(p_norm)


def test_t_two_sided_p_edge_cases():
    assert bstats.t_two_sided_p(0.0, 5) == pytest.approx(1.0)
    assert bstats.t_two_sided_p(float("inf"), 5) == 0.0
    assert bstats.t_two_sided_p(1.0, 0) == 1.0
    assert 0.0 <= bstats.t_two_sided_p(-3.0, 4) <= 1.0
    assert bstats.t_two_sided_p(-3.0, 4) == pytest.approx(bstats.t_two_sided_p(3.0, 4))


def test_paired_t_test_uses_exact_t_without_scipy():
    """n=4 strong effect: exact t gives a tiny but non-zero p (the value that
    `or 1` in blitz_upgrade used to flip to 1.0 when it underflowed)."""
    t, p, df = bstats.paired_t_test([8.0, 8.2, 8.1, 8.3], [7.0, 7.1, 6.9, 7.0])
    assert df == 3 and t > 10
    assert 0.0 < p < 0.001


# ===========================================================================
# jobs/blitz_upgrade.py — BUG-003: p=0.0 is significant, None is not
# ===========================================================================


def test_decision_rule_treats_p_zero_as_significant_and_none_as_not():
    import jobs.blitz_upgrade as bu

    base = {"mean_on": 8.0, "mean_off": 7.0, "delta": 1.0, "cohens_d": 1.5,
            "bootstrap_ci": [0.5, 1.5]}
    kw = dict(baseline_floor=7.0, keep_band=0.5)
    assert bu._decide_one_technique({**base, "p_value": 0.0}, **kw)["keep"] is True
    assert bu._decide_one_technique({**base, "p_value": 0.01}, **kw)["keep"] is True
    none_case = bu._decide_one_technique({**base, "p_value": None}, **kw)
    assert none_case["keep"] is False and none_case["p_value"] == 1.0


# ===========================================================================
# bench/runner.py — bench-01 contract + bench-03 report + bench-04 version
# ===========================================================================


def _rounds_clean() -> list[list[dict]]:
    # Two rounds, two agents each, well-formed outputs (no MAST trigger).
    return [[{"agent_id": f"r0{i}", "role": "researcher", "findings": "a" * 300,
              "vote": "ready", "confidence": 0.9, "_round": r}
             for i in range(2)] for r in range(2)]


def test_split_swarm_result_accepts_path_or_tuple(tmp_path):
    p = tmp_path / "o.md"
    assert brunner._split_swarm_result(p) == (p, [])
    rounds = _rounds_clean()
    assert brunner._split_swarm_result((p, rounds)) == (p, rounds)
    assert brunner._split_swarm_result((p, "garbage")) == (p, [])


def _metrics_record(metrics_path: Path, topic: str) -> None:
    import time

    metrics_path.write_text(json.dumps({
        "run_id": "r1", "topic": topic, "timestamp": time.time() + 1,
        "consensus_reached": True, "rounds_to_consensus": 1,
        "coverage": 8, "accuracy": 8, "clarity": 8, "depth": 8, "avg_quality": 8.0,
        "total_cost_usd": 0.1, "total_input_tokens": 10, "total_output_tokens": 5,
    }) + "\n")


@pytest.mark.asyncio
async def test_run_one_prompt_runs_detectors_only_with_per_agent_outputs(tmp_path, slate_path, capsys):
    from bench import load_slate

    prompt = load_slate(slate_path).prompts[0]
    cfg = brunner.BenchConfig(slate_path=slate_path, max_rounds=1)
    metrics_path = tmp_path / "metrics.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    async def path_only(topic, **kw):
        out = tmp_path / "a.md"
        out.write_text("# a\n\nfindings\n", encoding="utf-8")
        _metrics_record(metrics_path, topic)
        return out

    async def with_rounds(topic, **kw):
        out = tmp_path / "b.md"
        out.write_text("# b\n\nfindings\n", encoding="utf-8")
        _metrics_record(metrics_path, topic)
        return out, _rounds_clean()

    r1 = await brunner.run_one_prompt(prompt, cfg, run_dir, swarm_fn=path_only, metrics_path=metrics_path)
    assert r1.mast_detectors_ran is False and r1.mast_flags == []
    assert "MAST detectors not run" in capsys.readouterr().out

    r2 = await brunner.run_one_prompt(prompt, cfg, run_dir, swarm_fn=with_rounds, metrics_path=metrics_path)
    assert r2.mast_detectors_ran is True
    assert isinstance(r2.mast_flags, list)

    # A sibling <output>.rounds.json (what the orchestrator writes) also counts.
    async def sibling(topic, **kw):
        out = tmp_path / "c.md"
        out.write_text("# c\n", encoding="utf-8")
        out.with_suffix(".rounds.json").write_text(json.dumps(_rounds_clean()))
        _metrics_record(metrics_path, topic)
        return out

    r3 = await brunner.run_one_prompt(prompt, cfg, run_dir, swarm_fn=sibling, metrics_path=metrics_path)
    assert r3.mast_detectors_ran is True


def _summary(**over) -> dict:
    s = {
        "run_id": "run-x", "started_utc": "x", "ended_utc": "y",
        "slate": {"slate_id": "v1", "sha256": "0" * 64, "prompts_run": 2, "prompt_count": 2,
                  "prompts_timed_out": 0, "prompts_errored": 0},
        "aggregate_quality": {"avg": {"mean": 7.0, "stddev": 0.5, "ci95": [6.5, 7.5]}},
        "by_tier": {"easy": {"n": 2, "avg_quality": 7.0, "avg_cost": 0.05}},
        "mast_flags_summary": {},
        "mast_detectors_ran": 0,
        "cost": {"total_usd": 0.1, "input_tokens_total": 1, "output_tokens_total": 1},
        "functional_composite": {"mean": 0.5, "per_prompt": {}, "missing": []},
    }
    s.update(over)
    return s


def test_stats_md_distinguishes_not_run_from_clean(tmp_path):
    d = tmp_path / "r"; d.mkdir()
    text = brunner.write_stats_md(d, _summary(mast_detectors_ran=0)).read_text()
    assert "MAST detectors not run (no per-agent outputs)" in text
    assert "No MAST flags raised" not in text
    text = brunner.write_stats_md(d, _summary(mast_detectors_ran=3)).read_text()
    assert "No MAST flags raised (detectors ran on 3 prompt(s))" in text
    text = brunner.write_stats_md(d, _summary(mast_flags_summary={"FM-2.4": 1})).read_text()
    assert "| FM-2.4 | 1 |" in text


def test_aggregate_counts_detector_rows(tmp_path, slate_path):
    from bench import load_slate

    slate = load_slate(slate_path)
    d = tmp_path / "r"; d.mkdir()
    pid = slate.prompts[0].id
    rows = [
        {"prompt_id": pid, "quality": {"avg": 7.0}, "cost_usd": 0.1, "mast_flags": [],
         "mast_detectors_ran": True, "functional_composite": {"composite": 0.5}},
        {"prompt_id": pid, "quality": {"avg": 7.0}, "cost_usd": 0.1, "mast_flags": ["FM-1.1"],
         "mast_detectors_ran": False, "functional_composite": {"composite": 0.4}},
    ]
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = brunner.aggregate(d, slate, brunner.BenchConfig(slate_path=slate_path))
    assert summary["mast_detectors_ran"] == 1
    assert summary["mast_flags_summary"] == {"FM-1.1": 1}


def test_paired_comparison_wires_t_test_and_cohens_d(tmp_path):
    mine = _summary(functional_composite={"mean": 0.8, "per_prompt": {"p1": 0.8, "p2": 0.9, "p3": 0.85, "p4": 0.95}, "missing": []})
    base = _summary(run_id="base", functional_composite={"mean": 0.6, "per_prompt": {"p1": 0.6, "p2": 0.7, "p3": 0.65, "p4": 0.7, "p9": 0.1}, "missing": []})
    cmp = brunner.paired_comparison(mine, base)
    assert cmp["n_pairs"] == 4 and cmp["prompt_ids"] == ["p1", "p2", "p3", "p4"]
    assert cmp["mean_delta"] == pytest.approx(0.2125, abs=1e-4)
    assert cmp["t_stat"] > 0 and cmp["df"] == 3 and 0 < cmp["p_value"] < 0.05
    assert cmp["cohens_d"] > 1
    assert cmp["p_method"]
    d = tmp_path / "r"; d.mkdir()
    text = brunner.write_stats_md(d, mine, baseline_summary=base).read_text()
    assert "## Paired comparison vs baseline `base`" in text
    assert "Cohen's d_z" in text and "Paired t" in text


def test_paired_comparison_refuses_below_two_pairs():
    mine = _summary(functional_composite={"mean": 0.8, "per_prompt": {"p1": 0.8}, "missing": []})
    base = _summary(functional_composite={"mean": 0.6, "per_prompt": {"p1": 0.6}, "missing": []})
    cmp = brunner.paired_comparison(mine, base)
    assert cmp["p_value"] is None and "fewer than 2" in cmp["reason"]


def test_version_comes_from_pyproject_not_a_literal():
    import tomllib

    want = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert brunner._blitz_swarm_version() == want
    src = (ROOT / "bench" / "runner.py").read_text()
    assert '"blitz_swarm_version": _blitz_swarm_version()' in src


# ===========================================================================
# config.py — [evolve] gate + [guard] mode validation (cross_edit)
# ===========================================================================


def test_config_evolve_section_defaults_and_parse(tmp_path):
    assert BlitzConfig().evolve == EvolveConfig(enabled=False, gepa_max_metric_calls=1500, aflow_max_iters=50)
    p = tmp_path / "b.toml"
    p.write_text("[evolve]\nenabled = true\naflow_max_iters = 7\n")
    c = load_config(p)
    assert c.evolve.enabled is True and c.evolve.aflow_max_iters == 7
    p.write_text("[evolve]\nbogus = 1\n")
    with pytest.raises(ConfigError, match=r"\[evolve\]\.bogus"):
        load_config(p)


def test_config_guard_mode_validated(tmp_path):
    assert "max" in GUARD_MODES
    p = tmp_path / "b.toml"
    p.write_text('[guard]\nmode = "max"\n')
    assert load_config(p).guard.mode == "max"
    p.write_text('[guard]\nmode = "ludicrous"\n')
    with pytest.raises(ConfigError, match="ludicrous"):
        load_config(p)


def test_shipped_blitz_toml_has_evolve_disabled():
    assert load_config().evolve.enabled is False


# ===========================================================================
# mechanisms/cascade_guard.py — mechanisms-02
# ===========================================================================


@dataclass
class _FakeResult:
    parsed: dict | None
    errored: bool = False
    error: str | None = None


class _FakeBackend:
    """Scripted guard backend: decompose -> given claims; adjudicate -> given verdict."""

    def __init__(self, claims, verdict="green", errored=False):
        self.claims, self.verdict, self.errored = claims, verdict, errored
        self.calls = []

    def call(self, call):
        self.calls.append(call)
        if self.errored:
            return _FakeResult(parsed=None, errored=True, error="boom")
        if call.role == "guard_decompose":
            return _FakeResult(parsed={"claims": self.claims})
        return _FakeResult(parsed={"verdict": self.verdict, "reason": "scripted"})


def _out(agent="r01", role="researcher", findings="The sky is blue. Water is wet."):
    return {"agent_id": agent, "role": role, "findings": findings, "vote": "ready", "confidence": 0.9}


def test_max_mode_uses_llm_hooks_with_resolved_guard_tier(tmp_path, capsys):
    cfg = _cfg(tmp_path, mode="max")
    be = _FakeBackend(claims=["sky is blue", "water is wet"], verdict="green")
    guard = cg.CascadeGuard.from_config(cfg, "alpha", backend=be)
    assert guard.hook_kind == "llm" and guard.cfg.mode == "max"
    assert guard.cfg.adjudicate_model == "alpha-fast"        # [models] guard = fast
    assert "LLM claim decomposition" in capsys.readouterr().out

    # round 0: decomposition via the backend; no anchors yet -> yellow, no adjudication call
    o0 = guard.on_agent_output(_out(), round_n=0)
    assert o0["_claim_counts"] == {"green": 0, "yellow": 2, "red": 0}
    assert [c.role for c in be.calls] == ["guard_decompose"]
    assert all(c.model == "alpha-fast" and c.model_role == "guard" for c in be.calls)
    assert be.calls[0].reasoning_effort == "low"              # [effort] fast = low forwarded
    assert be.calls[0].schema == cg.DECOMPOSE_SCHEMA

    # promote the round-0 claims to green so round 1 has anchors to adjudicate against
    for c in guard.lineage.values():
        c.verdict = "green"
    be.verdict = "red"
    o1 = guard.on_agent_output(_out(agent="r02", findings="The sky is not blue."), round_n=1)
    assert any(c.role == "guard_adjudicate" for c in be.calls)
    assert o1["_claim_counts"]["red"] == 2 and o1["_blocked"] is True   # max blocks on red
    assert any(c.verify_notes == "scripted" for c in guard.lineage.values())


def test_max_mode_backend_error_falls_back_to_heuristic(tmp_path, capsys):
    cfg = _cfg(tmp_path, mode="max")
    be = _FakeBackend(claims=[], errored=True)
    guard = cg.CascadeGuard.from_config(cfg, "alpha", backend=be)
    o = guard.on_agent_output(_out(), round_n=0)
    # heuristic_decompose splits the two sentences
    assert len(o["_claim_ids"]) == 2
    assert "heuristic fallback" in capsys.readouterr().out


def test_balanced_mode_is_heuristic_and_says_so(tmp_path, capsys):
    cfg = _cfg(tmp_path, mode="balanced")
    guard = cg.CascadeGuard.from_config(cfg, "alpha", backend=_FakeBackend(claims=["x"]))
    assert guard.hook_kind == "heuristic"
    out = capsys.readouterr().out
    assert "heuristic screen" in out and "NOT LLM adjudication" in out
    guard.on_agent_output(_out(), round_n=0)      # no backend call ever happens
    assert guard.cfg.uses_llm_hooks is False


def test_guard_mode_policy_table():
    assert cg.GuardConfig(mode="max").verify_all_yellow is True
    assert cg.GuardConfig(mode="max").blocks_on_red is True
    assert cg.GuardConfig(mode="speed").blocks_on_red is False
    assert cg.GuardConfig(mode="strict").uses_llm_hooks is False
    assert cg.GuardConfig(mode="off").is_active is False


def test_orchestrator_builds_guard_via_from_config():
    src = (ROOT / "orchestrator.py").read_text()
    assert "CascadeGuard.from_config(runtime_cfg, selected_backend)" in src


# ===========================================================================
# mythos/verifier.py — backends-04: mechanical check + confidence gate
# ===========================================================================


@dataclass
class _Exec:
    spec_id: str
    deliverable: str
    acceptance_check: list = None
    concerns: list = None
    confidence: float = 0.9
    cost_usd: float | None = None
    tokens: int = 0
    elapsed_s: float = 0.0
    error: str | None = None

    def to_dict(self):
        return {"spec_id": self.spec_id, "deliverable": self.deliverable}


@dataclass
class _Plan:
    task_summary: str = "t"
    sub_specs: list = None
    global_invariants: list = None
    verification_strategy: str = "v"


def _invoke_result(parsed):
    from mythos._invoke import InvokeResult

    return InvokeResult(parsed=parsed, raw_stdout="", cost_usd=None, input_tokens=1,
                        output_tokens=1, elapsed_s=0.01)


def _patch_invoke(monkeypatch, verdict="pass", confidence=0.9, capture=None):
    from mythos import verifier as vmod

    def fake(**kw):
        if capture is not None:
            capture.append(kw)
        return _invoke_result({"verdict": verdict, "confidence": confidence,
                               "invariant_results": [], "spec_results": [],
                               "required_fixes": [], "summary": "s"})

    monkeypatch.setattr(vmod._invoke, "invoke", fake)
    return vmod


_PASSING = textwrap.dedent("""
    Implemented the helper.

    ### mathx.py
    ```python
    def add(a, b):
        return a + b
    ```

    ### test_mathx.py
    ```python
    from mathx import add

    def test_add():
        assert add(1, 2) == 3
    ```
""")
_FAILING = _PASSING.replace("assert add(1, 2) == 3", "assert add(1, 2) == 4")


def test_extract_files_recognises_named_fences():
    from mythos.verifier import extract_files

    files = extract_files(_PASSING)
    assert set(files) == {"mathx.py", "test_mathx.py"}
    assert "def add" in files["mathx.py"]
    info = "```python title=tests/test_a.py\nassert True\n```\n**File: src/b.py**\n```python\nx=1\n```\n```python\nunnamed\n```"
    files = extract_files(info)
    assert set(files) == {"tests/test_a.py", "src/b.py"}
    assert files["src/b.py"] == "x=1\n"
    assert extract_files("### ../escape.py\n```python\nx\n```") == {}


def test_run_artifact_tests_not_run_without_test_file():
    from mythos.verifier import run_artifact_tests

    r = run_artifact_tests([_Exec("s1", "### a.py\n```python\nx = 1\n```")])
    assert r["ran"] is False and r["passed"] is None and "no test file" in r["reason"]


@pytest.mark.spawns_subprocess
def test_verifier_mechanical_check_demotes_pass_when_artifact_tests_fail(monkeypatch):
    from mythos.policies import MythosConfig

    seen = []
    vmod = _patch_invoke(monkeypatch, verdict="pass", confidence=0.95, capture=seen)
    cfg = MythosConfig(verifier_run_tests=True, verifier_test_timeout_s=60)
    v = vmod.verify("task", _Plan(sub_specs=[], global_invariants=[]), [_Exec("s1", _FAILING)], cfg)
    assert v.llm_verdict == "pass" and v.verdict == "needs_work"
    assert v.mechanical["ran"] is True and v.mechanical["passed"] is False
    assert v.mechanical["test_files"] == ["test_mathx.py"]
    assert any("Mechanical check" in f for f in v.required_fixes)
    assert any("needs_work" in n for n in v.gate_notes)
    # the LLM verifier was shown the mechanical result as evidence
    assert "## Mechanical check" in seen[0]["user_prompt"]
    assert v.to_dict()["mechanical"]["returncode"] not in (None, 0)


@pytest.mark.spawns_subprocess
def test_verifier_mechanical_check_keeps_pass_when_artifact_tests_pass(monkeypatch):
    from mythos.policies import MythosConfig

    vmod = _patch_invoke(monkeypatch, verdict="pass", confidence=0.95)
    v = vmod.verify("task", _Plan(sub_specs=[], global_invariants=[]), [_Exec("s1", _PASSING)],
                    MythosConfig(verifier_run_tests=True, verifier_test_timeout_s=60))
    assert v.verdict == "pass" and v.mechanical["passed"] is True
    assert any("artifact tests passed" in n for n in v.gate_notes)


def test_verifier_confidence_gate(monkeypatch):
    from mythos.policies import MythosConfig

    vmod = _patch_invoke(monkeypatch, verdict="pass", confidence=0.4)
    cfg = MythosConfig(verifier_run_tests=False, verifier_min_confidence=0.6)
    v = vmod.verify("task", _Plan(sub_specs=[], global_invariants=[]), [_Exec("s1", "no code")], cfg)
    assert v.llm_verdict == "pass" and v.verdict == "needs_work"
    assert v.mechanical == {} and any("confidence 0.40 < floor 0.60" in n for n in v.gate_notes)
    assert any("below the 0.60 floor" in f for f in v.required_fixes)

    # at/above the floor the LLM pass stands
    vmod = _patch_invoke(monkeypatch, verdict="pass", confidence=0.6)
    assert vmod.verify("t", _Plan(sub_specs=[], global_invariants=[]), [_Exec("s1", "x")], cfg).verdict == "pass"


def test_verifier_run_tests_false_skips_subprocess(monkeypatch):
    from mythos.policies import MythosConfig
    from mythos import verifier as vmod

    monkeypatch.setattr(vmod.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("subprocess")))
    _patch_invoke(monkeypatch, verdict="pass", confidence=0.9)
    v = vmod.verify("t", _Plan(sub_specs=[], global_invariants=[]), [_Exec("s1", _FAILING)],
                    MythosConfig(verifier_run_tests=False))
    assert v.verdict == "pass" and v.mechanical == {}


def test_verifier_timeout_counts_as_failure(monkeypatch):
    from mythos import verifier as vmod

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(vmod.subprocess, "run", boom)
    r = vmod.run_artifact_tests([_Exec("s1", _PASSING)], timeout_s=1)
    assert r["ran"] is True and r["passed"] is False and "timed out" in r["reason"]


def test_mythos_config_verifier_knobs_load_from_toml(tmp_path):
    from mythos.policies import MythosConfig, load_mythos_config

    # Executing model-written tests is opt-in (security review 2026-10-11).
    assert MythosConfig().verifier_min_confidence == 0.6 and MythosConfig().verifier_run_tests is False
    p = tmp_path / "m.toml"
    p.write_text("[policies]\nverifier_run_tests = false\nverifier_min_confidence = 0.8\nverifier_test_timeout_s = 7\n")
    c = load_mythos_config(p)
    assert (c.verifier_run_tests, c.verifier_min_confidence, c.verifier_test_timeout_s) == (False, 0.8, 7)
    shipped = load_mythos_config()
    assert shipped.verifier_run_tests is False and shipped.verifier_min_confidence == 0.6


# ===========================================================================
# scripts/ — research-docs-02
# ===========================================================================


def test_optimize_prompts_refuses_when_evolve_disabled(monkeypatch):
    import importlib

    import evolve

    monkeypatch.setattr(evolve, "load_config", lambda *a, **k: BlitzConfig())
    mod = importlib.import_module("scripts.optimize_prompts")
    monkeypatch.setattr(mod, "load_config", lambda *a, **k: BlitzConfig())
    with pytest.raises(evolve.EvolveDisabled):
        mod.main(["--dry-run"])


def test_optimize_prompts_dry_run_resolves_without_gepa_or_llm(monkeypatch, capsys, tmp_path):
    import importlib

    mod = importlib.import_module("scripts.optimize_prompts")
    enabled = BlitzConfig(evolve=EvolveConfig(enabled=True))
    monkeypatch.setattr(mod, "load_config", lambda *a, **k: enabled)
    monkeypatch.setitem(sys.modules, "gepa", None)      # import would fail loudly
    rc = mod.main(["--dry-run", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    plan = json.loads(out[: out.index("dry-run")])
    assert plan["components"] == list(mod.DEFAULT_COMPONENTS)
    assert plan["train_tasks"] + plan["val_tasks"] > 0
    assert plan["max_metric_calls"] == 1500 and plan["domain"] == "general"
    assert "gepa not imported" in out


def test_optimize_prompts_prompt_workspace_restores_seed(tmp_path):
    from scripts.optimize_prompts import PromptWorkspace

    root = tmp_path / "prompts"; (root / "d").mkdir(parents=True)
    (root / "d" / "researcher.md").write_text("SEED")
    with PromptWorkspace(root, "d", ["researcher"], tmp_path / "bak") as ws:
        ws.write({"researcher": "CANDIDATE", "other": "ignored"})
        assert (root / "d" / "researcher.md").read_text() == "CANDIDATE"
    assert (root / "d" / "researcher.md").read_text() == "SEED"


def test_phase2_nightly_script_exists_and_is_executable():
    sh = ROOT / "scripts" / "nightly-gmemory-aggregate.sh"
    assert sh.exists() and sh.stat().st_mode & stat.S_IXUSR
    text = sh.read_text()
    assert "aggregate_overnight" in text and "set -euo pipefail" in text


@pytest.mark.spawns_subprocess
def test_phase2_nightly_script_parses_under_bash(tmp_path):
    """`bash -n` on the wrapper (the heredoc body needs a live gmemory db to run)."""
    sh = ROOT / "scripts" / "nightly-gmemory-aggregate.sh"
    env = {**os.environ, "PYTHON": PY}
    # point [storage] sqlite_path at a non-existent file via a temp blitz.toml? The
    # script reads the repo's blitz.toml; run from the repo and only assert the
    # script is well-formed (bash -n) plus a dry python parse of its heredoc.
    proc = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
