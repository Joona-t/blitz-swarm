"""AUDIT M11 (verifier pass vs failed specs), M5 (bench git sha)."""

from __future__ import annotations

import re

from bench.runner import _git_sha
from mythos.policies import MythosConfig
from mythos.verifier import Verification, apply_gates


def _v(**kw):
    base = dict(verdict="pass", invariant_results=[], spec_results=[], required_fixes=[],
                summary="s", confidence=0.95)
    base.update(kw)
    return Verification(**base)


def test_pass_verdict_with_failed_spec_is_demoted():
    v = apply_gates(_v(spec_results=[{"spec_id": "s1", "status": "pass"}, {"spec_id": "s2", "status": "fail"}]),
                    MythosConfig(), mechanical=None)
    assert v.verdict == "needs_work"
    assert any("s2" in n for n in v.gate_notes) and any("s2" in f for f in v.required_fixes)


def test_pass_verdict_with_all_specs_passing_stays():
    v = apply_gates(_v(spec_results=[{"spec_id": "s1", "status": "pass"}, {"spec_id": "s2", "status": "partial"}]),
                    MythosConfig(), mechanical=None)
    assert v.verdict == "pass"


def test_bench_git_sha_works_in_this_checkout():
    sha = _git_sha()
    assert sha != "nogit" and re.fullmatch(r"[0-9a-f]{12}", sha), sha
