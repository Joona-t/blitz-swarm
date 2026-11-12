"""`[swarm] judge_timeout_seconds` (2026-11-12, vLLM on the Forge): judge-ensemble,
selector and bench B2 judge calls had fixed limits (min(timeout_seconds, 90) and
300 s) that no config could raise. 0 keeps the old limits."""

from __future__ import annotations

import orchestrator
from backends import AgentResult
from bench import judge as J
from bench.arms import ArmContext
from config import load_config
from mechanisms.selector_synth import Span

TOML = '''
[swarm]
timeout_seconds = 600
judge_timeout_seconds = {jt}
[models]
default = "standard"
[backend]
default = "gen"
[backend.providers.gen]
adapter = "generic_cli"
command = ["gen-tool", "{{model}}", "{{prompt_file}}"]
allow_degrade = ["no_tools"]
[backend.providers.gen.tiers]
fast = "g"
standard = "g"
deep = "g"
'''


def _cfg(tmp_path, jt):
    p = tmp_path / f"blitz{jt}.toml"
    p.write_text(TOML.format(jt=jt), encoding="utf-8")
    return load_config(p)


class Rec:
    backend_id = "gen"
    adapter_kind = "fake"
    capabilities = None

    def __init__(self):
        self.timeouts = []

    def is_available(self):
        return True

    def call(self, call):
        self.timeouts.append(call.timeout_s)
        return AgentResult(backend_id="gen", adapter_kind="fake", model=call.model, text="", raw_stdout="",
                           parsed=None, elapsed_s=0.0, errored=True, error="x")


def test_config_default_keeps_old_limits(tmp_path):
    assert _cfg(tmp_path, 0).swarm.judge_timeout_seconds == 0
    assert orchestrator.JUDGE_TIMEOUT_SECONDS == (
        orchestrator._config.swarm.judge_timeout_seconds or min(orchestrator.AGENT_TIMEOUT_SECONDS, 90))


def test_swarm_judges_use_the_configured_limit(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, 600)
    monkeypatch.setattr(orchestrator, "JUDGE_TIMEOUT_SECONDS", cfg.swarm.judge_timeout_seconds)
    kw = dict(cfg=cfg, backend_id="gen", provider=cfg.backend.get_provider("gen"),
              sandbox="read-only", scratch_dir=tmp_path)
    b = Rec()
    judge = orchestrator._make_judge_fn(backend=b, **kw)
    judge = judge[0] if isinstance(judge, tuple) else judge
    try:
        judge("cand", "hist", "j0", "g", 1, ("coverage",))
    except Exception:
        pass
    pair = orchestrator._make_pairwise_judge_fn(backend=b, **kw)
    pair = pair[0] if isinstance(pair, tuple) else pair
    span = Span(id="s", source_agent="a", source_round=1, heading=None, text="t")
    try:
        pair(span, span, "topic", "j0", 1)
    except Exception:
        pass
    assert b.timeouts == [600, 600]


def test_bench_b2_judge_uses_the_configured_limit(tmp_path, monkeypatch):
    seen = {}
    real = J.make_ask

    def spy(*a, **k):
        seen["timeout_s"] = k.get("timeout_s")
        return real(*a, **k)

    monkeypatch.setattr(J, "make_ask", spy)
    from bench import arms
    for jt, want in ((600, 600), (0, 300)):
        cfg = _cfg(tmp_path, jt)
        ctx = ArmContext(run_dir=tmp_path / f"r{jt}", cfg=cfg, backend=Rec(), system_prompt="S",
                         n_drafts=2, allow_same_family=True)
        monkeypatch.setattr(arms, "_drafts", lambda *a, **k: ["a", "b"])
        monkeypatch.setattr(J, "compare", lambda *a, **k: {"x_wins": 1, "y_wins": 0, "ties": 0})
        arms.run_b2(ctx, "topic")
        assert seen["timeout_s"] == want
