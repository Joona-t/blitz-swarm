"""Phase 1 WP1-3, WP6, WP10: arms, the pairwise judge and preflight, all
against a fake backend. No model, no network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from backends import AgentResult
from bench import judge as J
from bench.arms import ArmContext, B1_MAX_CALLS, make_arm, prompt_key, tier_budgets
from config import load_config

TOPIC = "Explain how a write-ahead log makes commits durable."

TOML = '''
[models]
default = "standard"
judge_provider = "jud"

[backend]
default = "gen"

[backend.providers.gen]
adapter = "generic_cli"
command = ["gen-tool", "{model}", "{prompt_file}"]
allow_degrade = ["no_tools"]
[backend.providers.gen.tiers]
fast = "g"
standard = "g"
deep = "g"

[backend.providers.jud]
adapter = "generic_cli"
command = ["jud-tool", "{model}", "{prompt_file}"]
allow_degrade = ["no_tools"]
[backend.providers.jud.tiers]
fast = "j"
standard = "j"
deep = "j"
'''


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "blitz.toml"
    p.write_text(TOML, encoding="utf-8")
    return load_config(p)


class FakeBackend:
    """Scripted backend. Generation calls get numbered drafts; judge calls
    answer per `judge` (a function of the prompt)."""

    backend_id = "gen"
    adapter_kind = "fake"
    capabilities = None

    def __init__(self, *, out_tokens=1000, judge=None, fail_kinds=()):
        self.calls: list = []
        self.out_tokens = out_tokens
        self.judge = judge or (lambda prompt: {c: "tie" for c in J.CRITERIA})
        self.fail_kinds = set(fail_kinds)
        self.n = 0

    def is_available(self):
        return True

    def call(self, call):
        self.calls.append(call)
        kind = call.role.replace("bench_", "")
        if kind in self.fail_kinds:
            return AgentResult(backend_id="gen", adapter_kind="fake", model=call.model, text="", raw_stdout="",
                               parsed=None, elapsed_s=0.1, errored=True, error="boom")
        if call.model_role == "judge":
            verdict = {"expected_facts": ["f"], "rationale": "r", **self.judge(call.prompt)}
            return AgentResult(backend_id="jud", adapter_kind="fake", model=call.model, text=json.dumps(verdict),
                               raw_stdout="", parsed=verdict, elapsed_s=0.1, cost_usd=0.01,
                               input_tokens=50, output_tokens=20, cost_policy="reported")
        self.n += 1
        text = f"# Report {self.n}\nSENTINEL_{self.n} body for {kind}."
        if kind == "merge":
            text = "# Merged\n" + " ".join(f"SENTINEL_{i}" for i in range(1, self.n))
        return AgentResult(backend_id="gen", adapter_kind="fake", model=call.model, text=text, raw_stdout=text,
                           parsed=None, elapsed_s=0.2, cost_usd=None, input_tokens=100,
                           output_tokens=self.out_tokens)


def _ctx(tmp_path, cfg, backend, **kw):
    return ArmContext(run_dir=tmp_path / "run", cfg=cfg, backend=backend, system_prompt="SYS", n_drafts=3, **kw)


def _run(name, ctx):
    return asyncio.run(make_arm(name, ctx)(TOPIC, max_rounds=1, use_redis=False))


def _log_rows(ctx):
    return [json.loads(l) for l in (ctx.run_dir / "calls" / f"{prompt_key(TOPIC)}.jsonl").read_text().splitlines()]


def test_b0_is_one_researcher_call_with_no_tools(tmp_path, cfg):
    b = FakeBackend()
    ctx = _ctx(tmp_path, cfg, b)
    path, rounds, record = _run("B0", ctx)
    assert len(b.calls) == 1 and record["calls"] == 1
    c = b.calls[0]
    assert c.model_role == "researcher" and c.tools is None and c.max_turns == 1 and c.system_prompt == "SYS"
    assert TOPIC in c.prompt and path.read_text().startswith("# Report 1")
    assert record["total_output_tokens"] == 1000 and record["total_cost_usd"] is None and record["cost_incomplete"]
    assert rounds[0][0]["role"] == "researcher"
    assert sum(r["output_tokens"] for r in _log_rows(ctx)) == record["total_output_tokens"]


def test_b1_minimum_three_calls_then_stops_at_budget(tmp_path, cfg):
    b = FakeBackend(out_tokens=1000)
    ctx = _ctx(tmp_path, cfg, b, output_token_budget=0)
    _, _, record = _run("B1", ctx)
    assert record["calls"] == 3 and [c.role for c in b.calls] == ["bench_draft", "bench_critique", "bench_revise"]
    assert "Report 1" in b.calls[1].prompt, "the critique must see the draft"
    assert "Report 1" in b.calls[2].prompt and "Report 2" in b.calls[2].prompt, "the revision sees draft and critique"

    b2 = FakeBackend(out_tokens=1000)
    _, _, rec2 = _run("B1", _ctx(tmp_path / "x", cfg, b2, output_token_budget=4500))
    assert rec2["calls"] == 5, "3 calls = 3000 < 4500, so one more cycle; 5000 >= 4500 stops"


def test_b1_never_exceeds_max_calls_and_uses_per_prompt_budget(tmp_path, cfg):
    b = FakeBackend(out_tokens=10)
    ctx = _ctx(tmp_path, cfg, b, output_token_budget=10**9)
    _, _, record = _run("B1", ctx)
    assert record["calls"] == B1_MAX_CALLS
    b2 = FakeBackend(out_tokens=1000)
    ctx2 = _ctx(tmp_path / "y", cfg, b2, output_token_budget=10**9)
    ctx2.budget_by_key = {prompt_key(TOPIC): 2500}
    assert _run("B1", ctx2)[2]["calls"] == 3, "the per-prompt (tier) budget overrides the flat one"


def test_b2_returns_the_draft_the_judge_prefers_in_both_orders(tmp_path, cfg):
    def judge(prompt):
        a = prompt.split("<<<REPORT A>>>")[1].split("<<<END REPORT A>>>")[0]
        pick = "A" if "SENTINEL_2" in a else ("B" if "SENTINEL_2" in prompt.split("<<<REPORT B>>>")[1] else "tie")
        return {c: pick for c in J.CRITERIA}

    b = FakeBackend(judge=judge)
    ctx = _ctx(tmp_path, cfg, b)
    path, rounds, record = _run("B2", ctx)
    assert "SENTINEL_2" in path.read_text()
    assert len(rounds[0]) == 3 and len(list((ctx.run_dir / "drafts" / prompt_key(TOPIC)).iterdir())) == 3
    judge_calls = [c for c in b.calls if c.model_role == "judge"]
    assert len(judge_calls) == 6, "3 pairs x 2 orders"
    assert record["calls"] == 9 and record["generation_output_tokens"] == 3000
    assert record["total_cost_usd"] == pytest.approx(0.06) and record["cost_incomplete"] is True
    assert set(record["providers"]) == {"gen", "jud"}


def test_b2_refuses_a_same_family_judge(tmp_path, cfg):
    cfg.models.providers.clear()
    with pytest.raises(J.SameFamilyJudge, match="judge_provider"):
        _run("B2", _ctx(tmp_path, cfg, FakeBackend()))
    path, _, _ = _run("B2", _ctx(tmp_path / "ok", cfg, FakeBackend(), allow_same_family=True))
    assert path.exists()


def test_b3_merge_keeps_every_draft_sentinel(tmp_path, cfg):
    b = FakeBackend()
    path, rounds, record = _run("B3", _ctx(tmp_path, cfg, b))
    text = path.read_text()
    assert all(f"SENTINEL_{i}" in text for i in (1, 2, 3))
    merge = [c for c in b.calls if c.role == "bench_merge"][0]
    assert all(f"<<<REPORT {i}>>>" in merge.prompt for i in (1, 2, 3)) and "Keep every distinct claim" in merge.prompt
    assert record["calls"] == 4 and len(rounds[0]) == 3


def test_failed_generation_is_an_arm_error_not_an_empty_report(tmp_path, cfg):
    from bench.arms import ArmError
    with pytest.raises(ArmError, match="boom"):
        _run("B0", _ctx(tmp_path, cfg, FakeBackend(fail_kinds={"draft"})))


def test_unknown_arm_and_s_passthrough():
    with pytest.raises(ValueError, match="unknown arm"):
        make_arm("B9", None)
    import orchestrator
    assert make_arm("S", None) is orchestrator.run_swarm


# --- judge -----------------------------------------------------------------


def test_position_biased_judge_yields_ties_and_counts_flips():
    v = J.compare("t", "report x", "report y", lambda p, s, schema: {c: "A" for c in J.CRITERIA})
    assert v["winner"] == "tie" and v["ties"] == 4 and v["order_flips"] == 4


def test_judge_preferring_a_marker_wins_in_both_orders_and_is_cached():
    asked = []

    def ask(prompt, system, schema):
        asked.append(prompt)
        a = prompt.split("<<<REPORT A>>>")[1].split("<<<END REPORT A>>>")[0]
        return {c: ("A" if "GOOD" in a else "B") for c in J.CRITERIA}

    cache: dict = {}
    v = J.compare("t", "the GOOD one", "the other", ask, cache=cache)
    assert v["winner"] == "x" and v["x_wins"] == 4 and v["order_flips"] == 0
    assert (v["len_x"], v["len_y"]) == (12, 9)
    assert J.compare("t", "the GOOD one", "the other", ask, cache=cache) is v and len(asked) == 2
    assert J.compare("t", "the other", "the GOOD one", ask)["winner"] == "y"


def test_failed_judgement_is_errored_tie_and_not_cached():
    cache: dict = {}
    v = J.compare("t", "a", "b", lambda p, s, schema: None, cache=cache)
    assert v["errored"] and v["winner"] == "tie" and cache == {}


def test_judge_prompt_asks_for_expected_facts_before_the_reports():
    p = J.build_prompt("task", "AAA", "BBB")
    assert p.index("expected_facts") < p.index("<<<REPORT A>>>") < p.index("<<<REPORT B>>>")
    assert "Do not reward length" in p


# --- budgets and preflight ---------------------------------------------------


def test_tier_budgets_are_medians_of_scored_prompts(tmp_path):
    class Slate:
        def by_id(self, pid):
            return type("P", (), {"tier": {"a": "easy", "b": "easy", "c": "easy", "d": "hard"}[pid]})()

    rows = [{"prompt_id": "a", "output_tokens": 100}, {"prompt_id": "b", "output_tokens": 300},
            {"prompt_id": "c", "output_tokens": 0, "error": "x"}, {"prompt_id": "d", "output_tokens": 900}]
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert tier_budgets(tmp_path, Slate()) == {"easy": 200, "hard": 900}


def test_preflight_names_each_problem(tmp_path, cfg):
    from bench.preflight import run_preflight
    from bench.runner import BenchConfig

    ok = run_preflight(BenchConfig(slate_path=Path("x"), arm="B0", citations="off"), blitz_cfg=cfg,
                       which=lambda b: "/bin/" + b)
    assert ok == []
    missing = run_preflight(BenchConfig(slate_path=Path("x"), arm="B2", citations="off"), blitz_cfg=cfg,
                            which=lambda b: None)
    assert any("gen-tool" in m for m in missing) and any("jud-tool" in m for m in missing)
    no_budget = run_preflight(BenchConfig(slate_path=Path("x"), arm="B1", citations="off"), blitz_cfg=cfg,
                              which=lambda b: "/bin/" + b)
    assert any("B1 has no budget" in m for m in no_budget)
    cfg.models.providers.clear()
    same = run_preflight(BenchConfig(slate_path=Path("x"), arm="B2", citations="off"), blitz_cfg=cfg,
                         which=lambda b: "/bin/" + b)
    assert any("judge provider" in m for m in same)
    full = run_preflight(BenchConfig(slate_path=Path("x"), arm="B0", citations="off"), blitz_cfg=cfg,
                         which=lambda b: "/bin/" + b,
                         disk_usage=lambda p: type("U", (), {"free": 5e8})())
    assert any("GB free" in m for m in full)


def test_call_log_records_finish_reason_so_truncation_is_visible(tmp_path, cfg):
    """2026-11-12 (vLLM live run): the adapter put finish_reason in metadata but
    the call log dropped it, so a report cut at the token limit looked complete."""

    class Truncating(FakeBackend):
        def call(self, call):
            r = super().call(call)
            r.metadata = {"finish_reason": "length", "thinking_chars": 0}
            return r

    ctx = _ctx(tmp_path, cfg, Truncating())
    _, _, record = _run("B0", ctx)
    row = _log_rows(ctx)[0]
    assert row["finish_reason"] == "length" and row["thinking_chars"] == 0
    assert record["truncated_calls"] == 1
    # A backend that reports nothing logs None, not a guess.
    ctx2 = _ctx(tmp_path / "b", cfg, FakeBackend())
    _, _, record2 = _run("B0", ctx2)
    assert _log_rows(ctx2)[0]["finish_reason"] is None and record2["truncated_calls"] == 0

    class OllamaTruncating(FakeBackend):
        def call(self, call):
            r = super().call(call)
            r.metadata = {"done_reason": "length"}
            return r

    ctx3 = _ctx(tmp_path / "c", cfg, OllamaTruncating())
    _, _, record3 = _run("B0", ctx3)
    assert _log_rows(ctx3)[0]["finish_reason"] == "length" and record3["truncated_calls"] == 1
