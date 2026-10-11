"""metrics.py: per-run metrics collection, cost accounting (None = n/a, never
0), JSONL persistence, rolling averages, regression detection and the
human-readable report. Pure stdlib; no backend is invoked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import metrics as metrics_mod
from config import load_config
from metrics import (
    MetricsCollector,
    RoundMetrics,
    RunMetrics,
    compute_rolling_avg,
    cost_from_pricing,
    detect_regressions,
    format_metrics_report,
    load_all_metrics,
)

PRICED_TOML = """
[backend]
default = "prov"

[backend.providers.prov]
adapter = "generic_cli"
command = ["echo"]

[backend.providers.prov.tiers]
standard = "priced-model"

[backend.providers.prov.pricing."priced-model"]
input_per_mtok = 10.0
output_per_mtok = 30.0
"""


@pytest.fixture
def priced_cfg(tmp_path: Path):
    p = tmp_path / "blitz.toml"
    p.write_text(PRICED_TOML, encoding="utf-8")
    return load_config(p)


@pytest.fixture
def metrics_path(tmp_path: Path, monkeypatch):
    path = tmp_path / "out" / "metrics.jsonl"
    monkeypatch.setattr(metrics_mod, "METRICS_PATH", path)
    return path


# ---------------------------------------------------------------------------
# cost_from_pricing
# ---------------------------------------------------------------------------


def test_cost_from_pricing_uses_configured_per_mtok_prices(priced_cfg):
    # 1M input @ $10 + 500k output @ $30 = 10 + 15
    assert cost_from_pricing(priced_cfg, "prov", "priced-model", 1_000_000, 500_000) == pytest.approx(25.0)


@pytest.mark.parametrize("cfg_key, backend_id, model", [
    ("none", None, "priced-model"),        # no cfg at all
    ("cfg", None, "priced-model"),         # no backend id
    ("cfg", "prov", None),                 # no model
    ("cfg", "ghost", "priced-model"),      # unknown provider -> KeyError swallowed
    ("cfg", "prov", "unpriced-model"),     # provider has no pricing entry
])
def test_cost_from_pricing_returns_none_not_zero_when_unknown(priced_cfg, cfg_key, backend_id, model):
    cfg = None if cfg_key == "none" else priced_cfg
    assert cost_from_pricing(cfg, backend_id, model, 1000, 1000) is None


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


def test_round_metrics_success_rate_divides_by_max_total_1():
    assert RoundMetrics().success_rate == 0.0
    assert RoundMetrics(agents_total=4, agents_ok=3).success_rate == pytest.approx(0.75)


def test_run_metrics_defaults_cost_unknown():
    run = RunMetrics()
    assert run.total_cost_usd is None
    assert run.rounds == {}
    assert run.consensus_reached is False


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


def _outputs():
    return [
        {"agent_id": "a", "findings": "fine"},
        {"agent_id": "b", "_error": True, "_error_msg": "Timed out after 180s"},
        {"agent_id": "c", "_error": True, "_error_msg": "exit 1"},
        {"agent_id": "d", "_error": True, "findings": "TIMEOUT"},
        {"agent_id": "e", "_raw": True, "findings": "not json"},
    ]


def test_end_round_classifies_ok_error_timeout_raw():
    mc = MetricsCollector()
    mc.start_run("r1", "topic")
    mc.start_round("round_1")
    mc.end_round("round_1", _outputs())
    rm = mc.run.rounds["round_1"]
    assert (rm["agents_total"], rm["agents_ok"], rm["agents_error"],
            rm["agents_timeout"], rm["agents_raw"]) == (5, 1, 1, 2, 1)
    assert rm["cost_usd"] is None
    assert mc.run.total_cost_usd is None
    assert mc.run.total_agent_invocations == 5
    assert rm["wall_clock_s"] >= 0.0


def test_end_round_without_start_round_does_not_crash():
    mc = MetricsCollector()
    mc.start_run("r1", "topic")
    mc.end_round("unstarted", [{"findings": "x"}])
    assert mc.run.rounds["unstarted"]["agents_ok"] == 1


def test_end_round_sums_provider_reported_cost_and_tokens():
    mc = MetricsCollector()
    mc.start_run("r1", "t")
    mc.start_round("r")
    mc.end_round("r", [
        {"findings": "a", "_cost_usd": 0.5, "_input_tokens": 100, "_output_tokens": 10},
        {"findings": "b", "_cost_usd": 0.25, "_input_tokens": 50, "_output_tokens": 5},
        {"findings": "c"},   # unknown cost does not zero the sum
    ])
    rm = mc.run.rounds["r"]
    assert rm["cost_usd"] == pytest.approx(0.75)
    assert (rm["input_tokens"], rm["output_tokens"]) == (150, 15)
    assert mc.run.total_cost_usd == pytest.approx(0.75)
    assert (mc.run.total_input_tokens, mc.run.total_output_tokens) == (150, 15)


def test_end_round_falls_back_to_configured_pricing_per_output(priced_cfg):
    mc = MetricsCollector(cfg=priced_cfg, backend_id="prov")
    mc.start_run("r1", "t")
    mc.start_round("r")
    mc.end_round("r", [
        # collector-level backend id, priced model: 100k*10 + 100k*30 per 1M = 4.0
        {"findings": "a", "_model": "priced-model", "_input_tokens": 100_000, "_output_tokens": 100_000},
        # per-output backend id wins over the collector default
        {"findings": "b", "_backend_id": "ghost", "_model": "priced-model",
         "_input_tokens": 100_000, "_output_tokens": 100_000},
        # provider-reported cost wins over pricing
        {"findings": "c", "_model": "priced-model", "_cost_usd": 1.0,
         "_input_tokens": 100_000, "_output_tokens": 100_000},
    ])
    assert mc.run.rounds["r"]["cost_usd"] == pytest.approx(5.0)
    assert mc.run.total_cost_usd == pytest.approx(5.0)


def test_record_consensus_state_only_for_known_rounds():
    mc = MetricsCollector()
    mc.start_run("r1", "t")
    mc.record_consensus_state("missing", 1, 2, 0.5)
    assert "missing" not in mc.run.rounds
    mc.start_round("r")
    mc.end_round("r", [])
    mc.record_consensus_state("r", 2, 3, 0.83333)
    rm = mc.run.rounds["r"]
    assert (rm["ready_votes"], rm["total_voters"], rm["avg_confidence"]) == (2, 3, 0.833)


def test_record_quality_scores_averages_nonzero_only():
    mc = MetricsCollector()
    mc.record_quality_scores({"coverage_score": 8, "accuracy_score": 6, "clarity_score": 0, "depth_score": 7})
    assert (mc.run.coverage, mc.run.accuracy, mc.run.clarity, mc.run.depth) == (8, 6, 0, 7)
    assert mc.run.avg_quality == 7.0
    mc.record_quality_scores({})
    assert mc.run.avg_quality == 0


def test_record_consensus_result_and_agent_timing():
    mc = MetricsCollector()
    mc.start_run("r1", "t")
    mc.record_consensus_result(3, True, True)
    assert (mc.run.rounds_to_consensus, mc.run.consensus_reached, mc.run.override_applied) == (3, True, True)
    mc.record_agent_timing("researcher_00", "tier-model", 12.345, context_chars=1500)
    mc.record_agent_timing("critic", "tier-model", 1.0, context_chars=500)
    assert mc.run.estimated_cost_units == 2000
    assert mc._agent_timings[0] == {
        "agent_id": "researcher_00", "model": "tier-model", "elapsed_s": 12.3, "context_chars": 1500,
    }


def test_finalize_rounds_wall_clock_units_and_cost():
    mc = MetricsCollector()
    mc.start_run("r1", "t")
    mc.run.estimated_cost_units = 12.7
    mc.run.total_cost_usd = 1.23456789
    mc.finalize()
    assert mc.run.total_wall_clock_s >= 0.0
    assert mc.run.estimated_cost_units == 13
    assert mc.run.total_cost_usd == 1.2346


def test_finalize_keeps_unknown_cost_as_none():
    mc = MetricsCollector()
    mc.start_run("r1", "t")
    mc.finalize()
    assert mc.run.total_cost_usd is None


def test_save_appends_jsonl_and_load_all_metrics_reads_it_back(metrics_path):
    mc = MetricsCollector()
    mc.start_run("run-1", "topic one")
    mc.start_round("r")
    mc.end_round("r", [{"findings": "x"}])
    mc.record_agent_timing("a", "m", 1.0)
    rec = mc.save()
    mc2 = MetricsCollector()
    mc2.start_run("run-2", "topic two")
    mc2.save()
    assert metrics_path.exists()
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert rec["run_id"] == "run-1" and rec["_agent_timings"][0]["agent_id"] == "a"
    assert json.loads(lines[0])["rounds"]["r"]["agents_ok"] == 1
    loaded = load_all_metrics()
    assert [r["run_id"] for r in loaded] == ["run-1", "run-2"]


def test_load_all_metrics_skips_blank_and_corrupt_lines(metrics_path):
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text('{"run_id": "ok"}\n\nnot json\n{"run_id": "ok2"}\n', encoding="utf-8")
    assert [r["run_id"] for r in load_all_metrics()] == ["ok", "ok2"]


def test_load_all_metrics_missing_file_is_empty(metrics_path):
    assert not metrics_path.exists()
    assert load_all_metrics() == []


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def _run(quality: float, cost=None, rounds_ok=(3, 3), **extra) -> dict:
    ok, total = rounds_ok
    r = {
        "topic": "t", "avg_quality": quality, "coverage": quality, "accuracy": quality,
        "clarity": quality, "depth": quality, "rounds_to_consensus": 2,
        "consensus_reached": True, "total_wall_clock_s": 10.0,
        "rounds": {"round_1": {"agents_total": total, "agents_ok": ok}},
    }
    if cost is not None:
        r["total_cost_usd"] = cost
    r.update(extra)
    return r


def test_compute_rolling_avg_short_series_is_passthrough_and_ignores_non_numeric():
    recs = [_run(8), _run(6), {"avg_quality": "n/a"}]
    assert compute_rolling_avg(recs, "avg_quality", window=5) == [8, 6]


def test_compute_rolling_avg_windows():
    recs = [_run(q) for q in (2, 4, 6, 8)]
    assert compute_rolling_avg(recs, "avg_quality", window=2) == [2, 3.0, 5.0, 7.0]


def test_detect_regressions_needs_window_plus_one_records():
    assert detect_regressions([_run(8), _run(2)], window=3) == []


def test_detect_regressions_flags_quality_drop_round_rate_drop_and_cost_increase():
    recs = [_run(8, cost=1.0), _run(8, cost=1.0), _run(8, cost=1.0),
            _run(4, cost=2.0, rounds_ok=(1, 3))]
    regs = {r["metric"]: r for r in detect_regressions(recs, window=3, threshold=0.15)}
    for key in ("avg_quality", "coverage", "accuracy", "clarity", "depth"):
        assert regs[key]["direction"] == "dropped" and regs[key]["delta_pct"] == -50.0
    assert regs["round_1_success_rate"]["current"] == 0.33
    assert regs["round_1_success_rate"]["rolling_avg"] == 1.0
    assert regs["total_cost_usd"]["direction"].startswith("increased")
    assert regs["total_cost_usd"]["delta_pct"] == 100.0


def test_detect_regressions_is_quiet_within_threshold_and_with_unknown_cost():
    recs = [_run(8), _run(8), _run(8), _run(7.5, cost=5.0)]   # -6% quality, prior cost unknown
    assert detect_regressions(recs, window=3, threshold=0.15) == []


def test_format_metrics_report_empty():
    assert format_metrics_report([]) == "No metrics recorded yet."


def test_format_metrics_report_table_trends_and_regressions():
    recs = [_run(8, cost=1.0, topic="a" * 40), _run(8, cost=1.0), _run(8, cost=1.0), _run(4, cost=2.0)]
    out = format_metrics_report(recs, last_n=5)
    assert "4 total runs, showing last 4" in out
    assert "| " + "a" * 25 + " |" in out          # topic truncated to 25 chars
    assert "$1.00" in out and "$2.00" in out
    assert "### Trends (last 3 runs)" in out
    assert "Avg Quality: 8.0 -> 8.0 -> 4.0 v" in out
    assert "Cost (USD): 1.00 -> 1.00 -> 2.00 ^ (worse)" in out
    assert "### Regressions Detected" in out
    assert "avg_quality: 4 (avg: 8, -50.0%)" in out   # mean of ints stays int


def test_format_metrics_report_unknown_cost_is_na_and_override_marked():
    recs = [_run(8, override_applied=True), _run(8, consensus_reached=False)]
    out = format_metrics_report(recs)
    assert "| n/a |" in out
    assert "| yes* |" in out and "| no |" in out
    assert "Trends" not in out
