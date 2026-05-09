"""Tests for bench/runner.py — uses mocked swarm_fn so no LLM cost is incurred."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from bench import load_slate
from bench.runner import (
    BenchConfig,
    PromptResult,
    _append_jsonl,
    _coverage_count,
    _find_metrics_record_for_topic,
    _resumable_ids,
    aggregate,
    run_bench,
    run_one_prompt,
    write_stats_md,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_cfg(slate_path: Path, **overrides) -> BenchConfig:
    base = dict(
        slate_path=slate_path,
        slate_filter_ids=("s001",),
        parallel_prompts=1,
        per_prompt_timeout_s=10,
        total_budget_usd=1.0,
        max_rounds=1,
        use_redis=False,
        seed=42,
    )
    base.update(overrides)
    return BenchConfig(**base)


async def _fake_swarm_factory(tmp_path: Path, metrics_path: Path,
                               sleep_s: float = 0.0):
    """Create a fake swarm function that writes a fake output file
    and a corresponding metrics.jsonl record."""
    call_count = {"n": 0}

    async def fake_swarm(topic: str, *, max_rounds: int = 4, use_redis: bool = False) -> Path:
        if sleep_s:
            await asyncio.sleep(sleep_s)
        call_count["n"] += 1
        out_path = tmp_path / f"output_{call_count['n']}.md"
        out_path.write_text(
            f"# {topic}\n\nResearch findings:\n"
            "WAL file structure is documented; checkpointing happens periodically.\n"
            "Concurrency guarantees include reader-writer concurrency.\n"
            "fsync semantics control durability.\n",
            encoding="utf-8",
        )
        record = {
            "run_id": f"fake_run_{call_count['n']}",
            "topic": topic,
            "timestamp": time.time(),
            "consensus_reached": True,
            "rounds_to_consensus": 1,
            "coverage": 8.0,
            "accuracy": 8.5,
            "clarity": 7.5,
            "depth": 7.0,
            "avg_quality": 7.75,
            "total_cost_usd": 0.10,
            "total_input_tokens": 1500,
            "total_output_tokens": 300,
            "total_wall_clock_s": 12.0,
            "rounds_to_consensus_count": 1,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return out_path

    return fake_swarm, call_count


# ---------------------------------------------------------------------------
# Helper-function tests (no swarm call)
# ---------------------------------------------------------------------------


def test_coverage_count_counts_hits():
    md = "The WAL file structure is documented. Checkpointing happens. Fsync semantics matter."
    expected = ["WAL file structure", "checkpointing", "fsync semantics", "missing"]
    hits = _coverage_count(md, expected)
    assert hits == 3


def test_coverage_count_case_insensitive():
    md = "Wal File Structure Detail"
    expected = ["WAL file structure"]
    assert _coverage_count(md, expected) == 1


def test_coverage_count_empty_inputs():
    assert _coverage_count("", ["x"]) == 0
    assert _coverage_count("text", []) == 0


def test_resumable_ids_empty_dir(tmp_path):
    assert _resumable_ids(tmp_path) == set()


def test_resumable_ids_reads_jsonl(tmp_path):
    path = tmp_path / "results.jsonl"
    _append_jsonl(path, {"prompt_id": "s001", "x": 1})
    _append_jsonl(path, {"prompt_id": "s002", "x": 2})
    _append_jsonl(path, {"x": 3})  # no prompt_id — skipped
    ids = _resumable_ids(tmp_path)
    assert ids == {"s001", "s002"}


def test_find_metrics_record_for_topic(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    rows = [
        {"topic": "A", "timestamp": 100.0, "avg_quality": 5},
        {"topic": "B", "timestamp": 200.0, "avg_quality": 6},
        {"topic": "A", "timestamp": 300.0, "avg_quality": 7},
    ]
    with metrics_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rec = _find_metrics_record_for_topic("A", metrics_path=metrics_path)
    assert rec["timestamp"] == 300.0
    assert rec["avg_quality"] == 7


def test_find_metrics_after_timestamp(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    with metrics_path.open("w") as f:
        f.write(json.dumps({"topic": "A", "timestamp": 50.0}) + "\n")
        f.write(json.dumps({"topic": "A", "timestamp": 250.0}) + "\n")
    rec = _find_metrics_record_for_topic("A", metrics_path=metrics_path,
                                          after_timestamp=100.0)
    assert rec["timestamp"] == 250.0


# ---------------------------------------------------------------------------
# Single-prompt run (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_prompt_success(tmp_path, slate_path):
    metrics_path = tmp_path / "metrics.jsonl"
    fake_swarm, call_count = await _fake_swarm_factory(tmp_path, metrics_path)

    slate = load_slate(slate_path)
    prompt = slate.by_id("s001")
    cfg = make_cfg(slate_path)

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = await run_one_prompt(prompt, cfg, run_dir,
                                  swarm_fn=fake_swarm, metrics_path=metrics_path)

    assert isinstance(result, PromptResult)
    assert result.prompt_id == "s001"
    assert result.timeout is False
    assert result.error is None
    assert result.cost_usd > 0
    assert result.quality["avg"] > 0
    assert result.coverage_hits >= 1
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_run_one_prompt_timeout(tmp_path, slate_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.touch()

    async def slow_swarm(topic, *, max_rounds=4, use_redis=False):
        await asyncio.sleep(5)
        raise RuntimeError("should have timed out")

    slate = load_slate(slate_path)
    prompt = slate.by_id("s001")
    cfg = make_cfg(slate_path, per_prompt_timeout_s=0.1)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = await run_one_prompt(prompt, cfg, run_dir,
                                  swarm_fn=slow_swarm, metrics_path=metrics_path)
    assert result.timeout is True
    assert result.error == "timeout"
    assert "FM-3.1" in result.mast_flags


@pytest.mark.asyncio
async def test_run_one_prompt_exception(tmp_path, slate_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.touch()

    async def bad_swarm(topic, *, max_rounds=4, use_redis=False):
        raise ValueError("boom")

    slate = load_slate(slate_path)
    prompt = slate.by_id("s001")
    cfg = make_cfg(slate_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = await run_one_prompt(prompt, cfg, run_dir,
                                  swarm_fn=bad_swarm, metrics_path=metrics_path)
    assert result.error is not None
    assert "boom" in result.error
    assert result.timeout is False


# ---------------------------------------------------------------------------
# Full run_bench (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_bench_smoke(tmp_path, slate_path, monkeypatch):
    metrics_path = tmp_path / "metrics.jsonl"
    fake_swarm, call_count = await _fake_swarm_factory(tmp_path, metrics_path)

    monkeypatch.setattr("bench.runner.RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg(slate_path,
                   slate_filter_ids=("s001", "s002", "s003"),
                   parallel_prompts=2)

    run_dir = await run_bench(cfg, swarm_fn=fake_swarm, metrics_path=metrics_path)

    assert run_dir.exists()
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "stats.md").exists()
    assert call_count["n"] == 3

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["slate"]["prompts_run"] == 3
    assert summary["aggregate_quality"]["avg"]["mean"] > 0


@pytest.mark.asyncio
async def test_run_bench_resume(tmp_path, slate_path, monkeypatch):
    """Re-running with same run_id picks up results.jsonl, skips done prompts."""
    metrics_path = tmp_path / "metrics.jsonl"
    fake_swarm, call_count = await _fake_swarm_factory(tmp_path, metrics_path)

    monkeypatch.setattr("bench.runner.RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg(slate_path,
                   slate_filter_ids=("s001", "s002"),
                   parallel_prompts=1,
                   started_utc="2026-05-09T00:00:00+00:00",
                   git_sha="testsha000")

    run_dir = await run_bench(cfg, swarm_fn=fake_swarm, metrics_path=metrics_path)
    first_calls = call_count["n"]
    assert first_calls == 2

    # Re-run with same identity tuple — should skip both prompts
    cfg2 = make_cfg(slate_path,
                    slate_filter_ids=("s001", "s002"),
                    parallel_prompts=1,
                    started_utc="2026-05-09T00:00:00+00:00",
                    git_sha="testsha000")
    run_dir2 = await run_bench(cfg2, swarm_fn=fake_swarm, metrics_path=metrics_path)
    assert run_dir2 == run_dir
    assert call_count["n"] == first_calls  # no new calls — resumed


@pytest.mark.asyncio
async def test_run_bench_budget_gate(tmp_path, slate_path, monkeypatch):
    """Per-prompt budget exhausted → result has error='budget_exhausted'."""
    metrics_path = tmp_path / "metrics.jsonl"
    fake_swarm, _ = await _fake_swarm_factory(tmp_path, metrics_path)

    monkeypatch.setattr("bench.runner.RUNS_DIR", tmp_path / "runs")
    # Total budget 0.10 — first easy prompt (0.05) fits, but the second's
    # cost report ($0.10 in fake metrics) drains it before a third.
    cfg = make_cfg(slate_path,
                   slate_filter_ids=("s001", "s002", "s016"),
                   parallel_prompts=1,
                   total_budget_usd=0.05)

    await run_bench(cfg, swarm_fn=fake_swarm, metrics_path=metrics_path)

    rows: list[dict] = []
    results_path = (tmp_path / "runs").glob("*/results.jsonl")
    for path in results_path:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    assert any(r.get("error") == "budget_exhausted" for r in rows)


# ---------------------------------------------------------------------------
# Aggregation + stats output
# ---------------------------------------------------------------------------


def test_aggregate_writes_summary_json(tmp_path, slate_path):
    """Synthetic results.jsonl in tmp dir; aggregate produces summary.json."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "prompt_id": "s001",
            "run_id": run_dir.name,
            "swarm_run_id": "x",
            "prompt_sha256": "y",
            "output_md_path": "",
            "started_utc": "2026-05-09T00:00:00+00:00",
            "elapsed_s": 10.0,
            "cost_usd": 0.05,
            "input_tokens": 1000,
            "output_tokens": 200,
            "consensus_reached": True,
            "rounds_to_consensus": 1,
            "quality": {"coverage": 8, "accuracy": 8, "clarity": 7, "depth": 7, "avg": 7.5},
            "coverage_hits": 4,
            "coverage_total": 5,
            "timeout": False,
            "error": None,
            "mast_flags": [],
        },
        {
            "prompt_id": "s002",
            "run_id": run_dir.name,
            "swarm_run_id": "x2",
            "prompt_sha256": "z",
            "output_md_path": "",
            "started_utc": "2026-05-09T00:00:00+00:00",
            "elapsed_s": 12.0,
            "cost_usd": 0.05,
            "input_tokens": 1000,
            "output_tokens": 200,
            "consensus_reached": False,
            "rounds_to_consensus": 0,
            "quality": {"coverage": 6, "accuracy": 7, "clarity": 6, "depth": 6, "avg": 6.25},
            "coverage_hits": 3,
            "coverage_total": 5,
            "timeout": False,
            "error": None,
            "mast_flags": ["FM-2.4"],
        },
    ]
    with (run_dir / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    slate = load_slate(slate_path)
    cfg = BenchConfig(
        slate_path=slate_path,
        started_utc="2026-05-09T00:00:00+00:00",
        git_sha="abc",
    )
    summary = aggregate(run_dir, slate, cfg)

    assert summary["slate"]["prompts_run"] == 2
    assert summary["aggregate_quality"]["avg"]["mean"] == round((7.5 + 6.25) / 2, 3)
    assert summary["mast_flags_summary"].get("FM-2.4") == 1
    assert (run_dir / "summary.json").exists()


def test_write_stats_md_creates_file(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "run_id": "abc",
        "started_utc": "x",
        "ended_utc": "y",
        "slate": {"slate_id": "v1", "sha256": "abcdef" * 11,
                  "prompts_run": 2, "prompt_count": 30,
                  "prompts_timed_out": 0, "prompts_errored": 0},
        "aggregate_quality": {
            "avg": {"mean": 7.0, "stddev": 0.5, "ci95": [6.5, 7.5]},
        },
        "by_tier": {
            "easy": {"n": 2, "avg_quality": 7.0, "avg_cost": 0.05},
            "medium": {"n": 0, "avg_quality": 0, "avg_cost": 0},
            "hard": {"n": 0, "avg_quality": 0, "avg_cost": 0},
        },
        "mast_flags_summary": {"FM-2.4": 1},
        "cost": {"total_usd": 0.10, "input_tokens_total": 2000, "output_tokens_total": 400},
    }
    path = write_stats_md(run_dir, summary)
    assert path.exists()
    text = path.read_text()
    assert "# Bench run" in text
    assert "FM-2.4" in text
