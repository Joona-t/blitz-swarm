"""Deterministic tests for the functional research-quality scorer wiring
in bench/runner.py (G1) plus the best-effort MAST detector hook (G6).

No LLM calls, no network: a fake swarm_fn writes a fixed markdown artifact
and a matching metrics.jsonl record. We assert that:

  * run_one_prompt computes PromptResult.functional_composite from
    detectors.research_quality_composite()
  * aggregate() surfaces summary["functional_composite"] with the exact
    SHARED CONTRACT shape the driver reads:
        {"mean": <float 0..1> | None,
         "per_prompt": {prompt_id: composite},
         "missing": [prompt_id, ...]}
  * the LLM-judge scores remain under summary["aggregate_quality"] (SECONDARY)
  * C5a: timeout / error prompts get a MISSING (None) composite — they are
    EXCLUDED from the functional mean (never averaged in as 0.0) and surfaced
    in summary["functional_composite"]["missing"] + summary["timeouts"];
    when ALL prompts are missing the mean is None, not 0.0
  * H4: metrics-record matching normalizes topics (whitespace/case/120-char
    drift) and logs one loud MATCHED/MISS line per prompt; a miss never
    crashes — cost/tokens stay 0
  * the G6 detect_all hook fires only when per-agent round outputs exist
    (sibling rounds.json) and stays empty otherwise — no fabrication.

These tests are intentionally independent of the existing test_runner.py
fixtures so they document the contract on their own.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from bench import BenchPrompt, load_slate
from bench.detectors import research_quality_composite
from bench.runner import (
    BenchConfig,
    PromptResult,
    _append_jsonl,
    _find_metrics_record_for_topic,
    _load_round_outputs,
    _slate_entry_for,
    aggregate,
    run_one_prompt,
    write_stats_md,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A markdown artifact engineered to score well on the deterministic
# composite: it mentions every expected_coverage keyword for s001, cites a
# URL that we will also pass as a source, and uses a well-formed arXiv id.
_GOOD_MD = (
    "# SQLite WAL internals\n\n"
    "The WAL file structure is a header plus frames. Checkpointing happens "
    "when the WAL grows past a threshold. We compare the rollback journal vs "
    "WAL tradeoffs. The concurrency guarantees allow one writer with many "
    "readers. Finally, fsync semantics determine durability across crashes.\n\n"
    "Source: https://www.sqlite.org/wal.html — see also arXiv:2503.13657.\n"
)

_S001_SOURCES = ["https://www.sqlite.org/wal.html"]


def _make_prompt() -> BenchPrompt:
    """A self-contained prompt mirroring slate s001's coverage keywords."""
    return BenchPrompt(
        id="f001",
        tier="easy",
        domain="technical",
        parallelizable=True,
        tool_heavy=False,
        budget_usd=0.05,
        expected_coverage=(
            "WAL file structure",
            "checkpointing",
            "rollback journal vs WAL",
            "concurrency guarantees",
            "fsync semantics",
        ),
        text="Explain SQLite WAL mode internals.",
    )


def _make_cfg(slate_path: Path | None = None) -> BenchConfig:
    return BenchConfig(
        slate_path=slate_path or Path("bench/slate_v1.toml"),
        started_utc="2026-06-09T00:00:00+00:00",
        git_sha="testsha",
        slate_filter_ids=("f001",),
        max_rounds=1,
        seed=42,
    )


def _fake_swarm_writing(md: str, metrics_path: Path, out_path: Path):
    """Build a fake swarm_fn that writes `md` to out_path + a metrics row."""

    async def fake_swarm(topic: str, *, max_rounds: int = 4, use_redis: bool = False) -> Path:
        out_path.write_text(md, encoding="utf-8")
        record = {
            "run_id": "fake_run_1",
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
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return out_path

    return fake_swarm


# ---------------------------------------------------------------------------
# Pure-function tests (no swarm call)
# ---------------------------------------------------------------------------


def test_default_per_prompt_timeout_is_3600():
    """C5a: hard research prompts exceed 20 min; the default budget is 1 h."""
    assert BenchConfig(slate_path=Path("x")).per_prompt_timeout_s == 3600


def test_slate_entry_maps_coverage_to_subtopics():
    """_slate_entry_for exposes expected_coverage under expected_subtopics
    so the composite's coverage sub-score has ground-truth targets."""
    entry = _slate_entry_for(_make_prompt())
    assert entry["expected_subtopics"] == [
        "WAL file structure",
        "checkpointing",
        "rollback journal vs WAL",
        "concurrency guarantees",
        "fsync semantics",
    ]
    assert entry["sources"] == []


def test_composite_is_deterministic_float_in_unit_interval():
    """The fixture markdown + slate_entry produce a stable composite in [0,1]."""
    entry = {"expected_subtopics": list(_make_prompt().expected_coverage),
             "sources": _S001_SOURCES}
    r1 = research_quality_composite(_GOOD_MD, entry, prior=[])
    r2 = research_quality_composite(_GOOD_MD, entry, prior=[])
    assert r1 == r2  # determinism
    assert isinstance(r1["composite"], float)
    assert 0.0 <= r1["composite"] <= 1.0
    # All five coverage keywords present -> coverage sub-score is perfect.
    assert r1["coverage"] == 1.0
    # Grounded URL + well-formed arXiv id -> those sub-scores are perfect.
    assert r1["citation_grounding"] == 1.0
    assert r1["arxiv_validity"] == 1.0


# ---------------------------------------------------------------------------
# G1 — run_one_prompt computes + stores the composite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_prompt_populates_functional_composite(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.touch()
    out_path = tmp_path / "out.md"
    fake_swarm = _fake_swarm_writing(_GOOD_MD, metrics_path, out_path)

    prompt = _make_prompt()
    cfg = _make_cfg()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = await run_one_prompt(
        prompt, cfg, run_dir, swarm_fn=fake_swarm, metrics_path=metrics_path, prior=[],
    )

    assert isinstance(result, PromptResult)
    fc = result.functional_composite
    # Sub-scores all present (incl. the H3 'citations' component).
    for key in ("arxiv_validity", "citation_grounding", "citations", "coverage",
                "dedup_overlap", "novelty", "composite"):
        assert key in fc
    assert isinstance(fc["composite"], float)
    assert 0.0 <= fc["composite"] <= 1.0
    assert fc["coverage"] == 1.0  # every keyword present in _GOOD_MD

    # The composite must equal a direct call with the same slate_entry — i.e.
    # the runner really delegates to research_quality_composite, no drift.
    direct = research_quality_composite(_GOOD_MD, _slate_entry_for(prompt), prior=[])
    assert fc == direct


@pytest.mark.asyncio
async def test_run_one_prompt_novelty_drops_with_duplicate_prior(tmp_path):
    """Passing the identical artifact as prior collapses the novelty sub-score,
    proving `prior` is threaded into the scorer."""
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.touch()
    out_path = tmp_path / "out.md"
    fake_swarm = _fake_swarm_writing(_GOOD_MD, metrics_path, out_path)

    prompt = _make_prompt()
    cfg = _make_cfg()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    fresh = await run_one_prompt(
        prompt, cfg, run_dir, swarm_fn=fake_swarm, metrics_path=metrics_path, prior=[],
    )
    dup = await run_one_prompt(
        prompt, cfg, run_dir, swarm_fn=fake_swarm, metrics_path=metrics_path,
        prior=[_GOOD_MD],
    )
    assert fresh.functional_composite["novelty"] == 1.0
    assert dup.functional_composite["novelty"] < 0.05  # near-duplicate of prior
    # Lower novelty -> lower composite (everything else equal).
    assert dup.functional_composite["composite"] < fresh.functional_composite["composite"]


# ---------------------------------------------------------------------------
# G1 — aggregate() surfaces the SHARED CONTRACT
# ---------------------------------------------------------------------------


def _write_row(run_dir: Path, prompt_id: str, composite_dict: dict,
               *, error: str | None = None, timeout: bool = False) -> None:
    row = PromptResult(
        prompt_id=prompt_id,
        run_id=run_dir.name,
        swarm_run_id="x",
        prompt_sha256=hashlib.sha256(prompt_id.encode()).hexdigest(),
        output_md_path="",
        started_utc="2026-06-09T00:00:00+00:00",
        elapsed_s=1.0,
        cost_usd=0.05,
        input_tokens=100,
        output_tokens=50,
        consensus_reached=True,
        rounds_to_consensus=1,
        quality={"coverage": 8, "accuracy": 8, "clarity": 7, "depth": 7, "avg": 7.5},
        functional_composite=composite_dict,
        coverage_hits=5,
        coverage_total=5,
        timeout=timeout,
        error=error,
        mast_flags=[],
    )
    _append_jsonl(run_dir / "results.jsonl", asdict(row))


def test_aggregate_surfaces_functional_composite_contract(tmp_path):
    """summary['functional_composite'] == {'mean': float 0..1 | None,
    'per_prompt': {id: composite}, 'missing': [ids]} — the exact driver
    contract — and the LLM-judge scores stay under
    summary['aggregate_quality'].

    Uses real slate ids (s001/s002) because aggregate()'s by_tier pass looks
    every prompt_id up in the slate.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_row(run_dir, "s001", {**_zero(), "composite": 0.8})
    _write_row(run_dir, "s002", {**_zero(), "composite": 0.6})

    slate = load_slate("bench/slate_v1.toml")
    cfg = _make_cfg()
    summary = aggregate(run_dir, slate, cfg)

    assert "functional_composite" in summary
    fc = summary["functional_composite"]
    assert set(fc.keys()) == {"mean", "per_prompt", "missing"}
    assert isinstance(fc["mean"], float)
    assert 0.0 <= fc["mean"] <= 1.0
    assert fc["mean"] == round((0.8 + 0.6) / 2, 4)
    assert fc["per_prompt"] == {"s001": 0.8, "s002": 0.6}
    assert fc["missing"] == []  # nothing timed out / errored
    assert summary["timeouts"] == []

    # SECONDARY signal retained.
    assert "aggregate_quality" in summary
    assert summary["aggregate_quality"]["avg"]["mean"] > 0


def test_aggregate_excludes_missing_composite_from_mean(tmp_path):
    """C5a: a timed-out prompt's composite is MISSING (None) — excluded from
    the functional mean, never averaged in as 0.0. mean([0.9, None, 0.9])
    must be 0.9, with the missing id surfaced in fc['missing'] and the
    timeout surfaced in summary['timeouts']."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_row(run_dir, "s001", {**_zero(), "composite": 0.9})
    _write_row(run_dir, "s002", _missing(), timeout=True, error="timeout")
    _write_row(run_dir, "s003", {**_zero(), "composite": 0.9})

    slate = load_slate("bench/slate_v1.toml")
    summary = aggregate(run_dir, slate, _make_cfg())
    fc = summary["functional_composite"]
    assert fc["mean"] == 0.9  # NOT (0.9 + 0.0 + 0.9) / 3
    assert "s002" not in fc["per_prompt"]
    assert fc["per_prompt"] == {"s001": 0.9, "s003": 0.9}
    assert fc["missing"] == ["s002"]  # loud, machine-readable
    assert summary["timeouts"] == ["s002"]


def test_aggregate_errored_prompt_is_missing_not_zero(tmp_path):
    """C5a applies to errored (not just timed-out) prompts: excluded from the
    mean, listed in missing, but NOT in summary['timeouts']."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_row(run_dir, "s001", {**_zero(), "composite": 0.8})
    _write_row(run_dir, "s002", _missing(), error="RuntimeError('boom')")

    slate = load_slate("bench/slate_v1.toml")
    summary = aggregate(run_dir, slate, _make_cfg())
    fc = summary["functional_composite"]
    assert fc["mean"] == 0.8
    assert fc["missing"] == ["s002"]
    assert summary["timeouts"] == []  # errored, not timed out


def test_aggregate_all_missing_mean_is_none(tmp_path):
    """C5a: if EVERY prompt is missing, the mean is None — 'no signal', not
    'measured zero quality'. The driver then falls back to the judge avg."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_row(run_dir, "s001", _missing(), timeout=True, error="timeout")
    _write_row(run_dir, "s002", _missing(), timeout=True, error="timeout")

    slate = load_slate("bench/slate_v1.toml")
    summary = aggregate(run_dir, slate, _make_cfg())
    fc = summary["functional_composite"]
    assert fc["mean"] is None  # NOT 0.0
    assert fc["per_prompt"] == {}
    assert fc["missing"] == ["s001", "s002"]
    assert summary["timeouts"] == ["s001", "s002"]


def test_aggregate_handles_legacy_rows_without_composite(tmp_path):
    """Rows predating the wiring (no functional_composite key) are treated as
    missing, not crashed on — keeps aggregate backward-compatible."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # A row missing functional_composite entirely.
    legacy = {
        "prompt_id": "s001", "run_id": run_dir.name, "swarm_run_id": "",
        "prompt_sha256": "x", "output_md_path": "", "started_utc": "t",
        "elapsed_s": 1.0, "cost_usd": 0.05, "input_tokens": 1, "output_tokens": 1,
        "consensus_reached": True, "rounds_to_consensus": 1,
        "quality": {"coverage": 8, "accuracy": 8, "clarity": 8, "depth": 8, "avg": 8.0},
        "coverage_hits": 5, "coverage_total": 5, "timeout": False, "error": None,
        "mast_flags": [],
    }
    _append_jsonl(run_dir / "results.jsonl", legacy)

    slate = load_slate("bench/slate_v1.toml")
    summary = aggregate(run_dir, slate, _make_cfg())
    # No scoreable rows -> empty per_prompt, mean None (not 0.0), row listed
    # as missing, no exception.
    assert summary["functional_composite"] == {
        "mean": None, "per_prompt": {}, "missing": ["s001"],
    }


# ---------------------------------------------------------------------------
# stats.md surfaces the PRIMARY signal
# ---------------------------------------------------------------------------


def test_stats_md_shows_functional_composite(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "run_id": "abc", "started_utc": "x", "ended_utc": "y",
        "slate": {"slate_id": "v1", "sha256": "a" * 64,
                  "prompts_run": 1, "prompt_count": 30,
                  "prompts_timed_out": 0, "prompts_errored": 0},
        "functional_composite": {"mean": 0.77, "per_prompt": {"f001": 0.77}},
        "aggregate_quality": {"avg": {"mean": 7.0, "stddev": 0.0, "ci95": [7.0, 7.0]}},
        "by_tier": {"easy": {"n": 1, "avg_quality": 7.0, "avg_cost": 0.05},
                    "medium": {"n": 0, "avg_quality": 0, "avg_cost": 0},
                    "hard": {"n": 0, "avg_quality": 0, "avg_cost": 0}},
        "mast_flags_summary": {},
        "cost": {"total_usd": 0.05, "input_tokens_total": 100, "output_tokens_total": 50},
    }
    text = write_stats_md(run_dir, summary).read_text()
    assert "Functional research-quality composite" in text
    assert "PRIMARY" in text
    assert "0.77" in text
    assert "SECONDARY" in text  # judge section labelled secondary


def test_stats_md_handles_all_missing_mean_none(tmp_path):
    """C5a: a None mean renders as n/a (never crashes / never prints 0.0) and
    the missing + timed-out prompt ids are surfaced on the page."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "run_id": "abc", "started_utc": "x", "ended_utc": "y",
        "slate": {"slate_id": "v1", "sha256": "a" * 64,
                  "prompts_run": 0, "prompt_count": 2,
                  "prompts_timed_out": 2, "prompts_errored": 0},
        "functional_composite": {"mean": None, "per_prompt": {},
                                 "missing": ["s001", "s002"]},
        "timeouts": ["s001", "s002"],
        "aggregate_quality": {"avg": {"mean": 0.0, "stddev": 0.0, "ci95": [0.0, 0.0]}},
        "by_tier": {"easy": {"n": 0, "avg_quality": 0, "avg_cost": 0},
                    "medium": {"n": 0, "avg_quality": 0, "avg_cost": 0},
                    "hard": {"n": 0, "avg_quality": 0, "avg_cost": 0}},
        "mast_flags_summary": {},
        "cost": {"total_usd": 0.0, "input_tokens_total": 0, "output_tokens_total": 0},
    }
    text = write_stats_md(run_dir, summary).read_text()
    assert "n/a" in text
    assert "Missing (timeout/error" in text
    assert "Timed out: s001, s002" in text
    assert "**0.0**" not in text  # all-missing must never masquerade as zero


# ---------------------------------------------------------------------------
# H4 — robust metrics-record matching + loud MATCHED/MISS logging
# ---------------------------------------------------------------------------


def test_find_metrics_record_matches_despite_topic_drift(tmp_path):
    """Normalized matching: trailing newline, doubled spaces, leading space,
    and case drift in the logged topic must still match the prompt text."""
    metrics_path = tmp_path / "metrics.jsonl"
    logged = {
        "topic": "  explain SQLite  WAL mode  Internals.\n",
        "timestamp": 100.0,
        "total_cost_usd": 0.10,
    }
    metrics_path.write_text(json.dumps(logged) + "\n", encoding="utf-8")

    rec = _find_metrics_record_for_topic(
        "Explain SQLite WAL mode internals.", metrics_path=metrics_path,
    )
    assert rec is not None
    assert rec["total_cost_usd"] == 0.10


def test_find_metrics_record_matches_on_first_120_chars(tmp_path):
    """Long topics: only the first 120 normalized chars are compared, so a
    truncated/embellished tail in metrics.jsonl still matches."""
    head = "word " * 30  # 150 chars collapsed -> first 120 compared
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"topic": head + "logged-tail", "timestamp": 1.0}) + "\n",
        encoding="utf-8",
    )
    rec = _find_metrics_record_for_topic(head + "asked-tail", metrics_path=metrics_path)
    assert rec is not None


def test_find_metrics_record_different_topics_still_miss(tmp_path):
    """Normalization must not over-match: genuinely different topics miss."""
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"topic": "completely different subject", "timestamp": 1.0}) + "\n",
        encoding="utf-8",
    )
    assert _find_metrics_record_for_topic("SQLite WAL", metrics_path=metrics_path) is None


@pytest.mark.asyncio
async def test_run_one_prompt_matches_metrics_despite_drift_and_logs(tmp_path, capsys):
    """End-to-end H4: the swarm logs the topic with whitespace drift; the
    runner still finds the record (cost/tokens populated) and prints the loud
    MATCHED line for the prompt id."""
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.touch()
    out_path = tmp_path / "out.md"

    async def drifting_swarm(topic: str, *, max_rounds: int = 4, use_redis: bool = False) -> Path:
        out_path.write_text(_GOOD_MD, encoding="utf-8")
        record = {
            "run_id": "drift_run",
            "topic": topic.upper() + " \n",  # case + trailing-whitespace drift
            "timestamp": time.time(),
            "total_cost_usd": 0.10,
            "total_input_tokens": 1500,
            "total_output_tokens": 300,
            "consensus_reached": True,
            "rounds_to_consensus": 1,
            "avg_quality": 7.5,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return out_path

    result = await run_one_prompt(
        _make_prompt(), _make_cfg(), tmp_path, swarm_fn=drifting_swarm,
        metrics_path=metrics_path, prior=[],
    )
    assert result.cost_usd == 0.10  # record found despite drift
    assert result.input_tokens == 1500
    assert "metrics record MATCHED for f001" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_one_prompt_metrics_miss_is_loud_not_fatal(tmp_path, capsys):
    """A metrics miss must not crash: cost/tokens stay 0, the prompt still
    gets its functional composite, and the loud MISS line names the prompt."""
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.touch()
    out_path = tmp_path / "out.md"

    async def silent_swarm(topic: str, *, max_rounds: int = 4, use_redis: bool = False) -> Path:
        out_path.write_text(_GOOD_MD, encoding="utf-8")
        return out_path  # writes NO metrics record

    result = await run_one_prompt(
        _make_prompt(), _make_cfg(), tmp_path, swarm_fn=silent_swarm,
        metrics_path=metrics_path, prior=[],
    )
    assert result.error is None  # no crash
    assert result.cost_usd == 0.0
    assert result.input_tokens == 0 and result.output_tokens == 0
    assert result.functional_composite["composite"] is not None  # still scored
    assert "metrics record MISS for f001" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# C5a — timeout produces a MISSING composite on the row itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_prompt_timeout_composite_is_missing(tmp_path):
    """A timed-out prompt's functional_composite is all-None (MISSING), so
    aggregate() can exclude it from the mean instead of averaging 0.0."""
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.touch()

    async def hang_swarm(topic: str, *, max_rounds: int = 4, use_redis: bool = False) -> Path:
        await asyncio.sleep(30)
        raise AssertionError("unreachable")

    cfg = _make_cfg()
    cfg.per_prompt_timeout_s = 0.05
    result = await run_one_prompt(
        _make_prompt(), cfg, tmp_path, swarm_fn=hang_swarm,
        metrics_path=metrics_path, prior=[],
    )
    assert result.timeout is True
    assert result.functional_composite["composite"] is None
    assert all(v is None for v in result.functional_composite.values())


# ---------------------------------------------------------------------------
# G6 — best-effort MAST detector hook
# ---------------------------------------------------------------------------


def test_load_round_outputs_absent_returns_empty(tmp_path):
    out = tmp_path / "out.md"
    out.write_text("artifact", encoding="utf-8")
    assert _load_round_outputs(out) == []


def test_load_round_outputs_non_path_returns_empty():
    assert _load_round_outputs("not-a-path") == []
    assert _load_round_outputs(None) == []


def test_load_round_outputs_reads_sibling_rounds_json(tmp_path):
    out = tmp_path / "out.md"
    out.write_text("artifact", encoding="utf-8")
    rounds = [
        [{"role": "researcher", "agent_id": "r1", "findings": "alpha bravo charlie"}],
        [{"role": "critic", "agent_id": "c1", "findings": "missing flag concern issue gap"}],
    ]
    out.with_suffix(".rounds.json").write_text(json.dumps(rounds), encoding="utf-8")
    assert _load_round_outputs(out) == rounds


def test_load_round_outputs_malformed_returns_empty(tmp_path):
    out = tmp_path / "out.md"
    out.write_text("artifact", encoding="utf-8")
    out.with_suffix(".rounds.json").write_text("{not valid json", encoding="utf-8")
    assert _load_round_outputs(out) == []


def test_load_round_outputs_wrong_shape_returns_empty(tmp_path):
    out = tmp_path / "out.md"
    out.write_text("artifact", encoding="utf-8")
    # dict, not list-of-lists-of-dicts
    out.with_suffix(".rounds.json").write_text('{"rounds": []}', encoding="utf-8")
    assert _load_round_outputs(out) == []


@pytest.mark.asyncio
async def test_run_one_prompt_flags_empty_without_round_outputs(tmp_path):
    """On a normal run (no sibling rounds.json) mast_flags stays empty —
    we do NOT fabricate failure modes from aggregate metrics (G6 TODO)."""
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.touch()
    out_path = tmp_path / "out.md"
    fake_swarm = _fake_swarm_writing(_GOOD_MD, metrics_path, out_path)

    result = await run_one_prompt(
        _make_prompt(), _make_cfg(), tmp_path, swarm_fn=fake_swarm,
        metrics_path=metrics_path, prior=[],
    )
    assert result.mast_flags == []


@pytest.mark.asyncio
async def test_run_one_prompt_runs_detect_all_with_round_outputs(tmp_path):
    """When a sibling rounds.json with a real failure exists, detect_all fires
    and populates mast_flags. We inject an information-withholding failure
    (researcher with findings but empty key_points -> FM-2.4)."""
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.touch()
    out_path = tmp_path / "out.md"
    fake_swarm = _fake_swarm_writing(_GOOD_MD, metrics_path, out_path)

    # Pre-create the sibling transcript the swarm "would have written".
    rounds = [[
        {"role": "researcher", "agent_id": "r1",
         "findings": "substantive findings here", "key_points": []},
    ]]
    out_path.with_suffix(".rounds.json").write_text(json.dumps(rounds), encoding="utf-8")

    result = await run_one_prompt(
        _make_prompt(), _make_cfg(), tmp_path, swarm_fn=fake_swarm,
        metrics_path=metrics_path, prior=[],
    )
    assert "FM-2.4" in result.mast_flags


# ---------------------------------------------------------------------------
# small local helper
# ---------------------------------------------------------------------------


def _zero() -> dict:
    return {
        "arxiv_validity": 0.0,
        "citation_grounding": 0.0,
        "citations": 0.0,
        "coverage": 0.0,
        "dedup_overlap": 0.0,
        "novelty": 0.0,
        "composite": 0.0,
    }


def _missing() -> dict:
    """Mirror of runner._MISSING_COMPOSITE: what timeout/error rows carry."""
    return {key: None for key in _zero()}
