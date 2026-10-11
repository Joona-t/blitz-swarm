"""Tests for token totals in mythos.runner._build_metrics (C6).

The blitz-upgrade driver (jobs/blitz_upgrade._read_mythos_tokens) reads
"total_input_tokens"/"total_output_tokens" from the metrics.json that
mythos writes per run. _build_metrics must always emit both keys, summing
whatever per-round usage the history entries carry (top-level
input_tokens/output_tokens or the same keys nested under "usage"), and
honestly writing 0 when usage is absent — never fabricating.
"""

from __future__ import annotations

import json
import time

from mythos.policies import CostBudget
from mythos.runner import _build_metrics


def _budget(spent: float = 0.123, ceiling: float = 5.0) -> CostBudget:
    b = CostBudget(ceiling_usd=ceiling)
    b.charge(spent)
    return b


# ---------------------------------------------------------------------------
# Keys always present
# ---------------------------------------------------------------------------


def test_token_keys_always_present_even_with_empty_history():
    metrics = _build_metrics([], _budget(), time.monotonic())
    assert metrics["total_input_tokens"] == 0
    assert metrics["total_output_tokens"] == 0


def test_token_keys_present_with_current_runner_history_shape():
    # Exactly what run_mythos appends today: cost fields only, no usage.
    # Sums must be honest zeros, not fabricated.
    history = [
        {
            "round": 0,
            "verdict": "needs_work",
            "exec_cost_usd": 0.01,
            "verifier_cost_usd": 0.002,
            "cumulative_cost_usd": 0.012,
        },
        {
            "round": 1,
            "verdict": "pass",
            "exec_cost_usd": 0.01,
            "verifier_cost_usd": 0.002,
            "cumulative_cost_usd": 0.024,
        },
    ]
    metrics = _build_metrics(history, _budget(), time.monotonic())
    assert metrics["total_input_tokens"] == 0
    assert metrics["total_output_tokens"] == 0


def test_existing_metrics_fields_unchanged():
    budget = _budget(spent=0.5, ceiling=5.0)
    history = [{"round": 0, "verdict": "pass", "input_tokens": 10, "output_tokens": 5}]
    metrics = _build_metrics(history, budget, time.monotonic())
    assert metrics["rounds"] is history
    assert metrics["total_cost_usd"] == 0.5
    assert metrics["cost_ceiling_usd"] == 5.0
    assert isinstance(metrics["wall_clock_s"], float)


# ---------------------------------------------------------------------------
# Sums match a synthetic history fixture
# ---------------------------------------------------------------------------


def test_sums_match_synthetic_history():
    history = [
        {"round": 0, "verdict": "needs_work", "input_tokens": 1000, "output_tokens": 250},
        {"round": 1, "verdict": "needs_work", "input_tokens": 2500, "output_tokens": 750},
        {"round": 2, "verdict": "pass", "input_tokens": 500, "output_tokens": 125},
    ]
    metrics = _build_metrics(history, _budget(), time.monotonic())
    assert metrics["total_input_tokens"] == 4000
    assert metrics["total_output_tokens"] == 1125


def test_sums_include_usage_nested_entries():
    history = [
        {"round": 0, "input_tokens": 100, "output_tokens": 10},
        {"round": 1, "usage": {"input_tokens": 200, "output_tokens": 20}},
        {"round": 2},  # no usage at all -> contributes 0
    ]
    metrics = _build_metrics(history, _budget(), time.monotonic())
    assert metrics["total_input_tokens"] == 300
    assert metrics["total_output_tokens"] == 30


def test_top_level_tokens_preferred_over_nested_usage():
    history = [
        {"round": 0, "input_tokens": 7, "output_tokens": 3,
         "usage": {"input_tokens": 9999, "output_tokens": 9999}},
    ]
    metrics = _build_metrics(history, _budget(), time.monotonic())
    assert metrics["total_input_tokens"] == 7
    assert metrics["total_output_tokens"] == 3


def test_junk_entries_tolerated():
    history = [
        "not-a-dict",
        None,
        {"round": 0, "input_tokens": "not-an-int", "output_tokens": None},
        {"round": 1, "usage": "also-not-a-dict"},
        {"round": 2, "input_tokens": 50, "output_tokens": 5},
    ]
    metrics = _build_metrics(history, _budget(), time.monotonic())
    assert metrics["total_input_tokens"] == 50
    assert metrics["total_output_tokens"] == 5


# ---------------------------------------------------------------------------
# Contract with the driver
# ---------------------------------------------------------------------------


def test_driver_key_names_and_json_serializable():
    # jobs/blitz_upgrade._read_mythos_tokens sums these exact keys from
    # metrics.json; the metrics dict must serialize cleanly for
    # artifact.write_metrics.
    history = [{"round": 0, "input_tokens": 12, "output_tokens": 34}]
    metrics = _build_metrics(history, _budget(), time.monotonic())
    assert {"total_input_tokens", "total_output_tokens"} <= set(metrics)
    rec = json.loads(json.dumps(metrics))
    total = sum(
        int(rec.get(k, 0) or 0)
        for k in ("total_input_tokens", "input_tokens",
                  "total_output_tokens", "output_tokens")
    )
    assert total == 46
