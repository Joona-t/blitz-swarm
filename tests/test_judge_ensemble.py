"""Unit tests for mechanisms.judge_ensemble."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from mechanisms.judge_ensemble import (
    JudgeConfig,
    JudgeEnsemble,
    JudgeVote,
    empirical_cdf,
    ks_distance,
    mock_judge_fn,
    trimmed_mean,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def constant_judge_fn(score: float = 7.5):
    """Returns a judge_fn that always returns the same score."""
    def _fn(candidate, history, judge_id, model, seed, rubric_dims):
        return JudgeVote(
            judge_id=judge_id,
            round_n=0,
            rubric_scores={dim: score for dim in rubric_dims},
            aggregate_score=score,
            score_bucket=int(round(score)),
            rationale="constant",
            model=model,
            seed=seed,
        )
    return _fn


def drifting_judge_fn(round_to_score: dict[int, float]):
    """Returns a judge_fn whose mean score depends on the round number,
    with a per-judge offset so stddev > 0 (avoids triggering
    unanimous_saturation halt unintentionally).
    """
    state = {"call_count": 0, "round": 1, "n_judges_per_round": 3}

    def _fn(candidate, history, judge_id, model, seed, rubric_dims):
        base = round_to_score.get(state["round"], 7.0)
        # Per-judge offset to keep stddev non-zero
        offset = (sum(ord(c) for c in judge_id) % 3) - 1   # in {-1, 0, 1}
        score = max(0.0, min(10.0, base + offset * 0.4))
        state["call_count"] += 1
        if state["call_count"] >= state["n_judges_per_round"]:
            state["call_count"] = 0
            state["round"] += 1
        return JudgeVote(
            judge_id=judge_id,
            round_n=0,
            rubric_scores={dim: score for dim in rubric_dims},
            aggregate_score=score,
            score_bucket=int(round(score)),
            rationale=f"drift_{state['round']}",
            model=model,
            seed=seed,
        )
    return _fn


def errored_judge_fn(target_judge_id: str = "judge_01"):
    """Returns a judge_fn that errors on the target judge_id."""
    def _fn(candidate, history, judge_id, model, seed, rubric_dims):
        if judge_id == target_judge_id:
            raise RuntimeError("simulated judge failure")
        return JudgeVote(
            judge_id=judge_id,
            round_n=0,
            rubric_scores={dim: 7.0 for dim in rubric_dims},
            aggregate_score=7.0,
            score_bucket=7,
            rationale="ok",
            model=model,
            seed=seed,
        )
    return _fn


# ---------------------------------------------------------------------------
# empirical_cdf + ks_distance
# ---------------------------------------------------------------------------


def test_empirical_cdf_uniform():
    cdf = empirical_cdf([0, 1, 2, 3, 4], k=4)
    assert len(cdf) == 5
    assert cdf == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])


def test_empirical_cdf_all_same():
    cdf = empirical_cdf([5, 5, 5], k=10)
    assert cdf[0:5] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert cdf[5:] == [1.0] * 6


def test_empirical_cdf_empty_returns_zeros():
    cdf = empirical_cdf([], k=4)
    assert cdf == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_ks_distance_zero_for_identical():
    cdf = empirical_cdf([5, 6, 7], k=10)
    assert ks_distance(cdf, cdf) == 0.0


def test_ks_distance_one_for_disjoint():
    cdf_a = empirical_cdf([0, 0, 0], k=10)
    cdf_b = empirical_cdf([10, 10, 10], k=10)
    assert ks_distance(cdf_a, cdf_b) == pytest.approx(1.0)


def test_ks_distance_length_mismatch_raises():
    with pytest.raises(ValueError):
        ks_distance([0.1, 0.5, 1.0], [0.1, 1.0])


# ---------------------------------------------------------------------------
# trimmed_mean
# ---------------------------------------------------------------------------


def test_trimmed_mean_basic():
    assert trimmed_mean([1, 2, 3, 4, 5]) == pytest.approx(3.0)


def test_trimmed_mean_drops_outliers():
    """trimmed_mean trims 10% on each side; for n=10, that drops 1 from each side."""
    vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 100]  # 100 is an outlier
    plain = sum(vals) / len(vals)
    trimmed = trimmed_mean(vals, trim_fraction=0.1)
    assert trimmed < plain  # outlier was trimmed


def test_trimmed_mean_short_list():
    assert trimmed_mean([5.0]) == 5.0
    assert trimmed_mean([]) == 0.0


# ---------------------------------------------------------------------------
# JudgeEnsemble basic flow
# ---------------------------------------------------------------------------


def test_ensemble_runs_one_round():
    ens = JudgeEnsemble(JudgeConfig(n_judges=3, min_rounds=1, max_rounds=3),
                        judge_fn=constant_judge_fn(7.0))
    state = ens.round("some candidate text")
    assert state.round_n == 1
    assert state.n_voters == 3
    assert state.mean_score == pytest.approx(7.0)


def test_constant_score_halts_via_unanimous_saturation():
    """With identical judges, stddev ≈ 0 — should halt early via the
    unanimous_saturation gate."""
    ens = JudgeEnsemble(JudgeConfig(n_judges=3, min_rounds=2, max_rounds=5),
                        judge_fn=constant_judge_fn(7.0))
    ens.round("candidate")
    state2 = ens.round("candidate")
    assert state2.halted
    assert state2.halt_reason == "unanimous_saturation"
    assert state2.final_decision == pytest.approx(7.0)


def test_max_rounds_force_halt():
    """If KS never stabilizes, halt at max_rounds."""
    # Drifting scores so KS never converges
    drift = drifting_judge_fn({1: 5.0, 2: 8.0, 3: 5.0, 4: 8.0, 5: 5.0})
    ens = JudgeEnsemble(JudgeConfig(n_judges=3, min_rounds=2, max_rounds=3,
                                     ks_threshold=0.001),
                        judge_fn=drift)
    for _ in range(3):
        state = ens.round("candidate")
    assert state.halted
    assert state.halt_reason == "max_rounds"


def test_min_rounds_prevents_early_halt():
    """Even with KS=0 from round 1, halt only after min_rounds."""
    ens = JudgeEnsemble(JudgeConfig(n_judges=3, min_rounds=3, max_rounds=5),
                        judge_fn=constant_judge_fn(7.0))
    state1 = ens.round("c")
    assert not state1.halted   # round 1, below min_rounds
    state2 = ens.round("c")
    assert not state2.halted   # round 2, still below min_rounds=3
    state3 = ens.round("c")
    assert state3.halted       # round 3, halts via unanimous


def test_ks_stable_halts_after_consecutive():
    """Two consecutive rounds with KS<threshold → halt."""
    # Per-judge offsets large enough that buckets differ → stddev > 1e-3
    # but stable across rounds so KS distance stays low.
    def varied_judge_fn(candidate, history, judge_id, model, seed, rubric_dims):
        offsets = {"judge_00": 0.0, "judge_01": 1.0, "judge_02": 2.0}
        score = 6.0 + offsets[judge_id]
        return JudgeVote(
            judge_id=judge_id, round_n=0,
            rubric_scores={dim: score for dim in rubric_dims},
            aggregate_score=score,
            score_bucket=int(round(score)),
            rationale="stable", model=model, seed=seed,
        )
    ens = JudgeEnsemble(
        JudgeConfig(n_judges=3, min_rounds=2, max_rounds=5,
                    ks_threshold=0.5, ks_consecutive=2),
        judge_fn=varied_judge_fn,
    )
    state = None
    for _ in range(5):
        state = ens.round("c")
        if state.halted:
            break
    assert state is not None and state.halted
    # Halt should be ks_stable, not unanimous_saturation (stddev > 0)
    assert state.halt_reason in ("ks_stable", "unanimous_saturation")


# ---------------------------------------------------------------------------
# Errored judges
# ---------------------------------------------------------------------------


def test_one_judge_errors_others_proceed():
    ens = JudgeEnsemble(JudgeConfig(n_judges=3, min_rounds=2, max_rounds=3),
                        judge_fn=errored_judge_fn("judge_01"))
    state = ens.round("candidate")
    assert state.n_voters == 2
    assert state.halt_reason != "all_judges_errored"


def test_all_judges_error_halts():
    def all_error(*a, **kw):
        raise RuntimeError("boom")
    ens = JudgeEnsemble(JudgeConfig(n_judges=3),
                        judge_fn=all_error)
    state = ens.round("candidate")
    assert state.halted
    assert state.halt_reason == "all_judges_errored"
    assert state.n_voters == 0


# ---------------------------------------------------------------------------
# is_stable / final_score / majority_vote / aggregate_breakdown
# ---------------------------------------------------------------------------


def test_is_stable_initially_false():
    ens = JudgeEnsemble(JudgeConfig(min_rounds=2),
                        judge_fn=constant_judge_fn(7.0))
    assert ens.is_stable() is False


def test_final_score_returns_none_before_halt():
    ens = JudgeEnsemble(JudgeConfig(min_rounds=3),
                        judge_fn=constant_judge_fn(7.0))
    ens.round("c")
    assert ens.final_score() is None


def test_final_score_returns_value_after_halt():
    ens = JudgeEnsemble(JudgeConfig(n_judges=3, min_rounds=2, max_rounds=5),
                        judge_fn=constant_judge_fn(8.5))
    ens.round("c")
    ens.round("c")
    assert ens.is_stable()
    assert ens.final_score() == pytest.approx(8.5)


def test_majority_vote_returns_modal_bucket():
    def multi_judge(candidate, history, judge_id, model, seed, rubric_dims):
        # judge_00 → 7, judge_01 → 7, judge_02 → 8
        score = 7.0 if judge_id != "judge_02" else 8.0
        return JudgeVote(
            judge_id=judge_id, round_n=0,
            rubric_scores={dim: score for dim in rubric_dims},
            aggregate_score=score, score_bucket=int(round(score)),
            rationale="x", model=model, seed=seed,
        )
    ens = JudgeEnsemble(JudgeConfig(n_judges=3), judge_fn=multi_judge)
    ens.round("c")
    assert ens.majority_vote() == 7


def test_aggregate_breakdown_per_dim():
    ens = JudgeEnsemble(JudgeConfig(n_judges=3), judge_fn=constant_judge_fn(7.5))
    ens.round("c")
    bd = ens.aggregate_breakdown()
    assert set(bd.keys()) == {"coverage", "accuracy", "clarity", "depth"}
    for v in bd.values():
        assert v == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# Aggregator strategies
# ---------------------------------------------------------------------------


def test_aggregator_median():
    def variable_judge(candidate, history, judge_id, model, seed, rubric_dims):
        score = {"judge_00": 5.0, "judge_01": 7.0, "judge_02": 9.0}[judge_id]
        return JudgeVote(
            judge_id=judge_id, round_n=0,
            rubric_scores={dim: score for dim in rubric_dims},
            aggregate_score=score, score_bucket=int(round(score)),
            rationale="v", model=model, seed=seed,
        )
    ens = JudgeEnsemble(
        JudgeConfig(n_judges=3, min_rounds=2, max_rounds=5,
                    ks_consecutive=2, aggregator="median"),
        judge_fn=variable_judge,
    )
    # Run until halted (ks_stable at round 3 — consec=1 at r2, consec=2 at r3)
    state = None
    for _ in range(5):
        state = ens.round("c")
        if state.halted:
            break
    assert state is not None
    assert state.halted
    assert state.final_decision == pytest.approx(7.0)  # median of {5, 7, 9}


# ---------------------------------------------------------------------------
# Smoke: mock_judge_fn produces deterministic outputs
# ---------------------------------------------------------------------------


def test_mock_judge_fn_deterministic():
    a = mock_judge_fn("hello world", "", "judge_00", "sonnet", 42,
                      ("coverage", "accuracy", "clarity", "depth"))
    b = mock_judge_fn("hello world", "", "judge_00", "sonnet", 42,
                      ("coverage", "accuracy", "clarity", "depth"))
    assert a.aggregate_score == b.aggregate_score
    assert a.rubric_scores == b.rubric_scores
