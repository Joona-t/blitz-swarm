"""Multi-judge debate with adaptive stability detection.

Hu et al. arXiv 2510.12697 (NeurIPS 2025) — "Multi-Agent Debate for LLM
Judges with Adaptive Stability Detection." Replaces the single quality
judge with N (default 3) judges who run iterative debate rounds, each
producing a rubric-scored vote. The ensemble halts when KS distance
between consecutive rounds' score distributions falls below a threshold
for `ks_consecutive` rounds — adaptive instead of fixed-iteration.

Implementation note: this module ships the empirical-CDF KS variant
(simpler, scipy-free) instead of the paper's parametric Beta-Binomial
mixture. KS over the empirical CDF gives the same halting behavior at
the small N (3-7) the swarm uses. The BB-mixture variant can be
swapped in via JudgeConfig.distribution = "bb_mixture" once the
parametric estimator earns its complexity.

Pluggable LLM hook: `judge_fn(candidate, history, judge_id, model, seed,
rubric)` returns a JudgeVote. Default mock implementation is for tests;
orchestrator integration replaces with a Claude CLI subprocess.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class JudgeConfig:
    n_judges: int = 3
    max_rounds: int = 5                  # paper used 10; we cap at 5
    ks_threshold: float = 0.05
    ks_consecutive: int = 2
    min_rounds: int = 2
    score_buckets: int = 11              # 0..10 maps to k=10
    seed_strategy: Literal["prompt_seed", "model_mix", "both"] = "prompt_seed"
    judge_models: tuple[str, ...] = ("sonnet", "sonnet", "haiku")
    rubric_dims: tuple[str, ...] = ("coverage", "accuracy", "clarity", "depth")
    distribution: Literal["empirical", "bb_mixture"] = "empirical"
    aggregator: Literal["mean", "median", "trimmed_mean"] = "mean"

    @property
    def k(self) -> int:
        """Max bucket index = score_buckets - 1."""
        return self.score_buckets - 1


@dataclass
class JudgeVote:
    judge_id: str
    round_n: int
    rubric_scores: dict[str, float]      # criterion -> 0..10
    aggregate_score: float                # 0..10
    score_bucket: int                     # 0..k
    rationale: str
    model: str = ""
    seed: int = 0
    elapsed_s: float = 0.0
    errored: bool = False


@dataclass
class StabilityState:
    round_n: int
    n_voters: int
    mean_score: float
    median_score: float
    stddev: float
    ks_stat: float
    consecutive_below_threshold: int
    halted: bool
    halt_reason: str = ""
    final_decision: float | None = None


# ---------------------------------------------------------------------------
# Pluggable LLM hook
# ---------------------------------------------------------------------------


JudgeFn = Callable[[str, str, str, str, int, tuple[str, ...]], JudgeVote]


def _stable_score_for(candidate: str, judge_id: str, seed: int) -> float:
    """Deterministic stub score for tests — varies slightly by seed."""
    base = (sum(ord(c) for c in candidate[:200]) % 30) / 10.0 + 5.0  # 5..8
    perturbation = ((seed * 17 + sum(ord(c) for c in judge_id)) % 20 - 10) / 50.0
    return max(0.0, min(10.0, base + perturbation))


def mock_judge_fn(
    candidate: str,
    history: str,
    judge_id: str,
    model: str,
    seed: int,
    rubric_dims: tuple[str, ...],
) -> JudgeVote:
    """Deterministic mock judge for tests. Score is a stable hash of
    (candidate, judge_id, seed). Each rubric dim gets a small per-dim offset."""
    base = _stable_score_for(candidate, judge_id, seed)
    rubric = {dim: max(0.0, min(10.0, base + (i - 1) * 0.3))
              for i, dim in enumerate(rubric_dims)}
    aggregate = sum(rubric.values()) / max(len(rubric), 1)
    return JudgeVote(
        judge_id=judge_id,
        round_n=0,
        rubric_scores=rubric,
        aggregate_score=aggregate,
        score_bucket=int(round(aggregate)),
        rationale="mock judge rationale",
        model=model,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Empirical-CDF KS distance
# ---------------------------------------------------------------------------


def empirical_cdf(buckets: Iterable[int], k: int) -> list[float]:
    """Empirical CDF over discrete buckets [0..k]. Returns list of length k+1."""
    bucket_list = list(buckets)
    n = max(len(bucket_list), 1)
    cdf: list[float] = []
    cumulative = 0
    for s in range(k + 1):
        cumulative += sum(1 for b in bucket_list if b == s)
        cdf.append(cumulative / n)
    return cdf


def ks_distance(cdf_a: list[float], cdf_b: list[float]) -> float:
    """Kolmogorov-Smirnov distance between two discrete CDFs."""
    if not cdf_a or not cdf_b:
        return 1.0
    if len(cdf_a) != len(cdf_b):
        raise ValueError(f"CDF length mismatch: {len(cdf_a)} vs {len(cdf_b)}")
    return max(abs(a - b) for a, b in zip(cdf_a, cdf_b))


def trimmed_mean(values: list[float], trim_fraction: float = 0.1) -> float:
    """Mean with the most extreme `trim_fraction` of values dropped on each side."""
    if not values:
        return 0.0
    if len(values) <= 2:
        return statistics.mean(values)
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cut = max(1, int(n * trim_fraction))
    trimmed = sorted_vals[cut:n - cut] if n > 2 * cut else sorted_vals
    return statistics.mean(trimmed) if trimmed else statistics.mean(sorted_vals)


# ---------------------------------------------------------------------------
# JudgeEnsemble
# ---------------------------------------------------------------------------


class JudgeEnsemble:
    """N-judge debate with KS-stability halt.

    Lifecycle:
        ensemble = JudgeEnsemble(cfg, judge_fn=...)
        while not ensemble.is_stable():
            state = ensemble.round(candidate, debate_history)
            if state.halted: break
        score = ensemble.final_score()

    Returns the same 0..10 aggregate score the v0.1.1 single-judge
    produced, so downstream code doesn't need to change. New: per-judge
    breakdown, stability log, halt reason all available via `state_log`.
    """

    def __init__(
        self,
        cfg: JudgeConfig | None = None,
        *,
        judge_fn: JudgeFn | None = None,
    ):
        self.cfg = cfg or JudgeConfig()
        self.judge_fn = judge_fn or mock_judge_fn
        self.history: list[list[JudgeVote]] = []
        self.state_log: list[StabilityState] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def round(self, candidate: str, debate_history: str = "") -> StabilityState:
        """Run one debate round. Returns the stability state."""
        round_n = len(self.history) + 1
        votes = self._cast_votes(candidate, debate_history, round_n)
        self.history.append(votes)
        state = self._update_state(votes, round_n)
        self.state_log.append(state)
        return state

    def is_stable(self) -> bool:
        return bool(self.state_log) and self.state_log[-1].halted

    def final_score(self) -> float | None:
        if not self.is_stable():
            return None
        return self.state_log[-1].final_decision

    def majority_vote(self) -> int:
        """Return the modal score bucket from the latest round."""
        if not self.history:
            return 0
        latest_buckets = [v.score_bucket for v in self.history[-1]
                          if not v.errored]
        if not latest_buckets:
            return 0
        counts: dict[int, int] = {}
        for b in latest_buckets:
            counts[b] = counts.get(b, 0) + 1
        return max(counts, key=lambda k: (counts[k], k))

    def aggregate_breakdown(self) -> dict[str, float]:
        """Return per-rubric-dim mean across the latest round's voters."""
        if not self.history:
            return {dim: 0.0 for dim in self.cfg.rubric_dims}
        latest = [v for v in self.history[-1] if not v.errored]
        if not latest:
            return {dim: 0.0 for dim in self.cfg.rubric_dims}
        return {
            dim: statistics.mean(
                v.rubric_scores.get(dim, 0.0) for v in latest
            )
            for dim in self.cfg.rubric_dims
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cast_votes(
        self, candidate: str, history: str, round_n: int
    ) -> list[JudgeVote]:
        """Invoke each judge once for this round."""
        votes: list[JudgeVote] = []
        for i in range(self.cfg.n_judges):
            judge_id = f"judge_{i:02d}"
            seed = self._seed_for(i, round_n)
            model = self.cfg.judge_models[i % len(self.cfg.judge_models)]
            t0 = time.monotonic()
            try:
                vote = self.judge_fn(
                    candidate, history, judge_id, model, seed,
                    self.cfg.rubric_dims,
                )
                vote.round_n = round_n
                vote.elapsed_s = time.monotonic() - t0
            except Exception as e:
                vote = JudgeVote(
                    judge_id=judge_id,
                    round_n=round_n,
                    rubric_scores={},
                    aggregate_score=0.0,
                    score_bucket=0,
                    rationale=f"judge error: {e!r}",
                    model=model,
                    seed=seed,
                    elapsed_s=time.monotonic() - t0,
                    errored=True,
                )
            votes.append(vote)
        return votes

    def _seed_for(self, judge_idx: int, round_n: int) -> int:
        """Rotate seed assignments across rounds to control position bias."""
        # Standard prompt_seed strategy: judge i gets seed (i + round_n) mod N
        return (judge_idx + round_n) % max(self.cfg.n_judges, 1)

    def _update_state(
        self, votes: list[JudgeVote], round_n: int
    ) -> StabilityState:
        valid = [v for v in votes if not v.errored]
        n_voters = len(valid)
        if n_voters == 0:
            # All judges errored → can't proceed
            prev = self.state_log[-1] if self.state_log else None
            return StabilityState(
                round_n=round_n, n_voters=0,
                mean_score=0.0, median_score=0.0, stddev=0.0,
                ks_stat=1.0,
                consecutive_below_threshold=0,
                halted=True,
                halt_reason="all_judges_errored",
                final_decision=prev.final_decision if prev else None,
            )

        scores = [v.aggregate_score for v in valid]
        buckets = [v.score_bucket for v in valid]
        mean_score = statistics.mean(scores)
        median_score = statistics.median(scores)
        stddev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        # KS distance vs prior round's CDF
        ks_stat = 0.0
        if round_n >= 2 and len(self.history) >= 2:
            prev_buckets = [
                v.score_bucket for v in self.history[-2] if not v.errored
            ]
            if prev_buckets:
                cdf_prev = empirical_cdf(prev_buckets, self.cfg.k)
                cdf_curr = empirical_cdf(buckets, self.cfg.k)
                ks_stat = ks_distance(cdf_prev, cdf_curr)
        else:
            ks_stat = 1.0  # round 1 always has "max change" since no prior

        # Consecutive-below counter
        prev_consec = self.state_log[-1].consecutive_below_threshold \
            if self.state_log else 0
        consec = prev_consec + 1 if ks_stat < self.cfg.ks_threshold else 0

        # Halt condition
        halted = False
        halt_reason = ""

        # Early-halt for unanimous saturation: variance ≈ 0
        if round_n >= self.cfg.min_rounds and stddev < 1e-3:
            halted = True
            halt_reason = "unanimous_saturation"
        elif (round_n >= self.cfg.min_rounds
              and consec >= self.cfg.ks_consecutive):
            halted = True
            halt_reason = "ks_stable"
        elif round_n >= self.cfg.max_rounds:
            halted = True
            halt_reason = "max_rounds"

        final_decision = (
            self._aggregate(scores) if halted else None
        )

        return StabilityState(
            round_n=round_n,
            n_voters=n_voters,
            mean_score=mean_score,
            median_score=median_score,
            stddev=stddev,
            ks_stat=ks_stat,
            consecutive_below_threshold=consec,
            halted=halted,
            halt_reason=halt_reason,
            final_decision=final_decision,
        )

    def _aggregate(self, scores: list[float]) -> float:
        if not scores:
            return 0.0
        if self.cfg.aggregator == "median":
            return statistics.median(scores)
        if self.cfg.aggregator == "trimmed_mean":
            return trimmed_mean(scores, 0.1)
        return statistics.mean(scores)
