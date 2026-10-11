"""Meta-loop: config-patch proposer with bench-validated auto-merge.

Reads insights tagged `meta:` from gmemory.meta, proposes config edits,
runs the bench under each candidate patch, and auto-merges only if all
guards pass:
    - Cohen's d_z >= auto_merge_threshold (default 0.4)
    - No per-dim regression > regression_bound (default 0.3)
    - Cost <= 1.5 × baseline cost
    - High-risk paths require human review (escalate to L3)
    - Holdout slate must not regress (anti-bench-gaming)

Schema-validated patches via ALLOWED_KEY_PATHS — only specific TOML keys
are tunable by L2; everything else escalates to L3. Every key path names a
real `config.BlitzConfig` field (`<section>.<field>`); tests walk the
dataclass to assert this (evolve-01).

EXPERIMENTAL, UNWIRED: `evaluate_patch` / `propose_patches_from_insights`
refuse unless `[evolve] enabled = true`.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from evolve import require_enabled


class PatchScope(Enum):
    GLOBAL = "global"
    DOMAIN = "domain"
    TOPIC = "topic"


# Mapping: "<section>.<field>" of config.BlitzConfig -> (type, allowed range or set, risk_class)
ALLOWED_KEY_PATHS: dict[str, tuple] = {
    "swarm.max_rounds":                (int, range(1, 10), "high"),
    "swarm.max_agents":                (int, range(2, 26), "medium"),
    "swarm.persona_critics":           (bool, (True, False), "low"),
    "judge_ensemble.n_judges":         (int, range(1, 9), "low"),
    "judge_ensemble.min_rounds":       (int, range(1, 11), "low"),
    "judge_ensemble.ks_threshold":     (float, (0.001, 0.5), "medium"),
    "memory.top_k_retrieval":          (int, range(1, 13), "low"),
    "memory.query_link_threshold":     (float, (0.5, 0.95), "medium"),
    "evolve.gepa_max_metric_calls":    (int, range(100, 5001), "low"),
    "evolve.aflow_max_iters":          (int, range(20, 601), "low"),
    "guard.mode":                      (str, {"off", "speed", "balanced", "strict", "max"}, "medium"),
    "selector.granularity":            (str, {"whole", "section", "paragraph"}, "low"),
}


@dataclass(slots=True)
class ConfigPatch:
    patch_id: str
    scope: PatchScope
    scope_value: str | None
    key_path: str
    old_value: Any
    new_value: Any
    rationale: str
    source_insights: list[str] = field(default_factory=list)
    risk_class: str = "low"
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class PatchEvaluation:
    patch: ConfigPatch
    baseline_score: float
    candidate_score: float
    delta: float
    per_dim_regressions: list[str]
    cohens_d: float
    cost_ratio: float
    holdout_score_delta: float
    decision: str       # "merged", "rejected", "human_review"
    reason: str = ""


@dataclass(slots=True)
class MetaLoopConfig:
    auto_merge_threshold: float = 0.4   # Cohen's d_z required
    regression_bound: float = 0.3        # max per-dim drop allowed
    cost_ratio_cap: float = 1.5
    holdout_min_delta: float = -0.2      # holdout can't drop more than this
    high_risk_paths: tuple[str, ...] = ("swarm.max_rounds",)


class SchemaValidationError(Exception):
    pass


def validate_patch(patch: ConfigPatch) -> None:
    """Raise SchemaValidationError if the patch violates ALLOWED_KEY_PATHS."""
    spec = ALLOWED_KEY_PATHS.get(patch.key_path)
    if spec is None:
        raise SchemaValidationError(
            f"key_path {patch.key_path!r} is not in the L2 allow-list "
            f"(escalate to L3 review)"
        )
    expected_type, allowed, _risk = spec
    if not isinstance(patch.new_value, expected_type):
        raise SchemaValidationError(
            f"{patch.key_path}: expected {expected_type.__name__}, got "
            f"{type(patch.new_value).__name__}"
        )
    if isinstance(allowed, range):
        if patch.new_value not in allowed:
            raise SchemaValidationError(
                f"{patch.key_path}={patch.new_value} out of range "
                f"[{allowed.start}, {allowed.stop - 1}]"
            )
    elif isinstance(allowed, set):
        if patch.new_value not in allowed:
            raise SchemaValidationError(
                f"{patch.key_path}={patch.new_value!r} not in allowed set "
                f"{sorted(allowed)}"
            )
    elif isinstance(allowed, tuple) and len(allowed) == 2 and all(
        isinstance(v, (int, float)) for v in allowed
    ):
        lo, hi = allowed
        if not (lo <= patch.new_value <= hi):
            raise SchemaValidationError(
                f"{patch.key_path}={patch.new_value} out of range [{lo}, {hi}]"
            )


# Pluggable hooks
BenchScoreFn = Callable[[dict], dict]   # config -> {"aggregate": float, "per_dim": dict, "cost": float}


def evaluate_patch(
    patch: ConfigPatch,
    *,
    baseline_score: dict,
    candidate_score: dict,
    holdout_score: dict | None = None,
    cfg: MetaLoopConfig | None = None,
    blitz_cfg=None,
) -> PatchEvaluation:
    """Decide merged / rejected / human_review based on guards.

    Gated: raises `evolve.EvolveDisabled` unless `[evolve] enabled = true`.
    """
    require_enabled(blitz_cfg, what="evolve.meta_loop.evaluate_patch")
    cfg = cfg or MetaLoopConfig()

    base_agg = float(baseline_score.get("aggregate", 0.0))
    cand_agg = float(candidate_score.get("aggregate", 0.0))
    delta = cand_agg - base_agg

    # Per-dim regression check
    regressions: list[str] = []
    base_dims = baseline_score.get("per_dim", {})
    cand_dims = candidate_score.get("per_dim", {})
    for dim in base_dims:
        if dim in cand_dims:
            drop = float(base_dims[dim]) - float(cand_dims[dim])
            if drop > cfg.regression_bound:
                regressions.append(f"{dim}:-{drop:.2f}")

    # Cost ratio
    base_cost = max(float(baseline_score.get("cost", 0.0)), 1e-9)
    cand_cost = float(candidate_score.get("cost", 0.0))
    cost_ratio = cand_cost / base_cost

    # Holdout delta
    holdout_delta = 0.0
    if holdout_score is not None:
        h_base = float(holdout_score.get("baseline", 0.0))
        h_cand = float(holdout_score.get("candidate", 0.0))
        holdout_delta = h_cand - h_base

    # Cohen's d_z proxy: delta / pooled_stddev (using per-dim stddev average)
    base_sd = float(baseline_score.get("stddev", 1.0))
    if base_sd > 0:
        cohens_d = delta / base_sd
    else:
        cohens_d = float("inf") if delta > 0 else 0.0

    # Decision
    decision = "human_review"
    reason = ""
    if patch.key_path in cfg.high_risk_paths or patch.risk_class == "high":
        reason = f"high-risk path {patch.key_path}; L3 audit required"
        decision = "human_review"
    elif regressions:
        reason = f"per-dim regressions: {','.join(regressions)}"
        decision = "rejected"
    elif cost_ratio > cfg.cost_ratio_cap:
        reason = f"cost ratio {cost_ratio:.2f} > cap {cfg.cost_ratio_cap}"
        decision = "rejected"
    elif holdout_delta < cfg.holdout_min_delta:
        reason = f"holdout regressed by {-holdout_delta:.2f}"
        decision = "rejected"
    elif cohens_d >= cfg.auto_merge_threshold:
        reason = f"d_z={cohens_d:.2f} >= threshold {cfg.auto_merge_threshold}"
        decision = "merged"
    else:
        reason = f"d_z={cohens_d:.2f} below threshold; queued for review"
        decision = "human_review"

    return PatchEvaluation(
        patch=patch,
        baseline_score=base_agg,
        candidate_score=cand_agg,
        delta=delta,
        per_dim_regressions=regressions,
        cohens_d=cohens_d,
        cost_ratio=cost_ratio,
        holdout_score_delta=holdout_delta,
        decision=decision,
        reason=reason,
    )


def propose_patches_from_insights(
    insights: list[dict],
    *,
    max_patches: int = 5,
    blitz_cfg=None,
) -> list[ConfigPatch]:
    """Heuristic patch proposer. Gated by `[evolve] enabled`.

    Scans `meta:` insights for known patterns (e.g., `meta:gepa-converged`,
    `meta:judge-variance-high`) and produces ConfigPatch suggestions.
    Real implementation calls an LLM to synthesize patches from arbitrary
    insight content.
    """
    require_enabled(blitz_cfg, what="evolve.meta_loop.propose_patches_from_insights")
    patches: list[ConfigPatch] = []
    seen_keys: set[str] = set()

    for ins in insights:
        if len(patches) >= max_patches:
            break
        content = (ins.get("content") or "").lower()
        ins_id = ins.get("id", "?")

        if "judge variance" in content or "judge-variance-high" in content:
            if "judge_ensemble.n_judges" not in seen_keys:
                patches.append(ConfigPatch(
                    patch_id=str(uuid.uuid4()),
                    scope=PatchScope.GLOBAL, scope_value=None,
                    key_path="judge_ensemble.n_judges",
                    old_value=1, new_value=3,
                    rationale="meta-loop: high judge variance observed",
                    source_insights=[ins_id], risk_class="low",
                ))
                seen_keys.add("judge_ensemble.n_judges")

        elif "gepa converged" in content or "gepa-converged" in content:
            if "evolve.gepa_max_metric_calls" not in seen_keys:
                patches.append(ConfigPatch(
                    patch_id=str(uuid.uuid4()),
                    scope=PatchScope.GLOBAL, scope_value=None,
                    key_path="evolve.gepa_max_metric_calls",
                    old_value=1500, new_value=800,
                    rationale="meta-loop: GEPA converged at low K; reducing budget",
                    source_insights=[ins_id], risk_class="low",
                ))
                seen_keys.add("evolve.gepa_max_metric_calls")

        elif "saturated round 2" in content or "saturated-round-2" in content:
            if "swarm.max_rounds" not in seen_keys:
                patches.append(ConfigPatch(
                    patch_id=str(uuid.uuid4()),
                    scope=PatchScope.GLOBAL, scope_value=None,
                    key_path="swarm.max_rounds",
                    old_value=4, new_value=3,
                    rationale="meta-loop: most runs saturate at round 2",
                    source_insights=[ins_id], risk_class="high",
                ))
                seen_keys.add("swarm.max_rounds")

    return patches


# ---------------------------------------------------------------------------
# Recursion bounds
# ---------------------------------------------------------------------------


class RecursionLevelExceeded(Exception):
    pass


def assert_level(current: int, allowed: int = 2) -> None:
    """Hard bound on recursion depth. L3+ requires explicit human approval."""
    if current > allowed:
        raise RecursionLevelExceeded(
            f"recursion level {current} exceeds allowed {allowed}; "
            f"L3+ requires --allow-l3 flag and human audit"
        )
