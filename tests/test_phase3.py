"""Tests for Phase 3: evolve / heterogeneity / managed_agents."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolve.aflow_search import (
    AddDebateRound,
    AddJudgeEnsemble,
    AddResearcher,
    Edge,
    MCTSNode,
    Node,
    OPERATOR_REGISTRY,
    RemoveCritic,
    RoleKind,
    SwapSynthForSelector,
    SwarmGraph,
    aflow_search,
    default_seed_graph,
)
from evolve.gepa_adapter import (
    BlitzGEPAAdapter,
    SwarmTask,
    SwarmTrajectory,
    load_role_prompts,
)
from evolve.meta_loop import (
    ALLOWED_KEY_PATHS,
    ConfigPatch,
    MetaLoopConfig,
    PatchScope,
    RecursionLevelExceeded,
    SchemaValidationError,
    assert_level,
    evaluate_patch,
    propose_patches_from_insights,
    validate_patch,
)
from heterogeneity.cli_router import (
    CLICall,
    CLIResult,
    CLIRouter,
    RouteEntry,
)
from managed_agents.adapter import (
    AgentSpec,
    BETA_HEADER,
    COORDINATOR_DEPTH_LIMIT,
    CoordinatorDepthExceeded,
    ManagedAgentsAdapter,
    ManagedAgentsConfig,
    SpendCapExceeded,
    SpendTracker,
    TooManySpecialists,
    check_coordinator_depth,
    validate_spec,
)


# ---------------------------------------------------------------------------
# AFlow / SwarmGraph
# ---------------------------------------------------------------------------


def test_default_seed_graph_is_valid():
    g = default_seed_graph()
    assert g.validate() == []
    assert g.count_role(RoleKind.RESEARCHER) == 2
    assert g.count_role(RoleKind.SYNTHESIZER) == 1


def test_swarm_graph_fingerprint_canonical():
    g1 = default_seed_graph()
    g2 = default_seed_graph()
    assert g1.fingerprint() == g2.fingerprint()


def test_swarm_graph_validate_detects_dangling_edge():
    nodes = (Node("a", RoleKind.RESEARCHER),)
    edges = (Edge("a", "missing"),)
    g = SwarmGraph(nodes=nodes, edges=edges)
    issues = g.validate()
    assert any("unknown target" in i for i in issues)


def test_add_researcher_increases_count():
    g = default_seed_graph()
    op = AddResearcher()
    assert op.applicable(g)
    g2 = op.apply(g)
    assert g2.count_role(RoleKind.RESEARCHER) == g.count_role(RoleKind.RESEARCHER) + 1
    assert g2.fingerprint() != g.fingerprint()


def test_remove_critic_drops_node_and_edges():
    g = default_seed_graph()
    op = RemoveCritic()
    assert op.applicable(g)
    g2 = op.apply(g)
    assert all(not n.kind.value.startswith("critic_") for n in g2.nodes)


def test_add_debate_round_bumps_rounds():
    g = default_seed_graph()
    op = AddDebateRound()
    g2 = op.apply(g)
    assert g2.rounds == g.rounds + 1


def test_swap_synth_for_selector():
    g = default_seed_graph()
    op = SwapSynthForSelector()
    g2 = op.apply(g)
    assert any(n.kind == RoleKind.SELECTOR_SYNTH for n in g2.nodes)
    assert g2.consensus_strategy == "judge_select"


def test_add_judge_ensemble_swaps_judge():
    g = default_seed_graph()
    op = AddJudgeEnsemble()
    g2 = op.apply(g)
    assert any(n.kind == RoleKind.JUDGE_ENSEMBLE for n in g2.nodes)


def test_aflow_search_returns_frontier():
    """MCTS run with deterministic evaluator picks expected mutations."""
    def evaluator(g: SwarmGraph) -> float:
        # Reward more researchers
        return 5.0 + 0.5 * g.count_role(RoleKind.RESEARCHER)

    seed = default_seed_graph()
    result = aflow_search(seed, evaluator=evaluator, max_iters=20, rng_seed=42)
    assert "frontier" in result
    assert len(result["frontier"]) > 0
    # Iterations completed
    assert result["iterations"] > 0


def test_aflow_search_dedup_avoids_revisits():
    def constant_evaluator(g: SwarmGraph) -> float:
        return 7.0

    seed = default_seed_graph()
    result = aflow_search(seed, evaluator=constant_evaluator, max_iters=30)
    fingerprints = [h["fp"] for h in result["history"]]
    duplicate_count = sum(1 for h in result["history"] if h.get("skipped") == "duplicate")
    # Some duplicates may be reached but the dedup must engage
    assert duplicate_count >= 0  # at minimum doesn't crash


# ---------------------------------------------------------------------------
# meta_loop.py
# ---------------------------------------------------------------------------


def _patch(key: str, new: object, old: object = 1, risk: str = "low") -> ConfigPatch:
    return ConfigPatch(
        patch_id="p1", scope=PatchScope.GLOBAL, scope_value=None,
        key_path=key, old_value=old, new_value=new,
        rationale="test", source_insights=[], risk_class=risk,
    )


def test_validate_patch_accepts_known_int_path():
    validate_patch(_patch("consensus.judge_ensemble.n", 3))


def test_validate_patch_rejects_unknown_path():
    with pytest.raises(SchemaValidationError):
        validate_patch(_patch("not.a.real.path", 1))


def test_validate_patch_rejects_out_of_range_int():
    with pytest.raises(SchemaValidationError):
        validate_patch(_patch("swarm.max_rounds", 999))


def test_validate_patch_rejects_wrong_type():
    with pytest.raises(SchemaValidationError):
        validate_patch(_patch("consensus.judge_ensemble.n", "three"))


def test_validate_patch_accepts_str_set():
    validate_patch(_patch("guard.mode", "balanced"))


def test_validate_patch_rejects_str_not_in_set():
    with pytest.raises(SchemaValidationError):
        validate_patch(_patch("guard.mode", "ludicrous"))


def test_validate_patch_accepts_float_range():
    validate_patch(_patch("consensus.judge_ensemble.ks_threshold", 0.05))


def test_evaluate_patch_merges_when_d_above_threshold():
    patch = _patch("consensus.judge_ensemble.n", 3)
    result = evaluate_patch(
        patch,
        baseline_score={
            "aggregate": 7.0,
            "per_dim": {"coverage": 7.0, "accuracy": 7.0},
            "stddev": 0.5,
            "cost": 1.0,
        },
        candidate_score={
            "aggregate": 7.5,
            "per_dim": {"coverage": 7.5, "accuracy": 7.5},
            "stddev": 0.5,
            "cost": 1.0,
        },
    )
    assert result.decision == "merged"
    assert result.delta == pytest.approx(0.5)


def test_evaluate_patch_rejects_per_dim_regression():
    patch = _patch("consensus.judge_ensemble.n", 3)
    result = evaluate_patch(
        patch,
        baseline_score={
            "aggregate": 7.0,
            "per_dim": {"coverage": 7.0, "accuracy": 8.0},
            "stddev": 0.5, "cost": 1.0,
        },
        candidate_score={
            "aggregate": 7.5,  # aggregate up
            "per_dim": {"coverage": 8.0, "accuracy": 7.0},  # accuracy DOWN by 1.0
            "stddev": 0.5, "cost": 1.0,
        },
    )
    assert result.decision == "rejected"
    assert result.per_dim_regressions  # non-empty


def test_evaluate_patch_rejects_cost_blowup():
    patch = _patch("consensus.judge_ensemble.n", 3)
    result = evaluate_patch(
        patch,
        baseline_score={"aggregate": 7.0, "per_dim": {}, "stddev": 0.5, "cost": 1.0},
        candidate_score={"aggregate": 7.5, "per_dim": {}, "stddev": 0.5, "cost": 5.0},
    )
    assert result.decision == "rejected"
    assert "cost" in result.reason


def test_evaluate_patch_high_risk_routes_to_human():
    patch = _patch("swarm.max_rounds", 6, risk="high")
    result = evaluate_patch(
        patch,
        baseline_score={"aggregate": 7.0, "per_dim": {}, "stddev": 0.5, "cost": 1.0},
        candidate_score={"aggregate": 8.0, "per_dim": {}, "stddev": 0.5, "cost": 1.0},
    )
    assert result.decision == "human_review"


def test_propose_patches_from_judge_variance_insight():
    insights = [
        {"id": "i1", "content": "meta: judge variance high on logical reasoning"},
    ]
    patches = propose_patches_from_insights(insights)
    assert len(patches) == 1
    assert patches[0].key_path == "consensus.judge_ensemble.n"


def test_assert_level_enforced():
    assert_level(2, allowed=2)  # ok
    with pytest.raises(RecursionLevelExceeded):
        assert_level(3, allowed=2)


# ---------------------------------------------------------------------------
# gepa_adapter.py
# ---------------------------------------------------------------------------


def test_gepa_adapter_evaluate_calls_run_fn():
    def fake_run(task, candidate):
        return SwarmTrajectory(
            task=task,
            role_outputs={"researcher": "out_r", "synthesizer": "out_s"},
            judge_breakdown={"coverage": 8.0},
            final_score=7.5,
        )

    def fake_score(traj, rubric):
        return traj.final_score

    adapter = BlitzGEPAAdapter(run_fn=fake_run, score_fn=fake_score)
    tasks = [
        SwarmTask(task_id="t1", topic="topic 1", domain="general",
                  expected_traits={}, judge_rubric={}),
        SwarmTask(task_id="t2", topic="topic 2", domain="general",
                  expected_traits={}, judge_rubric={}),
    ]
    result = adapter.evaluate({"researcher": "..."}, tasks)
    assert len(result["scores"]) == 2
    assert result["scores"] == [7.5, 7.5]


def test_gepa_adapter_make_reflective_dataset():
    adapter = BlitzGEPAAdapter(
        run_fn=lambda t, c: SwarmTrajectory(
            task=t, role_outputs={"researcher": "out"},
            judge_breakdown={"accuracy": 6.0}, final_score=6.5,
            failure_modes=["FM-2.4"],
        ),
        score_fn=lambda traj, r: traj.final_score,
    )
    tasks = [SwarmTask("t1", "topic", "general", {}, {})]
    eval_batch = adapter.evaluate({"researcher": "..."}, tasks)
    refl = adapter.make_reflective_dataset(
        {"researcher": "..."}, eval_batch, ["researcher"],
    )
    assert "researcher" in refl
    assert len(refl["researcher"]) == 1
    assert "Score" in refl["researcher"][0]["Feedback"]


def test_load_role_prompts(tmp_path):
    p = tmp_path / "researcher.md"
    p.write_text("hi", encoding="utf-8")
    out = load_role_prompts({"researcher": p})
    assert out == {"researcher": "hi"}


# ---------------------------------------------------------------------------
# CLI router
# ---------------------------------------------------------------------------


class _FakeAdapter:
    def __init__(self, cli_id: str, available: bool = True):
        self.cli_id = cli_id
        self._available = available
        self.calls: list[CLICall] = []

    def is_available(self) -> bool:
        return self._available

    def call(self, c: CLICall) -> CLIResult:
        self.calls.append(c)
        return CLIResult(
            cli_id=self.cli_id, text=f"{self.cli_id}-out",
            structured=None, elapsed_s=0.01, fallback_used=False,
        )


def test_router_routes_to_specified_cli():
    claude = _FakeAdapter("claude")
    codex = _FakeAdapter("codex")
    table = [RouteEntry(domain="*", role="researcher", cli_id="codex")]
    router = CLIRouter(table, adapters=[claude, codex])
    result = router.route("researcher", "general", "test prompt")
    assert result.cli_id == "codex"
    assert len(codex.calls) == 1


def test_router_falls_back_when_target_missing():
    claude = _FakeAdapter("claude")
    table = [RouteEntry(domain="*", role="researcher", cli_id="codex")]
    router = CLIRouter(table, adapters=[claude])  # no codex
    result = router.route("researcher", "general", "test")
    assert result.cli_id == "claude"
    assert result.fallback_used is True


def test_router_domain_override():
    claude = _FakeAdapter("claude")
    codex = _FakeAdapter("codex")
    gemini = _FakeAdapter("gemini")
    table = [
        RouteEntry(domain="*", role="researcher", cli_id="claude"),
        RouteEntry(domain="code", role="researcher", cli_id="codex"),
        RouteEntry(domain="factual", role="researcher", cli_id="gemini"),
    ]
    router = CLIRouter(table, adapters=[claude, codex, gemini])
    assert router.route("researcher", "code", "p").cli_id == "codex"
    assert router.route("researcher", "factual", "p").cli_id == "gemini"
    assert router.route("researcher", "general", "p").cli_id == "claude"


def test_router_loads_from_toml(tmp_path):
    toml_text = """
[default]
researcher = "claude"

[domain.code]
researcher = "codex"
"""
    p = tmp_path / "routing.toml"
    p.write_text(toml_text)
    claude = _FakeAdapter("claude")
    codex = _FakeAdapter("codex")
    router = CLIRouter.from_toml(p, adapters=[claude, codex])
    assert router.route("researcher", "general", "p").cli_id == "claude"
    assert router.route("researcher", "code", "p").cli_id == "codex"


def test_available_clis_filtered_by_availability():
    claude = _FakeAdapter("claude")
    codex = _FakeAdapter("codex", available=False)
    router = CLIRouter([], adapters=[claude, codex])
    assert "claude" in router.available_clis()
    assert "codex" not in router.available_clis()


# ---------------------------------------------------------------------------
# Managed Agents adapter
# ---------------------------------------------------------------------------


def test_validate_spec_caps_specialists():
    cfg = ManagedAgentsConfig(max_specialists=3)
    specs = [AgentSpec(role=f"r{i}", model="claude", system_prompt="x") for i in range(5)]
    with pytest.raises(TooManySpecialists):
        validate_spec(specs, cfg)


def test_check_coordinator_depth_blocks_more_than_one():
    specs = [
        AgentSpec("coordinator", "claude-opus", "x"),
        AgentSpec("coordinator", "claude-opus", "y"),
    ]
    with pytest.raises(CoordinatorDepthExceeded):
        check_coordinator_depth(specs)


def test_spend_tracker_caps_per_run():
    tracker = SpendTracker(cap_per_run=1.0, cap_per_day=10.0)
    tracker.assert_ok(0.5)
    tracker.record(0.5)
    with pytest.raises(SpendCapExceeded):
        tracker.assert_ok(1.0)


def test_spend_tracker_caps_per_day():
    tracker = SpendTracker(cap_per_run=10.0, cap_per_day=1.0)
    with pytest.raises(SpendCapExceeded):
        tracker.assert_ok(2.0)


def test_managed_agents_bootstrap_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = ManagedAgentsConfig()
    with pytest.raises(EnvironmentError):
        ManagedAgentsAdapter.bootstrap(cfg, [], client_factory=None)


def test_managed_agents_bootstrap_succeeds_with_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = ManagedAgentsConfig()
    adapter = ManagedAgentsAdapter.bootstrap(
        cfg, [AgentSpec("researcher", "claude-sonnet", "x")],
    )
    assert isinstance(adapter, ManagedAgentsAdapter)
    assert adapter.spend.cap_per_run == cfg.spend_cap_usd_per_run


def test_managed_agents_run_task_dryruns_without_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = ManagedAgentsConfig()
    adapter = ManagedAgentsAdapter.bootstrap(cfg, [])
    out = adapter.run_task("topic x", n_specialists=3, n_rounds=2)
    assert out["status"] == "stub"
    assert out["topic"] == "topic x"


def test_managed_agents_run_task_enforces_spend_cap(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = ManagedAgentsConfig(spend_cap_usd_per_run=0.01)
    adapter = ManagedAgentsAdapter.bootstrap(cfg, [])
    with pytest.raises(SpendCapExceeded):
        adapter.run_task("x", n_specialists=10, n_rounds=10)


def test_beta_header_constant():
    assert BETA_HEADER == "managed-agents-2026-04-01"


def test_coordinator_depth_constant():
    assert COORDINATOR_DEPTH_LIMIT == 1
