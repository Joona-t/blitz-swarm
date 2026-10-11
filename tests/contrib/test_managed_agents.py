"""Tests for contrib/managed_agents — opt-in, PAID API scaffold. No network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contrib.managed_agents.adapter import (
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


def test_validate_spec_caps_specialists():
    cfg = ManagedAgentsConfig(max_specialists=3)
    specs = [AgentSpec(role=f"r{i}", model="model-a", system_prompt="x") for i in range(5)]
    with pytest.raises(TooManySpecialists):
        validate_spec(specs, cfg)


def test_check_coordinator_depth_blocks_more_than_one():
    specs = [
        AgentSpec("coordinator", "model-a", "x"),
        AgentSpec("coordinator", "model-a", "y"),
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
        cfg, [AgentSpec("researcher", "model-b", "x")],
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
    assert adapter.spend.spent_this_run == 0.0


def test_managed_agents_run_task_enforces_spend_cap(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = ManagedAgentsConfig(spend_cap_usd_per_run=0.01)
    adapter = ManagedAgentsAdapter.bootstrap(cfg, [])
    with pytest.raises(SpendCapExceeded):
        adapter.run_task("x", n_specialists=10, n_rounds=10)


def test_managed_agents_run_task_with_client_refuses_not_fakes_completion(monkeypatch):
    """backends-02: no fake 'completed' path — with a client attached the call
    raises NotImplementedError and records no spend."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = ManagedAgentsConfig()
    adapter = ManagedAgentsAdapter.bootstrap(cfg, [], client_factory=lambda c: object())
    assert adapter.client is not None
    with pytest.raises(NotImplementedError, match="not implemented"):
        adapter.run_task("topic", n_specialists=2, n_rounds=1)
    assert adapter.spend.spent_this_run == 0.0


def test_nothing_on_the_default_path_imports_contrib():
    """Rule #10: contrib/ is opt-in; no production module may import it."""
    import re

    offenders = []
    for py in ROOT.rglob("*.py"):
        rel = py.relative_to(ROOT).as_posix()
        if rel.startswith(("contrib/", "tests/")) or "__pycache__" in rel or "/." in f"/{rel}":
            continue
        if re.search(r"^\s*(from|import)\s+contrib(\.|\s)", py.read_text(encoding="utf-8", errors="ignore"), re.M):
            offenders.append(rel)
    assert not offenders, offenders


def test_readme_states_paid_api_opt_in_and_rule_10():
    text = (ROOT / "contrib" / "managed_agents" / "README.md").read_text(encoding="utf-8")
    for needle in ("PAID API", "opt-in", "rule #10", "not implemented"):
        assert needle in text, needle


def test_beta_header_constant():
    assert BETA_HEADER == "managed-agents-2026-04-01"


def test_coordinator_depth_constant():
    assert COORDINATOR_DEPTH_LIMIT == 1
