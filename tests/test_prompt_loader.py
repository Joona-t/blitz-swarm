"""Tests for prompts.loader.PromptLoader."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root on path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompts.loader import (
    PromptLoader,
    PromptLoaderError,
    PromptSet,
    PERSONA_ASSIGNMENT,
    assign_personas,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prompts_root() -> Path:
    return ROOT / "prompts"


@pytest.fixture
def general_loader(prompts_root) -> PromptLoader:
    return PromptLoader(root=prompts_root, domain="general")


@pytest.fixture
def crypto_loader(prompts_root) -> PromptLoader:
    return PromptLoader(root=prompts_root, domain="crypto")


# ---------------------------------------------------------------------------
# Basic load
# ---------------------------------------------------------------------------


def test_general_researcher_loads(general_loader):
    ps = general_loader.load("researcher")
    assert isinstance(ps, PromptSet)
    assert ps.role == "researcher"
    assert ps.persona is None
    assert ps.domain == "general"
    assert ps.system  # non-empty
    assert len(ps.sha256) == 64  # hex


def test_crypto_researcher_loads(crypto_loader):
    ps = crypto_loader.load("researcher")
    assert ps.domain == "crypto"
    assert "crypto" in ps.system.lower()


def test_general_critic_personas_load(general_loader):
    for persona in ("factual", "logical", "counterfactual", "steelman"):
        ps = general_loader.load("critic", persona=persona)
        assert ps.persona == persona
        assert persona in ps.system.lower() or "critic" in ps.system.lower()


def test_critic_no_persona_loads(general_loader):
    ps = general_loader.load("critic")
    assert ps.persona is None
    assert "critic" in ps.system.lower()


def test_quality_judge_general(general_loader):
    ps = general_loader.load("quality_judge")
    assert "coverage_score" in ps.system
    assert "accuracy_score" in ps.system


def test_synthesizer_general(general_loader):
    ps = general_loader.load("synthesizer")
    assert "synthesizer" in ps.system.lower() or "integrate" in ps.system.lower()


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def test_unknown_domain_falls_back_to_general(prompts_root):
    loader = PromptLoader(root=prompts_root, domain="nonexistent",
                          fallback_domain="general")
    ps = loader.load("researcher")
    assert ps.domain == "general"


def test_missing_role_raises(prompts_root):
    loader = PromptLoader(root=prompts_root, domain="general")
    with pytest.raises(PromptLoaderError):
        loader.load("nonexistent_role")


def test_missing_persona_raises(general_loader):
    with pytest.raises(PromptLoaderError):
        general_loader.load("critic", persona="nonexistent_persona")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_load_caches_result(general_loader):
    a = general_loader.load("researcher")
    b = general_loader.load("researcher")
    assert a is b  # same object from cache


def test_reload_clears_cache(general_loader):
    a = general_loader.load("researcher")
    general_loader.reload()
    b = general_loader.load("researcher")
    assert a is not b
    assert a.system == b.system  # same content


# ---------------------------------------------------------------------------
# list_personas
# ---------------------------------------------------------------------------


def test_list_personas_for_critic_general(general_loader):
    personas = general_loader.list_personas("critic")
    assert "factual" in personas
    assert "logical" in personas
    assert "counterfactual" in personas
    assert "steelman" in personas


def test_list_personas_for_researcher_returns_empty(general_loader):
    """No researcher_*.md files exist."""
    personas = general_loader.list_personas("researcher")
    assert personas == []


# ---------------------------------------------------------------------------
# Persona assignment policy
# ---------------------------------------------------------------------------


def test_assign_personas_count_1():
    assert assign_personas(1) == ["factual"]


def test_assign_personas_count_2():
    assert assign_personas(2) == ["factual", "counterfactual"]


def test_assign_personas_count_3():
    assert assign_personas(3) == ["factual", "logical", "counterfactual"]


def test_assign_personas_count_4_recycles():
    """4 slots: cap at 3 unique, recycle for the 4th."""
    out = assign_personas(4)
    assert len(out) == 4
    # First 3 should be the standard 3-persona set
    assert out[:3] == ["factual", "logical", "counterfactual"]


def test_assign_personas_round2_with_dissent_swaps_in_steelman():
    out = assign_personas(3, round_n=2, has_unresolved_dissent=True)
    assert out[-1] == "steelman"


def test_assign_personas_round2_no_dissent_keeps_default():
    out = assign_personas(3, round_n=2, has_unresolved_dissent=False)
    assert "steelman" not in out


def test_assign_personas_round1_never_steelman():
    out = assign_personas(3, round_n=1, has_unresolved_dissent=True)
    assert "steelman" not in out


# ---------------------------------------------------------------------------
# Integration with agents.py
# ---------------------------------------------------------------------------


def test_plan_agents_uses_general_by_default():
    from agents import plan_agents
    agents_list = plan_agents("test topic", use_llm=False, domain="general")
    by_role = {a.role: a for a in agents_list}

    # Each role should have a prompt_path pointing to prompts/general/
    for role in ("researcher", "critic", "quality_judge", "synthesizer"):
        assert role in by_role, f"missing role: {role}"
        agent = by_role[role]
        assert agent.system_prompt
        if agent.prompt_path:
            assert "prompts/general" in agent.prompt_path


def test_plan_agents_uses_crypto_when_requested():
    from agents import plan_agents
    agents_list = plan_agents("test topic", use_llm=False, domain="crypto")
    by_role = {a.role: a for a in agents_list}
    for role in ("researcher", "critic"):
        agent = by_role.get(role)
        if agent and agent.prompt_path:
            assert "prompts/crypto" in agent.prompt_path


def test_plan_agents_persona_critics_off_by_default():
    from agents import plan_agents
    agents_list = plan_agents("test topic", use_llm=False, domain="general",
                              persona_critics=False)
    critics = [a for a in agents_list if a.role == "critic"]
    for c in critics:
        assert c.persona is None


def test_plan_agents_persona_critics_when_enabled_with_multiple_critics():
    """If we had a swarm plan with critic_count >= 2, personas would activate.
    The default heuristic plan has critic_count=1, so personas just give factual."""
    from agents import plan_agents
    agents_list = plan_agents("test topic", use_llm=False, domain="general",
                              persona_critics=True)
    critics = [a for a in agents_list if a.role == "critic"]
    # With critic_count=1 (heuristic default), persona is "factual"
    if len(critics) == 1:
        assert critics[0].persona == "factual"
        assert "factual" in critics[0].id
