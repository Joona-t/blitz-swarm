"""Mythos Swarm smoke tests — imports + config + dry-run shape.

These do NOT invoke `claude -p`. They verify the package wiring is sound
so a real run can be attempted with confidence.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mythos_package_imports():
    """Public surface is importable."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from mythos import (
            MythosConfig,
            CostBudget,
            MythosResult,
            load_mythos_config,
            run_mythos,
        )
    finally:
        sys.path.pop(0)
    assert run_mythos is not None
    # v0.3.0: no alias table. Empty = use blitz.toml [models].<role> tier.
    assert MythosConfig().planner_model == ""
    assert not hasattr(__import__("mythos.policies", fromlist=["x"]), "MODEL_ALIASES")


def test_role_tier_resolution_order(monkeypatch):
    """mythos.toml tier beats blitz.toml [models]; env BLITZ_MODEL_<ROLE> and
    BLITZ_MODEL beat the mythos.toml tier; provider `model` never shadows the
    role tier (the v0.2 `provider.model or model` bug)."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from config import load_config
        from mythos._invoke import resolve_role_model
    finally:
        sys.path.pop(0)
    cfg = load_config()
    for var in ("BLITZ_MODEL", "BLITZ_MODEL_PLANNER", "BLITZ_MODEL_EXECUTOR", "BLITZ_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    claude = cfg.backend.get_provider("claude")
    claude.model = "legacy-provider-wide"          # must NOT win over the role tier

    deep = resolve_role_model("planner", "deep", cfg, backend_id="claude")
    assert (deep.model, deep.tier, deep.effort) == (claude.tiers["deep"], "deep", claude.effort["deep"])
    fast = resolve_role_model("planner", "fast", cfg, backend_id="claude")
    assert fast.model == claude.tiers["fast"]
    # no overlay tier -> blitz.toml [models].planner
    base = resolve_role_model("planner", "", cfg, backend_id="claude")
    assert base.tier == cfg.models.roles["planner"]

    monkeypatch.setenv("BLITZ_MODEL_PLANNER", "fast")
    assert resolve_role_model("planner", "deep", cfg, backend_id="claude").model == claude.tiers["fast"]
    monkeypatch.setenv("BLITZ_MODEL", "cli-override")
    assert resolve_role_model("planner", "deep", cfg, backend_id="claude").model == "cli-override"


def test_cost_budget_token_ceiling():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from mythos.policies import CostBudget
    finally:
        sys.path.pop(0)
    b = CostBudget(ceiling_usd=5.0, token_ceiling=100)
    b.charge(None, tokens=60)        # unknown cost is not 0 spend
    assert b.spent_usd == 0.0 and not b.exceeded()
    b.charge(None, tokens=40)
    assert b.tokens_exceeded() and b.exceeded() and not b.cost_exceeded()


def test_cost_budget():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from mythos.policies import CostBudget
    finally:
        sys.path.pop(0)

    b = CostBudget(ceiling_usd=1.0)
    assert not b.exceeded()
    b.charge(0.4)
    assert b.remaining() == pytest.approx(0.6)
    b.charge(0.7)
    assert b.exceeded()


def test_load_mythos_config_with_toml():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from mythos.policies import load_mythos_config
    finally:
        sys.path.pop(0)
    cfg = load_mythos_config(REPO_ROOT / "mythos.toml")
    # v0.3.0: mythos.toml [models] names TIERS, resolved per backend through
    # backends.models.resolve_model (claude deep tier = Fable 5.1).
    assert cfg.planner_model == "deep"
    assert cfg.executor_model == "fast"
    assert cfg.verifier_model == "deep"
    assert cfg.cost_ceiling_usd == 5.0
    assert cfg.token_ceiling == 2_000_000
    assert cfg.max_replans == 3


def test_schemas_are_valid_json():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from mythos.schemas import (
            EXECUTOR_SCHEMA_JSON,
            PLANNER_SCHEMA_JSON,
            VERIFIER_SCHEMA_JSON,
        )
    finally:
        sys.path.pop(0)
    for s in (PLANNER_SCHEMA_JSON, EXECUTOR_SCHEMA_JSON, VERIFIER_SCHEMA_JSON):
        parsed = json.loads(s)
        assert parsed["type"] == "object"
        assert "properties" in parsed
        assert "required" in parsed


def test_prompts_exist_and_nonempty():
    prompts_dir = REPO_ROOT / "mythos" / "prompts"
    for name in ("planner.md", "executor.md", "verifier.md"):
        path = prompts_dir / name
        assert path.exists(), f"missing {path}"
        assert len(path.read_text(encoding="utf-8")) > 200


def test_orchestrator_help_includes_mythos_flag():
    """The orchestrator CLI surface exposes --mode mythos."""
    out = subprocess.run(
        [sys.executable, "orchestrator.py", "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert "--mode" in out.stdout
    assert "mythos" in out.stdout
    assert "--cost-ceiling" in out.stdout
    assert "--max-replans" in out.stdout


def test_mythos_dry_run_no_cli_call():
    """--mode mythos --dry-run prints config without invoking claude."""
    out = subprocess.run(
        [
            sys.executable, "orchestrator.py",
            "--mode", "mythos", "--dry-run",
            "test task — please do not actually run",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert "MYTHOS DRY RUN" in out.stdout
    assert "planner_model" in out.stdout
    assert "executor_model" in out.stdout
    assert "verifier_model" in out.stdout
    assert "token_ceiling" in out.stdout
    assert "resolved" in out.stdout and "planner=" in out.stdout
    assert "No CLI calls will be made" in out.stdout


def test_mythos_swarm_shim_exists_and_executable():
    shim = REPO_ROOT / "mythos_swarm.py"
    assert shim.exists()
    # Just verify the shim can be parsed as Python without errors.
    out = subprocess.run(
        [sys.executable, "-c", "import ast; ast.parse(open('mythos_swarm.py').read())"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=10,
    )
    assert out.returncode == 0, out.stderr
