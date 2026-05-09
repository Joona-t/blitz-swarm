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
            resolve_model,
            run_mythos,
        )
    finally:
        sys.path.pop(0)
    assert run_mythos is not None
    assert MythosConfig().planner_model == "mythos"


def test_resolve_model_aliases():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from mythos.policies import resolve_model
    finally:
        sys.path.pop(0)
    assert resolve_model("mythos") == ("opus", "max")
    assert resolve_model("sonnet") == ("sonnet", None)
    assert resolve_model("opus") == ("opus", None)
    # Unknown alias passes through
    assert resolve_model("claude-opus-4-7") == ("claude-opus-4-7", None)


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
    # Defaults from mythos.toml
    assert cfg.planner_model == "mythos"
    assert cfg.executor_model == "sonnet"
    assert cfg.verifier_model == "mythos"
    assert cfg.cost_ceiling_usd == 5.0
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
