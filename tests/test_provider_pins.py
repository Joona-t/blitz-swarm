"""WP7: per-role provider pins and --list-models."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import backends
from backends import AgentCall, RoleRouter, make_role_router
from backends.models import format_models_table, list_models, provider_for_role
from config import ConfigError, load_config

REPO = Path(__file__).resolve().parents[1]

TOML = '''
[models]
default = "standard"
judge = "deep"
judge_provider = "b"
critic_provider = "b"

[backend]
default = "a"

[backend.providers.a]
adapter = "generic_cli"
command = ["tool-a", "{model}", "{prompt_file}"]
allow_degrade = ["no_tools"]
[backend.providers.a.tiers]
fast = "a-fast"
standard = "a-std"
deep = "a-deep"

[backend.providers.b]
adapter = "generic_cli"
command = ["tool-b", "{model}", "{prompt_file}"]
allow_degrade = ["no_tools"]
[backend.providers.b.tiers]
fast = "b-fast"
standard = "b-std"
deep = "b-deep"
'''


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "blitz.toml"
    p.write_text(TOML, encoding="utf-8")
    return load_config(p)


def test_pins_are_parsed_from_models_table(cfg):
    assert cfg.models.providers == {"judge": "b", "critic": "b"}
    assert cfg.models.roles == {"judge": "deep"}          # pins are not tiers
    assert cfg.models.provider_for("judge") == "b" and cfg.models.provider_for("researcher") is None


def test_provider_for_role_defaults_and_validates(cfg):
    assert provider_for_role("researcher", cfg) == "a"
    assert provider_for_role("judge", cfg) == "b"
    cfg.models.providers["judge"] = "ghost"
    with pytest.raises(ValueError, match="judge_provider = 'ghost'"):
        provider_for_role("judge", cfg)


def test_list_models_shows_provider_per_role(cfg):
    rows = {r["role"]: r for r in list_models(cfg)}
    assert rows["judge"]["backend"] == "b" and rows["judge"]["model"] == "b-deep"
    assert rows["critic"]["backend"] == "b" and rows["critic"]["model"] == "b-std"
    assert rows["*"]["backend"] == "a" and rows["*"]["model"] == "a-std"
    table = format_models_table(cfg)
    assert "provider" in table.splitlines()[1] and " b " in table


def test_router_routes_pinned_roles_and_re_resolves_the_model(cfg, monkeypatch):
    spawned: list[list[str]] = []
    from backends.adapters import generic_cli

    monkeypatch.setattr(generic_cli.shutil, "which", lambda n: f"/bin/{n}")

    def fake_run(cmd, **kw):
        spawned.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="The answer is forty-two.", stderr="")

    monkeypatch.setattr(generic_cli.subprocess, "run", fake_run)

    router = make_role_router("a", cfg)
    assert isinstance(router, RoleRouter)
    # model resolved for the default provider, as the orchestrator does
    r = router.call(AgentCall(role="judge", model_role="judge", prompt="p", model="a-deep", tools=[]))
    assert spawned[-1][:2] == ["tool-b", "b-deep"]
    assert r.metadata["routed_provider"] == "b" and r.metadata["routed_from"] == "a"
    router.call(AgentCall(role="researcher", prompt="p", model="a-std", tools=[]))
    assert spawned[-1][:2] == ["tool-a", "a-std"]
    assert "routed_provider" not in router.call(AgentCall(role="researcher", prompt="p", model="a-std", tools=[])).metadata


def test_router_is_plain_backend_without_pins(cfg):
    cfg.models.providers.clear()
    b = make_role_router("a", cfg)
    assert not isinstance(b, RoleRouter)


def test_router_refuses_undefined_pin_at_build(cfg):
    cfg.models.providers["judge"] = "ghost"
    with pytest.raises(ValueError, match="ghost"):
        make_role_router("a", cfg)


def test_list_models_cli_is_model_free(tmp_path):
    """`python orchestrator.py --list-models` must not call any model and must
    print the capability table."""
    proc = subprocess.run(
        [sys.executable, "orchestrator.py", "--list-models"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
        env={"PATH": "/nonexistent", "HOME": str(tmp_path)},   # no CLI on PATH -> any model call would fail loudly
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "Provider capabilities" in proc.stdout
    assert "provider" in proc.stdout.splitlines()[1]
