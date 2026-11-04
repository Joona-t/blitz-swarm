"""WP8: packaging and config lookup.

The wheel build itself runs in CI (tests.yml `wheel` job) and locally with
BLITZ_PACKAGING_TEST=1; the cheap checks here guard the declarations and
the config lookup order so a regression is caught without a build.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import config as config_mod

REPO = Path(__file__).resolve().parents[1]


def test_pyproject_ships_prompts_configs_and_entry_point():
    meta = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    st = meta["tool"]["setuptools"]
    assert "prompts" in st["packages"]
    assert st["package-data"]["prompts"] == ["*/*.md"]
    assert set(st["data-files"]["share/blitz-swarm"]) == {"blitz.toml", "mythos.toml"}
    assert meta["project"]["scripts"]["blitz"] == "orchestrator:main"
    assert meta["project"]["version"] == "0.4.0"


def test_find_config_file_order(tmp_path, monkeypatch):
    monkeypatch.delenv("BLITZ_CONFIG", raising=False)
    # 2. cwd
    monkeypatch.chdir(tmp_path)
    (tmp_path / "blitz.toml").write_text("[swarm]\nmax_rounds = 1\n", encoding="utf-8")
    assert config_mod.find_config_file() == tmp_path / "blitz.toml"
    # 1. env beats cwd
    other = tmp_path / "other.toml"
    other.write_text("[swarm]\nmax_rounds = 2\n", encoding="utf-8")
    monkeypatch.setenv("BLITZ_CONFIG", str(other))
    assert config_mod.find_config_file() == other
    assert config_mod.load_config().swarm.max_rounds == 2
    # 4. nothing anywhere -> the share path is returned (and load_config -> defaults)
    monkeypatch.delenv("BLITZ_CONFIG")
    (tmp_path / "blitz.toml").unlink()
    monkeypatch.setattr(config_mod, "__file__", str(tmp_path / "nowhere" / "config.py"))
    found = config_mod.find_config_file()
    assert found.parts[-3:] == ("share", "blitz-swarm", "blitz.toml")


def test_no_home_paths_in_tracked_files():
    out = subprocess.run(
        ["git", "grep", "-nE", r"/Users/[^/ ]+/|/home/[^/ ]+/|scratch-workspaces", "--",
         ":!tests/test_packaging.py", ":!.github/workflows/tests.yml"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode == 1, f"home paths in tracked files:\n{out.stdout}"


@pytest.mark.skipif(not os.environ.get("BLITZ_PACKAGING_TEST"), reason="set BLITZ_PACKAGING_TEST=1 to build and install the wheel (slow)")
def test_wheel_installs_and_runs_model_free(tmp_path):
    uv = subprocess.run(["uv", "--version"], capture_output=True)
    assert uv.returncode == 0, "uv required for the local wheel test"
    dist = tmp_path / "dist"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=REPO, check=True, capture_output=True)
    wheel = next(dist.glob("*.whl"))
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "-q", "-p", sys.executable, str(venv)], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "-p", str(venv / "bin" / "python"), str(wheel)], check=True)
    env = {"PATH": "/nonexistent", "HOME": str(tmp_path)}
    r = subprocess.run([str(venv / "bin" / "blitz"), "--list-models"], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-800:]
    r = subprocess.run([str(venv / "bin" / "blitz"), "--dry-run", "--no-llm-plan", "x"], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-800:]
    assert "DRY RUN" in r.stdout
