"""config.py strictness (core-loop-07): every blitz.toml key is read by code,
dead keys are gone, malformed TOML is a clear ConfigError, and the CLI
surfaces it as one line + exit 2 (no traceback)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from config import (
    CONFIG_PATH,
    ConfigError,
    JudgeEnsembleConfig,
    StorageConfig,
    _SIMPLE_SECTIONS,
    load_config,
)

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def test_shipped_blitz_toml_loads_and_judge_halt_defaults_false():
    cfg = load_config(CONFIG_PATH)
    assert cfg.judge_ensemble.halt is False
    assert JudgeEnsembleConfig().halt is False
    assert cfg.swarm.max_agents >= 4


def test_lancedb_path_is_gone_from_config_and_toml():
    assert "lancedb_path" not in {f.name for f in fields(StorageConfig)}
    assert "lancedb_path" not in CONFIG_PATH.read_text(encoding="utf-8")
    with pytest.raises(ConfigError, match=r"unknown key \[storage\]\.lancedb_path"):
        load_config_from_text('[storage]\nlancedb_path = "./x"\n')


def load_config_from_text(text: str, tmp: Path | None = None):
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="blitz-cfg-")) if tmp is None else tmp
    p = d / "blitz.toml"
    p.write_text(text, encoding="utf-8")
    return load_config(p)


def test_every_simple_section_key_in_blitz_toml_is_read_by_production_code():
    """A key in blitz.toml must be consumed somewhere outside config.py and
    tests/; otherwise it is dead config and must be deleted (core-loop-07)."""
    import tomllib
    raw = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sources = []
    for path in REPO.rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(("tests/", "docs/")) or rel == "config.py" or "__pycache__" in rel:
            continue
        sources.append(path.read_text(encoding="utf-8", errors="ignore"))
    corpus = "\n".join(sources)
    unread = []
    for section in _SIMPLE_SECTIONS:
        for key in raw.get(section, {}):
            if not re.search(rf"\b{re.escape(key)}\b", corpus):
                unread.append(f"[{section}].{key}")
    assert unread == [], f"dead blitz.toml keys (no reader outside config.py): {unread}"


def test_malformed_toml_and_bad_types_raise_config_error_naming_path():
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config_from_text("[swarm\nmax_rounds = 3\n")
    with pytest.raises(ConfigError, match=r"\[swarm\]\.max_rounds expects int"):
        load_config_from_text('[swarm]\nmax_rounds = "three"\n')
    with pytest.raises(ConfigError, match=r"unknown section \[nope\]"):
        load_config_from_text("[nope]\nx = 1\n")


def test_blitz_config_env_var_selects_the_file(monkeypatch, tmp_path):
    p = tmp_path / "alt.toml"
    p.write_text("[swarm]\nmax_rounds = 9\n", encoding="utf-8")
    monkeypatch.setenv("BLITZ_CONFIG", str(p))
    assert load_config().swarm.max_rounds == 9
    monkeypatch.delenv("BLITZ_CONFIG")
    assert load_config().swarm.max_rounds == load_config(CONFIG_PATH).swarm.max_rounds


@pytest.mark.parametrize("text, needle", [
    ("[swarm\nmax_rounds = 3\n", "invalid TOML"),
    ('[storage]\nlancedb_path = "./x"\n', "unknown key [storage].lancedb_path"),
])
def test_cli_surfaces_config_error_as_one_line_exit_2(tmp_path, text, needle):
    """`python orchestrator.py --list-models` (no model call) with a broken
    BLITZ_CONFIG: exit 2, the error on stderr, no traceback."""
    bad = tmp_path / "blitz.toml"
    bad.write_text(text, encoding="utf-8")
    env = dict(os.environ, BLITZ_CONFIG=str(bad))
    proc = subprocess.run(
        [PY, str(REPO / "orchestrator.py"), "--list-models"],
        capture_output=True, text=True, timeout=60, cwd=str(tmp_path), env=env,
    )
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "blitz-swarm: config error:" in proc.stderr and needle in proc.stderr
    assert "Traceback" not in proc.stderr
