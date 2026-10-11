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


# ---------------------------------------------------------------------------
# WP5 additions: defaults, coercion, [models] / [backend] parsing
# ---------------------------------------------------------------------------


def test_missing_file_and_missing_blitz_config_env_path_yield_code_defaults(monkeypatch, tmp_path):
    from config import BlitzConfig
    assert load_config(tmp_path / "absent.toml") == BlitzConfig()
    monkeypatch.setenv("BLITZ_CONFIG", str(tmp_path / "also-absent.toml"))
    cfg = load_config()
    assert cfg == BlitzConfig()
    assert cfg.backend.default == "" and cfg.backend.providers == {}
    assert cfg.models.tier_for("anything") == "standard"


def test_int_is_coerced_to_float_fields_but_bool_is_never_an_int():
    cfg = load_config_from_text("[memory]\ninsight_dedup_threshold = 1\n")
    assert cfg.memory.insight_dedup_threshold == 1.0 and isinstance(cfg.memory.insight_dedup_threshold, float)
    with pytest.raises(ConfigError, match=r"\[swarm\]\.max_rounds expects int, got bool"):
        load_config_from_text("[swarm]\nmax_rounds = true\n")
    with pytest.raises(ConfigError, match=r"\[guard\]\.enabled expects bool, got int"):
        load_config_from_text("[guard]\nenabled = 1\n")


def test_section_must_be_a_table_and_guard_mode_is_validated():
    with pytest.raises(ConfigError, match=r"\[swarm\] must be a table"):
        load_config_from_text("swarm = 3\n")
    with pytest.raises(ConfigError, match=r"\[guard\]\.mode = 'turbo'; expected one of off, speed"):
        load_config_from_text('[guard]\nmode = "turbo"\n')
    assert load_config_from_text('[guard]\nmode = "max"\n').guard.mode == "max"


def test_models_section_maps_roles_to_tiers_with_default():
    cfg = load_config_from_text('[models]\ndefault = "fast"\nplanner = "deep"\n')
    assert cfg.models.default == "fast"
    assert cfg.models.roles == {"planner": "deep"}
    assert cfg.models.tier_for("planner") == "deep"
    assert cfg.models.tier_for("executor") == "fast"
    with pytest.raises(ConfigError, match=r"\[models\]\.planner must be a non-empty tier name"):
        load_config_from_text('[models]\nplanner = ""\n')
    with pytest.raises(ConfigError, match=r"\[models\] must be a table"):
        load_config_from_text('models = "x"\n')


def test_models_tier_for_falls_back_to_none_when_default_is_blank():
    from config import ModelsConfig
    m = ModelsConfig(default="", roles={"planner": "deep"})
    assert m.tier_for("planner") == "deep"
    assert m.tier_for("executor") is None


def test_backend_providers_parse_known_keys_and_pass_unknown_keys_through_metadata():
    cfg = load_config_from_text(
        '[backend]\ndefault = "p1"\nfallback = "p2"\n'
        '[backend.providers.p1]\nadapter = "generic_cli"\ncommand = ["mycli", "--model", "{model}"]\n'
        'custom_flag = true\n[backend.providers.p1.tiers]\nfast = "f"\ndeep = "d"\n'
        '[backend.providers.p1.effort]\ndeep = "max"\n'
        '[backend.p2]\nadapter = "ollama_http"\nbase_url = "http://localhost:11434"\n'
    )
    assert cfg.backend.default == "p1"
    assert cfg.backend.fallback == ["p2"]                       # bare string coerced to a list
    assert set(cfg.backend.providers) == {"p1", "p2"}            # legacy [backend.<id>] form accepted
    p1 = cfg.backend.get_provider("p1")
    assert p1.adapter == "generic_cli" and p1.command == ["mycli", "--model", "{model}"]
    assert p1.tiers == {"fast": "f", "deep": "d"} and p1.effort == {"deep": "max"}
    assert p1.metadata == {"custom_flag": True}
    assert cfg.backend.get_provider("p2").base_url == "http://localhost:11434"
    with pytest.raises(KeyError):
        cfg.backend.get_provider("ghost")


@pytest.mark.parametrize("text, needle", [
    ('[backend]\ndefault = "nope"\n', "[backend].default = 'nope' is not defined under [backend.providers]"),
    ('[backend]\nfallback = ["nope"]\n', "[backend].fallback names 'nope', which is not defined"),
    ('[backend]\nfallback = 3\n', "[backend].fallback must be a list of provider ids"),
    ('[backend]\ndefault = 3\n', "[backend].default expects str"),
    ('[backend]\nproviders = 3\n', "[backend.providers] must be a table"),
    ('[backend]\nmystery = "x"\n', "unknown key [backend].mystery"),
    ('[backend.providers.p]\ntiers = { fast = "" }\n', "[backend.providers.p].tiers.fast must be a non-empty string"),
    ('[backend.providers.p]\ncommand = ["ok", 3]\n', "[backend.providers.p].command[1] must be a string"),
    ('[backend.providers.p]\nsandbox = 3\n', "[backend.providers.p].sandbox expects str"),
    ('[backend.providers.p.pricing]\nm = 3\n', "[backend.providers.p].pricing.m must be a table"),
    ('[backend.providers.p.pricing.m]\ninput_per_mtok = 1.0\n', "[backend.providers.p].pricing.m is missing output_per_mtok"),
    ('[backend.providers.p.pricing.m]\ninput_per_mtok = "1"\noutput_per_mtok = 2.0\n',
     "[backend.providers.p].pricing.m.input_per_mtok expects float"),
])
def test_backend_section_validation_errors_name_the_key(text, needle):
    with pytest.raises(ConfigError) as exc:
        load_config_from_text(text)
    assert needle in str(exc.value)
    assert str(exc.value).startswith(str(Path(exc.value.args[0].split(":")[0])))  # message starts with the path


def test_pricing_table_is_kept_verbatim_for_metrics():
    cfg = load_config_from_text(
        '[backend.providers.p.pricing."some-model"]\ninput_per_mtok = 1\noutput_per_mtok = 2.5\n'
    )
    assert cfg.backend.get_provider("p").pricing == {"some-model": {"input_per_mtok": 1, "output_per_mtok": 2.5}}
