"""backends.models.resolve_model: the 6-step resolution order, env overrides,
effort passthrough, loud errors, and the strict config loader behind it."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from backends.models import (
    ModelResolutionError,
    ResolvedModel,
    format_models_table,
    list_models,
    pricing_for,
    resolve_model,
    role_env_var,
)
from config import ConfigError, load_config

ROOT = Path(__file__).parent.parent

BASE_TOML = """
[models]
default = "standard"
researcher = "standard"
synthesizer = "deep"
planner = "deep"
executor = "fast"
oddball = "ultra"            # tier no provider defines -> falls to default

[backend]
default = "alpha"
fallback = ["beta"]

[backend.providers.alpha]
adapter = "claude_cli"
reasoning_effort = "high"
[backend.providers.alpha.tiers]
fast = "alpha-fast"
standard = "alpha-std"
deep = "alpha-deep"
[backend.providers.alpha.effort]
deep = "max"
[backend.providers.alpha.pricing."alpha-deep"]
input_per_mtok = 15
output_per_mtok = 75.0

[backend.providers.beta]
adapter = "codex_cli"
[backend.providers.beta.tiers]
standard = "beta-std"

[backend.providers.legacy]
adapter = "codex_cli"
model = "legacy-only"

[backend.providers.empty]
adapter = "codex_cli"
"""


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "blitz.toml"
    path.write_text(textwrap.dedent(BASE_TOML), encoding="utf-8")
    return load_config(path)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("BLITZ_MODEL", raising=False)
    for var in list(__import__("os").environ):
        if var.startswith("BLITZ_MODEL_"):
            monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


def test_step3_role_tier_maps_through_provider_tiers(cfg):
    r = resolve_model("synthesizer", "alpha", cfg)
    assert r == ResolvedModel(model="alpha-deep", tier="deep", effort="max",
                              backend_id="alpha", source="tier")


def test_step4_default_tier_when_role_unlisted(cfg):
    r = resolve_model("some_new_role", "alpha", cfg)
    assert (r.model, r.tier, r.source) == ("alpha-std", "standard", "default_tier")


def test_step4_default_tier_when_role_tier_missing_on_provider(cfg):
    # oddball -> "ultra", which alpha does not define -> default "standard"
    r = resolve_model("oddball", "alpha", cfg)
    assert (r.model, r.tier, r.source) == ("alpha-std", "standard", "default_tier")
    # beta only defines "standard": planner(deep) also lands on default
    r = resolve_model("planner", "beta", cfg)
    assert (r.model, r.tier) == ("beta-std", "standard")


def test_step5_legacy_single_model_key(cfg):
    r = resolve_model("planner", "legacy", cfg)
    assert (r.model, r.source) == ("legacy-only", "legacy")
    assert r.tier == "deep"   # the tier the role asked for is still reported


def test_step6_error_names_role_backend_tier(cfg):
    with pytest.raises(ModelResolutionError) as exc:
        resolve_model("planner", "empty", cfg)
    msg = str(exc.value)
    assert "role='planner'" in msg
    assert "backend='empty'" in msg
    assert "tier='deep'" in msg
    assert "[backend.providers.empty.tiers]" in msg
    assert exc.value.tier == "deep"


def test_unknown_backend_errors(cfg):
    with pytest.raises(ModelResolutionError) as exc:
        resolve_model("planner", "nope", cfg)
    assert "not defined under [backend.providers]" in str(exc.value)


def test_backend_defaults_to_config_default(cfg):
    assert resolve_model("researcher", None, cfg).backend_id == "alpha"


def test_no_backend_at_all_errors(cfg):
    cfg.backend.default = ""
    with pytest.raises(ModelResolutionError) as exc:
        resolve_model("researcher", None, cfg)
    assert "no backend selected" in str(exc.value)


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def test_step1_explicit_override_wins_over_everything(cfg, monkeypatch):
    monkeypatch.setenv("BLITZ_MODEL", "env-global")
    monkeypatch.setenv("BLITZ_MODEL_PLANNER", "env-planner")
    r = resolve_model("planner", "alpha", cfg, override="cli-model")
    assert (r.model, r.source) == ("cli-model", "override")
    assert r.tier == "deep" and r.effort == "max"   # role tier still drives effort


def test_step1_env_blitz_model(cfg, monkeypatch):
    monkeypatch.setenv("BLITZ_MODEL", "env-global")
    monkeypatch.setenv("BLITZ_MODEL_PLANNER", "env-planner")
    assert resolve_model("planner", "alpha", cfg).model == "env-global"


def test_step2_env_role_override(cfg, monkeypatch):
    monkeypatch.setenv("BLITZ_MODEL_PLANNER", "env-planner")
    r = resolve_model("planner", "alpha", cfg)
    assert (r.model, r.source) == ("env-planner", "env_role")
    assert resolve_model("executor", "alpha", cfg).model == "alpha-fast"


def test_override_naming_a_tier_resolves_through_tiers(cfg, monkeypatch):
    r = resolve_model("executor", "alpha", cfg, override="deep")
    assert (r.model, r.tier, r.effort) == ("alpha-deep", "deep", "max")
    monkeypatch.setenv("BLITZ_MODEL_EXECUTOR", "standard")
    assert resolve_model("executor", "alpha", cfg).model == "alpha-std"


def test_env_dict_injection_is_isolated_from_os_environ(cfg, monkeypatch):
    monkeypatch.setenv("BLITZ_MODEL", "env-global")
    assert resolve_model("planner", "alpha", cfg, env={}).model == "alpha-deep"
    assert resolve_model("planner", "alpha", cfg, env={"BLITZ_MODEL_PLANNER": "x"}).model == "x"


def test_role_env_var_sanitises_role_names():
    assert role_env_var("fact_checker") == "BLITZ_MODEL_FACT_CHECKER"
    assert role_env_var("critic-factual") == "BLITZ_MODEL_CRITIC_FACTUAL"
    assert role_env_var("*") is None


# ---------------------------------------------------------------------------
# Effort passthrough
# ---------------------------------------------------------------------------


def test_effort_per_tier_then_provider_wide_then_none(cfg):
    assert resolve_model("synthesizer", "alpha", cfg).effort == "max"        # [effort].deep
    assert resolve_model("researcher", "alpha", cfg).effort == "high"       # reasoning_effort
    assert resolve_model("researcher", "beta", cfg).effort is None          # nothing set


def test_pricing_lookup(cfg):
    alpha = cfg.backend.get_provider("alpha")
    assert pricing_for(alpha, "alpha-deep") == {"input_per_mtok": 15.0, "output_per_mtok": 75.0}
    assert pricing_for(alpha, "alpha-std") is None


# ---------------------------------------------------------------------------
# --list-models table (dry-run friendly: never raises)
# ---------------------------------------------------------------------------


def test_list_models_rows_and_error_rows(cfg):
    rows = {r["role"]: r for r in list_models(cfg, "beta")}
    assert rows["researcher"]["model"] == "beta-std"
    assert rows["*"]["model"] == "beta-std"
    rows_empty = {r["role"]: r for r in list_models(cfg, "empty")}
    assert rows_empty["planner"]["source"] == "error"
    assert "tier='deep'" in rows_empty["planner"]["error"]
    table = format_models_table(cfg, "alpha")
    assert "synthesizer" in table and "alpha-deep" in table and "max" in table


# ---------------------------------------------------------------------------
# Shipped blitz.toml / mythos.toml contract
# ---------------------------------------------------------------------------


def test_shipped_blitz_toml_claude_deep_tier_is_fable():
    cfg = load_config(ROOT / "blitz.toml")
    planner = resolve_model("planner", "claude", cfg)
    assert planner.model == "fable"
    assert planner.tier == "deep"
    assert planner.effort == "max"
    assert resolve_model("executor", "claude", cfg).tier == "fast"
    assert resolve_model("verifier", "claude", cfg).model == "fable"
    assert resolve_model("synthesizer", "claude", cfg).model == "fable"
    assert resolve_model("researcher", "codex", cfg).tier == "standard"
    # every shipped provider resolves every configured role (or is an explicit error row)
    for backend_id in cfg.backend.providers:
        for row in list_models(cfg, backend_id):
            assert row["error"] is None, (backend_id, row)


def test_shipped_mythos_toml_uses_tiers_not_model_names():
    import tomllib
    raw = tomllib.loads((ROOT / "mythos.toml").read_text(encoding="utf-8"))
    assert raw["models"] == {
        "planner_model": "deep", "executor_model": "fast", "verifier_model": "deep",
    }
    assert raw["policies"]["token_ceiling"] == 2_000_000


# ---------------------------------------------------------------------------
# Strict config loader
# ---------------------------------------------------------------------------


def _load(tmp_path, text: str):
    path = tmp_path / "blitz.toml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path, load_config(path)


def test_missing_file_gives_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.backend.providers == {}
    assert cfg.models.default == "standard"


def test_invalid_toml_raises_with_path(tmp_path):
    path = tmp_path / "blitz.toml"
    path.write_text("[swarm\nmax_rounds = ", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert str(path) in str(exc.value) and "invalid TOML" in str(exc.value)


@pytest.mark.parametrize("text, needle", [
    ("[swarm]\nmax_round = 3\n", "unknown key [swarm].max_round"),
    ("[swarm]\nmax_rounds = 'three'\n", "[swarm].max_rounds expects int"),
    ("[swarm]\nmax_rounds = true\n", "[swarm].max_rounds expects int"),
    ("[nonsense]\nx = 1\n", "unknown section [nonsense]"),
    ("[backend]\nbogus = 1\n", "unknown key [backend].bogus"),
    ("[backend]\ndefault = 'ghost'\n", "default = 'ghost' is not defined"),
    ("[backend]\nfallback = ['ghost']\n[backend.providers.a]\nadapter='codex_cli'\n", "fallback names 'ghost'"),
    ("[backend.providers.a]\nadapter = 'codex_cli'\nephemeral = 'yes'\n", "[backend.providers.a].ephemeral expects bool"),
    ("[backend.providers.a]\nadapter = 'codex_cli'\n[backend.providers.a.tiers]\nfast = 3\n", ".tiers.fast must be a non-empty string"),
    ("[backend.providers.a]\nadapter = 'codex_cli'\ncommand = ['x', 1]\n", ".command[1] must be a string"),
    ("[backend.providers.a]\nadapter = 'codex_cli'\n[backend.providers.a.pricing.m]\ninput_per_mtok = 1\n", "missing output_per_mtok"),
    ("[models]\nplanner = 7\n", "[models].planner must be a non-empty tier name"),
])
def test_malformed_config_raises_naming_path_and_key(tmp_path, text, needle):
    path = tmp_path / "blitz.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert str(path) in str(exc.value)
    assert needle in str(exc.value), str(exc.value)


def test_provider_tables_are_open_ended_and_legacy_form_accepted(tmp_path):
    _, cfg = _load(tmp_path, """
        [backend]
        default = "x"
        fallback = "y"

        [backend.providers.x]
        adapter = "openai_http"
        base_url = "http://127.0.0.1:9/v1"
        api_key_env = "X_TOKEN"
        custom_knob = 42
        [backend.providers.x.tiers]
        standard = "m"

        [backend.y]
        adapter = "codex_cli"
        model = "legacy"
    """)
    x = cfg.backend.get_provider("x")
    assert x.adapter == "openai_http"
    assert x.api_key_env == "X_TOKEN"
    assert x.metadata == {"custom_knob": 42}
    assert cfg.backend.fallback == ["y"]
    assert cfg.backend.get_provider("y").model == "legacy"
    assert not hasattr(cfg.backend, "codex")
    with pytest.raises(KeyError):
        cfg.backend.get_provider("zzz")


def test_int_is_widened_to_float_for_float_fields(tmp_path):
    _, cfg = _load(tmp_path, "[memory]\ninsight_dedup_threshold = 1\n")
    assert cfg.memory.insight_dedup_threshold == 1.0
    assert isinstance(cfg.memory.insight_dedup_threshold, float)
