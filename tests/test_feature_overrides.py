"""Tests for the BLITZ_FEATURE_OVERRIDES env override in _feature_enabled (C2).

Precedence contract:
    1. BLITZ_FEATURE_OVERRIDES env var (JSON object: feature name -> bool)
    2. explicit override argument (CLI flags)
    3. quality profile (max -> on, cheap -> off)
    4. configured value (blitz.toml)

The env check is ADDITIVE: with the env var unset (or malformed, or the
feature name absent from the mapping), behavior is byte-identical to the
pre-C2 logic.
"""

from __future__ import annotations

import json

import pytest

from orchestrator import _feature_enabled

ENV = "BLITZ_FEATURE_OVERRIDES"


# ---------------------------------------------------------------------------
# Env set — highest precedence, beats profile/config/override
# ---------------------------------------------------------------------------


def test_env_false_beats_everything(monkeypatch):
    monkeypatch.setenv(ENV, json.dumps({"judge_ensemble": False}))
    # profile=max, configured=True, explicit override=True would all say ON
    assert _feature_enabled(
        profile="max",
        configured=True,
        override=True,
        name="judge_ensemble",
    ) is False


def test_env_true_beats_everything(monkeypatch):
    monkeypatch.setenv(ENV, json.dumps({"cascade_guard": True}))
    # profile=cheap, configured=False, explicit override=False would all say OFF
    assert _feature_enabled(
        profile="cheap",
        configured=False,
        override=False,
        name="cascade_guard",
    ) is True


def test_env_overrides_each_known_mechanism_name(monkeypatch):
    monkeypatch.setenv(
        ENV,
        json.dumps({"cascade_guard": False, "judge_ensemble": False, "selector": True}),
    )
    for name, expected in (
        ("cascade_guard", False),
        ("judge_ensemble", False),
        ("selector", True),
    ):
        assert _feature_enabled(
            profile="max",
            configured=True,
            override=None,
            name=name,
        ) is expected


def test_env_supports_unknown_future_feature_names(monkeypatch):
    monkeypatch.setenv(ENV, json.dumps({"my-new-mech": True}))
    assert _feature_enabled(
        profile="cheap",
        configured=False,
        override=None,
        name="my-new-mech",
    ) is True


def test_env_truthy_nonbool_values_coerced(monkeypatch):
    monkeypatch.setenv(ENV, json.dumps({"selector": 1, "judge_ensemble": 0}))
    assert _feature_enabled(
        profile="cheap", configured=False, override=None, name="selector",
    ) is True
    assert _feature_enabled(
        profile="max", configured=True, override=None, name="judge_ensemble",
    ) is False


def test_env_name_not_in_mapping_falls_through(monkeypatch):
    monkeypatch.setenv(ENV, json.dumps({"judge_ensemble": False}))
    # "selector" isn't a key -> normal precedence applies (override wins)
    assert _feature_enabled(
        profile="cheap",
        configured=False,
        override=True,
        name="selector",
    ) is True


def test_env_ignored_when_name_not_passed(monkeypatch):
    # Backward-compat path: callers that don't thread a name never consult env
    monkeypatch.setenv(ENV, json.dumps({"judge_ensemble": False}))
    assert _feature_enabled(
        profile="max",
        configured=True,
        override=None,
    ) is True


# ---------------------------------------------------------------------------
# Malformed env — ignored, falls through to existing logic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        "",
        "judge_ensemble=false",
        '{"judge_ensemble": fals',  # truncated
    ],
)
def test_malformed_json_ignored(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    # Falls through: explicit override wins as before
    assert _feature_enabled(
        profile="balanced",
        configured=False,
        override=True,
        name="judge_ensemble",
    ) is True
    # And profile logic still applies
    assert _feature_enabled(
        profile="max",
        configured=False,
        override=None,
        name="judge_ensemble",
    ) is True


@pytest.mark.parametrize("raw", ['["judge_ensemble"]', '"judge_ensemble"', "42", "null"])
def test_non_dict_json_ignored(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    assert _feature_enabled(
        profile="cheap",
        configured=True,
        override=None,
        name="judge_ensemble",
    ) is False  # cheap profile still wins


# ---------------------------------------------------------------------------
# Env unset — pre-C2 behavior unchanged
# ---------------------------------------------------------------------------


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def test_unset_env_override_wins(no_env):
    assert _feature_enabled(
        profile="cheap", configured=False, override=True, name="selector",
    ) is True
    assert _feature_enabled(
        profile="max", configured=True, override=False, name="selector",
    ) is False


def test_unset_env_max_profile_enables(no_env):
    assert _feature_enabled(
        profile="max", configured=False, override=None, name="judge_ensemble",
    ) is True
    # ...unless the feature opts out of max
    assert _feature_enabled(
        profile="max",
        configured=False,
        override=None,
        enabled_in_max=False,
        name="judge_ensemble",
    ) is False


def test_unset_env_cheap_profile_disables(no_env):
    assert _feature_enabled(
        profile="cheap", configured=True, override=None, name="cascade_guard",
    ) is False


def test_unset_env_balanced_uses_configured(no_env):
    assert _feature_enabled(
        profile="balanced", configured=True, override=None, name="selector",
    ) is True
    assert _feature_enabled(
        profile="balanced", configured=False, override=None, name="selector",
    ) is False
    # None/empty profile defaults to balanced
    assert _feature_enabled(
        profile="", configured=True, override=None, name="selector",
    ) is True
