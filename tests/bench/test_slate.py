"""Schema and distribution tests for bench/slate_v1.toml."""

from __future__ import annotations

from collections import Counter

import pytest

from bench import BenchPrompt, BenchSlate, load_slate


def test_slate_loads(slate_path):
    slate = load_slate(slate_path)
    assert isinstance(slate, BenchSlate)
    assert len(slate) == 30
    assert slate.schema_version >= 1
    assert slate.slate_id == "v1"
    assert len(slate.sha256) == 64  # hex sha256


def test_slate_prompt_ids_unique(slate_path):
    slate = load_slate(slate_path)
    ids = [p.id for p in slate.prompts]
    assert len(ids) == len(set(ids)), "prompt ids must be unique"


def test_slate_tier_distribution(slate_path):
    slate = load_slate(slate_path)
    counts = Counter(p.tier for p in slate.prompts)
    assert counts["easy"] == 15, f"expected 15 easy prompts, got {counts['easy']}"
    assert counts["medium"] == 10, f"expected 10 medium prompts, got {counts['medium']}"
    assert counts["hard"] == 5, f"expected 5 hard prompts, got {counts['hard']}"


def test_slate_domain_coverage(slate_path):
    slate = load_slate(slate_path)
    domains = {p.domain for p in slate.prompts}
    expected = {"technical", "open", "adversarial", "multi", "compositional"}
    assert expected.issubset(domains), f"missing domains: {expected - domains}"


def test_slate_adversarial_count(slate_path):
    slate = load_slate(slate_path)
    adv = [p for p in slate.prompts if p.domain == "adversarial"]
    assert len(adv) >= 4, f"expected >=4 adversarial prompts, got {len(adv)}"


def test_slate_budgets_valid(slate_path):
    slate = load_slate(slate_path)
    for p in slate.prompts:
        assert p.budget_usd > 0, f"{p.id} has non-positive budget"
        if p.tier == "easy":
            assert p.budget_usd <= 0.10, f"{p.id} easy budget too high"
        elif p.tier == "hard":
            assert p.budget_usd >= 0.30, f"{p.id} hard budget too low"


def test_slate_expected_coverage_nonempty(slate_path):
    slate = load_slate(slate_path)
    for p in slate.prompts:
        assert p.expected_coverage, f"{p.id} has empty expected_coverage"
        assert len(p.expected_coverage) >= 3, f"{p.id} has too few coverage keywords"


def test_slate_filter_by_tier(slate_path):
    slate = load_slate(slate_path)
    hard = slate.filter(tiers=["hard"])
    assert len(hard) == 5
    assert all(p.tier == "hard" for p in hard)


def test_slate_filter_by_id(slate_path):
    slate = load_slate(slate_path)
    subset = slate.filter(ids=["s001", "s016", "s026"])
    assert {p.id for p in subset} == {"s001", "s016", "s026"}


def test_slate_filter_parallelizable(slate_path):
    slate = load_slate(slate_path)
    par = slate.filter(parallelizable=True)
    seq = slate.filter(parallelizable=False)
    assert len(par) + len(seq) == 30
    assert all(p.parallelizable for p in par)
    assert all(not p.parallelizable for p in seq)


def test_slate_by_id_lookup(slate_path):
    slate = load_slate(slate_path)
    p = slate.by_id("s001")
    assert p.id == "s001"
    assert "WAL" in p.text


def test_slate_by_id_missing_raises(slate_path):
    slate = load_slate(slate_path)
    with pytest.raises(KeyError):
        slate.by_id("nonexistent")


def test_slate_sha256_deterministic(slate_path):
    """Same file path → same sha256."""
    a = load_slate(slate_path)
    b = load_slate(slate_path)
    assert a.sha256 == b.sha256
