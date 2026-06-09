"""Deterministic unit tests for the research-quality detectors in
``bench/detectors.py``.

These are pure-python, no-LLM instruments: every fixture below maps a known
input to a known output. Edge cases covered: empty text, no arXiv IDs claimed,
full vs partial coverage, total duplication vs all-unique, no-prior novelty,
and ungrounded citations.
"""

from __future__ import annotations

import math

from bench.detectors import (
    _RQ_WEIGHTS,
    arxiv_id_validity,
    citation_grounding,
    coverage_hits,
    dedup_overlap,
    novelty_vs_prior,
    research_quality_composite,
)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, abs_tol=tol)


# ----------------------------------------------------------------------
# arxiv_id_validity
# ----------------------------------------------------------------------


def test_arxiv_no_ids_claimed_returns_one():
    assert arxiv_id_validity("a report with no paper identifiers at all") == 1.0


def test_arxiv_empty_text_returns_one():
    assert arxiv_id_validity("") == 1.0


def test_arxiv_all_wellformed():
    text = "See arXiv:2503.13657 and arXiv:2310.06825 for details."
    assert arxiv_id_validity(text) == 1.0


def test_arxiv_wellformed_bare_and_prefixed():
    # bare 4.5 id + prefixed 4.4 id, both well-formed
    text = "refs: 2503.13657 and arXiv:2401.1234"
    assert arxiv_id_validity(text) == 1.0


def test_arxiv_version_suffix_is_wellformed():
    assert arxiv_id_validity("arXiv:2503.13657v2") == 1.0


def test_arxiv_half_malformed():
    # 2503.13657 well-formed; 2503.13 too few sequence digits (malformed)
    text = "good arXiv:2503.13657 but bad arXiv:2503.13"
    assert _close(arxiv_id_validity(text), 0.5)


def test_arxiv_invalid_month_is_malformed():
    # month 99 is implausible -> malformed despite right digit shape
    text = "arXiv:2599.12345 and arXiv:2503.13657"
    assert _close(arxiv_id_validity(text), 0.5)


def test_arxiv_all_malformed_returns_zero():
    text = "arXiv:2503.13 and arXiv:2599.99999"
    assert arxiv_id_validity(text) == 0.0


# ----------------------------------------------------------------------
# citation_grounding
# ----------------------------------------------------------------------


def test_citation_no_urls_returns_one():
    assert citation_grounding("plain prose with zero links", ["https://a.com"]) == 1.0


def test_citation_empty_text_returns_one():
    assert citation_grounding("", []) == 1.0


def test_citation_all_grounded():
    text = "see https://arxiv.org/abs/2503.13657 and https://example.com/page"
    sources = ["https://arxiv.org/abs/2503.13657", "https://example.com/page"]
    assert citation_grounding(text, sources) == 1.0


def test_citation_trailing_slash_normalized():
    text = "ref https://example.com/page/"
    sources = ["https://example.com/page"]
    assert citation_grounding(text, sources) == 1.0


def test_citation_half_grounded():
    text = "real https://example.com/a fake https://hallucinated.example/zzz"
    sources = ["https://example.com/a"]
    assert _close(citation_grounding(text, sources), 0.5)


def test_citation_none_grounded():
    text = "made up https://nope.example/x"
    sources = ["https://different.example/y"]
    assert citation_grounding(text, sources) == 0.0


def test_citation_no_sources_with_urls_is_zero():
    text = "cite https://example.com/a"
    assert citation_grounding(text, []) == 0.0


# ----------------------------------------------------------------------
# coverage_hits
# ----------------------------------------------------------------------


def test_coverage_no_expected_returns_one():
    assert coverage_hits("anything", []) == 1.0


def test_coverage_full():
    text = "We discuss scaling laws and reinforcement learning and tokenization."
    expected = ["scaling laws", "reinforcement learning", "tokenization"]
    assert coverage_hits(text, expected) == 1.0


def test_coverage_full_via_stemming():
    # "scaling" should hit the "scale" stem; "agents" should hit "agent"
    text = "the system scales well across many agents in production"
    expected = ["scaling", "agent"]
    assert coverage_hits(text, expected) == 1.0


def test_coverage_partial():
    text = "We only discuss scaling laws here."
    expected = ["scaling laws", "reinforcement learning"]
    assert _close(coverage_hits(text, expected), 0.5)


def test_coverage_none_on_empty_text():
    assert coverage_hits("", ["alpha", "beta"]) == 0.0


def test_coverage_case_insensitive():
    assert coverage_hits("ALPHA topic covered", ["alpha"]) == 1.0


# ----------------------------------------------------------------------
# dedup_overlap
# ----------------------------------------------------------------------


def test_dedup_empty_returns_zero():
    assert dedup_overlap([]) == 0.0


def test_dedup_single_item_returns_zero():
    assert dedup_overlap(["only one block of text here"]) == 0.0


def test_dedup_all_unique():
    items = [
        "alpha bravo charlie delta echo",
        "foxtrot golf hotel india juliet",
        "kilo lima mike november oscar",
    ]
    assert dedup_overlap(items) == 0.0


def test_dedup_total_duplication():
    block = "the exact same sentence repeated several times over again"
    assert _close(dedup_overlap([block, block, block]), 1.0)


def test_dedup_partial_between_zero_and_one():
    items = [
        "shared opening words then unique alpha bravo charlie",
        "shared opening words then unique delta echo foxtrot",
    ]
    val = dedup_overlap(items)
    assert 0.0 < val < 1.0


# ----------------------------------------------------------------------
# novelty_vs_prior
# ----------------------------------------------------------------------


def test_novelty_no_prior_returns_one():
    assert novelty_vs_prior("a brand new artifact", []) == 1.0


def test_novelty_identical_prior_returns_zero():
    text = "exactly the same words in both texts"
    assert _close(novelty_vs_prior(text, [text]), 0.0)


def test_novelty_disjoint_prior_returns_one():
    assert _close(novelty_vs_prior("alpha bravo charlie", ["delta echo foxtrot"]), 1.0)


def test_novelty_takes_max_similarity():
    # text shares more with the second prior; novelty driven by the max sim
    text = "alpha bravo charlie delta"
    priors = ["zulu yankee xray", "alpha bravo charlie delta"]
    assert _close(novelty_vs_prior(text, priors), 0.0)


def test_novelty_partial():
    # text tokens {alpha,bravo,charlie}, prior {alpha,bravo,delta,echo}
    # intersection 2, union 5 -> Jaccard 0.4 -> novelty 0.6
    val = novelty_vs_prior("alpha bravo charlie", ["alpha bravo delta echo"])
    assert _close(val, 1.0 - (2 / 5))


# ----------------------------------------------------------------------
# research_quality_composite
# ----------------------------------------------------------------------


def test_weights_sum_to_one():
    assert _close(sum(_RQ_WEIGHTS.values()), 1.0)


def test_composite_keys_present():
    out = research_quality_composite("text", {}, prior=())
    for key in ("arxiv_validity", "citation_grounding", "coverage",
                "dedup_overlap", "novelty", "composite"):
        assert key in out


def test_composite_perfect_score():
    # Full coverage, all citations grounded, valid arXiv id, no prior (novel),
    # distinct sections (low dedup) -> composite must be exactly 1.0.
    md = (
        "# Scaling\nWe cover scaling laws here, citing https://example.com/a\n"
        "# Methods\nReinforcement learning details from arXiv:2503.13657\n"
        "# Refs\nAlso see https://example.com/b for tokenization."
    )
    slate = {
        "sources": ["https://example.com/a", "https://example.com/b"],
        "expected_subtopics": ["scaling laws", "reinforcement learning", "tokenization"],
        "sections": [
            "alpha bravo charlie delta echo",
            "foxtrot golf hotel india juliet",
        ],
    }
    out = research_quality_composite(md, slate, prior=())
    assert out["coverage"] == 1.0
    assert out["citation_grounding"] == 1.0
    assert out["arxiv_validity"] == 1.0
    assert out["novelty"] == 1.0
    assert out["dedup_overlap"] == 0.0
    assert _close(out["composite"], 1.0)


def test_composite_worst_score():
    # Empty markdown: no coverage of expected topics, no arxiv (vacuous 1.0),
    # no urls (grounding vacuous 1.0), identical prior (novelty 0).
    slate = {"expected_subtopics": ["alpha", "beta"], "sources": []}
    out = research_quality_composite("", slate, prior=[""])
    # empty prior text is filtered out -> treated as "no prior" -> novelty 1.0
    assert out["coverage"] == 0.0
    assert out["novelty"] == 1.0
    # composite = 0.4*0 + 0.25*1 + 0.15*1 + 0.15*1 + 0.05*1
    assert _close(out["composite"], 0.25 + 0.15 + 0.15 + 0.05)


def test_composite_matches_manual_weighting():
    md = "Only scaling laws discussed, cited badly https://nope.example/x"
    slate = {
        "expected_subtopics": ["scaling laws", "reinforcement learning"],  # 0.5 coverage
        "sources": ["https://real.example/y"],  # url not grounded -> 0.0
    }
    prior = ["Only scaling laws discussed, cited badly https://nope.example/x"]  # identical -> novelty 0
    out = research_quality_composite(md, slate, prior=prior)
    assert _close(out["coverage"], 0.5)
    assert out["citation_grounding"] == 0.0
    assert out["arxiv_validity"] == 1.0  # no arxiv claimed
    assert _close(out["novelty"], 0.0)
    expected = (
        _RQ_WEIGHTS["coverage"] * out["coverage"]
        + _RQ_WEIGHTS["citation_grounding"] * out["citation_grounding"]
        + _RQ_WEIGHTS["arxiv_validity"] * out["arxiv_validity"]
        + _RQ_WEIGHTS["novelty"] * out["novelty"]
        + _RQ_WEIGHTS["low_dedup"] * (1.0 - out["dedup_overlap"])
    )
    assert _close(out["composite"], expected)


def test_composite_in_unit_interval():
    md = "arbitrary text with https://example.com/a and arXiv:2503.13657"
    slate = {"expected_subtopics": ["x"], "sources": ["https://example.com/a"]}
    out = research_quality_composite(md, slate, prior=["something else entirely"])
    assert 0.0 <= out["composite"] <= 1.0
