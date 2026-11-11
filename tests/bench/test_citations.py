"""M1 citation resolver (Phase 1 WP4/WP5), with the known-bad fixture: a
real ID, a fabricated ID in valid format, and a real DOI given the wrong
title must be classified differently, or the check cannot fail."""

from __future__ import annotations

import json
import os

import pytest

from bench import citations as C
from bench.detectors import research_quality_composite

REAL_ARXIV = "1706.03762"
FAKE_ARXIV = "2399.12345"          # valid format, does not exist
REAL_DOI = "10.1038/s41586-021-03819-2"

REPORT = f"""# Report

The Transformer architecture was introduced in "Attention Is All You Need" (arXiv:{REAL_ARXIV}).

A follow-up claims quantum speedups for sparse attention heads (arXiv:{FAKE_ARXIV}).

Protein structure results appear in "A Survey of Reinforcement Learning for Traffic Signals"
(https://doi.org/{REAL_DOI}).

See also https://example.org/alive and https://example.org/gone and https://example.org/blocked.
"""

ATOM = f"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/{REAL_ARXIV}v7</id><title>Attention Is All
 You Need</title></entry></feed>"""


class FakeNet:
    def __init__(self, *, down=False):
        self.down = down
        self.calls: list[tuple[str, str]] = []

    def __call__(self, url: str, method: str):
        self.calls.append((method, url))
        if self.down:
            return 0, ""
        if "export.arxiv.org" in url:
            return 200, ATOM
        if "api.crossref.org" in url:
            if REAL_DOI in url:
                return 200, json.dumps({"message": {"title": ["Highly accurate protein structure prediction with AlphaFold"]}})
            return 404, ""
        if "archive.org/wayback" in url:
            return 200, json.dumps({"archived_snapshots": {}})
        if url.endswith("/alive"):
            return 200, ""
        if url.endswith("/gone"):
            return 404, ""
        if url.endswith("/blocked"):
            return 403, ""
        return 0, ""


def _by_ident(result):
    return {i["ident"]: i["status"] for i in result["items"]}


def test_extract_dedupes_and_maps_doi_and_arxiv_urls():
    cites = C.extract(REPORT + f"\nAgain https://arxiv.org/abs/{REAL_ARXIV}v2 and doi:{REAL_DOI}.")
    kinds = sorted((c.kind, c.ident) for c in cites)
    assert ("arxiv", REAL_ARXIV) in kinds and ("arxiv", FAKE_ARXIV) in kinds
    assert ("doi", REAL_DOI) in kinds
    assert sum(1 for k, _ in kinds if k == "url") == 3
    assert len(kinds) == 6, "arxiv.org / doi.org URLs must not be double-counted as URLs"


def test_known_bad_fixture_classes_differ(tmp_path):
    net = FakeNet()
    r = C.check_report(REPORT, fetch=net, cache_path=tmp_path / "c.json", sleep=lambda s: None)
    st = _by_ident(r)
    assert st[REAL_ARXIV] == C.RESOLVED
    assert st[FAKE_ARXIV] == C.NOT_FOUND
    assert st[REAL_DOI] == C.TITLE_MISMATCH, "a real DOI attached to the wrong paper is not a pass"
    assert st["https://example.org/alive"] == C.RESOLVED
    assert st["https://example.org/gone"] == C.NOT_FOUND
    assert st["https://example.org/blocked"] == C.UNVERIFIABLE
    assert (r["n_cited"], r["resolved"], r["title_mismatch"], r["not_found"], r["unverifiable"]) == (6, 2, 1, 2, 1)
    assert r["ratio"] == pytest.approx(2 / 5), "unverifiable is excluded from the denominator"


def test_cache_makes_reruns_offline_and_never_stores_unverifiable(tmp_path):
    cache = tmp_path / "c.json"
    C.check_report(REPORT, fetch=FakeNet(), cache_path=cache, sleep=lambda s: None)
    stored = json.loads(cache.read_text())
    assert "url:https://example.org/blocked" not in stored
    assert stored[f"arxiv:{FAKE_ARXIV}"]["exists"] is False
    dead = FakeNet(down=True)
    r = C.check_report(REPORT, fetch=dead, cache_path=cache, offline=True)
    assert dead.calls == []
    st = _by_ident(r)
    assert st[REAL_ARXIV] == C.RESOLVED and st[FAKE_ARXIV] == C.NOT_FOUND and st[REAL_DOI] == C.TITLE_MISMATCH
    assert st["https://example.org/blocked"] == C.UNVERIFIABLE


def test_network_down_is_unverifiable_not_zero(tmp_path):
    r = C.check_report(REPORT, fetch=FakeNet(down=True), cache_path=tmp_path / "c.json", sleep=lambda s: None)
    assert r["unverifiable"] == r["n_cited"] == 6
    assert r["ratio"] is None
    assert C.citations_v2_score(r) is None, "an outage must not read as a score of 0"


def test_bare_identifier_cannot_mismatch_and_no_citations_scores_zero():
    assert C.title_check("Attention Is All You Need", f"[1] arXiv:{REAL_ARXIV}", REAL_ARXIV) == C.RESOLVED
    assert C.citations_v2_score({"n_cited": 0, "ratio": None, "resolved": 0}) == 0.0
    assert C.citations_v2_score({"n_cited": 4, "ratio": 0.5, "resolved": 2}) == pytest.approx(0.5 * 2 / 3)
    assert C.citations_v2_score({"n_cited": 5, "ratio": 1.0, "resolved": 5}) == 1.0


def test_composite_v2_takes_away_credit_for_fabricated_citations(tmp_path):
    fake = "\n".join(f"Claim {i} is established (arXiv:230{i}.9999{i})." for i in range(1, 4)) + "\nkeyword_a keyword_b"
    entry = {"expected_subtopics": ["keyword_a", "keyword_b"]}

    def all_missing(url, method):
        return (200, '<feed xmlns="http://www.w3.org/2005/Atom"></feed>') if "arxiv" in url else (0, "")

    check = C.check_report(fake, fetch=all_missing, cache_path=tmp_path / "c.json", sleep=lambda s: None)
    assert check["not_found"] == 3
    v = research_quality_composite(fake, entry, citation_check=check)
    assert v["citations"] >= 0.9, "v1 still gives format credit (kept for comparability)"
    assert v["citations_v2"] == 0.0
    assert v["composite_v2"] < v["composite"] - 0.25
    plain = research_quality_composite(fake, entry)
    assert "composite_v2" in plain and plain["composite_v2"] is None, "no check supplied -> v2 is None, not a guess"


@pytest.mark.skipif(not os.environ.get("BLITZ_LIVE_NET"), reason="set BLITZ_LIVE_NET=1 for the live arXiv check")
def test_live_arxiv_resolves_real_and_rejects_fabricated(tmp_path):
    text = f'"Attention Is All You Need" (arXiv:{REAL_ARXIV}) and a made-up one (arXiv:{FAKE_ARXIV}).'
    r = C.check_report(text, cache_path=tmp_path / "c.json")
    st = _by_ident(r)
    assert st[REAL_ARXIV] == C.RESOLVED and st[FAKE_ARXIV] == C.NOT_FOUND


def test_beta_coverage_exact_values():
    from bench.detectors import beta_coverage
    expected = ["alpha", "beta", "gamma", "delta"]
    drafts = ["covers ALPHA and beta", "covers alpha and gamma"]
    b = beta_coverage(drafts, expected)
    assert b["beta"] == 0.25                       # only delta is missed by every draft
    assert b["miss_rates"] == [0.5, 0.5]
    assert beta_coverage([], expected)["beta"] is None
