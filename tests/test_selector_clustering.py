"""AUDIT M2: headings that mean the same thing must land in one cluster."""

from __future__ import annotations

from mechanisms.selector_synth import SelectorConfig, SelectorSynth, Span, heading_words


def _span(i, heading, agent):
    return Span(id=f"s{i}", source_agent=agent, source_round=1, heading=heading, text=f"body {i}")


def test_heading_words_normalise():
    assert heading_words("Risk Analysis") == heading_words("Risks") == frozenset({"risk"})
    assert heading_words("3. Key Risks:") == frozenset({"3", "risk"})     # enumerator kept
    assert heading_words("Section A") != heading_words("Section B")
    assert heading_words("Implementation Details") == frozenset({"implem"})
    assert heading_words(None) == frozenset()


def test_similar_headings_cluster_together():
    sel = SelectorSynth(SelectorConfig(granularity="section"))
    spans = [_span(1, "Risks", "a"), _span(2, "Risk Analysis", "b"), _span(3, "Key risks", "c"),
             _span(4, "Implementation", "a"), _span(5, "Implementation details", "b"), _span(6, None, "c")]
    clusters = sel._cluster_spans(spans)
    sizes = sorted(len(v) for v in clusters.values())
    assert sizes == [1, 2, 3], clusters.keys()
    assert "_unsorted" in clusters


def test_avg_quality_counts_reported_zeros():
    from metrics import MetricsCollector
    mc = MetricsCollector()
    mc.start_run("r", "t")
    mc.record_quality_scores({"coverage_score": 8, "accuracy_score": 0, "clarity_score": 8, "depth_score": 8})
    assert mc.run.avg_quality == 6.0
    mc.record_quality_scores({"coverage_score": 9})
    assert mc.run.avg_quality == 9.0
