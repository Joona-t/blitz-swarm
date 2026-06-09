"""MAST failure-mode detectors — rule-based, no LLM calls.

Implements 9 detectors from Cemri et al. arXiv 2503.13657 (NeurIPS 2025):
14 named failure modes across 3 categories (FC1 spec/design, FC2 inter-agent
misalignment, FC3 verification/termination).

Each detector returns True when the failure mode is present in the run
artifacts. The combined `detect_all()` returns the list of FM-X.Y codes
that fired. Detectors are deterministic and run cheaply during the bench
runner's post-processing — they do NOT inject failures (that's the job of
`bench/mast_regression.py` pytest cases).

Coverage: 9 of 14 modes have rule-based detectors. The 5 that require
mocked subprocess injection (FM-1.1, FM-2.2, FM-2.3, FM-3.1, FM-3.2) are
covered by the regression-suite pytest cases that monkey-patch agents.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _jaccard(a: str, b: str, *, n: int = 5) -> float:
    """Jaccard similarity over n-grams of two strings (rough approximation)."""
    if not a or not b:
        return 0.0
    a_tokens = a.lower().split()
    b_tokens = b.lower().split()
    a_grams = {tuple(a_tokens[i : i + n]) for i in range(len(a_tokens) - n + 1)}
    b_grams = {tuple(b_tokens[i : i + n]) for i in range(len(b_tokens) - n + 1)}
    if not a_grams or not b_grams:
        return 0.0
    inter = a_grams & b_grams
    union = a_grams | b_grams
    return len(inter) / len(union)


def _tokens(text: str) -> set[str]:
    """Lowercased word tokens (alphanumerics) of a string as a set."""
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _jaccard_tokens(a: str, b: str) -> float:
    """Jaccard similarity over the bag-of-word-tokens of two strings."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 0.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _stem(word: str) -> str:
    """Crude deterministic stemmer: strip a few common English suffixes.

    Not linguistically correct — just enough so that "scaling" matches
    "scale" and "agents" matches "agent" for coverage checks. Purely
    mechanical, no data, no dependencies.
    """
    w = word.lower()
    for suf in ("ization", "izations", "isation", "isations", "ies", "ing",
                "ers", "er", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            if suf == "ies":
                return w[: -len(suf)] + "y"
            return w[: -len(suf)]
    return w


def _shingles(text: str, *, n: int = 3) -> set[tuple[str, ...]]:
    """Word n-gram shingles of a string as a set of tuples."""
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


# ----------------------------------------------------------------------
# Research-quality detectors (pure-python, deterministic, no LLM)
# ----------------------------------------------------------------------
#
# These are additive instruments used by the bench runner to score the
# *content* of a produced research artifact, distinct from the MAST
# failure-mode detectors above (which score process failures). All are
# deterministic: same input -> same float, no network, no model calls.

# An arXiv identifier looks like YYMM.NNNNN (4-digit year+month, 4-5 digit
# sequence), optionally with a version suffix (vN). We split "claimed" from
# "well-formed": a candidate is anything that announces itself as an arXiv id
# (an `arXiv:` prefix, or a bare DDDD.DDDD+ run). Well-formed additionally
# requires a plausible month (01-12) and the canonical 4-5 digit sequence.
_ARXIV_CANDIDATE = re.compile(r"(?:arxiv:\s*)?(\d{4}\.\d{2,7})(?:v\d+)?", re.IGNORECASE)
_ARXIV_WELLFORMED = re.compile(r"^\d{4}\.\d{4,5}$")

_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.IGNORECASE)


def arxiv_id_validity(text: str) -> float:
    """Fraction of claimed arXiv IDs in `text` that are well-formed.

    A "claimed" id is any ``arXiv:NNNN.NNN…`` reference or bare
    ``DDDD.DDDD+`` token. "Well-formed" requires the canonical
    ``YYMM.NNNNN`` shape (4-5 digit sequence) AND a plausible month
    (01-12). Returns 1.0 when nothing arXiv-shaped is claimed (vacuously
    valid — the artifact made no arXiv claims to get wrong).
    """
    candidates = _ARXIV_CANDIDATE.findall(text or "")
    if not candidates:
        return 1.0
    good = 0
    for raw in candidates:
        stripped = raw.strip().rstrip(".")
        if not _ARXIV_WELLFORMED.match(stripped):
            continue
        month = int(stripped[2:4])
        if 1 <= month <= 12:
            good += 1
    return good / len(candidates)


def citation_grounding(text: str, sources: list[str]) -> float:
    """Fraction of URLs cited in `text` that appear in `sources`.

    Each URL found in the text counts as grounded if it occurs as a
    substring of any provided source string (after stripping a trailing
    slash and lowercasing both sides). Returns 1.0 when the text cites no
    URLs (nothing ungrounded can exist).
    """
    cited = _URL_RE.findall(text or "")
    if not cited:
        return 1.0
    norm_sources = [(s or "").strip().rstrip("/").lower() for s in (sources or [])]
    grounded = 0
    for url in cited:
        u = url.strip().rstrip("/").lower()
        if any(u in s or s in u for s in norm_sources if s):
            grounded += 1
    return grounded / len(cited)


def coverage_hits(text: str, expected_subtopics: list[str]) -> float:
    """Fraction of `expected_subtopics` mentioned in `text`.

    A subtopic is hit if it appears as a case-insensitive substring, OR if
    every stemmed token of the subtopic appears among the stemmed tokens of
    the text (so "scaling laws" matches text containing "scale" and "law").
    Returns 1.0 when no subtopics are expected (vacuously covered).
    """
    subtopics = [s for s in (expected_subtopics or []) if (s or "").strip()]
    if not subtopics:
        return 1.0
    haystack = (text or "").lower()
    text_stems = {_stem(t) for t in re.findall(r"[a-z0-9]+", haystack)}
    hits = 0
    for sub in subtopics:
        s = sub.lower().strip()
        if s in haystack:
            hits += 1
            continue
        sub_stems = {_stem(t) for t in re.findall(r"[a-z0-9]+", s)}
        if sub_stems and sub_stems <= text_stems:
            hits += 1
    return hits / len(subtopics)


def dedup_overlap(items: list[str]) -> float:
    """Shingle-overlap ratio across `items` (0 = all unique, 1 = identical).

    Mean pairwise Jaccard similarity over word-trigram shingles. With 0 or 1
    item there are no pairs, so redundancy is 0.0 by definition.
    """
    items = [i for i in (items or []) if (i or "").strip()]
    if len(items) < 2:
        return 0.0
    shingle_sets = [_shingles(i) for i in items]
    total = 0.0
    pairs = 0
    for a in range(len(shingle_sets)):
        for b in range(a + 1, len(shingle_sets)):
            sa, sb = shingle_sets[a], shingle_sets[b]
            union = sa | sb
            if union:
                total += len(sa & sb) / len(union)
            pairs += 1
    return total / pairs if pairs else 0.0


def novelty_vs_prior(text: str, prior_texts: list[str]) -> float:
    """Novelty of `text` against prior artifacts: 1 - max token-Jaccard.

    Returns 1.0 when there is no prior (everything is novel). A value near
    0 means `text` is near-duplicate of some prior artifact.
    """
    priors = [p for p in (prior_texts or []) if (p or "").strip()]
    if not priors:
        return 1.0
    max_sim = max(_jaccard_tokens(text, p) for p in priors)
    return 1.0 - max_sim


def citations_score(text: str, sources: list[str]) -> float:
    """Earned citation credit for the composite (audit fix H3).

    The individual detectors (``arxiv_id_validity``, ``citation_grounding``)
    return a *vacuous* 1.0 when the artifact makes no claims of their kind —
    correct for "fraction malformed/ungrounded", but wrong as composite
    credit: an output that cites NOTHING must score 0 on citations, not full
    marks.

      * 0.0 when the text cites nothing (no URLs AND no arXiv ids).
      * Otherwise ``wellformed_fraction * min(1.0, distinct_citations / 3)``
        where ``distinct_citations`` = distinct cited URLs + distinct claimed
        arXiv ids (3+ distinct citations earn full richness), and
        ``wellformed_fraction`` combines the existing detectors weighted by
        how many claims of each kind appear:
          - arXiv claims are scored by ``arxiv_id_validity``;
          - URL claims are scored by ``citation_grounding`` when ``sources``
            are provided; with no sources to ground against, URLs are taken
            at face value (1.0) rather than auto-penalised.

    Deterministic, pure-python, no LLM.
    """
    urls = _URL_RE.findall(text or "")
    arxiv_claims = [c.strip().rstrip(".") for c in _ARXIV_CANDIDATE.findall(text or "")]
    n_urls, n_arxiv = len(urls), len(arxiv_claims)
    if n_urls == 0 and n_arxiv == 0:
        return 0.0

    weighted = 0.0
    if n_arxiv:
        weighted += n_arxiv * arxiv_id_validity(text)
    if n_urls:
        url_quality = citation_grounding(text, list(sources)) if sources else 1.0
        weighted += n_urls * url_quality
    wellformed_fraction = weighted / (n_urls + n_arxiv)

    distinct = {u.strip().rstrip("/").lower() for u in urls} | set(arxiv_claims)
    richness = min(1.0, len(distinct) / 3.0)
    return wellformed_fraction * richness


# Composite weights — sum to 1.0. Rationale (audit fix H3):
#   coverage 0.45   answering the question fully is the primary goal
#   citations 0.30  citation credit must be EARNED (count + well-formedness +
#                   grounding) — replaces the old citation_grounding (0.25) +
#                   arxiv_validity (0.15) pair, whose vacuous 1.0s handed an
#                   uncited artifact 0.40 free composite credit
#   novelty 0.15    a fresh artifact, not a rehash of prior runs
#   low-dedup 0.10  (1 - dedup_overlap): penalty for internal redundancy
_RQ_WEIGHTS = {
    "coverage": 0.45,
    "citations": 0.30,
    "novelty": 0.15,
    "low_dedup": 0.10,
}


def research_quality_composite(
    output_md: str,
    slate_entry: dict,
    prior: Sequence[str] = (),
) -> dict:
    """Deterministic research-quality scorecard for one produced artifact.

    `slate_entry` supplies the ground-truth expectations for the prompt:
      - ``sources`` / ``expected_sources``: list[str] of allowed citation URLs
      - ``expected_subtopics`` / ``subtopics``: list[str] coverage targets
      - ``sections`` / ``items``: list[str] to score for internal redundancy
        (falls back to the markdown's own section blocks when absent)

    Composite (audit fix H3)::

        0.45*coverage + 0.30*citations + 0.15*novelty + 0.10*(1 - dedup)

    The ``citations`` component REPLACES the old ``citation_grounding`` +
    ``arxiv_validity`` pair in the composite. Rationale: both old detectors
    return a vacuous 1.0 when the artifact cites nothing, so under the old
    weights an uncited synthesis banked 0.40 of free credit — a research
    synthesis that cites NOTHING could outscore one that cites well. That is
    backwards: citing well is a core research-quality requirement, so credit
    is now earned via ``citations_score`` (0.0 for zero citations; otherwise
    well-formedness/grounding scaled by distinct-citation richness, capped
    at 3 distinct citations).

    The individual detector functions keep their original semantics and
    their raw values stay in the returned dict (``arxiv_validity``,
    ``citation_grounding``) for compatibility, alongside the new
    ``citations`` key. Only the composite weighting changed.

    Returns a dict with each sub-score plus the weighted ``composite`` in
    [0, 1]. The composite uses ``low_dedup = 1 - dedup_overlap`` so that
    less redundancy scores higher.
    """
    entry = slate_entry or {}
    sources = entry.get("sources") or entry.get("expected_sources") or []
    subtopics = entry.get("expected_subtopics") or entry.get("subtopics") or []
    items = entry.get("sections") or entry.get("items")
    if not items:
        # Fall back to splitting the markdown into section-ish blocks so
        # dedup still has something to chew on for a single artifact.
        items = [b for b in re.split(r"\n#{1,6}\s|\n\n", output_md or "") if b.strip()]

    arxiv = arxiv_id_validity(output_md)
    grounding = citation_grounding(output_md, list(sources))
    citations = citations_score(output_md, list(sources))
    coverage = coverage_hits(output_md, list(subtopics))
    dedup = dedup_overlap(list(items))
    novelty = novelty_vs_prior(output_md, list(prior))

    composite = (
        _RQ_WEIGHTS["coverage"] * coverage
        + _RQ_WEIGHTS["citations"] * citations
        + _RQ_WEIGHTS["novelty"] * novelty
        + _RQ_WEIGHTS["low_dedup"] * (1.0 - dedup)
    )
    # Clamp against floating-point drift so callers always get [0, 1].
    composite = max(0.0, min(1.0, composite))

    return {
        "arxiv_validity": arxiv,
        "citation_grounding": grounding,
        "citations": citations,
        "coverage": coverage,
        "dedup_overlap": dedup,
        "novelty": novelty,
        "composite": composite,
    }


# ----------------------------------------------------------------------
# FC1 — Specification & System Design
# ----------------------------------------------------------------------


def detect_FM_1_2_role_mismatch(outputs: list[dict], role_hints: dict[str, str]) -> bool:
    """FM-1.2 Disobey Role Specification.

    Hint heuristic: if a critic's findings have <50% overlap with critique
    keywords, role is mismatched. Returns True if ANY agent appears off-role.
    """
    critic_words = {"flag", "issue", "missing", "unsupported", "weak", "gap", "concern", "criticize"}
    researcher_words = {"finding", "evidence", "investigate", "research", "analyze", "data", "claim"}
    for o in outputs:
        role = o.get("role", "")
        text = (o.get("findings", "") or "").lower()
        if not text:
            continue
        if role == "critic":
            hits = sum(1 for w in critic_words if w in text)
            if hits < 2:
                return True
        elif role == "researcher":
            hits = sum(1 for w in researcher_words if w in text)
            if hits < 2:
                return True
    return False


def detect_FM_1_3_step_repetition(rounds: list[list[dict]]) -> bool:
    """FM-1.3 Step Repetition.

    Compares round-N findings to round-(N-1) findings for each agent.
    If Jaccard > 0.95 on findings text, that agent repeated itself.
    """
    if len(rounds) < 2:
        return False
    for i in range(1, len(rounds)):
        prev_by_id = {o.get("agent_id"): o.get("findings", "") for o in rounds[i - 1]}
        curr_by_id = {o.get("agent_id"): o.get("findings", "") for o in rounds[i]}
        for agent_id, curr_text in curr_by_id.items():
            prev_text = prev_by_id.get(agent_id, "")
            if prev_text and curr_text and _jaccard(prev_text, curr_text) > 0.95:
                return True
    return False


def detect_FM_1_4_loss_of_history(round_2_contexts: list[int]) -> bool:
    """FM-1.4 Loss of Conversation History.

    `round_2_contexts` is a list of context-string lengths seen by agents
    in round 2. If ANY agent received empty context in round 2+, history
    was lost.
    """
    return any(ctx_len == 0 for ctx_len in round_2_contexts)


# ----------------------------------------------------------------------
# FC2 — Inter-Agent Misalignment
# ----------------------------------------------------------------------


def detect_FM_2_1_oscillating_consensus(rounds: list[list[dict]]) -> bool:
    """FM-2.1 Conversation Reset (proxy: non-monotone ready-vote progress).

    If ready_votes go up then down across rounds, consensus is oscillating
    rather than approaching.
    """
    if len(rounds) < 3:
        return False
    ready_counts = []
    for outputs in rounds:
        ready_counts.append(sum(1 for o in outputs if o.get("quality_vote") == "ready"))
    # Oscillation: any peak followed by a strict decrease followed by another increase
    for i in range(1, len(ready_counts) - 1):
        if ready_counts[i] > ready_counts[i - 1] and ready_counts[i + 1] < ready_counts[i]:
            return True
    return False


def detect_FM_2_4_information_withholding(outputs: list[dict]) -> bool:
    """FM-2.4 Information Withholding.

    Researcher returned findings but `key_points` is empty.
    """
    for o in outputs:
        if o.get("role") == "researcher":
            findings = (o.get("findings", "") or "").strip()
            key_points = o.get("key_points") or []
            if findings and not key_points:
                return True
    return False


def detect_FM_2_5_ignored_other_agents(rounds: list[list[dict]]) -> bool:
    """FM-2.5 Ignored Other Agents' Input.

    Critic gave specific feedback in round N; round N+1 researcher findings
    have Jaccard > 0.95 with round N findings (i.e., they didn't change).
    """
    if len(rounds) < 2:
        return False
    for i in range(1, len(rounds)):
        prev_critics = [o for o in rounds[i - 1] if o.get("role") == "critic"
                        and o.get("quality_vote") == "needs_work"]
        if not prev_critics:
            continue
        prev_researchers = {o.get("agent_id"): o.get("findings", "")
                            for o in rounds[i - 1] if o.get("role") == "researcher"}
        curr_researchers = {o.get("agent_id"): o.get("findings", "")
                            for o in rounds[i] if o.get("role") == "researcher"}
        for agent_id, curr_text in curr_researchers.items():
            prev_text = prev_researchers.get(agent_id, "")
            if prev_text and curr_text and _jaccard(prev_text, curr_text) > 0.95:
                return True
    return False


def detect_FM_2_6_reasoning_action_mismatch(outputs: list[dict]) -> bool:
    """FM-2.6 Reasoning-Action Mismatch.

    confidence < 0.4 paired with quality_vote == "ready" is internally
    inconsistent — the model said it's uncertain but voted ready anyway.
    """
    for o in outputs:
        conf = o.get("confidence", 1.0)
        vote = o.get("quality_vote")
        if conf is not None and conf < 0.4 and vote == "ready":
            return True
    return False


# ----------------------------------------------------------------------
# FC3 — Verification & Termination
# ----------------------------------------------------------------------


def detect_FM_3_1_silent_partial(rounds: list[list[dict]]) -> bool:
    """FM-3.1 Premature Termination (proxy: silent partial output).

    Any agent recorded `_partial=True` (timeout recovered partial output)
    AND the run still produced a final document — i.e., we shipped something
    on partial signals without flagging it.
    """
    for outputs in rounds:
        for o in outputs:
            if o.get("_partial") and not o.get("_flagged_partial"):
                return True
    return False


def detect_FM_3_3_incorrect_verification(outputs: list[dict], output_md_chars: int) -> bool:
    """FM-3.3 Incorrect Verification.

    quality_judge gave avg score >8 BUT output document is suspiciously short
    (< 500 chars) — judge approved an obviously thin output.
    """
    judges = [o for o in outputs if o.get("role") == "quality_judge"]
    if not judges:
        return False
    judge = judges[0]
    avg = sum(
        judge.get(k, 0)
        for k in ("coverage_score", "accuracy_score", "clarity_score", "depth_score")
    ) / 4.0
    return avg > 8.0 and output_md_chars < 500


# ----------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------


def detect_all(
    rounds: list[list[dict]],
    *,
    output_md_chars: int = 0,
    round_2_context_lengths: list[int] | None = None,
    role_hints: dict[str, str] | None = None,
) -> list[str]:
    """Run all detectors over a completed swarm run. Returns FM-X.Y codes."""
    flat_outputs = [o for round_outputs in rounds for o in round_outputs]
    role_hints = role_hints or {}
    flags: list[str] = []

    if detect_FM_1_2_role_mismatch(flat_outputs, role_hints):
        flags.append("FM-1.2")
    if detect_FM_1_3_step_repetition(rounds):
        flags.append("FM-1.3")
    if round_2_context_lengths is not None and detect_FM_1_4_loss_of_history(round_2_context_lengths):
        flags.append("FM-1.4")
    if detect_FM_2_1_oscillating_consensus(rounds):
        flags.append("FM-2.1")
    if detect_FM_2_4_information_withholding(flat_outputs):
        flags.append("FM-2.4")
    if detect_FM_2_5_ignored_other_agents(rounds):
        flags.append("FM-2.5")
    if detect_FM_2_6_reasoning_action_mismatch(flat_outputs):
        flags.append("FM-2.6")
    if detect_FM_3_1_silent_partial(rounds):
        flags.append("FM-3.1")
    if detect_FM_3_3_incorrect_verification(flat_outputs, output_md_chars):
        flags.append("FM-3.3")

    return flags


# Mapping of FM codes to human-readable names (for reports)
FM_NAMES = {
    "FM-1.1": "Disobey Task Specification",
    "FM-1.2": "Disobey Role Specification",
    "FM-1.3": "Step Repetition",
    "FM-1.4": "Loss of Conversation History",
    "FM-1.5": "Unaware of Termination",
    "FM-2.1": "Conversation Reset",
    "FM-2.2": "Fail to Ask for Clarification",
    "FM-2.3": "Task Derailment",
    "FM-2.4": "Information Withholding",
    "FM-2.5": "Ignored Other Agents' Input",
    "FM-2.6": "Reasoning-Action Mismatch",
    "FM-3.1": "Premature Termination",
    "FM-3.2": "No or Incomplete Verification",
    "FM-3.3": "Incorrect Verification",
}
