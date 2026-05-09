"""Selection-bottleneck synthesizer — Maryanskyy arXiv 2603.20324 (Mar 2026).

Replaces blended synthesis ("synthesizer averages findings") with
judge-driven pairwise span selection ("synthesizer picks best span per
section"). Per the paper: diverse-team-with-judge-selection wins 0.81
vs MoA-style synthesis 0.51 across 42 tasks. The aggregation choice
(selection vs synthesis) dominates output quality once judge skill
exceeds the s* ≈ 0.567 crossover threshold.

Pipeline:
    1. split_into_spans   — break each researcher output into spans
                            (whole / section / paragraph granularity)
    2. cluster_spans      — group spans by sub-topic
    3. select_within_cluster — pairwise judging + Bradley-Terry MLE
                            picks the highest-skill span per cluster
    4. assemble           — concatenate selected spans, optional
                            haiku-smoothed transitions

Bradley-Terry MLE: iterative Minorization-Maximization (MM).
No scipy required. Converges in ~30 iterations for typical N<=10 spans.

Pluggable hooks: `pairwise_judge_fn` is the only LLM call. Default
mock implementation is for tests; orchestrator integration replaces
with a Claude Sonnet subprocess.
"""

from __future__ import annotations

import itertools
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SelectorConfig:
    granularity: Literal["whole", "section", "paragraph"] = "section"
    n_judges: int = 3
    judge_models: tuple[str, ...] = ("sonnet", "sonnet", "haiku")
    pairwise_random_order: bool = True
    bt_regularization: float = 1e-3
    bt_max_iterations: int = 100
    bt_tolerance: float = 1e-6
    min_section_chars: int = 200
    transition_strategy: Literal["smooth", "concat"] = "concat"
    smooth_model: str = "haiku"
    candidate_cap: int = 5             # qualifier round if more


@dataclass
class Span:
    id: str
    source_agent: str
    source_round: int
    heading: str | None
    text: str


@dataclass
class PairwiseVerdict:
    judge_id: str
    span_a_id: str
    span_b_id: str
    winner: Literal["a", "b", "tie"]
    rationale: str = ""


@dataclass
class SelectionResult:
    selected_span_ids: list[str]
    bt_scores: dict[str, float]
    sections_assembled: list[Span]
    final_text: str
    diagnostics: dict = field(default_factory=dict)


# Type alias for the LLM hook
PairwiseJudgeFn = Callable[[Span, Span, str, str, int], PairwiseVerdict]


# ---------------------------------------------------------------------------
# Default mock pairwise judge (deterministic, for tests)
# ---------------------------------------------------------------------------


def mock_pairwise_judge(
    span_a: Span,
    span_b: Span,
    topic: str,
    judge_id: str,
    seed: int,
) -> PairwiseVerdict:
    """Deterministic mock that picks the longer span as the winner.

    A real Sonnet judge picks based on factual accuracy, specificity,
    coverage, and clarity — see prompts/general/quality_judge.md or the
    pairwise prompt in the deep dive.
    """
    a_len = len(span_a.text)
    b_len = len(span_b.text)
    if a_len > b_len * 1.1:
        winner = "a"
    elif b_len > a_len * 1.1:
        winner = "b"
    else:
        winner = "tie"
    return PairwiseVerdict(
        judge_id=judge_id,
        span_a_id=span_a.id,
        span_b_id=span_b.id,
        winner=winner,
        rationale=f"mock: len(a)={a_len} len(b)={b_len}",
    )


# ---------------------------------------------------------------------------
# Span splitting
# ---------------------------------------------------------------------------


_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_H3_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)


def _split_by_h2(text: str) -> list[tuple[str | None, str]]:
    """Split markdown text by ## headings. Returns [(heading, body), ...].

    Content before the first heading goes under heading=None.
    """
    if not text:
        return []
    matches = list(_H2_RE.finditer(text))
    if not matches:
        return [(None, text.strip())]
    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        prelude = text[: matches[0].start()].strip()
        if prelude:
            sections.append((None, prelude))
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def _split_by_paragraph(text: str) -> list[str]:
    """Split text by blank-line-separated paragraphs."""
    if not text:
        return []
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_into_spans(
    output: dict,
    *,
    granularity: str = "section",
    min_section_chars: int = 200,
) -> list[Span]:
    """Convert one agent output into a list of Spans."""
    findings = output.get("findings", "") or ""
    if not findings.strip():
        return []
    agent_id = output.get("agent_id", "?")
    source_round = int(output.get("_round", 0))

    def _make(heading: str | None, body: str) -> Span:
        return Span(
            id=str(uuid.uuid4()),
            source_agent=agent_id,
            source_round=source_round,
            heading=heading,
            text=body.strip(),
        )

    if granularity == "whole":
        return [_make(None, findings)]

    if granularity == "section":
        sections = _split_by_h2(findings)
        if not sections:
            return [_make(None, findings)]
        # Drop tiny sections; merge into prior section if too small
        out: list[Span] = []
        for heading, body in sections:
            if (len(body) < min_section_chars
                    and out
                    and out[-1].heading is None):
                out[-1] = Span(
                    id=out[-1].id,
                    source_agent=out[-1].source_agent,
                    source_round=out[-1].source_round,
                    heading=out[-1].heading,
                    text=out[-1].text + "\n\n" + body,
                )
            else:
                out.append(_make(heading, body))
        return out

    # granularity == "paragraph"
    paragraphs = _split_by_paragraph(findings)
    return [_make(None, p) for p in paragraphs]


# ---------------------------------------------------------------------------
# Bradley-Terry MLE via Minorization-Maximization
# ---------------------------------------------------------------------------


def bt_mle(
    items: list[str],
    wins: dict[tuple[str, str], float],
    *,
    regularization: float = 1e-3,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """Bradley-Terry skill estimation via Minorization-Maximization.

    `wins[(i, j)]` = number of times item i beat item j (ties contribute
    0.5 to each side). Returns normalized skill scores summing to N
    (so a uniform skill assignment gives 1.0 each).

    Reference: Hunter (2004) "MM algorithms for generalized Bradley-Terry
    models." Annals of Statistics, 32(1):384-406.
    """
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0]: 1.0}

    # Initialize uniform
    skill: dict[str, float] = {item: 1.0 for item in items}

    # Total wins per item
    w_i: dict[str, float] = {item: 0.0 for item in items}
    for (i, j), count in wins.items():
        w_i[i] = w_i.get(i, 0.0) + count
    # Comparison count between each pair
    n_ij: dict[tuple[str, str], float] = {}
    for (i, j), count in wins.items():
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        n_ij[key] = n_ij.get(key, 0.0) + count

    for iteration in range(max_iterations):
        new_skill: dict[str, float] = {}
        max_change = 0.0
        for item in items:
            denom = regularization
            for other in items:
                if other == item:
                    continue
                key = (item, other) if item < other else (other, item)
                count = n_ij.get(key, 0.0)
                if count > 0:
                    denom += count / max(skill[item] + skill[other], 1e-12)
            new_skill[item] = (w_i.get(item, 0.0) + regularization) / max(denom, 1e-12)
            max_change = max(max_change, abs(new_skill[item] - skill[item]))
        # Normalize so sum = N (uniform == 1.0 per item)
        total = sum(new_skill.values())
        if total > 0:
            for item in new_skill:
                new_skill[item] = new_skill[item] * n / total
        skill = new_skill
        if max_change < tolerance:
            break
    return skill


def _aggregate_pairwise_verdicts(
    span_ids: list[str],
    verdicts: list[PairwiseVerdict],
) -> dict[tuple[str, str], float]:
    """Convert PairwiseVerdict list into a wins[(winner, loser)] count dict.
    Ties contribute 0.5 to each side."""
    wins: dict[tuple[str, str], float] = {}
    for v in verdicts:
        a, b = v.span_a_id, v.span_b_id
        if v.winner == "a":
            wins[(a, b)] = wins.get((a, b), 0.0) + 1.0
        elif v.winner == "b":
            wins[(b, a)] = wins.get((b, a), 0.0) + 1.0
        else:  # tie
            wins[(a, b)] = wins.get((a, b), 0.0) + 0.5
            wins[(b, a)] = wins.get((b, a), 0.0) + 0.5
    return wins


# ---------------------------------------------------------------------------
# SelectorSynth
# ---------------------------------------------------------------------------


class SelectorSynth:
    """Selection-bottleneck synthesizer.

    Lifecycle:
        synth = SelectorSynth(cfg, pairwise_judge_fn=...)
        result = synth.synthesize(researcher_outputs, topic="...")
    """

    def __init__(
        self,
        cfg: SelectorConfig | None = None,
        *,
        pairwise_judge_fn: PairwiseJudgeFn | None = None,
    ):
        self.cfg = cfg or SelectorConfig()
        self.pairwise_judge_fn = pairwise_judge_fn or mock_pairwise_judge

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        researcher_outputs: list[dict],
        topic: str,
        *,
        guard_filter: Callable[[list[dict]], list[dict]] | None = None,
    ) -> SelectionResult:
        """Run selection over researcher outputs and return final document."""
        outputs = list(researcher_outputs)
        if guard_filter is not None:
            outputs = guard_filter(outputs)

        if not outputs:
            return SelectionResult(
                selected_span_ids=[], bt_scores={},
                sections_assembled=[], final_text="[no researcher inputs]",
                diagnostics={"reason": "empty_inputs"},
            )

        # 1. Split each output into spans
        all_spans: list[Span] = []
        for out in outputs:
            all_spans.extend(split_into_spans(
                out,
                granularity=self.cfg.granularity,
                min_section_chars=self.cfg.min_section_chars,
            ))

        if not all_spans:
            return SelectionResult(
                selected_span_ids=[], bt_scores={},
                sections_assembled=[], final_text="[no spans extracted]",
                diagnostics={"reason": "no_spans"},
            )

        # 2. Cluster spans (by heading for "section", whole-set for "whole")
        clusters = self._cluster_spans(all_spans)

        # 3. For each cluster, run pairwise selection + BT
        selected: list[Span] = []
        bt_scores_all: dict[str, float] = {}
        diagnostics = {
            "n_outputs": len(outputs),
            "n_spans": len(all_spans),
            "n_clusters": len(clusters),
            "judge_calls": 0,
        }

        for cluster_key, cluster in clusters.items():
            if len(cluster) == 1:
                selected.append(cluster[0])
                bt_scores_all[cluster[0].id] = 1.0
                continue
            verdicts = self._judge_cluster(cluster, topic)
            diagnostics["judge_calls"] += len(verdicts)
            wins = _aggregate_pairwise_verdicts(
                [s.id for s in cluster], verdicts,
            )
            bt = bt_mle(
                [s.id for s in cluster], wins,
                regularization=self.cfg.bt_regularization,
                max_iterations=self.cfg.bt_max_iterations,
                tolerance=self.cfg.bt_tolerance,
            )
            bt_scores_all.update(bt)
            winner_id = max(bt, key=lambda sid: (bt[sid], -len([
                s for s in cluster if s.id == sid
            ])))
            winner = next(s for s in cluster if s.id == winner_id)
            selected.append(winner)

        # Diagnostic: span-similarity collapse warning
        if len(set(s.text[:200] for s in all_spans)) == 1:
            diagnostics["warning"] = "homogeneous_collapse"

        # 4. Assemble
        text = self._assemble(selected, topic)

        return SelectionResult(
            selected_span_ids=[s.id for s in selected],
            bt_scores=bt_scores_all,
            sections_assembled=selected,
            final_text=text,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cluster_spans(self, spans: list[Span]) -> dict[str, list[Span]]:
        """Cluster spans by heading. Whole-mode → single cluster."""
        if self.cfg.granularity == "whole":
            return {"_root": list(spans)}
        clusters: dict[str, list[Span]] = {}
        for s in spans:
            key = s.heading or "_unsorted"
            clusters.setdefault(key, []).append(s)
        return clusters

    def _judge_cluster(
        self, cluster: list[Span], topic: str
    ) -> list[PairwiseVerdict]:
        """Pairwise comparisons across all pairs × n_judges."""
        verdicts: list[PairwiseVerdict] = []
        pairs = list(itertools.combinations(cluster, 2))
        for span_a, span_b in pairs:
            for j in range(self.cfg.n_judges):
                # Optionally swap order to control position bias
                if self.cfg.pairwise_random_order and (j + hash(span_a.id)) % 2:
                    a, b = span_b, span_a
                else:
                    a, b = span_a, span_b
                judge_id = f"judge_{j:02d}"
                seed = j
                v = self.pairwise_judge_fn(a, b, topic, judge_id, seed)
                # Normalize: store with original ordering (span_a, span_b)
                if (a, b) != (span_a, span_b):
                    if v.winner == "a":
                        v = PairwiseVerdict(judge_id=v.judge_id,
                                            span_a_id=span_a.id,
                                            span_b_id=span_b.id,
                                            winner="b",
                                            rationale=v.rationale)
                    elif v.winner == "b":
                        v = PairwiseVerdict(judge_id=v.judge_id,
                                            span_a_id=span_a.id,
                                            span_b_id=span_b.id,
                                            winner="a",
                                            rationale=v.rationale)
                    else:
                        v = PairwiseVerdict(judge_id=v.judge_id,
                                            span_a_id=span_a.id,
                                            span_b_id=span_b.id,
                                            winner="tie",
                                            rationale=v.rationale)
                verdicts.append(v)
        return verdicts

    def _assemble(self, ordered_spans: list[Span], topic: str) -> str:
        if not ordered_spans:
            return "[no spans selected]"
        if self.cfg.transition_strategy == "concat":
            return self._concat_assemble(ordered_spans)
        # smooth — placeholder; orchestrator integration calls Haiku
        return self._concat_assemble(ordered_spans)

    def _concat_assemble(self, spans: list[Span]) -> str:
        chunks: list[str] = []
        for s in spans:
            if s.heading:
                chunks.append(f"## {s.heading}\n\n{s.text}")
            else:
                chunks.append(s.text)
        return "\n\n".join(chunks)
