"""Genealogy-graph cascade defense — Xie et al. arXiv 2603.04474 (Mar 2026).

Adds a message-layer governance layer that:
  1. Decomposes each outbound agent message into atomic claims.
  2. Screens each claim against a persistent Lineage Graph
     (Green / Yellow / Red verdict).
  3. Optionally verifies Yellow claims at hub roles.
  4. On agent errors, marks descendant messages tainted and excludes
     them from subsequent rounds' context.

Three modes — selected via GuardConfig.mode:
  speed     skip Yellow verification, threshold 0.5
  balanced  verify Yellow at hubs (synthesizer/judge), threshold 0.7
  strict    verify all Yellow, block on Red, threshold 0.9
  off       identity passthrough (for ablation)

This module ships the LLM-free skeleton: lineage graph, taint
propagation, context filtering, mode policy. The decompose() and
screen() methods are pluggable hooks — the default implementation is a
heuristic (sentence-split + cosine-similarity stub) that lets tests run
without LLM calls. Wire to Claude Haiku via subprocess in orchestrator
integration (Phase 1 v0.2.0-alpha.2).
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

ClaimVerdict = Literal["green", "yellow", "red"]
GuardMode = Literal["speed", "balanced", "strict", "off"]

# Hub roles get tighter screening — outbound claims from these agents have
# the highest blast radius if tainted.
HUB_ROLES: frozenset[str] = frozenset({"synthesizer", "quality_judge"})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AtomicClaim:
    """One atomic SPO-style claim extracted from an agent's findings."""

    id: str
    text: str
    source_agent: str
    source_round: int
    parent_claim_ids: list[str] = field(default_factory=list)
    verdict: ClaimVerdict = "yellow"
    supports: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    verify_notes: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class GuardConfig:
    mode: GuardMode = "balanced"
    decompose_model: str = "haiku"          # used by orchestrator integration
    adjudicate_model: str = "sonnet"
    max_claims_per_msg: int = 12
    yellow_release_threshold: float = 0.7
    cache_ttl_s: int = 600
    max_lineage_size: int = 2000             # eviction cap
    block_on_red: bool = True

    @property
    def verify_yellow_at_hubs_only(self) -> bool:
        return self.mode == "balanced"

    @property
    def verify_all_yellow(self) -> bool:
        return self.mode == "strict"

    @property
    def is_active(self) -> bool:
        return self.mode != "off"


@dataclass
class TaintMarker:
    """Records why a (agent_id, round_n) tuple is tainted."""

    agent_id: str
    round_n: int
    reason: str
    marked_at: float = field(default_factory=time.time)


@dataclass
class GuardSummary:
    """Per-round stats for the metrics layer."""

    round_n: int
    green: int = 0
    yellow: int = 0
    red: int = 0
    blocked_outputs: int = 0
    tainted_descendants: int = 0
    decompose_calls: int = 0
    screen_calls: int = 0


# Type aliases for pluggable LLM hooks
DecomposeFn = Callable[[str], list[str]]      # findings -> atomic claim texts
ScreenFn = Callable[[AtomicClaim, list[AtomicClaim]], ClaimVerdict]


# ---------------------------------------------------------------------------
# Default LLM-free implementations (used by tests + orchestrator scaffold)
# ---------------------------------------------------------------------------


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def heuristic_decompose(findings: str, *, max_claims: int = 12) -> list[str]:
    """Simple sentence-split decomposition. No LLM call.

    The orchestrator integration replaces this with a Haiku call that
    extracts SPO-style atomic claims. Heuristic version is a strict
    sentence-splitter — adequate for tests and a sane fallback.
    """
    if not findings or not findings.strip():
        return []
    parts = _SENTENCE_SPLIT.split(findings.strip())
    out = [p.strip() for p in parts if p.strip()]
    return out[:max_claims]


def heuristic_screen(
    claim: AtomicClaim, neighbors: list[AtomicClaim]
) -> ClaimVerdict:
    """Default screen — no LLM call.

    Behavior:
      - No neighbors → "yellow" (novel, no anchor)
      - Any neighbor with verdict "green" whose text shares >= 60%
        word overlap → "green"
      - Any neighbor with verdict "green" whose text contains a clear
        negation token AND >= 40% word overlap → "red"
      - Else "yellow"

    This is a placeholder — real LLM-driven entailment/contradiction
    judgment lives in the Phase 1 orchestrator integration.
    """
    if not neighbors:
        return "yellow"
    claim_words = set(re.findall(r"[a-z0-9]+", claim.text.lower()))
    if not claim_words:
        return "yellow"

    neg_tokens = {"not", "no", "never", "false", "incorrect", "wrong"}
    has_neg = bool(neg_tokens & claim_words)

    for n in neighbors:
        if n.verdict != "green":
            continue
        n_words = set(re.findall(r"[a-z0-9]+", n.text.lower()))
        if not n_words:
            continue
        overlap = len(claim_words & n_words) / max(
            len(claim_words | n_words), 1
        )
        if has_neg and overlap >= 0.4:
            return "red"
        if not has_neg and overlap >= 0.6:
            return "green"
    return "yellow"


# ---------------------------------------------------------------------------
# CascadeGuard
# ---------------------------------------------------------------------------


class CascadeGuard:
    """Lineage-graph governance layer for the Blitz-Swarm blackboard.

    Public API:
        on_agent_output(output, round_n) -> dict
        on_agent_error(agent_id, round_n, reason) -> int
        filter_context(outputs)         -> list[dict]
        round_summary(round_n)          -> GuardSummary

    Internal state:
        lineage     dict[claim_id, AtomicClaim]
        tainted     set[(agent_id, round_n)]
        markers     list[TaintMarker]
    """

    def __init__(
        self,
        cfg: GuardConfig | None = None,
        *,
        decompose_fn: DecomposeFn | None = None,
        screen_fn: ScreenFn | None = None,
    ):
        self.cfg = cfg or GuardConfig()
        self.lineage: dict[str, AtomicClaim] = {}
        self.tainted: set[tuple[str, int]] = set()
        self.markers: list[TaintMarker] = []
        self._summaries: dict[int, GuardSummary] = {}
        self._decompose_fn = decompose_fn or (
            lambda findings: heuristic_decompose(
                findings, max_claims=self.cfg.max_claims_per_msg
            )
        )
        self._screen_fn = screen_fn or heuristic_screen

    # ------------------------------------------------------------------
    # Public hooks
    # ------------------------------------------------------------------

    def on_agent_output(self, output: dict, round_n: int) -> dict:
        """Process one agent output. Tag with claim_ids + verdict counts.

        Mutates `output` in place to add `_round`, `_claim_ids`,
        `_claim_counts`, and (when blocked) `_blocked` + `_block_reason`.
        Returns the (possibly modified) output for blackboard write.
        """
        if not self.cfg.is_active:
            output.setdefault("_round", round_n)
            return output

        summary = self._summary_for(round_n)
        output["_round"] = round_n
        agent_id = output.get("agent_id", "?")
        role = output.get("role", "?")

        # Hard taint: agent errored
        if output.get("_error") or output.get("_raw"):
            self.on_agent_error(agent_id, round_n, "agent_error")
            return output

        findings = output.get("findings", "") or ""
        claim_texts = self._decompose_fn(findings)
        summary.decompose_calls += 1

        parent_ids = output.get("parent_claim_ids") or []
        new_claims: list[AtomicClaim] = []
        red_claims: list[AtomicClaim] = []
        counts = {"green": 0, "yellow": 0, "red": 0}

        for text in claim_texts:
            claim = AtomicClaim(
                id=str(uuid.uuid4()),
                text=text,
                source_agent=agent_id,
                source_round=round_n,
                parent_claim_ids=list(parent_ids),
            )
            anchors = self._anchors_for(claim, role=role)
            verdict = self._screen_fn(claim, anchors)
            summary.screen_calls += 1

            # Mode policy: yellow may be promoted via verify_yellow
            if verdict == "yellow" and self._should_verify(role):
                # Stub — real implementation calls adjudicate_model.
                # Heuristic version leaves Yellow as Yellow.
                verdict = "yellow"

            claim.verdict = verdict
            self.lineage[claim.id] = claim
            new_claims.append(claim)
            counts[verdict] += 1
            if verdict == "red":
                red_claims.append(claim)

        # Update per-round summary
        summary.green += counts["green"]
        summary.yellow += counts["yellow"]
        summary.red += counts["red"]

        output["_claim_ids"] = [c.id for c in new_claims]
        output["_claim_counts"] = counts

        # Block on Red claims (balanced/strict only; speed never blocks)
        if (
            red_claims
            and self.cfg.mode in ("balanced", "strict")
            and self.cfg.block_on_red
        ):
            self._mark_tainted(agent_id, round_n, "red_claim")
            output["_blocked"] = True
            output["_block_reason"] = [c.text[:200] for c in red_claims][:3]
            summary.blocked_outputs += 1

        # Eviction
        if len(self.lineage) > self.cfg.max_lineage_size:
            self._evict_oldest_non_green()

        return output

    def on_agent_error(self, agent_id: str, round_n: int, reason: str) -> int:
        """Mark this agent (and its descendants) tainted.

        Returns the count of descendant claims tainted.
        """
        self._mark_tainted(agent_id, round_n, reason)
        descendants = self._descendant_claim_ids(agent_id, round_n)
        for cid in descendants:
            claim = self.lineage.get(cid)
            if claim is not None:
                # Demote Green/Yellow descendants to Yellow ('tainted')
                # Do not change Red (already worst case)
                if claim.verdict != "red":
                    claim.verdict = "yellow"
                    claim.verify_notes = (
                        f"tainted via {agent_id}@r{round_n}: {reason}"
                    )
                self._mark_tainted(claim.source_agent, claim.source_round,
                                    f"descendant_of:{agent_id}")
        summary = self._summary_for(round_n)
        summary.tainted_descendants += len(descendants)
        return len(descendants)

    def filter_context(self, outputs: list[dict]) -> list[dict]:
        """Strip blocked or tainted outputs before context-build."""
        if not self.cfg.is_active:
            return list(outputs)
        out: list[dict] = []
        for o in outputs:
            if o.get("_blocked"):
                continue
            key = (o.get("agent_id", "?"), o.get("_round", -1))
            if key in self.tainted:
                continue
            out.append(o)
        return out

    def round_summary(self, round_n: int) -> GuardSummary:
        return self._summaries.get(round_n, GuardSummary(round_n=round_n))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _summary_for(self, round_n: int) -> GuardSummary:
        if round_n not in self._summaries:
            self._summaries[round_n] = GuardSummary(round_n=round_n)
        return self._summaries[round_n]

    def _mark_tainted(self, agent_id: str, round_n: int, reason: str) -> None:
        key = (agent_id, round_n)
        if key not in self.tainted:
            self.tainted.add(key)
            self.markers.append(TaintMarker(agent_id, round_n, reason))

    def _descendant_claim_ids(self, agent_id: str, round_n: int) -> list[str]:
        """Find all claims whose lineage transitively touches this agent's
        claims at this round."""
        # First, claims authored by the errored agent at this round
        seed_ids = {
            cid for cid, c in self.lineage.items()
            if c.source_agent == agent_id and c.source_round == round_n
        }
        if not seed_ids:
            return []
        # Reverse-edge index — for each claim, which children cite it as parent
        children: dict[str, list[str]] = {}
        for cid, claim in self.lineage.items():
            for pid in claim.parent_claim_ids:
                children.setdefault(pid, []).append(cid)
        # BFS over descendants
        seen: set[str] = set(seed_ids)
        frontier = list(seed_ids)
        while frontier:
            nxt: list[str] = []
            for cid in frontier:
                for child in children.get(cid, []):
                    if child not in seen:
                        seen.add(child)
                        nxt.append(child)
            frontier = nxt
        seen.difference_update(seed_ids)
        return list(seen)

    def _anchors_for(self, claim: AtomicClaim, *, role: str) -> list[AtomicClaim]:
        """Return prior-round Green claims to use as screening anchors.

        Lateral (same-round) claims do NOT count as anchors — that
        prevents echo-chamber promotion.
        """
        return [
            c for c in self.lineage.values()
            if c.verdict == "green" and c.source_round < claim.source_round
        ]

    def _should_verify(self, role: str) -> bool:
        """Decide whether to run verify_yellow for an agent in this role."""
        if self.cfg.verify_all_yellow:
            return True
        if self.cfg.verify_yellow_at_hubs_only and role in HUB_ROLES:
            return True
        return False

    def _evict_oldest_non_green(self) -> int:
        """Drop oldest Yellow/Red claims to stay under max_lineage_size."""
        candidates = [
            c for c in self.lineage.values() if c.verdict != "green"
        ]
        candidates.sort(key=lambda c: c.created_at)
        target = max(0, len(self.lineage) - self.cfg.max_lineage_size)
        evicted = 0
        for c in candidates[:target]:
            self.lineage.pop(c.id, None)
            evicted += 1
        return evicted
