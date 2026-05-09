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
from typing import Iterable


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
