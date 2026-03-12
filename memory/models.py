"""Data models for G-Memory's three-tier hierarchical memory."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    RESOLVED = "resolved"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Tier 1: Interaction Graph — raw agent communication traces
# ---------------------------------------------------------------------------


@dataclass
class Utterance:
    """Node in the Interaction Graph — a single agent output."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    content: str = ""
    epoch: int = 0  # round number
    timestamp: float = field(default_factory=time.time)


@dataclass
class InteractionGraph:
    """Per-query record of all agent communication."""
    query_id: str = ""
    utterances: dict[str, Utterance] = field(default_factory=dict)  # id -> Utterance
    edges: list[tuple[str, str]] = field(default_factory=list)  # (source_id, target_id)

    def add_utterance(
        self,
        agent_id: str,
        content: str,
        epoch: int,
        inspired_by: list[str] | None = None,
    ) -> str:
        """Add an utterance and optionally connect it to prior utterances."""
        u = Utterance(agent_id=agent_id, content=content, epoch=epoch)
        self.utterances[u.id] = u
        for src_id in (inspired_by or []):
            if src_id in self.utterances:
                self.edges.append((src_id, u.id))
        return u.id


# ---------------------------------------------------------------------------
# Tier 2: Query Graph — task-level associations
# ---------------------------------------------------------------------------


@dataclass
class QueryNode:
    """Node in the Query Graph — a completed task."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str = ""
    status: TaskStatus = TaskStatus.RESOLVED
    embedding: list[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Tier 3: Insight Graph — distilled wisdom
# ---------------------------------------------------------------------------


@dataclass
class InsightNode:
    """Node in the Insight Graph — a generalizable lesson."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    supporting_queries: set[str] = field(default_factory=set)  # query IDs
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class InsightEdge:
    """Ternary hyper-edge: insight_a contextualizes insight_b via query."""
    source_insight_id: str = ""
    target_insight_id: str = ""
    via_query_id: str = ""


# ---------------------------------------------------------------------------
# Agent output (typed version of the JSON schema)
# ---------------------------------------------------------------------------


@dataclass
class AgentOutput:
    """Typed representation of an agent's structured output."""
    agent_id: str = ""
    role: str = ""
    findings: str = ""
    key_points: list[str] = field(default_factory=list)
    confidence: float = 0.0
    gaps_identified: list[str] = field(default_factory=list)
    quality_vote: str = "needs_work"
    quality_notes: str = ""
    dissent: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "AgentOutput":
        return cls(
            agent_id=d.get("agent_id", ""),
            role=d.get("role", ""),
            findings=d.get("findings", ""),
            key_points=d.get("key_points", []),
            confidence=d.get("confidence", 0.0),
            gaps_identified=d.get("gaps_identified", []),
            quality_vote=d.get("quality_vote", "needs_work"),
            quality_notes=d.get("quality_notes", ""),
            dissent=d.get("dissent", ""),
        )
