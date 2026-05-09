"""AFlow-style architectural search over swarm graphs.

Frames swarm-config optimization as MCTS over `SwarmGraph` mutations.
Anchor: Liu et al. arXiv 2410.10762 (ICLR 2025 oral) "AFlow: Automating
Agentic Workflow Generation."

Six operators ship in v0.2:
    AddResearcher, RemoveCritic, AddDebateRound, SwapSynthForSelector,
    AddJudgeEnsemble, IncreaseRounds, FuseRoles, SwapCLI

This module implements the LLM-free skeleton: graph IR, operator
registry, MCTS with UCB1, fingerprint dedup. The mutator (the LLM
that picks which operator to apply next) is pluggable; default is a
deterministic round-robin so tests run without LLM cost.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Iterable


class RoleKind(Enum):
    RESEARCHER = "researcher"
    CRITIC_FACTUAL = "critic_factual"
    CRITIC_LOGICAL = "critic_logical"
    CRITIC_COUNTERFACTUAL = "critic_counterfactual"
    SYNTHESIZER = "synthesizer"
    SELECTOR_SYNTH = "selector_synth"
    DEBATER = "debater"
    JUDGE = "judge"
    JUDGE_ENSEMBLE = "judge_ensemble"


@dataclass(frozen=True)
class Node:
    id: str
    kind: RoleKind
    cli: str = "claude:sonnet"


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    payload: str = "findings"


@dataclass
class SwarmGraph:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    rounds: int = 4
    consensus_strategy: str = "judge_select"

    def fingerprint(self) -> str:
        canonical = {
            "nodes": sorted([(n.id, n.kind.value, n.cli) for n in self.nodes]),
            "edges": sorted([(e.src, e.dst, e.payload) for e in self.edges]),
            "rounds": self.rounds,
            "consensus_strategy": self.consensus_strategy,
        }
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True).encode()
        ).hexdigest()[:16]

    def validate(self) -> list[str]:
        """Return a list of issues; empty list means valid."""
        issues: list[str] = []
        node_ids = {n.id for n in self.nodes}
        if len(node_ids) != len(self.nodes):
            issues.append("duplicate node id")
        for e in self.edges:
            if e.src not in node_ids:
                issues.append(f"edge references unknown source {e.src!r}")
            if e.dst not in node_ids:
                issues.append(f"edge references unknown target {e.dst!r}")
        if self.rounds < 1 or self.rounds > 9:
            issues.append(f"rounds {self.rounds} out of bounds [1, 9]")
        return issues

    def count_role(self, kind: RoleKind) -> int:
        return sum(1 for n in self.nodes if n.kind == kind)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class Operator:
    name: str = "noop"

    def applicable(self, g: SwarmGraph) -> bool:
        return True

    def apply(self, g: SwarmGraph, ctx: dict | None = None) -> SwarmGraph:
        return g


class AddResearcher(Operator):
    name = "AddResearcher"

    def applicable(self, g: SwarmGraph) -> bool:
        return g.count_role(RoleKind.RESEARCHER) < 6

    def apply(self, g, ctx=None):
        new_id = f"researcher_{g.count_role(RoleKind.RESEARCHER):02d}_added"
        new_node = Node(id=new_id, kind=RoleKind.RESEARCHER)
        return replace(g, nodes=(*g.nodes, new_node))


class RemoveCritic(Operator):
    name = "RemoveCritic"

    def applicable(self, g):
        return any(n.kind.value.startswith("critic_") for n in g.nodes)

    def apply(self, g, ctx=None):
        critics = [n for n in g.nodes if n.kind.value.startswith("critic_")]
        if not critics:
            return g
        # Drop the last-added critic
        target = critics[-1]
        new_nodes = tuple(n for n in g.nodes if n.id != target.id)
        new_edges = tuple(e for e in g.edges
                          if e.src != target.id and e.dst != target.id)
        return replace(g, nodes=new_nodes, edges=new_edges)


class AddDebateRound(Operator):
    name = "AddDebateRound"

    def applicable(self, g):
        return g.rounds < 9

    def apply(self, g, ctx=None):
        return replace(g, rounds=g.rounds + 1)


class SwapSynthForSelector(Operator):
    name = "SwapSynthForSelector"

    def applicable(self, g):
        return any(n.kind == RoleKind.SYNTHESIZER for n in g.nodes)

    def apply(self, g, ctx=None):
        new_nodes = tuple(
            replace(n, kind=RoleKind.SELECTOR_SYNTH)
            if n.kind == RoleKind.SYNTHESIZER else n
            for n in g.nodes
        )
        return replace(g, nodes=new_nodes, consensus_strategy="judge_select")


class AddJudgeEnsemble(Operator):
    name = "AddJudgeEnsemble"

    def applicable(self, g):
        return any(n.kind == RoleKind.JUDGE for n in g.nodes)

    def apply(self, g, ctx=None):
        new_nodes = tuple(
            replace(n, kind=RoleKind.JUDGE_ENSEMBLE)
            if n.kind == RoleKind.JUDGE else n
            for n in g.nodes
        )
        return replace(g, nodes=new_nodes)


class IncreaseRounds(Operator):
    name = "IncreaseRounds"

    def applicable(self, g):
        return g.rounds < 9

    def apply(self, g, ctx=None):
        return replace(g, rounds=g.rounds + 1)


OPERATOR_REGISTRY: tuple[type[Operator], ...] = (
    AddResearcher, RemoveCritic, AddDebateRound,
    SwapSynthForSelector, AddJudgeEnsemble, IncreaseRounds,
)


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------


@dataclass
class MCTSNode:
    graph: SwarmGraph
    parent: "MCTSNode | None"
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_score: float = 0.0
    untried_ops: list[type[Operator]] = field(default_factory=list)
    op_applied: str = ""

    def ucb(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.total_score / max(self.visits, 1)
        exploit = self.total_score / self.visits
        explore = c * math.sqrt(math.log(max(self.parent.visits, 1)) / self.visits)
        return exploit + explore


# Pluggable evaluator hook
EvaluatorFn = Callable[[SwarmGraph], float]


def _select(root: MCTSNode) -> MCTSNode:
    node = root
    while node.children and not node.untried_ops:
        node = max(node.children, key=lambda c: c.ucb())
    return node


def _expand(
    leaf: MCTSNode,
    *,
    max_attempts: int = 5,
) -> tuple[MCTSNode, str] | None:
    """Try to apply one untried operator. Returns (new_node, op_name) or None."""
    if not leaf.untried_ops:
        return None
    for _ in range(min(max_attempts, len(leaf.untried_ops))):
        op_cls = leaf.untried_ops.pop(0)
        op = op_cls()
        if not op.applicable(leaf.graph):
            continue
        new_graph = op.apply(leaf.graph)
        if new_graph.validate():
            continue
        new_node = MCTSNode(
            graph=new_graph,
            parent=leaf,
            untried_ops=list(OPERATOR_REGISTRY),
            op_applied=op.name,
        )
        leaf.children.append(new_node)
        return new_node, op.name
    return None


def _backprop(node: MCTSNode, score: float) -> None:
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.total_score += score
        cur = cur.parent


def aflow_search(
    seed: SwarmGraph,
    *,
    evaluator: EvaluatorFn,
    max_iters: int = 50,
    seed_score: float | None = None,
    rng_seed: int = 42,
) -> dict:
    """Run MCTS over swarm graphs. Returns dict with frontier + history."""
    rng = random.Random(rng_seed)
    root = MCTSNode(
        graph=seed, parent=None,
        untried_ops=list(OPERATOR_REGISTRY),
    )
    if seed_score is None:
        seed_score = evaluator(seed)
    _backprop(root, seed_score)

    history: list[dict] = []
    seen_fingerprints: set[str] = {seed.fingerprint()}

    for i in range(max_iters):
        leaf = _select(root)
        result = _expand(leaf)
        if result is None:
            # No more operators to try at this leaf; pick a different leaf
            # Use rng for diversity
            if not root.children:
                break
            leaf = rng.choice(root.children)
            continue
        new_node, op_name = result
        fp = new_node.graph.fingerprint()
        if fp in seen_fingerprints:
            history.append({"i": i, "fp": fp, "score": None,
                            "op": op_name, "skipped": "duplicate"})
            continue
        seen_fingerprints.add(fp)
        score = evaluator(new_node.graph)
        _backprop(new_node, score)
        history.append({"i": i, "fp": fp, "score": score,
                        "op": op_name, "skipped": None})

    # Rank frontier: top-K by mean score
    all_nodes: list[MCTSNode] = []

    def _walk(n: MCTSNode) -> None:
        all_nodes.append(n)
        for c in n.children:
            _walk(c)
    _walk(root)
    ranked = sorted(
        [n for n in all_nodes if n.visits > 0],
        key=lambda n: n.total_score / max(n.visits, 1),
        reverse=True,
    )
    return {
        "iterations": len(history),
        "history": history,
        "frontier": [
            {"fingerprint": n.graph.fingerprint(),
             "score": n.total_score / max(n.visits, 1),
             "op": n.op_applied}
            for n in ranked[:5]
        ],
        "best": ranked[0].graph if ranked else seed,
    }


# ---------------------------------------------------------------------------
# Default seed graph
# ---------------------------------------------------------------------------


def default_seed_graph(*, n_researchers: int = 2, with_judge: bool = True) -> SwarmGraph:
    nodes = [Node(id=f"r{i:02d}", kind=RoleKind.RESEARCHER)
             for i in range(n_researchers)]
    nodes.append(Node(id="critic_factual", kind=RoleKind.CRITIC_FACTUAL))
    if with_judge:
        nodes.append(Node(id="judge", kind=RoleKind.JUDGE))
    nodes.append(Node(id="synthesizer", kind=RoleKind.SYNTHESIZER))
    edges = []
    for n in nodes:
        if n.kind == RoleKind.RESEARCHER:
            edges.append(Edge(src=n.id, dst="critic_factual"))
            edges.append(Edge(src=n.id, dst="synthesizer"))
    if with_judge:
        edges.append(Edge(src="critic_factual", dst="judge"))
        edges.append(Edge(src="judge", dst="synthesizer"))
    return SwarmGraph(nodes=tuple(nodes), edges=tuple(edges))
