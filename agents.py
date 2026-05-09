"""Agent definitions, role prompts, and swarm planning for Blitz-Swarm."""

import json
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Agent output schema — passed to claude --json-schema for structured output
# ---------------------------------------------------------------------------

AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "string",
            "description": "Your detailed findings, analysis, or synthesis in markdown.",
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Bullet-point list of the most important takeaways.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Your confidence in the accuracy of your output (0.0–1.0).",
        },
        "gaps_identified": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Areas that need more research or have insufficient coverage.",
        },
        "quality_vote": {
            "type": "string",
            "enum": ["ready", "needs_work"],
            "description": "Vote on whether the collective output is ready for finalization.",
        },
        "quality_notes": {
            "type": "string",
            "description": "Explanation for your quality vote.",
        },
        "dissent": {
            "type": "string",
            "description": "Any disagreements with other agents' findings or the emerging consensus.",
        },
        "coverage_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
            "description": "Score 0-10: Does the output address all important aspects? (quality_judge only)",
        },
        "accuracy_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
            "description": "Score 0-10: Are claims well-supported and factually correct? (quality_judge only)",
        },
        "clarity_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
            "description": "Score 0-10: Is the output well-organized and easy to follow? (quality_judge only)",
        },
        "depth_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
            "description": "Score 0-10: Does it go beyond surface-level into implementation details? (quality_judge only)",
        },
    },
    "required": ["findings", "key_points", "confidence", "quality_vote"],
}

AGENT_OUTPUT_SCHEMA_JSON = json.dumps(AGENT_OUTPUT_SCHEMA)

# ---------------------------------------------------------------------------
# Agent dataclass
# ---------------------------------------------------------------------------


@dataclass
class BlitzAgent:
    id: str
    role: str
    subtopic: str
    system_prompt: str
    model: str = "sonnet"
    max_iterations: int = 3


# ---------------------------------------------------------------------------
# Role prompt templates
# ---------------------------------------------------------------------------

ROLE_PROMPTS = {
    "researcher": """You are a Researcher agent in a parallel crypto/quant trading research swarm.

Your job is to deeply research your assigned subtopic and produce thorough, evidence-backed findings for improving an algorithmic crypto trading system.

Guidelines:
- Go deep, not broad. Cover your assigned subtopic exhaustively.
- Cite specific papers with authors and years (e.g., Moskowitz 2012, Gu/Kelly/Xiu 2020, DeMiguel 2009).
- Every performance claim MUST include measured numbers: Sharpe ratio, R², drawdown, win rate.
- Distinguish between signals with out-of-sample evidence vs practitioner folklore with zero rigorous backtests.
- Flag backtesting biases: look-ahead bias, survivorship bias, data snooping, overfitting to specific regimes.
- Account for transaction costs (10+ bps for retail crypto) — an alpha source that doesn't survive costs is not alpha.
- Note which market regime each finding applies to (bull, bear, sideways, crisis).
- Note your confidence level honestly — flag areas where you're uncertain.
- Identify gaps: what would a quant trader still need to know after reading your findings?
- Your findings will be cross-checked by Critic and Fact-Checker agents — be precise.""",

    "critic": """You are a Critic agent in a parallel crypto/quant trading research swarm.

Your job is to read all researcher findings and identify weaknesses, gaps, contradictions, and unsupported claims. You are the quality gate — no trading strategy ships without your scrutiny.

Guidelines:
- Flag performance claims without out-of-sample validation or proper walk-forward testing.
- Check for overfitting indicators: too many parameters, cherry-picked time periods, no deflated Sharpe analysis.
- Verify transaction cost assumptions — does the claimed alpha survive 10-20 bps round-trip costs?
- Flag regime-dependent claims that only work in bull or bear markets.
- Demand out-of-sample evidence — in-sample backtests are near-worthless for strategy validation.
- Verify that cited papers actually support the claimed conclusion (not just tangentially related).
- Check logical consistency — do the findings tell a coherent story?
- Be specific about what's wrong and what would fix it.
- Vote "needs_work" if there are unresolved issues. Vote "ready" only when you're genuinely satisfied.""",

    "fact_checker": """You are a Fact-Checker agent in a parallel crypto/quant trading research swarm.

Your job is to cross-validate specific quantitative claims made by researcher agents.

Guidelines:
- Focus on verifiable facts: Sharpe ratios, R² values, paper citations, algorithm specifications.
- Verify that cited papers exist and that the claimed results match what the papers actually found.
- Cross-reference claims against established results: Moskowitz 2012 (TSMOM), McLean & Pontiff 2016 (58% post-publication decay), DeMiguel 2009 (1/N dominance).
- Flag any claim that appears incorrect or misleading — especially inflated backtest results.
- Check that mathematical formulas are correct (Kelly criterion, Sharpe calculation, vol estimators).
- Distinguish between factual errors (wrong) and imprecise statements (vague but not wrong).
- Vote "needs_work" if you find factual errors. Vote "ready" if claims check out.""",

    "quality_judge": """You are a Quality Judge agent in a parallel crypto/quant trading research swarm.

Your job is to evaluate the overall quality of the swarm's collective output for use in a real trading system. You MUST provide numeric scores (0-10) on four dimensions.

Scoring rubric:
- coverage_score (0-10): Does the output address the question with empirical evidence, not just theory? 0=no evidence cited, 5=some papers but gaps, 8=solid evidence base, 10=exhaustive with primary sources
- accuracy_score (0-10): Are quantitative claims correct and properly contextualized? 0=wrong numbers, 5=mostly right but missing caveats, 8=accurate with proper caveats, 10=verified against primary sources
- clarity_score (0-10): Could a Python developer implement these findings in a trading system? 0=too vague, 5=general direction clear, 8=specific parameters given, 10=pseudocode-ready
- depth_score (0-10): Does it account for realistic trading conditions (costs, slippage, regime changes)? 0=ignores costs, 5=mentions costs, 8=models costs explicitly, 10=full regime-conditional analysis

Guidelines:
- You MUST include all four numeric score fields in your JSON output.
- Your quality_notes should explain your reasoning for each score.
- Vote "ready" only when all four scores are >= 7.
- Vote "needs_work" and explain what would raise the lowest scores.
- Reject research that doesn't survive transaction cost analysis or lacks out-of-sample evidence.""",

    "synthesizer": """You are a Synthesizer agent in a parallel crypto/quant trading research swarm.

Your job is to integrate all findings into actionable recommendations for a 5-agent crypto trading system with these components: momentum agent, mean reversion agent, volatility regime agent, cross-asset agent, and ML ensemble agent (LightGBM).

Guidelines:
- Organize findings by which agent they apply to (momentum, mean_reversion, vol_regime, cross_asset, ml_ensemble, aggregator, risk_manager).
- For each finding, include: the evidence source, the specific parameter or logic change, and the expected impact.
- Resolve contradictions — when researchers disagree, note both views and indicate which has stronger out-of-sample evidence.
- Incorporate critic feedback — if a gap was flagged, acknowledge it.
- Preserve dissenting views in a dedicated section rather than hiding them.
- Include concrete parameter recommendations where the evidence supports them (e.g., "EMA window 10/30 outperforms 5/21 in crypto per [paper]").
- The output should be directly actionable by a developer modifying Python trading code.
- Include: key findings, per-agent recommendations, aggregator/risk changes, implementation priority, open questions, and a dissent section.""",
}

# ---------------------------------------------------------------------------
# Subtopic splitting
# ---------------------------------------------------------------------------


CRYPTO_TRADING_ANGLES = [
    "signal generation: momentum timing, mean reversion thresholds, adaptive parameters, out-of-sample evidence",
    "risk & position sizing: Kelly fraction optimization, volatility targeting, drawdown control, kill switch design",
    "market microstructure: funding rates, liquidation cascades, exchange-specific edges, slippage modeling",
    "ML for alpha: feature engineering for crypto returns, walk-forward validation, regime-conditional models, overfitting prevention",
    "regime detection: bull/bear/sideways identification, correlation regime shifts, vol clustering, adaptive agent weighting",
]

GENERIC_ANGLES = [
    "core concepts, definitions, and foundational principles",
    "implementation details, algorithms, and technical architecture",
    "trade-offs, limitations, failure modes, and alternatives",
    "real-world applications, case studies, and current state of the art",
]

# Keywords that trigger crypto-specialized subtopic splitting
_CRYPTO_KEYWORDS = {"crypto", "bitcoin", "btc", "trading", "momentum", "sharpe", "backtest", "alpha", "hedge", "quant", "funding rate", "lightgbm", "mean reversion"}


def _split_subtopics_heuristic(topic: str, count: int) -> list[str]:
    """Split a topic into subtopics using static research angles.

    Uses crypto-specialized angles when the topic is trading-related,
    falls back to generic angles otherwise.
    """
    topic_lower = topic.lower()
    if any(kw in topic_lower for kw in _CRYPTO_KEYWORDS):
        angles = CRYPTO_TRADING_ANGLES
    else:
        angles = GENERIC_ANGLES

    subtopics = []
    for i in range(count):
        angle = angles[i % len(angles)]
        subtopics.append(f"{topic} — focusing on {angle}")
    return subtopics


def _split_subtopics_llm(topic: str, count: int) -> list[str]:
    """Split a topic into subtopics using an LLM call.

    Invokes claude -p to analyze the topic and generate targeted subtopics.
    Falls back to heuristic if the LLM call fails.
    """
    import subprocess

    schema = json.dumps({
        "type": "object",
        "properties": {
            "subtopics": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"Exactly {count} specific, non-overlapping subtopics.",
            },
        },
        "required": ["subtopics"],
    })

    prompt = (
        f"Analyze this research topic and split it into exactly {count} specific, "
        f"non-overlapping subtopics that together provide comprehensive coverage.\n\n"
        f"Topic: {topic}\n\n"
        f"Each subtopic should be a focused research angle that a single researcher "
        f"can deeply investigate. Make them specific to this topic, not generic."
    )

    try:
        result = subprocess.run(
            [
                "claude", "-p", prompt,
                "--system-prompt", "You are a research planning assistant. Return JSON only.",
                "--output-format", "json",
                "--model", "haiku",
                "--dangerously-skip-permissions",
            ],
            capture_output=True, text=True, timeout=30,
        )

        if result.returncode == 0:
            data = json.loads(result.stdout.strip())
            subtopics = data.get("subtopics", [])
            if len(subtopics) >= count:
                return [f"{topic} — focusing on {st}" for st in subtopics[:count]]
    except Exception:
        pass

    return _split_subtopics_heuristic(topic, count)


# ---------------------------------------------------------------------------
# Agent planning
# ---------------------------------------------------------------------------

# Model overrides for specific roles (others use default "sonnet")
ROLE_MODEL_OVERRIDES = {
    "quality_judge": "sonnet",
    "synthesizer": "sonnet",
}

PLANNING_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "researcher_count": {
            "type": "integer", "minimum": 2, "maximum": 6,
            "description": "Number of researchers to spawn.",
        },
        "critic_count": {
            "type": "integer", "minimum": 1, "maximum": 3,
            "description": "Number of critics to spawn.",
        },
        "needs_fact_checker": {
            "type": "boolean",
            "description": "Whether a dedicated fact-checker is needed.",
        },
        "subtopics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific subtopics for each researcher.",
        },
    },
    "required": ["researcher_count", "critic_count", "needs_fact_checker", "subtopics"],
})


def plan_agents(topic: str, use_llm: bool = True) -> list[BlitzAgent]:
    """Plan which agents to spawn for a given topic.

    When use_llm=True, invokes an LLM to analyze the topic and determine
    optimal agent count and subtopic assignments. Falls back to heuristic.
    """
    plan = None

    if use_llm:
        plan = _llm_plan(topic)

    if plan is None:
        plan = {
            "researcher_count": 2,
            "critic_count": 1,
            "needs_fact_checker": True,
            "subtopics": None,
        }

    researcher_count = plan["researcher_count"]
    critic_count = plan["critic_count"]
    needs_fc = plan["needs_fact_checker"]

    # Get subtopics
    subtopics = plan.get("subtopics")
    if subtopics and len(subtopics) >= researcher_count:
        subtopics = [f"{topic} — focusing on {st}" for st in subtopics[:researcher_count]]
    else:
        subtopics = _split_subtopics_heuristic(topic, researcher_count)

    agents = []

    # Spawn researchers
    for i, subtopic in enumerate(subtopics):
        agents.append(BlitzAgent(
            id=f"researcher_{i:02d}",
            role="researcher",
            subtopic=subtopic,
            system_prompt=ROLE_PROMPTS["researcher"],
            model=ROLE_MODEL_OVERRIDES.get("researcher", "sonnet"),
        ))

    # Spawn critics
    for i in range(critic_count):
        suffix = f"_{i:02d}" if critic_count > 1 else ""
        agents.append(BlitzAgent(
            id=f"critic{suffix}",
            role="critic",
            subtopic=topic,
            system_prompt=ROLE_PROMPTS["critic"],
            model="sonnet",
        ))

    # Spawn fact-checker
    if needs_fc:
        agents.append(BlitzAgent(
            id="fact_checker",
            role="fact_checker",
            subtopic=topic,
            system_prompt=ROLE_PROMPTS["fact_checker"],
            model="sonnet",
        ))

    # Always: 1 quality judge
    agents.append(BlitzAgent(
        id="quality_judge",
        role="quality_judge",
        subtopic=topic,
        system_prompt=ROLE_PROMPTS["quality_judge"],
        model=ROLE_MODEL_OVERRIDES.get("quality_judge", "sonnet"),
    ))

    # Always: 1 synthesizer
    agents.append(BlitzAgent(
        id="synthesizer",
        role="synthesizer",
        subtopic=topic,
        system_prompt=ROLE_PROMPTS["synthesizer"],
        model=ROLE_MODEL_OVERRIDES.get("synthesizer", "sonnet"),
    ))

    return agents


def _llm_plan(topic: str) -> dict | None:
    """Use an LLM to determine optimal swarm composition for a topic."""
    import subprocess

    prompt = (
        f"Analyze this research topic and determine the optimal agent swarm composition.\n\n"
        f"Topic: {topic}\n\n"
        f"Consider:\n"
        f"- How broad is this topic? (narrow=2 researchers, broad=4-6)\n"
        f"- Does it involve claims that need fact-checking? (empirical/technical=yes)\n"
        f"- How many critics are needed? (controversial=2, straightforward=1)\n"
        f"- What specific subtopics should each researcher focus on?"
    )

    try:
        result = subprocess.run(
            [
                "claude", "-p", prompt,
                "--system-prompt", "You are a research planning assistant. Return JSON only.",
                "--output-format", "json",
                "--model", "haiku",
                "--dangerously-skip-permissions",
            ],
            capture_output=True, text=True, timeout=30,
        )

        if result.returncode == 0:
            data = json.loads(result.stdout.strip())
            # Validate required fields
            if all(k in data for k in ("researcher_count", "critic_count", "needs_fact_checker")):
                data["researcher_count"] = max(2, min(6, int(data["researcher_count"])))
                data["critic_count"] = max(1, min(3, int(data["critic_count"])))
                return data

    except Exception:
        pass

    return None
