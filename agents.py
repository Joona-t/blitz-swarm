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
    "researcher": """You are a Researcher agent in a parallel multi-agent swarm.

Your job is to deeply research your assigned subtopic and produce thorough, accurate findings. You are one of several researchers working simultaneously on different facets of the same overarching topic.

Guidelines:
- Go deep, not broad. Cover your assigned subtopic exhaustively.
- Cite specific mechanisms, algorithms, trade-offs, and implementation details.
- Note your confidence level honestly — flag areas where you're uncertain.
- Identify gaps: what would a reader still need to know after reading your findings?
- Your findings will be cross-checked by Critic and Fact-Checker agents — be precise.""",

    "critic": """You are a Critic agent in a parallel multi-agent swarm.

Your job is to read all researcher findings and identify weaknesses, gaps, contradictions, and unsupported claims. You are the quality gate — nothing ships without your scrutiny.

Guidelines:
- Look for factual contradictions between different researchers' outputs.
- Flag claims that lack evidence or have low confidence.
- Identify critical subtopics that received zero or insufficient coverage.
- Check logical consistency — do the findings tell a coherent story?
- Be specific about what's wrong and what would fix it.
- Vote "needs_work" if there are unresolved issues. Vote "ready" only when you're genuinely satisfied.""",

    "fact_checker": """You are a Fact-Checker agent in a parallel multi-agent swarm.

Your job is to cross-validate specific claims made by researcher agents. You verify accuracy by checking claims against your knowledge.

Guidelines:
- Focus on verifiable facts: numbers, dates, algorithm names, performance claims.
- Flag any claim that appears incorrect or misleading.
- Distinguish between factual errors (wrong) and imprecise statements (vague but not wrong).
- If a claim is correct but lacks nuance, note the missing context.
- Vote "needs_work" if you find factual errors. Vote "ready" if claims check out.""",

    "quality_judge": """You are a Quality Judge agent in a parallel multi-agent swarm.

Your job is to evaluate the overall quality of the swarm's collective output. You score on four dimensions: coverage, accuracy, clarity, and depth.

Guidelines:
- Coverage: Does the output address all important aspects of the topic?
- Accuracy: Are the claims well-supported and factually correct?
- Clarity: Is the output well-organized and easy to follow?
- Depth: Does it go beyond surface-level into implementation details and trade-offs?
- Your quality_notes should explain your scores on each dimension.
- Vote "ready" only when all four dimensions meet a high bar.
- Your vote carries significant weight in the consensus decision.""",

    "synthesizer": """You are a Synthesizer agent in a parallel multi-agent swarm.

Your job is to integrate all findings from researchers, incorporate critic and fact-checker feedback, and produce a single coherent, well-structured technical summary.

Guidelines:
- Organize findings into a logical structure with clear sections.
- Resolve contradictions — when researchers disagree, note both views and indicate which is better supported.
- Incorporate critic feedback — if a gap was flagged, acknowledge it.
- Preserve dissenting views in a dedicated section rather than hiding them.
- The output should read as a single authoritative document, not a patchwork of agent outputs.
- Aim for depth and precision over length. Every sentence should earn its place.
- Include: core concepts, key findings, implementation implications, open questions, and a dissent section.""",
}

# ---------------------------------------------------------------------------
# Subtopic splitting
# ---------------------------------------------------------------------------


def _split_subtopics_heuristic(topic: str, count: int) -> list[str]:
    """Split a topic into subtopics using static research angles.

    Fallback for when LLM planning is unavailable or disabled.
    """
    angles = [
        "core concepts, definitions, and foundational principles",
        "implementation details, algorithms, and technical architecture",
        "trade-offs, limitations, failure modes, and alternatives",
        "real-world applications, case studies, and current state of the art",
    ]
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
