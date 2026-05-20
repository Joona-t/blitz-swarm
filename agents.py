"""Agent definitions, role prompts, and swarm planning for Blitz-Swarm.

v0.2: prompts are loaded from `prompts/<domain>/<role>[_<persona>].md`
via `prompts.loader.PromptLoader`. The legacy inline ROLE_PROMPTS dict
below is kept as a backward-compat fallback when no prompt files are on
disk; v0.3 will drop it.

Domain selection comes from `blitz.toml [swarm] domain = "..."`. Persona
critics activate when `[swarm] persona_critics = true` (MAR mechanism,
arXiv 2512.20845).
"""

import json
from dataclasses import dataclass, field

from prompts.loader import (
    PromptLoader,
    PromptLoaderError,
    PromptSet,
    assign_personas,
)

# ---------------------------------------------------------------------------
# Agent output schema — passed to local backends for structured output
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
    # v0.2: persona-typed critics (MAR mechanism). None for non-critic roles
    # or single-critic mode. Populated by `plan_agents` when
    # blitz.toml `[swarm] persona_critics = true`.
    persona: str | None = None
    # Path to the prompt file used (for trace metadata in metrics.jsonl).
    prompt_path: str | None = None
    # SHA-256 of the prompt file content (for cross-version diffing).
    prompt_sha256: str | None = None


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


def _split_subtopics_llm(
    topic: str,
    count: int,
    *,
    backend_id: str | None = None,
    sandbox: str | None = None,
) -> list[str]:
    """Split a topic into subtopics using an LLM call.

    Invokes the configured local backend to generate targeted subtopics.
    Falls back to heuristic if the LLM call fails.
    """
    schema = {
        "type": "object",
        "properties": {
            "subtopics": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"Exactly {count} specific, non-overlapping subtopics.",
            },
        },
        "required": ["subtopics"],
    }

    prompt = (
        f"Analyze this research topic and split it into exactly {count} specific, "
        f"non-overlapping subtopics that together provide comprehensive coverage.\n\n"
        f"Topic: {topic}\n\n"
        f"Each subtopic should be a focused research angle that a single researcher "
        f"can deeply investigate. Make them specific to this topic, not generic."
    )

    data = _planning_backend_json(
        prompt,
        "You are a research planning assistant. Return JSON only.",
        schema,
        model_hint="cheap",
        timeout_s=30,
        backend_id=backend_id,
        sandbox=sandbox,
    )
    if data:
        subtopics = data.get("subtopics", [])
        if len(subtopics) >= count:
            return [f"{topic} — focusing on {st}" for st in subtopics[:count]]

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


def _resolve_role_prompt(
    loader: PromptLoader,
    role: str,
    *,
    persona: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Load a role prompt via PromptLoader; fall back to ROLE_PROMPTS dict on miss.

    Returns: (system_prompt, prompt_path_str_or_None, sha256_or_None).
    Fallback path returns (None, None) for path/sha so the metrics layer
    can distinguish disk-loaded prompts from inline-fallback prompts.
    """
    try:
        prompt_set: PromptSet = loader.load(role, persona=persona)
        return prompt_set.system, str(prompt_set.template_path), prompt_set.sha256
    except PromptLoaderError:
        if role in ROLE_PROMPTS:
            return ROLE_PROMPTS[role], None, None
        raise


def plan_agents(
    topic: str,
    use_llm: bool = True,
    *,
    domain: str | None = None,
    persona_critics: bool | None = None,
    backend_id: str | None = None,
    sandbox: str | None = None,
) -> list[BlitzAgent]:
    """Plan which agents to spawn for a given topic.

    When use_llm=True, invokes an LLM to analyze the topic and determine
    optimal agent count and subtopic assignments. Falls back to heuristic.

    `domain` selects which preset under prompts/<domain>/ is used. If None,
    reads from blitz.toml [swarm] domain.

    `persona_critics` enables MAR-style persona-typed critic prompts (Phase 1).
    If None, reads from blitz.toml [swarm] persona_critics.
    """
    if domain is None or persona_critics is None:
        try:
            from config import load_config
            cfg = load_config()
            if domain is None:
                domain = cfg.swarm.domain
            if persona_critics is None:
                persona_critics = cfg.swarm.persona_critics
        except Exception:
            domain = domain or "general"
            persona_critics = bool(persona_critics)

    loader = PromptLoader(domain=domain)

    plan = None

    if use_llm:
        plan = _llm_plan(topic, backend_id=backend_id, sandbox=sandbox)

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

    subtopics = plan.get("subtopics")
    if subtopics and len(subtopics) >= researcher_count:
        subtopics = [f"{topic} — focusing on {st}" for st in subtopics[:researcher_count]]
    else:
        subtopics = _split_subtopics_heuristic(topic, researcher_count)

    agents: list[BlitzAgent] = []

    # Researchers
    for i, subtopic in enumerate(subtopics):
        sys_prompt, path, sha = _resolve_role_prompt(loader, "researcher")
        agents.append(BlitzAgent(
            id=f"researcher_{i:02d}",
            role="researcher",
            subtopic=subtopic,
            system_prompt=sys_prompt,
            model=ROLE_MODEL_OVERRIDES.get("researcher", "sonnet"),
            prompt_path=path,
            prompt_sha256=sha,
        ))

    # Critics — with personas if persona_critics enabled and we have files for them
    use_personas = bool(persona_critics) and critic_count >= 1
    if use_personas:
        # assign_personas returns N persona names for N slots
        personas = assign_personas(critic_count, round_n=1, has_unresolved_dissent=False)
        # Verify each persona prompt actually exists for the active domain (with fallback);
        # if any is missing, drop back to plain critic.
        for p in personas:
            try:
                loader.load("critic", persona=p)
            except PromptLoaderError:
                use_personas = False
                break

    if use_personas:
        for i, persona in enumerate(personas):
            sys_prompt, path, sha = _resolve_role_prompt(loader, "critic", persona=persona)
            suffix = f"_{persona}" if critic_count > 1 else f"_{persona}"
            agents.append(BlitzAgent(
                id=f"critic{suffix}",
                role="critic",
                subtopic=topic,
                system_prompt=sys_prompt,
                model="sonnet",
                persona=persona,
                prompt_path=path,
                prompt_sha256=sha,
            ))
    else:
        for i in range(critic_count):
            sys_prompt, path, sha = _resolve_role_prompt(loader, "critic")
            suffix = f"_{i:02d}" if critic_count > 1 else ""
            agents.append(BlitzAgent(
                id=f"critic{suffix}",
                role="critic",
                subtopic=topic,
                system_prompt=sys_prompt,
                model="sonnet",
                prompt_path=path,
                prompt_sha256=sha,
            ))

    # Fact-checker
    if needs_fc:
        sys_prompt, path, sha = _resolve_role_prompt(loader, "fact_checker")
        agents.append(BlitzAgent(
            id="fact_checker",
            role="fact_checker",
            subtopic=topic,
            system_prompt=sys_prompt,
            model="sonnet",
            prompt_path=path,
            prompt_sha256=sha,
        ))

    # Quality judge — always exactly 1
    sys_prompt, path, sha = _resolve_role_prompt(loader, "quality_judge")
    agents.append(BlitzAgent(
        id="quality_judge",
        role="quality_judge",
        subtopic=topic,
        system_prompt=sys_prompt,
        model=ROLE_MODEL_OVERRIDES.get("quality_judge", "sonnet"),
        prompt_path=path,
        prompt_sha256=sha,
    ))

    # Synthesizer — always exactly 1
    sys_prompt, path, sha = _resolve_role_prompt(loader, "synthesizer")
    agents.append(BlitzAgent(
        id="synthesizer",
        role="synthesizer",
        subtopic=topic,
        system_prompt=sys_prompt,
        model=ROLE_MODEL_OVERRIDES.get("synthesizer", "sonnet"),
        prompt_path=path,
        prompt_sha256=sha,
    ))

    return agents


def _llm_plan(
    topic: str,
    *,
    backend_id: str | None = None,
    sandbox: str | None = None,
) -> dict | None:
    """Use an LLM to determine optimal swarm composition for a topic."""
    prompt = (
        f"Analyze this research topic and determine the optimal agent swarm composition.\n\n"
        f"Topic: {topic}\n\n"
        f"Consider:\n"
        f"- How broad is this topic? (narrow=2 researchers, broad=4-6)\n"
        f"- Does it involve claims that need fact-checking? (empirical/technical=yes)\n"
        f"- How many critics are needed? (controversial=2, straightforward=1)\n"
        f"- What specific subtopics should each researcher focus on?"
    )

    data = _planning_backend_json(
        prompt,
        "You are a research planning assistant. Return JSON only.",
        json.loads(PLANNING_SCHEMA),
        model_hint="cheap",
        timeout_s=30,
        backend_id=backend_id,
        sandbox=sandbox,
    )
    if data and all(k in data for k in ("researcher_count", "critic_count", "needs_fact_checker")):
        data["researcher_count"] = max(2, min(6, int(data["researcher_count"])))
        data["critic_count"] = max(1, min(3, int(data["critic_count"])))
        return data

    return None


def _planning_backend_json(
    prompt: str,
    system_prompt: str,
    schema: dict,
    *,
    model_hint: str = "cheap",
    timeout_s: int = 30,
    backend_id: str | None = None,
    sandbox: str | None = None,
) -> dict | None:
    """Run a small planning call through the configured backend."""
    try:
        from backends import AgentCall, make_backend
        from config import load_config

        cfg = load_config()
        backend_id = backend_id or cfg.backend.default or "codex"
        provider = cfg.backend.get_provider(backend_id)
        model = provider.model or cfg.swarm.default_model
        resolved_sandbox = sandbox or provider.sandbox
        backend = make_backend(
            backend_id,
            adapter=provider.adapter,
            model=model,
            reasoning_effort=provider.reasoning_effort,
            sandbox=resolved_sandbox,
            approval_policy=provider.approval_policy,
            ephemeral=provider.ephemeral,
            base_url=provider.base_url,
            **provider.metadata,
        )
        result = backend.call(AgentCall(
            role="planner",
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
            model=model,
            timeout_s=timeout_s,
            sandbox=resolved_sandbox,
            approval_policy=provider.approval_policy,
            reasoning_effort=provider.reasoning_effort,
            ephemeral=provider.ephemeral,
        ))
        if result.errored:
            return None
        return result.parsed
    except Exception:
        return None
