"""Blitz-Swarm orchestrator — main entrypoint for the parallel agent swarm.

Usage:
    python orchestrator.py "topic string"
    python orchestrator.py "topic string" --max-rounds 3
    python orchestrator.py "topic string" --no-redis
"""

import asyncio
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from agents import (
    AGENT_OUTPUT_SCHEMA,
    BlitzAgent,
    plan_agents,
)
from backends import AgentBackend, AgentCall, make_backend
from config import load_config
from consensus import (
    check_consensus,
    extract_dissent,
    format_convergence_report,
    format_dissent_section,
    should_override_holdout,
)
from mechanisms.cascade_guard import CascadeGuard
from mechanisms.cascade_guard import GuardConfig as CascadeGuardConfig
from mechanisms.judge_ensemble import JudgeConfig, JudgeEnsemble, JudgeVote
from mechanisms.selector_synth import (
    PairwiseVerdict,
    SelectorConfig as MechanismSelectorConfig,
    SelectorSynth,
    Span,
)
from metrics import MetricsCollector

# ---------------------------------------------------------------------------
# Constants — loaded from config.py / blitz.toml
# ---------------------------------------------------------------------------

_config = load_config()

DEFAULT_MAX_ROUNDS = _config.swarm.max_rounds
AGENT_TIMEOUT_SECONDS = _config.swarm.timeout_seconds
OUTPUT_DIR = Path(_config.storage.output_dir) if not Path(_config.storage.output_dir).is_absolute() else Path(_config.storage.output_dir)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = Path(__file__).parent / _config.storage.output_dir
MAX_CONTEXT_TOKENS = _config.memory.max_context_tokens
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * 4

JUDGE_VOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "rubric_scores": {"type": "object"},
        "aggregate_score": {"type": "number", "minimum": 0, "maximum": 10},
        "rationale": {"type": "string"},
    },
    "required": ["aggregate_score", "rationale"],
}

PAIRWISE_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["a", "b", "tie"]},
        "rationale": {"type": "string"},
    },
    "required": ["winner", "rationale"],
}

# ---------------------------------------------------------------------------
# Backend/runtime helpers
# ---------------------------------------------------------------------------


FeatureOverride = Literal[True, False, None]


def _backend_settings(
    *,
    cfg=None,
    backend_id: str | None = None,
    sandbox: str | None = None,
):
    """Resolve backend/provider settings without mutating loaded config."""
    cfg = cfg or load_config()
    selected = backend_id or cfg.backend.default or "codex"
    try:
        provider = cfg.backend.get_provider(selected)
    except KeyError:
        raise ValueError(f"Unknown backend '{selected}'")
    resolved_sandbox = sandbox or provider.sandbox or "read-only"
    return cfg, selected, provider, resolved_sandbox


def _model_for_agent(agent: BlitzAgent, backend_id: str, provider, cfg) -> str:
    """Map legacy role models to backend-native models."""
    return provider.model or agent.model or cfg.swarm.default_model


def _make_runtime_backend(
    *,
    cfg=None,
    backend_id: str | None = None,
    sandbox: str | None = None,
) -> AgentBackend:
    cfg, selected, provider, resolved_sandbox = _backend_settings(
        cfg=cfg,
        backend_id=backend_id,
        sandbox=sandbox,
    )
    return make_backend(
        selected,
        adapter=provider.adapter,
        model=provider.model or cfg.swarm.default_model,
        reasoning_effort=provider.reasoning_effort,
        sandbox=resolved_sandbox,
        approval_policy=provider.approval_policy,
        ephemeral=provider.ephemeral,
        base_url=provider.base_url,
        **provider.metadata,
    )


def _feature_enabled(
    *,
    profile: str,
    configured: bool,
    override: FeatureOverride,
    enabled_in_max: bool = True,
    name: str | None = None,
) -> bool:
    # Highest precedence: BLITZ_FEATURE_OVERRIDES env var — a JSON object
    # mapping feature name -> bool, e.g. '{"judge_ensemble": false}'.
    # Lets jobs/CI force-toggle a mechanism without touching profile or
    # config. Malformed JSON or a non-dict payload is ignored (fall through
    # to the normal precedence chain) so a bad env var can never break a run.
    if name:
        raw = os.environ.get("BLITZ_FEATURE_OVERRIDES")
        if raw:
            try:
                env_overrides = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                env_overrides = None
            if isinstance(env_overrides, dict) and name in env_overrides:
                return bool(env_overrides[name])
    if override is not None:
        return override
    profile = (profile or "balanced").lower()
    if profile == "max" and enabled_in_max:
        return True
    if profile == "cheap":
        return False
    return bool(configured)


def _apply_guard(
    guard: CascadeGuard | None,
    outputs: list[dict],
    round_n: int,
) -> list[dict]:
    if guard is None:
        for output in outputs:
            output.setdefault("_round", round_n)
        return outputs
    guarded = [guard.on_agent_output(output, round_n) for output in outputs]
    summary = guard.round_summary(round_n)
    if summary.blocked_outputs or summary.tainted_descendants:
        print(
            f"  Cascade guard: blocked={summary.blocked_outputs} "
            f"tainted_descendants={summary.tainted_descendants}"
        )
    return guarded


def _filter_guarded(guard: CascadeGuard | None, outputs: list[dict]) -> list[dict]:
    return guard.filter_context(outputs) if guard is not None else list(outputs)


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------


def invoke_agent(
    agent: BlitzAgent,
    context: str,
    task: str,
    max_retries: int = 1,
    *,
    backend: AgentBackend | None = None,
    backend_id: str | None = None,
    sandbox: str | None = None,
    cfg=None,
) -> dict:
    """Invoke a single agent through the configured backend layer.

    Retries once on parse/schema failure and normalizes telemetry across
    Codex, Claude, and Gemini adapters.
    """
    agent_label = f"{agent.role}({agent.id})"
    trace_id = str(uuid.uuid4())
    cfg, selected_backend, provider, resolved_sandbox = _backend_settings(
        cfg=cfg,
        backend_id=backend_id,
        sandbox=sandbox,
    )
    active_backend = backend or _make_runtime_backend(
        cfg=cfg,
        backend_id=selected_backend,
        sandbox=resolved_sandbox,
    )
    model = _model_for_agent(agent, selected_backend, provider, cfg)
    _telemetry = {}

    def _tag(output: dict) -> dict:
        """Inject trace metadata and backend telemetry into every output."""
        output["_trace_id"] = trace_id
        output["_wall_clock_s"] = round(time.monotonic() - _invoke_start, 1)
        output.update(_telemetry)
        return output

    _invoke_start = time.monotonic()

    for attempt in range(1 + max_retries):
        user_prompt = _build_user_prompt(agent, context, task)

        try:
            result = active_backend.call(AgentCall(
                role=agent.role,
                prompt=user_prompt,
                system_prompt=agent.system_prompt,
                schema=AGENT_OUTPUT_SCHEMA,
                model=model,
                timeout_s=AGENT_TIMEOUT_SECONDS,
                cwd=Path(__file__).parent,
                sandbox=resolved_sandbox,
                approval_policy=provider.approval_policy,
                reasoning_effort=provider.reasoning_effort,
                ephemeral=provider.ephemeral,
            ))
            _telemetry = {
                "_backend_id": result.backend_id,
                "_model": result.model or model,
                "_elapsed_s": round(result.elapsed_s, 1),
                "_cost_usd": result.cost_usd,
                "_input_tokens": result.input_tokens,
                "_output_tokens": result.output_tokens,
                "_validation_errors": result.validation_errors,
                "backend_id": result.backend_id,
                "model": result.model or model,
                "elapsed_s": round(result.elapsed_s, 1),
                "errored": result.errored,
                "raw_stdout": result.raw_stdout[-4000:],
                "parsed": result.parsed,
                "cost_usd": result.cost_usd,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }

            if result.errored and result.parsed is None:
                print(f"  {agent_label} ERROR: {result.error or 'backend failure'}")
                return _tag(_error_output(agent, result.error or "backend failure"))

            if result.parsed:
                output = dict(result.parsed)
                output["agent_id"] = agent.id
                output["role"] = agent.role
            else:
                output = _parse_agent_output(agent, result.text or result.raw_stdout)

            # Retry on malformed output (raw text, not structured JSON)
            if (output.get("_raw") or result.validation_errors) and attempt < max_retries:
                print(
                    f"  {agent_label} malformed output, retrying "
                    f"[{result.elapsed_s:.1f}s]"
                )
                task = (
                    f"{task}\n\n"
                    "IMPORTANT: Your previous response did not satisfy the "
                    "required JSON schema. Respond with ONLY a JSON object, "
                    "no other text."
                )
                continue

            if result.validation_errors:
                output["_raw"] = True
                output["gaps_identified"] = output.get("gaps_identified", []) + [
                    "Backend result failed schema validation"
                ]
                output["quality_vote"] = output.get("quality_vote") or "needs_work"
                output["quality_notes"] = (
                    output.get("quality_notes", "")
                    or "; ".join(result.validation_errors[:3])
                )

            if not output.get("_error"):
                retry_note = f" (retry {attempt})" if attempt > 0 else ""
                print(f"  {agent_label} done [{result.elapsed_s:.1f}s]{retry_note}")
            return _tag(output)

        except Exception as e:
            print(f"  {agent_label} EXCEPTION: {e}")
            return _tag(_error_output(agent, str(e)))

    return _tag(_error_output(agent, "All retries exhausted"))


def _build_user_prompt(agent: BlitzAgent, context: str, task: str) -> str:
    """Build the user-facing prompt for an agent invocation."""
    sections = []

    sections.append(f"## Your Assigned Subtopic\n{agent.subtopic}")

    if context:
        sections.append(f"## Shared Context (current blackboard state)\n{context}")

    sections.append(f"## Task\n{task}")

    sections.append(
        "## Output Format\n"
        "CRITICAL: Do NOT use any tools. Do NOT read files or run commands. "
        "Respond directly from your knowledge with ONLY a JSON object.\n\n"
        "JSON fields:\n"
        "- findings (string): Your detailed analysis in markdown\n"
        "- key_points (array of strings): Most important takeaways\n"
        "- confidence (number 0-1): Your confidence in accuracy\n"
        "- gaps_identified (array of strings): Areas needing more research\n"
        "- quality_vote (string): 'ready' or 'needs_work'\n"
        "- quality_notes (string): Explanation for your vote\n"
        "- dissent (string): Any disagreements with other findings"
    )

    if agent.role == "quality_judge":
        sections.append(
            "## Quality Scores (REQUIRED for quality_judge)\n"
            "Include these numeric fields in your JSON:\n"
            "- coverage_score (number 0-10)\n"
            "- accuracy_score (number 0-10)\n"
            "- clarity_score (number 0-10)\n"
            "- depth_score (number 0-10)"
        )

    return "\n\n".join(sections)


def _parse_agent_output(agent: BlitzAgent, stdout: str) -> dict:
    """Parse agent stdout into a structured dict.

    Handles CLI envelope unwrapping, then tries JSON parsing, markdown
    code blocks, and brace extraction. Sets _raw flag on unstructured output.
    """
    stdout = stdout.strip()
    if not stdout:
        return _error_output(agent, "Empty output")

    # --output-format json wraps response in a CLI envelope.
    # Unwrap the "result" field first.
    try:
        envelope = json.loads(stdout)
        if isinstance(envelope, dict) and "result" in envelope:
            inner = envelope["result"]
            if isinstance(inner, dict):
                inner["agent_id"] = agent.id
                inner["role"] = agent.role
                return inner
            if inner is None:
                return _error_output(agent, "No result (model may have exhausted turns)")
            # result is a string — continue parsing below
            stdout = str(inner).strip()
        elif isinstance(envelope, dict) and "type" in envelope and "num_turns" in envelope:
            # CLI envelope without result — model exhausted turns
            return {
                "agent_id": agent.id,
                "role": agent.role,
                "findings": f"Agent exhausted {envelope.get('num_turns', '?')} turns",
                "_raw": True,
                "key_points": [],
                "confidence": 0.0,
                "quality_vote": "needs_work",
                "quality_notes": "Agent exhausted turn limit",
                "dissent": "",
            }
    except (json.JSONDecodeError, TypeError):
        pass

    if not stdout:
        return _error_output(agent, "Empty result after envelope unwrap")

    # Try direct JSON parse
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            data["agent_id"] = agent.id
            data["role"] = agent.role
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", stdout, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict):
                data["agent_id"] = agent.id
                data["role"] = agent.role
                return data
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the output
    brace_match = re.search(r"\{.*\}", stdout, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            if isinstance(data, dict):
                data["agent_id"] = agent.id
                data["role"] = agent.role
                return data
        except json.JSONDecodeError:
            pass

    # Last resort: wrap raw text as findings (flagged as _raw for retry)
    return {
        "agent_id": agent.id,
        "role": agent.role,
        "findings": stdout[:3000],
        "key_points": [],
        "confidence": 0.3,
        "gaps_identified": ["Output was not structured JSON — raw text captured"],
        "quality_vote": "needs_work",
        "quality_notes": "Agent produced unstructured output",
        "dissent": "",
        "_raw": True,
    }


def _error_output(agent: BlitzAgent, error_msg: str) -> dict:
    """Create a standardized error output dict for a failed agent."""
    return {
        "agent_id": agent.id,
        "role": agent.role,
        "findings": f"[ERROR] {error_msg}",
        "key_points": [],
        "confidence": 0.0,
        "gaps_identified": [f"Agent failed: {error_msg}"],
        "quality_vote": "needs_work",
        "quality_notes": f"Agent error: {error_msg}",
        "dissent": "",
        "_error": True,
        "_error_msg": error_msg,
    }


# ---------------------------------------------------------------------------
# Parallel blast
# ---------------------------------------------------------------------------


async def blast_agents(
    agents: list[BlitzAgent],
    context: str,
    task: str,
    *,
    backend: AgentBackend | None = None,
    backend_id: str | None = None,
    sandbox: str | None = None,
    cfg=None,
) -> list[dict]:
    """Invoke all agents in parallel and collect their outputs."""
    coros = [
        asyncio.to_thread(
            invoke_agent,
            agent,
            context,
            task,
            backend=backend,
            backend_id=backend_id,
            sandbox=sandbox,
            cfg=cfg,
        )
        for agent in agents
    ]
    return list(await asyncio.gather(*coros))


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def build_context(outputs: list[dict], for_role: str = "") -> str:
    """Build a role-filtered context string from agent outputs.

    - Researchers see: prior researcher findings + critic feedback
    - Critics/Fact-checkers/Judges see: all researcher findings + prior critic feedback
    - Synthesizer sees: everything
    """
    sections = []

    by_role = {}
    for o in outputs:
        role = o.get("role", "unknown")
        by_role.setdefault(role, []).append(o)

    researcher_outputs = by_role.get("researcher", [])
    critic_outputs = by_role.get("critic", [])
    fc_outputs = by_role.get("fact_checker", [])
    judge_outputs = by_role.get("quality_judge", [])

    # All roles see researcher findings
    if researcher_outputs:
        sections.append("### Researcher Findings")
        for o in researcher_outputs:
            aid = o.get("agent_id", "unknown")
            findings = o.get("findings", "")[:800]
            kps = o.get("key_points", [])
            conf = o.get("confidence", 0)
            sections.append(f"**{aid}** (confidence: {conf:.0%}):")
            sections.append(findings)
            if kps:
                sections.append("Key points: " + "; ".join(kps[:5]))
            sections.append("")

    # Evaluators and synthesizer see critic + fact-checker feedback
    if for_role in ("critic", "fact_checker", "quality_judge", "synthesizer", "researcher"):
        if critic_outputs:
            sections.append("### Critic Feedback")
            for o in critic_outputs:
                sections.append(o.get("findings", "")[:500])
                sections.append("")

        if fc_outputs:
            sections.append("### Fact-Checker Verification")
            for o in fc_outputs:
                sections.append(o.get("findings", "")[:500])
                sections.append("")

    # Synthesizer also sees judge feedback
    if for_role == "synthesizer" and judge_outputs:
        sections.append("### Quality Judge Assessment")
        for o in judge_outputs:
            sections.append(o.get("findings", "")[:500])
            sections.append("")

    context = "\n".join(sections)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated to fit token limit]"

    return context


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_final_output(
    topic: str,
    all_round_outputs: list[list[dict]],
    synthesizer_output: dict | None,
) -> str:
    """Format the final research document from all swarm outputs."""
    lines = [
        f"# {topic}",
        "",
        f"*Generated by Blitz-Swarm on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]

    if synthesizer_output and synthesizer_output.get("findings"):
        lines.append(synthesizer_output["findings"])
        lines.append("")
    else:
        lines.append("## Research Findings")
        lines.append("")
        last_round = all_round_outputs[-1] if all_round_outputs else []
        for o in last_round:
            if o.get("role") == "researcher":
                lines.append(f"### {o.get('agent_id', 'Researcher')}")
                lines.append(o.get("findings", ""))
                lines.append("")

    all_flat = [o for rnd in all_round_outputs for o in rnd]
    dissent = extract_dissent(all_flat)
    dissent_section = format_dissent_section(dissent)
    if dissent_section:
        lines.append(dissent_section)

    lines.append(format_convergence_report(all_round_outputs))

    return "\n".join(lines)


def save_output(topic: str, content: str) -> Path:
    """Save the final output to a markdown file."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower().strip())[:50].strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = OUTPUT_DIR / f"{slug}_{timestamp}.md"
    filepath.write_text(content, encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Blackboard integration helpers
# ---------------------------------------------------------------------------


def _try_connect_blackboard(use_redis: bool):
    """Attempt to connect to Redis blackboard. Returns None if disabled or unavailable."""
    if not use_redis:
        return None
    try:
        from blackboard import Blackboard
        bb = Blackboard()
        return bb
    except Exception as e:
        print(f"Warning: Redis unavailable ({e}). Running in memory-only mode.\n")
        return None


def _write_to_blackboard(bb, round_n: int, outputs: list[dict]):
    """Write round outputs to the blackboard if connected."""
    if bb is None:
        return
    for o in outputs:
        bb.write_agent_output(round_n, o.get("agent_id", "unknown"), o)


def _try_init_memory():
    """Try to initialize the memory reader and writer. Returns (reader, writer) or (None, None)."""
    try:
        from memory.reader import MemoryReader
        from memory.writer import MemoryWriter

        reader = MemoryReader()
        reader.initialize()

        writer = MemoryWriter()
        writer.initialize()

        return reader, writer
    except Exception as e:
        print(f"Warning: Memory system unavailable ({e})")
        return None, None


def _persist_to_memory(
    mem_writer, bb, topic, consensus_reached, override_applied,
    all_round_outputs, agents,
):
    """Persist completed task to G-Memory storage and finalize blackboard."""
    # Build interaction graph from round outputs
    utterances = []
    utterance_edges = []
    prior_ids_by_round = {}

    for round_n, round_outputs in enumerate(all_round_outputs):
        current_ids = []
        for o in round_outputs:
            import uuid
            utt_id = str(uuid.uuid4())
            utterances.append({
                "id": utt_id,
                "agent_id": o.get("agent_id", ""),
                "content": o.get("findings", "")[:500],
                "epoch": round_n,
                "timestamp": time.time(),
            })
            current_ids.append(utt_id)

            # Link to prior round outputs (temporal causation)
            for prior_id in prior_ids_by_round.get(round_n - 1, []):
                utterance_edges.append((prior_id, utt_id))

        prior_ids_by_round[round_n] = current_ids

    # Generate insight from synthesizer output
    synth_round = all_round_outputs[-1] if all_round_outputs else []
    synth_output = next((o for o in synth_round if o.get("role") == "synthesizer"), None)
    insight = ""
    if synth_output:
        # Use key points as a compact insight
        kps = synth_output.get("key_points", [])
        if kps:
            insight = f"On '{topic}': " + "; ".join(kps[:3])

    # Write to persistent memory
    if mem_writer:
        mem_writer.store_task(
            topic=topic,
            status="resolved" if consensus_reached else "failed",
            utterances=utterances,
            utterance_edges=utterance_edges,
            insight=insight,
        )

    # Write to Redis + finalize blackboard
    if bb:
        bb.queue_memory_write({
            "type": "task_complete",
            "topic": topic,
            "rounds": len(all_round_outputs) - 1,
            "consensus": consensus_reached,
            "override": override_applied,
            "agent_count": len(agents),
        })
        bb.finalize()

    # Cleanup
    if mem_writer:
        mem_writer.stop()


# ---------------------------------------------------------------------------
# Max-quality mechanisms
# ---------------------------------------------------------------------------


def _make_judge_fn(
    *,
    backend: AgentBackend,
    cfg,
    backend_id: str,
    provider,
    sandbox: str,
):
    """Build JudgeEnsemble's pluggable LLM hook on top of AgentBackend."""

    def _judge(
        candidate: str,
        history: str,
        judge_id: str,
        model_alias: str,
        seed: int,
        rubric_dims: tuple[str, ...],
    ) -> JudgeVote:
        model = provider.model or model_alias or cfg.swarm.default_model
        prompt = (
            f"You are {judge_id}, an independent quality judge.\n\n"
            f"Evaluate this candidate research output for the task.\n"
            f"Rubric dimensions: {', '.join(rubric_dims)}.\n"
            f"Score each dimension and provide an aggregate_score from 0 to 10.\n\n"
            f"Prior debate history:\n{history or '[none]'}\n\n"
            f"Candidate:\n{candidate}\n\n"
            "Return JSON with aggregate_score, rationale, and optional "
            "rubric_scores object."
        )
        result = backend.call(AgentCall(
            role="judge_ensemble",
            prompt=prompt,
            system_prompt="Return JSON only. Be strict, independent, and concise.",
            schema=JUDGE_VOTE_SCHEMA,
            model=model or cfg.swarm.default_model,
            timeout_s=min(AGENT_TIMEOUT_SECONDS, 90),
            cwd=Path(__file__).parent,
            sandbox=sandbox,
            approval_policy=provider.approval_policy,
            reasoning_effort=provider.reasoning_effort,
            ephemeral=provider.ephemeral,
        ))
        parsed = result.parsed or {}
        if result.errored and not parsed:
            return JudgeVote(
                judge_id=judge_id,
                round_n=0,
                rubric_scores={dim: 0.0 for dim in rubric_dims},
                aggregate_score=0.0,
                score_bucket=0,
                rationale=result.error or "judge backend error",
                model=model or "",
                seed=seed,
                errored=True,
            )

        raw_rubric = parsed.get("rubric_scores", {})
        rubric: dict[str, float] = {}
        for dim in rubric_dims:
            value = 0.0
            if isinstance(raw_rubric, dict) and raw_rubric.get(dim) is not None:
                value = raw_rubric.get(dim, 0.0)
            elif parsed.get(f"{dim}_score") is not None:
                value = parsed.get(f"{dim}_score", 0.0)
            elif parsed.get("aggregate_score") is not None:
                value = parsed.get("aggregate_score", 0.0)
            try:
                rubric[dim] = max(0.0, min(10.0, float(value)))
            except (TypeError, ValueError):
                rubric[dim] = 0.0

        aggregate = parsed.get("aggregate_score")
        try:
            aggregate_score = max(0.0, min(10.0, float(aggregate)))
        except (TypeError, ValueError):
            aggregate_score = sum(rubric.values()) / max(len(rubric), 1)

        return JudgeVote(
            judge_id=judge_id,
            round_n=0,
            rubric_scores=rubric,
            aggregate_score=aggregate_score,
            score_bucket=int(round(aggregate_score)),
            rationale=str(parsed.get("rationale", "")),
            model=model or "",
            seed=seed,
            errored=bool(result.validation_errors),
        )

    return _judge


def _run_judge_ensemble(
    *,
    topic: str,
    candidate_outputs: list[dict],
    history_outputs: list[dict],
    backend: AgentBackend,
    cfg,
    backend_id: str,
    provider,
    sandbox: str,
) -> dict:
    candidate = build_context(candidate_outputs, for_role="quality_judge")
    history = build_context(history_outputs, for_role="quality_judge")
    model = provider.model or cfg.swarm.default_model
    judge_cfg = JudgeConfig(
        n_judges=cfg.judge_ensemble.n_judges,
        ks_threshold=cfg.judge_ensemble.ks_threshold,
        ks_consecutive=cfg.judge_ensemble.ks_consecutive,
        min_rounds=cfg.judge_ensemble.min_rounds,
        judge_models=tuple(model for _ in range(cfg.judge_ensemble.n_judges)),
    )
    ensemble = JudgeEnsemble(
        judge_cfg,
        judge_fn=_make_judge_fn(
            backend=backend,
            cfg=cfg,
            backend_id=backend_id,
            provider=provider,
            sandbox=sandbox,
        ),
    )

    state = None
    while not ensemble.is_stable():
        state = ensemble.round(candidate, history)
        if state.halted:
            break

    score = ensemble.final_score()
    if score is None and state is not None:
        score = state.mean_score
    score = score or 0.0
    breakdown = ensemble.aggregate_breakdown()
    halt_reason = state.halt_reason if state is not None else "not_started"
    notes = f"Judge ensemble score {score:.1f}/10; halt_reason={halt_reason}"
    return {
        "agent_id": "judge_ensemble",
        "role": "quality_judge",
        "findings": (
            f"## Judge Ensemble Assessment\n\n{notes}\n\n"
            f"Topic: {topic}"
        ),
        "key_points": [
            notes,
            f"majority_bucket={ensemble.majority_vote()}",
        ],
        "confidence": max(0.0, min(1.0, score / 10.0)),
        "gaps_identified": [] if score >= 7.0 else ["Quality score below readiness threshold"],
        "quality_vote": "ready" if score >= 7.0 else "needs_work",
        "quality_notes": notes,
        "dissent": "" if score >= 7.0 else "Judge ensemble requested another iteration.",
        "coverage_score": breakdown.get("coverage", 0.0),
        "accuracy_score": breakdown.get("accuracy", 0.0),
        "clarity_score": breakdown.get("clarity", 0.0),
        "depth_score": breakdown.get("depth", 0.0),
        "_judge_ensemble": True,
        "_judge_state_log": [
            {
                "round_n": s.round_n,
                "mean_score": s.mean_score,
                "ks_stat": s.ks_stat,
                "halted": s.halted,
                "halt_reason": s.halt_reason,
            }
            for s in ensemble.state_log
        ],
    }


def _make_pairwise_judge_fn(
    *,
    backend: AgentBackend,
    cfg,
    backend_id: str,
    provider,
    sandbox: str,
):
    """Build SelectorSynth's pairwise judge hook on top of AgentBackend."""

    def _pairwise(span_a: Span, span_b: Span, topic: str, judge_id: str, seed: int) -> PairwiseVerdict:
        model = provider.model or cfg.swarm.default_model
        prompt = (
            f"You are {judge_id}, judging two candidate research spans for: {topic}\n\n"
            "Choose the span that is more accurate, specific, complete, and clear. "
            "Return winner='a', winner='b', or winner='tie'.\n\n"
            f"Span A:\n{span_a.text}\n\n"
            f"Span B:\n{span_b.text}\n"
        )
        result = backend.call(AgentCall(
            role="selector",
            prompt=prompt,
            system_prompt="Return JSON only with winner and rationale.",
            schema=PAIRWISE_VERDICT_SCHEMA,
            model=model or provider.model or cfg.swarm.default_model,
            timeout_s=min(AGENT_TIMEOUT_SECONDS, 90),
            cwd=Path(__file__).parent,
            sandbox=sandbox,
            approval_policy=provider.approval_policy,
            reasoning_effort=provider.reasoning_effort,
            ephemeral=provider.ephemeral,
        ))
        parsed = result.parsed or {}
        winner = parsed.get("winner", "tie")
        if winner not in ("a", "b", "tie"):
            winner = "tie"
        return PairwiseVerdict(
            judge_id=judge_id,
            span_a_id=span_a.id,
            span_b_id=span_b.id,
            winner=winner,
            rationale=str(parsed.get("rationale", result.error or "")),
        )

    return _pairwise


def _run_selector_synthesis(
    *,
    topic: str,
    researcher_outputs: list[dict],
    guard: CascadeGuard | None,
    backend: AgentBackend,
    cfg,
    backend_id: str,
    provider,
    sandbox: str,
) -> dict:
    selector_cfg = MechanismSelectorConfig(
        granularity=cfg.selector.granularity,
        n_judges=cfg.selector.n_judges,
        judge_models=tuple(
            (provider.model or cfg.swarm.default_model)
            for _ in range(cfg.selector.n_judges)
        ),
    )
    selector = SelectorSynth(
        selector_cfg,
        pairwise_judge_fn=_make_pairwise_judge_fn(
            backend=backend,
            cfg=cfg,
            backend_id=backend_id,
            provider=provider,
            sandbox=sandbox,
        ),
    )
    result = selector.synthesize(
        researcher_outputs,
        topic=topic,
        guard_filter=(lambda outs: _filter_guarded(guard, outs)) if guard else None,
    )
    ready = not result.final_text.startswith("[no ")
    return {
        "agent_id": "selector_synth",
        "role": "synthesizer",
        "findings": result.final_text,
        "key_points": [
            f"selected_spans={len(result.selected_span_ids)}",
            f"judge_calls={result.diagnostics.get('judge_calls', 0)}",
        ],
        "confidence": 0.85 if ready else 0.0,
        "gaps_identified": [] if ready else ["Selector synthesis had no usable inputs"],
        "quality_vote": "ready" if ready else "needs_work",
        "quality_notes": "Selection-bottleneck synthesis completed." if ready else result.final_text,
        "dissent": "",
        "_selector_synth": True,
        "_selector_diagnostics": result.diagnostics,
    }


def _contribution_scores(outputs: list[dict]) -> dict[str, float]:
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for output in outputs:
        aid = output.get("agent_id")
        if not aid:
            continue
        score = 0.0
        if not output.get("_error") and not output.get("_raw") and not output.get("_blocked"):
            score += 0.4
        score += max(0.0, min(0.3, float(output.get("confidence", 0.0) or 0.0) * 0.3))
        score += min(0.2, len(output.get("findings", "") or "") / 4000)
        score += min(0.1, len(output.get("key_points", []) or []) / 50)
        scores[aid] = scores.get(aid, 0.0) + score
        counts[aid] = counts.get(aid, 0) + 1
    return {aid: scores[aid] / max(counts.get(aid, 1), 1) for aid in scores}


def _prune_low_contribution_agents(
    agents: list[BlitzAgent],
    outputs: list[dict],
    *,
    selector_enabled: bool,
) -> list[BlitzAgent]:
    """AgentDropout-style conservative pruning with role floors."""
    if not outputs or len(agents) <= 3:
        return agents
    scores = _contribution_scores(outputs)
    floors = {
        "researcher": 2 if selector_enabled else 1,
        "critic": 1,
        "fact_checker": 1,
        "quality_judge": 1,
    }
    by_role: dict[str, list[BlitzAgent]] = {}
    for agent in agents:
        by_role.setdefault(agent.role, []).append(agent)

    removable = [
        agent for agent in agents
        if len(by_role.get(agent.role, [])) > floors.get(agent.role, 0)
        and scores.get(agent.id, 1.0) < 0.2
    ]
    if not removable:
        return agents
    remove = min(removable, key=lambda a: scores.get(a.id, 0.0))
    print(f"  AgentDropout: pruning {remove.id} (score={scores.get(remove.id, 0.0):.2f})")
    return [agent for agent in agents if agent.id != remove.id]


# ---------------------------------------------------------------------------
# Main swarm loop — full iterative consensus
# ---------------------------------------------------------------------------


async def run_swarm(
    topic: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    use_redis: bool = True,
    *,
    backend_id: str | None = None,
    quality_profile: str | None = None,
    sandbox: str | None = None,
    use_selector: FeatureOverride = None,
    use_judge_ensemble: FeatureOverride = None,
    use_cascade_guard: FeatureOverride = None,
    use_llm_plan: bool = True,
) -> Path:
    """Run the Blitz-Swarm pipeline on a topic.

    Full lifecycle: SPAWN -> BLAST -> WRITE -> CHECK -> ITERATE -> FINALIZE

    All agents (researchers + evaluators) blast simultaneously each round.
    After each round, consensus is checked. If not reached, the swarm
    re-blasts with updated blackboard context. The synthesizer runs once
    after consensus or max rounds.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_cfg = load_config()
    profile = quality_profile or runtime_cfg.swarm.quality_profile
    _, selected_backend, provider, resolved_sandbox = _backend_settings(
        cfg=runtime_cfg,
        backend_id=backend_id,
        sandbox=sandbox,
    )
    backend = _make_runtime_backend(
        cfg=runtime_cfg,
        backend_id=selected_backend,
        sandbox=resolved_sandbox,
    )
    guard_enabled = _feature_enabled(
        profile=profile,
        configured=runtime_cfg.guard.enabled,
        override=use_cascade_guard,
        name="cascade_guard",
    )
    judge_ensemble_enabled = _feature_enabled(
        profile=profile,
        configured=runtime_cfg.judge_ensemble.enabled,
        override=use_judge_ensemble,
        name="judge_ensemble",
    )
    selector_enabled = _feature_enabled(
        profile=profile,
        configured=runtime_cfg.selector.enabled,
        override=use_selector,
        name="selector",
    )
    guard = (
        CascadeGuard(CascadeGuardConfig(mode=runtime_cfg.guard.mode))
        if guard_enabled else None
    )

    print(f"\n{'='*60}")
    print(f"BLITZ-SWARM — {topic}")
    print(f"{'='*60}\n")
    print(
        f"Backend: {selected_backend} | model={provider.model or runtime_cfg.swarm.default_model} "
        f"| adapter={provider.adapter} | profile={profile} | sandbox={resolved_sandbox}"
    )
    print(
        "Mechanisms: "
        f"cascade_guard={'on' if guard_enabled else 'off'}, "
        f"judge_ensemble={'on' if judge_ensemble_enabled else 'off'}, "
        f"selector={'on' if selector_enabled else 'off'}"
    )
    print()

    # --- METRICS: Initialize ---
    mc = MetricsCollector()
    mc.start_run(run_id, topic)

    # --- SPAWN ---
    agents = plan_agents(
        topic,
        use_llm=use_llm_plan,
        domain=runtime_cfg.swarm.domain,
        persona_critics=runtime_cfg.swarm.persona_critics,
        backend_id=selected_backend,
        sandbox=resolved_sandbox,
    )
    researchers = [a for a in agents if a.role == "researcher"]
    if judge_ensemble_enabled:
        evaluators = [a for a in agents if a.role in ("critic", "fact_checker")]
    else:
        evaluators = [a for a in agents if a.role in ("critic", "fact_checker", "quality_judge")]
    synthesizer_agents = [a for a in agents if a.role == "synthesizer"]
    blast_agents_list = researchers + evaluators  # everyone except synthesizer

    print(f"Swarm: {len(agents)} agents ({len(researchers)} researchers, "
          f"{len(evaluators)} evaluators, {len(synthesizer_agents)} synthesizer)")
    for a in agents:
        print(f"  {a.id:20s} | {a.role:15s} | model={a.model}")
    print()

    # Connect blackboard (optional)
    bb = _try_connect_blackboard(use_redis)
    if bb:
        bb.initialize(topic, [a.id for a in agents])
        print("Blackboard: Redis connected")
    else:
        print("Blackboard: in-memory mode")

    # --- MEMORY: Retrieve historical context ---
    memory_context = ""
    mem_reader, mem_writer = _try_init_memory()
    if mem_reader and mem_reader.is_available():
        memory = mem_reader.retrieve_memory(topic)
        memory_context = mem_reader.build_memory_context(memory)
        n_insights = len(memory.get("insights", []))
        n_queries = len(memory.get("related_queries", []))
        print(f"Memory: {n_insights} insights, {n_queries} related queries loaded")
    else:
        print("Memory: no historical data yet")
    print()

    all_round_outputs = []  # list of lists, one per round
    accumulated_outputs = []  # flat list of all outputs across rounds
    consensus_reached = False
    override_applied = False
    final_round_n = 0

    # --- BLAST + CHECK loop ---
    for round_n in range(1, max_rounds + 1):
        final_round_n = round_n
        if bb:
            bb.advance_round()

        round_label = f"round_{round_n}"
        mc.start_round(round_label)

        print(f"--- Round {round_n}/{max_rounds}: "
              f"{len(blast_agents_list)} agents blasting ---")

        # Build per-role context from accumulated outputs
        # Round 1: researchers get no context, evaluators get no context
        # Round 2+: everyone gets prior round outputs
        task_prompt = f"Research and analyze: {topic}"

        if round_n == 1:
            # First round: blast researchers only (evaluators need findings first)
            # Inject historical memory context if available
            initial_context = memory_context if memory_context else ""
            print(f"  Phase A: {len(researchers)} researchers...")
            start = time.monotonic()
            r_outputs = await blast_agents(
                researchers,
                initial_context,
                task_prompt,
                backend=backend,
                backend_id=selected_backend,
                sandbox=resolved_sandbox,
                cfg=runtime_cfg,
            )
            r_outputs = _apply_guard(guard, r_outputs, round_n)
            elapsed = time.monotonic() - start
            print(f"  Researchers done [{elapsed:.1f}s]")

            # Now blast evaluators with researcher context
            r_context = build_context(_filter_guarded(guard, r_outputs), for_role="critic")
            eval_task = (
                f"Evaluate the research findings on: {topic}\n\n"
                "Review the researcher outputs. Identify gaps, contradictions, "
                "and quality issues."
            )
            print(f"  Phase B: {len(evaluators)} evaluators...")
            start = time.monotonic()
            e_outputs = await blast_agents(
                evaluators,
                r_context,
                eval_task,
                backend=backend,
                backend_id=selected_backend,
                sandbox=resolved_sandbox,
                cfg=runtime_cfg,
            )
            e_outputs = _apply_guard(guard, e_outputs, round_n)
            elapsed = time.monotonic() - start
            print(f"  Evaluators done [{elapsed:.1f}s]")

            if judge_ensemble_enabled:
                print("  Phase C: judge ensemble...")
                judge_output = _run_judge_ensemble(
                    topic=topic,
                    candidate_outputs=_filter_guarded(guard, r_outputs + e_outputs),
                    history_outputs=_filter_guarded(guard, accumulated_outputs),
                    backend=backend,
                    cfg=runtime_cfg,
                    backend_id=selected_backend,
                    provider=provider,
                    sandbox=resolved_sandbox,
                )
                e_outputs.extend(_apply_guard(guard, [judge_output], round_n))

            round_outputs = r_outputs + e_outputs
        else:
            # Round 2+: all agents blast simultaneously with full context
            # Each agent gets role-appropriate context
            round_outputs = []
            coros = []

            for agent in blast_agents_list:
                ctx = build_context(
                    _filter_guarded(guard, accumulated_outputs),
                    for_role=agent.role,
                )
                if agent.role == "researcher":
                    task = (
                        f"Continue researching: {topic}\n\n"
                        "Review the shared context — prior findings and critic "
                        "feedback. Address gaps and improve coverage."
                    )
                else:
                    task = (
                        f"Re-evaluate the research on: {topic}\n\n"
                        "Review updated findings. Have prior issues been addressed? "
                        "Vote 'ready' if quality is sufficient, 'needs_work' if not."
                    )
                coros.append(asyncio.to_thread(
                    invoke_agent,
                    agent,
                    ctx,
                    task,
                    backend=backend,
                    backend_id=selected_backend,
                    sandbox=resolved_sandbox,
                    cfg=runtime_cfg,
                ))

            start = time.monotonic()
            round_outputs = list(await asyncio.gather(*coros))
            round_outputs = _apply_guard(guard, round_outputs, round_n)
            if judge_ensemble_enabled:
                judge_output = _run_judge_ensemble(
                    topic=topic,
                    candidate_outputs=_filter_guarded(guard, round_outputs),
                    history_outputs=_filter_guarded(guard, accumulated_outputs),
                    backend=backend,
                    cfg=runtime_cfg,
                    backend_id=selected_backend,
                    provider=provider,
                    sandbox=resolved_sandbox,
                )
                round_outputs.extend(_apply_guard(guard, [judge_output], round_n))
            elapsed = time.monotonic() - start
            print(f"  All agents done [{elapsed:.1f}s]")

        # --- WRITE ---
        _write_to_blackboard(bb, round_n, round_outputs)
        all_round_outputs.append(round_outputs)
        accumulated_outputs = [o for rnd in all_round_outputs for o in rnd]

        # --- METRICS: End round ---
        mc.end_round(round_label, round_outputs)

        # --- CHECK ---
        guarded_round_outputs = _filter_guarded(guard, round_outputs)
        eval_votes = [o for o in guarded_round_outputs
                      if o.get("role") in ("critic", "fact_checker", "quality_judge")]
        consensus_reached = check_consensus(eval_votes)

        voters = [o for o in eval_votes if o.get("quality_vote") is not None]
        ready_count = sum(1 for v in voters if v["quality_vote"] == "ready")
        total_voters = len(voters)
        avg_conf = (
            sum(v.get("confidence", 0) for v in voters) / total_voters
            if total_voters > 0 else 0
        )

        # --- METRICS: Record consensus state ---
        mc.record_consensus_state(round_label, ready_count, total_voters, avg_conf)

        # --- METRICS: Record quality scores from judge ---
        judge_outputs = [o for o in round_outputs if o.get("role") == "quality_judge"]
        for jo in judge_outputs:
            if jo.get("coverage_score") is not None:
                mc.record_quality_scores(jo)

        print(f"\n  Consensus: {ready_count}/{total_voters} ready | "
              f"avg confidence: {avg_conf:.0%}")

        if consensus_reached:
            print(f"  CONSENSUS REACHED in round {round_n}.\n")
            break

        # Check holdout override
        if should_override_holdout(round_n, eval_votes):
            dissent = extract_dissent(eval_votes)
            holdout = dissent[0] if dissent else {}
            print(f"  HOLDOUT OVERRIDE: {holdout.get('agent_id', '?')} overridden "
                  f"after round {round_n}. Dissent preserved.\n")
            override_applied = True
            consensus_reached = True
            break

        if round_n < max_rounds:
            dissent = extract_dissent(eval_votes)
            print(f"  {len(dissent)} dissenting vote(s) — iterating...")
            for d in dissent:
                print(f"    {d['agent_id']}: {d.get('quality_notes', '')[:80]}")
            print()
        else:
            print(f"  Max rounds ({max_rounds}) reached. Force-finalizing.\n")

        if round_n < max_rounds and not consensus_reached:
            blast_agents_list = _prune_low_contribution_agents(
                blast_agents_list,
                _filter_guarded(guard, accumulated_outputs),
                selector_enabled=selector_enabled,
            )
            researchers = [a for a in blast_agents_list if a.role == "researcher"]
            evaluators = [
                a for a in blast_agents_list
                if a.role in ("critic", "fact_checker", "quality_judge")
            ]

    # --- FINALIZE: Synthesizer ---
    mc.start_round("synthesis")

    guarded_accumulated = _filter_guarded(guard, accumulated_outputs)
    synth_context = build_context(guarded_accumulated, for_role="synthesizer")
    synth_task = (
        f"Synthesize all findings on: {topic}\n\n"
        "Integrate all researcher findings, critic feedback, and fact-checker "
        "verification into a single coherent, well-structured technical summary. "
        "Preserve dissenting views in a dedicated section."
    )

    if not consensus_reached:
        synth_task += (
            "\n\nNote: The swarm did not reach full consensus. Include a "
            "quality warning noting unresolved issues."
        )

    print(f"--- Final: Synthesizer ---")
    start = time.monotonic()
    if selector_enabled:
        researcher_outputs = [
            o for o in guarded_accumulated if o.get("role") == "researcher"
        ]
        synth_output = _run_selector_synthesis(
            topic=topic,
            researcher_outputs=researcher_outputs,
            guard=guard,
            backend=backend,
            cfg=runtime_cfg,
            backend_id=selected_backend,
            provider=provider,
            sandbox=resolved_sandbox,
        )
        synth_outputs = _apply_guard(guard, [synth_output], final_round_n + 1)
    else:
        synth_outputs = await blast_agents(
            synthesizer_agents,
            synth_context,
            synth_task,
            backend=backend,
            backend_id=selected_backend,
            sandbox=resolved_sandbox,
            cfg=runtime_cfg,
        )
        synth_outputs = _apply_guard(guard, synth_outputs, final_round_n + 1)
    elapsed = time.monotonic() - start
    print(f"  Synthesis complete [{elapsed:.1f}s]\n")
    all_round_outputs.append(synth_outputs)

    mc.end_round("synthesis", synth_outputs)

    synth_output = synth_outputs[0] if synth_outputs else None

    # --- METRICS: Record consensus result + finalize + save ---
    mc.record_consensus_result(final_round_n, consensus_reached, override_applied)
    mc.finalize()
    metrics_record = mc.save()

    # --- Persist to memory (non-fatal if it fails) ---
    try:
        _persist_to_memory(
            mem_writer, bb, topic, consensus_reached, override_applied,
            all_round_outputs, agents,
        )
    except Exception as e:
        print(f"Warning: Memory persistence failed ({e})")

    # --- Output ---
    final_doc = format_final_output(topic, all_round_outputs, synth_output)
    filepath = save_output(topic, final_doc)

    total_agents_invoked = sum(len(rnd) for rnd in all_round_outputs)
    cost_str = f"${metrics_record.get('total_cost_usd', 0):.4f}"
    tokens_str = f"{metrics_record.get('total_input_tokens', 0) + metrics_record.get('total_output_tokens', 0):,}"

    print(f"{'='*60}")
    print(f"OUTPUT SAVED: {filepath}")
    print(f"Rounds: {final_round_n} + synthesis")
    print(f"Total agent invocations: {total_agents_invoked}")
    print(f"Consensus: {'yes' if consensus_reached else 'no'}"
          f"{' (override)' if override_applied else ''}")
    print(f"Cost: {cost_str} | Tokens: {tokens_str}")
    print(f"Quality: avg={metrics_record.get('avg_quality', 0)} "
          f"(cov={metrics_record.get('coverage', 0)} "
          f"acc={metrics_record.get('accuracy', 0)} "
          f"clar={metrics_record.get('clarity', 0)} "
          f"dep={metrics_record.get('depth', 0)})")
    print(f"Wall clock: {metrics_record.get('total_wall_clock_s', 0):.1f}s")
    print(f"Metrics saved to: metrics.jsonl")
    print(f"{'='*60}\n")

    return filepath


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _dry_run(
    topic: str,
    use_llm: bool,
    *,
    backend_id: str | None = None,
    quality_profile: str | None = None,
    sandbox: str | None = None,
):
    """Show the agent plan without executing."""
    from agents import plan_agents

    cfg, selected_backend, provider, resolved_sandbox = _backend_settings(
        backend_id=backend_id,
        sandbox=sandbox,
    )
    profile = quality_profile or cfg.swarm.quality_profile

    print(f"\n{'='*60}")
    print(f"BLITZ-SWARM DRY RUN — {topic}")
    print(f"{'='*60}\n")
    print(
        f"Backend: {selected_backend} | model={provider.model or cfg.swarm.default_model} "
        f"| adapter={provider.adapter} | profile={profile} | sandbox={resolved_sandbox}"
    )
    print()

    agents = plan_agents(
        topic,
        use_llm=use_llm,
        domain=cfg.swarm.domain,
        persona_critics=cfg.swarm.persona_critics,
        backend_id=selected_backend,
        sandbox=resolved_sandbox,
    )
    researchers = [a for a in agents if a.role == "researcher"]
    evaluators = [a for a in agents if a.role in ("critic", "fact_checker", "quality_judge")]
    synthesizer = [a for a in agents if a.role == "synthesizer"]

    print(f"Swarm composition: {len(agents)} agents")
    print(f"  Researchers: {len(researchers)}")
    print(f"  Evaluators:  {len(evaluators)}")
    print(f"  Synthesizer: {len(synthesizer)}")
    print()

    for a in agents:
        print(f"  {a.id:20s} | {a.role:15s} | model={a.model}")
        if a.role == "researcher":
            # Show subtopic assignment
            sub = a.subtopic
            if " — focusing on " in sub:
                sub = sub.split(" — focusing on ", 1)[1]
            print(f"  {'':20s}   subtopic: {sub[:60]}")

    print(f"\nNo agents will be invoked. Use without --dry-run to execute.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Blitz-Swarm: parallel multi-agent research swarm",
    )
    parser.add_argument("topic", help="The topic / task spec")
    parser.add_argument(
        "--mode", choices=["consensus", "mythos"], default="consensus",
        help="Swarm mode: 'consensus' (default, flat parallel) or "
             "'mythos' (hierarchical planner+executors+verifier)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
        help=f"Maximum consensus rounds (default: {DEFAULT_MAX_ROUNDS}) — consensus mode only",
    )
    parser.add_argument(
        "--no-redis", action="store_true",
        help="Run without Redis (in-memory blackboard only)",
    )
    parser.add_argument(
        "--no-llm-plan", action="store_true",
        help="Use heuristic agent planning instead of LLM — consensus mode only",
    )
    parser.add_argument(
        "--backend", default=None,
        help="Backend provider id from blitz.toml [backend.providers]",
    )
    parser.add_argument(
        "--quality-profile", choices=["max", "balanced", "cheap"], default=None,
        help="Mechanism profile controlling quality/cost tradeoffs",
    )
    parser.add_argument(
        "--sandbox", choices=["read-only", "workspace-write"], default=None,
        help="Sandbox passed to local backend invocations",
    )
    parser.add_argument(
        "--no-selector", action="store_true",
        help="Disable selector synthesis even when max-quality enables it",
    )
    parser.add_argument(
        "--no-judge-ensemble", action="store_true",
        help="Disable judge ensemble even when max-quality enables it",
    )
    parser.add_argument(
        "--no-cascade-guard", action="store_true",
        help="Disable cascade guard context filtering",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the agent plan without executing",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable detailed logging",
    )
    # Mythos-mode flags
    parser.add_argument(
        "--max-replans", type=int, default=None,
        help="Mythos mode: max planner→exec→verify→replan rounds (default: 3)",
    )
    parser.add_argument(
        "--cost-ceiling", type=float, default=None,
        help="Mythos mode: hard $ ceiling per task (default: 5.0)",
    )

    args = parser.parse_args()

    if args.mode == "mythos":
        _run_mythos_mode(args)
        return

    if args.dry_run:
        _dry_run(
            args.topic,
            use_llm=not args.no_llm_plan,
            backend_id=args.backend,
            quality_profile=args.quality_profile,
            sandbox=args.sandbox,
        )
        return

    filepath = asyncio.run(
        run_swarm(
            args.topic,
            max_rounds=args.max_rounds,
            use_redis=not args.no_redis,
            backend_id=args.backend,
            quality_profile=args.quality_profile,
            sandbox=args.sandbox,
            use_selector=False if args.no_selector else None,
            use_judge_ensemble=False if args.no_judge_ensemble else None,
            use_cascade_guard=False if args.no_cascade_guard else None,
            use_llm_plan=not args.no_llm_plan,
        )
    )
    print(f"Done. Output at: {filepath}")


def _run_mythos_mode(args) -> None:
    """Dispatch to the mythos package."""
    import os

    from mythos import load_mythos_config, run_mythos

    if args.backend:
        os.environ["BLITZ_BACKEND"] = args.backend
    if args.sandbox:
        os.environ["BLITZ_SANDBOX"] = args.sandbox

    cfg = load_mythos_config()
    if args.max_replans is not None:
        cfg.max_replans = args.max_replans
    if args.cost_ceiling is not None:
        cfg.cost_ceiling_usd = args.cost_ceiling
    if args.no_redis:
        cfg.use_redis = False

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"MYTHOS DRY RUN — {args.topic}")
        print(f"{'='*60}\n")
        print(f"Config:")
        print(f"  planner_model     = {cfg.planner_model}")
        print(f"  executor_model    = {cfg.executor_model}")
        print(f"  verifier_model    = {cfg.verifier_model}")
        print(f"  max_replans       = {cfg.max_replans}")
        print(f"  max_executors     = {cfg.max_executors}")
        print(f"  cost_ceiling_usd  = {cfg.cost_ceiling_usd}")
        print(f"  output_dir        = {cfg.output_dir}")
        print(f"  backend           = {os.environ.get('BLITZ_BACKEND') or load_config().backend.default}")
        loaded = load_config()
        default_provider = loaded.backend.get_provider(loaded.backend.default)
        print(f"  sandbox           = {os.environ.get('BLITZ_SANDBOX') or default_provider.sandbox}")
        print(f"\nNo CLI calls will be made. Use without --dry-run to execute.")
        return

    asyncio.run(run_mythos(args.topic, cfg))


if __name__ == "__main__":
    main()
