"""Blitz-Swarm orchestrator — main entrypoint for the parallel agent swarm.

Usage:
    python orchestrator.py "topic string"
    python orchestrator.py "topic string" --max-rounds 3
    python orchestrator.py "topic string" --no-redis
"""

import asyncio
import json
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from agents import (
    AGENT_OUTPUT_SCHEMA_JSON,
    BlitzAgent,
    plan_agents,
)
from config import load_config
from consensus import (
    check_consensus,
    extract_dissent,
    format_convergence_report,
    format_dissent_section,
    should_override_holdout,
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

# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------


def invoke_agent(agent: BlitzAgent, context: str, task: str,
                  max_retries: int = 1) -> dict:
    """Invoke a single agent as a subprocess via the Claude CLI.

    Retries once on parse failure. Extracts cost data from the CLI envelope.
    Attempts partial output recovery on timeout.
    """
    agent_label = f"{agent.role}({agent.id})"
    trace_id = str(uuid.uuid4())
    _envelope_cost = {}  # cost data extracted from CLI envelope

    def _tag(output: dict) -> dict:
        """Inject trace metadata and cost data into every output."""
        output["_trace_id"] = trace_id
        output["_wall_clock_s"] = round(time.monotonic() - _invoke_start, 1)
        output.update(_envelope_cost)
        return output

    _invoke_start = time.monotonic()

    for attempt in range(1 + max_retries):
        user_prompt = _build_user_prompt(agent, context, task)

        cmd = [
            "claude",
            "-p", user_prompt,
            "--system-prompt", agent.system_prompt,
            "--output-format", "json",
            "--model", agent.model,
            "--max-turns", "3",
            "--dangerously-skip-permissions",
        ]

        start = time.monotonic()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=AGENT_TIMEOUT_SECONDS,
            )
            elapsed = time.monotonic() - start

            # Extract cost data from CLI envelope before parsing
            try:
                _env = json.loads(result.stdout)
                if isinstance(_env, dict):
                    _envelope_cost["_cost_usd"] = _env.get("total_cost_usd", 0)
                    _usage = _env.get("usage", {})
                    _envelope_cost["_input_tokens"] = _usage.get("input_tokens", 0)
                    _envelope_cost["_output_tokens"] = _usage.get("output_tokens", 0)
            except (json.JSONDecodeError, TypeError):
                pass

            if result.returncode != 0:
                stderr = (result.stderr or "")[:300]
                stdout_preview = (result.stdout or "")[:300]
                print(f"  {agent_label} ERROR: exit {result.returncode}")
                if stderr:
                    print(f"    stderr: {stderr}")
                if stdout_preview:
                    print(f"    stdout: {stdout_preview}")
                return _tag(_error_output(agent, f"Exit code {result.returncode}"))

            output = _parse_agent_output(agent, result.stdout)

            # Retry on malformed output (raw text, not structured JSON)
            if output.get("_raw") and attempt < max_retries:
                print(f"  {agent_label} malformed output, retrying [{elapsed:.1f}s]")
                task = (
                    f"{task}\n\n"
                    f"IMPORTANT: Your previous response was not valid JSON. "
                    f"Respond with ONLY a JSON object, no other text."
                )
                continue

            if not output.get("_error"):
                retry_note = f" (retry {attempt})" if attempt > 0 else ""
                print(f"  {agent_label} done [{elapsed:.1f}s]{retry_note}")
            return _tag(output)

        except subprocess.TimeoutExpired as e:
            elapsed = time.monotonic() - start
            # Try to recover partial output from the timed-out process
            partial_stdout = ""
            if e.stdout:
                partial_stdout = (
                    e.stdout if isinstance(e.stdout, str)
                    else e.stdout.decode("utf-8", errors="replace")
                )
            if partial_stdout.strip():
                output = _parse_agent_output(agent, partial_stdout)
                if not output.get("_raw") and not output.get("_error"):
                    output["_partial"] = True
                    print(f"  {agent_label} TIMEOUT [{elapsed:.1f}s] (recovered partial output)")
                    return _tag(output)
            print(f"  {agent_label} TIMEOUT [{elapsed:.1f}s]")
            return _tag(_error_output(agent, f"Timed out after {AGENT_TIMEOUT_SECONDS}s"))

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
    }


# ---------------------------------------------------------------------------
# Parallel blast
# ---------------------------------------------------------------------------


async def blast_agents(agents: list[BlitzAgent], context: str, task: str) -> list[dict]:
    """Invoke all agents in parallel and collect their outputs."""
    coros = [
        asyncio.to_thread(invoke_agent, agent, context, task)
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
# Main swarm loop — full iterative consensus
# ---------------------------------------------------------------------------


async def run_swarm(
    topic: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    use_redis: bool = True,
) -> Path:
    """Run the Blitz-Swarm pipeline on a topic.

    Full lifecycle: SPAWN -> BLAST -> WRITE -> CHECK -> ITERATE -> FINALIZE

    All agents (researchers + evaluators) blast simultaneously each round.
    After each round, consensus is checked. If not reached, the swarm
    re-blasts with updated blackboard context. The synthesizer runs once
    after consensus or max rounds.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"BLITZ-SWARM — {topic}")
    print(f"{'='*60}\n")

    # --- METRICS: Initialize ---
    mc = MetricsCollector()
    mc.start_run(run_id, topic)

    # --- SPAWN ---
    agents = plan_agents(topic)
    researchers = [a for a in agents if a.role == "researcher"]
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
            r_outputs = await blast_agents(researchers, initial_context, task_prompt)
            elapsed = time.monotonic() - start
            print(f"  Researchers done [{elapsed:.1f}s]")

            # Now blast evaluators with researcher context
            r_context = build_context(r_outputs, for_role="critic")
            eval_task = (
                f"Evaluate the research findings on: {topic}\n\n"
                "Review the researcher outputs. Identify gaps, contradictions, "
                "and quality issues."
            )
            print(f"  Phase B: {len(evaluators)} evaluators...")
            start = time.monotonic()
            e_outputs = await blast_agents(evaluators, r_context, eval_task)
            elapsed = time.monotonic() - start
            print(f"  Evaluators done [{elapsed:.1f}s]")

            round_outputs = r_outputs + e_outputs
        else:
            # Round 2+: all agents blast simultaneously with full context
            # Each agent gets role-appropriate context
            round_outputs = []
            coros = []

            for agent in blast_agents_list:
                ctx = build_context(accumulated_outputs, for_role=agent.role)
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
                coros.append(asyncio.to_thread(invoke_agent, agent, ctx, task))

            start = time.monotonic()
            round_outputs = list(await asyncio.gather(*coros))
            elapsed = time.monotonic() - start
            print(f"  All agents done [{elapsed:.1f}s]")

        # --- WRITE ---
        _write_to_blackboard(bb, round_n, round_outputs)
        all_round_outputs.append(round_outputs)
        accumulated_outputs = [o for rnd in all_round_outputs for o in rnd]

        # --- METRICS: End round ---
        mc.end_round(round_label, round_outputs)

        # --- CHECK ---
        eval_votes = [o for o in round_outputs
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

    # --- FINALIZE: Synthesizer ---
    mc.start_round("synthesis")

    synth_context = build_context(accumulated_outputs, for_role="synthesizer")
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
    synth_outputs = await blast_agents(synthesizer_agents, synth_context, synth_task)
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


def _dry_run(topic: str, use_llm: bool):
    """Show the agent plan without executing."""
    from agents import plan_agents

    print(f"\n{'='*60}")
    print(f"BLITZ-SWARM DRY RUN — {topic}")
    print(f"{'='*60}\n")

    agents = plan_agents(topic, use_llm=use_llm)
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
        _dry_run(args.topic, use_llm=not args.no_llm_plan)
        return

    filepath = asyncio.run(
        run_swarm(
            args.topic,
            max_rounds=args.max_rounds,
            use_redis=not args.no_redis,
        )
    )
    print(f"Done. Output at: {filepath}")


def _run_mythos_mode(args) -> None:
    """Dispatch to the mythos package."""
    from mythos import load_mythos_config, run_mythos

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
        print(f"\nNo CLI calls will be made. Use without --dry-run to execute.")
        return

    asyncio.run(run_mythos(args.topic, cfg))


if __name__ == "__main__":
    main()
