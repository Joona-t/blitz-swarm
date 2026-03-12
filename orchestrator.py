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
from datetime import datetime
from pathlib import Path

from agents import (
    AGENT_OUTPUT_SCHEMA_JSON,
    BlitzAgent,
    plan_agents,
)
from consensus import (
    check_consensus,
    extract_dissent,
    format_convergence_report,
    format_dissent_section,
    should_override_holdout,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_ROUNDS = 5
AGENT_TIMEOUT_SECONDS = 180
OUTPUT_DIR = Path(__file__).parent / "output"
MAX_CONTEXT_TOKENS = 2000
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * 4

# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------


def invoke_agent(agent: BlitzAgent, context: str, task: str) -> dict:
    """Invoke a single agent as a subprocess via the Claude CLI.

    Returns parsed JSON output from the agent, or an error dict on failure.
    """
    user_prompt = _build_user_prompt(agent, context, task)

    cmd = [
        "claude",
        "-p", user_prompt,
        "--system-prompt", agent.system_prompt,
        "--output-format", "json",
        "--model", agent.model,
        "--dangerously-skip-permissions",
    ]

    start = time.monotonic()
    agent_label = f"{agent.role}({agent.id})"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=AGENT_TIMEOUT_SECONDS,
        )

        elapsed = time.monotonic() - start
        print(f"  {agent_label} done [{elapsed:.1f}s]")

        if result.returncode != 0:
            print(f"  {agent_label} ERROR: exit code {result.returncode}")
            stderr_snippet = (result.stderr or "")[:200]
            if stderr_snippet:
                print(f"    stderr: {stderr_snippet}")
            return _error_output(agent, f"Process exited with code {result.returncode}")

        return _parse_agent_output(agent, result.stdout)

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        print(f"  {agent_label} TIMEOUT [{elapsed:.1f}s]")
        return _error_output(agent, f"Timed out after {AGENT_TIMEOUT_SECONDS}s")

    except Exception as e:
        print(f"  {agent_label} EXCEPTION: {e}")
        return _error_output(agent, str(e))


def _build_user_prompt(agent: BlitzAgent, context: str, task: str) -> str:
    """Build the user-facing prompt for an agent invocation."""
    sections = []

    sections.append(f"## Your Assigned Subtopic\n{agent.subtopic}")

    if context:
        sections.append(f"## Shared Context (current blackboard state)\n{context}")

    sections.append(f"## Task\n{task}")

    sections.append(
        "## Output Format\n"
        "Respond with a JSON object containing these fields:\n"
        "- findings (string): Your detailed analysis in markdown\n"
        "- key_points (array of strings): Most important takeaways\n"
        "- confidence (number 0-1): Your confidence in accuracy\n"
        "- gaps_identified (array of strings): Areas needing more research\n"
        "- quality_vote (string): 'ready' or 'needs_work'\n"
        "- quality_notes (string): Explanation for your vote\n"
        "- dissent (string): Any disagreements with other findings"
    )

    return "\n\n".join(sections)


def _parse_agent_output(agent: BlitzAgent, stdout: str) -> dict:
    """Parse agent stdout into a structured dict.

    Tries JSON parsing first, then falls back to extracting JSON from
    markdown code blocks if the raw output isn't valid JSON.
    """
    stdout = stdout.strip()
    if not stdout:
        return _error_output(agent, "Empty output")

    # Try direct JSON parse
    try:
        data = json.loads(stdout)
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
            data["agent_id"] = agent.id
            data["role"] = agent.role
            return data
        except json.JSONDecodeError:
            pass

    # Last resort: wrap raw text as findings
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
    print(f"\n{'='*60}")
    print(f"BLITZ-SWARM — {topic}")
    print(f"{'='*60}\n")

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

    # --- BLAST + CHECK loop ---
    for round_n in range(1, max_rounds + 1):
        if bb:
            bb.advance_round()

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

    synth_output = synth_outputs[0] if synth_outputs else None

    # --- Persist to memory ---
    _persist_to_memory(
        mem_writer, bb, topic, consensus_reached, override_applied,
        all_round_outputs, agents,
    )

    # --- Output ---
    final_doc = format_final_output(topic, all_round_outputs, synth_output)
    filepath = save_output(topic, final_doc)

    total_agents_invoked = sum(len(rnd) for rnd in all_round_outputs)
    print(f"{'='*60}")
    print(f"OUTPUT SAVED: {filepath}")
    print(f"Rounds: {len(all_round_outputs) - 1} + synthesis")
    print(f"Total agent invocations: {total_agents_invoked}")
    print(f"Consensus: {'yes' if consensus_reached else 'no'}"
          f"{' (override)' if override_applied else ''}")
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
    parser.add_argument("topic", help="The topic to research")
    parser.add_argument(
        "--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
        help=f"Maximum consensus rounds (default: {DEFAULT_MAX_ROUNDS})",
    )
    parser.add_argument(
        "--no-redis", action="store_true",
        help="Run without Redis (in-memory blackboard only)",
    )
    parser.add_argument(
        "--no-llm-plan", action="store_true",
        help="Use heuristic agent planning instead of LLM",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the agent plan without executing",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable detailed logging",
    )

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
