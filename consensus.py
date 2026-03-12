"""Convergence logic and dissent preservation for Blitz-Swarm."""


def check_consensus(agent_outputs: list[dict]) -> bool:
    """Check if all voting agents have reached unanimous 'ready' consensus.

    Only agents that include a quality_vote field are counted as voters.
    Returns True if all voters say 'ready', False otherwise.
    Returns False if there are no voters.
    """
    voters = [o for o in agent_outputs if o.get("quality_vote") is not None]
    if not voters:
        return False
    return all(v["quality_vote"] == "ready" for v in voters)


def extract_dissent(agent_outputs: list[dict]) -> list[dict]:
    """Collect all 'needs_work' votes with their notes and agent IDs.

    Returns a list of dicts with agent_id, quality_notes, and dissent fields.
    """
    dissent = []
    for output in agent_outputs:
        if output.get("quality_vote") == "needs_work":
            dissent.append({
                "agent_id": output.get("agent_id", "unknown"),
                "role": output.get("role", "unknown"),
                "quality_notes": output.get("quality_notes", ""),
                "dissent": output.get("dissent", ""),
                "confidence": output.get("confidence", 0.0),
            })
    return dissent


def should_override_holdout(round_n: int, agent_outputs: list[dict]) -> bool:
    """Check if a single holdout should be overridden.

    Override condition: one agent says 'needs_work' after 3+ rounds
    while all other voters say 'ready'.
    """
    if round_n < 3:
        return False

    voters = [o for o in agent_outputs if o.get("quality_vote") is not None]
    if len(voters) < 2:
        return False

    needs_work = [v for v in voters if v["quality_vote"] == "needs_work"]
    return len(needs_work) == 1


def format_dissent_section(dissent: list[dict]) -> str:
    """Format dissent entries into a markdown section for the final output."""
    if not dissent:
        return ""

    lines = ["## Dissent & Open Questions", ""]
    for d in dissent:
        agent_label = f"{d['role']} ({d['agent_id']})"
        lines.append(f"### {agent_label}")
        if d.get("quality_notes"):
            lines.append(f"**Concern:** {d['quality_notes']}")
        if d.get("dissent"):
            lines.append(f"**Dissent:** {d['dissent']}")
        lines.append(f"**Confidence:** {d['confidence']:.1%}")
        lines.append("")

    return "\n".join(lines)


def format_convergence_report(all_rounds: list[list[dict]]) -> str:
    """Summarize how consensus evolved across rounds.

    all_rounds is a list where each entry is the list of agent outputs
    for that round.
    """
    lines = ["## Convergence Report", ""]

    for round_n, outputs in enumerate(all_rounds, 1):
        voters = [o for o in outputs if o.get("quality_vote") is not None]
        ready = sum(1 for v in voters if v["quality_vote"] == "ready")
        total = len(voters)
        avg_conf = (
            sum(v.get("confidence", 0) for v in voters) / total
            if total > 0
            else 0
        )
        status = "CONSENSUS" if ready == total and total > 0 else f"{ready}/{total} ready"
        lines.append(f"**Round {round_n}:** {status} | avg confidence: {avg_conf:.1%}")

    lines.append("")
    return "\n".join(lines)
