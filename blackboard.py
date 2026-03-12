"""Redis-backed blackboard for Blitz-Swarm agent coordination.

The blackboard is the shared memory surface that all agents read from and
write to. It uses Redis for sub-millisecond reads/writes with built-in
event broadcasting.

Redis key design:
    blackboard:meta              -> Hash {topic, started_at, current_round, status}
    blackboard:round:{n}:{id}    -> String (JSON agent output)
    blackboard:agent:{id}        -> Hash {status, role, last_updated}
    blackboard:findings          -> Stream (append-only log of all findings)
    memory:writes                -> Stream (write queue for MemoryWriter)
"""

import json
import time

import redis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CONTEXT_TOKENS = 2000
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * 4  # rough chars-to-tokens ratio

BLACKBOARD_META_KEY = "blackboard:meta"
FINDINGS_STREAM_KEY = "blackboard:findings"
MEMORY_WRITES_STREAM_KEY = "memory:writes"


class Blackboard:
    """Redis-backed shared blackboard for the swarm."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis = redis.Redis(
            host=host, port=port, db=db,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        self._verify_connection()

    def _verify_connection(self):
        """Verify Redis is reachable. Raises with clear error if not."""
        try:
            self.redis.ping()
        except redis.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Redis. Start it with:\n"
                "  docker run -d --name blitz-redis -p 6379:6379 redis:7-alpine\n"
                "Or install locally:\n"
                "  brew install redis && redis-server --daemonize yes"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, topic: str, agent_ids: list[str]):
        """Initialize the blackboard for a new swarm run.

        Clears any prior state and sets up metadata.
        """
        self.cleanup()

        self.redis.hset(BLACKBOARD_META_KEY, mapping={
            "topic": topic,
            "started_at": str(time.time()),
            "current_round": "0",
            "status": "running",
        })

        for agent_id in agent_ids:
            self.set_agent_status(agent_id, "idle")

    def cleanup(self):
        """Remove all blackboard keys from prior runs."""
        pipe = self.redis.pipeline()
        for key in self.redis.scan_iter("blackboard:*"):
            pipe.delete(key)
        pipe.delete(MEMORY_WRITES_STREAM_KEY)
        pipe.execute()

    def finalize(self):
        """Mark the swarm run as complete."""
        self.redis.hset(BLACKBOARD_META_KEY, "status", "complete")

    # ------------------------------------------------------------------
    # Round management
    # ------------------------------------------------------------------

    def get_current_round(self) -> int:
        val = self.redis.hget(BLACKBOARD_META_KEY, "current_round")
        return int(val) if val else 0

    def advance_round(self) -> int:
        """Increment and return the new round number."""
        new_round = self.redis.hincrby(BLACKBOARD_META_KEY, "current_round", 1)
        return new_round

    # ------------------------------------------------------------------
    # Agent output read/write
    # ------------------------------------------------------------------

    def write_agent_output(self, round_n: int, agent_id: str, output: dict):
        """Write an agent's output to the blackboard for a given round."""
        key = f"blackboard:round:{round_n}:{agent_id}"
        self.redis.set(key, json.dumps(output))

        # Also append to the findings stream for durability
        self.redis.xadd(FINDINGS_STREAM_KEY, {
            "round": str(round_n),
            "agent_id": agent_id,
            "role": output.get("role", ""),
            "confidence": str(output.get("confidence", 0)),
            "quality_vote": output.get("quality_vote", ""),
            "findings_preview": (output.get("findings", ""))[:200],
        })

    def read_round_outputs(self, round_n: int) -> list[dict]:
        """Read all agent outputs for a given round."""
        outputs = []
        pattern = f"blackboard:round:{round_n}:*"
        for key in self.redis.scan_iter(pattern):
            raw = self.redis.get(key)
            if raw:
                outputs.append(json.loads(raw))
        return outputs

    def read_all_outputs(self) -> list[dict]:
        """Read all agent outputs across all rounds."""
        current_round = self.get_current_round()
        all_outputs = []
        for r in range(1, current_round + 1):
            all_outputs.extend(self.read_round_outputs(r))
        return all_outputs

    # ------------------------------------------------------------------
    # Agent status (LWW-Register pattern)
    # ------------------------------------------------------------------

    def set_agent_status(self, agent_id: str, status: str, role: str = ""):
        """Set agent status using Last-Writer-Wins timestamp."""
        key = f"blackboard:agent:{agent_id}"
        mapping = {
            "status": status,
            "last_updated": str(time.time()),
        }
        if role:
            mapping["role"] = role
        self.redis.hset(key, mapping=mapping)

    def get_agent_status(self, agent_id: str) -> str:
        key = f"blackboard:agent:{agent_id}"
        return self.redis.hget(key, "status") or "unknown"

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_context_string(self, up_to_round: int, for_role: str = "") -> str:
        """Build a context string from blackboard state for agent injection.

        Collects outputs from all rounds up to (and including) up_to_round,
        filtered and formatted based on the target role.
        """
        all_outputs = []
        for r in range(1, up_to_round + 1):
            all_outputs.extend(self.read_round_outputs(r))

        return _format_context(all_outputs, for_role)

    # ------------------------------------------------------------------
    # Memory write queue
    # ------------------------------------------------------------------

    def queue_memory_write(self, payload: dict):
        """Append a memory write to the Redis Stream for the MemoryWriter."""
        self.redis.xadd(MEMORY_WRITES_STREAM_KEY, {
            "type": payload.get("type", "task_complete"),
            "payload": json.dumps(payload),
            "timestamp": str(time.time()),
        })

    def consume_memory_writes(self, last_id: str = "0-0", count: int = 10) -> list[tuple[str, dict]]:
        """Read pending memory writes from the stream.

        Returns list of (stream_id, payload_dict) tuples.
        """
        entries = self.redis.xread(
            {MEMORY_WRITES_STREAM_KEY: last_id},
            count=count,
            block=100,
        )
        results = []
        if entries:
            for stream_name, messages in entries:
                for msg_id, fields in messages:
                    payload = json.loads(fields.get("payload", "{}"))
                    results.append((msg_id, payload))
        return results


# ---------------------------------------------------------------------------
# Context formatting (shared between blackboard and in-memory modes)
# ---------------------------------------------------------------------------


def _format_context(outputs: list[dict], for_role: str = "") -> str:
    """Format agent outputs into a context string for prompt injection.

    Filters by role relevance and caps at MAX_CONTEXT_CHARS.
    """
    sections = []

    researcher_outputs = [o for o in outputs if o.get("role") == "researcher"]
    critic_outputs = [o for o in outputs if o.get("role") == "critic"]
    fact_checker_outputs = [o for o in outputs if o.get("role") == "fact_checker"]
    judge_outputs = [o for o in outputs if o.get("role") == "quality_judge"]

    if researcher_outputs:
        sections.append("### Researcher Findings")
        for o in researcher_outputs:
            agent_id = o.get("agent_id", "unknown")
            findings = o.get("findings", "")[:800]
            key_points = o.get("key_points", [])
            conf = o.get("confidence", 0)
            sections.append(f"**{agent_id}** (confidence: {conf:.0%}):")
            sections.append(findings)
            if key_points:
                sections.append("Key points: " + "; ".join(key_points[:5]))
            sections.append("")

    if for_role in ("critic", "fact_checker", "quality_judge", "synthesizer", "researcher"):
        if critic_outputs:
            sections.append("### Critic Feedback")
            for o in critic_outputs:
                sections.append(o.get("findings", "")[:500])
                sections.append("")

        if fact_checker_outputs:
            sections.append("### Fact-Checker Verification")
            for o in fact_checker_outputs:
                sections.append(o.get("findings", "")[:500])
                sections.append("")

    if for_role == "synthesizer" and judge_outputs:
        sections.append("### Quality Judge Assessment")
        for o in judge_outputs:
            sections.append(o.get("findings", "")[:500])
            sections.append("")

    context = "\n".join(sections)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated to fit token limit]"

    return context
