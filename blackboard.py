"""Shared blackboard for Blitz-Swarm agent coordination.

The blackboard is the ONLY round store the orchestrator reads context from
(core-loop-02): every round's outputs are written here and read back before
the next prompt is built, so "shared live blackboard" is true in both modes.

Two implementations of one interface:

    Blackboard          Redis-backed (sub-millisecond reads/writes, findings
                        stream for durability). `[redis]` in blitz.toml.
    InMemoryBlackboard  Same interface on plain dicts; used when Redis is
                        disabled (--no-redis) or unreachable.

Interface: initialize / cleanup / finalize / get_current_round /
advance_round / write_agent_output / read_round_outputs / read_all_outputs /
set_agent_status / get_agent_status.

Redis key design:
    blackboard:meta              -> Hash {topic, started_at, current_round, status}
    blackboard:round:{n}:{id}    -> String (JSON agent output)
    blackboard:agent:{id}        -> Hash {status, role, last_updated}
    blackboard:findings          -> Stream (append-only log of all findings)
"""

from __future__ import annotations

import json
import time

BLACKBOARD_META_KEY = "blackboard:meta"
FINDINGS_STREAM_KEY = "blackboard:findings"


def _round_trip(output: dict) -> dict:
    """JSON round-trip so both implementations hand back the same shape
    (tuples -> lists, non-JSON values stringified) and no caller can mutate
    stored state through a shared reference."""
    return json.loads(json.dumps(output, default=str))


class Blackboard:
    """Redis-backed shared blackboard for the swarm."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, *, client=None):
        if client is None:
            import redis  # lazy: the in-memory mode must not need the package

            client = redis.Redis(
                host=host, port=port, db=db,
                decode_responses=True,
                socket_connect_timeout=5,
            )
        self.redis = client
        self._verify_connection()

    def _verify_connection(self):
        """Verify Redis is reachable. Raises with clear error if not."""
        try:
            self.redis.ping()
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to Redis ({exc}). Start it with:\n"
                "  docker run -d --name blitz-redis -p 6379:6379 redis:7-alpine\n"
                "Or install locally:\n"
                "  brew install redis && redis-server --daemonize yes"
            ) from exc

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
        return int(self.redis.hincrby(BLACKBOARD_META_KEY, "current_round", 1))

    # ------------------------------------------------------------------
    # Agent output read/write
    # ------------------------------------------------------------------

    def write_agent_output(self, round_n: int, agent_id: str, output: dict):
        """Write an agent's output to the blackboard for a given round."""
        key = f"blackboard:round:{round_n}:{agent_id}"
        self.redis.set(key, json.dumps(output, default=str))

        # Also append to the findings stream for durability
        self.redis.xadd(FINDINGS_STREAM_KEY, {
            "round": str(round_n),
            "agent_id": agent_id,
            "role": output.get("role", ""),
            "confidence": str(output.get("confidence", 0)),
            "quality_vote": output.get("quality_vote", "") or "",
            "findings_preview": (output.get("findings", "") or "")[:200],
        })

    def read_round_outputs(self, round_n: int) -> list[dict]:
        """Read all agent outputs for a given round (sorted by key so the
        order is deterministic across scans)."""
        outputs = []
        pattern = f"blackboard:round:{round_n}:*"
        for key in sorted(self.redis.scan_iter(pattern)):
            raw = self.redis.get(key)
            if raw:
                outputs.append(json.loads(raw))
        return outputs

    def read_all_outputs(self) -> list[dict]:
        """Read all agent outputs across all rounds, round order first."""
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


class InMemoryBlackboard:
    """The same interface on plain dicts, for --no-redis / Redis-down runs."""

    def __init__(self):
        self.meta: dict[str, str] = {}
        self.rounds: dict[int, dict[str, dict]] = {}
        self.agents: dict[str, dict[str, str]] = {}
        self.findings: list[dict] = []

    # Lifecycle
    def initialize(self, topic: str, agent_ids: list[str]):
        self.cleanup()
        self.meta = {
            "topic": topic,
            "started_at": str(time.time()),
            "current_round": "0",
            "status": "running",
        }
        for agent_id in agent_ids:
            self.set_agent_status(agent_id, "idle")

    def cleanup(self):
        self.meta.clear()
        self.rounds.clear()
        self.agents.clear()
        self.findings.clear()

    def finalize(self):
        self.meta["status"] = "complete"

    # Round management
    def get_current_round(self) -> int:
        return int(self.meta.get("current_round") or 0)

    def advance_round(self) -> int:
        new_round = self.get_current_round() + 1
        self.meta["current_round"] = str(new_round)
        return new_round

    # Agent output read/write
    def write_agent_output(self, round_n: int, agent_id: str, output: dict):
        self.rounds.setdefault(round_n, {})[agent_id] = _round_trip(output)
        self.findings.append({
            "round": str(round_n),
            "agent_id": agent_id,
            "role": output.get("role", ""),
            "confidence": str(output.get("confidence", 0)),
            "quality_vote": output.get("quality_vote", "") or "",
            "findings_preview": (output.get("findings", "") or "")[:200],
        })

    def read_round_outputs(self, round_n: int) -> list[dict]:
        return [_round_trip(o) for o in self.rounds.get(round_n, {}).values()]

    def read_all_outputs(self) -> list[dict]:
        all_outputs = []
        for r in range(1, self.get_current_round() + 1):
            all_outputs.extend(self.read_round_outputs(r))
        return all_outputs

    # Agent status
    def set_agent_status(self, agent_id: str, status: str, role: str = ""):
        entry = self.agents.setdefault(agent_id, {})
        entry["status"] = status
        entry["last_updated"] = str(time.time())
        if role:
            entry["role"] = role

    def get_agent_status(self, agent_id: str) -> str:
        return self.agents.get(agent_id, {}).get("status") or "unknown"
