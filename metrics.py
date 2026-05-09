"""Blitz-Swarm — Per-run metrics collection and analysis.

Every run produces a structured metrics record appended to metrics.jsonl.
Adapted from research-swarm's phase-based metrics for blitz-swarm's
round-based iterative consensus model.

Metrics collected per run:
  - Round success rates (ok/total per round)
  - Per-agent timing and cost (from CLI envelope)
  - Consensus tracking (rounds to consensus, ready votes, confidence)
  - Quality scores from judge (coverage, accuracy, clarity, depth — 0-10)
  - Token usage and USD cost
  - Total wall clock time
"""

import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

METRICS_PATH = Path(__file__).parent / "metrics.jsonl"

# Relative cost weights per model tier (normalized to haiku = 1)
MODEL_COST_WEIGHT = {
    "haiku": 1.0,
    "sonnet": 3.0,
    "opus": 15.0,
}


@dataclass
class RoundMetrics:
    """Metrics for a single round execution."""
    round_name: str = ""
    agents_total: int = 0
    agents_ok: int = 0
    agents_error: int = 0
    agents_timeout: int = 0
    agents_raw: int = 0
    wall_clock_s: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    ready_votes: int = 0
    total_voters: int = 0
    avg_confidence: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.agents_ok / max(self.agents_total, 1)


@dataclass
class RunMetrics:
    """Complete metrics for a single blitz-swarm run."""
    # Identity
    run_id: str = ""
    topic: str = ""
    timestamp: float = 0.0

    # Round metrics
    rounds: dict = field(default_factory=dict)  # round_name -> RoundMetrics dict

    # Consensus
    rounds_to_consensus: int = 0
    consensus_reached: bool = False
    override_applied: bool = False

    # Quality (0-10 each, from judge)
    coverage: float = 0.0
    accuracy: float = 0.0
    clarity: float = 0.0
    depth: float = 0.0
    avg_quality: float = 0.0

    # Cost
    total_wall_clock_s: float = 0.0
    total_agent_invocations: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_units: float = 0.0


class MetricsCollector:
    """Collects metrics during a blitz-swarm run, then persists to JSONL."""

    def __init__(self):
        self.run = RunMetrics()
        self._round_timers: dict[str, float] = {}
        self._agent_timings: list[dict] = []
        self._run_start: float = 0.0

    def start_run(self, run_id: str, topic: str):
        self.run = RunMetrics(
            run_id=run_id,
            topic=topic,
            timestamp=time.time(),
        )
        self._run_start = time.monotonic()

    def start_round(self, round_name: str):
        self._round_timers[round_name] = time.monotonic()

    def end_round(self, round_name: str, outputs: list[dict]):
        """Record metrics for a completed round."""
        wall_clock = time.monotonic() - self._round_timers.get(
            round_name, time.monotonic()
        )

        ok = 0
        errors = 0
        timeouts = 0
        raw = 0
        round_cost = 0.0
        round_input_tokens = 0
        round_output_tokens = 0

        for o in outputs:
            if o.get("_error"):
                msg = o.get("_error_msg", "") or o.get("findings", "")
                if "Timed out" in msg or "TIMEOUT" in msg:
                    timeouts += 1
                else:
                    errors += 1
            elif o.get("_raw"):
                raw += 1
            else:
                ok += 1
            # Accumulate cost from CLI envelope data
            round_cost += o.get("_cost_usd", 0)
            round_input_tokens += o.get("_input_tokens", 0)
            round_output_tokens += o.get("_output_tokens", 0)

        rm = RoundMetrics(
            round_name=round_name,
            agents_total=len(outputs),
            agents_ok=ok,
            agents_error=errors,
            agents_timeout=timeouts,
            agents_raw=raw,
            wall_clock_s=round(wall_clock, 1),
            cost_usd=round(round_cost, 6),
            input_tokens=round_input_tokens,
            output_tokens=round_output_tokens,
        )

        self.run.rounds[round_name] = asdict(rm)
        self.run.total_agent_invocations += len(outputs)
        self.run.total_cost_usd += round_cost
        self.run.total_input_tokens += round_input_tokens
        self.run.total_output_tokens += round_output_tokens

    def record_consensus_state(self, round_name: str, ready_count: int,
                                total_voters: int, avg_confidence: float):
        """Record consensus voting state for a round."""
        if round_name in self.run.rounds:
            self.run.rounds[round_name]["ready_votes"] = ready_count
            self.run.rounds[round_name]["total_voters"] = total_voters
            self.run.rounds[round_name]["avg_confidence"] = round(avg_confidence, 3)

    def record_quality_scores(self, judge_output: dict):
        """Extract numeric quality scores from judge agent output."""
        self.run.coverage = judge_output.get("coverage_score", 0)
        self.run.accuracy = judge_output.get("accuracy_score", 0)
        self.run.clarity = judge_output.get("clarity_score", 0)
        self.run.depth = judge_output.get("depth_score", 0)
        scores = [self.run.coverage, self.run.accuracy,
                  self.run.clarity, self.run.depth]
        nonzero = [s for s in scores if s > 0]
        self.run.avg_quality = round(sum(nonzero) / len(nonzero), 1) if nonzero else 0

    def record_consensus_result(self, rounds_to_consensus: int,
                                 consensus_reached: bool,
                                 override_applied: bool):
        self.run.rounds_to_consensus = rounds_to_consensus
        self.run.consensus_reached = consensus_reached
        self.run.override_applied = override_applied

    def record_agent_timing(self, agent_id: str, model: str,
                            elapsed_s: float, context_chars: int = 0):
        """Record per-agent timing for cost estimation."""
        self._agent_timings.append({
            "agent_id": agent_id,
            "model": model,
            "elapsed_s": round(elapsed_s, 1),
            "context_chars": context_chars,
        })
        weight = MODEL_COST_WEIGHT.get(model, 3.0)
        self.run.estimated_cost_units += context_chars * weight

    def finalize(self):
        self.run.total_wall_clock_s = round(
            time.monotonic() - self._run_start, 1
        )
        self.run.estimated_cost_units = round(self.run.estimated_cost_units)
        self.run.total_cost_usd = round(self.run.total_cost_usd, 4)

    def save(self):
        """Append metrics record to JSONL file."""
        record = asdict(self.run)
        record["_agent_timings"] = self._agent_timings
        with open(METRICS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        return record


# ---------------------------------------------------------------------------
# Analysis — compute trends and detect regressions
# ---------------------------------------------------------------------------


def load_all_metrics() -> list[dict]:
    """Load all metrics records from JSONL."""
    if not METRICS_PATH.exists():
        return []
    records = []
    with open(METRICS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def compute_rolling_avg(records: list[dict], key: str,
                        window: int = 5) -> list[float]:
    """Compute rolling average for a top-level numeric metric."""
    values = [r.get(key, 0) for r in records if isinstance(r.get(key), (int, float))]
    if len(values) < window:
        return values
    avgs = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        avgs.append(round(statistics.mean(values[start:i + 1]), 2))
    return avgs


def detect_regressions(records: list[dict], window: int = 3,
                       threshold: float = 0.15) -> list[dict]:
    """Detect metrics that regressed >threshold from rolling average.

    Checks: avg_quality, coverage, accuracy, clarity, depth,
    and per-round success rates.
    """
    if len(records) < window + 1:
        return []

    latest = records[-1]
    prior = records[-(window + 1):-1]
    regressions = []

    # Quality metrics (higher is better)
    for key in ("avg_quality", "coverage", "accuracy", "clarity", "depth"):
        current = latest.get(key, 0)
        prior_vals = [r.get(key, 0) for r in prior if isinstance(r.get(key), (int, float))]
        if not prior_vals:
            continue
        avg = statistics.mean(prior_vals)
        if avg > 0:
            delta = (current - avg) / avg
            if delta < -threshold:
                regressions.append({
                    "metric": key,
                    "current": current,
                    "rolling_avg": round(avg, 2),
                    "delta_pct": round(delta * 100, 1),
                    "direction": "dropped",
                })

    # Per-round success rates
    for round_name, round_data in latest.get("rounds", {}).items():
        if not isinstance(round_data, dict):
            continue
        total = round_data.get("agents_total", 0)
        ok = round_data.get("agents_ok", 0)
        if total == 0:
            continue
        current_rate = ok / total

        prior_rates = []
        for r in prior:
            p = r.get("rounds", {}).get(round_name, {})
            if isinstance(p, dict) and p.get("agents_total", 0) > 0:
                prior_rates.append(p["agents_ok"] / p["agents_total"])

        if not prior_rates:
            continue
        avg_rate = statistics.mean(prior_rates)
        if avg_rate > 0:
            delta = (current_rate - avg_rate) / avg_rate
            if delta < -threshold:
                regressions.append({
                    "metric": f"{round_name}_success_rate",
                    "current": round(current_rate, 2),
                    "rolling_avg": round(avg_rate, 2),
                    "delta_pct": round(delta * 100, 1),
                    "direction": "dropped",
                })

    # Cost regression (higher is worse)
    current_cost = latest.get("total_cost_usd", 0)
    prior_costs = [r.get("total_cost_usd", 0) for r in prior
                   if isinstance(r.get("total_cost_usd"), (int, float))]
    if prior_costs:
        avg_cost = statistics.mean(prior_costs)
        if avg_cost > 0:
            delta = (current_cost - avg_cost) / avg_cost
            if delta > threshold:
                regressions.append({
                    "metric": "total_cost_usd",
                    "current": round(current_cost, 4),
                    "rolling_avg": round(avg_cost, 4),
                    "delta_pct": round(delta * 100, 1),
                    "direction": "increased (worse)",
                })

    return regressions


def format_metrics_report(records: list[dict], last_n: int = 5) -> str:
    """Format a human-readable metrics report from the last N runs."""
    if not records:
        return "No metrics recorded yet."

    recent = records[-last_n:]

    lines = [
        f"## Blitz-Swarm Metrics Report ({len(records)} total runs, showing last {len(recent)})",
        "",
        "| Run | Topic | Quality | Cov | Acc | Clar | Depth | Rounds | Consensus | Wall(s) | Cost |",
        "|-----|-------|---------|-----|-----|------|-------|--------|-----------|---------|------|",
    ]

    for i, r in enumerate(recent):
        topic = r.get("topic", "?")[:25]
        quality = r.get("avg_quality", 0)
        cov = r.get("coverage", 0)
        acc = r.get("accuracy", 0)
        clar = r.get("clarity", 0)
        dep = r.get("depth", 0)
        rounds = r.get("rounds_to_consensus", 0)
        consensus = "yes" if r.get("consensus_reached") else "no"
        if r.get("override_applied"):
            consensus += "*"
        wall = r.get("total_wall_clock_s", 0)
        cost_usd = r.get("total_cost_usd", 0)
        cost_str = f"${cost_usd:.2f}" if cost_usd > 0 else "--"

        lines.append(
            f"| {len(records) - last_n + i + 1} "
            f"| {topic} "
            f"| {quality} "
            f"| {cov} "
            f"| {acc} "
            f"| {clar} "
            f"| {dep} "
            f"| {rounds} "
            f"| {consensus} "
            f"| {wall:.0f} "
            f"| {cost_str} |"
        )

    # Trends
    if len(records) >= 3:
        lines.extend(["", "### Trends (last 3 runs)"])
        last3 = records[-3:]
        for key, label in [
            ("avg_quality", "Avg Quality"),
            ("total_cost_usd", "Cost (USD)"),
            ("total_wall_clock_s", "Wall Clock (s)"),
            ("rounds_to_consensus", "Rounds to Consensus"),
        ]:
            vals = [r.get(key, 0) for r in last3]
            if all(isinstance(v, (int, float)) for v in vals):
                trend = "->"
                if vals[-1] > vals[0] * 1.05:
                    if key in ("total_cost_usd", "total_wall_clock_s", "rounds_to_consensus"):
                        trend = "^ (worse)"
                    else:
                        trend = "^"
                elif vals[-1] < vals[0] * 0.95:
                    if key in ("total_cost_usd", "total_wall_clock_s", "rounds_to_consensus"):
                        trend = "v (better)"
                    else:
                        trend = "v"
                fmt = ".2f" if key == "total_cost_usd" else ".1f"
                lines.append(
                    f"  {label}: {vals[0]:{fmt}} -> {vals[1]:{fmt}} -> {vals[2]:{fmt}} {trend}"
                )

    # Regressions
    regressions = detect_regressions(records)
    if regressions:
        lines.extend(["", "### Regressions Detected"])
        for reg in regressions:
            lines.append(
                f"  {reg['metric']}: {reg['current']} "
                f"(avg: {reg['rolling_avg']}, {reg['delta_pct']:+.1f}%)"
            )

    return "\n".join(lines)
