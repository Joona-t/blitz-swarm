"""Comparison arms for the bench (Phase 1, WP1-3).

Every arm is an async function with the runner's `swarm_fn` signature and
returns `(report_path, rounds, record)`:

  S   the swarm as shipped (orchestrator.run_swarm), untouched
  B0  one researcher call
  B1  one agent revising its own work: draft -> critique -> revise, repeated
      until its cumulative OUTPUT tokens reach the budget (min 3, max 9 calls)
  B2  N independent drafts; a judge from another family picks one
  B3  N independent drafts; one merge call that must keep every distinct claim

Fairness rules (docs/audit/2026-11-11/PHASE1-RESEARCH.md §2):
- every generation call in B0-B3 uses role "researcher", the swarm's own
  researcher system prompt, the same provider and tier, no tools, one turn;
- only B2's selection runs on the judge provider;
- every call is logged to <run_dir>/calls/<prompt_key>.jsonl with tokens,
  wall-clock, cost and provider, so any budget axis can be plotted later.

`record` mirrors the metrics.jsonl fields the runner reads, so cost and
tokens for baseline arms come from the call log, not from a topic match.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

ARMS = ("S", "B0", "B1", "B2", "B3")
B1_MIN_CALLS, B1_MAX_CALLS = 3, 9

REPORT_TASK = (
    "Write a complete, well-organised research report in Markdown answering the task below. "
    "Explain mechanisms, give concrete details, and state limits and open questions. Cite a "
    "source only when you are confident it exists, and say so plainly when a claim is unsourced.\n\n"
    "## Task\n{topic}"
)
CRITIQUE_TASK = (
    "Below is your current draft for the task. Review it as its strictest reader: list the factual "
    "errors, unsupported claims, missing parts of the task, vague passages and contradictions. "
    "Be specific and brief; do not rewrite the draft.\n\n## Task\n{topic}\n\n## Draft\n{draft}"
)
REVISE_TASK = (
    "Revise your draft using the review. Fix every valid point, keep what was right, and return the "
    "full improved report in Markdown (not a diff, not commentary).\n\n## Task\n{topic}\n\n"
    "## Draft\n{draft}\n\n## Review\n{critique}"
)
MERGE_TASK = (
    "Below are {n} independent reports on the same task. Merge them into ONE report in Markdown. "
    "Keep every distinct claim, detail and caveat that appears in any of them; where they disagree, "
    "keep both positions and say they disagree. Do not summarise content away and do not add new "
    "claims.\n\n## Task\n{topic}\n\n{drafts}"
)


@dataclass
class ArmContext:
    """Everything an arm needs; built once per bench run."""
    run_dir: Path
    cfg: object                       # BlitzConfig
    backend: object                   # AgentBackend (RoleRouter when roles are pinned)
    backend_id: str | None = None
    output_token_budget: int = 0      # B1 stop rule; 0 = stop at B1_MIN_CALLS
    n_drafts: int = 0                 # B2/B3; 0 = the planner's researcher count
    timeout_s: int = 900              # per model call (bench --arm-call-timeout)
    prompt_timeout_s: int = 0         # the runner's per-prompt limit; 0 = none. No call starts after it
    allow_same_family: bool = False
    system_prompt: str = ""           # researcher system prompt; "" = load the swarm's
    judge_cache: dict = field(default_factory=dict)
    budget_by_key: dict = field(default_factory=dict)   # prompt_key -> B1 output-token budget (per tier)


def prompt_key(topic: str) -> str:
    return hashlib.sha256(topic.encode("utf-8")).hexdigest()[:12]


class _CallLog:
    """One JSONL row per model call, plus running totals."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.gen_output_tokens = 0     # generation only (excludes judge calls)
        self.cost_usd: float | None = None
        self.cost_unknown = 0
        self.providers: set[str] = set()
        self.truncated = 0             # calls with finish_reason == "length"
        self.errored = 0               # calls that returned an error (their tokens are still counted above)
        # Generation tokens from calls that ended normally: no error and not
        # cut at the token limit. This is what B1's budget is matched on, so a
        # runaway or an all-reasoning reply is a cost, not budget.
        self.finished_gen_output_tokens = 0
        self.started = time.monotonic()

    def add(self, kind: str, result) -> None:
        self.calls += 1
        self.input_tokens += int(result.input_tokens or 0)
        self.output_tokens += int(result.output_tokens or 0)
        if kind != "judge":
            self.gen_output_tokens += int(result.output_tokens or 0)
        if isinstance(result.cost_usd, (int, float)):
            self.cost_usd = (self.cost_usd or 0.0) + float(result.cost_usd)
        else:
            self.cost_unknown += 1
        meta = result.metadata or {}
        provider = meta.get("routed_provider") or result.backend_id
        self.providers.add(str(provider))
        # finish_reason "length" = the reply was cut at the token limit; without
        # it in the log a truncated report looks like a complete one.
        finish_reason = meta.get("finish_reason") or meta.get("done_reason")   # openai_http | ollama_http
        if finish_reason == "length":
            self.truncated += 1
        if result.errored:
            self.errored += 1
        elif kind != "judge" and finish_reason != "length":
            self.finished_gen_output_tokens += int(result.output_tokens or 0)
        row = {
            "kind": kind, "provider": provider, "model": result.model,
            "input_tokens": int(result.input_tokens or 0), "output_tokens": int(result.output_tokens or 0),
            "elapsed_s": round(float(result.elapsed_s or 0), 1), "cost_usd": result.cost_usd,
            "cost_policy": result.cost_policy, "errored": bool(result.errored), "error": result.error,
            "finish_reason": finish_reason, "thinking_chars": meta.get("thinking_chars"),
            "reasoning_tokens": meta.get("reasoning_tokens"),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def record(self, arm: str, topic: str, started: float) -> dict:
        return {
            "run_id": f"{arm}_{prompt_key(topic)}", "topic": topic, "timestamp": time.time(), "arm": arm,
            "total_cost_usd": self.cost_usd, "cost_incomplete": self.cost_unknown > 0,
            "total_input_tokens": self.input_tokens, "total_output_tokens": self.output_tokens,
            "generation_output_tokens": self.gen_output_tokens, "calls": self.calls,
            "finished_generation_output_tokens": self.finished_gen_output_tokens,
            "truncated_calls": self.truncated, "errored_calls": self.errored,
            "providers": sorted(self.providers), "total_wall_clock_s": round(time.monotonic() - started, 1),
            "consensus_reached": False, "rounds_to_consensus": 0,
        }


class ArmError(RuntimeError):
    pass


def _researcher_setup(ctx: ArmContext, topic: str) -> tuple[str, int]:
    """(system prompt, researcher count) from the swarm's own heuristic plan,
    so baselines get the same role prompt and the same N. No model call."""
    if ctx.system_prompt and ctx.n_drafts:
        return ctx.system_prompt, ctx.n_drafts
    from agents import plan_agents
    agents = plan_agents(topic, use_llm=False)
    researchers = [a for a in agents if a.role == "researcher"]
    system = ctx.system_prompt or (researchers[0].system_prompt if researchers else "")
    return system, ctx.n_drafts or max(2, len(researchers))


def _call_timeout(ctx: ArmContext, log: _CallLog, kind: str) -> int:
    """Seconds this call may take. Raises ArmError once the prompt's time
    limit has passed: the runner stops waiting at that limit, but it cannot
    stop this thread, so the arm stops itself instead of calling on."""
    if not ctx.prompt_timeout_s:
        return ctx.timeout_s
    left = ctx.prompt_timeout_s - (time.monotonic() - log.started)
    if left <= 0:
        raise ArmError(f"{kind} call not started: prompt time limit of {ctx.prompt_timeout_s}s reached")
    return max(1, min(ctx.timeout_s, int(left)))


def _generate(ctx: ArmContext, log: _CallLog, kind: str, prompt: str, system: str) -> str:
    """One no-tools researcher-role call returning plain text; raises ArmError.

    An empty reply is asked for once more (a thinking model can spend the
    whole request on reasoning). Both calls are logged and counted. Other
    errors (HTTP, timeout) are not retried."""
    from backends import AgentCall
    from backends.models import resolve_model
    resolved = resolve_model("researcher", ctx.backend_id or ctx.cfg.backend.default, ctx.cfg)
    for attempt in (1, 2):
        result = ctx.backend.call(AgentCall(
            role=f"bench_{kind}", model_role="researcher", prompt=prompt, system_prompt=system,
            model=resolved.model, reasoning_effort=resolved.effort, timeout_s=_call_timeout(ctx, log, kind),
            cwd=ctx.run_dir / "scratch", tools=None, max_turns=1,
        ))
        log.add(kind, result)
        text = (result.text or "").strip()
        if text and not result.errored:
            return text
        empty_reply = not result.errored or "empty content" in (result.error or "")
        if attempt == 1 and empty_reply:
            continue
        raise ArmError(f"{kind} call failed: {result.error or 'empty output'}"
                       + (" (after one retry)" if attempt == 2 else ""))
    raise AssertionError("unreachable")


def _finish(ctx: ArmContext, arm: str, topic: str, report: str, drafts: list[str],
            log: _CallLog, started: float):
    key = prompt_key(topic)
    out = ctx.run_dir / "reports" / f"{key}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    if drafts:
        ddir = ctx.run_dir / "drafts" / key
        ddir.mkdir(parents=True, exist_ok=True)
        for i, d in enumerate(drafts):
            (ddir / f"draft_{i}.md").write_text(d, encoding="utf-8")
    rounds = [[{"role": "researcher", "agent_id": f"{arm.lower()}_draft_{i}", "findings": d,
                "key_points": [], "confidence": 0.0} for i, d in enumerate(drafts or [report])]]
    return out, rounds, log.record(arm, topic, started)


def _log_for(ctx: ArmContext, topic: str) -> _CallLog:
    path = ctx.run_dir / "calls" / f"{prompt_key(topic)}.jsonl"
    if path.exists():
        path.unlink()          # a re-run of this prompt starts a fresh log
    return _CallLog(path)


# ---------------------------------------------------------------------------
# Arms (sync bodies; the async wrappers run them in a thread)
# ---------------------------------------------------------------------------


def run_b0(ctx: ArmContext, topic: str):
    started = time.monotonic()
    log = _log_for(ctx, topic)
    system, _ = _researcher_setup(ctx, topic)
    report = _generate(ctx, log, "draft", REPORT_TASK.format(topic=topic), system)
    return _finish(ctx, "B0", topic, report, [], log, started)


def run_b1(ctx: ArmContext, topic: str):
    started = time.monotonic()
    log = _log_for(ctx, topic)
    system, _ = _researcher_setup(ctx, topic)
    draft = _generate(ctx, log, "draft", REPORT_TASK.format(topic=topic), system)
    # One critique+revise cycle is the minimum (3 calls). Further cycles run
    # while generation output tokens are under budget and calls allow.
    while True:
        critique = _generate(ctx, log, "critique", CRITIQUE_TASK.format(topic=topic, draft=draft), system)
        draft = _generate(ctx, log, "revise",
                          REVISE_TASK.format(topic=topic, draft=draft, critique=critique), system)
        if (log.calls - log.errored) + 2 > B1_MAX_CALLS:      # a retried empty reply is not a cycle
            break
        if log.finished_gen_output_tokens >= ctx.budget_by_key.get(prompt_key(topic), ctx.output_token_budget):
            break
    return _finish(ctx, "B1", topic, draft, [], log, started)


def _drafts(ctx: ArmContext, log: _CallLog, topic: str, system: str, n: int) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor
    prompt = REPORT_TASK.format(topic=topic)
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_generate, ctx, log, "draft", prompt, system) for _ in range(n)]
        drafts, errors = [], []
        for f in futures:
            try:
                drafts.append(f.result())
            except ArmError as exc:
                errors.append(str(exc))
    if not drafts:
        raise ArmError(f"all {n} drafts failed: {errors[:2]}")
    return drafts


def run_b2(ctx: ArmContext, topic: str):
    from bench import judge as J
    from mechanisms.selector_synth import bt_mle
    started = time.monotonic()
    log = _log_for(ctx, topic)
    J.assert_cross_family(ctx.cfg, backend_id=ctx.backend_id, allow_same=ctx.allow_same_family)
    system, n = _researcher_setup(ctx, topic)
    drafts = _drafts(ctx, log, topic, system, n)
    best = 0
    if len(drafts) > 1:
        ask = J.make_ask(ctx.backend, ctx.cfg, backend_id=ctx.backend_id,
                         scratch_dir=ctx.run_dir / "scratch", on_result=log.add,
                         timeout_s=ctx.cfg.swarm.judge_timeout_seconds or 300)
        ids = [str(i) for i in range(len(drafts))]
        wins: dict[tuple[str, str], float] = {}
        for i in range(len(drafts)):
            for j in range(i + 1, len(drafts)):
                _call_timeout(ctx, log, "judge")          # stop judging once the prompt's time is up
                v = J.compare(topic, drafts[i], drafts[j], ask, cache=ctx.judge_cache)
                # criteria-level wins feed Bradley-Terry; ties split
                wins[(ids[i], ids[j])] = wins.get((ids[i], ids[j]), 0.0) + v["x_wins"] + v["ties"] / 2
                wins[(ids[j], ids[i])] = wins.get((ids[j], ids[i]), 0.0) + v["y_wins"] + v["ties"] / 2
        scores = bt_mle(ids, wins)
        best = max(range(len(drafts)), key=lambda k: (scores.get(ids[k], 0.0), -k))
    return _finish(ctx, "B2", topic, drafts[best], drafts, log, started)


def run_b3(ctx: ArmContext, topic: str):
    started = time.monotonic()
    log = _log_for(ctx, topic)
    system, n = _researcher_setup(ctx, topic)
    drafts = _drafts(ctx, log, topic, system, n)
    if len(drafts) == 1:
        merged = drafts[0]
    else:
        blocks = "\n\n".join(f"<<<REPORT {i + 1}>>>\n{d}\n<<<END REPORT {i + 1}>>>" for i, d in enumerate(drafts))
        merged = _generate(ctx, log, "merge", MERGE_TASK.format(n=len(drafts), topic=topic, drafts=blocks), system)
    return _finish(ctx, "B3", topic, merged, drafts, log, started)


_SYNC = {"B0": run_b0, "B1": run_b1, "B2": run_b2, "B3": run_b3}


def make_arm(name: str, ctx: ArmContext | None):
    """The `swarm_fn` for arm `name`. "S" returns run_swarm itself."""
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; choose from {', '.join(ARMS)}")
    if name == "S":
        from orchestrator import run_swarm
        return run_swarm
    if ctx is None:
        raise ValueError(f"arm {name} needs an ArmContext")
    body = _SYNC[name]

    async def arm_fn(topic: str, **_ignored):
        return await asyncio.to_thread(body, ctx, topic)

    arm_fn.__name__ = f"arm_{name}"
    return arm_fn


def tier_budgets(run_dir: Path, slate) -> dict[str, int]:
    """Median per tier of an S run's normally finished generation tokens
    (results.jsonl): the B1 budget. Errored / timed-out prompts are excluded.
    Rows from before that field existed fall back to all output tokens."""
    import statistics
    by_tier: dict[str, list[int]] = {}
    for line in (Path(run_dir) / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tokens = row.get("finished_generation_output_tokens") or row.get("output_tokens")
        if row.get("error") or row.get("timeout") or not tokens:
            continue
        try:
            tier = slate.by_id(row["prompt_id"]).tier
        except Exception:
            continue
        by_tier.setdefault(tier, []).append(int(tokens))
    return {t: int(statistics.median(v)) for t, v in by_tier.items()}


__all__ = ["ARMS", "ArmContext", "ArmError", "make_arm", "prompt_key", "tier_budgets",
           "run_b0", "run_b1", "run_b2", "run_b3", "B1_MIN_CALLS", "B1_MAX_CALLS"]
