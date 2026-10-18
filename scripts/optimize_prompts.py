#!/usr/bin/env python3
"""GEPA prompt-evolution driver for blitz-swarm (EXPERIMENTAL, gated).

Minimal, real, import-guarded (research-docs-02 / PHASE_3_RECURSION.md §1.3):

  * refuses unless `[evolve] enabled = true` in blitz.toml;
  * `gepa` (pip install gepa) is imported lazily — a missing package is a
    clear exit, not a traceback at import time;
  * the seed candidate is the role prompts under `prompts/<domain>/`;
  * tasks come from the bench slate (`bench.load_slate`), split 70/30;
  * the evaluator is the deterministic functional composite from
    `bench.detectors.research_quality_composite` (no LLM judge);
  * `run_fn` runs the real swarm (`orchestrator.run_swarm`) with the candidate
    prompts materialised into `prompts/<domain>/` for the duration of the run
    and the originals restored afterwards (the rollback the research doc
    describes). It spends tokens through the user's own CLI backend only
    (rule #10) — never a paid API.
  * `--dry-run` does everything except import gepa and run the swarm: it
    prints the resolved plan and exits 0.

The reflection LM is whatever `gepa` accepts for `reflection_lm`; by default
we pass a callable that routes through the configured local backend on the
`synthesizer` role tier, so no model name is hard-coded here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench import load_slate  # noqa: E402
from bench.detectors import research_quality_composite  # noqa: E402
from config import load_config  # noqa: E402
from evolve import require_enabled  # noqa: E402
from evolve.gepa_adapter import BlitzGEPAAdapter, SwarmTask, SwarmTrajectory, load_role_prompts  # noqa: E402

DEFAULT_COMPONENTS = ("researcher", "critic_factual", "critic_logical", "synthesizer")


def split_slate(slate_path: Path, *, train_frac: float = 0.7, seed: int = 42) -> tuple[list[SwarmTask], list[SwarmTask]]:
    """Deterministic train/val split of the bench slate as GEPA SwarmTasks."""
    import random

    slate = load_slate(slate_path)
    tasks = [
        SwarmTask(
            task_id=p.id, topic=p.text, domain=p.domain,
            expected_traits={"expected_subtopics": list(p.expected_coverage), "tier": p.tier},
            judge_rubric={"expected_subtopics": list(p.expected_coverage)},
        )
        for p in slate.prompts
    ]
    rng = random.Random(seed)
    rng.shuffle(tasks)
    k = max(1, int(len(tasks) * train_frac)) if len(tasks) > 1 else len(tasks)
    return tasks[:k], tasks[k:]


def seed_candidate(prompts_root: Path, domain: str, components: list[str]) -> dict[str, str]:
    files = {c: prompts_root / domain / f"{c}.md" for c in components}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"optimize_prompts: seed prompt file(s) missing: {', '.join(missing)}")
    return load_role_prompts(files)


def functional_score(traj: SwarmTrajectory, rubric: dict) -> float:
    """Deterministic score in [0, 1]: the bench's functional composite."""
    md = traj.role_outputs.get("synthesizer", "") or ""
    comp = research_quality_composite(md, {"expected_subtopics": rubric.get("expected_subtopics", [])})
    return float(comp.get("composite") or 0.0)


class PromptWorkspace:
    """Materialise a candidate into prompts/<domain>/ and restore on exit."""

    def __init__(self, prompts_root: Path, domain: str, components: list[str], backup_dir: Path):
        self.dir = prompts_root / domain
        self.components = components
        self.backup_dir = backup_dir

    def __enter__(self) -> "PromptWorkspace":
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        for c in self.components:
            shutil.copy2(self.dir / f"{c}.md", self.backup_dir / f"{c}.md")
        return self

    def write(self, candidate: dict[str, str]) -> None:
        for c, text in candidate.items():
            if c in self.components:
                (self.dir / f"{c}.md").write_text(text, encoding="utf-8")

    def __exit__(self, *exc) -> None:
        for c in self.components:
            shutil.copy2(self.backup_dir / f"{c}.md", self.dir / f"{c}.md")


def make_run_fn(workspace: PromptWorkspace, *, max_rounds: int, backend_id: str | None):
    """run_fn for BlitzGEPAAdapter: real swarm run under the candidate prompts."""
    from orchestrator import run_swarm  # lazy: pulls the whole orchestrator

    def run(task: SwarmTask, candidate: dict[str, str]) -> SwarmTrajectory:
        workspace.write(candidate)
        out_path = asyncio.run(run_swarm(
            task.topic, max_rounds=max_rounds, use_redis=False,
            **({"backend_id": backend_id} if backend_id else {}),
        ))
        if isinstance(out_path, tuple):           # (Path, rounds) contract
            out_path = out_path[0]
        md = Path(out_path).read_text(encoding="utf-8") if Path(out_path).exists() else ""
        return SwarmTrajectory(
            task=task, role_outputs={"synthesizer": md}, judge_breakdown={},
            final_score=functional_score(SwarmTrajectory(task, {"synthesizer": md}, {}, 0.0), task.judge_rubric),
        )

    return run


def make_reflection_lm(backend_id: str | None):
    """Reflection LM callable on the configured local backend (`synthesizer` tier)."""
    from backends import AgentCall, make_provider_backend
    from backends.models import resolve_model

    cfg = load_config()
    resolved = resolve_model("synthesizer", backend_id or cfg.backend.default or None, cfg)
    backend = make_provider_backend(resolved.backend_id, cfg)

    def reflect(prompt: str) -> str:
        res = backend.call(AgentCall(role="gepa_reflection", model_role="synthesizer",
                                     prompt=prompt, model=resolved.model,
                                     reasoning_effort=resolved.effort))
        return res.text or ""

    return reflect


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GEPA prompt evolution for blitz-swarm (experimental)")
    ap.add_argument("--slate", default=str(_ROOT / "bench" / "slate_v1.toml"))
    ap.add_argument("--domain", default=None, help="prompts/<domain>/ (default: [swarm] domain)")
    ap.add_argument("--components", nargs="+", default=list(DEFAULT_COMPONENTS))
    ap.add_argument("--max-metric-calls", type=int, default=None,
                    help="default: [evolve] gepa_max_metric_calls")
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--out", default=str(_ROOT / "output" / "evolve"))
    ap.add_argument("--dry-run", action="store_true", help="resolve everything, run nothing")
    args = ap.parse_args(argv)

    cfg = load_config()
    require_enabled(cfg, what="scripts/optimize_prompts.py")

    domain = args.domain or cfg.swarm.domain
    max_calls = args.max_metric_calls or cfg.evolve.gepa_max_metric_calls
    prompts_root = _ROOT / "prompts"
    seed = seed_candidate(prompts_root, domain, args.components)
    train, val = split_slate(Path(args.slate))
    run_dir = Path(args.out) / f"gepa_{int(time.time())}"

    plan = {
        "domain": domain, "components": args.components,
        "seed_chars": {k: len(v) for k, v in seed.items()},
        "train_tasks": len(train), "val_tasks": len(val),
        "max_metric_calls": max_calls, "max_rounds": args.max_rounds,
        "backend": args.backend or cfg.backend.default, "run_dir": str(run_dir),
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("dry-run: gepa not imported, swarm not run.")
        return 0

    try:
        import gepa  # type: ignore
    except ImportError:
        print("optimize_prompts: the `gepa` package is not installed. "
              "`pip install gepa` (optional extra) to run prompt evolution.", file=sys.stderr)
        return 2

    run_dir.mkdir(parents=True, exist_ok=True)
    with PromptWorkspace(prompts_root, domain, args.components, run_dir / "seed_backup") as ws:
        adapter = BlitzGEPAAdapter(
            run_fn=make_run_fn(ws, max_rounds=args.max_rounds, backend_id=args.backend),
            score_fn=functional_score, blitz_cfg=cfg,
        )
        result = gepa.optimize(
            seed_candidate=seed, trainset=train, valset=val, adapter=adapter,
            reflection_lm=make_reflection_lm(args.backend),
            max_metric_calls=max_calls, run_dir=str(run_dir), seed=42,
        )
    best = getattr(result, "best_candidate", None) or {}
    (run_dir / "best_candidate.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"optimize_prompts: done — best candidate written to {run_dir / 'best_candidate.json'} "
          f"(prompts/{domain}/ restored to the seed; apply the candidate by hand).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
