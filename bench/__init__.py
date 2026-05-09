"""Blitz-Swarm bench harness — Phase 0 of the v0.2 frontier upgrade.

The bench is the fitness signal that gates every later phase. Without it,
mechanism A/B tests are theater, GEPA optimizes against noise, and the
recursive self-improvement meta-loop has no closed-form gradient.

Public API:
    load_slate(path, *, split=None, seed=42) -> list[BenchPrompt] | tuple[...]
    score(swarm_config, slate, *, seed=42, backend="cli") -> BenchScore
    regression_check(baseline, candidate, *, bound=0.3) -> list[str]

See:
    bench/slate_v1.toml      — canonical 30-prompt slate
    bench/runner.py          — orchestrates a benchmark run
    bench/detectors.py       — rule-based MAST failure-mode detectors
    bench/mast_regression.py — pytest cases injecting the 14 MAST failure modes
    bench/parallel_vs_sequential.py — paired comparison (Shen 2603.29632)
    bench/stats.py           — paired t-test, Cohen's d, bootstrap CI
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

__all__ = [
    "BenchPrompt",
    "BenchScore",
    "BenchSlate",
    "load_slate",
]


@dataclass(slots=True, frozen=True)
class BenchPrompt:
    """One row of `bench/slate_v1.toml`."""

    id: str
    tier: str             # "easy" | "medium" | "hard"
    domain: str           # "technical" | "open" | "adversarial" | "multi" | "compositional"
    parallelizable: bool
    tool_heavy: bool
    budget_usd: float
    expected_coverage: tuple[str, ...]
    text: str

    def coverage_keywords(self) -> list[str]:
        return list(self.expected_coverage)


@dataclass(slots=True, frozen=True)
class BenchSlate:
    """A loaded slate file plus metadata."""

    schema_version: int
    slate_id: str
    created_utc: str
    prompts: tuple[BenchPrompt, ...]
    sha256: str           # of the underlying file bytes

    def __len__(self) -> int:
        return len(self.prompts)

    def by_id(self, prompt_id: str) -> BenchPrompt:
        for p in self.prompts:
            if p.id == prompt_id:
                return p
        raise KeyError(prompt_id)

    def filter(
        self,
        *,
        tiers: Iterable[str] | None = None,
        domains: Iterable[str] | None = None,
        ids: Iterable[str] | None = None,
        parallelizable: bool | None = None,
    ) -> list[BenchPrompt]:
        out: list[BenchPrompt] = []
        tiers_s = set(tiers) if tiers is not None else None
        domains_s = set(domains) if domains is not None else None
        ids_s = set(ids) if ids is not None else None
        for p in self.prompts:
            if tiers_s is not None and p.tier not in tiers_s:
                continue
            if domains_s is not None and p.domain not in domains_s:
                continue
            if ids_s is not None and p.id not in ids_s:
                continue
            if parallelizable is not None and p.parallelizable != parallelizable:
                continue
            out.append(p)
        return out


@dataclass(slots=True)
class BenchScore:
    """Aggregated bench result for one swarm config."""

    aggregate: float                      # 0-10 weighted across dims
    per_dim: dict[str, float]             # {"coverage": 7.2, ...}
    per_task: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    seed: int = 42
    slate_sha256: str = ""
    config_sha256: str = ""

    def beats(self, baseline: "BenchScore", *, threshold: float = 0.4) -> bool:
        """True if aggregate exceeds baseline by `threshold`."""
        return (self.aggregate - baseline.aggregate) >= threshold


def load_slate(path: str | Path = "bench/slate_v1.toml") -> BenchSlate:
    """Load a TOML slate file. Returns BenchSlate."""
    p = Path(path)
    raw_bytes = p.read_bytes()
    raw = tomllib.loads(raw_bytes.decode("utf-8"))
    prompts = tuple(
        BenchPrompt(
            id=row["id"],
            tier=row["tier"],
            domain=row["domain"],
            parallelizable=row["parallelizable"],
            tool_heavy=row["tool_heavy"],
            budget_usd=float(row["budget_usd"]),
            expected_coverage=tuple(row["expected_coverage"]),
            text=row["text"],
        )
        for row in raw["prompt"]
    )
    return BenchSlate(
        schema_version=int(raw.get("schema_version", 1)),
        slate_id=raw.get("slate_id", "v1"),
        created_utc=raw.get("created_utc", ""),
        prompts=prompts,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
