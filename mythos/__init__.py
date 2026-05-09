"""Mythos Swarm — hierarchical orchestrator-executor mode for blitz-swarm.

Mythos planner decomposes a task into sub-specs + invariants. Sonnet executors
run the sub-specs in parallel. Mythos verifier gates the outputs and either
accepts or sends a replan signal. Up to N replans before failing loudly.

Distinct from blitz-swarm's flat consensus mode — this is the right shape for
long-horizon coherence tasks (legacy modernization, multi-hop security audit,
formal verification, distributed-systems debugging, compiler construction)
where a single deep planner + cheap parallel executors beats a swarm of peers.

Public surface:
    run_mythos(task, config) -> MythosResult
    load_mythos_config(path=None) -> MythosConfig
"""

from .policies import (
    MythosConfig,
    CostBudget,
    load_mythos_config,
    resolve_model,
)
from .runner import MythosResult, run_mythos

__all__ = [
    "MythosConfig",
    "CostBudget",
    "MythosResult",
    "load_mythos_config",
    "resolve_model",
    "run_mythos",
]
