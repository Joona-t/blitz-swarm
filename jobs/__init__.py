"""Long-running job drivers for the blitz-swarm v0.next upgrade.

This package holds resumable, budget-aware drivers that orchestrate the
existing blitz-swarm surfaces (consensus orchestrator, research-swarm,
mythos, bench) into multi-phase jobs that can span hours and survive
interruption.

Public surface:
    blitz_upgrade.JobState     — load/save/checkpoint the phase manifest
    blitz_upgrade.BudgetTracker — cumulative-token ceiling enforcement
    blitz_upgrade.main         — CLI entrypoint for the 10M-token upgrade job
"""

from __future__ import annotations

__all__ = ["blitz_upgrade"]
