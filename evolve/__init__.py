"""Phase 3 — Recursive self-improvement. **EXPERIMENTAL, UNWIRED.**

Nothing in this package is called by the orchestrator or the bench. It is
gated: every entry point calls `require_enabled()` and refuses with a clear
message unless `blitz.toml` has `[evolve] enabled = true` (mechanisms-01).

Three orthogonal evolution loops, all gated by the bench from Phase 0:

  evolve/gepa_adapter.py   Reflective prompt evolution (GEPA, arXiv 2507.19457).
                           Wraps the standalone gepa-ai/gepa library.
  evolve/aflow_search.py   MCTS over swarm graphs (AFlow, arXiv 2410.10762).
                           Mutates topology, role count, debate rounds.
  evolve/meta_loop.py      Config-patch proposer + bench-validated auto-merge.
                           Reads insights tagged `meta:` and proposes blitz.toml
                           edits gated by Cohen's d / regression / cost / p-value.

Recursion is hard-capped at L3 (humans audit). No L4 — see PHASE_3_RECURSION.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import load_config  # noqa: E402

DISABLED_MESSAGE = (
    "{what} is EXPERIMENTAL and disabled: set `[evolve] enabled = true` in "
    "blitz.toml to run it (it is unwired from the orchestrator/bench and may "
    "spend tokens on evolution loops)."
)


class EvolveDisabled(RuntimeError):
    """Raised by evolve/ and heterogeneity/ entry points while [evolve] enabled = false."""


def require_enabled(cfg=None, *, what: str = "evolve/") -> None:
    """Refuse unless `[evolve] enabled = true`. `cfg` defaults to blitz.toml."""
    if cfg is None:
        cfg = load_config()
    if not getattr(getattr(cfg, "evolve", None), "enabled", False):
        raise EvolveDisabled(DISABLED_MESSAGE.format(what=what))
