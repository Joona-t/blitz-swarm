"""Phase 3 — Recursive self-improvement.

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
