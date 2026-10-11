# Roadmap (after v0.3.0, 2026-10-11)

Ordered by what unblocks the most. Items 1–3 need live tokens on the user's own CLI subscription (rule #10 holds: no paid API). The 2026-03-14 roadmap is superseded; its still-open items are folded in below.

## Next

1. **Live bench slate run.** `python -m bench.runner --budget …` on the 30-prompt `bench/slate_v1.toml` for at least one backend, n≥20, with the MAST detectors now actually running (bench-01 fixed). Produces the first `bench/runs/<id>/summary.json` with `blitz_swarm_version = 0.3.0`. Until this exists, every quality claim stays "not measured" (`docs/CLAIMS_AND_EVIDENCE.md` rows 22, 33).

2. **Regenerate mythos-bench** on the v0.3 defaults (planner/verifier = `fable`, executors = `haiku`) and on `--backend codex`, same three tasks, so `mythos-bench/results/comparison.md` stops being a 2026-05-10 artefact. Add the three deferred probes from ITER-MYTHOS-003 (non-local invariant, IEEE-754 boundary, debug-from-trace).

3. **RQ1 tools on/off.** Consensus roles are tool-less (`AgentCall.tools = None`). Run the same slate with researcher roles allowed a read-only tool set via the per-call field and compare quality/cost. This is the "decide the tool question" item from the audit.

4. **research-swarm search-server port.** research-swarm's MCP web-search server is the one capability consensus mode lacks (limitation 12). Port it as an optional tool backend behind `AgentCall.tools`, off by default.

5. **MCP surface for blitz-swarm.** Expose `run_swarm` / `run_mythos` / `--list-models` as an MCP server so other agents (Sparky, Claude Code) can call the swarm; separate plan, recommended by the audit.

6. **Agents-make-tools design.** The audit's Part 4 idea 1: let a Mythos executor emit a tool spec that the verifier validates and later runs register. Design doc first (research → plan), not code.

## Housekeeping

7. **WP5 landing check.** `pyproject.toml` version 0.3.0 + extras, `.gitignore`, CI green on 3.12/3.13.
8. **Finish or archive the upgrade job.** Resume `jobs/` from its last healthy cell or archive the worktree with a note; stop describing it as a working loop until one run completes.
9. **Public `main`.** Merge `fable/audit-remediation` → `main`; public `main` was 5 months behind at audit time.
10. **Gemini adapter** verification on a machine with the CLI installed; fix `blitz.toml` placeholder ids.

## Still speculative

- Cross-provider judge ensemble (heterogeneous judges) once more than one provider is routinely available.
- Adaptive `max_rounds` from topic complexity (needs the bench data from item 1 first).
- Turning `[evolve]` on: GEPA against the live slate (item 1 is the prerequisite — there is no fitness signal without it).
- CRDT blackboard conflict resolution — never started, no evidence it is needed; stays listed only so nobody rediscovers it.
