# MAST Regression Scoreboard — v0.1.1

**Detected: 9/14** (9 marked as detectable; 5 require orchestrator integration or LLM judgment).

Source: 14 synthetic scenarios from `bench/mast_regression.py`. Detectors live in `bench/detectors.py`.

| Code | Name | FC | Expected | Fired | Notes |
|---|---|---|---|---|---|
| FM-1.1 | Disobey Task Specification | FC1 | no | no | Requires LLM-based topic-relevance judgment; no rule-based detector. |
| FM-1.2 | Disobey Role Specification | FC1 | yes | YES | Heuristic keyword overlap with role-template vocabulary. |
| FM-1.3 | Step Repetition | FC1 | yes | YES | Jaccard 5-gram > 0.95 between rounds for same agent. |
| FM-1.4 | Loss of Conversation History | FC1 | yes | YES | Round-2+ agent received zero-length context. |
| FM-1.5 | Unaware of Termination | FC1 | no | no | Orchestrator-level failure; needs integration test. |
| FM-2.1 | Conversation Reset | FC2 | yes | YES | Non-monotone ready_votes across rounds (oscillation). |
| FM-2.2 | Fail to Ask for Clarification | FC2 | no | no | Requires ambiguity detection; orchestrator integration. |
| FM-2.3 | Task Derailment | FC2 | no | no | Synthesizer-content drift; needs LLM judgment beyond rule heuristics. |
| FM-2.4 | Information Withholding | FC2 | yes | YES | Researcher findings non-empty but key_points list empty. |
| FM-2.5 | Ignored Other Agents' Input | FC2 | yes | YES | Critic feedback present but researcher findings unchanged. |
| FM-2.6 | Reasoning-Action Mismatch | FC2 | yes | YES | Confidence < 0.4 paired with quality_vote=='ready'. |
| FM-3.1 | Premature Termination | FC3 | yes | YES | _partial flag set without _flagged_partial. |
| FM-3.2 | No or Incomplete Verification | FC3 | no | no | Roster-level check; needs metrics.jsonl `agents_used` field. |
| FM-3.3 | Incorrect Verification | FC3 | yes | YES | Judge avg > 8 with output_md_chars < 500. |

## Coverage by category

- **FC1** (3/5 detected, 3 marked detectable)
- **FC2** (4/6 detected, 4 marked detectable)
- **FC3** (2/3 detected, 2 marked detectable)

## Path to full coverage

- FM-1.1, FM-2.3 require LLM-based content judgment (Phase 1: judge_ensemble).
- FM-1.5, FM-2.2, FM-3.2 require orchestrator integration (Phase 1: roster + halting hooks in `run_swarm`).
