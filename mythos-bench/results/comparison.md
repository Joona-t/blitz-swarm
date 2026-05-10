# mythos-bench — comparison

_Generated 2026-05-10 20:27 from mythos-bench/results/runs.jsonl (4 run(s))._

## All runs

| Task | Mode | Verdict | Cost | Wall | Inv | Rounds | Verifier | Primary | Adversarial |
|---|---|---|---|---|---|---|---|---|---|
| binary_search | mythos | `passed` | $1.07 | 203s | 4 | 1 | pass | 9p/0f/0e | — |
| binary_search | consensus | `passed` | $1.43 | 481s | 11 | 2 | no | 31p/0f/0e | — |
| merge_sort | mythos | `passed` | $1.56 | 294s | 4 | 1 | pass | 7p/0f/0e | — |
| monetary_decimal | mythos | `passed` | $1.50 | 256s | 3 | 1 | pass | 12p/0f/0e | 17p/0f/0e |

## Head-to-head per task

### binary_search

- **Mythos**: passed | $1.07 | 203s | 4 invocations | rounds=1
- **Consensus**: passed | $1.43 | 481s | 11 invocations | rounds=2
- Cost delta: Mythos -25% vs consensus
