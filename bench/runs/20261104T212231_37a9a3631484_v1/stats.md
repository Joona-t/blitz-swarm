# Bench run 20261104T212231_37a9a3631484_v1

Started: 2026-11-04T21:22:31+00:00  →  Ended: 2026-11-04T21:28:16+00:00
Slate: v1 (sha256=f14ac96dbcf4...)
Prompts: ran 2 / 30, timed out 0, errored 0

## Functional research-quality composite (0-1) — PRIMARY

Mean composite: **0.8463**

| Prompt | Composite |
|---|---|
| s001 | 0.7298 |
| s003 | 0.9629 |

## Aggregate quality (0-10) — SECONDARY (LLM-judge)

| Dim | Mean | SD | CI95 |
|---|---|---|---|
| avg | 7.75 | 0.212 | [7.6, 7.9] |
| coverage | 6.92 | 0.354 | [6.67, 7.17] |
| accuracy | 8.585 | 0.12 | [8.5, 8.67] |
| clarity | 8.25 | 0.113 | [8.17, 8.33] |
| depth | 7.25 | 0.113 | [7.17, 7.33] |

## By tier

| Tier | N | Avg quality | Avg cost ($) |
|---|---|---|---|
| easy | 2 | 7.75 | 0.5093 |
| medium | 0 | 0 | 0 |
| hard | 0 | 0 | 0 |

## MAST flags

_No MAST flags raised (detectors ran on 2 prompt(s))._

## Paired comparison vs baseline `20261104T211647_f48cde461b11_v1`

- Pairs: 2 (functional composite, PRIMARY)
- Mean delta (this - baseline): -0.0019
- Paired t = -1.0, df = 1, p = 0.5 (student-t (incomplete beta))
- Cohen's d_z = -0.7071

## Cost

- Total: $1.0186
- Input tokens:  515,890
- Output tokens: 49,940
