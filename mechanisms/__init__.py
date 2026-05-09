"""v0.2 Phase 1 mechanism upgrades — each isolated in its own module.

Mechanisms ship behind blitz.toml flags so v0.1.x configs run unchanged.

  cascade_guard   genealogy-graph fault containment (Xie 2603.04474)
  judge_ensemble  N-judge debate + KS-stop (Hu 2510.12697)
  selector_synth  selection-bottleneck synthesis (Maryanskyy 2603.20324)

Default: cascade_guard.enabled = true (unambiguous safety improvement),
judge_ensemble.enabled = false, selector.enabled = false.
"""
