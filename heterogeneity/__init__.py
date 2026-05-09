"""Cross-CLI heterogeneity routing.

Routes role calls to local CLIs Joona already has (`claude`, `codex`,
`gemini`, `llm`). Rule-#10 compliant: all subprocess invocations against
existing subscriptions, no paid API keys required.

Anchor: Maryanskyy 2603.20324 (Mar 2026) — heterogeneous diversity
helps iff paired with judge-driven selection. Phase 1 ships the
selector_synth; this layer gives it heterogeneous candidates to pick from.
"""
