# Limitations

## Data Limitations

1. **No n>=20 live benchmark yet.** The mechanism and backend suites are green, but the harness-agnostic stack has not been run against a statistically meaningful bench slate.

2. **No baseline comparison yet.** The v0.1.1 baseline metrics exist for a few historical runs, but there is not yet a controlled Codex/Claude/Gemini/Ollama/single-agent comparison.

3. **Codex cost telemetry is partial.** `AgentResult` standardizes `cost_usd`, `input_tokens`, and `output_tokens`, but Codex CLI does not currently expose the same cost envelope as Claude CLI. Cost fields are optional.

## Architecture Limitations

4. **Cascade adjudication is heuristic by default.** The guard is wired into orchestration and filters errored/raw/blocked outputs before they enter future context, but claim decomposition and contradiction screening still use LLM-free hooks unless swapped.

5. **Memory consolidation is compatibility-first.** `gmemory/` is treated as the canonical implementation and `memory/` routes LLM helper calls through the backend layer, but `memory/reader.py` and `memory/writer.py` still preserve legacy storage APIs for orchestrator compatibility.

6. **Selector synthesis increases judge calls.** Max-quality mode uses pairwise span selection, which can grow quickly with many spans. `candidate_cap` and section granularity keep this bounded, but latency/cost need live measurement.

7. **No web retrieval tool is wired into agents.** Research outputs still depend on the model backend's knowledge unless a future retrieval/search backend is added.

## External Validity

8. **Codex is the default, not the only supported backend.** Claude, Gemini, and Ollama adapters exist, and future harnesses should plug into the same registry contract.

9. **Local CLI behavior can drift.** The implementation is grounded in `codex-cli 0.132.0`; future CLI flag changes may require adapter updates.

10. **Solo-operator evaluation.** Design decisions and qualitative assessments still need external review and repeatable benchmark data.
