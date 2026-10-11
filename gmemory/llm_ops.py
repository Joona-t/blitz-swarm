"""LLM helpers for memory operations, routed through the `memory_ops` role.

Three hooks -- insight distillation, relevance scoring, trace
sparsification -- share one JSON call path:

    resolve_model("memory_ops", backend_id, cfg)  ->  make_provider_backend(...)

so the model is whatever `[models] memory_ops` maps to on the active
provider (never a literal here), the `[backend] fallback` chain applies,
and rule #10 holds: the user's own CLI subscription, no paid API.

`build_memory_llm` returns None when no backend / tier is configured for
the role. Callers then stay on the heuristic hooks in gmemory.insight_graph
and gmemory.retrieval. Nothing in this module runs at import time and the
test-suite never constructs a real backend (it injects a fake).
"""

from __future__ import annotations

from dataclasses import dataclass

from gmemory.insight_graph import DISTILL_MAX_INSIGHTS, InsightCandidate, heuristic_distill
from gmemory.retrieval import heuristic_relevance, heuristic_sparsify

_RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {"relevance": {"type": "number", "minimum": 0, "maximum": 1}},
    "required": ["relevance"],
}
_SPARSIFY_SCHEMA = {
    "type": "object",
    "properties": {"relevant_indices": {"type": "array", "items": {"type": "integer"}}},
    "required": ["relevant_indices"],
}
_DISTILL_SCHEMA = {
    "type": "object",
    "properties": {
        "insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["content"],
            },
        }
    },
    "required": ["insights"],
}


@dataclass
class MemoryLLM:
    """A resolved `memory_ops` backend plus the three hook methods."""

    backend: object
    backend_id: str
    model: str
    effort: str | None
    sandbox: str
    approval_policy: str
    ephemeral: bool
    timeout_s: int = 60

    def call_json(self, role: str, prompt: str, system_prompt: str, schema: dict) -> tuple[dict | None, float | None]:
        """One structured call. Returns (parsed_or_None, cost_or_None)."""
        from backends import AgentCall  # noqa: PLC0415  (lazy: keep import-time free)

        result = self.backend.call(AgentCall(
            role=role,
            model_role="memory_ops",
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
            model=self.model,
            timeout_s=self.timeout_s,
            sandbox=self.sandbox,
            approval_policy=self.approval_policy,
            reasoning_effort=self.effort,
            ephemeral=self.ephemeral,
        ))
        if result.errored or not isinstance(result.parsed, dict):
            return None, result.cost_usd
        return result.parsed, result.cost_usd

    # -- hooks ------------------------------------------------------------

    def distill(self, topic: str, status: str, utterances: list[dict], cost_str: str = "") -> list[InsightCandidate]:
        trace = "\n".join(
            f"[{u.get('agent_id', '?')}]: {(u.get('findings') or u.get('content') or '')[:300]}"
            for u in utterances[:15]
        )
        prompt = (
            f"A multi-agent swarm just completed a task.\n\nTopic: {topic}\nOutcome: {status}\n\n"
            f"Agent trace:\n{trace}\n\n"
            f"Return up to {DISTILL_MAX_INSIGHTS} distinct, generalizable one-line lessons "
            "useful for FUTURE tasks on similar topics (what worked, what failed, a key fact). "
            "Tags use axis:value with axis in domain|pattern|pitfall."
        )
        data, _ = self.call_json("memory_distill", prompt, "Return JSON only.", _DISTILL_SCHEMA)
        if not data:
            return heuristic_distill(topic, status, utterances, cost_str)
        out: list[InsightCandidate] = []
        for item in data.get("insights") or []:
            content = str((item or {}).get("content") or "").strip()
            if not content:
                continue
            try:
                conf = max(0.0, min(1.0, float(item.get("confidence", 0.6))))
            except (TypeError, ValueError):
                conf = 0.6
            tags = [t for t in (item.get("tags") or []) if isinstance(t, str) and ":" in t]
            out.append(InsightCandidate(content=content[:180], tags=tags, confidence=conf))
            if len(out) >= DISTILL_MAX_INSIGHTS:
                break
        return out or heuristic_distill(topic, status, utterances, cost_str)

    def relevance(self, new_query: str, candidate_text: str, model: str = "") -> tuple[float, float | None]:
        prompt = (
            "Rate how relevant this historical task is to the new query.\n\n"
            f"New query: {new_query}\nHistorical task: {candidate_text}\n\n"
            "Return JSON with a single 'relevance' field (0.0-1.0)."
        )
        data, cost = self.call_json("memory_relevance", prompt, "Return JSON only.", _RELEVANCE_SCHEMA)
        if not data:
            score, _ = heuristic_relevance(new_query, candidate_text)
            return score, cost
        try:
            return max(0.0, min(1.0, float(data.get("relevance", 0.5)))), cost
        except (TypeError, ValueError):
            return 0.5, cost

    def sparsify(self, utterances: list[dict], new_query: str, model: str = "") -> tuple[list[dict], float | None]:
        if len(utterances) <= 5:
            return utterances, 0.0
        trace = "\n".join(
            f"{i}. [{u.get('agent_id', '?')}]: {(u.get('content') or '')[:200]}"
            for i, u in enumerate(utterances)
        )
        prompt = (
            "Given this agent trace and a new query, pick only the steps relevant to the new query.\n\n"
            f"New query: {new_query}\n\nTrace:\n{trace}\n\n"
            "Return JSON with 'relevant_indices': the 0-based indices to keep (max 5)."
        )
        data, cost = self.call_json("memory_sparsify", prompt, "Return JSON only.", _SPARSIFY_SCHEMA)
        if not data:
            kept, _ = heuristic_sparsify(utterances, new_query)
            return kept, cost
        idx = [i for i in data.get("relevant_indices") or [] if isinstance(i, int) and 0 <= i < len(utterances)]
        return [utterances[i] for i in idx[:5]] or utterances[:5], cost


def build_memory_llm(cfg=None, backend_id: str | None = None) -> MemoryLLM | None:
    """Resolve `memory_ops` on the active provider and build its backend.

    None (-> heuristic hooks) when the role/tier is not defined for the
    provider, the provider is unknown, or the backend package cannot build.
    """
    try:
        from backends import make_provider_backend, resolve_model  # noqa: PLC0415
        from backends.models import ModelResolutionError  # noqa: PLC0415
    except ImportError:
        return None
    try:
        if cfg is None:
            from config import load_config  # noqa: PLC0415
            cfg = load_config()
        resolved = resolve_model("memory_ops", backend_id, cfg)
        provider = cfg.backend.get_provider(resolved.backend_id)
        backend = make_provider_backend(resolved.backend_id, cfg, sandbox=provider.sandbox)
    except (ModelResolutionError, ValueError, KeyError) as exc:
        print(f"[memory] LLM helpers off (heuristics): {exc}")
        return None
    return MemoryLLM(
        backend=backend,
        backend_id=resolved.backend_id,
        model=resolved.model,
        effort=resolved.effort or provider.reasoning_effort or None,
        sandbox=provider.sandbox,
        approval_policy=provider.approval_policy,
        ephemeral=provider.ephemeral,
    )
