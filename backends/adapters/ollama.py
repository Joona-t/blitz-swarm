"""Ollama HTTP backend adapter.

This adapter intentionally uses only the Python standard library so the
core package does not acquire optional HTTP dependencies.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from backends.core import (
    COST_PRICED,
    SCHEMA_NATIVE,
    TOOL_ALLOWLIST,
    AgentCall,
    AgentResult,
    BackendCapabilities,
    combined_prompt,
    normalize_result,
)


NO_MODEL_ERROR = (
    "no model resolved for this call: configure [models] + "
    "[backend.providers.<id>.tiers] in blitz.toml (see --list-models)"
)


_GATES: dict[tuple[str, int], threading.BoundedSemaphore] = {}
_GATES_LOCK = threading.Lock()


def _gate(base_url: str, n: int) -> threading.BoundedSemaphore:
    """One semaphore per (server, limit), shared by every backend object in
    the process, so role-routed backends cannot exceed the server's slots."""
    with _GATES_LOCK:
        return _GATES.setdefault((base_url.rstrip("/"), n), threading.BoundedSemaphore(n))


@dataclass
class OllamaHTTPBackend:
    model: str | None = None
    base_url: str = "http://localhost:11434"
    backend_id: str = "ollama"
    adapter_kind: str = "ollama_http"
    # `think = false` in the provider table turns a thinking model's
    # reasoning off; None leaves the server default. Thinking tokens are
    # counted in eval_count, so leaving it on inflates output tokens.
    think: bool | None = None
    # 0 = no limit. Set to the server's parallel slots (1 when
    # OLLAMA_NUM_PARALLEL is unset) so a swarm's concurrent calls queue
    # here, before the request timer starts, instead of timing out while
    # waiting inside the server.
    max_parallel: int = 0

    capabilities = BackendCapabilities(
        adapter_kind="ollama_http",
        structured_output=True,
        local=True,
        schema_policy=SCHEMA_NATIVE,      # `format` accepts a JSON schema
        tool_policy=TOOL_ALLOWLIST,       # /api/generate has no tools
        cost_policy=COST_PRICED,          # eval counts; USD via pricing table
    )

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url.rstrip('/')}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    def call(self, call: AgentCall) -> AgentResult:
        model = call.model or self.model
        if not model:
            return AgentResult(
                backend_id=self.backend_id, adapter_kind=self.adapter_kind,
                model=None, text="", raw_stdout="", parsed=None, elapsed_s=0.0,
                errored=True, error=NO_MODEL_ERROR,
            )
        if call.tools:
            # WP3: /api/generate has no tools to grant.
            return AgentResult(
                backend_id=self.backend_id, adapter_kind=self.adapter_kind,
                model=model, text="", raw_stdout="", parsed=None, elapsed_s=0.0,
                errored=True, error=f"ollama has no tools; call asked for {list(call.tools)}",
            )
        payload = {
            "model": model,
            "prompt": combined_prompt(call),
            "stream": False,
        }
        if call.schema:
            payload["format"] = call.schema
        if self.think is not None:
            payload["think"] = bool(self.think)

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        gate = _gate(self.base_url, self.max_parallel) if self.max_parallel > 0 else None
        queued_from = time.monotonic()
        if gate is not None:
            gate.acquire()
        start = time.monotonic()
        queued_s = start - queued_from
        try:
            with urllib.request.urlopen(req, timeout=call.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                status = resp.status
        except TimeoutError:
            return AgentResult(
                backend_id=self.backend_id,
                adapter_kind=self.adapter_kind,
                model=model,
                text="",
                raw_stdout="",
                parsed=None,
                elapsed_s=time.monotonic() - start,
                errored=True,
                error=f"ollama timed out after {call.timeout_s}s",
            )
        except urllib.error.URLError as exc:
            return AgentResult(
                backend_id=self.backend_id,
                adapter_kind=self.adapter_kind,
                model=model,
                text="",
                raw_stdout="",
                parsed=None,
                elapsed_s=time.monotonic() - start,
                errored=True,
                error=f"ollama request failed: {exc.reason!r}",
            )
        finally:
            if gate is not None:
                gate.release()

        elapsed = time.monotonic() - start
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            envelope = {}
        text = envelope.get("response") if isinstance(envelope, dict) else None
        if not isinstance(text, str):
            text = raw
        errored = not (200 <= status < 300)
        return normalize_result(
            backend_id=self.backend_id,
            adapter_kind=self.adapter_kind,
            model=model,
            text=text,
            raw_stdout=raw,
            schema=call.schema,
            elapsed_s=elapsed,
            errored=errored,
            error=f"ollama HTTP {status}" if errored else None,
            input_tokens=int(envelope.get("prompt_eval_count", 0) or 0)
            if isinstance(envelope, dict) else 0,
            output_tokens=int(envelope.get("eval_count", 0) or 0)
            if isinstance(envelope, dict) else 0,
            metadata={
                "done": envelope.get("done"),
                "done_reason": envelope.get("done_reason"),
                "thinking_chars": len(envelope.get("thinking") or ""),
                "queued_s": round(queued_s, 3),
            } if isinstance(envelope, dict) else {},
        )
