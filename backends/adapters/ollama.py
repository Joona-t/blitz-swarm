"""Ollama HTTP backend adapter.

This adapter intentionally uses only the Python standard library so the
core package does not acquire optional HTTP dependencies.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from backends.core import (
    AgentCall,
    AgentResult,
    BackendCapabilities,
    combined_prompt,
    normalize_result,
)


@dataclass
class OllamaHTTPBackend:
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    backend_id: str = "ollama"
    adapter_kind: str = "ollama_http"

    capabilities = BackendCapabilities(
        adapter_kind="ollama_http",
        structured_output=True,
        local=True,
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
        payload = {
            "model": model,
            "prompt": combined_prompt(call),
            "stream": False,
        }
        if call.schema:
            payload["format"] = call.schema

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.monotonic()
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
            metadata={"done": envelope.get("done")} if isinstance(envelope, dict) else {},
        )
