"""OpenAI-compatible chat/completions adapter for LOCAL servers.

Targets sparkd, LM Studio, llama.cpp server, vLLM, Ollama's /v1 shim — any
endpoint speaking `POST {base_url}/chat/completions`. Standard library only.

`api_key_env` is the NAME of an environment variable holding a bearer token
for local servers that require one (vLLM `--api-key`, LM Studio auth). It
is optional and defaults to none. This adapter is not a route to a paid
hosted API: no default base_url, no default key (LoveSpark rule #10).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from backends.core import (
    AgentCall,
    AgentResult,
    BackendCapabilities,
    normalize_result,
    to_strict_output_schema,
)

NO_MODEL_ERROR = (
    "no model resolved for this call: configure [models] + "
    "[backend.providers.<id>.tiers] in blitz.toml (see --list-models)"
)


@dataclass
class OpenAIHTTPBackend:
    base_url: str = ""
    model: str | None = None
    api_key_env: str = ""
    reasoning_effort: str | None = None
    backend_id: str = "local"
    adapter_kind: str = "openai_http"
    metadata: dict = field(default_factory=dict)

    capabilities = BackendCapabilities(
        adapter_kind="openai_http",
        structured_output=True,
        schema_enforcement=True,
        local=True,
    )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def is_available(self) -> bool:
        if not self.base_url:
            return False
        try:
            req = urllib.request.Request(self._url("models"), headers=self._headers())
            with urllib.request.urlopen(req, timeout=2) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    def _error(self, model: str | None, error: str, *, elapsed_s: float = 0.0,
               raw: str = "") -> AgentResult:
        return AgentResult(
            backend_id=self.backend_id, adapter_kind=self.adapter_kind, model=model,
            text="", raw_stdout=raw, parsed=None, elapsed_s=elapsed_s,
            errored=True, error=error,
        )

    def call(self, call: AgentCall) -> AgentResult:
        model = call.model or self.model
        if not self.base_url:
            return self._error(model, f"provider {self.backend_id!r} has no base_url")
        if not model:
            return self._error(None, NO_MODEL_ERROR)

        messages = []
        if call.system_prompt:
            messages.append({"role": "system", "content": call.system_prompt})
        messages.append({"role": "user", "content": call.prompt})
        payload: dict = {"model": model, "messages": messages, "stream": False}
        if call.schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "blitz_output",
                    "strict": True,
                    "schema": to_strict_output_schema(call.schema),
                },
            }
        effort = call.reasoning_effort or self.reasoning_effort
        if effort:
            payload["reasoning_effort"] = effort

        req = urllib.request.Request(
            self._url("chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=call.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                status = resp.status
        except TimeoutError:
            return self._error(
                model, f"openai_http timed out after {call.timeout_s}s",
                elapsed_s=time.monotonic() - start,
            )
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            return self._error(
                model, f"openai_http HTTP {exc.code}: {body}",
                elapsed_s=time.monotonic() - start, raw=body,
            )
        except urllib.error.URLError as exc:
            return self._error(
                model, f"openai_http request failed: {exc.reason!r}",
                elapsed_s=time.monotonic() - start,
            )
        except Exception as exc:
            return self._error(
                model, f"openai_http exception: {exc!r}",
                elapsed_s=time.monotonic() - start,
            )
        elapsed = time.monotonic() - start

        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            envelope = {}
        text = ""
        if isinstance(envelope, dict):
            choices = envelope.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(message.get("refusal"), str):
                    text = message["refusal"]
        if not text:
            text = raw
        usage = envelope.get("usage", {}) if isinstance(envelope, dict) else {}
        usage = usage or {}
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
            error=f"openai_http HTTP {status}" if errored else None,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            metadata={"effort": effort},
            strict_nulls=True,
        )


__all__ = ["OpenAIHTTPBackend"]
