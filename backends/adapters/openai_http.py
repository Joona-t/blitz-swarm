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
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from backends.core import (
    COST_PRICED,
    SCHEMA_NATIVE,
    TOOL_ALLOWLIST,
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


_GATES: dict[tuple[str, int], threading.BoundedSemaphore] = {}
_GATES_LOCK = threading.Lock()


def _gate(base_url: str, n: int) -> threading.BoundedSemaphore:
    """One semaphore per (server, limit), shared by every backend object in
    the process (mirrors ollama_http), so role-routed backends and the
    selector's fan-out cannot exceed the slots this client may use."""
    with _GATES_LOCK:
        return _GATES.setdefault((base_url.rstrip("/"), n), threading.BoundedSemaphore(n))


@dataclass
class OpenAIHTTPBackend:
    base_url: str = ""
    model: str | None = None
    api_key_env: str = ""
    reasoning_effort: str | None = None
    backend_id: str = "local"
    adapter_kind: str = "openai_http"
    metadata: dict = field(default_factory=dict)
    # 0 = no limit. `max_parallel` in the provider table caps this process's
    # requests in flight to the server (a shared vLLM server: our share of its
    # slots). Calls wait here before the request timer starts; the wait is
    # reported as metadata["queued_s"].
    max_parallel: int = 0

    # A bare chat endpoint has no tools at all, so "no tools" is met by
    # construction (allowlist with an empty set); a call that ASKS for tools
    # is refused by the adapter.
    capabilities = BackendCapabilities(
        adapter_kind="openai_http",
        structured_output=True,
        schema_enforcement=True,
        local=True,
        schema_policy=SCHEMA_NATIVE,      # response_format json_schema strict
        tool_policy=TOOL_ALLOWLIST,
        cost_policy=COST_PRICED,          # usage tokens; USD via pricing table
        effort=True,
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

        if call.tools:
            # WP3: a bare completion endpoint has no tools to grant.
            return self._error(model, f"openai_http has no tools; call asked for {list(call.tools)}")
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
        # `extra_body` in the provider table: server-specific request fields,
        # e.g. vLLM's {chat_template_kwargs = {enable_thinking = false}}.
        # It cannot replace the fields this adapter owns.
        extra = self.metadata.get("extra_body") if isinstance(self.metadata, dict) else None
        if isinstance(extra, dict):
            for key, value in extra.items():
                payload.setdefault(key, value)

        req = urllib.request.Request(
            self._url("chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        gate = _gate(self.base_url, self.max_parallel) if self.max_parallel > 0 else None
        queued_from = time.monotonic()
        if gate is not None:
            gate.acquire()
        start = time.monotonic()
        queued_s = round(start - queued_from, 3)
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
        finally:
            if gate is not None:
                gate.release()
        elapsed = time.monotonic() - start

        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            envelope = {}
        text = ""
        finish_reason = None
        thinking_chars = 0
        has_choice = False
        if isinstance(envelope, dict):
            choices = envelope.get("choices") or []
            if choices and isinstance(choices[0], dict):
                has_choice = True
                finish_reason = choices[0].get("finish_reason")
                message = choices[0].get("message") or {}
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                thinking_chars = len(reasoning) if isinstance(reasoning, str) else 0
                content = message.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(message.get("refusal"), str):
                    text = message["refusal"]
        # Raw body only when the server did not answer in the chat shape. A
        # well-formed reply with no content (e.g. a thinking model that spent
        # every token on reasoning, finish_reason "length") is an error, not a
        # report whose text is the JSON envelope.
        if not text and not has_choice:
            text = raw
        usage = envelope.get("usage", {}) if isinstance(envelope, dict) else {}
        usage = usage or {}
        http_ok = 200 <= status < 300
        empty = http_ok and has_choice and not text.strip()
        errored = not http_ok or empty
        return normalize_result(
            backend_id=self.backend_id,
            adapter_kind=self.adapter_kind,
            model=model,
            text=text,
            raw_stdout=raw,
            schema=call.schema,
            elapsed_s=elapsed,
            errored=errored,
            error=(f"openai_http HTTP {status}" if not http_ok
                   else f"openai_http: empty content (finish_reason={finish_reason!r})" if empty
                   else None),
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            # reasoning_tokens: the part of completion_tokens (= output_tokens)
            # spent thinking, when the server reports it (vLLM does).
            metadata={"effort": effort, "finish_reason": finish_reason, "thinking_chars": thinking_chars,
                      "queued_s": queued_s,
                      "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")},
            strict_nulls=True,
        )


__all__ = ["OpenAIHTTPBackend"]
