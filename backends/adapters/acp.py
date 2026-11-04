"""Agent Client Protocol (ACP) adapter: any ACP agent as a backend (WP5).

ACP (agentclientprotocol.com) is JSON-RPC 2.0 over the agent's stdio. The
client (us) spawns the agent, calls `initialize`, `session/new`, then
`session/prompt`, and streams `session/update` notifications until the
prompt returns a `stopReason`. Tool use is gated by the agent asking us
`session/request_permission`, which is exactly the hook WP3 wants: with
`tools=None` every request is rejected; with a list, requests whose tool
kind/title match are allowed once.

Standard library only, on purpose. The official Python SDK pins
Python <3.15 and was at 1.0rc3 on 2026-10-12; the wire protocol we need
is five messages, so speaking it directly keeps the adapter importable
everywhere and stable across SDK churn. Protocol v1 (stable).

Config example (blitz.toml):

    [backend.providers.gemini_acp]
    adapter = "acp"
    command = ["gemini", "--experimental-acp", "-m", "{model}"]
    mcp_servers = []            # optional, forwarded to session/new
    [backend.providers.gemini_acp.tiers]
    fast = "<the agent's fast model id>"
    standard = "<its standard model id>"
    deep = "<its deep model id>"

Model selection: ACP v1 has no standard model selector, so `{model}` in
the command template is how the model reaches the agent. Schema: ACP has
no structured-output facility -> `schema_policy = validate` (PolicyBackend
appends the schema and validates the reply). Cost: `unknown` unless the
agent reports usage tokens on the prompt response.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from backends.core import (
    COST_UNKNOWN,
    SCHEMA_VALIDATE,
    TOOL_ALLOWLIST,
    AgentCall,
    AgentResult,
    BackendCapabilities,
    combined_prompt,
    normalize_result,
)

PROTOCOL_VERSION = 1
NO_MODEL_ERROR = (
    "no model resolved for this call: configure [models] + "
    "[backend.providers.<id>.tiers] in blitz.toml (see --list-models)"
)
STOP_OK = "end_turn"
STOP_ERRORS = {
    "max_tokens": "agent hit max_tokens",
    "max_turn_requests": "agent hit max_turn_requests",
    "refusal": "agent refused",
    "cancelled": "prompt cancelled",
}


class _Timeout(Exception):
    pass


class _AgentExited(Exception):
    pass


class _Rpc:
    """Minimal JSON-RPC client over a child process's stdio."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.inbox: queue.Queue = queue.Queue()
        self._next_id = 1
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self.stderr_tail: list[str] = []
        self._err = threading.Thread(target=self._pump_err, daemon=True)
        self._err.start()

    def _pump(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.inbox.put(json.loads(line))
            except json.JSONDecodeError:
                self.inbox.put({"_garbage": line})
        self.inbox.put(None)  # EOF

    def _pump_err(self):
        for line in self.proc.stderr:
            self.stderr_tail.append(line.rstrip())
            del self.stderr_tail[:-20]

    def send(self, obj: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            # the agent died between our messages (seen as a race in the
            # crash test): report it as an exit, not a traceback
            raise _AgentExited()

    def request(self, method: str, params: dict) -> int:
        rid = self._next_id
        self._next_id += 1
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        return rid

    def respond(self, rid, result: dict | None = None, error: dict | None = None) -> None:
        msg: dict = {"jsonrpc": "2.0", "id": rid}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result or {}
        self.send(msg)


@dataclass
class ACPBackend:
    command: list[str] = field(default_factory=list)
    model: str | None = None
    reasoning_effort: str | None = None
    backend_id: str = "acp"
    adapter_kind: str = "acp"
    metadata: dict = field(default_factory=dict)

    capabilities = BackendCapabilities(
        adapter_kind="acp",
        structured_output=False,
        schema_enforcement=False,
        tool_sandbox=True,
        local=True,
        schema_policy=SCHEMA_VALIDATE,   # no schema facility in ACP v1
        tool_policy=TOOL_ALLOWLIST,      # session/request_permission is the gate
        cost_policy=COST_UNKNOWN,        # usage tokens only if the agent sends them
        effort=False,
        max_turns=False,
    )

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return bool(self.command) and shutil.which(self.command[0]) is not None

    def _error(self, model, error, *, elapsed_s=0.0, raw="") -> AgentResult:
        return AgentResult(
            backend_id=self.backend_id, adapter_kind=self.adapter_kind, model=model,
            text="", raw_stdout=raw, parsed=None, elapsed_s=elapsed_s,
            errored=True, error=error,
        )

    def _argv(self, model: str) -> list[str]:
        return [c.replace("{model}", model) for c in self.command]

    # ------------------------------------------------------------------
    def call(self, call: AgentCall) -> AgentResult:
        model = call.model or self.model
        if not self.command:
            return self._error(model, f"provider {self.backend_id!r} has no 'command' template")
        if not self.is_available():
            return self._error(model, f"{self.command[0]} not found on PATH")
        if not model:
            return self._error(None, NO_MODEL_ERROR)

        deadline = time.monotonic() + call.timeout_s
        start = time.monotonic()
        transcript: list[dict] = []
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        permissions: list[dict] = []
        usage: dict = {}

        with tempfile.TemporaryDirectory(prefix="blitz-acp-") as td:
            cwd = Path(call.cwd) if call.cwd else Path(td)
            cwd.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env.update({str(k): str(v) for k, v in (self.metadata.get("env") or {}).items()})
            try:
                proc = subprocess.Popen(
                    self._argv(model), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, bufsize=1, cwd=str(cwd), env=env,
                )
            except OSError as exc:
                return self._error(model, f"acp spawn failed: {exc!r}")
            rpc = _Rpc(proc)

            def _wait(rid: int) -> dict:
                """Pump messages until the response to `rid` arrives, answering
                the agent's own requests on the way."""
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _Timeout()
                    try:
                        msg = rpc.inbox.get(timeout=min(remaining, 0.5))
                    except queue.Empty:
                        if proc.poll() is not None:
                            raise _AgentExited()
                        continue
                    if msg is None:
                        raise _AgentExited()
                    transcript.append(msg)
                    if "_garbage" in msg:
                        continue
                    if msg.get("id") == rid and ("result" in msg or "error" in msg):
                        return msg
                    method = msg.get("method")
                    if method is None:
                        continue  # a response we are not waiting for
                    if "id" in msg:
                        self._answer_request(rpc, msg, call, permissions)
                    else:
                        self._on_notification(msg, text_parts, tool_calls, usage)

            error = None
            stop_reason = None
            try:
                rid = rpc.request("initialize", {
                    "protocolVersion": PROTOCOL_VERSION,
                    "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False},
                                           "terminal": False},
                    "clientInfo": {"name": "blitz-swarm", "version": "0.4"},
                })
                init = _wait(rid)
                if "error" in init:
                    raise RuntimeError(f"initialize failed: {init['error']}")
                rid = rpc.request("session/new", {
                    "cwd": str(cwd),
                    "mcpServers": list(self.metadata.get("mcp_servers") or []),
                })
                new = _wait(rid)
                if "error" in new:
                    raise RuntimeError(f"session/new failed: {new['error']}")
                session_id = (new.get("result") or {}).get("sessionId")
                if not session_id:
                    raise RuntimeError("session/new returned no sessionId")
                rid = rpc.request("session/prompt", {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": combined_prompt(call)}],
                })
                done = _wait(rid)
                if "error" in done:
                    raise RuntimeError(f"session/prompt failed: {done['error']}")
                result = done.get("result") or {}
                stop_reason = result.get("stopReason")
                usage.update(result.get("usage") or {})
                if stop_reason != STOP_OK:
                    error = f"acp stop_reason={stop_reason}: {STOP_ERRORS.get(stop_reason, 'unexpected stop')}"
            except _Timeout:
                error = f"acp timed out after {call.timeout_s}s"
            except _AgentExited:
                code = proc.poll()
                tail = " | ".join(rpc.stderr_tail[-3:])
                error = f"acp agent exited (code {code}) before answering: {tail[:300]}"
            except RuntimeError as exc:
                error = f"acp {exc}"
            finally:
                try:
                    proc.kill()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass

        elapsed = time.monotonic() - start
        text = "".join(text_parts).strip()
        raw = "\n".join(json.dumps(m) for m in transcript)
        in_tok = int(usage.get("inputTokens", usage.get("input_tokens", 0)) or 0)
        out_tok = int(usage.get("outputTokens", usage.get("output_tokens", 0)) or 0)
        if error:
            r = self._error(model, error, elapsed_s=elapsed, raw=raw)
            r.text = text
            r.input_tokens, r.output_tokens = in_tok, out_tok
            r.metadata = {"stop_reason": stop_reason, "tool_calls": tool_calls, "permissions": permissions}
            return r
        return normalize_result(
            backend_id=self.backend_id, adapter_kind=self.adapter_kind, model=model,
            text=text, raw_stdout=raw, schema=call.schema, elapsed_s=elapsed,
            input_tokens=in_tok, output_tokens=out_tok,
            cost_policy=COST_UNKNOWN,
            schema_mode=SCHEMA_VALIDATE if call.schema else "",
            metadata={"stop_reason": stop_reason, "tool_calls": tool_calls,
                      "permissions": permissions, "tools": list(call.tools or [])},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _on_notification(msg: dict, text_parts: list[str], tool_calls: list[dict], usage: dict) -> None:
        if msg.get("method") != "session/update":
            return
        update = (msg.get("params") or {}).get("update") or {}
        kind = update.get("sessionUpdate")
        if kind == "agent_message_chunk":
            content = update.get("content") or {}
            if content.get("type") == "text" and isinstance(content.get("text"), str):
                text_parts.append(content["text"])
        elif kind in ("tool_call", "tool_call_update"):
            tool_calls.append({k: update.get(k) for k in ("toolCallId", "title", "kind", "status")})
        elif kind == "usage_update":
            if isinstance(update.get("cost"), dict):
                usage["cost"] = update["cost"]

    @staticmethod
    def _answer_request(rpc: _Rpc, msg: dict, call: AgentCall, permissions: list[dict]) -> None:
        method = msg.get("method")
        rid = msg.get("id")
        params = msg.get("params") or {}
        if method == "session/request_permission":
            tool = params.get("toolCall") or {}
            options = params.get("options") or []
            allowed = _tool_allowed(tool, call.tools)
            wanted = ("allow_once", "allow_always") if allowed else ("reject_once", "reject_always")
            choice = next((o for k in wanted for o in options if o.get("kind") == k), None)
            permissions.append({"tool": tool.get("title") or tool.get("kind"), "allowed": allowed})
            if choice is None:
                rpc.respond(rid, {"outcome": {"outcome": "cancelled"}})
            else:
                rpc.respond(rid, {"outcome": {"outcome": "selected", "optionId": choice.get("optionId")}})
            return
        # fs/*, terminal/*, elicitation/*: we advertised none of these.
        rpc.respond(rid, error={"code": -32601, "message": f"{method} not supported by blitz-swarm client"})


def _tool_allowed(tool: dict, tools: list[str] | None) -> bool:
    """tools=None -> nothing. A list matches the tool call's `kind` or a
    case-insensitive substring of its `title` (ACP has no canonical tool
    names across agents)."""
    if tools is None:
        return False
    kind = str(tool.get("kind") or "").lower()
    title = str(tool.get("title") or "").lower()
    for t in tools:
        t = t.lower()
        if t == "*" or t == kind or (t and t in title):
            return True
    return False


__all__ = ["ACPBackend", "PROTOCOL_VERSION"]
