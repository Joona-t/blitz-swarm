"""CLI backend adapters for Codex, Claude, and Gemini."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from backends.core import (
    DEFAULT_APPROVAL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING,
    DEFAULT_SANDBOX,
    AgentCall,
    AgentResult,
    BackendCapabilities,
    combined_prompt,
    normalize_result,
    parse_json_loose,
    validate_json_schema,
)


def _extract_last_codex_agent_message(stdout: str) -> str:
    last = ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, dict) else None
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str):
                last = text
    return last


def _read_text_if_present(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return ""


@dataclass
class CodexLocalBackend:
    model: str = DEFAULT_CODEX_MODEL
    reasoning_effort: str = DEFAULT_CODEX_REASONING
    sandbox: str = DEFAULT_SANDBOX
    approval_policy: str = DEFAULT_APPROVAL
    ephemeral: bool = True
    backend_id: str = "codex"
    adapter_kind: str = "codex_cli"

    capabilities = BackendCapabilities(
        adapter_kind="codex_cli",
        structured_output=True,
        schema_enforcement=True,
        tool_sandbox=True,
        local=True,
        supports_ephemeral=True,
    )

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    def call(self, call: AgentCall) -> AgentResult:
        if not self.is_available():
            return AgentResult(
                backend_id=self.backend_id,
                adapter_kind=self.adapter_kind,
                model=call.model or self.model,
                text="",
                raw_stdout="",
                parsed=None,
                elapsed_s=0.0,
                errored=True,
                error="codex CLI not found on PATH",
            )

        model = call.model or self.model
        sandbox = call.sandbox or self.sandbox
        approval = call.approval_policy or self.approval_policy
        prompt = combined_prompt(call)

        with tempfile.TemporaryDirectory(prefix="blitz-codex-") as td:
            tmp = Path(td)
            output_path = tmp / "last_message.txt"
            schema_path: Path | None = None
            if call.schema:
                schema_path = tmp / "schema.json"
                schema_path.write_text(json.dumps(call.schema), encoding="utf-8")

            cmd = [
                "codex",
                "exec",
                "-c", f"approval_policy={json.dumps(approval)}",
                "--model", model,
                "--sandbox", sandbox,
                "--json",
                "--output-last-message", str(output_path),
            ]
            if call.reasoning_effort or self.reasoning_effort:
                cmd.extend([
                    "-c",
                    f"model_reasoning_effort={json.dumps(call.reasoning_effort or self.reasoning_effort)}",
                ])
            if call.cwd:
                cmd.extend(["--cd", str(call.cwd)])
            if call.ephemeral and self.ephemeral:
                cmd.append("--ephemeral")
            if schema_path is not None:
                cmd.extend(["--output-schema", str(schema_path)])
            cmd.append("-")

            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=call.timeout_s,
                    cwd=str(call.cwd) if call.cwd else None,
                )
            except subprocess.TimeoutExpired as exc:
                return AgentResult(
                    backend_id=self.backend_id,
                    adapter_kind=self.adapter_kind,
                    model=model,
                    text="",
                    raw_stdout=exc.stdout if isinstance(exc.stdout, str) else "",
                    parsed=None,
                    elapsed_s=time.monotonic() - start,
                    errored=True,
                    error=f"codex timed out after {call.timeout_s}s",
                )
            except Exception as exc:
                return AgentResult(
                    backend_id=self.backend_id,
                    adapter_kind=self.adapter_kind,
                    model=model,
                    text="",
                    raw_stdout="",
                    parsed=None,
                    elapsed_s=time.monotonic() - start,
                    errored=True,
                    error=f"codex subprocess exception: {exc!r}",
                )

            elapsed = time.monotonic() - start
            final_text = _read_text_if_present(output_path)
            if not final_text:
                final_text = _extract_last_codex_agent_message(proc.stdout)
            if not final_text:
                final_text = proc.stdout.strip()
            result = normalize_result(
                backend_id=self.backend_id,
                adapter_kind=self.adapter_kind,
                model=model,
                text=final_text,
                raw_stdout=proc.stdout,
                schema=call.schema,
                elapsed_s=elapsed,
                errored=proc.returncode != 0,
                error=(
                    f"codex exit {proc.returncode}: {(proc.stderr or '')[:500]}"
                    if proc.returncode != 0 else None
                ),
            )
            return result


@dataclass
class ClaudeCLIBackend:
    model: str = "sonnet"
    backend_id: str = "claude"
    adapter_kind: str = "claude_cli"

    capabilities = BackendCapabilities(
        adapter_kind="claude_cli",
        structured_output=True,
        schema_enforcement=True,
        local=True,
    )

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    def call(self, call: AgentCall) -> AgentResult:
        if not self.is_available():
            return AgentResult(
                backend_id=self.backend_id, adapter_kind=self.adapter_kind,
                model=call.model or self.model, text="", raw_stdout="",
                parsed=None, elapsed_s=0.0, errored=True,
                error="claude CLI not found on PATH",
            )
        model = call.model or self.model
        cmd = ["claude", "-p", call.prompt, "--output-format", "json", "--model", model]
        if call.system_prompt:
            cmd.extend(["--system-prompt", call.system_prompt])
        if call.schema:
            cmd.extend(["--json-schema", json.dumps(call.schema)])
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=call.timeout_s,
                cwd=str(call.cwd) if call.cwd else None,
            )
        except subprocess.TimeoutExpired:
            return AgentResult(
                backend_id=self.backend_id, adapter_kind=self.adapter_kind,
                model=model, text="", raw_stdout="", parsed=None,
                elapsed_s=time.monotonic() - start, errored=True,
                error=f"claude timed out after {call.timeout_s}s",
            )
        elapsed = time.monotonic() - start
        envelope = parse_json_loose(proc.stdout)
        text = proc.stdout.strip()
        parsed = None
        cost = 0.0
        in_tok = 0
        out_tok = 0
        if envelope:
            cost = float(envelope.get("total_cost_usd", 0.0) or 0.0)
            usage = envelope.get("usage", {}) or {}
            in_tok = int(usage.get("input_tokens", 0) or 0)
            out_tok = int(usage.get("output_tokens", 0) or 0)
            if isinstance(envelope.get("structured_output"), dict):
                parsed = envelope["structured_output"]
                text = json.dumps(parsed)
            elif isinstance(envelope.get("result"), dict):
                parsed = envelope["result"]
                text = json.dumps(parsed)
            elif isinstance(envelope.get("result"), str):
                text = envelope["result"]
                parsed = parse_json_loose(text)
        if parsed is None:
            parsed = parse_json_loose(text)
        validation_errors = validate_json_schema(parsed, call.schema)
        error = None
        if proc.returncode != 0:
            error = f"claude exit {proc.returncode}: {(proc.stderr or '')[:500]}"
        elif validation_errors:
            error = "; ".join(validation_errors[:3])
        return AgentResult(
            backend_id=self.backend_id, adapter_kind=self.adapter_kind,
            model=model, text=text, raw_stdout=proc.stdout, parsed=parsed,
            elapsed_s=elapsed, errored=proc.returncode != 0 or bool(validation_errors),
            error=error, cost_usd=cost, input_tokens=in_tok,
            output_tokens=out_tok, validation_errors=validation_errors,
        )


@dataclass
class GeminiCLIBackend:
    model: str | None = None
    backend_id: str = "gemini"
    adapter_kind: str = "gemini_cli"

    capabilities = BackendCapabilities(adapter_kind="gemini_cli", local=True)

    def is_available(self) -> bool:
        return shutil.which("gemini") is not None

    def call(self, call: AgentCall) -> AgentResult:
        if not self.is_available():
            return AgentResult(
                backend_id=self.backend_id, adapter_kind=self.adapter_kind,
                model=call.model or self.model, text="", raw_stdout="",
                parsed=None, elapsed_s=0.0, errored=True,
                error="gemini CLI not found on PATH",
            )
        prompt = combined_prompt(call)
        cmd = ["gemini", "-p", prompt]
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=call.timeout_s,
                cwd=str(call.cwd) if call.cwd else None,
            )
        except subprocess.TimeoutExpired:
            return AgentResult(
                backend_id=self.backend_id, adapter_kind=self.adapter_kind,
                model=call.model or self.model, text="", raw_stdout="",
                parsed=None, elapsed_s=time.monotonic() - start, errored=True,
                error=f"gemini timed out after {call.timeout_s}s",
            )
        elapsed = time.monotonic() - start
        return normalize_result(
            backend_id=self.backend_id,
            adapter_kind=self.adapter_kind,
            model=call.model or self.model,
            text=proc.stdout.strip(),
            raw_stdout=proc.stdout,
            schema=call.schema,
            elapsed_s=elapsed,
            errored=proc.returncode != 0,
            error=(
                f"gemini exit {proc.returncode}: {(proc.stderr or '')[:500]}"
                if proc.returncode != 0 else None
            ),
        )
