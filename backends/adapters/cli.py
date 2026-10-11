"""CLI backend adapters for Codex, Claude, and Gemini.

Adapters carry no model defaults: the model string arrives on every
`AgentCall` from `backends.models.resolve_model`, or as the provider's
legacy `model` key via the registry. A call without a model is an explicit
error result, never a silent CLI default.
"""

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
    DEFAULT_SANDBOX,
    AgentCall,
    AgentResult,
    BackendCapabilities,
    combined_prompt,
    normalize_result,
    parse_json_loose,
    strip_strict_nulls,
    to_strict_output_schema,
    validate_json_schema,
)

# Prompts larger than this go to `claude -p` via stdin instead of argv
# (argv has an OS limit; stdin does not). backends-07.
CLAUDE_STDIN_PROMPT_BYTES = 100 * 1024

NO_MODEL_ERROR = (
    "no model resolved for this call: configure [models] + "
    "[backend.providers.<id>.tiers] in blitz.toml (see --list-models)"
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


def _error_result(backend, model: str | None, error: str, *, elapsed_s: float = 0.0,
                  raw_stdout: str = "") -> AgentResult:
    return AgentResult(
        backend_id=backend.backend_id,
        adapter_kind=backend.adapter_kind,
        model=model,
        text="",
        raw_stdout=raw_stdout,
        parsed=None,
        elapsed_s=elapsed_s,
        errored=True,
        error=error,
    )


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------


@dataclass
class CodexLocalBackend:
    model: str | None = None
    reasoning_effort: str | None = None
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
        model = call.model or self.model
        if not self.is_available():
            return _error_result(self, model, "codex CLI not found on PATH")
        if not model:
            return _error_result(self, None, NO_MODEL_ERROR)

        sandbox = call.sandbox or self.sandbox
        approval = call.approval_policy or self.approval_policy
        effort = call.reasoning_effort or self.reasoning_effort
        prompt = combined_prompt(call)

        with tempfile.TemporaryDirectory(prefix="blitz-codex-") as td:
            tmp = Path(td)
            output_path = tmp / "last_message.txt"
            schema_path: Path | None = None
            if call.schema:
                schema_path = tmp / "schema.json"
                schema_path.write_text(
                    json.dumps(to_strict_output_schema(call.schema)),
                    encoding="utf-8",
                )

            cmd = [
                "codex",
                "exec",
                "-c", f"approval_policy={json.dumps(approval)}",
                "--model", model,
                "--sandbox", sandbox,
                "--json",
                "--output-last-message", str(output_path),
            ]
            if effort:
                # Codex effort vocabulary comes from the provider config
                # (`reasoning_effort` / `[effort]`), forwarded verbatim.
                cmd.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])
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
                return _error_result(
                    self, model, f"codex timed out after {call.timeout_s}s",
                    elapsed_s=time.monotonic() - start,
                    raw_stdout=exc.stdout if isinstance(exc.stdout, str) else "",
                )
            except Exception as exc:
                return _error_result(
                    self, model, f"codex subprocess exception: {exc!r}",
                    elapsed_s=time.monotonic() - start,
                )

            elapsed = time.monotonic() - start
            final_text = _read_text_if_present(output_path)
            if not final_text:
                final_text = _extract_last_codex_agent_message(proc.stdout)
            if not final_text:
                final_text = proc.stdout.strip()
            return normalize_result(
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
                strict_nulls=True,
            )


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

_CLAUDE_EFFORT_SUPPORT: bool | None = None


def claude_supports_effort() -> bool:
    """Probe `claude --help` once for `--effort`; cached for the process.

    Tests reset the cache with `reset_claude_probe_cache()` and mock
    `subprocess.run`. The probe is not a model invocation.
    """
    global _CLAUDE_EFFORT_SUPPORT
    if _CLAUDE_EFFORT_SUPPORT is None:
        try:
            proc = subprocess.run(
                ["claude", "--help"], capture_output=True, text=True, timeout=20,
            )
            _CLAUDE_EFFORT_SUPPORT = "--effort" in f"{proc.stdout or ''}{proc.stderr or ''}"
        except Exception:
            _CLAUDE_EFFORT_SUPPORT = False
    return _CLAUDE_EFFORT_SUPPORT


def reset_claude_probe_cache() -> None:
    global _CLAUDE_EFFORT_SUPPORT
    _CLAUDE_EFFORT_SUPPORT = None


@dataclass
class ClaudeCLIBackend:
    model: str | None = None
    reasoning_effort: str | None = None
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
        model = call.model or self.model
        if not self.is_available():
            return _error_result(self, model, "claude CLI not found on PATH")
        if not model:
            return _error_result(self, None, NO_MODEL_ERROR)

        prompt = call.prompt
        use_stdin = len(prompt.encode("utf-8")) > CLAUDE_STDIN_PROMPT_BYTES
        cmd = ["claude", "-p"]
        if not use_stdin:
            cmd.append(prompt)
        cmd.extend(["--output-format", "json", "--model", model])
        effort = call.reasoning_effort or self.reasoning_effort
        if effort and claude_supports_effort():
            cmd.extend(["--effort", effort])
        if call.system_prompt:
            cmd.extend(["--system-prompt", call.system_prompt])
        if call.schema:
            cmd.extend(["--json-schema", json.dumps(call.schema)])

        run_kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": call.timeout_s,
            "cwd": str(call.cwd) if call.cwd else None,
        }
        if use_stdin:
            run_kwargs["input"] = prompt
        else:
            run_kwargs["stdin"] = subprocess.DEVNULL

        start = time.monotonic()
        try:
            proc = subprocess.run(cmd, **run_kwargs)
        except subprocess.TimeoutExpired:
            return _error_result(
                self, model, f"claude timed out after {call.timeout_s}s",
                elapsed_s=time.monotonic() - start,
            )
        except Exception as exc:
            return _error_result(
                self, model, f"claude subprocess exception: {exc!r}",
                elapsed_s=time.monotonic() - start,
            )
        elapsed = time.monotonic() - start
        envelope = parse_json_loose(proc.stdout)
        text = proc.stdout.strip()
        parsed = None
        cost = 0.0
        in_tok = 0
        out_tok = 0
        if envelope:
            raw_cost = envelope.get("total_cost_usd")
            cost = float(raw_cost) if isinstance(raw_cost, (int, float)) else None
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
            metadata={"effort": effort, "prompt_via": "stdin" if use_stdin else "argv"},
        )


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


@dataclass
class GeminiCLIBackend:
    model: str | None = None
    backend_id: str = "gemini"
    adapter_kind: str = "gemini_cli"

    capabilities = BackendCapabilities(adapter_kind="gemini_cli", local=True)

    def is_available(self) -> bool:
        return shutil.which("gemini") is not None

    def call(self, call: AgentCall) -> AgentResult:
        model = call.model or self.model
        if not self.is_available():
            return _error_result(self, model, "gemini CLI not found on PATH")
        if not model:
            return _error_result(self, None, NO_MODEL_ERROR)
        prompt = combined_prompt(call)
        cmd = ["gemini", "-m", model, "-p", prompt]
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=call.timeout_s,
                cwd=str(call.cwd) if call.cwd else None,
            )
        except subprocess.TimeoutExpired:
            return _error_result(
                self, model, f"gemini timed out after {call.timeout_s}s",
                elapsed_s=time.monotonic() - start,
            )
        except Exception as exc:
            return _error_result(
                self, model, f"gemini subprocess exception: {exc!r}",
                elapsed_s=time.monotonic() - start,
            )
        elapsed = time.monotonic() - start
        return normalize_result(
            backend_id=self.backend_id,
            adapter_kind=self.adapter_kind,
            model=model,
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
