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
    COST_PRICED,
    COST_REPORTED,
    COST_UNKNOWN,
    DEFAULT_APPROVAL,
    DEFAULT_SANDBOX,
    SCHEMA_NATIVE,
    SCHEMA_VALIDATE,
    TOOL_ALLOWLIST,
    TOOL_NONE,
    TOOL_SANDBOX,
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


def _codex_error_message(stdout: str) -> str:
    """The last `{"type": "error"|"turn.failed", ...}` message in the JSONL
    stream. codex exec prints these on stdout and exits 1 with an empty
    stderr, so without this the error text is lost (live finding 2026-11-04:
    "The 'x' model is not supported when using Codex with a ChatGPT account")."""
    last = ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "error" and isinstance(event.get("message"), str):
            last = event["message"]
        elif event.get("type") == "turn.failed":
            err = event.get("error") or {}
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                last = err["message"]
    return last


def _codex_usage(stdout: str) -> tuple[int, int, int]:
    """(input_tokens, cached_input_tokens, output_tokens) summed over every
    JSONL event that carries a `usage` object (`turn.completed` in current
    builds; older builds nest it under `item`/`info`). Missing -> zeros."""
    in_tok = cached = out_tok = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = None
        if isinstance(event, dict):
            usage = event.get("usage")
            if usage is None and isinstance(event.get("item"), dict):
                usage = event["item"].get("usage")
            if usage is None and isinstance(event.get("info"), dict):
                usage = event["info"].get("usage") or event["info"].get("total_token_usage")
        if isinstance(usage, dict):
            in_tok += int(usage.get("input_tokens", 0) or 0)
            cached += int(usage.get("cached_input_tokens", 0) or 0)
            out_tok += int(usage.get("output_tokens", 0) or 0)
    return in_tok, cached, out_tok


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
        schema_policy=SCHEMA_NATIVE,      # --output-schema (strict mode)
        tool_policy=TOOL_SANDBOX,         # --sandbox read-only / -a never; no per-tool list
        cost_policy=COST_PRICED,          # tokens from JSONL; USD only via a pricing table
        effort=True,
    )

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    def call(self, call: AgentCall) -> AgentResult:
        model = call.model or self.model
        if not self.is_available():
            return _error_result(self, model, "codex CLI not found on PATH")
        if not model:
            return _error_result(self, None, NO_MODEL_ERROR)

        # WP3 tool policy. Codex has no per-tool allowlist, only a sandbox.
        # tools=None (no tools) -> the tightest sandbox and no approvals,
        # whatever the provider default says; a tool list -> the configured
        # sandbox (workspace-write at most). danger-full-access is refused.
        if call.tools is None:
            sandbox, approval = "read-only", "never"
        else:
            sandbox = call.sandbox or self.sandbox
            approval = call.approval_policy or self.approval_policy
        if sandbox == "danger-full-access":
            return _error_result(self, model, "codex sandbox danger-full-access is not allowed (WP3 tool policy)")
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
                # Live finding 2026-11-04: `codex exec` refuses to run in a
                # directory that is not a trusted git checkout ("Not inside
                # a trusted directory and --skip-git-repo-check was not
                # specified", exit 1). Our scratch cwd is never a repo, by
                # design (core-loop-04), so the check is skipped explicitly.
                "--skip-git-repo-check",
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
            cwd = _scratch_cwd(call, tmp)
            cmd.extend(["--cd", str(cwd)])
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
                    cwd=str(cwd),
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
            in_tok, cached_tok, out_tok = _codex_usage(proc.stdout)
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
                    f"codex exit {proc.returncode}: "
                    f"{((proc.stderr or '').strip() or _codex_error_message(proc.stdout))[:500]}"
                    if proc.returncode != 0 else None
                ),
                strict_nulls=True,
                # WP4: codex reports tokens, never USD. input_tokens includes
                # cached input (it is still billed, at a lower rate); the
                # split is kept in metadata for a pricing table that cares.
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_policy=COST_UNKNOWN,
                schema_mode=SCHEMA_NATIVE if call.schema else "",
                metadata={"sandbox": sandbox, "approval": approval,
                          "tools": list(call.tools or []),
                          "cached_input_tokens": cached_tok},
            )


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

_CLAUDE_HELP_TEXT: str | None = None


def _claude_help_text() -> str:
    """`claude --help` output, probed once per process (not a model call).

    Tests reset the cache with `reset_claude_probe_cache()` and mock
    `subprocess.run`.
    """
    global _CLAUDE_HELP_TEXT
    if _CLAUDE_HELP_TEXT is None:
        try:
            proc = subprocess.run(
                ["claude", "--help"], capture_output=True, text=True, timeout=20,
            )
            _CLAUDE_HELP_TEXT = f"{proc.stdout or ''}{proc.stderr or ''}"
        except Exception:
            _CLAUDE_HELP_TEXT = ""
    return _CLAUDE_HELP_TEXT


def claude_supports_flag(flag: str) -> bool:
    """True when `claude --help` advertises `flag` (e.g. "--effort",
    "--max-turns"). A flag the installed CLI does not list is never sent."""
    return flag in _claude_help_text()


def claude_supports_effort() -> bool:
    return claude_supports_flag("--effort")


def reset_claude_probe_cache() -> None:
    global _CLAUDE_HELP_TEXT
    _CLAUDE_HELP_TEXT = None


def _scratch_cwd(call: AgentCall, tmp: Path) -> Path:
    """Working directory for a CLI invocation: the call's explicit cwd (the
    orchestrator passes output/scratch/<run_id>/) or a throwaway temp dir.
    Never the inherited process cwd, which is usually the repo root
    (core-loop-04, critic gap A)."""
    if call.cwd:
        cwd = Path(call.cwd)
        cwd.mkdir(parents=True, exist_ok=True)
        return cwd
    return tmp


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
        schema_policy=SCHEMA_NATIVE,      # --json-schema -> structured_output
        tool_policy=TOOL_ALLOWLIST,       # --tools "" / --tools a,b
        cost_policy=COST_REPORTED,        # total_cost_usd in the envelope
        effort=True,
        max_turns=True,
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
        # Tool access is explicit: `--tools ""` disables every built-in tool
        # (the CLI documents "" as "disable all tools"); a list is forwarded
        # comma-joined. `--max-turns` only when this CLI build advertises it.
        if claude_supports_flag("--tools"):
            cmd.extend(["--tools", ",".join(call.tools) if call.tools else ""])
        elif call.tools is None:
            # WP3: this CLI build cannot disable tools; running the prompt
            # with tools would be AUDIT H4. Refuse instead of pretending.
            return _error_result(
                self, model,
                "claude CLI build lacks --tools; cannot guarantee a no-tools call "
                "(upgrade claude, or set allow_degrade = [\"no_tools\"] on the provider)",
            )
        if call.max_turns is not None and claude_supports_flag("--max-turns"):
            cmd.extend(["--max-turns", str(call.max_turns)])
        if call.system_prompt:
            cmd.extend(["--system-prompt", call.system_prompt])
        if call.schema:
            cmd.extend(["--json-schema", json.dumps(call.schema)])

        start = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="blitz-claude-") as td:
                run_kwargs: dict = {
                    "capture_output": True,
                    "text": True,
                    "timeout": call.timeout_s,
                    "cwd": str(_scratch_cwd(call, Path(td))),
                }
                if use_stdin:
                    run_kwargs["input"] = prompt
                else:
                    run_kwargs["stdin"] = subprocess.DEVNULL
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
        cost = None
        in_tok = 0
        out_tok = 0
        if envelope:
            raw_cost = envelope.get("total_cost_usd")
            cost = float(raw_cost) if isinstance(raw_cost, (int, float)) else None
            usage = envelope.get("usage", {}) or {}
            # WP4: the envelope splits input into fresh + cache_creation +
            # cache_read; metrics.py summed only the first and reported
            # 3-15 input tokens per round (AUDIT, metrics). Count all three.
            cache_create = int(usage.get("cache_creation_input_tokens", 0) or 0)
            cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
            in_tok = int(usage.get("input_tokens", 0) or 0) + cache_create + cache_read
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
        # The JSON envelope flags failures (max turns, refusals, API errors)
        # with `is_error: true` even when the process exits 0.
        is_error = bool(envelope.get("is_error")) if envelope else False
        if is_error:
            parsed = None
        validation_errors = validate_json_schema(parsed, call.schema)
        error = None
        if proc.returncode != 0:
            error = f"claude exit {proc.returncode}: {(proc.stderr or '')[:500]}"
        elif is_error:
            detail = envelope.get("result") if isinstance(envelope.get("result"), str) else ""
            error = f"claude is_error: {detail[:500] or envelope.get('subtype') or 'unspecified'}"
        elif validation_errors:
            error = "; ".join(validation_errors[:3])
        return AgentResult(
            backend_id=self.backend_id, adapter_kind=self.adapter_kind,
            model=model, text=text, raw_stdout=proc.stdout, parsed=parsed,
            elapsed_s=elapsed,
            errored=proc.returncode != 0 or is_error or bool(validation_errors),
            error=error, cost_usd=cost, input_tokens=in_tok,
            output_tokens=out_tok, validation_errors=validation_errors,
            cost_policy=COST_REPORTED if isinstance(cost, (int, float)) else COST_UNKNOWN,
            schema_mode=SCHEMA_NATIVE if call.schema else "",
            metadata={"effort": effort, "prompt_via": "stdin" if use_stdin else "argv",
                      "tools": list(call.tools or []), "max_turns": call.max_turns,
                      "cache_creation_input_tokens": (
                          int((envelope or {}).get("usage", {}).get("cache_creation_input_tokens", 0) or 0)),
                      "cache_read_input_tokens": (
                          int((envelope or {}).get("usage", {}).get("cache_read_input_tokens", 0) or 0))},
        )


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


@dataclass
class GeminiCLIBackend:
    model: str | None = None
    backend_id: str = "gemini"
    adapter_kind: str = "gemini_cli"

    # Gemini headless has `--output-format json` ({response, stats, error})
    # but no user schema and no documented tool allowlist, so: schema is
    # prompt+validate, tools cannot be restricted (a no-tools call is
    # refused unless the provider sets allow_degrade = ["no_tools"]).
    capabilities = BackendCapabilities(
        adapter_kind="gemini_cli", local=True,
        schema_policy=SCHEMA_VALIDATE, tool_policy=TOOL_NONE, cost_policy=COST_UNKNOWN,
    )

    def is_available(self) -> bool:
        return shutil.which("gemini") is not None

    def call(self, call: AgentCall) -> AgentResult:
        model = call.model or self.model
        if not self.is_available():
            return _error_result(self, model, "gemini CLI not found on PATH")
        if not model:
            return _error_result(self, None, NO_MODEL_ERROR)
        prompt = combined_prompt(call)
        # `--output-format json` -> {"response": str, "stats": {...}, "error"?}
        # (geminicli.com/docs/cli/headless). A build without the flag prints
        # plain text; both shapes are handled below.
        cmd = ["gemini", "-m", model, "--output-format", "json", "-p", prompt]
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
        text = proc.stdout.strip()
        in_tok = out_tok = 0
        env_error = None
        envelope = parse_json_loose(proc.stdout)
        if isinstance(envelope, dict) and "response" in envelope:
            text = str(envelope.get("response") or "").strip()
            stats = envelope.get("stats") or {}
            if isinstance(stats, dict):
                in_tok = int(stats.get("input_tokens", stats.get("prompt_tokens", 0)) or 0)
                out_tok = int(stats.get("output_tokens", stats.get("candidates_tokens", 0)) or 0)
            if envelope.get("error"):
                env_error = f"gemini error: {str(envelope['error'])[:500]}"
        error = None
        if proc.returncode != 0:
            error = f"gemini exit {proc.returncode}: {(proc.stderr or '')[:500]}"
        elif env_error:
            error = env_error
        return normalize_result(
            backend_id=self.backend_id,
            adapter_kind=self.adapter_kind,
            model=model,
            text=text,
            raw_stdout=proc.stdout,
            schema=call.schema,
            elapsed_s=elapsed,
            errored=error is not None,
            error=error,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_policy=COST_UNKNOWN,
            schema_mode=SCHEMA_VALIDATE if call.schema else "",
        )
