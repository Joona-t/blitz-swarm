"""Generic CLI adapter: any command-line model tool via an argv template.

Configured entirely from blitz.toml (`adapter = "generic_cli"`, `command =
[...]`); see `backends.adapters.templates` for the placeholders.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from backends.adapters.templates import placeholders_in, render_command
from backends.core import (
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


@dataclass
class GenericCLIBackend:
    command: list[str] = field(default_factory=list)
    model: str | None = None
    reasoning_effort: str | None = None
    backend_id: str = "generic"
    adapter_kind: str = "generic_cli"
    metadata: dict = field(default_factory=dict)

    capabilities = BackendCapabilities(adapter_kind="generic_cli", local=True)

    def is_available(self) -> bool:
        return bool(self.command) and shutil.which(self.command[0]) is not None

    def _error(self, model: str | None, error: str, *, elapsed_s: float = 0.0) -> AgentResult:
        return AgentResult(
            backend_id=self.backend_id, adapter_kind=self.adapter_kind, model=model,
            text="", raw_stdout="", parsed=None, elapsed_s=elapsed_s,
            errored=True, error=error,
        )

    def call(self, call: AgentCall) -> AgentResult:
        model = call.model or self.model
        if not self.command:
            return self._error(model, f"provider {self.backend_id!r} has no 'command' template")
        if not self.is_available():
            return self._error(model, f"{self.command[0]} not found on PATH")
        if not model:
            return self._error(None, NO_MODEL_ERROR)

        used = placeholders_in(self.command)
        prompt = call.prompt if "system_prompt" in used else combined_prompt(call)
        effort = call.reasoning_effort or self.reasoning_effort

        with tempfile.TemporaryDirectory(prefix="blitz-generic-") as td:
            tmp = Path(td)
            prompt_file = tmp / "prompt.md"
            prompt_file.write_text(prompt, encoding="utf-8")
            output_file = tmp / "output.txt"
            schema_file: Path | None = None
            if call.schema:
                schema_file = tmp / "schema.json"
                schema_file.write_text(json.dumps(call.schema), encoding="utf-8")

            values = {
                "model": model,
                "prompt": prompt,
                "prompt_file": str(prompt_file),
                "system_prompt": call.system_prompt or None,
                "schema_file": str(schema_file) if schema_file else None,
                "output_file": str(output_file),
                "effort": effort or None,
                "cwd": str(call.cwd) if call.cwd else None,
            }
            try:
                cmd = render_command(self.command, values)
            except ValueError as exc:
                return self._error(model, f"bad command template: {exc}")
            stdin_input = None if used & {"prompt", "prompt_file"} else prompt

            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    input=stdin_input,
                    capture_output=True,
                    text=True,
                    timeout=call.timeout_s,
                    cwd=str(call.cwd) if call.cwd else None,
                )
            except subprocess.TimeoutExpired:
                return self._error(
                    model, f"{cmd[0]} timed out after {call.timeout_s}s",
                    elapsed_s=time.monotonic() - start,
                )
            except Exception as exc:
                return self._error(
                    model, f"{cmd[0]} subprocess exception: {exc!r}",
                    elapsed_s=time.monotonic() - start,
                )
            elapsed = time.monotonic() - start

            text = ""
            try:
                if output_file.exists():
                    text = output_file.read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
            if not text:
                text = proc.stdout.strip()
            return normalize_result(
                backend_id=self.backend_id,
                adapter_kind=self.adapter_kind,
                model=model,
                text=text,
                raw_stdout=proc.stdout,
                schema=call.schema,
                elapsed_s=elapsed,
                errored=proc.returncode != 0,
                error=(
                    f"{cmd[0]} exit {proc.returncode}: {(proc.stderr or '')[:500]}"
                    if proc.returncode != 0 else None
                ),
                metadata={"command": cmd[0], "effort": effort},
            )


__all__ = ["GenericCLIBackend"]
