"""Cross-CLI router — selects which local CLI handles each role call.

Configuration: a TOML routing table maps `(domain, role)` -> `cli_id`.
Default fallback is `claude`. Adapters are detected at construction
time via `shutil.which`; missing CLIs gracefully fall back.

All adapters use subprocess invocation. No CLI is required for testing —
the adapters are pluggable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol


@dataclass(slots=True)
class CLICall:
    cli_id: str
    model: str | None
    prompt: str
    schema: dict | None
    timeout_s: int


@dataclass(slots=True)
class CLIResult:
    cli_id: str
    text: str
    structured: dict | None
    elapsed_s: float
    fallback_used: bool
    errored: bool = False


class CLIAdapter(Protocol):
    cli_id: str
    def is_available(self) -> bool: ...
    def call(self, c: CLICall) -> CLIResult: ...


def _try_parse_json(stdout: str) -> dict | None:
    if not stdout:
        return None
    try:
        return json.loads(stdout.strip())
    except json.JSONDecodeError:
        # Attempt to find embedded JSON
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stdout[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


@dataclass
class ClaudeCLI:
    cli_id: str = "claude"

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    def call(self, c: CLICall) -> CLIResult:
        cmd = ["claude", "-p", c.prompt]
        if c.model:
            cmd.extend(["--model", c.model])
        if c.schema:
            cmd.extend(["--json-schema", json.dumps(c.schema)])
        return _run_subprocess(cmd, c, cli_id=self.cli_id)


@dataclass
class CodexCLI:
    cli_id: str = "codex"

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    def call(self, c: CLICall) -> CLIResult:
        cmd = ["codex", "exec", "-"]
        return _run_subprocess(cmd, c, cli_id=self.cli_id, stdin_input=c.prompt)


@dataclass
class GeminiCLI:
    cli_id: str = "gemini"

    def is_available(self) -> bool:
        return shutil.which("gemini") is not None

    def call(self, c: CLICall) -> CLIResult:
        cmd = ["gemini", "-p", c.prompt]
        return _run_subprocess(cmd, c, cli_id=self.cli_id)


def _run_subprocess(
    cmd: list[str], c: CLICall, *, cli_id: str, stdin_input: str | None = None,
) -> CLIResult:
    t0 = time.monotonic()
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=c.timeout_s,
            input=stdin_input,
        )
        elapsed = time.monotonic() - t0
        if out.returncode != 0:
            return CLIResult(
                cli_id=cli_id,
                text=(out.stderr or "")[:500],
                structured=None,
                elapsed_s=elapsed,
                fallback_used=False,
                errored=True,
            )
        return CLIResult(
            cli_id=cli_id,
            text=out.stdout,
            structured=_try_parse_json(out.stdout),
            elapsed_s=elapsed,
            fallback_used=False,
        )
    except subprocess.TimeoutExpired:
        return CLIResult(
            cli_id=cli_id, text="", structured=None,
            elapsed_s=time.monotonic() - t0,
            fallback_used=False, errored=True,
        )
    except Exception as e:
        return CLIResult(
            cli_id=cli_id, text=f"exception: {e!r}", structured=None,
            elapsed_s=time.monotonic() - t0,
            fallback_used=False, errored=True,
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class RouteEntry:
    domain: str
    role: str
    cli_id: str


class CLIRouter:
    """Domain × role -> CLI routing with graceful fallback."""

    def __init__(
        self,
        routing_table: list[RouteEntry] | None = None,
        *,
        adapters: list[CLIAdapter] | None = None,
        fallback_cli: str = "claude",
    ):
        self.routing_table = routing_table or []
        if adapters is None:
            adapters = [ClaudeCLI(), CodexCLI(), GeminiCLI()]
        self.adapters: dict[str, CLIAdapter] = {
            a.cli_id: a for a in adapters if a.is_available()
        }
        self.fallback_cli = fallback_cli

    @classmethod
    def from_toml(cls, path: Path, *, adapters: list[CLIAdapter] | None = None) -> "CLIRouter":
        """Parse a routing-table TOML file.

        Schema:
            [default]
            <role> = "<cli_id>"
            ...

            [domain.<name>]
            <role> = "<cli_id>"
            ...
        """
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        entries: list[RouteEntry] = []
        defaults = raw.get("default", {}) or {}
        for role, cli_id in defaults.items():
            entries.append(RouteEntry(domain="*", role=role, cli_id=cli_id))
        domain_block = raw.get("domain", {}) or {}
        if isinstance(domain_block, dict):
            for domain_name, role_map in domain_block.items():
                if not isinstance(role_map, dict):
                    continue
                for role, cli_id in role_map.items():
                    entries.append(RouteEntry(domain=domain_name, role=role, cli_id=cli_id))
        return cls(entries, adapters=adapters)

    def route(
        self,
        role: str,
        domain: str,
        prompt: str,
        *,
        schema: dict | None = None,
        timeout_s: int = 120,
        model: str | None = None,
    ) -> CLIResult:
        cli_id = self._lookup(role, domain) or self.fallback_cli
        adapter = self.adapters.get(cli_id)
        fallback = False
        if adapter is None:
            adapter = self.adapters.get(self.fallback_cli)
            fallback = True
        if adapter is None:
            # No CLI available at all
            return CLIResult(
                cli_id=cli_id, text="no CLI available", structured=None,
                elapsed_s=0.0, fallback_used=True, errored=True,
            )
        c = CLICall(
            cli_id=adapter.cli_id, model=model, prompt=prompt,
            schema=schema, timeout_s=timeout_s,
        )
        result = adapter.call(c)
        result.fallback_used = fallback
        return result

    def _lookup(self, role: str, domain: str) -> str | None:
        # Exact (domain, role) match first
        for entry in self.routing_table:
            if entry.domain == domain and entry.role == role:
                return entry.cli_id
        # Then (*, role) default
        for entry in self.routing_table:
            if entry.domain == "*" and entry.role == role:
                return entry.cli_id
        return None

    def available_clis(self) -> list[str]:
        return sorted(self.adapters.keys())
