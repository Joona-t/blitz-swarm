"""Command-template rendering for the generic_cli adapter, plus the shape
every one-call harness adapter follows.

A provider that wraps any CLI needs no code: it declares an argv template in
blitz.toml and the placeholders below are filled per call.

    [backend.providers.mycli]
    adapter = "generic_cli"
    command = ["mycli", "--model", "{model}", "--prompt-file", "{prompt_file}"]

Placeholders:

    {model}          resolved model string
    {prompt}         full prompt text (system + user when {system_prompt} is unused)
    {prompt_file}    path of a temp file holding the same prompt
    {system_prompt}  system prompt alone (prompt then holds only the user task)
    {schema_file}    path of a temp JSON file with the output schema
    {output_file}    path the CLI may write its final answer to; read if present
    {effort}         per-tier effort value
    {cwd}            working directory of the call

If neither {prompt} nor {prompt_file} appears, the prompt is piped on stdin.
An argv element that references a placeholder with no value (no schema, no
effort, ...) is dropped together with the bare flag that precedes it, so
`["--schema", "{schema_file}"]` vanishes cleanly on schema-less calls.
"""

from __future__ import annotations

import re

from backends.core import AgentBackend, AgentCall, AgentResult

PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")
KNOWN_PLACEHOLDERS = (
    "model", "prompt", "prompt_file", "system_prompt",
    "schema_file", "output_file", "effort", "cwd",
)


def placeholders_in(template: list[str]) -> set[str]:
    found: set[str] = set()
    for arg in template:
        found.update(PLACEHOLDER_RE.findall(arg))
    return found


def render_command(template: list[str], values: dict[str, str | None]) -> list[str]:
    """Fill `template` from `values`; see module doc for the drop rule."""
    rendered: list[str] = []
    for arg in template:
        names = PLACEHOLDER_RE.findall(arg)
        unknown = [n for n in names if n not in KNOWN_PLACEHOLDERS]
        if unknown:
            raise ValueError(f"unknown placeholder(s) {unknown} in command template element {arg!r}")
        if any(values.get(n) in (None, "") for n in names):
            # Drop this element and a bare preceding flag ("--schema").
            if rendered and rendered[-1].startswith("-") and "=" not in rendered[-1]:
                rendered.pop()
            continue
        out = arg
        for n in names:
            out = out.replace("{" + n + "}", str(values[n]))
        rendered.append(out)
    return rendered


class OneCallHarnessAdapterTemplate:
    """Minimal shape for future harness adapters (OpenAI Agents SDK,
    LangGraph, CrewAI, Aider ...).

    Keep the public contract stateless: AgentCall -> adapter-owned
    execution -> AgentResult. If the harness has sessions, threads, memory,
    graphs or crews, keep that state inside the adapter and honour
    `AgentCall.ephemeral` by discarding it before returning. Normalise every
    harness-specific failure into an errored AgentResult; never raise.
    Register with `backends.register_backend("<adapter_id>", factory)`.
    """

    backend_id = "template"
    adapter_kind = "template"

    def is_available(self) -> bool:
        return False

    def call(self, call: AgentCall) -> AgentResult:
        raise NotImplementedError("Copy this template into a concrete adapter.")


__all__ = [
    "AgentBackend",
    "AgentCall",
    "AgentResult",
    "KNOWN_PLACEHOLDERS",
    "OneCallHarnessAdapterTemplate",
    "placeholders_in",
    "render_command",
]
