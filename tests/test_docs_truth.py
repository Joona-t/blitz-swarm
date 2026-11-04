"""WP9: the README's "Supported harnesses" table is generated from the same
facts the code declares. Every registered adapter must appear with its
declared schema / tools / cost words; a stale row fails the build."""

from __future__ import annotations

import re
from pathlib import Path

import backends
from backends.adapters.acp import ACPBackend
from backends.adapters.cli import ClaudeCLIBackend, CodexLocalBackend, GeminiCLIBackend
from backends.adapters.generic_cli import GenericCLIBackend
from backends.adapters.ollama import OllamaHTTPBackend
from backends.adapters.openai_http import OpenAIHTTPBackend

README = Path(__file__).resolve().parents[1] / "README.md"

DECLARED = {
    "acp": ACPBackend(command=["x"]).capabilities,
    "claude_cli": ClaudeCLIBackend().capabilities,
    "codex_cli": CodexLocalBackend().capabilities,
    "gemini_cli": GeminiCLIBackend().capabilities,
    "generic_cli": GenericCLIBackend(command=["x", "{prompt_file}"]).capabilities,
    "ollama_http": OllamaHTTPBackend().capabilities,
    "openai_http": OpenAIHTTPBackend(base_url="http://127.0.0.1:1").capabilities,
}


def _table_rows() -> dict[str, list[str]]:
    text = README.read_text(encoding="utf-8")
    m = re.search(r"### Supported harnesses.*?\n\n(\|.*?)\n\n", text, re.DOTALL)
    assert m, "README lacks a '### Supported harnesses' table"
    rows = {}
    for line in m.group(1).splitlines()[2:]:
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells
    return rows


def test_every_registered_adapter_has_a_readme_row():
    rows = _table_rows()
    builtins = {b for b in backends.registered_backends() if not b.startswith(("fake", "test"))}
    assert builtins <= set(rows), f"missing README rows: {builtins - set(rows)}"


def test_readme_rows_match_declared_capabilities():
    rows = _table_rows()
    for adapter_id, caps in DECLARED.items():
        cells = rows[adapter_id]
        assert cells[1] == caps.schema_policy, (adapter_id, "schema")
        assert cells[2] == caps.tool_policy, (adapter_id, "tools")
        assert cells[3] == caps.cost_policy, (adapter_id, "cost")
