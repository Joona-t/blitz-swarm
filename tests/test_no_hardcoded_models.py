"""Grep guard: no literal model name in any non-test .py / .toml.

The only files allowed to name a model are the two config files
(blitz.toml, mythos.toml). Everything else names roles and tiers and
resolves through backends.models.resolve_model.

The scan skips pure comment lines; docstrings and string literals count.
Run artifacts (output/, mythos-bench/artifacts/) are data, not code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

ALLOWED_FILES = {"blitz.toml", "mythos.toml"}
SKIP_DIR_PARTS = {
    "tests", "__pycache__", ".git", ".venv", "venv", "node_modules",
    "output", "results", "docs", "artifacts",
}

MODEL_LITERAL = re.compile(
    r"\b(?:opus|sonnet|haiku|fable)\b"      # claude aliases
    r"|\bgpt-[0-9o]"                        # gpt-5.5, gpt-4o
    r"|\bclaude-(?:[0-9]|opus|sonnet|haiku|fable)"
    r"|\bgemini-[0-9]"
    r"|\bqwen[0-9]"
    r"|\bllama[0-9]"
    r"|\bo[1-9]-(?:mini|pro)\b"
)


def candidate_files(root: Path = ROOT) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in (".py", ".toml") or not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.name in ALLOWED_FILES and rel.parent == Path("."):
            continue
        if SKIP_DIR_PARTS & set(rel.parts[:-1]):
            continue
        files.append(path)
    return files


def scan_file(path: Path, root: Path = ROOT) -> list[str]:
    hits: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return hits
    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith("#"):
            continue
        m = MODEL_LITERAL.search(line)
        if m:
            hits.append(f"{path.relative_to(root)}:{n}: {m.group(0)!r} in {line.strip()[:90]}")
    return hits


def scan(root: Path = ROOT) -> list[str]:
    hits: list[str] = []
    for path in candidate_files(root):
        hits.extend(scan_file(path, root))
    return hits


def test_scanner_detects_a_planted_literal(tmp_path):
    (tmp_path / "x.py").write_text('MODEL = "sonnet"  # bad\n# opus in a comment is fine\n', encoding="utf-8")
    (tmp_path / "blitz.toml").write_text('m = "opus"\n', encoding="utf-8")
    (tmp_path / "y.toml").write_text('m = "gpt-5.5"\n', encoding="utf-8")
    hits = scan(tmp_path)
    assert len(hits) == 2, hits
    assert any("x.py:1" in h for h in hits)
    assert any("y.toml:1" in h for h in hits)


def test_wp1_owned_files_are_clean():
    owned = [ROOT / "config.py", *sorted((ROOT / "backends").rglob("*.py"))]
    hits = [h for p in owned for h in scan_file(p)]
    assert hits == [], "\n".join(hits)


def test_no_hardcoded_model_literals_outside_config():
    """WP1 sweep landed: every non-test .py/.toml outside blitz.toml/mythos.toml is clean."""
    hits = scan()
    assert hits == [], "hard-coded model literals (move to blitz.toml tiers):\n" + "\n".join(hits)
