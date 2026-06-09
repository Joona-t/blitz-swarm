#!/usr/bin/env python3
"""Candidate extraction from research-phase markdown for the upgrade job.

Turns a pile of research-swarm reports into a ranked, structured slate of
implementation candidates for the implement phase. ONE LLM call over all
reports (each truncated), strict-JSON out, lenient parse in, validate,
dedupe by id, return the top N.

    research *.md  ->  extract_candidates()  ->  [candidate dict, ...]

Candidate schema (every value a non-empty ``str``)::

    {
      "id":                    kebab-case slug (<=40 chars),
      "title":                 short human title,
      "mechanism_sketch":      2-5 sentences on what the mechanism does + why,
      "blitz_extension_point": e.g. "mechanisms/<id>.py wired into the round loop",
      "expected_gain":         expected quality/cost effect,
      "cost_class":            "cheap" | "moderate" | "expensive",
      "source_md":             path of the source report,
    }

DRY-RUN CONTRACT (critical for testing):
    ``dry_run=True`` returns deterministic fake candidates derived from a
    stable SHA-256 hash of the input path list (mirror of
    ``jobs.blitz_upgrade._stable_hash`` — PYTHONHASHSEED-independent) and
    NEVER touches ``subprocess``, the LLM, or the filesystem beyond the
    paths' *names*. Identical path lists produce identical candidates
    across processes, OSes and Python builds.

LIVE CONTRACT (fail-soft):
    One ``claude -p`` subprocess over a single extraction prompt covering
    every readable report. On ANY failure — unknown backend, timeout,
    nonzero exit, unparseable output, nothing valid after validation —
    print ONE warning line and return ``[]`` so the caller falls back to
    its default technique slate. This function never raises for backend
    misbehaviour.

Pure stdlib only: hashlib, json, re, subprocess, pathlib.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants — schema, cost classes, CLI wiring
# ---------------------------------------------------------------------------

# Exact key set of a candidate dict; validation requires every one non-empty.
_REQUIRED_KEYS: tuple[str, ...] = (
    "id",
    "title",
    "mechanism_sketch",
    "blitz_extension_point",
    "expected_gain",
    "cost_class",
    "source_md",
)

# Closed vocabulary for "cost_class" (compared lowercase).
_COST_CLASSES: tuple[str, ...] = ("cheap", "moderate", "expensive")

# Per-report truncation cap so the single extraction prompt stays bounded
# even when the research phase produced very long reports.
_TRUNCATE_CHARS: int = 12_000

# Live CLI wiring (rule: user's own CLI subscription via subprocess — never a
# paid API). The extraction call is text-in/text-out with all tools disabled.
_CLI_MODEL: str = "sonnet"
_CLI_DISALLOWED_TOOLS: str = "Bash,Edit,Write,Read,Task,WebFetch,WebSearch"
DEFAULT_TIMEOUT_S: int = 240

# Slug cap — ids feed file names (mechanisms/<id>.py) and manifest cell ids.
_SLUG_MAX_LEN: int = 40

# Theme bank for dry-run fakes (content is fake; the SHAPE is contract-true).
_DRY_THEMES: tuple[str, ...] = (
    "debate-routing",
    "verifier-cascade",
    "memory-distill",
    "selector-synthesis",
    "judge-calibration",
    "round-early-exit",
)

# JSON schema block embedded verbatim in the extraction prompt.
_SCHEMA_BLOCK: str = """\
  {
    "id": "kebab-case-slug",
    "title": "short human title",
    "mechanism_sketch": "2-5 sentences: what the mechanism does and why it should lift quality",
    "blitz_extension_point": "where it plugs in, e.g. 'mechanisms/<id>.py wired into the round loop'",
    "expected_gain": "expected effect, e.g. '+0.3 avg quality at ~1.1x tokens'",
    "cost_class": "cheap | moderate | expensive",
    "source_md": "path of the source document"
  }"""


# ---------------------------------------------------------------------------
# Small helpers — slug, stable hash, warnings
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Kebab-case slug: lowercase, ASCII alnum + hyphen, <=40 chars.

    Runs of anything else collapse to a single hyphen; leading/trailing
    hyphens are stripped (again after truncation so a cut never ends ``-``).
    Returns ``""`` when nothing survives — callers treat that as invalid.
    """
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s[:_SLUG_MAX_LEN].rstrip("-")


def _stable_hash(key: Any) -> int:
    """Process-stable hash of an arbitrary key.

    Mirror of ``jobs.blitz_upgrade._stable_hash`` (B4): the builtin ``hash()``
    is salted by ``PYTHONHASHSEED``, which would break the dry-run determinism
    contract across processes. SHA-256 of ``repr(key)`` truncated to 32 bits
    is identical across processes, OSes and Python builds.
    """
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _warn(msg: str) -> None:
    """Emit ONE clear warning line to stderr (fail-soft contract)."""
    print(f"WARNING: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public API — extract / save / load
# ---------------------------------------------------------------------------


def extract_candidates(
    research_md_paths: list[str | Path],
    top_n: int,
    *,
    dry_run: bool = False,
    backend: str = "claude",
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Extract a ranked candidate slate from research markdown reports.

    Dry-run returns deterministic fakes (no subprocess, no file reads —
    see module docstring). Live mode makes one ``claude -p`` call over all
    readable reports and parses/validates the JSON reply. On any live
    failure this returns ``[]`` after one warning line — never raises.

    Ranking preserves model order (best-first by instruction), dedupes by
    slugified ``id``, and caps at ``top_n``.
    """
    paths = [str(p) for p in research_md_paths]
    n = max(0, int(top_n))
    if n == 0 or not paths:
        return []
    if dry_run:
        return _dry_run_candidates(paths, n)
    return _live_candidates(paths, n, backend=backend, timeout_s=timeout_s)


def save_candidates(cands: list[dict[str, Any]], path: str | Path) -> None:
    """Atomically persist a candidate slate as pretty JSON (parents created)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(list(cands), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    """Load a saved slate; ``[]`` if the file is missing or corrupt.

    Corrupt means unreadable, non-JSON, or not a JSON array. Non-dict
    elements inside an otherwise-valid array are dropped, not fatal.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError covers JSON + Unicode errors
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


# ---------------------------------------------------------------------------
# Dry-run path — deterministic fakes, no subprocess, no file reads
# ---------------------------------------------------------------------------


def _dry_run_candidates(paths: list[str], top_n: int) -> list[dict[str, Any]]:
    """Deterministic fake slate derived from a stable hash of the path list.

    Every field is a pure function of ``(paths, i)`` via :func:`_stable_hash`,
    so two calls — even in separate processes — produce identical output.
    ``source_md`` cycles through the given paths; the files are NEVER read.
    """
    run_h = _stable_hash(tuple(paths))
    out: list[dict[str, Any]] = []
    for i in range(top_n):
        h = _stable_hash((run_h, i))
        theme = _DRY_THEMES[h % len(_DRY_THEMES)]
        cid = slugify(f"{theme}-{run_h & 0xFFFF:04x}-{i + 1:02d}")
        out.append(
            {
                "id": cid,
                "title": f"Dry-run candidate {i + 1}: {theme}",
                "mechanism_sketch": (
                    f"Deterministic dry-run sketch for '{theme}'. Every field "
                    f"derives from a stable hash of the input path list "
                    f"({run_h & 0xFFFF:04x}). No file was read and no "
                    f"subprocess was spawned."
                ),
                "blitz_extension_point": (
                    f"mechanisms/{cid}.py wired into the round loop"
                ),
                "expected_gain": f"+{0.5 + (h % 15) / 10:.1f} avg quality (simulated)",
                "cost_class": _COST_CLASSES[h % len(_COST_CLASSES)],
                "source_md": paths[i % len(paths)],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Live path — one claude -p call, lenient parse, strict validation
# ---------------------------------------------------------------------------


def _live_candidates(
    paths: list[str], top_n: int, *, backend: str, timeout_s: int
) -> list[dict[str, Any]]:
    """Real-mode extraction. Fail-soft: any failure -> one warning + ``[]``."""
    argv_tail = _build_argv_tail(backend)
    if argv_tail is None:
        _warn(f"candidate extraction: unknown backend {backend!r}; returning []")
        return []
    docs = _read_docs(paths)
    if not docs:
        _warn("candidate extraction: no readable research md files; returning []")
        return []
    prompt = _build_prompt(docs)
    argv = [backend, "-p", prompt, *argv_tail]
    try:
        proc = subprocess.run(  # noqa: PLW1510  (we inspect returncode below)
            argv, capture_output=True, text=True, timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        _warn(
            f"candidate extraction: {backend} invocation failed "
            f"({type(exc).__name__}); returning [] — caller falls back"
        )
        return []
    if proc.returncode != 0:
        _warn(
            f"candidate extraction: {backend} exited {proc.returncode}; "
            f"returning [] — caller falls back"
        )
        return []
    raw = _parse_json_array(proc.stdout or "")
    if raw is None:
        _warn(
            "candidate extraction: unparseable model output (no JSON array); "
            "returning [] — caller falls back"
        )
        return []
    ranked = _rank(raw, top_n)
    if not ranked:
        _warn(
            "candidate extraction: model output had no valid candidates; "
            "returning [] — caller falls back"
        )
    return ranked


def _build_argv_tail(backend: str) -> list[str] | None:
    """Flags appended after ``[backend, '-p', prompt]``; None if unsupported.

    Only the ``claude`` CLI is wired (text-out, sonnet, all tools disabled —
    pure extraction, the model must not touch the repo). Unknown backends are
    a fail-soft ``[]`` upstream, not an exception.
    """
    if backend == "claude":
        return [
            "--output-format", "text",
            "--model", _CLI_MODEL,
            "--disallowedTools", _CLI_DISALLOWED_TOOLS,
        ]
    return None


def _read_docs(paths: list[str]) -> list[tuple[str, str]]:
    """Read each report, truncated to ``_TRUNCATE_CHARS``; skip unreadables."""
    docs: list[tuple[str, str]] = []
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = text[:_TRUNCATE_CHARS]
        if text.strip():
            docs.append((p, text))
    return docs


def _build_prompt(docs: list[tuple[str, str]]) -> str:
    """One extraction prompt over ALL reports, demanding a bare JSON array."""
    n = len(docs)
    parts: list[str] = [
        "You are mining research reports for upgrade candidates for "
        "blitz-swarm, a parallel multi-agent LLM swarm with a consensus "
        "round loop, pluggable mechanisms/ modules, and a statistically "
        "gated bench.\n\n"
        f"Extract the strongest, most implementable upgrade candidates "
        f"across all {n} documents below, ordered best-first.\n\n"
        "Output contract — follow it EXACTLY:\n"
        "- Output ONLY a JSON array of candidate objects. No prose, no "
        "fences, no markdown: the first character of your reply must be "
        "'[' and the last must be ']'.\n"
        "- Every object has exactly these keys, all values non-empty "
        "strings:\n"
        f"{_SCHEMA_BLOCK}\n"
        '- "cost_class" must be exactly one of: cheap, moderate, expensive.\n'
        '- "id" must be a kebab-case slug, unique across the array.\n'
        '- "source_md" must be the path shown in the header of the document '
        "the candidate came from.\n"
    ]
    for i, (path, text) in enumerate(docs, start=1):
        parts.append(f"\n--- document {i}/{n}: {path} ---\n{text}\n")
    return "".join(parts)


def _parse_json_array(text: str) -> list[Any] | None:
    """Lenient JSON-array parse: strip fences, find the first ``[...]`` block.

    Models add fences and prose despite instructions. Strategy: drop fence
    lines, try a direct parse, then scan each ``[`` and ``raw_decode`` from
    it until a JSON array decodes (skips prose brackets like "[roughly]").
    Returns None when no array can be decoded — the fail-soft signal.
    """
    t = _strip_fences(text.strip()).strip()
    if not t:
        return None
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, list):
        return obj
    dec = json.JSONDecoder()
    pos = t.find("[")
    while pos != -1:
        try:
            found, _ = dec.raw_decode(t, pos)
        except json.JSONDecodeError:
            pos = t.find("[", pos + 1)
            continue
        return found if isinstance(found, list) else None
    return None


def _strip_fences(text: str) -> str:
    """Drop markdown code-fence lines (``` / ```json) from a model reply."""
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("```")
    )


def _validate_candidate(entry: Any) -> dict[str, Any] | None:
    """Validate + normalize one model-emitted entry; None when invalid.

    Required: every schema key present as a non-empty string; ``cost_class``
    (lowercased) in the closed vocabulary; ``id`` slugifies non-empty (falls
    back to slugified ``title``). Extra keys are discarded so the output
    schema is exact.
    """
    if not isinstance(entry, dict):
        return None
    vals: dict[str, str] = {}
    for key in _REQUIRED_KEYS:
        v = entry.get(key)
        if not isinstance(v, str) or not v.strip():
            return None
        vals[key] = v.strip()
    cost = vals["cost_class"].lower()
    if cost not in _COST_CLASSES:
        return None
    cid = slugify(vals["id"]) or slugify(vals["title"])
    if not cid:
        return None
    return {**vals, "id": cid, "cost_class": cost}


def _rank(raw: list[Any], top_n: int) -> list[dict[str, Any]]:
    """Validate each entry, dedupe by id preserving model order, cap top_n."""
    seen: set[str] = set()
    ranked: list[dict[str, Any]] = []
    for entry in raw:
        cand = _validate_candidate(entry)
        if cand is None or cand["id"] in seen:
            continue
        seen.add(cand["id"])
        ranked.append(cand)
        if len(ranked) >= top_n:
            break
    return ranked
