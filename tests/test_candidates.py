"""Deterministic tests for the candidate-extraction module (jobs/candidates.py).

NO test in this file spawns a real subprocess. An autouse fixture replaces
``subprocess.run`` with a tripwire that raises; live-path tests overlay their
own fake ``run``. Contract coverage:

  1. Dry-run is deterministic across calls (and across str|Path mixes) and
     NEVER shells out or reads files (paths passed in do not exist).
  2. Dry-run honors top_n exactly and emits schema-exact candidates.
  3. save/load roundtrip; missing or corrupt file -> [].
  4. Live path parses a fenced-JSON-with-prose fixture via monkeypatched
     subprocess.run, slugifies ids, drops invalid entries, dedupes, caps top_n,
     and passes the exact contract argv + timeout to the CLI.
  5. Live path returns [] (with ONE warning line) on nonzero exit, garbage
     output, timeout, unknown backend, and unreadable inputs.
  6. slugify + validation edge cases.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure the repo root is importable (mirror conftest.py).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import candidates as cand  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_subprocess(monkeypatch):
    """Hard guarantee: nothing in this file may shell out for real."""

    def _forbidden(*a, **k):  # pragma: no cover - only fires on regression
        raise AssertionError(f"real subprocess.run called: {a!r}")

    monkeypatch.setattr(cand.subprocess, "run", _forbidden)


def _fake_run(monkeypatch, *, stdout: str = "", returncode: int = 0,
              raises: BaseException | None = None) -> list[tuple[list, dict]]:
    """Install a fake subprocess.run; returns the captured (argv, kwargs) list."""
    calls: list[tuple[list, dict]] = []

    def _run(argv, *a, **kw):
        calls.append((list(argv), dict(kw)))
        if raises is not None:
            raise raises
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(cand.subprocess, "run", _run)
    return calls


@pytest.fixture
def md_paths(tmp_path: Path) -> list[str]:
    """Two real research md files for live-path tests."""
    a = tmp_path / "research" / "a.md"
    b = tmp_path / "research" / "b.md"
    a.parent.mkdir(parents=True)
    a.write_text("# Report A\nJudge ensembles look promising. UNIQUE_TOKEN_A\n")
    b.write_text("# Report B\nVerifier cascades cut tokens. UNIQUE_TOKEN_B\n")
    return [str(a), str(b)]


# Model reply with prose, a prose bracket, fences, a duplicate id, an entry
# missing required keys, and a bad cost_class — only 3 entries survive.
_FENCED_STDOUT = textwrap.dedent("""\
    Sure! Here are [roughly] the best candidates I found:

    ```json
    [
      {
        "id": "Judge Ensemble V2!",
        "title": "Judge ensemble v2",
        "mechanism_sketch": "Run three cheap judges. Take the median verdict. Calibrate weekly.",
        "blitz_extension_point": "mechanisms/judge-ensemble-v2.py wired into the round loop",
        "expected_gain": "+0.4 avg quality at ~1.1x tokens",
        "cost_class": "Moderate",
        "source_md": "research/a.md"
      },
      {
        "id": "judge-ensemble-v2",
        "title": "Duplicate of the first after slugify",
        "mechanism_sketch": "Same id once slugified. Must be deduped.",
        "blitz_extension_point": "mechanisms/judge-ensemble-v2.py wired into the round loop",
        "expected_gain": "+0.0",
        "cost_class": "cheap",
        "source_md": "research/a.md"
      },
      {
        "title": "Missing id and expected_gain",
        "mechanism_sketch": "Invalid entry. Dropped by validation.",
        "blitz_extension_point": "mechanisms/x.py wired into the round loop",
        "cost_class": "cheap",
        "source_md": "research/a.md"
      },
      {
        "id": "free-lunch",
        "title": "Bad cost class",
        "mechanism_sketch": "cost_class is outside the closed vocabulary. Dropped.",
        "blitz_extension_point": "mechanisms/free-lunch.py wired into the round loop",
        "expected_gain": "+9.9",
        "cost_class": "free",
        "source_md": "research/b.md"
      },
      {
        "id": "verifier-cascade-gate",
        "title": "Verifier cascade gate",
        "mechanism_sketch": "Gate each round behind a cheap verifier. Escalate only on disagreement.",
        "blitz_extension_point": "mechanisms/verifier-cascade-gate.py wired into the round loop",
        "expected_gain": "-15% tokens at equal quality",
        "cost_class": "cheap",
        "source_md": "research/b.md"
      },
      {
        "id": "memory-aware-router",
        "title": "Memory-aware router",
        "mechanism_sketch": "Route subtasks using G-Memory hits. Skip agents with stale context.",
        "blitz_extension_point": "mechanisms/memory-aware-router.py wired into the round loop",
        "expected_gain": "+0.2 avg quality",
        "cost_class": "expensive",
        "source_md": "research/b.md"
      }
    ]
    ```

    Let me know if you want more!
""")


# ---------------------------------------------------------------------------
# 1-2. Dry-run: deterministic, honors top_n, never reads files or shells out
# ---------------------------------------------------------------------------


def test_dry_run_deterministic_across_calls(tmp_path: Path):
    # Paths are NEVER created — proves dry-run reads nothing but the names.
    paths = [tmp_path / "r1.md", str(tmp_path / "r2.md")]  # mixed str | Path
    first = cand.extract_candidates(paths, 4, dry_run=True)
    second = cand.extract_candidates(paths, 4, dry_run=True)
    assert first == second
    # str|Path normalization: pure-str input gives the identical slate.
    assert cand.extract_candidates([str(p) for p in paths], 4, dry_run=True) == first
    # A different path list produces a different (hash-derived) slate.
    other = cand.extract_candidates([tmp_path / "other.md"], 4, dry_run=True)
    assert other != first


def test_dry_run_honors_top_n(tmp_path: Path):
    paths = [str(tmp_path / "r1.md"), str(tmp_path / "r2.md")]
    assert len(cand.extract_candidates(paths, 2, dry_run=True)) == 2
    assert len(cand.extract_candidates(paths, 7, dry_run=True)) == 7
    assert cand.extract_candidates(paths, 0, dry_run=True) == []
    assert cand.extract_candidates([], 5, dry_run=True) == []


def test_dry_run_candidate_shape(tmp_path: Path):
    paths = [str(tmp_path / "r1.md"), str(tmp_path / "r2.md")]
    out = cand.extract_candidates(paths, 5, dry_run=True)
    ids = [c["id"] for c in out]
    assert len(set(ids)) == len(ids)  # unique ids, dedupe-clean
    for i, c in enumerate(out):
        assert set(c) == set(cand._REQUIRED_KEYS)
        assert all(isinstance(v, str) and v for v in c.values())
        assert c["cost_class"] in cand._COST_CLASSES
        assert c["id"] == cand.slugify(c["id"]) and len(c["id"]) <= 40
        assert c["source_md"] == paths[i % len(paths)]


# ---------------------------------------------------------------------------
# 3. save / load roundtrip + missing/corrupt -> []
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path):
    slate = cand.extract_candidates([str(tmp_path / "r.md")], 3, dry_run=True)
    path = tmp_path / "state" / "nested" / "candidates.json"  # parents created
    cand.save_candidates(slate, path)
    assert cand.load_candidates(path) == slate
    assert cand.load_candidates(str(path)) == slate  # str path accepted


def test_load_missing_returns_empty(tmp_path: Path):
    assert cand.load_candidates(tmp_path / "nope.json") == []


@pytest.mark.parametrize(
    "content",
    ["not json {{{", '{"a": 1}', '"just a string"', "", "[1, 2,"],
    ids=["garbage", "object-not-array", "scalar", "empty", "truncated-array"],
)
def test_load_corrupt_returns_empty(tmp_path: Path, content: str):
    p = tmp_path / "candidates.json"
    p.write_text(content)
    assert cand.load_candidates(p) == []


def test_load_drops_non_dict_elements(tmp_path: Path):
    p = tmp_path / "candidates.json"
    p.write_text('[1, "x", {"id": "keep-me"}]')
    assert cand.load_candidates(p) == [{"id": "keep-me"}]


# ---------------------------------------------------------------------------
# 4. Live path: fenced-JSON fixture, validation, dedupe, top_n, exact argv
# ---------------------------------------------------------------------------


def test_live_parses_fenced_json_and_validates(monkeypatch, md_paths):
    calls = _fake_run(monkeypatch, stdout=_FENCED_STDOUT)
    out = cand.extract_candidates(md_paths, 5, timeout_s=77)

    # 6 entries in -> 3 out: slugified, deduped, invalids dropped, order kept.
    assert [c["id"] for c in out] == [
        "judge-ensemble-v2", "verifier-cascade-gate", "memory-aware-router",
    ]
    assert [c["cost_class"] for c in out] == ["moderate", "cheap", "expensive"]
    for c in out:
        assert set(c) == set(cand._REQUIRED_KEYS)

    # Exactly ONE CLI call with the contract argv and the given timeout.
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[0] == "claude" and argv[1] == "-p"
    assert argv[3:] == [
        "--output-format", "text",
        "--model", "sonnet",
        "--disallowedTools", "Bash,Edit,Write,Read,Task,WebFetch,WebSearch",
    ]
    assert kwargs.get("timeout") == 77

    # ONE prompt over all docs: paths + contents + the no-prose instruction.
    prompt = argv[2]
    assert all(p in prompt for p in md_paths)
    assert "UNIQUE_TOKEN_A" in prompt and "UNIQUE_TOKEN_B" in prompt
    assert "no prose" in prompt.lower() and "json array" in prompt.lower()


def test_live_top_n_caps_valid_candidates(monkeypatch, md_paths):
    _fake_run(monkeypatch, stdout=_FENCED_STDOUT)
    out = cand.extract_candidates(md_paths, 2)
    assert [c["id"] for c in out] == ["judge-ensemble-v2", "verifier-cascade-gate"]


def test_live_prompt_truncates_each_doc(monkeypatch, tmp_path: Path):
    big = tmp_path / "big.md"
    big.write_text("A" * (cand._TRUNCATE_CHARS + 100) + "TAIL_MARKER")
    calls = _fake_run(monkeypatch, stdout="[]")
    cand.extract_candidates([str(big)], 3)
    prompt = calls[0][0][2]
    assert "TAIL_MARKER" not in prompt


# ---------------------------------------------------------------------------
# 5. Live failure modes -> [] + one warning line (caller falls back)
# ---------------------------------------------------------------------------


def test_live_nonzero_exit_returns_empty(monkeypatch, md_paths, capsys):
    _fake_run(monkeypatch, stdout=_FENCED_STDOUT, returncode=1)
    assert cand.extract_candidates(md_paths, 5) == []
    err = capsys.readouterr().err.strip()
    assert err.startswith("WARNING:") and "\n" not in err  # exactly one line


@pytest.mark.parametrize(
    "stdout",
    ["utter nonsense, no brackets at all", '{"id": "x"}', "[1, 2", ""],
    ids=["prose", "object-not-array", "truncated", "empty"],
)
def test_live_garbage_output_returns_empty(monkeypatch, md_paths, capsys, stdout):
    _fake_run(monkeypatch, stdout=stdout)
    assert cand.extract_candidates(md_paths, 5) == []
    assert "WARNING:" in capsys.readouterr().err


def test_live_parsable_but_no_valid_entries_returns_empty(monkeypatch, md_paths, capsys):
    _fake_run(monkeypatch, stdout='[1, 2, {"id": "only-key"}]')
    assert cand.extract_candidates(md_paths, 5) == []
    assert "WARNING:" in capsys.readouterr().err


def test_live_timeout_returns_empty(monkeypatch, md_paths, capsys):
    _fake_run(monkeypatch,
              raises=subprocess.TimeoutExpired(cmd=["claude"], timeout=1))
    assert cand.extract_candidates(md_paths, 5, timeout_s=1) == []
    assert "WARNING:" in capsys.readouterr().err


def test_live_missing_cli_returns_empty(monkeypatch, md_paths, capsys):
    _fake_run(monkeypatch, raises=FileNotFoundError("claude not on PATH"))
    assert cand.extract_candidates(md_paths, 5) == []
    assert "WARNING:" in capsys.readouterr().err


def test_live_unknown_backend_returns_empty(md_paths, capsys):
    # Autouse tripwire still active: proves NO subprocess call is attempted.
    assert cand.extract_candidates(md_paths, 5, backend="codex") == []
    assert "unknown backend" in capsys.readouterr().err


def test_live_unreadable_inputs_return_empty(tmp_path: Path, capsys):
    # Files don't exist; tripwire proves we fail before any subprocess call.
    missing = [str(tmp_path / "ghost.md")]
    assert cand.extract_candidates(missing, 5) == []
    assert "WARNING:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 6. slugify + validation edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello, World!", "hello-world"),
        ("  --Multi   space__and.dots--  ", "multi-space-and-dots"),
        ("UPPER-case-OK", "upper-case-ok"),
        ("émile's café", "mile-s-caf"),  # ASCII-only by design
        ("", ""),
        ("!!!", ""),
        ("a" * 60, "a" * 40),  # hard cap at 40
    ],
)
def test_slugify_edge_cases(raw: str, expected: str):
    assert cand.slugify(raw) == expected


def test_slugify_truncation_never_ends_with_hyphen():
    slug = cand.slugify("a" * 39 + "-bcd")  # cut would land on the hyphen
    assert slug == "a" * 39
    assert not slug.endswith("-")


def test_validation_drops_entries_missing_required_keys():
    full = {
        "id": "Good Candidate",
        "title": "Good candidate",
        "mechanism_sketch": "Does a thing. Helps quality.",
        "blitz_extension_point": "mechanisms/good-candidate.py wired into the round loop",
        "expected_gain": "+0.1",
        "cost_class": "CHEAP",
        "source_md": "research/a.md",
    }
    valid = cand._validate_candidate(full)
    assert valid is not None
    assert valid["id"] == "good-candidate"      # slugified
    assert valid["cost_class"] == "cheap"        # normalized lowercase
    # Dropping ANY single required key invalidates the entry.
    for key in cand._REQUIRED_KEYS:
        broken = {k: v for k, v in full.items() if k != key}
        assert cand._validate_candidate(broken) is None, key
    # Non-dict / empty-value / bad-enum / unsluggable entries also drop.
    assert cand._validate_candidate("not a dict") is None
    assert cand._validate_candidate({**full, "title": "   "}) is None
    assert cand._validate_candidate({**full, "cost_class": "free"}) is None
    assert cand._validate_candidate({**full, "id": "!!!", "title": "???"}) is None
    # id falls back to slugified title when the id itself is unsluggable.
    fallback = cand._validate_candidate({**full, "id": "!!!"})
    assert fallback is not None and fallback["id"] == "good-candidate"


def test_parse_json_array_leniency():
    assert cand._parse_json_array('[{"a": 1}]') == [{"a": 1}]
    assert cand._parse_json_array('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert cand._parse_json_array('See [these] results:\n[{"a": 1}]\nok?') == [{"a": 1}]
    assert cand._parse_json_array('{"candidates": [{"a": 1}]}') == [{"a": 1}]
    assert cand._parse_json_array("no array here") is None
    assert cand._parse_json_array("[broken") is None
