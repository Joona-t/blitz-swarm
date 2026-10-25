# manifest-validator Report: Blitz Swarm

_Generated: 2026-03-12 | Agent: manifest-validator | Reporting to: Matz (Code Quality Lead)_

---

## Summary

Blitz Swarm is structurally well-organized for v0.1.0. Core files are present and internally consistent. Several meaningful deviations from BLITZ-SWARM.md spec and configuration/packaging gaps need attention.

---

## pyproject.toml

- **Valid metadata:** Yes — name, version, description, license, authors all present
- **Python version:** `>=3.12` — correct
- **ISSUE — `aiosqlite` declared but unused.** All SQLite uses synchronous `sqlite3` stdlib
- **ISSUE — `pandas` used but undeclared.** `.to_pandas()` called on LanceDB results
- **ISSUE — No `[build-system]` table.** Can't `pip install -e .`
- **ISSUE — No `[project.scripts]` entry point.** No `blitz-swarm` command after install
- **Dev deps:** pytest/pytest-asyncio listed but no test files exist yet

## blitz.toml

- **Valid TOML:** Yes
- **All values match config.py defaults:** 100% sync (20/20 fields)
- **ISSUE — config.py is never imported in orchestrator.py.** Constants are hardcoded directly, bypassing the entire configuration system. `blitz.toml` + `config.py` have zero effect at runtime.

## Project Structure vs Spec

| Item | Status |
|------|--------|
| All spec-defined files | PRESENT |
| `config.py` | PRESENT — not in spec |
| `memory/__init__.py` | PRESENT — not in spec |
| `docs/` directory | MISSING — spec references two research docs that don't exist |
| compass_artifact files | STRAY — gitignored correctly |

## README Accuracy

- CLI options documented correctly
- **ISSUE — `--verbose` flag is parsed but silently does nothing**
- **ISSUE — BLITZ-SWARM.md says Researcher 2-4, Critic 1-2. Code allows 2-6, 1-3. README matches code.**
- Quickstart, architecture, memory docs all accurate

## .gitignore

- Well-covered for Python/macOS/output
- **Missing:** `.pytest_cache/`, `.env`, `_reports/`

## Verdict: ISSUES

### High Priority
1. Remove `aiosqlite` from deps or actually use it
2. Add `pandas` to deps or avoid `.to_pandas()`
3. Wire `load_config()` into orchestrator — config system is built but bypassed

### Medium Priority
4. Implement `--verbose` or remove it
5. Update BLITZ-SWARM.md structure + agent counts to match code

### Low Priority
6. Add `.pytest_cache/` to .gitignore
7. Add `[build-system]` table for future packaging
