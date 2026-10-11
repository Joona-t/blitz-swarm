#!/usr/bin/env python3
"""Migrate a blitz-swarm memory.db to the canonical gmemory schema.

    python scripts/migrate_memory_db.py                 # [storage] sqlite_path from blitz.toml
    python scripts/migrate_memory_db.py path/to/db      # explicit file
    python scripts/migrate_memory_db.py --check path    # report only, no writes

Idempotent: a database already on the canonical schema is left as it is
(only `CREATE ... IF NOT EXISTS` statements run). A legacy memory/-schema
database is backed up next to itself (`<name>.pre-gmemory.bak`) before its
tables are rebuilt. stdlib only; no model call.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gmemory.migrate import is_legacy, migrate_db  # noqa: E402


def default_db_path() -> Path:
    try:
        from config import load_config  # noqa: PLC0415
        p = Path(load_config().storage.sqlite_path)
    except Exception:
        p = Path("memory.db")
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="SQLite file (default: [storage] sqlite_path)")
    ap.add_argument("--check", action="store_true", help="report the schema state, write nothing")
    ap.add_argument("--no-backup", action="store_true", help="skip the .pre-gmemory.bak copy")
    args = ap.parse_args(argv)

    path = Path(args.path) if args.path else default_db_path()
    if not path.exists():
        print(f"no database at {path}; nothing to migrate (it is created on first run)")
        return 0

    db = sqlite3.connect(str(path))
    try:
        legacy = is_legacy(db)
    finally:
        db.close()

    if args.check:
        print(f"{path}: {'LEGACY memory/ schema -> migration needed' if legacy else 'canonical gmemory schema'}")
        return 1 if legacy else 0

    if legacy and not args.no_backup:
        backup = path.with_name(path.name + ".pre-gmemory.bak")
        shutil.copy2(path, backup)
        print(f"backup: {backup}")

    db = sqlite3.connect(str(path))
    try:
        report = migrate_db(db)
    finally:
        db.close()
    kind = "migrated legacy schema" if report["legacy"] else "already canonical (schema re-applied)"
    print(f"{path}: {kind}")
    for key in ("queries", "query_edges", "insights", "insight_edges", "utterances",
                "utterance_edges", "queries_archive"):
        if report.get(key):
            print(f"  {key:16s} {report[key]}")
    if report.get("columns_added"):
        print(f"  columns added: {', '.join(report['columns_added'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
