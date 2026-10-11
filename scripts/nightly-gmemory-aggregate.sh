#!/usr/bin/env bash
# Nightly G-Memory aggregation (docs/research/PHASE_2_GMEMORY.md §7.2).
#
# Runs the Tier-3 promotion sweep (`InsightGraph.aggregate_overnight`) against
# the SQLite memory database from blitz.toml `[storage] sqlite_path`. Pure
# SQLite: no LLM call, no optional dependency. Exit 2 with a clear message if
# the gmemory API is not importable (e.g. mid-refactor).
#
# Usage: scripts/nightly-gmemory-aggregate.sh [since_hours]   (default 24)
set -euo pipefail
cd "$(dirname "$0")/.."
SINCE_HOURS="${1:-24}"
exec "${PYTHON:-python3}" - "$SINCE_HOURS" <<'PY'
import sqlite3, sys
from pathlib import Path
try:
    from config import load_config
    from gmemory.query_graph import initialize_schema
    from gmemory.insight_graph import InsightGraph
except Exception as e:  # noqa: BLE001
    print(f"nightly-gmemory-aggregate: gmemory API unavailable ({e!r}); nothing run", file=sys.stderr)
    sys.exit(2)
since_hours = int(sys.argv[1])
cfg = load_config()
db_path = Path(cfg.storage.sqlite_path)
if not db_path.exists():
    print(f"nightly-gmemory-aggregate: no memory db at {db_path}; nothing to aggregate")
    sys.exit(0)
db = sqlite3.connect(db_path)
initialize_schema(db)
print(InsightGraph(db).aggregate_overnight(since_hours=since_hours))
PY
