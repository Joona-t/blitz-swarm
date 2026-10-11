"""Compatibility shim — the memory package is `gmemory` since v0.3.0.

The v0.1-v0.2 reader/writer pair (Redis single-writer daemon, LanceDB L2
distance read as a similarity, its own schema) is gone. `gmemory.store`
is the one memory surface; `gmemory.migrate` upgrades old `memory.db`
files (also `scripts/migrate_memory_db.py`). This package only re-exports
so an old import resolves to the real thing.
"""

from gmemory.store import MemoryStore, PersistReport, build_trace  # noqa: F401

__all__ = ["MemoryStore", "PersistReport", "build_trace"]
