"""Removed in v0.3.0 — use `gmemory.store.MemoryStore` (see memory/__init__.py)."""

raise ImportError(
    "memory.writer was replaced by gmemory in v0.3.0: use gmemory.store.MemoryStore "
    "and run scripts/migrate_memory_db.py on an existing memory.db"
)
