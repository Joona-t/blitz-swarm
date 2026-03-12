"""Configuration loading for Blitz-Swarm.

Reads from blitz.toml if present, otherwise uses defaults.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "blitz.toml"


@dataclass
class SwarmConfig:
    max_rounds: int = 5
    default_model: str = "sonnet"
    timeout_seconds: int = 180
    max_agents: int = 12


@dataclass
class MemoryConfig:
    max_context_tokens: int = 2000
    top_k_retrieval: int = 2
    hop_expansion: int = 1
    insight_dedup_threshold: float = 0.85
    query_link_threshold: float = 0.7
    llm_ops_threshold: int = 10


@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0


@dataclass
class StorageConfig:
    sqlite_path: str = "memory.db"
    lancedb_path: str = "./memory_vectors"
    output_dir: str = "./output"


@dataclass
class EvictionConfig:
    query_archive_days: int = 90
    interaction_purge_days: int = 30
    max_archive_per_cycle: int = 50
    cycle_frequency: int = 10


@dataclass
class BlitzConfig:
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    eviction: EvictionConfig = field(default_factory=EvictionConfig)


def load_config(path: Path = CONFIG_PATH) -> BlitzConfig:
    """Load configuration from blitz.toml, falling back to defaults."""
    config = BlitzConfig()

    if not path.exists():
        return config

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except Exception:
        return config

    # Apply overrides from TOML
    if "swarm" in raw:
        for k, v in raw["swarm"].items():
            if hasattr(config.swarm, k):
                setattr(config.swarm, k, v)

    if "memory" in raw:
        for k, v in raw["memory"].items():
            if hasattr(config.memory, k):
                setattr(config.memory, k, v)

    if "redis" in raw:
        for k, v in raw["redis"].items():
            if hasattr(config.redis, k):
                setattr(config.redis, k, v)

    if "storage" in raw:
        for k, v in raw["storage"].items():
            if hasattr(config.storage, k):
                setattr(config.storage, k, v)

    if "eviction" in raw:
        for k, v in raw["eviction"].items():
            if hasattr(config.eviction, k):
                setattr(config.eviction, k, v)

    return config
