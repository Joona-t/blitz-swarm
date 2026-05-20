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
    domain: str = "general"           # which prompts/<domain>/ preset to load
    persona_critics: bool = False     # MAR-style persona-typed critics (Phase 1 opt-in)
    quality_profile: str = "max"      # max / balanced / cheap


@dataclass
class MemoryConfig:
    max_context_tokens: int = 2000
    top_k_retrieval: int = 2
    hop_expansion: int = 1
    insight_dedup_threshold: float = 0.85
    query_link_threshold: float = 0.7
    llm_ops_threshold: int = 10
    gmemory_tier: int = 3


@dataclass
class BackendProviderConfig:
    model: str = ""
    reasoning_effort: str = "high"
    sandbox: str = "read-only"
    approval_policy: str = "never"
    ephemeral: bool = True


@dataclass
class BackendConfig:
    default: str = "codex"
    fallback: str | None = None
    codex: BackendProviderConfig = field(
        default_factory=lambda: BackendProviderConfig(model="gpt-5.5")
    )
    claude: BackendProviderConfig = field(
        default_factory=lambda: BackendProviderConfig(model="sonnet")
    )
    gemini: BackendProviderConfig = field(default_factory=BackendProviderConfig)


@dataclass
class GuardConfig:
    enabled: bool = True
    mode: str = "balanced"


@dataclass
class JudgeEnsembleConfig:
    enabled: bool = False
    n_judges: int = 3
    ks_threshold: float = 0.05
    ks_consecutive: int = 2
    min_rounds: int = 2


@dataclass
class SelectorConfig:
    enabled: bool = False
    granularity: str = "section"
    n_judges: int = 3


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
    backend: BackendConfig = field(default_factory=BackendConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    judge_ensemble: JudgeEnsembleConfig = field(default_factory=JudgeEnsembleConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
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

    if "backend" in raw:
        for k, v in raw["backend"].items():
            if isinstance(v, dict):
                provider = getattr(config.backend, k, None)
                if provider is not None:
                    for pk, pv in v.items():
                        if hasattr(provider, pk):
                            setattr(provider, pk, pv)
            elif hasattr(config.backend, k):
                setattr(config.backend, k, v)
        if config.backend.fallback == "":
            config.backend.fallback = None

    if "guard" in raw:
        for k, v in raw["guard"].items():
            if hasattr(config.guard, k):
                setattr(config.guard, k, v)

    if "judge_ensemble" in raw:
        for k, v in raw["judge_ensemble"].items():
            if hasattr(config.judge_ensemble, k):
                setattr(config.judge_ensemble, k, v)

    if "selector" in raw:
        for k, v in raw["selector"].items():
            if hasattr(config.selector, k):
                setattr(config.selector, k, v)

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
