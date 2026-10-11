"""Configuration loading for Blitz-Swarm.

Reads blitz.toml. Nothing model-, provider- or harness-specific is
hard-coded here: `[models]` maps roles to abstract tiers and
`[backend.providers.<id>]` maps tiers to provider-native model strings.
Resolution lives in `backends.models.resolve_model`.

Loading is strict: a malformed file, an unknown key or a wrongly typed
value raises `ConfigError` naming the path and the key. There is no silent
fallback to defaults for a file that exists but cannot be read.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "blitz.toml"


class ConfigError(ValueError):
    """Raised for a malformed blitz.toml; the message names path and key."""


@dataclass
class SwarmConfig:
    max_rounds: int = 5
    timeout_seconds: int = 180
    max_agents: int = 12
    domain: str = "general"           # which prompts/<domain>/ preset to load
    persona_critics: bool = False     # MAR-style persona-typed critics (Phase 1 opt-in)
    quality_profile: str = "max"      # max / balanced / cheap


@dataclass
class ModelsConfig:
    """`[models]`: role -> tier. Tiers are abstract names ("fast",
    "standard", "deep", ...); each provider maps them to real models."""

    default: str = "standard"
    roles: dict[str, str] = field(default_factory=dict)

    def tier_for(self, role: str) -> str | None:
        return self.roles.get(role) or self.default or None


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
    """One `[backend.providers.<id>]` table.

    `tiers` maps tier -> model string, `effort` maps tier -> provider
    effort value (forwarded by adapters that support it), `pricing` maps
    model -> {input_per_mtok, output_per_mtok}, `command` is the argv
    template for the generic_cli adapter. Unknown keys land in `metadata`
    and are passed through to the adapter factory, so a provider table is
    open-ended.
    """

    adapter: str = ""
    model: str = ""                   # legacy single-model key (resolution step 5)
    reasoning_effort: str = ""        # provider-wide effort; `effort` overrides per tier
    sandbox: str = "read-only"
    approval_policy: str = "never"
    ephemeral: bool = True
    base_url: str = ""
    api_key_env: str = ""             # openai_http only; env var NAME, optional
    command: list = field(default_factory=list)     # generic_cli argv template
    tiers: dict = field(default_factory=dict)       # tier -> model
    effort: dict = field(default_factory=dict)      # tier -> effort value
    pricing: dict = field(default_factory=dict)     # model -> {input_per_mtok, output_per_mtok}
    metadata: dict = field(default_factory=dict)


@dataclass
class BackendConfig:
    default: str = ""
    fallback: list[str] = field(default_factory=list)
    providers: dict[str, BackendProviderConfig] = field(default_factory=dict)

    def get_provider(self, name: str) -> BackendProviderConfig:
        provider = self.providers.get(name)
        if provider is None:
            raise KeyError(name)
        return provider


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
    models: ModelsConfig = field(default_factory=ModelsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    judge_ensemble: JudgeEnsembleConfig = field(default_factory=JudgeEnsembleConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    eviction: EvictionConfig = field(default_factory=EvictionConfig)


# ---------------------------------------------------------------------------
# Strict loader
# ---------------------------------------------------------------------------

_SIMPLE_SECTIONS = (
    "swarm", "memory", "guard", "judge_ensemble", "selector",
    "redis", "storage", "eviction",
)
_KNOWN_SECTIONS = _SIMPLE_SECTIONS + ("models", "backend")


def _type_ok(value, expected: type) -> bool:
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is str:
        return isinstance(value, str)
    if expected is list:
        return isinstance(value, list)
    if expected is dict:
        return isinstance(value, dict)
    return True


def _check_type(value, expected: type, *, path: Path, key: str) -> None:
    if not _type_ok(value, expected):
        raise ConfigError(
            f"{path}: {key} expects {expected.__name__}, got {type(value).__name__} ({value!r})"
        )


def _apply_section(target, raw: dict, *, path: Path, section: str) -> None:
    """Copy `raw` onto dataclass `target`; unknown key or bad type -> ConfigError."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: [{section}] must be a table")
    known = {f.name: f for f in fields(target)}
    for k, v in raw.items():
        if k not in known:
            raise ConfigError(f"{path}: unknown key [{section}].{k}")
        _check_type(v, type(getattr(target, k)), path=path, key=f"[{section}].{k}")
        if isinstance(v, int) and not isinstance(v, bool) and isinstance(getattr(target, k), float):
            v = float(v)
        setattr(target, k, v)


def _provider_from_raw(name: str, raw, *, path: Path) -> BackendProviderConfig:
    key = f"[backend.providers.{name}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: {key} must be a table")
    provider = BackendProviderConfig()
    known = {f.name: f for f in fields(provider) if f.name != "metadata"}
    for k, v in raw.items():
        if k in known:
            _check_type(v, type(getattr(provider, k)), path=path, key=f"{key}.{k}")
            setattr(provider, k, v)
        else:
            provider.metadata[k] = v
    for table in ("tiers", "effort"):
        for tier, value in getattr(provider, table).items():
            if not isinstance(value, str) or not value:
                raise ConfigError(
                    f"{path}: {key}.{table}.{tier} must be a non-empty string, got {value!r}"
                )
    for model, price in provider.pricing.items():
        if not isinstance(price, dict):
            raise ConfigError(f"{path}: {key}.pricing.{model} must be a table")
        for pk in ("input_per_mtok", "output_per_mtok"):
            if pk not in price:
                raise ConfigError(f"{path}: {key}.pricing.{model} is missing {pk}")
            _check_type(price[pk], float, path=path, key=f"{key}.pricing.{model}.{pk}")
    for i, arg in enumerate(provider.command):
        if not isinstance(arg, str):
            raise ConfigError(f"{path}: {key}.command[{i}] must be a string, got {arg!r}")
    return provider


def _apply_models(target: ModelsConfig, raw, *, path: Path) -> None:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: [models] must be a table")
    for k, v in raw.items():
        if not isinstance(v, str) or not v:
            raise ConfigError(f"{path}: [models].{k} must be a non-empty tier name, got {v!r}")
        if k == "default":
            target.default = v
        else:
            target.roles[k] = v


def _apply_backend(target: BackendConfig, raw, *, path: Path) -> None:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: [backend] must be a table")
    for k, v in raw.items():
        if k == "default":
            _check_type(v, str, path=path, key="[backend].default")
            target.default = v
        elif k == "fallback":
            if isinstance(v, str):
                v = [v] if v else []
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ConfigError(f"{path}: [backend].fallback must be a list of provider ids")
            target.fallback = list(v)
        elif k == "providers":
            if not isinstance(v, dict):
                raise ConfigError(f"{path}: [backend.providers] must be a table")
            for name, provider_raw in v.items():
                target.providers[name] = _provider_from_raw(name, provider_raw, path=path)
        elif isinstance(v, dict):
            # Legacy form `[backend.<id>]` == `[backend.providers.<id>]`.
            target.providers[k] = _provider_from_raw(k, v, path=path)
        else:
            raise ConfigError(f"{path}: unknown key [backend].{k}")
    if target.default and target.default not in target.providers:
        raise ConfigError(
            f"{path}: [backend].default = {target.default!r} is not defined under [backend.providers]"
        )
    for fid in target.fallback:
        if fid not in target.providers:
            raise ConfigError(
                f"{path}: [backend].fallback names {fid!r}, which is not defined under [backend.providers]"
            )


def load_config(path: Path = CONFIG_PATH) -> BlitzConfig:
    """Load blitz.toml. Missing file -> code defaults; malformed file -> ConfigError."""
    config = BlitzConfig()
    path = Path(path)

    if not path.exists():
        return config

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read config: {exc}") from exc

    for section in raw:
        if section not in _KNOWN_SECTIONS:
            raise ConfigError(f"{path}: unknown section [{section}]")

    for section in _SIMPLE_SECTIONS:
        if section in raw:
            _apply_section(getattr(config, section), raw[section], path=path, section=section)

    if "models" in raw:
        _apply_models(config.models, raw["models"], path=path)

    if "backend" in raw:
        _apply_backend(config.backend, raw["backend"], path=path)

    return config
