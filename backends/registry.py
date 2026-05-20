"""Backend adapter registry."""

from __future__ import annotations

from collections.abc import Callable

from backends.core import AgentBackend

BackendFactory = Callable[..., AgentBackend]

_FACTORIES: dict[str, BackendFactory] = {}
_ALIASES: dict[str, str] = {}
_BUILTINS_REGISTERED = False


def register_backend(
    adapter_id: str,
    factory: BackendFactory,
    *,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> None:
    if not replace and adapter_id in _FACTORIES:
        raise ValueError(f"backend adapter already registered: {adapter_id}")
    _FACTORIES[adapter_id] = factory
    for alias in aliases:
        if not replace and alias in _ALIASES and _ALIASES[alias] != adapter_id:
            raise ValueError(f"backend alias already registered: {alias}")
        _ALIASES[alias] = adapter_id


def registered_backends() -> list[str]:
    _ensure_builtins()
    return sorted(_FACTORIES)


def resolve_adapter_id(adapter_id: str) -> str:
    _ensure_builtins()
    return _ALIASES.get(adapter_id, adapter_id)


def make_backend(
    backend_id: str = "codex",
    *,
    adapter: str | None = None,
    **kwargs,
) -> AgentBackend:
    _ensure_builtins()
    adapter_id = resolve_adapter_id(adapter or backend_id)
    factory = _FACTORIES.get(adapter_id)
    if factory is None:
        raise ValueError(f"unknown backend adapter: {adapter or backend_id}")
    return factory(backend_id=backend_id, **kwargs)


class BackendRegistry:
    """Select a backend by provider id and optionally fail over."""

    def __init__(
        self,
        backends: list[AgentBackend] | None = None,
        *,
        default_backend: str = "codex",
        fallback_backend: str | list[str] | None = None,
    ):
        _ensure_builtins()
        self.backends = {b.backend_id: b for b in (backends or [])}
        self.default_backend = default_backend
        if fallback_backend is None:
            self.fallback_backends: list[str] = []
        elif isinstance(fallback_backend, str):
            self.fallback_backends = [fallback_backend]
        else:
            self.fallback_backends = list(fallback_backend)

    def get(self, backend_id: str | None = None) -> AgentBackend:
        chosen = backend_id or self.default_backend
        backend = self.backends.get(chosen) or make_backend(chosen)
        if backend.is_available():
            return backend
        for fallback_id in self.fallback_backends:
            fallback = self.backends.get(fallback_id) or make_backend(fallback_id)
            if fallback.is_available():
                fallback_result = fallback
                return fallback_result
        return backend


def _ensure_builtins() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from backends.adapters.cli import (
        ClaudeCLIBackend,
        CodexLocalBackend,
        GeminiCLIBackend,
    )
    from backends.adapters.ollama import OllamaHTTPBackend

    register_backend(
        "codex_cli",
        lambda **kw: CodexLocalBackend(
            backend_id=kw.get("backend_id", "codex"),
            model=kw.get("model") or "gpt-5.5",
            reasoning_effort=kw.get("reasoning_effort") or "high",
            sandbox=kw.get("sandbox") or "read-only",
            approval_policy=kw.get("approval_policy") or "never",
            ephemeral=bool(kw.get("ephemeral", True)),
        ),
        aliases=("codex",),
        replace=True,
    )
    register_backend(
        "claude_cli",
        lambda **kw: ClaudeCLIBackend(
            backend_id=kw.get("backend_id", "claude"),
            model=kw.get("model") or "sonnet",
        ),
        aliases=("claude",),
        replace=True,
    )
    register_backend(
        "gemini_cli",
        lambda **kw: GeminiCLIBackend(
            backend_id=kw.get("backend_id", "gemini"),
            model=kw.get("model"),
        ),
        aliases=("gemini",),
        replace=True,
    )
    register_backend(
        "ollama_http",
        lambda **kw: OllamaHTTPBackend(
            backend_id=kw.get("backend_id", "ollama"),
            model=kw.get("model") or "qwen2.5:7b",
            base_url=kw.get("base_url") or "http://localhost:11434",
        ),
        aliases=("ollama",),
        replace=True,
    )
    _BUILTINS_REGISTERED = True
