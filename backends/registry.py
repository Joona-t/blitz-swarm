"""Backend adapter registry, data-driven by adapter id.

`[backend.providers.<id>] adapter = "<adapter_id>"` picks the factory; the
provider table's fields are passed through to it. Provider ids are
open-ended: nothing here knows about any specific provider, only about
adapter kinds. `make_provider_backend` builds the backend for a configured
provider and wraps it with `FallbackBackend` when `[backend] fallback` is
set, so an "unavailable"-class error is retried once on the next provider.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable

from backends.core import DEFAULT_APPROVAL, DEFAULT_SANDBOX, AgentBackend, AgentCall, AgentResult, is_unavailable_error
from backends.models import ModelResolutionError, resolve_model

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
    backend_id: str,
    *,
    adapter: str | None = None,
    **kwargs,
) -> AgentBackend:
    """Instantiate the adapter `adapter` (or the one aliased by `backend_id`)."""
    _ensure_builtins()
    adapter_id = resolve_adapter_id(adapter or backend_id)
    factory = _FACTORIES.get(adapter_id)
    if factory is None:
        raise ValueError(
            f"unknown backend adapter {adapter or backend_id!r} for provider {backend_id!r}; "
            f"registered: {', '.join(sorted(_FACTORIES))}"
        )
    return factory(backend_id=backend_id, **kwargs)


def backend_from_provider(backend_id: str, provider, *, sandbox: str | None = None) -> AgentBackend:
    """Build one adapter instance from a `BackendProviderConfig`."""
    if not provider.adapter:
        raise ValueError(
            f"[backend.providers.{backend_id}] has no 'adapter' key; "
            f"registered adapters: {', '.join(registered_backends())}"
        )
    return make_backend(
        backend_id,
        adapter=provider.adapter,
        model=provider.model or None,
        reasoning_effort=provider.reasoning_effort or None,
        sandbox=sandbox or provider.sandbox or DEFAULT_SANDBOX,
        approval_policy=provider.approval_policy or DEFAULT_APPROVAL,
        ephemeral=provider.ephemeral,
        base_url=provider.base_url,
        api_key_env=provider.api_key_env,
        command=list(provider.command or []),
        **provider.metadata,
    )


def make_provider_backend(
    backend_id: str | None,
    cfg,
    *,
    sandbox: str | None = None,
    with_fallback: bool = True,
) -> AgentBackend:
    """Backend for provider `backend_id` (default: `[backend] default`),
    wrapped with the configured `[backend] fallback` chain."""
    selected = backend_id or cfg.backend.default
    if not selected:
        raise ValueError("no backend selected: set [backend] default in blitz.toml or pass --backend")
    try:
        provider = cfg.backend.get_provider(selected)
    except KeyError:
        raise ValueError(
            f"Unknown backend '{selected}' (not under [backend.providers]; "
            f"defined: {', '.join(sorted(cfg.backend.providers)) or 'none'})"
        ) from None
    primary = backend_from_provider(selected, provider, sandbox=sandbox)
    if not with_fallback:
        return primary
    fallbacks: list[tuple[str, AgentBackend]] = []
    for fid in cfg.backend.fallback:
        if fid == selected:
            continue
        try:
            fallbacks.append((fid, backend_from_provider(fid, cfg.backend.get_provider(fid), sandbox=sandbox)))
        except (KeyError, ValueError) as exc:
            print(f"[backend] fallback {fid!r} skipped: {exc}", file=sys.stderr)
    if not fallbacks:
        return primary
    return FallbackBackend(primary, fallbacks, cfg)


class FallbackBackend:
    """Primary backend + `[backend] fallback` chain, honoured once per call.

    When the primary returns an errored result of the "unavailable" class
    (`backends.core.is_unavailable_error`), the first available fallback
    provider is called exactly once with the model re-resolved for THAT
    provider through its tier table (global `BLITZ_MODEL` overrides are
    ignored for the hop: they name a model on the primary). The fallback
    result carries `fallback_used=True` and `metadata["fallback_from"]`.
    Any other error (timeout, schema failure) is returned as-is.
    """

    def __init__(self, primary: AgentBackend, fallbacks: list[tuple[str, AgentBackend]], cfg, *, log=None):
        self.primary = primary
        self.fallbacks = list(fallbacks)
        self.cfg = cfg
        self.backend_id = primary.backend_id
        self.adapter_kind = getattr(primary, "adapter_kind", "")
        self.capabilities = getattr(primary, "capabilities", None)
        self._log = log or (lambda msg: print(msg, file=sys.stderr))

    def is_available(self) -> bool:
        return self.primary.is_available() or any(b.is_available() for _, b in self.fallbacks)

    def call(self, call: AgentCall) -> AgentResult:
        result = self.primary.call(call)
        if not is_unavailable_error(result):
            return result
        for fid, backend in self.fallbacks:
            if not backend.is_available():
                continue
            try:
                resolved = resolve_model(call.role, fid, self.cfg, env={})
            except ModelResolutionError as exc:
                self._log(f"[backend] fallback {fid!r} skipped: {exc}")
                continue
            self._log(
                f"[backend] {self.backend_id!r} unavailable ({result.error}); "
                f"falling back once to {fid!r} model={resolved.model} role={call.role}"
            )
            hop = dataclasses.replace(call, model=resolved.model, reasoning_effort=resolved.effort)
            fb_result = backend.call(hop)
            fb_result.fallback_used = True
            fb_result.metadata = {
                **(fb_result.metadata or {}),
                "fallback_from": self.backend_id,
                "fallback_reason": result.error,
            }
            return fb_result
        return result


class BackendRegistry:
    """Select a backend by provider id and optionally fail over on
    availability (pre-call). See `FallbackBackend` for per-call fallback."""

    def __init__(
        self,
        backends: list[AgentBackend] | None = None,
        *,
        default_backend: str = "",
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
                return fallback
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
    from backends.adapters.generic_cli import GenericCLIBackend
    from backends.adapters.ollama import OllamaHTTPBackend
    from backends.adapters.openai_http import OpenAIHTTPBackend

    def _extras(kw: dict, *known: str) -> dict:
        skip = {"backend_id", *known}
        return {k: v for k, v in kw.items() if k not in skip}

    register_backend(
        "codex_cli",
        lambda **kw: CodexLocalBackend(
            backend_id=kw.get("backend_id", "codex"),
            model=kw.get("model") or None,
            reasoning_effort=kw.get("reasoning_effort") or None,
            sandbox=kw.get("sandbox") or DEFAULT_SANDBOX,
            approval_policy=kw.get("approval_policy") or DEFAULT_APPROVAL,
            ephemeral=bool(kw.get("ephemeral", True)),
        ),
        aliases=("codex",),
        replace=True,
    )
    register_backend(
        "claude_cli",
        lambda **kw: ClaudeCLIBackend(
            backend_id=kw.get("backend_id", "claude"),
            model=kw.get("model") or None,
            reasoning_effort=kw.get("reasoning_effort") or None,
        ),
        aliases=("claude",),
        replace=True,
    )
    register_backend(
        "gemini_cli",
        lambda **kw: GeminiCLIBackend(
            backend_id=kw.get("backend_id", "gemini"),
            model=kw.get("model") or None,
        ),
        aliases=("gemini",),
        replace=True,
    )
    register_backend(
        "ollama_http",
        lambda **kw: OllamaHTTPBackend(
            backend_id=kw.get("backend_id", "ollama"),
            model=kw.get("model") or None,
            base_url=kw.get("base_url") or OllamaHTTPBackend.base_url,
        ),
        aliases=("ollama",),
        replace=True,
    )
    register_backend(
        "generic_cli",
        lambda **kw: GenericCLIBackend(
            backend_id=kw.get("backend_id", "generic"),
            command=list(kw.get("command") or []),
            model=kw.get("model") or None,
            reasoning_effort=kw.get("reasoning_effort") or None,
            metadata=_extras(
                kw, "command", "model", "reasoning_effort", "sandbox",
                "approval_policy", "ephemeral", "base_url", "api_key_env",
            ),
        ),
        replace=True,
    )
    register_backend(
        "openai_http",
        lambda **kw: OpenAIHTTPBackend(
            backend_id=kw.get("backend_id", "local"),
            base_url=kw.get("base_url") or "",
            model=kw.get("model") or None,
            api_key_env=kw.get("api_key_env") or "",
            reasoning_effort=kw.get("reasoning_effort") or None,
            metadata=_extras(
                kw, "command", "model", "reasoning_effort", "sandbox",
                "approval_policy", "ephemeral", "base_url", "api_key_env",
            ),
        ),
        replace=True,
    )
    _BUILTINS_REGISTERED = True
