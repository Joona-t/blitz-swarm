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

from backends.core import (
    DEFAULT_APPROVAL,
    DEFAULT_SANDBOX,
    SCHEMA_RETRY_NOTE,
    SCHEMA_VALIDATE,
    AgentBackend,
    AgentCall,
    AgentResult,
    BackendCapabilityError,
    check_requirements,
    implied_requirements,
    is_unavailable_error,
    prepare_schema_call,
    schema_failed,
)
from backends.models import ModelResolutionError, provider_for_role, resolve_model

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
    """Build one adapter instance from a `BackendProviderConfig`, wrapped in
    `PolicyBackend` so every call is capability-checked (WP1) and schema-
    policed (WP2). `[backend.providers.<id>] allow_degrade = ["no_tools"]`
    names requirements that may be waived with a logged warning."""
    if not provider.adapter:
        raise ValueError(
            f"[backend.providers.{backend_id}] has no 'adapter' key; "
            f"registered adapters: {', '.join(registered_backends())}"
        )
    extras = dict(provider.metadata)
    allow_degrade = extras.pop("allow_degrade", None) or []
    if isinstance(allow_degrade, str):
        allow_degrade = [allow_degrade]
    raw = make_backend(
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
        **extras,
    )
    return PolicyBackend(raw, allow_degrade=tuple(allow_degrade))


class PolicyBackend:
    """Capability check + schema policy around one adapter.

    Order per call: (1) unmet requirements -> `BackendCapabilityError`
    unless listed in `allow_degrade`, in which case one stderr line and
    `metadata["degraded"]`; (2) `prepare_schema_call` picks native/validate;
    (3) adapter call; (4) in validate mode a schema failure is retried
    exactly once with the errors quoted, then returned as an error.
    """

    def __init__(self, inner: AgentBackend, *, allow_degrade: tuple[str, ...] = (), log=None):
        self.inner = inner
        self.backend_id = inner.backend_id
        self.adapter_kind = getattr(inner, "adapter_kind", "")
        self.allow_degrade = tuple(allow_degrade)
        self._log = log or (lambda msg: print(msg, file=sys.stderr))

    @property
    def capabilities(self):
        return getattr(self.inner, "capabilities", None)

    def is_available(self) -> bool:
        return self.inner.is_available()

    def call(self, call: AgentCall) -> AgentResult:
        caps = self.capabilities
        probe = dataclasses.replace(call, requires=implied_requirements(call))
        unmet = check_requirements(probe, caps)
        degraded = [r for r in unmet if r in self.allow_degrade]
        hard = [r for r in unmet if r not in self.allow_degrade]
        if hard:
            raise BackendCapabilityError(
                f"backend {self.backend_id!r} ({self.adapter_kind}) cannot meet "
                f"{', '.join(hard)} required by role {call.role!r}; "
                f"capabilities: schema={getattr(caps, 'schema_policy', '?')} "
                f"tools={getattr(caps, 'tool_policy', '?')} cost={getattr(caps, 'cost_policy', '?')}"
            )
        if degraded:
            self._log(
                f"[backend] {self.backend_id!r} DEGRADED: waiving {', '.join(degraded)} "
                f"for role {call.role!r} (allow_degrade in blitz.toml)"
            )
        sent, mode = prepare_schema_call(call, caps)
        result = self.inner.call(sent)
        if mode == SCHEMA_VALIDATE and schema_failed(result):
            retry = dataclasses.replace(
                sent, prompt=sent.prompt + SCHEMA_RETRY_NOTE.format(
                    errors="; ".join(result.validation_errors[:3])),
            )
            second = self.inner.call(retry)
            second.metadata = {**(second.metadata or {}), "schema_retry": True,
                               "first_errors": result.validation_errors[:3]}
            result = second
            if schema_failed(result):
                result.errored = True
                result.error = "schema: " + "; ".join(result.validation_errors[:3])
        result.schema_mode = mode or result.schema_mode
        if degraded:
            result.metadata = {**(result.metadata or {}), "degraded": degraded}
        return result


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


def make_role_router(backend_id: str | None, cfg, *, sandbox: str | None = None) -> AgentBackend:
    """The runtime backend: the default provider (with its fallback chain),
    plus one lazily built backend per `[models] <role>_provider` pin. With no
    pins this is exactly `make_provider_backend`."""
    default = make_provider_backend(backend_id, cfg, sandbox=sandbox)
    pins = dict(getattr(cfg.models, "providers", {}) or {})
    if not pins:
        return default
    for role in pins:
        provider_for_role(role, cfg, backend_id)   # raises on an undefined pin
    return RoleRouter(default, cfg, sandbox=sandbox)


class RoleRouter:
    """Routes each call to the provider pinned for its role (WP7).

    The orchestrator resolves a call's model for the DEFAULT provider; when
    a role is pinned elsewhere the model is re-resolved for that provider
    through its own tier table (env overrides still apply), the same way
    `FallbackBackend` re-resolves for a hop. `metadata["routed_provider"]`
    records where the call went.
    """

    def __init__(self, default: AgentBackend, cfg, *, sandbox: str | None = None, log=None):
        self.default = default
        self.cfg = cfg
        self.sandbox = sandbox
        self.backend_id = default.backend_id
        self.adapter_kind = getattr(default, "adapter_kind", "")
        self.capabilities = getattr(default, "capabilities", None)
        self.pins: dict[str, str] = dict(getattr(cfg.models, "providers", {}) or {})
        self._built: dict[str, AgentBackend] = {}
        self._log = log or (lambda msg: print(msg, file=sys.stderr))

    def is_available(self) -> bool:
        return self.default.is_available()

    def backend_for(self, role: str) -> tuple[str, AgentBackend]:
        pid = self.pins.get(role)
        if not pid or pid == self.backend_id:
            return self.backend_id, self.default
        if pid not in self._built:
            self._built[pid] = make_provider_backend(pid, self.cfg, sandbox=self.sandbox)
        return pid, self._built[pid]

    def call(self, call: AgentCall) -> AgentResult:
        role = call.model_role or call.role
        pid, backend = self.backend_for(role)
        if backend is self.default:
            return backend.call(call)
        resolved = resolve_model(role, pid, self.cfg)
        routed = dataclasses.replace(call, model=resolved.model, reasoning_effort=resolved.effort)
        result = backend.call(routed)
        result.metadata = {**(result.metadata or {}), "routed_provider": pid, "routed_from": self.backend_id}
        return result


class FallbackBackend:
    """Primary backend + `[backend] fallback` chain, honoured once per call.

    When the primary returns an errored result of the "unavailable" class
    (`backends.core.is_unavailable_error`), the first available fallback
    provider is called exactly once with the model re-resolved for THAT
    provider through its tier table, using `call.model_role` (the resolver
    role) when set, else `call.role` (global `BLITZ_MODEL` overrides are
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
            model_role = call.model_role or call.role
            try:
                resolved = resolve_model(model_role, fid, self.cfg, env={})
            except ModelResolutionError as exc:
                self._log(f"[backend] fallback {fid!r} skipped: {exc}")
                continue
            self._log(
                f"[backend] {self.backend_id!r} unavailable ({result.error}); "
                f"falling back once to {fid!r} model={resolved.model} role={model_role}"
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
    from backends.adapters.acp import ACPBackend
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
        "acp",
        lambda **kw: ACPBackend(
            backend_id=kw.get("backend_id", "acp"),
            command=list(kw.get("command") or []),
            model=kw.get("model") or None,
            reasoning_effort=kw.get("reasoning_effort") or None,
            metadata=_extras(
                kw, "command", "model", "reasoning_effort", "sandbox",
                "approval_policy", "ephemeral", "base_url", "api_key_env",
            ),
        ),
        aliases=("acp_stdio",),
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
