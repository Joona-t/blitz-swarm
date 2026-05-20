"""Harness-agnostic backend package."""

from backends.core import (
    DEFAULT_APPROVAL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING,
    DEFAULT_SANDBOX,
    AgentBackend,
    AgentCall,
    AgentResult,
    BackendCapabilities,
    RuntimePolicy,
    combined_prompt,
    normalize_result,
    parse_json_loose,
    validate_json_schema,
)
from backends.adapters.cli import ClaudeCLIBackend, CodexLocalBackend, GeminiCLIBackend
from backends.adapters.ollama import OllamaHTTPBackend
from backends.registry import (
    BackendRegistry,
    make_backend,
    register_backend,
    registered_backends,
    resolve_adapter_id,
)

__all__ = [
    "DEFAULT_APPROVAL",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_REASONING",
    "DEFAULT_SANDBOX",
    "AgentBackend",
    "AgentCall",
    "AgentResult",
    "BackendCapabilities",
    "RuntimePolicy",
    "combined_prompt",
    "normalize_result",
    "parse_json_loose",
    "validate_json_schema",
    "ClaudeCLIBackend",
    "CodexLocalBackend",
    "GeminiCLIBackend",
    "OllamaHTTPBackend",
    "BackendRegistry",
    "make_backend",
    "register_backend",
    "registered_backends",
    "resolve_adapter_id",
]
