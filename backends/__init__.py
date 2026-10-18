"""Harness-agnostic backend package."""

from backends.core import (
    DEFAULT_APPROVAL,
    DEFAULT_SANDBOX,
    UNAVAILABLE_MARKERS,
    AgentBackend,
    AgentCall,
    AgentResult,
    BackendCapabilities,
    RuntimePolicy,
    combined_prompt,
    is_unavailable_error,
    normalize_result,
    parse_json_loose,
    strip_strict_nulls,
    to_strict_output_schema,
    validate_json_schema,
)
from backends.models import (
    ModelResolutionError,
    ResolvedModel,
    format_models_table,
    list_models,
    pricing_for,
    resolve_model,
)
from backends.adapters.cli import ClaudeCLIBackend, CodexLocalBackend, GeminiCLIBackend
from backends.adapters.generic_cli import GenericCLIBackend
from backends.adapters.ollama import OllamaHTTPBackend
from backends.adapters.openai_http import OpenAIHTTPBackend
from backends.registry import (
    BackendRegistry,
    FallbackBackend,
    backend_from_provider,
    make_backend,
    make_provider_backend,
    register_backend,
    registered_backends,
    resolve_adapter_id,
)

__all__ = [
    "DEFAULT_APPROVAL",
    "DEFAULT_SANDBOX",
    "UNAVAILABLE_MARKERS",
    "AgentBackend",
    "AgentCall",
    "AgentResult",
    "BackendCapabilities",
    "RuntimePolicy",
    "combined_prompt",
    "is_unavailable_error",
    "normalize_result",
    "parse_json_loose",
    "strip_strict_nulls",
    "to_strict_output_schema",
    "validate_json_schema",
    "ModelResolutionError",
    "ResolvedModel",
    "format_models_table",
    "list_models",
    "pricing_for",
    "resolve_model",
    "ClaudeCLIBackend",
    "CodexLocalBackend",
    "GeminiCLIBackend",
    "GenericCLIBackend",
    "OllamaHTTPBackend",
    "OpenAIHTTPBackend",
    "BackendRegistry",
    "FallbackBackend",
    "backend_from_provider",
    "make_backend",
    "make_provider_backend",
    "register_backend",
    "registered_backends",
    "resolve_adapter_id",
]
