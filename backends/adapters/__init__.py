"""Built-in backend adapters."""

from backends.adapters.cli import ClaudeCLIBackend, CodexLocalBackend, GeminiCLIBackend
from backends.adapters.generic_cli import GenericCLIBackend
from backends.adapters.ollama import OllamaHTTPBackend
from backends.adapters.openai_http import OpenAIHTTPBackend

__all__ = [
    "ClaudeCLIBackend",
    "CodexLocalBackend",
    "GeminiCLIBackend",
    "GenericCLIBackend",
    "OllamaHTTPBackend",
    "OpenAIHTTPBackend",
]
