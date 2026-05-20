"""Built-in backend adapters."""

from backends.adapters.cli import ClaudeCLIBackend, CodexLocalBackend, GeminiCLIBackend
from backends.adapters.ollama import OllamaHTTPBackend

__all__ = [
    "ClaudeCLIBackend",
    "CodexLocalBackend",
    "GeminiCLIBackend",
    "OllamaHTTPBackend",
]
