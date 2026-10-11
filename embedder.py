"""Optional embedding model wrapper for memory retrieval.

sentence-transformers (extra `memory`) provides the 384-dim MiniLM vectors
used for query kNN and insight clustering. It is optional: when it is not
installed, `load_embedder()` returns None and gmemory runs on the FTS5/BM25
path with rows flagged `vector_source='none'`. Importing this module never
imports torch; the model loads lazily on first `encode`.
"""

from __future__ import annotations

import importlib.util
import math
import os

DEFAULT_MODEL = "all-MiniLM-L6-v2"
ENV_MODEL = "BLITZ_EMBED_MODEL"

_SINGLETON: "Embedder | None" = None


class EmbedderUnavailable(RuntimeError):
    """sentence-transformers is not installed (pip install -e '.[memory]')."""


def is_available() -> bool:
    """True when sentence-transformers can be imported (no import performed)."""
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return False


class Embedder:
    """Lazy wrapper around sentence-transformers. Use `load_embedder()`."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.environ.get(ENV_MODEL) or DEFAULT_MODEL
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise EmbedderUnavailable(
                "sentence-transformers is required for embeddings; "
                "install the `memory` extra: pip install -e '.[memory]'"
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name, backend="onnx")   # 2-3x CPU speedup
        except Exception:
            self._model = SentenceTransformer(self.model_name)

    def encode(self, text: str) -> list[float]:
        """Encode one string into a normalized embedding vector."""
        self._load_model()
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        return [e.tolist() for e in self._model.encode(texts, normalize_embeddings=True)]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Pure Python, no numpy."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_embedder() -> Embedder | None:
    """The singleton Embedder when sentence-transformers is installed, else None."""
    if not is_available():
        return None
    return get_embedder()


def get_embedder() -> Embedder:
    """Singleton Embedder (raises EmbedderUnavailable on first encode if missing)."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Embedder()
    return _SINGLETON
