"""Embedding model wrapper for Blitz-Swarm memory retrieval.

Uses all-MiniLM-L6-v2 (22M params, 384 dimensions) for coarse-grained
query similarity search. The embedding is just the first filter in a
multi-stage retrieval pipeline — precision matters less than speed.
"""

import math

_SINGLETON = None


class Embedder:
    """Wrapper around sentence-transformers for text embedding.

    Loads the model once and reuses it. Use get_embedder() for singleton access.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the embedding model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            # Try ONNX backend for 2-3x CPU speedup
            try:
                self._model = SentenceTransformer(self.model_name, backend="onnx")
            except Exception:
                self._model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings.\n"
                "Install with: pip install sentence-transformers"
            )

    def encode(self, text: str) -> list[float]:
        """Encode a single text string into a 384-dim embedding vector."""
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts in a single batch."""
        self._load_model()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors. Pure Python, no numpy."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def get_embedder() -> Embedder:
    """Get or create the singleton Embedder instance."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Embedder()
    return _SINGLETON
