"""embedder.py: the optional sentence-transformers wrapper. The `memory`
extra is never installed in the test venv, so these tests pin the fallback
contract (load_embedder() -> None, EmbedderUnavailable on first encode)
and exercise the loaded path through a fake module injected in sys.modules."""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

import embedder
from embedder import (
    DEFAULT_MODEL,
    ENV_MODEL,
    Embedder,
    EmbedderUnavailable,
    cosine_similarity,
    get_embedder,
    is_available,
    load_embedder,
)


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    monkeypatch.setattr(embedder, "_SINGLETON", None)
    yield
    monkeypatch.setattr(embedder, "_SINGLETON", None)


@pytest.fixture
def no_sentence_transformers(monkeypatch):
    """Simulate the package being absent even if the host happens to have it."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *a, **kw):
        if name == "sentence_transformers":
            return None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)   # import -> ImportError


class _Vec(list):
    def tolist(self):
        return list(self)


class _FakeSentenceTransformer:
    constructed: list[tuple[str, dict]] = []
    fail_backend_kw = False

    def __init__(self, name, **kw):
        type(self).constructed.append((name, kw))
        if kw.get("backend") and type(self).fail_backend_kw:
            raise TypeError("backend kw unsupported")
        self.name = name

    def encode(self, text, normalize_embeddings=False):
        assert normalize_embeddings is True
        if isinstance(text, list):
            return [_Vec([float(len(t)), 1.0]) for t in text]
        return _Vec([float(len(text)), 1.0])


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    _FakeSentenceTransformer.constructed = []
    _FakeSentenceTransformer.fail_backend_kw = False
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **kw: object() if name == "sentence_transformers" else None)
    return _FakeSentenceTransformer


# ---------------------------------------------------------------------------
# fallback path (what the dev venv actually exercises)
# ---------------------------------------------------------------------------


def test_importing_embedder_does_not_import_sentence_transformers_or_torch():
    # The module docstring promises laziness: the only sentence_transformers
    # import lives inside Embedder._load_model, and torch is never named.
    src = open(embedder.__file__, encoding="utf-8").read()
    assert "import torch" not in src
    assert src.index("def _load_model") < src.index("from sentence_transformers import")


def test_is_available_false_and_load_embedder_none_without_package(no_sentence_transformers):
    assert is_available() is False
    assert load_embedder() is None
    assert embedder._SINGLETON is None


def test_get_embedder_is_lazy_and_encode_raises_embedder_unavailable(no_sentence_transformers):
    e = get_embedder()
    assert isinstance(e, Embedder) and e._model is None
    assert get_embedder() is e
    with pytest.raises(EmbedderUnavailable, match=r"\.\[memory\]"):
        e.encode("hello")
    with pytest.raises(EmbedderUnavailable):
        e.encode_batch(["a", "b"])
    assert issubclass(EmbedderUnavailable, RuntimeError)


def test_is_available_swallows_find_spec_value_error(monkeypatch):
    def boom(name, *a, **kw):
        raise ValueError("__spec__ is None")
    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert is_available() is False


def test_model_name_resolution_order(monkeypatch):
    monkeypatch.delenv(ENV_MODEL, raising=False)
    assert Embedder().model_name == DEFAULT_MODEL
    monkeypatch.setenv(ENV_MODEL, "env-model")
    assert Embedder().model_name == "env-model"
    assert Embedder("explicit-model").model_name == "explicit-model"


# ---------------------------------------------------------------------------
# loaded path via a fake package
# ---------------------------------------------------------------------------


def test_load_embedder_returns_singleton_and_encodes_via_onnx_backend(fake_sentence_transformers, monkeypatch):
    monkeypatch.delenv(ENV_MODEL, raising=False)
    assert is_available() is True
    e = load_embedder()
    assert isinstance(e, Embedder) and load_embedder() is e
    assert fake_sentence_transformers.constructed == []        # still lazy
    assert e.encode("abc") == [3.0, 1.0]
    assert fake_sentence_transformers.constructed == [(DEFAULT_MODEL, {"backend": "onnx"})]
    assert e.encode_batch(["a", "abcd"]) == [[1.0, 1.0], [4.0, 1.0]]
    assert len(fake_sentence_transformers.constructed) == 1       # model loaded once


def test_load_model_falls_back_to_default_backend_when_onnx_rejected(fake_sentence_transformers):
    fake_sentence_transformers.fail_backend_kw = True
    e = Embedder("m")
    assert e.encode("xy") == [2.0, 1.0]
    assert fake_sentence_transformers.constructed == [("m", {"backend": "onnx"}), ("m", {})]


# ---------------------------------------------------------------------------
# cosine_similarity (pure python)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a, b, expected", [
    ([1.0, 0.0], [1.0, 0.0], 1.0),
    ([1.0, 0.0], [0.0, 1.0], 0.0),
    ([1.0, 2.0], [-1.0, -2.0], -1.0),
    ([3.0, 4.0], [6.0, 8.0], 1.0),
])
def test_cosine_similarity_values(a, b, expected):
    assert cosine_similarity(a, b) == pytest.approx(expected)


def test_cosine_similarity_degenerate_inputs_are_zero_not_errors():
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0          # length mismatch
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0     # zero norm
    assert cosine_similarity([], []) == 0.0
