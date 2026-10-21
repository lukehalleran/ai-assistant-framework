"""WebSearchCache writes never run an embedding model (incident 2026-10-19/20, BC-41).

Live: a LoopStall of 1.8 s whose ``other threads:`` field named
``onnxruntime ... run <- web_search_manager.py put`` -- the cache collection had no
embedding function, so ``upsert(documents=...)`` made Chroma cold-load and run its
default ONNX embedder on a worker thread. Lookups are by deterministic id only, so
the cache now supplies one constant unit vector.
"""
import chromadb
import pytest

from knowledge.web_search_manager import (
    WebPage,
    WebSearchCache,
    WebSearchDepth,
    WebSearchResult,
)


class _ExplodingEmbedder:
    calls = 0

    def __call__(self, input):  # noqa: A002 - chroma's EF signature
        type(self).calls += 1
        raise AssertionError("an embedding function ran for a cache write")

    @staticmethod
    def name():
        return "exploding-test-embedder"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return _ExplodingEmbedder()


class _Client:
    """Hands WebSearchCache a collection whose embedding function must never run."""

    def __init__(self):
        self._real = chromadb.EphemeralClient()

    def get_or_create_collection(self, name, metadata=None, **kw):
        try:
            self._real.delete_collection(name)
        except Exception:  # degrades: nothing to clear in a fresh ephemeral client
            pass
        return self._real.create_collection(
            name=name, metadata=metadata, embedding_function=_ExplodingEmbedder()
        )


class _Store:
    def __init__(self):
        self.client = _Client()


def _result(query="rowan weather"):
    return WebSearchResult(
        query=query,
        pages=[WebPage(url="https://example.test/a", title="A", content="body", score=0.9)],
        search_depth=WebSearchDepth.QUICK,
    )


def test_put_then_get_round_trips_without_running_an_embedder():
    _ExplodingEmbedder.calls = 0
    cache = WebSearchCache(chroma_store=_Store())
    cache.put(_result())
    got = cache.get("rowan weather", WebSearchDepth.QUICK)
    assert got is not None, "put failed silently (embedder ran?)"
    assert got.pages[0].url == "https://example.test/a"
    assert _ExplodingEmbedder.calls == 0


def test_constant_embedding_is_a_nonzero_unit_vector_of_the_on_disk_dimension():
    from knowledge.web_search_manager import CACHE_CONSTANT_EMBEDDING, CACHE_EMBEDDING_DIM

    assert CACHE_EMBEDDING_DIM == 384
    assert len(CACHE_CONSTANT_EMBEDDING) == CACHE_EMBEDDING_DIM
    assert sum(x * x for x in CACHE_CONSTANT_EMBEDDING) == pytest.approx(1.0)
