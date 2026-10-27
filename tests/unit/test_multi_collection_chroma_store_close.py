"""MultiCollectionChromaStore.close() releases the chromadb client (fd leak).

Incident 2026-10-27 (measured 10-26): chromadb caches one System per persist
path and nothing stops it, so ~238 function-scoped non-unit test stores held
~10 fds each (> the 1024 soft limit). class: BC-81 sibling (private heavyweight
resource never released). Drives the DEPLOYED store with a stubbed embedder so
no model loads; the fd source is the chroma client itself.
"""
import gc
import os
import shutil

import pytest

import memory.storage.multi_collection_chroma_store as mod


class _StubEmbed:
    def __init__(self, *a, **k):
        pass

    def __call__(self, input):
        return [[0.1, 0.2, 0.3, 0.4] for _ in input]


@pytest.fixture(autouse=True)
def _stub_embedder(monkeypatch):
    monkeypatch.setattr(mod.embedding_functions, "SentenceTransformerEmbeddingFunction", _StubEmbed)


def _fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def _build(tmp_path, i):
    store = mod.MultiCollectionChromaStore(persist_directory=str(tmp_path / f"c{i}"))
    store._get_collection("facts")  # opens a real collection (the segment files)
    return store


def test_building_and_closing_20_stores_returns_fds_to_baseline(tmp_path):
    warm = _build(tmp_path, "warm")  # one-time process-level handles (runtime threads, sqlite) are not a leak
    warm.close()
    gc.collect()
    start = _fds()
    for i in range(20):
        store = _build(tmp_path, i)
        store.close()
        shutil.rmtree(tmp_path / f"c{i}")
        del store
    gc.collect()
    assert _fds() - start <= 6, f"fds {start} -> {_fds()} after 20 closed stores"


def test_unclosed_stores_leak_so_the_counter_is_sensitive(tmp_path):
    gc.collect()
    start = _fds()
    leaked = []
    for i in range(5):
        store = _build(tmp_path, i)
        leaked.append(store)
    grew = _fds() - start
    for s in leaked:
        s.close()
    assert grew >= 20, f"control: 5 open stores grew fds by only {grew}"


def test_close_is_idempotent_and_only_stops_its_own_path(tmp_path):
    a = _build(tmp_path, "a")
    b = _build(tmp_path, "b")
    a.close()
    a.close()  # second call is a no-op
    assert a.client is None
    # the sibling store (different path) still works
    assert b._get_collection("facts") is not None
    b.close()
