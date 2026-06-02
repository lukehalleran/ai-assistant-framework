"""
Regression guard: an embedder mismatch must NEVER auto-delete a collection.

Deleting + recreating on mismatch (the old behavior) silently wiped every
document in the collection — including protected ones — violating the critical
no-auto-delete rule. The store must instead open the existing collection as-is
to preserve its data. See memory/storage/multi_collection_chroma_store.py.
"""

import pytest

from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore


class _FakeClient:
    """Stand-in Chroma client whose get_or_create raises the mismatch error."""
    def __init__(self):
        self.deleted = []
        self.created = []

    def get_or_create_collection(self, name, embedding_function=None):
        raise ValueError(
            "Embedding function name mismatch: collection expected 'default' "
            "but got 'sentence_transformer'"
        )

    def get_collection(self, name):
        return f"existing::{name}"  # sentinel for the preserved collection

    def delete_collection(self, name):
        self.deleted.append(name)

    def list_collections(self):
        return []


def _store_with(client):
    # Skip __init__ so no real Chroma client / SentenceTransformer is constructed.
    store = object.__new__(MultiCollectionChromaStore)
    store.collections = {"facts": None}
    store.embedding_fn = object()
    store.embedding_model_name = "BAAI/bge-small-en-v1.5"
    store.client = client
    return store


def test_embedder_mismatch_opens_as_is_and_never_deletes():
    client = _FakeClient()
    store = _store_with(client)

    coll = store._get_collection("facts")

    assert coll == "existing::facts"      # opened the existing collection as-is
    assert client.deleted == []           # CRITICAL: nothing was deleted


def test_embedder_mismatch_unopenable_raises_instead_of_deleting():
    """If the collection can't even be opened, refuse loudly — still no delete."""
    client = _FakeClient()
    client.get_collection = lambda name: (_ for _ in ()).throw(RuntimeError("cannot open"))
    store = _store_with(client)

    with pytest.raises(RuntimeError):
        store._get_collection("facts")

    assert client.deleted == []           # still never deletes


def test_non_mismatch_valueerror_propagates():
    """Unrelated ValueErrors must NOT be swallowed by the mismatch handler."""
    client = _FakeClient()
    client.get_or_create_collection = lambda name, embedding_function=None: (
        (_ for _ in ()).throw(ValueError("some other problem"))
    )
    store = _store_with(client)

    with pytest.raises(ValueError, match="some other problem"):
        store._get_collection("facts")
    assert client.deleted == []
