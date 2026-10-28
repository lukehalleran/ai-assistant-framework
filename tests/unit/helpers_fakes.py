"""Shared fakes for unit tests (lane C4 test-scaffolding compaction).

Each class here was previously copied near-verbatim into several test files.
Behaviour is identical to the copies it replaced; per-file differences are
constructor/factory parameters. Same flat-module pattern as helpers_orchestrator.py.
"""

from datetime import datetime

import numpy as np


# --- embedders ----------------------------------------------------------------

class ConceptBagEmbedder:
    """Deterministic bag-of-concepts embedder: one dimension per concept word-set
    plus a small constant tail, L2-normalised. Subclass via make_concept_embedder."""

    concepts: list = []
    calls = 0

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kw):
        type(self).calls += len(texts)
        out = []
        for t in texts:
            words = {w.strip(".,:;'\"?!").lower() for w in t.replace("_", " ").split()}
            v = np.array([float(len(words & c)) for c in self.concepts] + [0.01])
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def make_concept_embedder(concepts):
    """A fresh ConceptBagEmbedder subclass bound to ``concepts`` (own ``calls`` counter)."""
    return type("FakeEmbedder", (ConceptBagEmbedder,), {"concepts": concepts, "calls": 0})


class ToneAxisEmbedder:
    """4-axis tone-space embedder: FEELBAD count on axis 2, MATH count (+0.01) on axis 3."""

    def encode(self, text, convert_to_numpy=True):
        v = np.zeros(4)
        v[2] = text.count("FEELBAD")
        v[3] = text.count("MATH") + 0.01
        return v


# --- chroma stores ------------------------------------------------------------

class RangeQueryFakeChromaStore:
    """Data-backed stand-in for MultiCollectionChromaStore as read by MemoryExpander.

    data: {collection_name: [doc_dicts]}. Call counters and on-demand raises
    (``_get_by_id_raises`` doc_id -> exception, ``_range_raises``) are inert unless set.
    """

    def __init__(self, data: dict | None = None):
        self._data = data or {}
        self._get_by_id_raises: dict = {}
        self._range_raises = None
        self.get_by_id_calls = 0
        self.range_calls = 0

    def get_by_id(self, collection_name: str, doc_id: str):
        self.get_by_id_calls += 1
        if doc_id in self._get_by_id_raises:
            raise self._get_by_id_raises[doc_id]
        for doc in self._data.get(collection_name, []):
            if doc["id"] == doc_id:
                return dict(doc)
        return None

    def list_all(self, collection_name: str):
        return [dict(d) for d in self._data.get(collection_name, [])]

    def get_ids_by_timestamp_range(self, collection_name: str, start_iso: str, end_iso: str):
        self.range_calls += 1
        if self._range_raises is not None:
            raise self._range_raises
        try:
            start = datetime.fromisoformat(start_iso)
            end = datetime.fromisoformat(end_iso)
        except (ValueError, TypeError):
            return []
        ids = []
        for doc in self._data.get(collection_name, []):
            ts_str = (doc.get("metadata") or {}).get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue
            if start <= ts <= end:
                ids.append(doc["id"])
        return ids


def make_staged_replacement_fakes(marker, collection_name):
    """(FakeCollection, FakeChromaStore) for the staged-replacement store tests.

    Records adds/deletes and raises RuntimeError(f"{marker}_add"/"_delete") on demand.
    ``collection_name`` keys the store's single collection.
    """

    class FakeCollection:
        """Fake ChromaDB collection: count()/delete(), with injectable delete failures."""

        def __init__(self, items):
            self._items = items  # shared list reference with FakeChromaStore
            self._fail_delete_ids = set()
            self.delete_calls = []

        def count(self):
            return len(self._items)

        def fail_delete(self, doc_id):
            """The next delete() call naming doc_id raises instead of deleting."""
            self._fail_delete_ids.add(doc_id)

        def delete(self, ids=None):
            ids = list(ids or [])
            self.delete_calls.append(ids)
            if any(i in self._fail_delete_ids for i in ids):
                raise RuntimeError(f"{marker}_delete")
            matched = set(ids)
            self._items[:] = [item for item in self._items if item.get("id") not in matched]

    class FakeChromaStore:
        """Fake of the MultiCollectionChromaStore subset used by the store under test."""

        def __init__(self, items=None):
            self._items = items if items is not None else []
            self._id_counter = 0
            self.collection = FakeCollection(self._items)
            self.collections = {collection_name: self.collection}
            self._add_outcomes = []  # queue of bools; True = next add() raises

        def queue_add_failure(self):
            self._add_outcomes.append(True)

        def add_to_collection(self, name, text, metadata):
            fail = self._add_outcomes.pop(0) if self._add_outcomes else False
            if fail:
                raise RuntimeError(f"{marker}_add")
            self._id_counter += 1
            doc_id = f"doc_{self._id_counter}"
            self._items.append({"id": doc_id, "content": text, "metadata": dict(metadata)})
            return doc_id

        def list_all(self, name, where=None):
            return list(self._items)

        def create_collection(self, name):
            return self.collection

    return FakeCollection, FakeChromaStore
