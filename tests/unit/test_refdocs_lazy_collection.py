"""
Regression tests for the 2026-08-02 reference_docs duplicate-accumulation bug.

ReferenceDocsManager read the raw `chroma_store.collections` dict, which holds
None placeholders until the store's lazy `_get_collection()` opens them. On a
cold startup every title/hash lookup silently returned empty, so the auto-seed
re-uploaded the alphabetically-first doc WITHOUT deleting its previous chunks
— one duplicate copy per startup (AGENTIC_SEARCH: 334 batches, 15,309 junk
chunks). All collection access must go through the manager's _collection()
helper → store._get_collection().
"""

from unittest.mock import MagicMock

from knowledge.reference_docs_manager import ReferenceDocsManager


def _store_with_unopened_collection(chunks):
    """A store whose .collections dict is still a None placeholder, but whose
    lazy _get_collection() opens a working collection — the exact cold-startup
    shape that used to make lookups silently return empty."""
    store = MagicMock()
    store.collections = {"reference_docs": None}  # raw dict access sees None
    coll = MagicMock()
    coll.get.return_value = {
        "ids": [c["id"] for c in chunks],
        "documents": [c["content"] for c in chunks],
        "metadatas": [c["metadata"] for c in chunks],
    }
    store._get_collection.return_value = coll
    return store, coll


CHUNKS = [
    {"id": "c1", "content": "chunk one",
     "metadata": {"title": "AGENTIC_SEARCH", "content_hash": "abc123",
                  "chunk_index": 0}},
    {"id": "c2", "content": "chunk two",
     "metadata": {"title": "AGENTIC_SEARCH", "content_hash": "abc123",
                  "chunk_index": 1}},
]


class TestLazyCollectionAccess:
    def test_get_document_chunks_opens_collection_lazily(self):
        store, coll = _store_with_unopened_collection(CHUNKS)
        mgr = ReferenceDocsManager(chroma_store=store)
        chunks = mgr._get_document_chunks("AGENTIC_SEARCH")
        assert len(chunks) == 2
        store._get_collection.assert_called_with("reference_docs")

    def test_stored_content_hash_visible_on_cold_store(self):
        # Pre-fix this returned None on a cold store → sync re-uploaded every
        # startup and the existing-chunks delete never fired.
        store, _ = _store_with_unopened_collection(CHUNKS)
        mgr = ReferenceDocsManager(chroma_store=store)
        assert mgr._get_stored_content_hash("AGENTIC_SEARCH") == "abc123"

    def test_sync_file_skips_unchanged_on_cold_store(self, tmp_path):
        doc = tmp_path / "AGENTIC_SEARCH.md"
        doc.write_text("# doc\ncontent\n")
        store, _ = _store_with_unopened_collection(CHUNKS)
        mgr = ReferenceDocsManager(chroma_store=store)
        real_hash = mgr._compute_file_hash(str(doc))
        for c in CHUNKS:
            c["metadata"]["content_hash"] = real_hash
        assert mgr.sync_file(str(doc), title="AGENTIC_SEARCH") == "skipped"

    def test_no_raw_collections_dict_access_remains(self):
        # Grep-level guard: the buggy access pattern must not come back.
        import inspect
        import knowledge.reference_docs_manager as m
        src = inspect.getsource(m)
        assert "collections.get('reference_docs')" not in src
